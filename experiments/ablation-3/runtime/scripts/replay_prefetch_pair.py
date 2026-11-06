"""Frozen-input A/A and A/B/C/D KV-transfer replay on an otherwise idle GPU.

A retains ordinary demand loading. B performs CPU staging only, C imports
the actual previous transfer implementation from Git, and D calls the current
production transfer manager with measured calibration. Profiling samples are
kept out of latency distributions. This is a forward/transport experiment;
request TTFT/TPOT/E2E require a separate end-to-end workload.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values, fraction):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def paired_summary(rows, *, resamples=2000, seed=617):
    """Resample complete pairs; preserve A/D covariance in confidence bounds."""
    if not rows:
        return {'pairs': 0}
    a, d = [r[0] for r in rows], [r[1] for r in rows]
    rng = random.Random(seed)
    deltas = [y - x for x, y in rows]
    boots = {'mean_delta_ms': [], 'p50_ratio': [], 'p95_ratio': []}
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        x, y = [r[0] for r in sample], [r[1] for r in sample]
        boots['mean_delta_ms'].append(sum(v - u for u, v in sample) / len(sample))
        for name, q in [('p50_ratio', .5), ('p95_ratio', .95)]:
            boots[name].append(percentile(y, q) / percentile(x, q) - 1)
    result = {'pairs': len(rows), 'mean_delta_ms': sum(deltas) / len(rows),
              'p50_delta_ms': percentile(deltas, .5), 'p95_delta_ms': percentile(deltas, .95)}
    for name, q in [('p50', .5), ('p95', .95)]:
        result[name + '_a_ms'] = percentile(a, q)
        result[name + '_d_ms'] = percentile(d, q)
        result[name + '_ratio'] = percentile(d, q) / percentile(a, q) - 1
    result['ci95'] = {name: [percentile(values, .025), percentile(values, .975)]
                      for name, values in boots.items()}
    return result


def assess_forward(aa, ad, *, actual_overlap, output_equal, h2d_wait_reduced,
                   total_host_wait_reduced, startup_ok, minimum_pairs=20):
    if min(aa.get('pairs', 0), ad.get('pairs', 0)) < minimum_pairs:
        return 'inconclusive_insufficient_pairs'
    if not output_equal:
        return 'failed_output_mismatch'
    if not actual_overlap:
        return 'failed_no_measured_memcpy_kernel_overlap'
    if not h2d_wait_reduced:
        return 'failed_no_exposed_h2d_reduction'
    if not total_host_wait_reduced:
        return 'failed_no_total_next_batch_host_wait_reduction'
    if not startup_ok:
        return 'failed_forward_start_delay'
    # A/A must resolve the proposed 1%/2% tolerances before A/D is accepted.
    for key, limit in [('p50_ratio', .01), ('p95_ratio', .02)]:
        noise = aa['ci95'][key]
        if max(abs(noise[0]), abs(noise[1])) > limit:
            return 'inconclusive_baseline_noise'
        if ad[key] > limit:
            return 'failed_forward_disturbance'
        if ad['ci95'][key][1] > limit:
            return 'inconclusive_disturbance_confidence_interval'
    return 'passed_measured_shape'


def trace_overlap(path):
    """Attribute GPU activities to labeled production work through correlation IDs.

    Missing background-thread attribution fails closed. Unrelated H2D, demand
    restores and the next forward must never establish speculative overlap.
    """
    events = json.loads(Path(path).read_text())['traceEvents']
    labels = {'round3_target_forward', 'round3_speculative_h2d'}
    ranges = {label: [e for e in events if e.get('ph') == 'X' and e.get('name') == label]
              for label in labels}

    def identifiers(event):
        args = {str(k).lower(): v for k, v in event.get('args', {}).items()}
        return {(key, str(args[key])) for key in ('external id', 'correlation', 'correlation id')
                if key in args}

    allowed = {}
    for label, spans in ranges.items():
        ids = set()
        for span in spans:
            for e in events:
                if (e.get('ph') == 'X' and (e.get('pid'), e.get('tid')) == (span.get('pid'), span.get('tid'))
                        and span['ts'] <= e.get('ts', -1) <= span['ts'] + span.get('dur', 0)):
                    ids.update(identifiers(e))
        allowed[label] = ids
    kernels, copies = [], []
    for e in events:
        if e.get('ph') != 'X' or not e.get('dur'):
            continue
        category, name = e.get('cat', '').lower(), e.get('name', '').lower()
        span = (float(e['ts']), float(e['ts']) + float(e['dur']))
        if category == 'kernel' and identifiers(e) & allowed['round3_target_forward']:
            kernels.append(span)
        elif ('memcpy' in category and ('htod' in name or 'h2d' in name)
              and identifiers(e) & allowed['round3_speculative_h2d']):
            copies.append(span)
    intersections = sorted((max(a, c), min(b, d)) for a, b in kernels
                           for c, d in copies if min(b, d) > max(a, c))
    merged = []
    for start, end in intersections:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return {'memcpy_kernel_overlap_us': sum(b - a for a, b in merged),
            'kernel_activities': len(kernels), 'h2d_activities': len(copies),
            'attribution': 'correlated_labeled_activities' if kernels and copies else 'incomplete_activity_attribution',
            'target_forward_labels': len(ranges['round3_target_forward']),
            'speculative_labels': len(ranges['round3_speculative_h2d'])}


def production_forward(manager, model, residuals, proc_ids, *, is_prefill):
    """Use the exact production primitive and probability-window behavior."""
    from src.kvcache_varlen import varlen_generate
    return varlen_generate(manager, model, residuals, proc_ids, 0,
                           is_prefill=is_prefill, sample_next=False)


def gpu_idle(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    others = [line for line in rows if int(line.split(',')[0]) != os.getpid()]
    if others:
        raise RuntimeError('Selected GPU has existing/concurrent compute processes: ' + '; '.join(others))
    return rows


def legacy_manager_class(ref):
    code = subprocess.check_output(['git', '-C', str(ROOT), 'show', ref + ':src/cache_transfer.py'], text=True)
    module = types.ModuleType('src._round3_previous_cache_transfer')
    module.__package__ = 'src'
    exec(compile(code, ref + ':src/cache_transfer.py', 'exec'), module.__dict__)
    return module.CacheTransferManager, hashlib.sha256(code.encode()).hexdigest()


class FrozenReplay:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM
        from src.cache_transfer import CacheTransferManager
        from src.cache_views import dynamic_cache_from_views
        from src.kvcache_batching import KVCacheModel_batching
        self.torch, self.cache_from_views, self.args = torch, dynamic_cache_from_views, args
        self.compute = torch.cuda.Stream()
        self.dtype = getattr(torch, args.dtype)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.cpu_threads)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=self.dtype,
            attn_implementation=args.attention_backend, local_files_only=True).to('cuda:0').eval()
        self.Manager = CacheTransferManager
        self.ModelCache = KVCacheModel_batching
        self.Legacy, self.legacy_sha = legacy_manager_class(args.legacy_ref)
        self.managers = {}
        self.future_input = torch.arange(args.future_context, device='cuda:0').remainder(self.model.config.vocab_size)[None]
        self.compute.wait_stream(torch.cuda.current_stream())
        future_manager = self.new_model_cache()
        with torch.inference_mode(), torch.cuda.stream(self.compute):
            production_forward(future_manager, self.model, [self.future_input], ['future'], is_prefill=True)
            self.future_values = tuple(tuple(t.detach().clone() for t in pair)
                for pair in future_manager._past_key_values['future'].to_legacy_cache())
            self.future_prob = future_manager._prob_history['future'].clone()
            self.future_prob_start = future_manager._prob_start['future']
        self.compute.synchronize()
        self.future_bytes = sum(t.numel() * t.element_size() for p in self.future_values for t in p)
        if self.future_bytes > args.pool_mib * 1024 ** 2:
            raise ValueError('Future KV must fit the configured GPU/pinned slab budget')
        self.host = dynamic_cache_from_views(tuple(tuple(t.cpu().pin_memory() for t in p) for p in self.future_values))
        self.pageable = tuple(tuple(t.cpu() for t in p) for p in self.future_values)
        self.staging = tuple(tuple(torch.empty_like(t, pin_memory=True) for t in p) for p in self.pageable)
        self.staged_cache = dynamic_cache_from_views(self.staging)
        self.future_tail = torch.arange(args.future_context, args.future_context + 8,
            device='cuda:0').remainder(self.model.config.vocab_size)[None]
        self.compute.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(self.compute):
            reference = self.new_future_cache(self.cache_from_views(self.future_values))
            tokens = production_forward(reference, self.model, [self.future_tail], ['future'], is_prefill=False)
            self.future_expected = self.capture(reference, tokens, ['future'])
        self.compute.synchronize()

    def new_model_cache(self):
        manager = self.ModelCache(self.model, temperature=0,
                                  attention_backend=self.args.runtime_attention_backend)
        manager.vocab_size = self.model.config.vocab_size
        manager.bounded_probabilities = True
        return manager

    def new_future_cache(self, cache):
        manager = self.new_model_cache()
        manager._past_key_values['future'] = cache
        manager._prob_history['future'] = self.future_prob.clone()
        manager._prob_start['future'] = self.future_prob_start
        return manager

    @staticmethod
    def capture(manager, tokens, pids):
        return {'tokens': [t.detach().clone() for t in tokens],
                'probabilities': {pid: manager._prob_history[pid].detach().clone() for pid in pids},
                'prob_start': {pid: manager._prob_start[pid] for pid in pids},
                'kv': {pid: tuple(tuple(t.detach().clone() for t in pair)
                                 for pair in manager._past_key_values[pid].to_legacy_cache()) for pid in pids}}

    def assert_result(self, manager, tokens, pids, expected):
        torch = self.torch
        for observed, reference in zip(tokens, expected['tokens']):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        for pid in pids:
            torch.testing.assert_close(manager._prob_history[pid], expected['probabilities'][pid], rtol=0, atol=0)
            if manager._prob_start[pid] != expected['prob_start'][pid]:
                raise AssertionError('Probability window offset changed')
            for a, b in zip(manager._past_key_values[pid].to_legacy_cache(), expected['kv'][pid]):
                for observed, reference in zip(a, b):
                    torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    def manager(self, condition, group):
        key = condition, group
        if key not in self.managers:
            cls = self.Legacy if condition == 'C' else self.Manager
            extra = {} if condition == 'C' else {'rate_bytes_per_second': self.args.rate_mib_s * 1024 ** 2}
            self.managers[key] = cls('cuda:0', mode='overlap', prefetch_bytes=self.args.pool_mib * 1024 ** 2,
                pinned_pool_bytes=self.args.pool_mib * 1024 ** 2, chunk_bytes=group,
                headroom_bytes=0, record_timeline=False, **extra)
        return self.managers[key]

    def freeze(self, spec):
        torch = self.torch
        phase, batch, new_tokens, context = spec
        prefix = torch.arange(context, device='cuda:0').remainder(self.model.config.vocab_size)[None].expand(batch, -1).contiguous()
        tokens = torch.arange(context, context + new_tokens, device='cuda:0').remainder(self.model.config.vocab_size)[None].expand(batch, -1).contiguous()
        self.spec, self.pids = spec, ['active-' + str(i) for i in range(batch)]
        prefix_manager = self.new_model_cache()
        self.frozen, self.active_buffers = {}, {}
        self.active_manager = self.new_model_cache()
        self.compute.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(self.compute):
            if context:
                production_forward(prefix_manager, self.model, [prefix[i:i + 1] for i in range(batch)], self.pids, is_prefill=True)
                for pid in self.pids:
                    cache = prefix_manager._past_key_values[pid]
                    self.frozen[pid] = {'kv': tuple(tuple(t.detach().clone() for t in pair) for pair in cache.to_legacy_cache()),
                        'prob': prefix_manager._prob_history[pid].clone(), 'start': prefix_manager._prob_start[pid]}
                    self.active_buffers[pid] = tuple(tuple(torch.empty_like(t) for t in pair)
                        for pair in cache._fastsd_buffers)
        self.compute.synchronize()
        self.tokens = [tokens[i:i + 1] for i in range(batch)]
        self.shape = (phase, batch, tuple([new_tokens] * batch), tuple([context] * batch),
                      str(self.dtype), self.args.runtime_attention_backend)
        with torch.inference_mode(), torch.cuda.stream(self.compute):
            self.reset_active()
            output = production_forward(self.active_manager, self.model, self.tokens, self.pids, is_prefill=phase == 'prefill')
            self.expected = self.capture(self.active_manager, output, self.pids)
        self.compute.synchronize()

    def reset_active(self):
        manager = self.active_manager
        manager._past_key_values.clear()
        manager._prob_history.clear()
        manager._prob_start.clear()
        context = self.spec[3]
        for pid, frozen in self.frozen.items():
            buffers = self.active_buffers[pid]
            views = tuple(tuple(t[..., :context, :] for t in pair) for pair in buffers)
            for source_pair, target_pair in zip(frozen['kv'], views):
                for source, target in zip(source_pair, target_pair):
                    target.copy_(source)
            manager._past_key_values[pid] = self.cache_from_views(views, buffers=buffers)
            manager._prob_history[pid] = frozen['prob'].clone()
            manager._prob_start[pid] = frozen['start']

    def trial(self, condition, group, *, probe=False):
        torch, compute = self.torch, self.compute
        mgr = self.manager(condition, group)
        with torch.inference_mode(), torch.cuda.stream(compute):
            self.reset_active()
        start_signal, stop_signal = threading.Event(), threading.Event()
        state = {'ticket': None, 'error': None}
        before = dict(mgr.metrics)
        if condition == 'D' and probe:
            mgr.install_probe(self.shape, group_bytes=group,
                rate_bytes_per_second=self.args.rate_mib_s * 1024 ** 2,
                evidence={'kind': 'explicit_calibration_probe', 'gpu_uuid': self.args.gpu_uuid,
                          'model': str(Path(self.args.model).resolve()), 'shape': self.shape})

        def background():
            try:
                start_signal.wait()
                if condition == 'B':
                    for sources, targets in zip(self.pageable, self.staging):
                        for source, target in zip(sources, targets):
                            target.copy_(source)
                    return
                with torch.profiler.record_function('round3_speculative_h2d'):
                    state['ticket'] = mgr.submit_h2d(self.host, purpose='prefetch')
                while state['ticket'] is not None and not stop_signal.is_set():
                    with torch.profiler.record_function('round3_speculative_h2d'):
                        mgr.advance_prefetch(state['ticket'])
                    if state['ticket'].ready():
                        break
                    stop_signal.wait(.00025)
            except BaseException as exc:
                state['error'] = exc

        thread = None
        if condition != 'A':
            thread = threading.Thread(target=background, daemon=False)
            thread.start()
        anchor, start, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        anchor.record(compute)
        anchor.synchronize()
        mgr.admit()
        if condition == 'C':
            mgr.begin_execution()
        else:
            mgr.begin_execution(shape=self.shape)
        try:
            with torch.inference_mode(), torch.cuda.stream(compute):
                start.record(compute)
                start_signal.set()
                with torch.profiler.record_function('round3_target_forward'):
                    output = production_forward(self.active_manager, self.model, self.tokens, self.pids,
                                                is_prefill=self.spec[0] == 'prefill')
                end.record(compute)
            end.synchronize()
            # The next batch is selected now, so any staging/remaining copy is exposed.
            next_ready = time.perf_counter()
            stop_signal.set()
            if thread is not None:
                thread.join()
            if state['error'] is not None:
                raise state['error']
            mgr.end_execution(compute_stream=compute)
            mgr.release_admission()
            ticket = state['ticket']
            if ticket is None:
                ticket = mgr.submit_h2d(self.staged_cache if condition == 'B' else self.host, purpose='demand')
            # Claim the same production ticket; outstanding chunks become demand.
            ticket.claimed_by = 'paired-replay-next-batch'
            lease = mgr.bind_to_consumer(ticket, compute_stream=compute)
            compute.synchronize()
            wait_ms = (time.perf_counter() - next_ready) * 1000
            self.assert_result(self.active_manager, output, self.pids, self.expected)
            for pair_a, pair_b in zip(lease.cache.to_legacy_cache(), self.future_values):
                for a, b in zip(pair_a, pair_b):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            with torch.inference_mode(), torch.cuda.stream(compute):
                restored_manager = self.new_future_cache(self.cache_from_views(lease.cache.to_legacy_cache()))
                restored_tokens = production_forward(restored_manager, self.model, [self.future_tail], ['future'], is_prefill=False)
            compute.synchronize()
            self.assert_result(restored_manager, restored_tokens, ['future'], self.future_expected)
            result = {'condition': condition, 'group_bytes': group, 'shape': self.shape,
                'forward_gpu_ms': start.elapsed_time(end),
                'plan_ready_to_forward_start_ms': anchor.elapsed_time(start),
                'plan_ready_to_forward_start_clock': 'CUDA event after synchronized anchor immediately before plan readiness',
                'exposed_h2d_wait_ms': lease.exposed_stall_ms(),
                'next_batch_ready_to_kv_usable_host_ms': wait_ms,
                'output_and_kv_equal': True,
                'speculative_submitted_bytes': mgr.metrics.get('speculative_submitted_bytes', 0) - before.get('speculative_submitted_bytes', 0),
                'controller': mgr.controller_status() if condition != 'C' else {'legacy_ref': self.args.legacy_ref}}
            mgr.release_consumer(lease, compute_stream=compute)
            del restored_manager, restored_tokens, output, lease, ticket
            state['ticket'] = None
            gc.collect()
            mgr.poll_completed()
            return result
        finally:
            stop_signal.set()
            start_signal.set()
            if thread is not None:
                thread.join()
            if mgr.executing:
                mgr.end_execution(compute_stream=compute)
                mgr.release_admission()

    def close(self):
        for manager in self.managers.values():
            manager.close()


def parse_shape(value):
    phase, batch, tokens, context = value.split(':')
    spec = phase, int(batch), int(tokens), int(context)
    if phase not in ('prefill', 'verify') or min(spec[1:3]) < 1 or spec[3] < 0:
        raise argparse.ArgumentTypeError('Expected prefill|verify:batch:new_tokens:context')
    return spec


def run(args):
    gpu_idle(args.gpu_uuid)
    if os.environ.get('FASTSD_DIAGNOSTIC_DIR'):
        raise RuntimeError('Unset FASTSD_DIAGNOSTIC_DIR for matched replay; profiler traces are captured separately')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, args.gpu_uuid):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be unset or equal the requested physical UUID')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    import torch
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'running', 'arguments': vars(args).copy(), 'results': [],
              'request_ttft_tpot_e2e': None,
              'request_metric_note': 'Frozen forward replay does not establish request-level latency.',
              'gpu_uuid': args.gpu_uuid, 'torch': torch.__version__,
              'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    report['arguments']['output'] = str(args.output)
    output = args.output / 'paired-report.json'
    runner = None
    try:
        runner = FrozenReplay(args)
        report['legacy_source_sha256'] = runner.legacy_sha
        report['future_kv_bytes'] = runner.future_bytes
        calibration = {'context': {'gpu_uuid': args.gpu_uuid,
            'model': str(Path(args.model).resolve()), 'dtype': str(runner.dtype),
            'attention_backend': args.runtime_attention_backend,
            'torch': torch.__version__, 'forward_primitive': 'varlen_generate',
            'kv_batch_mode': 'varlen'}, 'profiles': []}
        for spec in args.shape:
            runner.freeze(spec)
            for group in args.group_bytes:
                for _ in range(args.warmup):
                    runner.trial('A', group)
                aa, samples = [], []
                for index in range(args.pairs):
                    gpu_idle(args.gpu_uuid)
                    a1, a2 = runner.trial('A', group), runner.trial('A', group)
                    aa.append((a1['forward_gpu_ms'], a2['forward_gpu_ms']))
                    samples.extend([dict(a1, pair=index, stage='AA'), dict(a2, pair=index, stage='AA')])
                paired = {key: [] for key in 'BCD'}
                rng = random.Random(args.seed)
                for condition in 'BCD':
                    for index in range(args.pairs):
                        gpu_idle(args.gpu_uuid)
                        order = ['A', condition]
                        rng.shuffle(order)
                        pair = {c: runner.trial(c, group, probe=condition == 'D') for c in order}
                        paired[condition].append(pair)
                        samples.extend(dict(pair[c], pair=index, stage='A' + condition) for c in order)
                ad = paired_summary([(p['A']['forward_gpu_ms'], p['D']['forward_gpu_ms']) for p in paired['D']])
                baseline = [p['A']['forward_gpu_ms'] for p in paired['D']]
                candidate = [p['D']['forward_gpu_ms'] for p in paired['D']]
                evidence = {'report': str(output), 'kind': 'randomized_frozen_input_pairs',
                    'gpu_uuid': args.gpu_uuid, 'model': str(Path(args.model).resolve()),
                    'dtype': str(runner.dtype), 'attention_backend': args.runtime_attention_backend,
                    'pairs': args.pairs}
                installed = runner.manager('D', group).install_calibration(runner.shape,
                    baseline_ms=baseline, sample_ms=candidate, group_bytes=group,
                    rate_bytes_per_second=args.rate_mib_s * 1024 ** 2,
                    evidence=evidence)
                trace_path = args.output / ('trace-' + '-'.join(map(str, spec)) + '-' + str(group) + '.json')
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    profiled = runner.trial('D', group, probe=not bool(installed))
                prof.export_chrome_trace(str(trace_path))
                overlap = trace_overlap(trace_path)
                aa_summary = paired_summary(aa)
                wait_a = percentile([p['A']['exposed_h2d_wait_ms'] for p in paired['D']], .5)
                wait_d = percentile([p['D']['exposed_h2d_wait_ms'] for p in paired['D']], .5)
                startup = paired_summary([(p['A']['plan_ready_to_forward_start_ms'], p['D']['plan_ready_to_forward_start_ms']) for p in paired['D']])
                total_wait = paired_summary([(p['A']['next_batch_ready_to_kv_usable_host_ms'],
                                              p['D']['next_batch_ready_to_kv_usable_host_ms']) for p in paired['D']])
                total_wait_reduced = (total_wait['p50_d_ms'] < total_wait['p50_a_ms']
                                      and total_wait['p95_d_ms'] < total_wait['p95_a_ms'])
                # Startup also has an absolute floor to avoid ratio explosions below 1ms.
                startup_ok = startup['p95_d_ms'] <= startup['p95_a_ms'] + max(.2, .02 * aa_summary['p50_a_ms'])
                decision = assess_forward(aa_summary, ad,
                    actual_overlap=overlap['memcpy_kernel_overlap_us'] > 0 and profiled['speculative_submitted_bytes'] > 0,
                    output_equal=all(s['output_and_kv_equal'] for s in samples),
                    h2d_wait_reduced=wait_d < wait_a,
                    total_host_wait_reduced=total_wait_reduced, startup_ok=startup_ok)
                if overlap['attribution'] == 'incomplete_activity_attribution':
                    decision = 'inconclusive_profiler_activity_attribution'
                if decision != 'passed_measured_shape':
                    # Leave shape disabled unless strict acceptance was established.
                    runner.manager('D', group).disable_shape(runner.shape, reason=decision)
                else:
                    calibration['profiles'].append({'shape': runner.shape,
                        'baseline_ms': baseline, 'sample_ms': candidate, 'group_bytes': group,
                        'rate_bytes_per_second': args.rate_mib_s * 1024 ** 2,
                        'evidence': dict(evidence, strict_acceptance=decision,
                                         aa_noise=aa_summary, ad_confidence=ad,
                                         actual_activity_overlap=overlap)})
                report['results'].append({'shape': runner.shape, 'group_bytes': group,
                    'AA_forward': aa_summary,
                    'comparisons': {c: paired_summary([(p['A']['forward_gpu_ms'], p[c]['forward_gpu_ms']) for p in pairs]) for c, pairs in paired.items()},
                    'startup_AD': startup, 'total_next_batch_host_wait_AD': total_wait,
                    'total_next_batch_host_wait_includes_worker_join': True,
                    'exposed_wait_p50_A_ms': wait_a,
                    'exposed_wait_p50_D_ms': wait_d, 'trace': str(trace_path),
                    'actual_activity_overlap': overlap, 'decision': decision,
                    'production_calibration_installed': bool(installed) and decision == 'passed_measured_shape',
                    'profile_sample_excluded_from_latency_statistics': profiled,
                    'samples': samples})
                output.write_text(json.dumps(report, indent=2, default=str) + '\n')
                (args.output / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
        report['state'] = 'completed'
    except BaseException as exc:
        report['state'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        if runner is not None:
            runner.close()
        output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({'state': report['state'], 'report': str(output),
                      'decisions': [r['decision'] for r in report['results']]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Existing local model; never downloads weights')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New unique output directory')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--attention-backend', default='sdpa')
    parser.add_argument('--runtime-attention-backend', choices=['sdpa_per_sequence', 'flash_attn_varlen'],
                        default='sdpa_per_sequence', help='Backend used by the timed production varlen_generate primitive')
    parser.add_argument('--shape', action='append', type=parse_shape)
    parser.add_argument('--group-bytes', type=int, nargs='+', default=[262144, 1048576, 4194304])
    parser.add_argument('--rate-mib-s', type=float, default=64)
    parser.add_argument('--pool-mib', type=int, default=64)
    parser.add_argument('--future-context', type=int, default=128)
    parser.add_argument('--legacy-ref', default='7637595a2ea43433b62f3a5d6929a59fe95b84cf')
    parser.add_argument('--pairs', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=617)
    args = parser.parse_args()
    args.shape = args.shape or [('verify', 1, 8, 128), ('verify', 16, 8, 128), ('prefill', 1, 128, 0)]
    if args.pairs < 1 or args.warmup < 0 or min(args.group_bytes) < 1 or args.rate_mib_s <= 0:
        parser.error('Pair count, group bytes and rate must be positive; warmup must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
