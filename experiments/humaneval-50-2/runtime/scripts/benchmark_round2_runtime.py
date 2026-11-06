"""Bounded planner microbench and optional tiny-model CUDA dependency timeline.

This is a correctness/overhead probe, not a production-model serving benchmark.
"""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fastsd_scheduler import WorkItem, cost_features, plan_iteration, project_after_plan
from src.progress_control import AcceptanceProfile, BatchCostModel
from src.run_artifacts import git_sha


def summary(values):
    return {'median_ms': statistics.median(values),
            'p95_ms': sorted(values)[int((len(values) - 1) * 0.95)]}


def planner_benchmark(iterations):
    results = {}
    for count in (32, 64):
        queues = {'verify': {'short': []}, 'prefill': {'short': []}}
        profiles = {}
        for index in range(count):
            item = WorkItem(str(index), str(index), 'verify', 'short',
                total_tokens=8, base_prefix_len=128 * (index + 1),
                request={'task_type': 'verify', 'proc_id': str(index), 'prefix_len': 128 * (index + 1), 'gamma': 8})
            queues['verify']['short'].append(item)
            p = profiles[item.proc_id] = AcceptanceProfile()
            for i in range(128):
                p.observe(i % 9, 8)
        costs, plans, looks = BatchCostModel(), [], []
        def build(state):
            items = {i.work_id: i for cats in state.values() for rows in cats.values() for i in rows}
            def score(item, sl, selected):
                rows = [cost_features(items[s.work_id], s) for s in selected if s.task_type == item.task_type]
                marginal = max(1e-6, costs.estimate(item.task_type, rows + [cost_features(item, sl)]) - costs.estimate(item.task_type, rows))
                return profiles[item.proc_id].slice_progress(item.accepted_so_far, sl.draft_token_count,
                    sl.offset + sl.draft_token_count >= item.total_tokens) / marginal
            return plan_iteration(state, 32, max_num_seqs=4, min_prefill_chunk_tokens=1,
                                  selection_policy='progress', candidate_value=score)
        for i in range(iterations + 5):
            start = time.perf_counter()
            plan = build(queues)
            plan_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            build(project_after_plan(queues, plan))
            look_ms = (time.perf_counter() - start) * 1000
            if i >= 5:
                plans.append(plan_ms)
                looks.append(look_ms)
        results[str(count)] = {'plan': summary(plans), 'lookahead': summary(looks), 'iterations': iterations}
    return results


def gpu_benchmark(iterations):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.cache_transfer import CacheTransferManager
    from src.cache_views import dynamic_cache_from_views
    torch.manual_seed(11)
    config = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).to('cuda').eval()
    prefix, tail = torch.randint(0, 97, (1, 512), device='cuda'), torch.randint(0, 97, (1, 4), device='cuda')
    with torch.no_grad():
        base = model(prefix, use_cache=True).past_key_values.to_legacy_cache()
        for _ in range(3):
            model(tail, past_key_values=dynamic_cache_from_views(base), use_cache=True)
    torch.cuda.synchronize()
    host = dynamic_cache_from_views([(torch.ones((1, 2, 8192, 64), pin_memory=True),
                                     torch.zeros((1, 2, 8192, 64), pin_memory=True))])
    reference, results = None, {}
    for mode in ('off', 'protect_forward', 'overlap'):
        manager = CacheTransferManager('cuda:0', mode='protect_forward' if mode == 'off' else mode,
                                       chunk_bytes=256 * 1024, record_timeline=True)
        compute = torch.cuda.Stream()
        records, forward_intervals = [], []
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        origin.synchronize()
        try:
            for _ in range(iterations):
                # Prepare before admission, including all allocations.
                ticket = manager.submit_h2d(host, purpose='prefetch') if mode != 'off' else None
                cache = dynamic_cache_from_views(base)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                done = threading.Event()
                failures = []
                def background():
                    torch.cuda.set_device(0)
                    while not done.is_set():
                        try:
                            if ticket is not None:
                                manager.advance_prefetch(ticket)
                        except BaseException as exc:
                            failures.append(repr(exc))
                            return
                        done.wait(0.0005)
                if ticket is not None:
                    manager.advance_prefetch(ticket)
                ready_at = time.perf_counter()
                manager.admit()
                drain_ms = manager.begin_execution()
                worker = threading.Thread(target=background)
                worker.start()
                try:
                    with torch.cuda.stream(compute), torch.no_grad():
                        start.record(compute)
                        logits = model(tail, past_key_values=cache, use_cache=True).logits
                        end.record(compute)
                    end.synchronize()
                    e2e = (time.perf_counter() - ready_at) * 1000
                finally:
                    done.set()
                    worker.join()
                if failures:
                    raise RuntimeError(failures)
                manager.end_execution(compute_stream=compute)
                manager.release_admission()
                actual = logits.cpu()
                if reference is None:
                    reference = actual
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
                forward_intervals.append((origin.elapsed_time(start), origin.elapsed_time(end)))
                if ticket is None:
                    ticket = manager.submit_h2d(host)
                    hit = 'demand_miss'
                else:
                    hit = 'ready_hit' if ticket.ready() else 'pending_hit'
                    ticket.claimed_by = 'B'
                lease = manager.bind_to_consumer(ticket, compute_stream=compute)
                stall = lease.exposed_stall_ms()
                compute.synchronize()
                torch.testing.assert_close(lease.cache.to_legacy_cache()[0][0].cpu(), host.to_legacy_cache()[0][0])
                manager.release_consumer(lease, compute_stream=compute)
                compute.synchronize()
                manager.collect()
                records.append({'forward_gpu_ms': start.elapsed_time(end),
                    'batch_start_delay_ms': drain_ms, 'exposed_h2d_stall_ms': stall,
                    'request_e2e_ms': e2e, 'B_cache_path': hit})
            torch.cuda.synchronize()
            copies = [{'start_ms': origin.elapsed_time(a), 'end_ms': origin.elapsed_time(b),
                       'direction': direction, 'purpose': purpose}
                      for a, b, direction, purpose in manager.timeline]
            overlap = sum(max(0, min(end, c['end_ms']) - max(start, c['start_ms']))
                          for start, end in forward_intervals for c in copies if c['purpose'] == 'prefetch')
            if mode == 'protect_forward' and overlap > 0.001:
                raise AssertionError(f'protect_forward DMA overlap: {overlap} ms')
            results[mode] = {'records': records, 'forward': summary([r['forward_gpu_ms'] for r in records]),
                'speculative_dma_forward_overlap_ms': overlap, 'forward_intervals_ms': forward_intervals,
                'copy_intervals': copies, 'transfer_metrics': dict(manager.metrics), 'output_parity': True}
        finally:
            manager.close()
    results['shape'] = {'model': 'random tiny Llama', 'batch': 1, 'context': 512, 'gamma': 4,
                        'dtype': 'float32', 'gpu': torch.cuda.get_device_name(0)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu-iterations', type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = {'source_sha': git_sha(ROOT), 'parameters': vars(args),
              'actual_import_paths': {name: importlib.import_module(name).__file__ for name in
                  ('src.fastsd_scheduler', 'src.progress_control')},
              'planner': planner_benchmark(args.iterations),
              'scope': 'tiny-model dependency and overhead probe; no production performance claim'}
    if args.gpu:
        result['gpu'] = gpu_benchmark(args.gpu_iterations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'planner': result['planner'],
                      'gpu_modes': list(result.get('gpu', {}))}, indent=2))


if __name__ == '__main__':
    main()
