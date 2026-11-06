"""Fixed-GPU correctness, memory, performance, and full HumanEval promotion gates."""
import gc
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest
from unittest import mock

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'runtime'), str(ROOT/'runtime/tests')]
CONFIG = json.loads((ROOT/'config.json').read_text())
OLD = Path('/home/hdd3/zhangh/workspace/fastsd/config1-humaneval-full-p20-1031T144115Z')
STATE = {'state':'starting', 'started_epoch_s':time.time(), 'formal_launch_allowed':False,
         'gpu_uuid':CONFIG['gpu_uuid'], 'gates':{}, 'progress':{}}

def write(name, data):
    target = OUT/name
    temp = target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temp.replace(target)

def status(**patch):
    STATE.update(patch, updated_epoch_s=time.time())
    write('gpu-status.json', STATE)
    print(json.dumps(patch,ensure_ascii=False), flush=True)

def new_manager(model, backend):
    manager = KVCacheModel_batching(model, temperature=0, top_k=0, top_p=1, attention_backend=backend)
    manager.vocab_size = model.config.vocab_size
    manager.bounded_probabilities = True
    return manager

def memory():
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

def engine_for(manager, policy):
    engine = AREngine({**CONFIG, 'trace_rows':[], 'target_only_kv_policy':policy,
                       'target_attention_backend':manager.attention_backend})
    engine._manager = manager
    return engine

def legacy_forward(manager, model, parts, pids, pad, **kwargs):
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    @functools.wraps(original_sdpa)
    def measured(*args, **kw):
        manager.attention_calls += 1
        if manager.diagnostic_timer is not None:
            with manager.diagnostic_timer.measure('attention'):
                return original_sdpa(*args, **kw)
        return original_sdpa(*args, **kw)
    with mock.patch('torch.nn.functional.scaled_dot_product_attention', measured):
        return old_varlen.varlen_generate(manager, model, parts, pids, pad, **kwargs)

VARIANTS = [
    ('legacy','sdpa_per_sequence','cpu_offload_legacy'),
    ('resident_sdpa','sdpa_per_sequence','gpu_resident'),
    ('resident_fa2','flash_attn_varlen','gpu_resident'),
]

def call_forward(label, manager, model, parts, pids, **kwargs):
    forward = legacy_forward if label == 'legacy' else varlen_generate
    return forward(manager, model, parts, pids, CONFIG['pad_token_id'], **kwargs)

def performance(model):
    results = []
    for label, backend, policy in VARIANTS:
        for batch in (1, 8, 16):
            manager = new_manager(model, backend)
            engine = engine_for(manager, policy)
            pids = list(range(batch))
            parts = [torch.arange(256+i%7,device='cuda').remainder(1000).unsqueeze(0)+1 for i in pids]
            call_forward(label,manager,model,parts,pids,is_prefill=True,sample_next=False)
            if policy == 'cpu_offload_legacy':
                for pid in pids:engine._move_cache(pid,'cpu')
            del parts
            tails = [torch.tensor([[17]],device='cuda') for _ in pids]
            def step(timer):
                with timer.measure('kv_h2d'):
                    if policy == 'cpu_offload_legacy':
                        for pid in pids:engine._move_cache(pid,'cuda:0')
                with timer.measure('full_forward'):
                    call_forward(label,manager,model,tails,pids,is_prefill=True,sample_next=False)
                with timer.measure('kv_d2h'):
                    if policy == 'cpu_offload_legacy':
                        for pid in pids:engine._move_cache(pid,'cpu')
            step(PhaseTimer(False))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            manager.attention_calls = 0
            counters_before = dict(engine.stats)
            start = time.perf_counter()
            for _ in range(16):step(PhaseTimer(False))
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            clean_calls = manager.attention_calls
            timer = PhaseTimer(True)
            manager.diagnostic_timer = timer
            step(timer)
            torch.cuda.synchronize()
            timing = timer.collect()
            manager.diagnostic_timer = None
            record = {'variant':label, 'batch':batch, 'history_start':256,
                      'clean_iterations':16, 'clean_wall_s':elapsed,
                      'clean_decode_tokens_per_s':16*batch/elapsed,
                      'attention_calls_per_forward':clean_calls/16,
                      'profiled_one_step':timing, **memory(),
                      'kv_transfer_bytes':{key:engine.stats[key]-counters_before[key]
                          for key in ('kv_h2d_bytes','kv_d2h_bytes','kv_transfer_calls')},
                      'workspace':manager._attention_workspace.snapshot() if manager._attention_workspace else None}
            if backend == 'flash_attn_varlen':
                assert clean_calls == 16*len(model.model.layers)
            else:
                assert clean_calls == 16*len(model.model.layers)*batch
            if policy == 'gpu_resident':
                assert engine.stats['kv_transfer_calls'] == 0
            results.append(record)
            write('performance.json',results)
            status(state='performance',progress={'variant':label,'batch':batch})
            engine._manager = None
            del manager, engine
            gc.collect()
            torch.cuda.empty_cache()
    return results

def peak_memory_check(model):
    manager = new_manager(model, 'flash_attn_varlen')
    pids = list(range(16))
    torch.cuda.reset_peak_memory_stats()
    for start in range(0,2048,32):
        parts = [torch.arange(start,start+32,device='cuda').remainder(1000).unsqueeze(0)+1 for _ in pids]
        varlen_generate(manager,model,parts,pids,CONFIG['pad_token_id'],is_prefill=True,sample_next=False)
    torch.cuda.synchronize()
    buffers = [t for cache in manager._past_key_values.values() for pair in cache._fastsd_buffers for t in pair]
    kv_bytes = sum(t.numel()*t.element_size() for t in buffers)
    assert all(cache.get_seq_length()==2048 for cache in manager._past_key_values.values())
    expected = 16*2048*len(model.model.layers)*2*model.config.num_key_value_heads*model.model.layers[0].self_attn.head_dim*2
    assert kv_bytes == expected
    result = {'active_requests':16,'context_tokens':2048,'kv_bytes':kv_bytes, **memory(),
              'workspace':manager._attention_workspace.snapshot(), 'memory_fraction':CONFIG['gpu_memory_fraction']}
    write('peak-memory.json',result)
    del buffers, manager, parts
    gc.collect()
    torch.cuda.empty_cache()
    return result

def greedy_dataset(model, rows, label, backend, policy):
    manager = new_manager(model,backend)
    engine = engine_for(manager,policy)
    jobs = [ARJob(row['source_id'],row['input_token_ids'],CONFIG['max_tokens'],session_id=row['source_id']) for row in rows]
    pending = iter(jobs)
    active, results = [], {}
    emitted = {job.session_id:[] for job in jobs}
    cursor, steps = 0, 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    output = (OUT/('greedy-'+label+'.jsonl')).open('x')
    try:
        while len(results) < len(rows):
            while len(active) < CONFIG['batch_size']:
                job = next(pending,None)
                if job is None:break
                active.append(job)
            plans = plan_batch(active, CONFIG['batch_size'], CONFIG['token_budget'], cursor)
            cursor += 1
            if not plans:raise RuntimeError('Greedy validation stalled')
            parts, pids = [], []
            for job,kind,length in plans:
                if policy == 'cpu_offload_legacy':engine._move_cache(job.session_id,'cuda:0')
                values = job.input_ids[job.prompt_position:job.prompt_position+length] if kind=='prefill' else [job.next_token]
                parts.append(torch.tensor([values],device='cuda'))
                pids.append(job.session_id)
            next_rows = call_forward(label,manager,model,parts,pids,is_prefill=True,sample_next=False)
            for (job,kind,length), tokens in zip(plans,next_rows):
                if policy == 'cpu_offload_legacy':engine._move_cache(job.session_id,'cpu')
                if kind=='prefill':job.prompt_position += length
                if job.prompt_position < len(job.input_ids):continue
                token = int(tokens[-1].item())
                job.next_token = token
                job.emitted += 1
                emitted[job.session_id].append(token)
                reason = completion_reason(token,job.emitted,job.max_tokens,set(CONFIG['eos_token_ids']))
                if reason:
                    result = {'source_id':job.session_id,'output_token_ids':emitted[job.session_id],
                              'stop_reason':reason,'generated_tokens':job.emitted}
                    results[job.session_id] = result
                    output.write(json.dumps(result)+'\n')
                    output.flush()
                    manager.reset(job.session_id)
                    active.remove(job)
            steps += 1
            if steps % 40 == 0:
                status(state='greedy164',progress={'variant':label,'completed':len(results),'total':164,
                       'forward_steps':steps,'elapsed_s':time.perf_counter()-start,
                       'generated_tokens':sum(map(len,emitted.values()))})
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        summary = {'variant':label,'requests':len(results),'elapsed_s':elapsed,
                   'generated_tokens':sum(r['generated_tokens'] for r in results.values()),
                   'kv_transfer_calls':engine.stats['kv_transfer_calls'],**memory()}
        write('greedy-'+label+'-summary.json',summary)
        if policy=='gpu_resident':assert engine.stats['kv_transfer_calls']==0
        assert len(results)==164 and not manager._past_key_values
        return results, summary
    finally:
        output.close()
        engine._manager = None
        del manager, engine
        gc.collect()
        torch.cuda.empty_cache()

def compare(reference, actual):
    differences = []
    for source in sorted(reference,key=lambda x:int(x.split('/')[-1])):
        a,b = reference[source]['output_token_ids'],actual[source]['output_token_ids']
        if a != b:
            offset = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            differences.append({'source_id':source,'first_token_index':offset,
                'reference_token':a[offset] if offset<len(a) else None,
                'actual_token':b[offset] if offset<len(b) else None,
                'reference_length':len(a),'actual_length':len(b)})
    return {'exact':not differences,'compared':len(reference),'differing_tasks':len(differences),
            'first_divergence':differences[0] if differences else None,'differences':differences}

try:
    status(state='preflight')
    cpu = json.loads((OUT/'cpu-status.json').read_text())
    dependency = json.loads(Path('/home/hdd/zhangh/tmp/config1_fa2_20261031/build-status.json').read_text())
    if cpu['state']!='passed' or dependency['state']!='complete':
        raise RuntimeError('CPU contracts or dependency ABI gate not passed')
    from promotion import content_digest
    from suite import verify_bundle
    bundle = verify_bundle()
    digest = content_digest(ROOT)
    if digest != cpu['validated_content_sha256']:raise RuntimeError('Candidate changed since CPU validation')
    STATE.update(bundle_id=bundle,validated_content_sha256=digest)
    if int(os.environ.get('FASTSD_KV_BLOCK_TOKENS','128')) != 128:
        raise RuntimeError('Config1 validation requires the original 128-token capacity blocks')
    STATE['dependency_build'] = dependency
    STATE['effective_runtime_options'] = {'kv_block_tokens':128,
        'native_gqa':os.environ.get('FASTSD_NATIVE_GQA','1')!='0',
        'torch_num_threads':1,'formal_heavy_profiler':False}
    STATE['gates'].update(dependency_abi=True,cpu_contracts=True)
    from campaign import wait_gpu_memory
    status(state='waiting_for_fixed_gpu')
    wait_gpu_memory(CONFIG,OUT,'target_only')
    import torch
    from transformers import AutoModelForCausalLM
    from src.attention_backend import PhaseTimer, require_attention_backend
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import varlen_generate
    from ar_backend import AREngine, ARJob, plan_batch, completion_reason
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'],0)
    torch.manual_seed(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    require_attention_backend('flash_attn_varlen',device='cuda:0',dtype=torch.bfloat16)
    import transformers, flash_attn, flash_attn_2_cuda
    if (torch.__version__,torch.version.cuda,transformers.__version__,flash_attn.__version__) != ('2.9.0+cu128','12.8','4.57.1','2.8.3'):
        raise RuntimeError('Pinned runtime dependency versions changed')
    with open(flash_attn_2_cuda.__file__,'rb') as binary:
        extension_sha=hashlib.file_digest(binary,'sha256').hexdigest()
    dependency_runtime={'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,
        'extension':flash_attn_2_cuda.__file__,'extension_sha256':extension_sha,
        'environment':sys.prefix}
    write('dependency-runtime.json',dependency_runtime)
    status(state='gpu_attention_cache')
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
        ('test_flash_varlen','test_cache_transfer','test_round2_cache','test_review_engine'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write('gpu-tests.json',{'tests_run':result.testsRun,'failures':len(result.failures),
                          'errors':len(result.errors),'skipped':[(str(t),r) for t,r in result.skipped]})
    if not result.wasSuccessful() or result.skipped:raise RuntimeError('GPU correctness gate failed or skipped')
    STATE['gates']['gpu_attention_cache'] = True
    gc.collect()
    torch.cuda.empty_cache()
    # Use the original frozen implementation as the independent legacy baseline.
    old_path = OLD/'runtime/src/kvcache_varlen.py'
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==old_manifest['files']['runtime/src/kvcache_varlen.py']
    spec = importlib.util.spec_from_file_location('src._frozen_legacy_varlen',old_path)
    old_varlen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_varlen)
    status(state='loading_qwen3_8b')
    model = AutoModelForCausalLM.from_pretrained(CONFIG['target_model'],torch_dtype=torch.bfloat16,
               device_map={'':'cuda:0'},local_files_only=True).eval()
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    with torch.inference_mode():
        peak = peak_memory_check(model)
        perf = performance(model)
        STATE['gates']['performance'] = True
        rows = [json.loads(line) for line in (ROOT/'traces/humaneval.jsonl').read_text().splitlines()]
        rows = sorted([r for r in rows if r['node']=='node1' and r['worker']==0],key=lambda r:r['request_index'])
        assert len(rows)==164 and {r['source_id'] for r in rows} == {'HumanEval/'+str(i) for i in range(164)}
        assert all(len(r['input_token_ids'])+CONFIG['max_tokens']<CONFIG['max_context_length'] for r in rows)
        greedy, summaries = {}, []
        comparisons = {}
        for label,backend,policy in VARIANTS:
            greedy[label],summary = greedy_dataset(model,rows,label,backend,policy)
            summaries.append(summary)
            if label!='legacy':
                comparisons[label] = compare(greedy['legacy'],greedy[label])
                write('greedy-comparison.json',comparisons)
                if not comparisons[label]['exact']:
                    STATE['first_divergence'] = STATE.get('first_divergence') or {'variant':label,**comparisons[label]['first_divergence']}
                    status(formal_launch_allowed=False)
        exact = all(comparison['exact'] for comparison in comparisons.values())
        STATE['gates']['humaneval164_exact_greedy'] = exact
        summary = {'passed':exact,'validated_content_sha256':digest,'bundle_id':bundle,
                   'dependency_runtime':dependency_runtime,'dependency_build':dependency,
                   'gates':STATE['gates'],'peak_memory':peak,'performance':perf,
                   'greedy_summaries':summaries,'comparisons':comparisons,
                   'heavy_profiling_in_formal_runs':False,'metadata_polling_unchanged':True}
        write('gpu-summary.json',summary)
        if exact:
            write('validation_certificate.json',summary)
        status(state='passed' if exact else 'blocked_output_difference',formal_launch_allowed=exact,
               finished_epoch_s=time.time())
except BaseException as exc:
    traceback.print_exc()
    status(state='failed',error=repr(exc),traceback=traceback.format_exc(),formal_launch_allowed=False)
    raise
