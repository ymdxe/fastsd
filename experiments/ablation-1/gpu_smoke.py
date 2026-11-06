"""One owned, bounded GPU readiness check; all samples are outside measurement."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT/'runtime'),str(ROOT/'runtime/tests')]
from common import write_json
from suite import CONFIG, verify_bundle


def model_check(output):
    import torch
    from transformers import AutoModelForCausalLM
    from src.cache_transfer import CacheTransferManager
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import varlen_generate
    from diagnostic_trace import trace

    model = AutoModelForCausalLM.from_pretrained(CONFIG['target_model'],torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    compute = torch.cuda.Stream()
    transfers = CacheTransferManager('cuda:0',mode='overlap',prefetch_bytes=64*1024**2,
                                     headroom_bytes=0,record_timeline=True)
    reference = KVCacheModel_batching(model,temperature=0,attention_backend='sdpa_per_sequence')
    actual = KVCacheModel_batching(model,temperature=0,attention_backend='sdpa_per_sequence')
    for manager in (reference,actual):
        manager.vocab_size=model.config.vocab_size
        manager.bounded_probabilities=True
    prefix=torch.arange(128,device='cuda:0').reshape(1,-1)
    tail=torch.arange(128,136,device='cuda:0').reshape(1,-1)
    try:
        with torch.no_grad(),torch.cuda.stream(compute):
            varlen_generate(reference,model,[prefix],['probe'],0,is_prefill=True,sample_next=False)
            source=reference._past_key_values['probe']
            transfers.mark_computed(source,producer_stream=compute)
            offload=transfers.submit_d2h(source)
            offload.copy_done.synchronize()
            actual._prob_history['probe']=reference._prob_history['probe'].clone()
            actual._prob_start['probe']=reference._prob_start['probe']
            expected=varlen_generate(reference,model,[tail],['probe'],0,sample_next=False)[0]
        compute.synchronize()
        ticket=transfers.submit_h2d(offload.destination_cache,purpose='prefetch')
        if ticket is None:
            raise RuntimeError('Idle KV prefetch was not admitted')
        lease=transfers.bind_to_consumer(ticket,compute_stream=compute)
        actual._past_key_values['probe']=lease.cache
        with torch.no_grad(),torch.cuda.stream(compute):
            observed=varlen_generate(actual,model,[tail],['probe'],0,sample_next=False)[0]
        compute.synchronize()
        torch.testing.assert_close(observed,expected,rtol=0,atol=0)
        torch.testing.assert_close(actual._prob_history['probe'][:,-8:],reference._prob_history['probe'][:,-8:],rtol=0,atol=0)
        for a,b in zip(reference._past_key_values['probe'].to_legacy_cache(),actual._past_key_values['probe'].to_legacy_cache()):
            for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
        transfers.release_consumer(lease,compute_stream=compute)
        compute.synchronize()
        return {'model':CONFIG['target_model'],'dtype':'bfloat16','backend':'sdpa_per_sequence',
            'exact_probability_and_kv_match':True,'prefix_tokens':128,'continuation_tokens':8,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'scope':'Single functional KV offload/reload and continuation; no performance or corpus equivalence claim'}
    finally:
        transfers.close()
        trace.close()


def main():
    output=Path(sys.argv[1])
    bundle=verify_bundle()
    if (output/'status.json').exists():
        raise FileExistsError('Readiness check already attempted; inspect retained status')
    status={'state':'running','source_sha':CONFIG['source_sha'],'bundle_id':bundle,
            'started_epoch_s':time.time(),'pid':os.getpid(),'formal_output_equivalence':False}
    write_json(output/'status.json',status)
    try:
        from campaign import query_gpu_memory
        memory=query_gpu_memory(CONFIG['gpu_uuid'])
        if memory['compute_pids'] or memory['free_mib']<43008:
            raise RuntimeError('GPU is occupied before CUDA initialization: '+repr(memory))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'],0)
        tests=unittest.defaultTestLoader.loadTestsFromName('test_round3_transfer.CudaSlabTests')
        result=unittest.TextTestRunner(verbosity=2).run(tests)
        if not result.wasSuccessful() or result.skipped or result.testsRun!=3:
            raise RuntimeError('Existing CUDA slab checks did not all run and pass')
        status['cuda_tests']={'run':result.testsRun,'skipped':len(result.skipped),'passed':True}
        gc.collect()
        torch.cuda.empty_cache()
        status['real_model']=model_check(output)
        status.update(state='complete',passed=True)
    except BaseException as exc:
        status.update(state='failed',passed=False,error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        status['finished_epoch_s']=time.time()
        write_json(output/'status.json',status)


if __name__=='__main__': main()
