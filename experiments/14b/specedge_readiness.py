"""Real model capacity and diagnostic probes outside formal measurements."""
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]


def cpu_check(output):
    import gzip
    config=json.loads((ROOT/'config.json').read_text())
    node=platform.node().split('.')[0]
    worker=json.loads((ROOT/'affinity.json').read_text())[node]['workers'][0]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,worker['logical_cpus'])
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['FASTSD_DIAGNOSTIC_DIR']=str(output/'diagnostics')
    from specedge.bootstrap import prepare_imports
    provenance=prepare_imports()
    from client import IntegratedSpecExecClient
    from cpu_adapter import CPUCompatibleSpecEdgeEngine
    import torch
    import util
    import diagnostic_trace
    from measurements import observe_cpu_model,set_task
    if Path(diagnostic_trace.__file__).resolve()!=ROOT/'runtime/diagnostic_trace.py':
        raise RuntimeError('SpecEdge imported a diagnostic module outside this frozen runtime')
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    model=util.load_graph_model(config['nodes'][node]['model'],torch.device('cpu'),torch.bfloat16)
    observe_cpu_model(model)
    engine=CPUCompatibleSpecEdgeEngine(model,config['specedge_max_len'],1)
    row=json.loads((ROOT/'traces/humaneval.jsonl').read_text().splitlines()[0])
    ids=torch.tensor([row['input_token_ids'][:64]],dtype=torch.long)
    n=ids.size(1)
    positions=torch.arange(n).unsqueeze(0)
    mask=(torch.arange(engine.max_len)[None,:]<=torch.arange(n)[:,None]).float()[None,None]
    set_task('readiness-only')
    started=time.monotonic()
    engine.prefill(ids,positions,0,positions,mask)
    token=torch.tensor([[int(ids[0,-1])]])
    decode_mask=(torch.arange(engine.max_len)<=n).float()[None,None,None]
    logits=engine.forward(token,torch.tensor([[n]]),torch.tensor([0]),torch.tensor([n]),decode_mask)
    if not torch.isfinite(logits).all() or logits.shape[:2]!=(1,1):
        raise RuntimeError('Actual CPU model returned invalid decode logits')
    engine.reset()
    diagnostic_trace.trace.close()
    files=list((output/'diagnostics').glob('events-*.jsonl.gz'))
    if len(files)!=1:raise RuntimeError('Frozen gzip diagnostic stream is absent')
    with gzip.open(files[0],'rt',encoding='utf-8') as handle:
        events=[json.loads(line) for line in handle]
    forwards=[e for e in events if e['kind']=='cpu_model_forward']
    if len(forwards)!=2 or any(e['task_id']!='readiness-only' for e in forwards):
        raise RuntimeError('Actual CPU forwards are missing from diagnostic evidence')
    if torch.cuda.is_initialized():raise RuntimeError('CPU probe initialized CUDA')
    return {'passed':True,'host':node,'model':config['nodes'][node]['model'],
        'diagnostic_module':str(Path(diagnostic_trace.__file__).resolve()),'gzip_stream':str(files[0]),
        'cpu_forwards':len(forwards),'elapsed_s':time.monotonic()-started,'provenance':provenance}


def gpu_check(output):
    import queue
    import torch
    from specedge_server import configure_environment,_runtime_dataset_hook,_cuda_memory
    config=json.loads((ROOT/'config.json').read_text())
    target={**config,'model':config['target_model'],'target_device':'cuda:0',
        'output_dir':str(Path(output)/'specedge-capacity'),
        'expected_workers':[f'{node}-w{w}' for node,c in config['nodes'].items() for w in range(c['workers'])]}
    configure_environment(target)
    from specedge.bootstrap import prepare_imports
    provenance=prepare_imports()
    import util
    from single_chain_server import SingleChainInferenceController
    from wire_codec import SpecEdgeWireCodec
    from specedge_grpc import specedge_pb2
    _runtime_dataset_hook(util)
    fraction=config['specedge_gpu_memory_fraction']
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    torch.cuda.reset_peak_memory_stats(0)
    responses=queue.Queue()
    started=time.monotonic()
    controller=SingleChainInferenceController(32,queue.Queue(),responses)
    torch.cuda.synchronize()
    ready=_cuda_memory(torch,0)
    length=config['specedge_max_len']-config['max_tokens']-2*config['tree_budget']-1
    tokenizer=controller._tokenizer
    tokens=tokenizer.encode('The small cat sat on the wooden chair. '*300)[:length]
    prefix=tokenizer.decode(tokens,skip_special_tokens=False)
    if len(tokens)!=length or tokenizer.encode(prefix)!=tokens:
        raise RuntimeError('Long-prefix probe did not round-trip exact canonical token IDs')
    def request(client,prefill):
        gamma=controller._client_gammas[client]
        root=length-1 if prefill else controller._cache_lengths[client]
        slots=torch.arange(root,root+gamma+1)
        mask=(torch.arange(config['specedge_max_len'])[None,:]<=slots[:,None]).to(torch.bfloat16)[None]
        return specedge_pb2.ValidateRequest(client_idx=client,req_idx=100000+client,
            input_ids=SpecEdgeWireCodec.encode(torch.tensor([tokens[-1]]*(gamma+1))),
            position_ids=SpecEdgeWireCodec.encode(slots),cache_seq_indices=SpecEdgeWireCodec.encode(slots),
            parent_indices=SpecEdgeWireCodec.encode(slots[:-1]),attention_mask=SpecEdgeWireCodec.encode(mask),
            prefill=prefill,prefix=prefix if prefill else '')
    batches=[]
    for start in (0,16):
        for prefill in (True,False):
            before=time.monotonic()
            elapsed,indices=controller._inference([request(client,prefill) for client in range(start,start+16)])
            for unused in range(16):responses.get_nowait()
            torch.cuda.synchronize()
            batches.append({'first_client':start,'prefill':prefill,'rows':16,
                'prefills':len(indices),'decode_event_ms':elapsed,'wall_s':time.monotonic()-before,
                'memory':_cuda_memory(torch,0)})
    return {'passed':True,'model':config['target_model'],'gpu_memory_fraction':fraction,
        'num_clients':32,'max_batch_size':16,'max_len':config['specedge_max_len'],
        'prefix_tokens':length,'ready_memory':ready,'batches':batches,
        'memory':_cuda_memory(torch,0),'elapsed_s':time.monotonic()-started,'provenance':provenance,
        'scope':'Actual official graph and resident KV capacity; synthetic candidate tokens, no output-equivalence/performance claim'}


if __name__=='__main__':
    result=cpu_check(sys.argv[1])
    (Path(sys.argv[1])/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
