"""Paired 32-Draft BF16 experiment; upstream runtime is immutable."""
import argparse, asyncio, hashlib, json, multiprocessing as mp, os, platform, signal, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
sys.path[:0]=[str(ROOT),str(RUNTIME)]
from common import write_json,digest,valid_id,trace_hash,summarize_dir,install_coordinator,monitor
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def verify_bundle():
    doc=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in doc['files'].items():
        if digest(ROOT/name)!=expected: raise RuntimeError('Changed bundle file: '+name)
    return doc['bundle_id']
def common_flags():
    c=CONFIG
    return ['--target_model',c['target_model'],'--target_dtype','bfloat16',
      '--batch_size',str(c['batch_size']),'--token_budget',str(c['token_budget']),
      '--max_tokens',str(c['max_tokens']),'--verify_method','greedy','--temp','0','--top_k','0','--top_p','1',
      '--seed',str(c['seed']),'--stop_policy','eos','--no-measure_energy','--no-pipeline_gamma_adapt',
      '--session_idle_timeout','1800']
def method_flags(method):
    if method not in ('sd','fastsd'):raise ValueError(method)
    enabled=method=='fastsd'
    return ['--profile','custom' if enabled else 'vanilla','--server_sched_mode','fastsd' if enabled else 'vanilla',
      '--kv_batch_mode','varlen','--scheduler_policy','progress' if enabled else 'fcfs',
      '--enable_pipeline' if enabled else '--no-enable_pipeline',
      '--enable_proactive_draft' if enabled else '--no-enable_proactive_draft',
      '--enable_latency_priority' if enabled else '--no-enable_latency_priority',
      '--overlap_prefill_first_draft' if enabled else '--no-overlap_prefill_first_draft',
      '--kv_prefetch' if enabled else '--no-kv_prefetch','--kv_prefetch_mode','protect_forward']
def target_worker(args,request_queue,response_queue,ready,output):
    import torch,transformers
    from cloud.cloud_service import CloudTargetWorker,_ResponseQueuesProxy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'],0)
    class ObservedWorker(CloudTargetWorker):
        def _load_target_model_for_service(self,*a,**kw):
            started=time.monotonic()
            model=super()._load_target_model_for_service(*a,**kw)
            dtypes=sorted({str(p.dtype) for p in model.parameters()})
            if dtypes!=['torch.bfloat16']:raise RuntimeError('Target dtype mismatch: '+str(dtypes))
            write_json(Path(output)/'target-model-ready.json',{
                'epoch_s':time.time(),'load_s':time.monotonic()-started,'dtypes':dtypes,
                'device':str(model.device),'parameters':sum(p.numel() for p in model.parameters()),
                'torch':torch.__version__,'transformers':transformers.__version__,
                'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'gpu_memory_fraction':CONFIG['gpu_memory_fraction'],'allocated_bytes':torch.cuda.memory_allocated(0)})
            ready.set()
            return model
    worker=ObservedWorker(args)
    worker.load_tokenizer()
    try:
        worker.run_target_process_batching(worker.tokenizer,request_queue,_ResponseQueuesProxy(response_queue))
    finally:
        write_json(Path(output)/'target-memory.json',{'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'epoch_s':time.time()})
def serve(args):
    import uvicorn
    from cloud import cloud_service as cloud
    from src.util import parse_arguments
    bundle_id=verify_bundle()
    rows=[json.loads(x) for x in (ROOT/'reference'/(args.trace+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    output=Path(CONFIG['gpu_result_root'])/valid_id(args.run_id)
    output.mkdir(parents=True,exist_ok=False)
    (output/'reports').mkdir()
    cfg={**CONFIG,'run_id':args.run_id,'method':args.method,'trace':args.trace,'trace_rows':rows,
      'trace_sha256':trace_hash(rows),'bundle_id':bundle_id,'output_dir':str(output),
      'arrival_window_s':max(r['scheduled_arrival_s'] for r in rows),'trial_type':'finite_burst_paired_single_repeat',
      'tasks_per_worker':1 if args.trace=='smoke' else CONFIG['tasks_per_worker'],
      'expected_workers':[f'{node}-w{w}' for node,ncfg in CONFIG['nodes'].items() for w in range(ncfg['workers'])],
      'gpu_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'target_physical_gpu':2,
      'sharing_authorized':True,'sharing_note':'Other users remain on GPU2; descriptive shared-resource result.',
      'output_equivalence_verified':None}
    write_json(output/'config.json',cfg)
    flags=common_flags()+method_flags(args.method)+['--draft_model',CONFIG['tokenizer_model'],'--draft_dtype','bfloat16',
       '--target_device','cuda:0','--exp_name',str(output/'target'),'--gamma','4','--num_drafts','32']
    sys.argv=[sys.argv[0]]+flags
    parsed=parse_arguments()
    write_json(output/'target-command.json',{'argv':flags,'resolved_parameters':vars(parsed),'python':sys.executable})
    cloud._request_max_tokens=CONFIG['max_tokens']
    cloud._server_sched_mode=parsed.server_sched_mode
    cloud._enable_latency_priority=parsed.enable_latency_priority
    ctx=mp.get_context('spawn')
    cloud.request_queue,cloud.response_queue=ctx.Queue(),ctx.Queue()
    ready=ctx.Event()
    worker=ctx.Process(target=target_worker,args=(parsed,cloud.request_queue,cloud.response_queue,ready,str(output)))
    worker.start()
    cloud.worker_proc=worker
    write_json(output/'processes.json',{'server_pid':os.getpid(),'target_pid':worker.pid})
    app=cloud.app
    @app.get('/experiment/health')
    def experiment_health():
        alive=worker.is_alive(); loaded=ready.is_set() and alive
        return {'status':'ok' if loaded else ('loading' if alive else 'failed'),'model_ready':loaded,'worker_alive':alive}
    coordinator=install_coordinator(app,cfg,output,None)
    stop=threading.Event()
    observer=threading.Thread(target=monitor,args=(output,stop),daemon=True)
    observer.start()
    server=uvicorn.Server(uvicorn.Config(app,host=CONFIG['bind_host'],port=CONFIG['port'],log_level='warning'))
    @app.on_event('startup')
    async def watch():
        async def loop():
            while not server.should_exit:
                if coordinator.aborted or len(coordinator.reports)==len(cfg['expected_workers']) or not worker.is_alive():
                    await asyncio.sleep(1);server.should_exit=True;return
                await asyncio.sleep(.5)
        asyncio.create_task(loop())
    try:
        server.run()
    finally:
        if worker.is_alive():
            cloud.request_queue.put(None)
            worker.join(timeout=45)
            if worker.is_alive():
                print('Own target did not drain in 45s; terminating own worker',flush=True)
                worker.terminate();worker.join(timeout=10)
        stop.set();observer.join(timeout=7)
        state=coordinator.state()
        write_json(output/'server-exit.json',{'epoch_s':time.time(),'target_exitcode':worker.exitcode,'state':state})
    if worker.exitcode or coordinator.aborted or len(coordinator.reports)!=len(cfg['expected_workers']):
        raise RuntimeError('Incomplete run; inspect retained results')
def check():
    bundle=verify_bundle()
    out={'bundle_id':bundle,'source_sha':CONFIG['source_sha'],'methods':{}}
    for method in ('sd','fastsd'):
        out['methods'][method]=common_flags()+method_flags(method)
    for trace in ('smoke','main'):
        rows=[json.loads(x) for x in (ROOT/'reference'/(trace+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(rows)==(32 if trace=='smoke' else 320)
        assert len({x['sample_id'] for x in rows})==len(rows)
        for node,cfg in CONFIG['nodes'].items():
            for w in range(cfg['workers']):
                subset=[x for x in rows if x['node']==node and x['worker']==w]
                assert len(subset)==(1 if trace=='smoke' else 10)
                assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(subset,subset[1:]))
        out[trace]={'requests':len(rows),'trace_sha256':trace_hash(rows),'arrival_window_s':max(x['scheduled_arrival_s'] for x in rows)}
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    p=sub.add_parser('server');p.add_argument('method',choices=['sd','fastsd']);p.add_argument('trace',choices=['smoke','main']);p.add_argument('run_id')
    args=parser.parse_args()
    if args.command=='check':check()
    else:serve(args)

