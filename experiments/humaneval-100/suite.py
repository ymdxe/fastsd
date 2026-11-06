"""Config1: four methods, frozen HumanEval subset, 32 clients and 100 requests each, mean arrivals 30 seconds."""
import argparse, asyncio, hashlib, json, multiprocessing as mp, os, platform, signal, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
sys.path[:0]=[str(ROOT),str(RUNTIME)]
from common import write_json,digest,valid_id,trace_hash,summarize_dir,install_coordinator,monitor,validate_trace
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
      '--target_attention_backend',c.get('target_attention_backend','sdpa_per_sequence'),
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
    from campaign import wait_gpu_memory
    method='fastsd' if args.server_sched_mode=='fastsd' else 'sd'
    wait_gpu_memory(CONFIG,output,method)
    import torch,transformers
    from runtime_fixes import install
    fixes=install(output,source_commit=CONFIG['source_sha'])
    from cloud.cloud_service import CloudTargetWorker,_ResponseQueuesProxy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'],0)
    from src.attention_backend import require_attention_backend
    attention_metadata=require_attention_backend(args.target_attention_backend,device='cuda:0',dtype=torch.bfloat16)
    if args.target_attention_backend=='flash_attn_varlen' and args.kv_batch_mode!='varlen':
        raise ValueError('flash_attn_varlen requires varlen batching')
    class ObservedWorker(CloudTargetWorker):
        def _load_target_model_for_service(self,*a,**kw):
            started=time.monotonic()
            model=super()._load_target_model_for_service(*a,**kw)
            dtypes=sorted({str(p.dtype) for p in model.parameters()})
            if dtypes!=['torch.bfloat16']:raise RuntimeError('Target dtype mismatch: '+str(dtypes))
            from src.kvcache_varlen import require_full_attention,supports_varlen_model
            require_full_attention(model)
            if args.kv_batch_mode=='varlen' and not supports_varlen_model(model):
                raise ValueError('Target does not support varlen execution')
            write_json(Path(output)/'target-model-ready.json',{
                'epoch_s':time.time(),'load_s':time.monotonic()-started,'dtypes':dtypes,
                'device':str(model.device),'parameters':sum(p.numel() for p in model.parameters()),
                'torch':torch.__version__,'transformers':transformers.__version__,
                'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'gpu_memory_fraction':CONFIG['gpu_memory_fraction'],'allocated_bytes':torch.cuda.memory_allocated(0),
                **attention_metadata})
            ready.set()
            return model
    worker=ObservedWorker(args)
    worker.load_tokenizer()
    try:
        worker.run_target_process_batching(worker.tokenizer,request_queue,_ResponseQueuesProxy(response_queue))
    finally:
        fixes.write_snapshot()
        write_json(Path(output)/'target-memory.json',{'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'epoch_s':time.time()})
def serve(args):
    import uvicorn
    from fastapi import FastAPI
    bundle_id=verify_bundle()
    if CONFIG.get('require_validation_certificate',False):
        from promotion import require_promotion
        require_promotion(ROOT)
    rows=[json.loads(x) for x in (ROOT/'traces'/(args.trace+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    validate_trace(rows,CONFIG)
    output=Path(CONFIG['gpu_result_root'])/valid_id(args.run_id)
    output.mkdir(parents=True,exist_ok=False)
    (output/'reports').mkdir()
    cfg={**CONFIG,'run_id':args.run_id,'method':args.method,'trace':args.trace,'dataset':args.trace,'trace_rows':rows,
      'model':CONFIG['target_model'],'target_device':'cuda:0',
      'trace_sha256':trace_hash(rows),'bundle_id':bundle_id,'output_dir':str(output),
      'arrival_window_s':max(r['scheduled_arrival_s'] for r in rows),'trial_type':'finite_trace_single_repeat',
      'tasks_per_worker':CONFIG['tasks_per_worker'],
      'expected_workers':[f'{node}-w{w}' for node,ncfg in CONFIG['nodes'].items() for w in range(ncfg['workers'])],
      'gpu_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'sharing_note':'Fixed GPU1 admission requires no compute processes; record concurrent shared-host load.',
      'output_equivalence_verified':None}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != CONFIG['gpu_uuid']:
        raise RuntimeError('Expected authorized physical GPU1 UUID in CUDA_VISIBLE_DEVICES')
    write_json(output/'config.json',cfg)
    specedge=args.method=='specedge'
    autoregressive=args.method=='target_only'
    target_log=None
    if autoregressive:
        from ar_backend import install_ar
        app=FastAPI()
        engine=install_ar(app,cfg)
        worker=None
        def alive(): return engine.snapshot().get('worker_alive',True)
        def loaded(): return engine.snapshot().get('model_ready',False)
    elif specedge:
        app=FastAPI()
        @app.get('/clock/sync')
        def clock_sync():
            return {'cloud_time_ns':time.time_ns()}
        command=[CONFIG['gpu_python'],'-B','-u',str(ROOT/'specedge_server.py'),
            '--config',str(output/'config.json'),'--host','127.0.0.1','--port',str(CONFIG['grpc_port'])]
        write_json(output/'target-command.json',{'argv':command,'cwd':str(ROOT)})
        target_log=(output/'specedge-target.log').open('x',encoding='utf-8')
        worker=subprocess.Popen(command,cwd=ROOT,stdout=target_log,stderr=subprocess.STDOUT,start_new_session=True)
        def alive(): return worker.poll() is None
        def loaded():
            return (output/'specedge-ready.json').exists() and not any((output/n).exists() for n in ('specedge-failed.json','specedge-wrapper-failed.json')) and alive()
    else:
        from cloud import cloud_service as cloud
        from src.util import parse_arguments
        flags=common_flags()+method_flags(args.method)+['--draft_model',CONFIG['tokenizer_model'],'--draft_dtype','bfloat16',
           '--target_device','cuda:0','--exp_name',str(output/'target'),'--gamma','4','--num_drafts','32']
        saved=sys.argv
        try:
            sys.argv=[sys.argv[0]]+flags
            parsed=parse_arguments()
        finally: sys.argv=saved
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
        app=cloud.app
        def alive(): return worker.is_alive()
        def loaded(): return ready.is_set() and alive()
    write_json(output/'processes.json',{'server_pid':os.getpid(),'target_pid':worker.pid if worker else os.getpid(),'specedge':specedge})
    @app.get('/experiment/health')
    def experiment_health():
        from campaign import model_start_health
        return model_start_health(output,alive=alive(),ready=loaded())
    coordinator=install_coordinator(app,cfg,output,None)
    from energy import EnergySampler
    sampler=EnergySampler(output/'energy',cfg,'gpu')
    sampler.start()
    server=uvicorn.Server(uvicorn.Config(app,host=CONFIG['bind_host'],port=CONFIG['port'],log_level='warning',timeout_keep_alive=600))
    started=time.monotonic()
    @app.on_event('startup')
    async def watch():
        async def loop():
            model_ready_started=None
            while not server.should_exit:
                if loaded() and model_ready_started is None:
                    model_ready_started=time.monotonic()
                timed_out=model_ready_started is not None and time.monotonic()-model_ready_started > CONFIG['stage_timeout_s']
                if timed_out:
                    coordinator.aborted='Stage exceeded configured timeout'
                    coordinator.save()
                if coordinator.aborted or not alive():
                    await asyncio.sleep(2);server.should_exit=True;return
                if len(coordinator.reports)==len(cfg['expected_workers']):
                    await asyncio.sleep(20);server.should_exit=True;return
                await asyncio.sleep(.5)
        asyncio.create_task(loop())
    try:
        server.run()
    finally:
        if autoregressive:
            engine.stop()
            target_exitcode=1 if engine.snapshot().get('error') else 0
        elif specedge:
            if alive():
                worker.terminate()
                try: worker.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    if os.getpgid(worker.pid)!=worker.pid: raise RuntimeError('Unexpected SpecEdge process group')
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=10)
            target_exitcode=worker.returncode
            target_log.close()
        else:
            if alive():
                cloud.request_queue.put(None)
                worker.join(timeout=45)
                if alive():
                    print('Own target did not drain in 45s; terminating own worker',flush=True)
                    worker.terminate();worker.join(timeout=10)
            target_exitcode=worker.exitcode
        sampler.finish(coordinator.t0,coordinator.measurement_end)
        write_json(output/'server-exit.json',{'epoch_s':time.time(),'target_exitcode':target_exitcode,'state':coordinator.state()})
    if target_exitcode or coordinator.aborted or len(coordinator.reports)!=len(cfg['expected_workers']):
        raise RuntimeError('Incomplete run; inspect retained results')

def check():
    bundle=verify_bundle()
    out={'bundle_id':bundle,'source_sha':CONFIG['source_sha'],'methods':{}}
    for method in ('sd','fastsd'):
        out['methods'][method]=common_flags()+method_flags(method)
    for trace in CONFIG['datasets']:
        rows=[json.loads(x) for x in (ROOT/'traces'/(trace+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        validate_trace(rows,CONFIG)
        assert len(rows)==CONFIG['tasks_per_worker']*sum(n['workers'] for n in CONFIG['nodes'].values())
        assert len({x['sample_id'] for x in rows})==len(rows)
        for node,cfg in CONFIG['nodes'].items():
            for w in range(cfg['workers']):
                subset=[x for x in rows if x['node']==node and x['worker']==w]
                assert len(subset)==CONFIG['tasks_per_worker']
                assert len({x['input_key'] for x in subset})==CONFIG['tasks_per_worker']
                assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(subset,subset[1:]))
        out[trace]={'requests':len(rows),'trace_sha256':trace_hash(rows),'arrival_window_s':max(x['scheduled_arrival_s'] for x in rows)}
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    p=sub.add_parser('server');p.add_argument('method',choices=CONFIG['methods']);p.add_argument('trace',choices=CONFIG['datasets']);p.add_argument('run_id')
    args=parser.parse_args()
    if args.command=='check':check()
    else:serve(args)

