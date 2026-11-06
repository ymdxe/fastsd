#!/usr/bin/env python3
"""Manually launched AR / Standard SD / FastSD paired experiment suite."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import random
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
sys.path[:0] = [str(ROOT), str(RUNTIME)]
from experiment_hooks import rpc, wait_for_start, finish_worker

NODES = {
    'node1': {'python': '/home/zhangh/miniconda3/envs/fastsd/bin/python',
              'model': '/mnt/nfs0/zhangh/models/Qwen3-0.6B', 'gamma': 4, 'rate': 5.0},
    'node2': {'python': '/home/zhangh/venvs/fastsd_cpu/bin/python',
              'model': '/vol1/zhangh/models/Qwen3-0.6B', 'gamma': 4, 'rate': 5.0},
    'node3': {'python': '/home/zhangh/envs/fastsd_cpu_py314/bin/python314_glibc',
              'model': '/mnt/nfs0/zhangh/models/Qwen3-1.7B', 'gamma': 8, 'rate': 10.0},
}
GPU_REPO = Path('/home/hdd3/zhangh/workspace/fastsd')
GPU_PYTHON = '/home/hdd/zhangh/envs/specedge/bin/python'
TARGET = '/home/hdd/zhangh/models/Qwen3-8B'
TOKENIZER = '/home/hdd/zhangh/models/Qwen3-0.6B'


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise ValueError('Use a run ID containing only letters, numbers, underscore, dot and hyphen')
    return run_id


def manifest():
    data = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in data['files'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Bundle changed since validation: ' + name)
    return data['bundle_id']


def rows_for_trace(name):
    if name in ('load05', 'load08', 'load1', 'smoke'):
        inputs = [json.loads(x) for x in (ROOT/'reference/inputs.jsonl').read_text(encoding='utf-8').splitlines()]
        rate = {'load05':0.5, 'load08':0.8, 'load1':1.0, 'smoke':1.0}[name]
        records = []
        for node,seed in (('node1',11000),('node3',31000)):
            for worker in range(8):
                rng=random.Random(seed+worker); arrival=0.0
                for index in range(1 if name=='smoke' else 10):
                    arrival += rng.expovariate(rate)
                    question=inputs[(worker+8*index)%80]
                    records.append({'sample_id':f'{node}-w{worker}-q{index}','node':node,'worker':worker,
                        'source_index':question['source_index'],'prompt':question['prompt'],
                        'scheduled_arrival_s':arrival})
        records.sort(key=lambda row:(row['scheduled_arrival_s'],row['sample_id']))
        return records, max(row['scheduled_arrival_s'] for row in records), False
    inputs = [json.loads(x) for x in (ROOT/'reference/inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    if name in ('single', 'parity'):
        rows = []
        for node in ('node1', 'node3'):
            for worker in range(1 if name == 'single' else 8):
                for index in range(2 if name == 'single' else 10):
                    q = inputs[(worker + 8*index) % 80]
                    rows.append({'sample_id': f'{node}-w{worker}-q{index}', 'node': node,
                        'worker': worker, 'source_index': q['source_index'], 'prompt': q['prompt'],
                        'scheduled_arrival_s': index*0.01})
        return rows, 0.01*(1 if name == 'single' else 9), False
    matrix = json.loads((ROOT/'reference/matrix.json').read_text(encoding='utf-8'))
    selected = 'mid_rep1' if name == 'pilot' else name
    for run in matrix['runs']:
        if f"{run['level']}_rep{run['repeat']}" == selected:
            path = ROOT/'reference/traces'/Path(run['trace'].replace('\\','/')).name
            if digest(path) != run['trace_sha256']:
                raise RuntimeError('Frozen trace hash mismatch')
            rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
            if name == 'pilot':
                rows = [row for row in rows if row['scheduled_arrival_s'] <= 60]
            return rows, (60 if name == 'pilot' else run['arrival_window_s']), name != 'pilot'
    raise ValueError('Unknown trace: ' + name)


def trace_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def method_flags(method):
    enabled = method == 'fastsd'
    return ['--profile', 'custom' if enabled else 'vanilla', '--server_sched_mode',
        'fastsd' if enabled else 'vanilla', '--kv_batch_mode', 'varlen',
        '--enable_pipeline' if enabled else '--no-enable_pipeline',
        '--enable_proactive_draft' if enabled else '--no-enable_proactive_draft',
        '--overlap_prefill_first_draft' if enabled else '--no-overlap_prefill_first_draft',
        '--enable_latency_priority' if enabled else '--no-enable_latency_priority',
        '--no-pipeline_gamma_adapt']


def common_flags():
    return ['--target_model', TARGET, '--target_dtype', 'bfloat16', '--batch_size', '16',
        '--token_budget', '512', '--max_tokens', '32', '--verify_method', 'greedy',
        '--temp', '0', '--top_k', '0', '--top_p', '1', '--seed', '1234',
        '--stop_policy', 'eos', '--no-measure_energy']


def summarize_dir(output):
    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
    rows = []
    reports = {}
    for path in sorted((output/'reports').glob('*.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        reports[report['worker']] = report
        rows.extend(report['records'])
    coordinator_file = output/'coordinator.json'
    coordinator_state = json.loads(coordinator_file.read_text(encoding='utf-8')) if coordinator_file.exists() else {}
    aborted = coordinator_state.get('aborted')
    expected = {r['sample_id']: r for r in config['trace_rows']}
    results = {r['task_id']: r for r in rows}
    if len(results) != len(rows):
        raise RuntimeError('Duplicate result IDs')
    unexpected = sorted(set(results)-set(expected))
    if unexpected:
        raise RuntimeError('Unexpected result IDs: ' + str(unexpected))
    def stats(values):
        values = sorted(float(v) for v in values if v is not None)
        if not values:
            return None
        return {'mean': sum(values)/len(values), 'p50': values[max(0, math.ceil(len(values)*.5)-1)],
            'p95': values[max(0, math.ceil(len(values)*.95)-1)]}
    total_time = max((r.get('completion_s',0) for r in rows), default=0)
    window = config['arrival_window_s']
    completed_in_window = sum(r.get('completion_s', float('inf')) <= window for r in rows) if window else None
    groups = {}
    for node in ('node1','node3','all'):
        subset = [r for r in rows if node == 'all' or r['task_id'].startswith(node+'-')]
        groups[node] = {'requests': len(subset),
            'scheduled_ttft_ms': stats([r['arrival_lag_ms']+r['ttft_ms'] for r in subset if r.get('ttft_ms') is not None]),
            'scheduled_e2e_ms': stats([r['arrival_lag_ms']+r['request_e2e_ms'] for r in subset]),
            'service_ttft_ms': stats([r.get('ttft_ms') for r in subset]),
            'service_e2e_ms': stats([r['request_e2e_ms'] for r in subset]),
            'tpot_ms': stats([r.get('tpot_ms') for r in subset])}
        clock_rows = [row for row in subset if 'clock_recovery_count' in row]
        groups[node]['clock_recovery'] = ({
            'requests_with_clock_metrics': len(clock_rows),
            'requests_with_recovery': sum(row['clock_recovery_count'] > 0 for row in clock_rows),
            'events': sum(row['clock_recovery_count'] for row in clock_rows),
            'posthoc_pull_events': sum(row['clock_pull_recovery_count'] for row in clock_rows),
            'pre_enqueue_upload_retry_events': sum(row['clock_upload_recovery_count'] for row in clock_rows),
            'recovery_ms_total': sum(row['clock_recovery_ms'] for row in clock_rows),
            'calibration_rpc_count': sum(row['clock_sync_rpc_count'] for row in clock_rows),
            'calibration_ms_total': sum(row['clock_sync_ms'] for row in clock_rows),
            'rejected_upload_http_ms_total': sum(row['clock_rejected_upload_http_ms'] for row in clock_rows),
        } if clock_rows else None)
    summary = {'run_id': config['run_id'], 'method': config['method'], 'trace': config['trace'],
        'bundle_id': config['bundle_id'], 'trace_sha256': config['trace_sha256'],
        'expected_requests': len(expected), 'completed_requests': len(rows),
        'missing_request_ids': sorted(set(expected)-set(results)),
        'all_workers_reported': sorted(reports) == sorted(config['expected_workers']),
        'all_requests_completed': len(rows) == len(expected) and sorted(reports) == sorted(config['expected_workers']),
        'aborted': aborted, 'measurement_valid': not aborted and len(rows) == len(expected) and sorted(reports) == sorted(config['expected_workers']),
        'total_output_tokens': sum(r['generated_tokens'] for r in rows), 'drain_time_s': total_time,
        'drain_token_per_s': sum(r['generated_tokens'] for r in rows)/total_time if total_time else None,
        'arrival_window_s': window, 'completed_in_window': completed_in_window,
        'backlog_at_window_end': len(expected)-completed_in_window if completed_in_window is not None else None,
        'completed_requests_per_s_in_window': completed_in_window/window if window and completed_in_window is not None else None,
        'groups': groups,
        'window_token_throughput': None,
        'clock_window_boundary_note': 'Completion times within clock uncertainty of a window boundary are ambiguous; consult barrier calibrations.',
        'clock_recovery_note': ('Request E2E includes failed pre-enqueue upload attempts and all clock recovery. '
            'Successful HTTP/transport timing excludes those costs, which are reported separately. '
            'Recovered one-way pull is a posthoc calibration estimate selected to be nonnegative, '
            'not an unbiased one-way latency measurement. Raw negative estimates, offsets, uncertainty, '
            'all calibration samples, warmup events and failures are in each CPU worker clock-recovery.jsonl. '
            'Group recovery totals describe completed measured request records only.'),
        'note': 'Window token throughput omitted: SD does not record every token visibility time. Clock uncertainty is in worker barrier.json and coordinator.json.'}
    write_json(output/'summary.json', summary)
    return summary


class Coordinator:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.lock = threading.RLock()
        self.ready, self.reports = {}, {}
        self.t0 = None
        self.aborted = None
    def state(self):
        with self.lock:
            return {'run_id': self.config['run_id'], 'ready': sorted(self.ready),
                'expected': self.config['expected_workers'], 'reported': sorted(self.reports),
                't0_server_epoch_s': self.t0, 'aborted': self.aborted}
    def save(self):
        write_json(self.output/'coordinator.json', {**self.state(), 'calibrations': self.ready})
    def register(self, body):
        with self.lock:
            worker = body['worker']
            if body['run_id'] != self.config['run_id'] or worker not in self.config['expected_workers']:
                raise ValueError('Wrong run ID or unexpected worker')
            for key in ('bundle_id','trace_sha256'):
                if body[key] != self.config[key]:
                    raise ValueError(key+' mismatch; sync the same bundle')
            if self.aborted or self.t0 is not None or worker in self.ready:
                raise ValueError('Run aborted, already started, or worker registered twice; use a new run ID')
            self.ready[worker] = body['clock']
            if len(self.ready) == len(self.config['expected_workers']):
                self.t0 = time.time()+10
            self.save()
            return self.state()
    def report(self, body):
        with self.lock:
            worker = body['worker']
            if self.aborted:
                raise ValueError('Run aborted: '+str(self.aborted))
            if body['run_id'] != self.config['run_id'] or worker not in self.ready or self.t0 is None:
                raise ValueError('Worker has not joined this experiment')
            if worker in self.reports:
                raise ValueError('Duplicate report')
            expected_ids = {r['sample_id'] for r in self.config['trace_rows'] if f"{r['node']}-w{r['worker']}" == worker}
            ids = [r['task_id'] for r in body['records']]
            if set(ids) != expected_ids or len(ids) != len(expected_ids):
                raise ValueError('Incomplete or duplicate worker records')
            self.reports[worker] = len(ids)
            write_json(self.output/'reports'/(worker+'.json'), body)
            self.save()
            if len(self.reports) == len(self.config['expected_workers']):
                summarize_dir(self.output)
            return self.state()


def install_coordinator(app, config, output, cache_snapshot=None):
    from fastapi import HTTPException
    coordinator = Coordinator(config, output)
    @app.get('/experiment/info')
    def info():
        return config
    @app.get('/experiment/state')
    def state():
        return coordinator.state()
    @app.post('/experiment/register')
    def register(body: dict):
        try:
            return coordinator.register(body)
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc
    @app.post('/experiment/report')
    async def report(body: dict):
        try:
            state = coordinator.report(body)
            if len(coordinator.reports) == len(config['expected_workers']) and cache_snapshot is not None:
                snapshot = await cache_snapshot()
                write_json(output/'cache-final.json', snapshot)
            return state
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc
    @app.post('/experiment/abort')
    def abort(body: dict):
        if body.get('run_id') != config['run_id']:
            raise HTTPException(409, 'Wrong run')
        with coordinator.lock:
            coordinator.aborted = body.get('reason','worker failure')
            coordinator.save()
        return coordinator.state()
    return coordinator


def monitor(output, stop):
    while not stop.is_set():
        item = {'epoch_s': time.time(), 'loadavg': os.getloadavg()}
        try:
            result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            item['gpu'] = result.stdout.strip().splitlines()
            item['gpu_query_exit'] = result.returncode
            item['process_rss_kib'] = {}
            targets = [os.getpid()]
            children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children')
            if children.exists():
                targets += [int(x) for x in children.read_text().split()]
            for pid in targets:
                try:
                    line = next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('VmRSS:'))
                    item['process_rss_kib'][str(pid)] = int(line.split()[1])
                except (OSError, StopIteration):
                    pass
        except Exception as exc:
            item['error'] = str(exc)
        with (output/'resources.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item)+'\n')
        stop.wait(1)


def serve(args):
    import multiprocessing as mp
    import uvicorn
    bundle_id = manifest()
    rows, window, formal = rows_for_trace(args.trace)
    run_id = valid_id(args.run_id)
    if formal:
        gate = json.loads((ROOT/'gate.json').read_text(encoding='utf-8')) if (ROOT/'gate.json').exists() else {}
        if not gate.get('passed') or gate.get('bundle_id') != bundle_id:
            raise RuntimeError('Formal matrix locked: run oracle, single/parity tests, then gate. See README.zh.md')
    output = GPU_REPO/'exp/four_method_20261026'/run_id
    output.mkdir(parents=True, exist_ok=False)
    (output/'reports').mkdir()
    config = {'run_id': run_id, 'method': args.method, 'trace': args.trace, 'trace_rows': rows,
        'arrival_window_s': window, 'formal': formal, 'bundle_id': bundle_id, 'trace_sha256': trace_hash(rows),
        'expected_workers': [f'{node}-w{worker}' for node in ('node1','node3')
            for worker in range(1 if args.trace=='single' else 8)],
        'model': TARGET, 'tokenizer': TOKENIZER, 'max_tokens': 32, 'batch_size': 16,
        'token_budget': 512, 'output_dir': str(output), 'network': 'HTTP/gRPC over fixed desktop-mediated SSH broker; not IB direct',
        'kv_policy': 'CPU offload after each forward; FastSD may prefetch as part of method',
        'model_settings': 'BF16 greedy; Qwen3 enable_thinking=False; EOS inclusive; max32',
        'gpu_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'specedge_grpc_target':'127.0.0.1:29636','specedge_max_len':2048,'tree_budget':32,
        'num_clients':16,'target_device':'cuda:0','clock_uncertainty_limit_ms':100,
        'intended_output_equivalence_to_8b':args.method!='draft_only',
        'output_equivalent_to_8b':False if args.method=='draft_only' else None,
        'trial_type':'finite_burst_single_repeat','tasks_per_worker':1 if args.trace=='smoke' else 10,
        'fixed_gamma':{'node1':4,'node3':8},
        'rate_per_worker':{'load05':0.5,'load08':0.8,'load1':1.0,'smoke':1.0}.get(args.trace)}
    write_json(output/'config.json', config)
    write_json(output/'launch.json', {'argv': sys.argv, 'python': sys.executable, 'hostname': platform.node()})
    for filename, command in [('gpu-before.txt',['nvidia-smi']), ('disk-before.txt',['df','-h',str(GPU_REPO)]),
        ('git-head.txt',['git','-C',str(GPU_REPO),'rev-parse','HEAD']),
        ('git-status.txt',['git','-C',str(GPU_REPO),'status','--short'])]:
        result = subprocess.run(command,capture_output=True,text=True)
        (output/filename).write_text(result.stdout+result.stderr,encoding='utf-8')
    worker = None
    spec_process = None
    spec_log = None
    if args.method in ('draft_only','specedge_cpu_adapted'):
        from fastapi import FastAPI
        app = FastAPI(title='Four-method experiment coordinator')
        if args.method == 'specedge_cpu_adapted':
            spec_output = output/'specedge'
            spec_output.mkdir()
            spec_config = dict(config, output_dir=str(spec_output))
            write_json(output/'specedge-config.json', spec_config)
            spec_command = [sys.executable,'-B','-u',str(ROOT/'specedge_server.py'),
                '--config',str(output/'specedge-config.json'),'--host','127.0.0.1','--port','19636']
            spec_log = (output/'specedge.log').open('x',encoding='utf-8')
            spec_process = subprocess.Popen(spec_command,cwd=ROOT,stdout=spec_log,stderr=subprocess.STDOUT)
            write_json(output/'specedge-command.json',spec_command)
        @app.get('/health')
        def health():
            if args.method=='draft_only':
                return {'status':'ok','model_ready':True,'target_model_loaded':False,'method':'draft_only'}
            alive=spec_process.poll() is None
            ready=(output/'specedge/specedge-ready.json').exists() and alive
            return {'status':'ok' if ready else ('loading' if alive else 'failed'),
                'model_ready':ready,'worker_alive':alive,'method':'specedge_cpu_adapted'}
        @app.get('/clock/sync')
        def clock_sync():
            return {'cloud_time_ns':time.time_ns()}
        async def cache_snapshot():
            return {'method':args.method,'target_cache_not_applicable':args.method=='draft_only',
                'policy':'no Target for draft_only; official preallocated per-client KV for SpecEdge'}
    elif args.method == 'ar':
        from fastapi import FastAPI
        from ar_backend import install_ar
        app = FastAPI(title='AR paired experiment')
        engine = install_ar(app, config)
        async def cache_snapshot():
            return engine.snapshot()
    else:
        from cloud import cloud_service as cloud
        from src.util import parse_arguments
        flags = common_flags()+method_flags(args.method)+['--draft_model',TOKENIZER,
            '--target_device','cuda:0','--exp_name',str(output/'target'), '--gamma','4','--num_drafts','16']
        saved = sys.argv
        try:
            sys.argv = [saved[0]]+flags
            parsed = parse_arguments()
        finally:
            sys.argv = saved
        cloud._request_max_tokens = 32
        cloud._server_sched_mode = parsed.server_sched_mode
        cloud._enable_latency_priority = parsed.enable_latency_priority
        ctx = mp.get_context('spawn')
        cloud.request_queue, cloud.response_queue = ctx.Queue(), ctx.Queue()
        cloud.worker_proc = cloud._start_worker(parsed,cloud.request_queue,cloud.response_queue)
        worker = cloud.worker_proc
        app = cloud.app
        write_json(output/'target-flags.json',flags)
        cache_snapshot = cloud.experiment_cache_stats
    coordinator = install_coordinator(app,config,output,cache_snapshot)
    stop = threading.Event()
    observer = threading.Thread(target=monitor,args=(output,stop),daemon=True)
    observer.start()
    print('Output:',output,flush=True)
    print('Waiting for actual model-ready, then clients:',', '.join(config['expected_workers']),flush=True)
    server = uvicorn.Server(uvicorn.Config(app,host='219.216.65.42',port=19635,workers=1,log_level='info'))
    @app.on_event('startup')
    async def start_completion_watcher():
        import asyncio
        async def completion_watcher():
            while not server.should_exit:
                if coordinator.aborted or len(coordinator.reports)==len(config['expected_workers']):
                    await asyncio.sleep(1)
                    server.should_exit=True
                    return
                await asyncio.sleep(0.5)
        asyncio.create_task(completion_watcher())
    try:
        server.run()
    finally:
        stop.set()
        observer.join(timeout=7)
        if worker is not None and worker.is_alive():
            cloud.request_queue.put(None)
            worker.join(timeout=15)
            if worker.is_alive():
                print('Stopping this launch\'s own Target worker after Ctrl+C',flush=True)
                worker.terminate()
                worker.join(timeout=5)
        if spec_process is not None and spec_process.poll() is None:
            spec_process.send_signal(signal.SIGINT)
            try:
                spec_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                spec_process.terminate()
                spec_process.wait(timeout=10)
        if spec_log is not None:
            spec_log.close()
        write_json(output/'server-exit.json',{'epoch_s':time.time(),'state':coordinator.state(),
            'target_exitcode':worker.exitcode if worker else None})


def ar_worker(args):
    import requests
    from transformers import AutoTokenizer
    info = rpc(args.server_url,'/experiment/info')
    node = platform.node().split('.')[0]
    cfg = NODES[node]
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    samples = [json.loads(x) for x in Path(args.inputs).read_text(encoding='utf-8').splitlines()]
    def encode(prompt):
        return tokenizer.apply_chat_template([{'role':'user','content':prompt.strip()}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
    session = requests.Session()
    session.trust_env = False
    def generate(ids, sample_id, limit):
        start = time.monotonic()
        first = None
        tokens = []
        done = False
        with session.post(args.server_url+'/ar/generate',json={'input_token_ids':ids,
            'sample_id':sample_id,'max_tokens':limit},stream=True,timeout=(10,1800)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                event = json.loads(line)
                if event.get('error'):
                    raise RuntimeError(str(event))
                if 'token_id' in event:
                    now = time.monotonic()
                    first = now if first is None else first
                    tokens.append(int(event['token_id']))
                if event.get('done'):
                    done = True
        end = time.monotonic()
        if not done or not tokens:
            raise RuntimeError('Incomplete AR stream')
        return start,first,end,tokens
    fixture = samples[0] if samples else json.loads((ROOT/'reference/inputs.jsonl').read_text(encoding='utf-8').splitlines()[0])
    generate(encode(fixture['prompt']),os.environ['FASTSD_SUITE_WORKER']+'-warmup',2)
    origin = wait_for_start(args)
    with Path(args.exp_name,'edge_metrics_proc0.jsonl').open('x',encoding='utf-8') as handle:
        for sample in samples:
            scheduled = sample['scheduled_arrival_s']
            wait = origin+scheduled-time.monotonic()
            if wait > 0:
                time.sleep(wait)
            # Start before tokenization, matching Edge's measured request path.
            start = time.monotonic()
            _,first,end,tokens = generate(encode(sample['prompt']),sample['sample_id'],32)
            row = {'task_id':sample['sample_id'],'generated_tokens':len(tokens),
                'output_token_ids':tokens,'output_text':tokenizer.decode(tokens,skip_special_tokens=True),
                'scheduled_arrival_s':scheduled,'arrival_lag_ms':max(0,start-origin-scheduled)*1000,
                'request_e2e_ms':(end-start)*1000,'ttft_ms':(first-start)*1000,
                'tpot_ms':(end-first)*1000/(len(tokens)-1) if len(tokens)>1 else None,
                'completion_s':end-origin,'request_start_monotonic_s':start,'completion_monotonic_s':end,
                'experiment_t0_monotonic_s':origin,'accepted_total':0,'drafted_total':0}
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            handle.flush()
    finish_worker(args)


def clients(args):
    bundle_id = manifest()
    node = platform.node().split('.')[0]
    if node not in ('node1','node3'):
        raise RuntimeError('This matrix uses node1 and node3; node2 is prepared for status/backup only')
    info = rpc(args.server_url,'/experiment/info')
    if info['run_id'] != valid_id(args.run_id) or info['bundle_id'] != bundle_id:
        raise RuntimeError('Wrong Target run ID or bundle; inspect status')
    health = rpc(args.server_url,'/health')
    if not health.get('model_ready') or health.get('status') != 'ok':
        raise RuntimeError('Target still loading or failed: '+str(health)+'; wait for model loading to finish')
    output = Path('/home/zhangh/workspace/fastsd/exp/four_method_20261026')/args.run_id/node
    output.mkdir(parents=True,exist_ok=False)
    cfg = NODES[node]
    children = []
    def interrupted(signum,frame):
        raise KeyboardInterrupt('launcher signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGHUP):
        signal.signal(signum,interrupted)
    write_json(output/'config.json',info)
    try:
        for identity in info['expected_workers']:
            if not identity.startswith(node+'-'):
                continue
            worker = int(identity.rsplit('w',1)[1])
            sample_rows = [r for r in info['trace_rows'] if r['node']==node and r['worker']==worker]
            worker_dir = output/f'worker{worker}'
            worker_dir.mkdir()
            inputs = worker_dir/'inputs.jsonl'
            inputs.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample_rows),encoding='utf-8')
            env = os.environ.copy()
            env.update({name:'8' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCH_NUM_THREADS')})
            env.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT)+os.pathsep+str(RUNTIME),
                'FASTSD_SUITE_ROOT':str(ROOT),
                'FASTSD_SUITE_WORKER':identity,'FASTSD_SUITE_RUN_ID':args.run_id,
                'FASTSD_SUITE_BUNDLE_ID':bundle_id,'FASTSD_SUITE_TRACE_SHA256':info['trace_sha256'],
                'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'})
            executable = os.environ.get('PYTHON_BIN', sys.executable)
            env['PYTHON_BIN'] = cfg['python']
            if info['method'] in ('draft_only','specedge_cpu_adapted'):
                command = [executable,'-B','-u',str(ROOT/'suite.py'),'external-worker',
                    '--method',info['method'],'--inputs',str(inputs),'--exp-name',str(worker_dir),
                    '--server-url',args.server_url,'--draft-model',cfg['model'],
                    '--gamma',str(cfg['gamma']),'--edge-threads','8','--max-tokens','32']
            elif info['method']=='ar':
                command = [executable,'-B','-u',str(ROOT/'suite.py'),'ar-worker',
                    '--inputs',str(inputs),'--exp-name',str(worker_dir),'--server-url',args.server_url]
            else:
                command = [executable,'-B','-u',str(RUNTIME/'edge/edge.py')]+common_flags()+method_flags(info['method'])+[
                    '--draft_model',cfg['model'],'--draft_dtype','float32','--draft_device','cpu',
                    '--edge_use_cpu','--edge_threads','8','--num_drafts','1','--gamma',str(cfg['gamma']),
                    '--server_url',args.server_url,'--request_timeout','1800',
                    '--timing_max_clock_uncertainty_ms',str(info.get('clock_uncertainty_limit_ms',100)),
                    '--dataset','mt_bench',
                    '--dataset_file',str(inputs),'--max_tasks_per_draft','0','--warmup_requests','1',
                    '--arrival_distribution','poisson','--arrival_rate',str(info.get('rate_per_worker') or cfg['rate']),
                    '--exp_name',str(worker_dir)]
            write_json(worker_dir/'command.json',{'argv':command,'environment':{k:env[k] for k in (
                'CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'FASTSD_SUITE_WORKER','FASTSD_SUITE_RUN_ID','FASTSD_SUITE_BUNDLE_ID')}})
            log = (worker_dir/'worker.log').open('x',encoding='utf-8')
            process = subprocess.Popen(command,cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((identity,process,log))
        write_json(output/'processes.json',{'launcher_pid':os.getpid(),'workers':[{"worker":name,"pid":proc.pid} for name,proc,_ in children]})
        print(f'{node}: started {len(children)} clients. Loading/warmup -> READY -> common T0',flush=True)
        while any(proc.poll() is None for _,proc,_ in children):
            failed = [(name,proc.returncode) for name,proc,_ in children if proc.poll() not in (None,0)]
            if failed:
                raise RuntimeError('Worker failed: '+str(failed)+'; inspect '+str(output))
            print('Running:',[name for name,proc,_ in children if proc.poll() is None],flush=True)
            time.sleep(15)
        failed = [(name,proc.returncode) for name,proc,_ in children if proc.returncode]
        if failed:
            raise RuntimeError(str(failed))
        print('All clients completed. GPU results:',info['output_dir']+'/summary.json',flush=True)
    except BaseException as exc:
        try:
            rpc(args.server_url,'/experiment/abort',{'run_id':args.run_id,'reason':node+': '+str(exc)})
        except Exception:
            pass
        # Only processes created by this invocation are stopped.
        for _,proc,_ in children:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
        for _,proc,_ in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=5)
        raise
    finally:
        for _,_,log in children:
            log.close()
        write_json(output/'exit-status.json',{name:proc.poll() for name,proc,_ in children})


def gate(args):
    bundle_id = manifest()
    oracle = Path(args.oracle).resolve()
    reference = {int(r['source_index']):r['hf_token_ids'] for r in
        map(json.loads,(oracle/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    oracle_summary = json.loads((oracle/'summary.json').read_text(encoding='utf-8'))
    if (oracle.parent/'bundle-id.txt').read_text(encoding='utf-8').strip()!=bundle_id:
        raise RuntimeError('Oracle came from a different source bundle')
    if len(reference)!=80:
        raise RuntimeError('Oracle must cover all 80 prompts')
    for key in ('varlen_single','hf_repeat'):
        check = oracle_summary['comparisons'][key]
        if check['missing'] or check['mismatches']:
            raise RuntimeError('Oracle '+key+' does not match; inspect diagnostic results')
    evidence = []
    for method,run_id in zip(('ar','sd','fastsd'),args.runs):
        output = GPU_REPO/'exp/four_method_20261026'/valid_id(run_id)
        summary = summarize_dir(output)
        config = json.loads((output/'config.json').read_text(encoding='utf-8'))
        if summary['method'] != method or summary['trace'] != 'parity' or not summary['measurement_valid']:
            raise RuntimeError('Need a complete 160-request parity run for '+method)
        if summary['bundle_id']!=bundle_id:
            raise RuntimeError('Correctness run used a different source bundle')
        snapshot = json.loads((output/'cache-final.json').read_text(encoding='utf-8'))
        check_keys = ('active','pending','cache_count','prob_history_count') if method == 'ar' else (
            'cache_sessions','prob_history_sessions','work_items','pending_releases','http_pending',
            'http_prefill_arrivals','http_prefill_timing','accept_stats_sessions','rng_sessions','ema_sessions')
        if any(snapshot.get(key) != 0 for key in check_keys):
            raise RuntimeError(method+' state did not drain cleanly: '+str(snapshot))
        compared = 0
        for path in (output/'reports').glob('*.json'):
            for row in json.loads(path.read_text(encoding='utf-8'))['records']:
                source_index = next(r['source_index'] for r in config['trace_rows'] if r['sample_id']==row['task_id'])
                if row['output_token_ids'] != reference[source_index]:
                    raise RuntimeError(method+' token mismatch: '+row['task_id']+'; inspect reports before timing')
                compared += 1
        if compared != 160:
            raise RuntimeError('Expected 160 paired outputs')
        evidence.append({'method':method,'run_id':run_id,'summary_sha256':digest(output/'summary.json')})
    result = {'passed':True,'bundle_id':bundle_id,'oracle':str(oracle),
        'oracle_requests_sha256':digest(oracle/'requests.jsonl'),'evidence':evidence,'epoch_s':time.time()}
    if (ROOT/'gate.json').exists():
        raise RuntimeError('gate.json already exists; retain it and inspect rather than overwrite')
    write_json(ROOT/'gate.json',result)
    print('PASS: all current AR/SD/FastSD parity outputs match the 80-prompt oracle. Formal matrix unlocked.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    server = commands.add_parser('server')
    server.add_argument('method',choices=['sd','fastsd','specedge_cpu_adapted','draft_only','ar'])
    server.add_argument('trace')
    server.add_argument('run_id')
    client = commands.add_parser('clients')
    client.add_argument('run_id')
    client.add_argument('--server-url',default='http://127.0.0.1:29635')
    worker = commands.add_parser('ar-worker')
    worker.add_argument('--inputs',required=True)
    worker.add_argument('--exp-name',required=True)
    worker.add_argument('--server-url',required=True)
    external = commands.add_parser('external-worker')
    external.add_argument('--method',choices=['draft_only','specedge_cpu_adapted'],required=True)
    external.add_argument('--inputs',required=True)
    external.add_argument('--exp-name',required=True)
    external.add_argument('--server-url',required=True)
    external.add_argument('--draft-model',required=True)
    external.add_argument('--gamma',type=int,required=True)
    external.add_argument('--edge-threads',type=int,default=8)
    external.add_argument('--max-tokens',type=int,default=32)
    status = commands.add_parser('status')
    status.add_argument('--server-url',default='http://127.0.0.1:29635')
    summary = commands.add_parser('summary')
    summary.add_argument('run_id')
    check = commands.add_parser('gate')
    check.add_argument('oracle')
    check.add_argument('runs',nargs=3,metavar='AR_SD_FASTSD_RUN_IDS')
    commands.add_parser('check')
    args = parser.parse_args()
    if args.command=='server':
        serve(args)
    elif args.command=='clients':
        clients(args)
    elif args.command=='ar-worker':
        ar_worker(args)
    elif args.command=='external-worker':
        if args.method=='draft_only':
            from draftonly import run
        else:
            from specedge_worker import run
        run(args)
    elif args.command=='status':
        print(json.dumps({'health':rpc(args.server_url,'/health'),
            'experiment':rpc(args.server_url,'/experiment/state')},indent=2))
    elif args.command=='summary':
        output = GPU_REPO/'exp/four_method_20261026'/valid_id(args.run_id)
        print(json.dumps(summarize_dir(output),ensure_ascii=False,indent=2))
    elif args.command=='gate':
        gate(args)
    else:
        print('Bundle:',manifest())
        print('Host:',platform.node(),'Python:',sys.executable)
        for name in ('smoke','load05','load08','load1'):
            rows,window,formal = rows_for_trace(name)
            print(name,len(rows),'requests; window=',window,'formal=',formal)


if __name__ == '__main__':
    main()
