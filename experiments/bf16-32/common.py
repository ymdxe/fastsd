import hashlib,json,math,os,re,subprocess,threading,time
from pathlib import Path

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def valid_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise ValueError('Use a run ID containing only letters, numbers, underscore, dot and hyphen')
    return run_id

def trace_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

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
    for node in ('node1','node2','node3','all'):
        subset = [r for r in rows if node == 'all' or r['task_id'].startswith(node+'-')]
        groups[node] = {'requests': len(subset),
            'scheduled_ttft_ms': stats([r['arrival_lag_ms']+r['ttft_ms'] for r in subset if r.get('ttft_ms') is not None]),
            'scheduled_e2e_ms': stats([r['arrival_lag_ms']+r['request_e2e_ms'] for r in subset]),
            'service_ttft_ms': stats([r.get('ttft_ms') for r in subset]),
            'service_e2e_ms': stats([r['request_e2e_ms'] for r in subset]),
            'tpot_ms': stats([r.get('tpot_ms') for r in subset])}
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