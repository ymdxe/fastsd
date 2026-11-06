import hashlib,json,math,os,re,subprocess,threading,time
from pathlib import Path

def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+str(threading.get_ident())+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def valid_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise ValueError('Use a run ID containing only letters, numbers, underscore, dot and hyphen')
    return run_id

def trace_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

def metric_record_error(record, expected_ids):
    """Validate completion evidence without changing inference or its timing."""
    if not isinstance(record, dict) or not isinstance(record.get('task_id'), str) or record['task_id'] not in expected_ids:
        return 'unexpected or missing task_id'
    if record.get('error') or record.get('status') in ('error', 'failed', 'cancelled') or record.get('success') is False:
        return 'failed request record'
    tokens = record.get('generated_tokens')
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        return 'missing or invalid generated_tokens'
    for key in ('completion_s', 'request_e2e_ms', 'arrival_lag_ms'):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return 'missing or invalid ' + key
    return None


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
        subset = [r for r in rows if node == 'all' or expected[r['task_id']]['node'] == node]
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
        self.heartbeats = {}
        self.expected_ids = {worker: {row['sample_id'] for row in config['trace_rows']
            if f"{row['node']}-w{row['worker']}" == worker} for worker in config['expected_workers']}
        self.t0 = None
        self.measurement_end = None
        self.aborted = None
    def state(self):
        with self.lock:
            now = time.time()
            workers, nodes = {}, {}
            for worker in self.config['expected_workers']:
                node = worker.rsplit('-w', 1)[0]
                heartbeat = self.heartbeats.get(node)
                reported = worker in self.reports
                sample = dict(heartbeat['workers'][worker]) if heartbeat else {}
                expected = len(self.expected_ids[worker])
                age = max(0, now-heartbeat['received_epoch_s']) if heartbeat else None
                sample.update(expected_requests=expected, reported=reported,
                    completed_requests=self.reports[worker] if reported else sample.get('completed_requests', 0),
                    source='report' if reported else ('heartbeat' if heartbeat else 'unavailable'),
                    heartbeat_age_s=round(age, 3) if age is not None else None,
                    stale=not reported and (age is None or age > 90))
                workers[worker] = sample
                subtotal = nodes.setdefault(node, {'completed_requests': 0, 'expected_requests': 0,
                    'reported_workers': 0, 'expected_workers': 0, 'stale_workers': 0})
                subtotal['completed_requests'] += sample['completed_requests']
                subtotal['expected_requests'] += expected
                subtotal['reported_workers'] += int(reported)
                subtotal['expected_workers'] += 1
                subtotal['stale_workers'] += int(sample['stale'])
            return {'run_id': self.config['run_id'], 'ready': sorted(self.ready),
                'expected': self.config['expected_workers'], 'reported': sorted(self.reports),
                't0_server_epoch_s': self.t0, 'measurement_end_server_epoch_s': self.measurement_end, 'aborted': self.aborted,
                'progress': {'completed_requests': sum(row['completed_requests'] for row in workers.values()),
                    'expected_requests': len(self.config['trace_rows']), 'workers': workers, 'nodes': nodes,
                    'all_workers_reported': len(self.reports) == len(self.config['expected_workers']),
                    'heartbeat_stale_after_s': 90,
                    'note': 'Heartbeat counts are provisional; only validated full reports establish completion.'}}
    def heartbeat(self, body):
        with self.lock:
            node = body['node']
            if body['run_id'] != self.config['run_id'] or node not in self.config['nodes']:
                raise ValueError('Wrong run ID or unexpected node')
            for key in ('bundle_id', 'trace_sha256'):
                if body[key] != self.config[key]:
                    raise ValueError(key+' mismatch')
            expected = {worker for worker in self.config['expected_workers'] if worker.startswith(node+'-w')}
            incoming = body['workers']
            if not isinstance(incoming, dict) or set(incoming) != expected:
                raise ValueError('Heartbeat worker set differs from node assignment')
            normalized = {}
            for worker, sample in incoming.items():
                if not isinstance(sample, dict):
                    raise ValueError('Invalid heartbeat worker record')
                limit = len(self.expected_ids[worker])
                item = {}
                for key in ('completed_requests', 'metric_rows', 'invalid_rows', 'duplicate_rows', 'partial_lines'):
                    value = sample[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError('Invalid heartbeat count: '+key)
                    item[key] = value
                if item['completed_requests'] > limit or item['completed_requests'] > item['metric_rows']:
                    raise ValueError('Heartbeat exceeds assigned requests or metric rows')
                code = sample.get('process_returncode')
                if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
                    raise ValueError('Invalid worker return code')
                if sample.get('state') not in ('running', 'exited', 'failed', 'not_started'):
                    raise ValueError('Invalid heartbeat process state')
                errors = sample.get('read_errors', [])
                if not isinstance(errors, list) or len(errors) > 4 or not all(isinstance(error, str) for error in errors):
                    raise ValueError('Invalid heartbeat read errors')
                item.update(process_returncode=code, state=sample['state'], read_errors=errors)
                normalized[worker] = item
            self.heartbeats[node] = {'received_epoch_s': time.time(), 'workers': normalized}
            self.save()
            return self.state()
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
                self.t0 = time.time()+float(self.config.get('idle_baseline_s',60))+5
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
            expected_ids = self.expected_ids[worker]
            if not isinstance(body['records'], list):
                raise ValueError('Expected a list of request records')
            for record in body['records']:
                error = metric_record_error(record, expected_ids)
                if error:
                    raise ValueError('Invalid completion record: '+error)
            ids = [record['task_id'] for record in body['records']]
            if set(ids) != expected_ids or len(ids) != len(expected_ids):
                raise ValueError('Incomplete or duplicate worker records')
            self.reports[worker] = len(ids)
            write_json(self.output/'reports'/(worker+'.json'), body)
            if len(self.reports) == len(self.config['expected_workers']):
                completion = max(r['completion_s'] for p in (self.output/'reports').glob('*.json')
                    for r in json.loads(p.read_text(encoding='utf-8'))['records'])
                self.measurement_end = self.t0 + completion
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
    @app.post('/experiment/heartbeat')
    def heartbeat(body: dict):
        try:
            return coordinator.heartbeat(body)
        except (ValueError, KeyError, TypeError) as exc:
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
