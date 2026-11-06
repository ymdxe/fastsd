"""Common experiment protocol. All intervals use each client's monotonic clock."""
import json
import os
from pathlib import Path
import time
import urllib.request


def rpc(url, route, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url.rstrip('/') + route, data=data,
        headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def clock_sample(url, samples=7):
    observations = []
    for _ in range(samples):
        start = time.monotonic()
        server = rpc(url, '/clock/sync')['cloud_time_ns'] / 1e9
        finish = time.monotonic()
        observations.append({'rtt_s': finish-start,
            'server_minus_local_monotonic_s': server-(start+finish)/2})
    best = min(observations, key=lambda row: row['rtt_s'])
    return {**best, 'uncertainty_s': best['rtt_s']/2, 'samples': observations}


def wait_for_start(args):
    url = args.server_url
    identity = os.environ['FASTSD_SUITE_WORKER']
    run_id = os.environ['FASTSD_SUITE_RUN_ID']
    config = rpc(url, '/experiment/info')
    if config['run_id'] != run_id:
        raise RuntimeError('Wrong Target run: ' + config['run_id'])
    calibration = clock_sample(url, samples=11)
    limit_ms=float(config.get('clock_uncertainty_limit_ms',100))
    Path(args.exp_name,'clock-calibration.json').write_text(json.dumps({'clock':calibration,'limit_ms':limit_ms},indent=2),encoding='utf-8')
    if calibration['uncertainty_s']*1000 > limit_ms:
        raise RuntimeError(f'Clock uncertainty {calibration["uncertainty_s"]*1000:.3f}ms exceeds {limit_ms}ms')
    rpc(url, '/experiment/register', {'worker': identity, 'run_id': run_id,
        'bundle_id': os.environ['FASTSD_SUITE_BUNDLE_ID'],
        'trace_sha256': os.environ['FASTSD_SUITE_TRACE_SHA256'], 'clock': calibration})
    print('READY ' + identity + '; waiting for common T0', flush=True)
    deadline = time.monotonic() + 1800
    while True:
        state = rpc(url, '/experiment/state')
        if state.get('aborted'):
            raise RuntimeError('Experiment aborted: ' + str(state['aborted']))
        if state.get('t0_server_epoch_s') is not None:
            origin = state['t0_server_epoch_s'] - calibration['server_minus_local_monotonic_s']
            if origin < time.monotonic():
                raise RuntimeError('Common T0 was missed; use a new run ID')
            break
        if time.monotonic() > deadline:
            raise TimeoutError('READY barrier exceeded 30 minutes; inspect missing workers')
        time.sleep(0.2)
    metadata = {'worker': identity, 'run_id': run_id, 'local_t0_monotonic_s': origin,
        't0_server_epoch_s': state['t0_server_epoch_s'], 'clock': calibration}
    Path(args.exp_name, 'barrier.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    remaining = origin-time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    print('START ' + identity, flush=True)
    return origin


def finish_worker(args):
    records = []
    for path in sorted(Path(args.exp_name).glob('edge_metrics_proc*.jsonl')):
        records.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    rpc(args.server_url, '/experiment/report', {'worker': os.environ['FASTSD_SUITE_WORKER'],
        'run_id': os.environ['FASTSD_SUITE_RUN_ID'], 'records': records}, timeout=120)