"""Campaign control and immutable, auditable configuration."""
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RELEASE_CACHE = {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def control(endpoint, payload=None, **query):
    url = os.environ.get('CAMPAIGN_CONTROL', 'http://127.0.0.1:28634')
    url += '/' + endpoint + ('?' + urllib.parse.urlencode(query) if query else '')
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        'Authorization': 'Bearer ' + (ROOT / '.control-token').read_text().strip(),
        'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=30) as response:
        return json.loads(response.read())


def barrier(endpoint):
    started = time.monotonic()
    while True:
        reply = control(endpoint, run=os.environ['EXP_RUN'], worker=os.environ['EXP_WORKER'])
        if reply.get('abort'):
            raise RuntimeError('Campaign run aborted: ' + str(reply))
        if reply.get('release'):
            return reply
        if time.monotonic() - started > 1800:
            raise TimeoutError(endpoint)
        time.sleep(.2)


def origin_monotonic(clock_offset_ns=0):
    s = control('state', run=os.environ['EXP_RUN'])
    return time.monotonic() + s['release_epoch'] - (time.time() + clock_offset_ns / 1e9)


def annotate(row,run=None,node=None,gamma=None):
    row = dict(row)
    lag = float(row.get('arrival_lag_ms', 0))
    row['service_ttft_ms'] = row['ttft_ms']
    row['service_request_e2e_ms'] = row['request_e2e_ms']
    row['arrival_ttft_ms'] = lag + row['ttft_ms'] if row['ttft_ms'] is not None else None
    row['arrival_request_e2e_ms'] = lag + row['request_e2e_ms']
    run=run or os.environ['EXP_RUN']
    if run not in RELEASE_CACHE:RELEASE_CACHE[run]=control('state',run=run)['release_epoch']
    row['completion_epoch'] = RELEASE_CACHE[run]+float(row['completion_s'])
    row['node'] = node or os.environ['EXP_NODE']
    row['gamma'] = int(gamma if gamma is not None else os.environ['EXP_GAMMA'])
    return row
