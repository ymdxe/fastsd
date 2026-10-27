"""Authenticated loopback-only task control; no unauthenticated model endpoint."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse
from common import ROOT, write_json

LOCK = threading.RLock()
STATES = {}
JOBS = {}
ALLOWED_RUNS = {r['id'] for r in json.loads((ROOT/'matrix.json').read_text())}
TOKEN = (ROOT/'.control-token').read_text().strip()


def allowed_runs():
    return {r['id'] for r in json.loads((ROOT/'matrix.json').read_text())}


def state(run):
    if run not in allowed_runs():
        raise ValueError('Invalid run')
    if run not in STATES:
        p = ROOT/run/'control.json'
        STATES[run] = json.loads(p.read_text()) if p.exists() else {
            'ready': {}, 'clock': {}, 'done': {}, 'results': {}, 'clock_owner': None,
            'release_epoch': None, 'abort': False}
    return STATES[run]


def save(run):
    # Persist identifiers, not repeated full outputs on every heartbeat.
    write_json(ROOT/run/'control.json', state(run))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, code=200):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if not hmac.compare_digest(self.headers.get('Authorization',''), 'Bearer '+TOKEN):
            self.reply({'error':'authentication required'},401)
            return False
        return True

    def do_GET(self):
        if not self.authorized(): return
        p = urlparse(self.path)
        q = {k:v[0] for k,v in parse_qs(p.query).items()}
        with LOCK:
            if p.path == '/health': return self.reply({'ok': True, 'epoch':time.time()})
            if p.path == '/bundle':
                data = (ROOT/'bundle.tar.gz').read_bytes()
                self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                return
            if p.path == '/job':
                return self.reply(JOBS.get(q.get('node'), {}))
            try:
                run = q['run']; s = state(run)
            except (KeyError,ValueError): return self.reply({'error':'invalid run'},400)
            worker = q.get('worker','')
            if worker and worker not in {f'{n}-{i}' for n in ('node1','node3') for i in range(16)}:
                return self.reply({'error':'invalid worker'},400)
            if p.path == '/state':
                return self.reply(s)
            if p.path == '/model_ready':
                s['ready'].setdefault(worker,time.time()); save(run)
                return self.reply({'release':len(s['ready'])==32,'abort':s['abort']})
            if p.path == '/clock_acquire':
                if s['clock_owner'] in (None,worker): s['clock_owner']=worker
                return self.reply({'release':s['clock_owner']==worker,'abort':s['abort']})
            if p.path == '/clock_release':
                if s['clock_owner']==worker:s['clock_owner']=None
                return self.reply({'ok':True})
            if p.path == '/clock_done':
                s['clock'].setdefault(worker,time.time())
                if len(s['clock'])==32 and s['release_epoch'] is None:s['release_epoch']=time.time()+5
                save(run)
                return self.reply({'release':s['release_epoch'] is not None,'abort':s['abort'],'epoch':s['release_epoch']})
            return self.reply({'error':'unknown endpoint'},404)

    def do_POST(self):
        if not self.authorized(): return
        size=int(self.headers.get('Content-Length',0))
        if not 0<size<32_000_000:return self.reply({'error':'invalid size'},400)
        try:
            v=json.loads(self.rfile.read(size))
            with LOCK:
                if self.path=='/set_job':
                    if v['node'] not in ('node1','node3'):raise ValueError('node')
                    if v['job'].get('run') not in allowed_runs():raise ValueError('run')
                    JOBS[v['node']]=v['job'];return self.reply({'ok':True})
                run=v['run'];s=state(run)
                if self.path=='/abort':s['abort']=v.get('error',True)
                elif self.path=='/results':
                    for row in v['rows']:
                        key=str(row['global_sample_index'])
                        if not 0<=int(key)<1319:raise ValueError('sample')
                        if key not in s['results']:
                            s['results'][key]={'node':v['node'],'completion_epoch':row['completion_epoch']}
                            dest=ROOT/run/v['node'];dest.mkdir(parents=True,exist_ok=True)
                            with (dest/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                elif self.path=='/node_status':
                    write_json(ROOT/run/v['node']/'status.json',v)
                    if v.get('finished'):s['done'][v['node']]=v
                else:return self.reply({'error':'unknown endpoint'},404)
                save(run);self.reply({'ok':True,'completed':len(s['results'])})
        except (KeyError,ValueError,TypeError,json.JSONDecodeError) as exc:self.reply({'error':str(exc)},400)


if __name__=='__main__':
    write_json(ROOT/'coordinator.json',{'pid':os.getpid(),'host':'127.0.0.1','port':18634})
    ThreadingHTTPServer(('127.0.0.1',18634),Handler).serve_forever()
