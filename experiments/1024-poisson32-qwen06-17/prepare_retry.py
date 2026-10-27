"""Keep the partial failed run and prepare a fresh first-cell attempt."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import ROOT,RELEASE_CACHE,annotate,write_json

RELEASE_CACHE['unit']=1000
for key in ('EXP_RUN','EXP_NODE','EXP_GAMMA'):os.environ.pop(key,None)
r=annotate({'ttft_ms':100,'request_e2e_ms':1000,'arrival_lag_ms':500,'completion_s':50},run='unit',node='node1',gamma=20)
assert r['arrival_ttft_ms']==600 and r['arrival_request_e2e_ms']==1500 and r['completion_epoch']==1050 and r['gamma']==20
for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
directory=ROOT/'fastsd-g20-l1'
assert json.loads((directory/'status.json').read_text())['status']=='failed'
assert json.loads((directory/'target/exit.json').read_text())['exit_status']==0
state=json.loads((directory/'control.json').read_text())
assert len(state['done'])==2 and all(d['finished'] for d in state['done'].values())
matrix=json.loads((ROOT/'matrix.json').read_text())
assert matrix[0]['id']=='fastsd-g20-l1'
matrix[0].update(id='fastsd-g20-l1-a2',retry_of='fastsd-g20-l1')
write_json(ROOT/'matrix.json',matrix)
write_json(ROOT/'attempts.json',[{'run':'fastsd-g20-l1','status':'excluded_partial','reason':'Result uploader missing EXP_RUN; inference exited cleanly; no final performance claim.'}])
names=['node_agent.py','common.py','campaign.py','start_control.py','matrix.json','README.zh.md','probe_nvml.py','recover_prelaunch.py','prepare_retry.py']
old=json.loads((ROOT/'manifest.json').read_text())
payload={name:{'text':(ROOT/name).read_text(),'old_sha256':old['hashes'].get(name),'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in names}
manifest=dict(old)
manifest['hashes']=dict(old['hashes'],**{k:v['sha256'] for k,v in payload.items()})
manifest['bootstrap_bundle_sha256']=json.loads((ROOT/'bundle.json').read_text())['sha256']
manifest['harness_revision']=2
manifest['initial_partial_attempt']='fastsd-g20-l1 excluded after uploader context failure'
write_json(ROOT/'manifest.json',manifest)
payload['manifest.json']={'text':(ROOT/'manifest.json').read_text(),'old_sha256':None,'sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()}
write_json(ROOT/'sync2.json',payload)
pid=json.loads((ROOT/'control-launch.json').read_text())['pid']
assert str(ROOT/'coordinator.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    if not Path(f'/proc/{pid}').exists():break
    time.sleep(.1)
subprocess.run([sys.executable,str(ROOT/'start_control.py')],check=True)
write_json(ROOT/'checks/uploader.json',{'passed':True,'explicit_run_without_parent_env':True,'arrival_ttft_ms':r['arrival_ttft_ms'],'completion_epoch':r['completion_epoch'],'harness_revision':2})
