"""Record an intentional cancellation and free the preceding private controller."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'1024-fp32-g100-l10-cpu32'
sys.path.insert(0,str(OLD))
from common import write_json

record=json.loads((ROOT/'previous-experiment.json').read_text());run=record['run']
assert json.loads((OLD/run/'target/exit.json').read_text())['exit_status']==0
state=json.loads((OLD/run/'control.json').read_text());assert set(state['done'])=={'node1','node3'}
for filename in ('gpu-supervisor.json',run+'/target/process.json'):
    pid=json.loads((OLD/filename).read_text())['pid'];path=Path(f'/proc/{pid}/cmdline')
    assert not path.exists() or str(OLD).encode() not in path.read_bytes(),'Old GPU process still running'
rows=[json.loads(line) for path in (OLD/run).glob('node*/requests.jsonl') for line in path.read_text().splitlines() if line.strip()]
record.update(status='interrupted_by_user',completed_requests_retained=len(rows),target_exit_status=0,included_in_new_experiment=False)
write_json(ROOT/'previous-experiment.json',record)
write_json(ROOT/'previous-raw-status.json',json.loads((OLD/'status.json').read_text()))
for path in (OLD/'status.json',OLD/run/'status.json'):
    value=json.loads(path.read_text());value.update(status='interrupted_by_user',reason=record['reason'],completed_requests=len(rows),epoch=time.time());write_json(path,value)
manifest=json.loads((OLD/'manifest.json').read_text());manifest.update(status='interrupted_by_user',completed_requests_retained=len(rows));write_json(OLD/'manifest.json',manifest)
pid=json.loads((OLD/'control-launch.json').read_text())['pid'];path=Path(f'/proc/{pid}/cmdline')
if path.exists() and path.read_bytes():
    assert str(OLD/'coordinator.py').encode() in path.read_bytes();os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not path.exists() or not path.read_bytes():break
        time.sleep(.1)
fd=os.open(ROOT/'.control-token',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as stream:stream.write((OLD/'.control-token').read_text())
gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
process_csv=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
occupied={line.split(',')[0].strip() for line in process_csv.splitlines() if line.strip()}
gpus=[]
for line in gpu_csv.splitlines():
    i,u,n,total,used,util,power=[x.strip() for x in line.split(',')]
    gpus.append({'index':int(i),'uuid':u,'name':n,'total_mib':int(total),'used_mib':int(used),'utilization_pct':int(util),'power_w':float(power)})
chosen=next(g for g in gpus if g['index']==1)
assert chosen['uuid']=='GPU-422501a2-5189-7f5b-5ce6-95464276b27f' and chosen['uuid'] not in occupied and chosen['used_mib']<300,'GPU1 is not idle'
disk=os.statvfs(ROOT)
write_json(ROOT/'preflight/gpunode2.json',{'epoch':time.time(),'gpus':gpus,'selected_gpu':chosen,'disk_free_gib':disk.f_bavail*disk.f_frsize/1024**3})
import report
report.main()
print('PREVIOUS_RETIRED',len(rows),'requests retained; GPU1 idle',flush=True)
