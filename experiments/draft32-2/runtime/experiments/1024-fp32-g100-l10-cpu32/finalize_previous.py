"""Archive an intentional stop and release the old task's private controller."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
record=json.loads((ROOT/'previous-experiment.json').read_text());old=Path(record['root']);run=record['run']
assert json.loads((old/run/'target/exit.json').read_text())['exit_status']==0
control=json.loads((old/run/'control.json').read_text());assert len(control['done'])==2
rows=[json.loads(s) for node in ('node1','node3') for s in (old/run/node/'requests.jsonl').read_text().splitlines() if s.strip()]
record.update(completed_requests_retained=len(rows),target_exit_status=0,status='interrupted_by_user',included_in_new_experiment=False)
(ROOT/'previous-experiment.json').write_text(json.dumps(record,indent=2)+'\n')
previous_status=json.loads((old/run/'status.json').read_text())
(ROOT/'previous-raw-status.json').write_text(json.dumps(previous_status,indent=2)+'\n')
previous_status.update(status='interrupted_by_user',reason='New FP32 multicore gamma100/lambda10 run requested',completed_requests=len(rows))
(old/run/'status.json').write_text(json.dumps(previous_status,indent=2)+'\n')
for node in ('node1','node3'):
    d=json.loads((ROOT/'preflight'/(node+'.json')).read_text());assert d['mem_available_gib']>=80
pid=json.loads((old/'control-launch.json').read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
if cmd.exists() and cmd.read_bytes():
    assert str(old/'coordinator.py').encode() in cmd.read_bytes();os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        p=Path(f'/proc/{pid}/stat')
        if not p.exists() or p.read_text().split()[2]=='Z':break
        time.sleep(.1)
fd=os.open(ROOT/'.control-token',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f:f.write((old/'.control-token').read_text())
gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
process_csv=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
occupied={line.split(',')[0].strip() for line in process_csv.splitlines() if line.strip()}
gpus=[]
for line in gpu_csv.splitlines():
    i,u,n,total,used,util,power=[x.strip() for x in line.split(',')]
    gpus.append({'index':int(i),'uuid':u,'name':n,'total_mib':int(total),'used_mib':int(used),'utilization_pct':int(util),'power_w':float(power)})
free=[g for g in gpus if g['uuid'] not in occupied and g['used_mib']<300]
assert free,'No idle GPU; do not overlap other jobs'
chosen=next((g for g in free if g['index']==1),free[0])
mem=Path('/proc/meminfo').read_text();disk=os.statvfs(ROOT)
report={'epoch':time.time(),'gpus':gpus,'selected_gpu':chosen,'meminfo':mem,'disk_free_gib':disk.f_bavail*disk.f_frsize/1024**3}
(ROOT/'preflight/gpunode2.json').write_text(json.dumps(report,indent=2)+'\n')
