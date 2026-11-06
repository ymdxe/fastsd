"""Choose idle GPU2 before any requests start; preserve the prelaunch record."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import control,write_json
os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'
state=json.loads((ROOT/'status.json').read_text());assert state['status']=='waiting_for_gpu'
assert not (ROOT/'fastsd-g4-l1/target/process.json').exists()
for node in ('node1','node3'):control('set_job',payload={'node':node,'job':{'run':'fastsd-g4-l1','action':'stop'}})
time.sleep(4)
for filename,program in [('gpu-supervisor.json','run_gpu.py'),('control-launch.json','coordinator.py')]:
    pid=json.loads((ROOT/filename).read_text())['pid'];p=Path(f'/proc/{pid}/cmdline')
    if p.exists() and p.read_bytes():
        assert str(ROOT/program).encode() in p.read_bytes();os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not p.exists() or not p.read_bytes():break
            time.sleep(.1)
archive=ROOT/'attempts/prelaunch-gpu2-change';assert not archive.exists()
(ROOT/'fastsd-g4-l1').rename(archive)
for name in ('manifest.json','bundle.tar.gz','bundle.json','status.json'):
    shutil.copy2(ROOT/name,archive/('campaign-'+name))
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
occupied={line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
available=[]
for line in gpus.splitlines():
    i,u,n,total,used,util,power=[x.strip() for x in line.split(',')]
    available.append({'index':int(i),'uuid':u,'name':n,'total_mib':int(total),'used_mib':int(used),'utilization_pct':int(util),'power_w':float(power)})
chosen=next(g for g in available if g['index']==2)
assert chosen['uuid']=='GPU-131d576e-3215-d317-dab1-4b1e4fbc0fca' and chosen['uuid'] not in occupied and chosen['used_mib']<300
manifest=json.loads((ROOT/'manifest.json').read_text());manifest.update(physical_gpu=2,gpu_uuid=chosen['uuid'],gpu_selection_reason='GPU1 occupied before formal admission; use idle GPU2 for all6 cells')
write_json(ROOT/'manifest.json',manifest)
write_json(ROOT/'preflight/gpunode2.json',{'epoch':time.time(),'gpus':available,'selected_gpu':chosen})
path=ROOT/'campaign.py';text=path.read_text();assert 'GPU=1' in text
text=text.replace('GPU=1',"GPU=int(json.loads((ROOT/'manifest.json').read_text())['physical_gpu'])",1)
text=text.replace("GPU_UUID='GPU-422501a2-5189-7f5b-5ce6-95464276b27f'","GPU_UUID=json.loads((ROOT/'manifest.json').read_text())['gpu_uuid']",1);path.write_text(text)
path=ROOT/'finalize.py';text=path.read_text();assert "'GPU-422501a2-5189-7f5b-5ce6-95464276b27f'" in text
text=text.replace("'GPU-422501a2-5189-7f5b-5ce6-95464276b27f'","json.loads((ROOT/'manifest.json').read_text())['gpu_uuid']",1);path.write_text(text)
path=ROOT/'report.py';text=path.read_text();assert 'gpunode2 GPU1' in text;text=text.replace('gpunode2 GPU1','gpunode2 GPU2');path.write_text(text)
path=ROOT/'README.zh.md';text=path.read_text().replace('空闲GPU1','空闲GPU2');path.write_text(text+'\n时钟处理修复及失败记录见diagnostics/CLOCK-RECOVERY.zh.md；本轮6组统一使用GPU2，GPU1上的失败尝试不混入正式统计。\n')
for name in ('validate_config.py','bundle.py','start_control.py','check_control.py','report.py'):
    subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,check=True)
write_json(ROOT/'setup-status.json',{'status':'gpu2_ready_to_launch','epoch':time.time(),'physical_gpu':2,'cells':6,'model_smokes':0})
print('GPU2_READY_TO_LAUNCH; zero formal requests admitted before retarget',flush=True)
