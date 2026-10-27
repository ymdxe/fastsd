"""Preserve the failed attempt, rebuild the transport config, and stage a fresh run."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
assert json.loads((ROOT/'diagnostics/clock-runtime-check.json').read_text())['ok']
run='fastsd-g4-l1'
status=json.loads((ROOT/'status.json').read_text());assert status['status']=='failed'
assert json.loads((ROOT/run/'target/exit.json').read_text())['exit_status']==0
for node in ('node1','node3'):
    assert json.loads((ROOT/'diagnostics'/(node+'.json')).read_text())['agent_stopped']
pid=json.loads((ROOT/'gpu-supervisor.json').read_text())['pid'];path=Path(f'/proc/{pid}/cmdline')
assert not path.exists() or str(ROOT/'run_gpu.py').encode() not in path.read_bytes()
archive=ROOT/'attempts/fastsd-g4-l1-clock-calibration';assert not archive.exists()
archive.parent.mkdir(exist_ok=True)
(ROOT/run).rename(archive)
for filename in ('status.json','campaign.log','results.json','results.csv','RESULTS.zh.md','manifest.json','bundle.tar.gz','bundle.json','gpu-supervisor.json','runtime-validation.json'):
    p=ROOT/filename
    if p.exists():shutil.copy2(p,archive/('campaign-'+filename))
pid=json.loads((ROOT/'control-launch.json').read_text())['pid'];path=Path(f'/proc/{pid}/cmdline')
if path.exists() and path.read_bytes():
    assert str(ROOT/'coordinator.py').encode() in path.read_bytes();os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not path.exists() or not path.read_bytes():break
        time.sleep(.1)
manifest=json.loads((ROOT/'manifest.json').read_text())
history=manifest.get('recovery_history',[])
if manifest.get('recovery'):history.append(manifest['recovery'])
manifest.update(status='prepared',http_timeout_keep_alive_s=7200,recovery_history=history,
                clock_timing_policy={'periodic_calibration_s':60,'unreliable_pull':'mark unavailable; last valid estimate for scheduler only','post_retry':'only explicit422 negative upload before server admission'},
                recovery={'failed_attempt':str(archive.relative_to(ROOT)),'reason':'Negative cross-node pull timing aborted a completed Verify response','restarted_from_beginning':True})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for filename in ('validate_config.py','bundle.py','start_control.py','check_control.py','report.py'):
    subprocess.run([sys.executable,str(ROOT/filename)],cwd=ROOT,check=True)
(ROOT/'setup-status.json').write_text(json.dumps({'status':'recovery_ready_to_launch','epoch':time.time(),'failed_requests_retained':156,'model_smokes':0},indent=2)+'\n')
print('RECOVERY_READY_TO_LAUNCH; failed attempt retained separately; six cells unchanged',flush=True)
