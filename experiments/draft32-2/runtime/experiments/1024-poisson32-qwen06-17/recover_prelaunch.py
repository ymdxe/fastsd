"""Archive and retry the setup-only NVML idle-check failure; no inference ran."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT,write_json

cell=ROOT/'fastsd-g20-l1'
assert not (cell/'target/process.json').exists()
state=json.loads((cell/'control.json').read_text())
assert not state['ready'] and not state['results'] and not state['done']
pid=json.loads((ROOT/'control-launch.json').read_text())['pid']
assert str(ROOT/'coordinator.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    if not Path(f'/proc/{pid}').exists():break
    time.sleep(.1)
dest=ROOT/'setup-attempts/nvml-reserved-memory';dest.mkdir(parents=True,exist_ok=False)
for p in (ROOT/'status.json',cell/'status.json',cell/'control.json'):
    if p.exists():p.rename(dest/(('campaign-' if p.parent==ROOT else 'run-')+p.name))
write_json(cell/'control.json',dict(state,abort=False))
subprocess.run([sys.executable,str(ROOT/'start_control.py')],check=True)
write_json(dest/'reason.json',{'reason':'NVML v1 counts driver reservation; nvidia-smi confirms idle GPU1. No Target or draft job had started.','epoch':time.time()})
