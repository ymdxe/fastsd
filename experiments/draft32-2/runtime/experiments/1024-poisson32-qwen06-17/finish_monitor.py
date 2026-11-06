"""Finalize automatically after the experiment, without launching any model checks."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import ROOT,write_json

write_json(ROOT/'finish-monitor.json',{'pid':os.getpid(),'epoch':time.time()})
while True:
    p=ROOT/'status.json'
    try:status=json.loads(p.read_text()).get('status') if p.exists() else None
    except json.JSONDecodeError:status=None
    if status=='completed':break
    time.sleep(30)
# Allow the main controller's final report and CPU stop jobs to finish.
time.sleep(15)
try:
    subprocess.run([sys.executable,str(ROOT/'finalize.py')],cwd=ROOT,check=True)
    pid=json.loads((ROOT/'control-launch.json').read_text())['pid']
    cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and str(ROOT/'coordinator.py').encode() in cmd.read_bytes():os.kill(pid,signal.SIGTERM)
except BaseException as exc:
    write_json(ROOT/'finalization.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'epoch':time.time()})
    raise
