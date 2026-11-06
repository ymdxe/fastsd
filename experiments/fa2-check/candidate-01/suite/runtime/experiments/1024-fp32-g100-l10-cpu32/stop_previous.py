"""Retire only the preceding FP16 campaign after the user's new FP32 request."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'1024-poisson32-qwen06-17'
sys.path.insert(0,str(PREVIOUS))
from common import control,write_json
os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'
state=json.loads((PREVIOUS/'status.json').read_text())
assert state['status']=='running'
run=state['cell']['id']
write_json(ROOT/'previous-experiment.json',{'root':str(PREVIOUS),'run':run,'status_at_stop':state,'reason':'New FP32 multicore gamma100/lambda10 experiment requested','epoch':time.time()})
monitor=PREVIOUS/'finish-monitor-launch.json'
if monitor.exists():
    pid=json.loads(monitor.read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and str(PREVIOUS/'finish_monitor.py').encode() in cmd.read_bytes():os.kill(pid,signal.SIGTERM)
control('abort',payload={'run':run,'error':'User requested a new FP32 multicore gamma100/lambda10 experiment; preserve this FP16 partial run and stop.'})
write_json(ROOT/'setup-status.json',{'status':'stopping_previous_campaign','previous_run':run,'epoch':time.time()})
