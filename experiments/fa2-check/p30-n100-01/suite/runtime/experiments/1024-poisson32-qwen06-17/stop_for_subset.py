"""Stop only this campaign's obsolete run and monitors for a user scope change."""
import json
import os
from pathlib import Path
import signal
import time
from common import ROOT,control,write_json

os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'
directory=ROOT/'subset-g200-g500-l50';directory.mkdir(exist_ok=False)
state=json.loads((ROOT/'status.json').read_text());active=state['cell']['id']
assert state['cell']['gamma']==20 and state['cell']['rate']==1
write_json(directory/'previous-status.json',state)
stopped=[]
for name,script in [('rate-update-1-10-50/launch.json','rate_handoff.py'),('finish-monitor-launch.json','finish_monitor.py')]:
    pid=json.loads((ROOT/name).read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists():
        assert str(ROOT/script).encode() in cmd.read_bytes()
        os.kill(pid,signal.SIGTERM);stopped.append({'pid':pid,'script':script})
control('abort',payload={'run':active,'error':'User requested only gamma200/500 at lambda50; preserve this partial gamma20/lambda1 run and stop it.'})
write_json(directory/'transition.json',{'status':'stopping_previous_run','previous_run':active,'stopped_monitors':stopped,'requested_gamma':[200,500],
    'requested_rates':[50],'expected_runs':6,'epoch':time.time()})
