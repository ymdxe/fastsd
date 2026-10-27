"""Stop only the gamma100 task, retaining its partial request records."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'1024-fp32-g100-l10-cpu32'
sys.path.insert(0,str(PREVIOUS))
from common import control,write_json

os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'
state=json.loads((PREVIOUS/'status.json').read_text())
assert state['status']=='running',state
run=state['cell']['id']
reason='User cancelled gamma100; replace with gamma4 and Poisson rates1/10, SpecEdge single chain'
write_json(ROOT/'previous-experiment.json',{'root':str(PREVIOUS),'run':run,'status_at_stop':state,'reason':reason,'epoch':time.time()})
for node in ('node1','node3'):
    control('set_job',payload={'node':node,'job':{'run':run,'action':'stop'}})
control('abort',payload={'run':run,'error':reason})
write_json(ROOT/'setup-status.json',{'status':'stopping_previous','epoch':time.time()})
print('PREVIOUS_ABORT_REQUESTED',run,flush=True)
