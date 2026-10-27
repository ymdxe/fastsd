"""Verify authenticated control/metadata transport before starting model work."""
import os
import time
from common import ROOT,control,write_json

os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'
for _ in range(50):
    try:
        if control('health').get('ok'):break
    except Exception:pass
    time.sleep(.1)
else:raise RuntimeError('Controller not ready')
result=control('metadata',payload={'run':'fastsd-g4-l1','node':'node1','files':{}})
assert result['ok'] and result['completed']==0
state=control('state',run='fastsd-g4-l1')
assert not state['ready'] and not state['results'] and state['release_epoch'] is None
write_json(ROOT/'control-check.json',{'ok':True,'authenticated_control':True,'metadata_endpoint':True,'no_model_smoke':True,'epoch':time.time()})
