import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import control
os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'
for node in ('node1','node3'):
    control('set_job',payload={'node':node,'job':{'run':'fastsd-g4-l1','action':'stop'}})
print('FAILED_RUN_AGENTS_STOP_REQUESTED',flush=True)
