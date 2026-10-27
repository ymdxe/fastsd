"""Perform essential setup checks, package, and start authenticated control."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
for name in ('validate_config.py','bundle.py','start_control.py','check_control.py','report.py'):
    subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,check=True)
(ROOT/'setup-status.json').write_text(json.dumps({'status':'ready_to_launch','epoch':time.time(),'cells':6,'model_smokes':0},indent=2)+'\n')
print('GAMMA4_READY_TO_LAUNCH: six cells; FP32 both ends; SpecEdge single chain; no model smoke',flush=True)
