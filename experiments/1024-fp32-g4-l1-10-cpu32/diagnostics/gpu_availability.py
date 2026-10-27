import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
(ROOT/'diagnostics/gpu-availability.json').write_text(json.dumps({'epoch':time.time(),'gpus':gpus,'processes':processes},indent=2)+'\n')
