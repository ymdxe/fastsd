"""Target entry point: actual load readiness, long keepalive, graceful exit."""
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
import torch
from src.engine import Decoding

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
original_load = Decoding._load_target_model_for_service


def load(self, path, device):
    model = original_load(self, path, device)
    if model.dtype != torch.float32:
        raise RuntimeError('Target must be FP32')
    torch.cuda.synchronize()
    ready = {'ready': True, 'dtype': str(model.dtype), 'pid': os.getpid(),
             'visible_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'model': path}
    (Path(self.args.exp_name) / 'target_model_ready.json').write_text(json.dumps(ready) + '\n')
    print('MANUAL_SMOKE_TARGET_READY ' + json.dumps(ready), flush=True)
    return model


Decoding._load_target_model_for_service = load

if __name__ == '__main__':
    import uvicorn
    import cloud.cloud_service as cloud

    def run_server(app, **kwargs):
        server = uvicorn.Server(uvicorn.Config(app, timeout_keep_alive=7200, **kwargs))

        def watch():
            while not server.should_exit:
                if cloud.worker_proc is not None and not cloud.worker_proc.is_alive():
                    server.should_exit = True
                    return
                time.sleep(0.2)

        threading.Thread(target=watch, daemon=True).start()
        server.run()
        if cloud.worker_proc is not None:
            cloud.worker_proc.join(timeout=10)

    uvicorn.run = run_server
    cloud.main()
