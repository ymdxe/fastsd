"""Native cloud entrypoint with readiness and nonintrusive campaign telemetry."""
import os
from pathlib import Path
import sys
import time
from common import ROOT,write_json

sys.path.insert(0,str(ROOT/'source'))
from telemetry import install_runtime_metrics
install_runtime_metrics()
from src.engine import Decoding

original=Decoding._load_target_model_for_service


def load(self,path,device):
    model=original(self,path,device)
    import torch
    torch.cuda.synchronize()
    write_json(Path(self.args.exp_name)/'target_model_ready.json',{
        'ready':True,'pid':os.getpid(),'epoch':time.time(),'model':path,'dtype':str(model.dtype),
        'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'gpu_name':torch.cuda.get_device_name(0),
        'memory_allocated_bytes':torch.cuda.memory_allocated(),'weight_offload':False})
    return model


Decoding._load_target_model_for_service=load
if __name__=='__main__':
    import threading
    import uvicorn
    import cloud.cloud_service as cloud
    def run_server(app,**kwargs):
        server=uvicorn.Server(uvicorn.Config(app,**kwargs))
        def watch():
            while not server.should_exit:
                if cloud._admission_closed and cloud.worker_proc is not None and not cloud.worker_proc.is_alive():
                    server.should_exit=True;return
                time.sleep(.2)
        threading.Thread(target=watch,daemon=True).start()
        server.run()
        if cloud.worker_proc is not None:cloud.worker_proc.join(timeout=10)
    uvicorn.run=run_server
    cloud.main()
