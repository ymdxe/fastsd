"""Observe real CPU forwards and GPU execution without changing their results."""
import functools
import time
from diagnostic_trace import trace

_task_id=None
def set_task(task_id):
    global _task_id
    _task_id=task_id

def observe_cpu_model(model):
    original=model.forward
    @functools.wraps(original)
    def forward(*args,**kwargs):
        wall=time.monotonic_ns()
        cpu=time.process_time_ns()
        inputs=args[0] if args else kwargs['input_ids']
        try:
            return original(*args,**kwargs)
        finally:
            trace.emit('cpu_model_forward',task_id=_task_id,start_ns=wall,
                wall_ns=time.monotonic_ns()-wall,process_cpu_ns=time.process_time_ns()-cpu,
                input_shape=list(inputs.shape))
    model.forward=forward

def observe_gpu_call(original,operation,device):
    @functools.wraps(original)
    def forward(*args,**kwargs):
        import torch
        trace.gpu_init(torch,device)
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        wall=time.monotonic_ns()
        start.record(torch.cuda.current_stream(device))
        try:
            return original(*args,**kwargs)
        finally:
            end.record(torch.cuda.current_stream(device))
            trace.gpu_span('measured_target_compute',start,end,primitive=operation)
            trace.emit('target_compute_host',primitive=operation,start_ns=wall,
                wall_ns=time.monotonic_ns()-wall)
    return forward
