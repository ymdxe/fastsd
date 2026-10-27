"""Observe cache moves without adding CUDA synchronization or changing policy."""
import functools
import json
import os
from pathlib import Path
import threading
import time

LOCK=threading.Lock()
METRICS=None


def emit(row):
    directory=os.environ.get('EXP_TARGET_OUTPUT')
    if directory:
        row.update(pid=os.getpid(),thread=threading.current_thread().name)
        with LOCK:
            with (Path(directory)/'kv_moves.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def trace_cache_move(fn):
    @functools.wraps(fn)
    def wrapped(cache,device):
        tensors=[t for pair in cache.to_legacy_cache() for t in pair]
        source=str(tensors[0].device) if tensors else 'empty'
        target=str(device)
        if source==target:return fn(cache,device)
        start=time.perf_counter();epoch=time.time()
        try:
            out=fn(cache,device)
        except BaseException as exc:
            emit({'start_epoch':epoch,'host_wall_ms':(time.perf_counter()-start)*1000,'source':source,'destination':target,'ok':False,'error':repr(exc)})
            raise
        emit({'start_epoch':epoch,'host_wall_ms':(time.perf_counter()-start)*1000,'source':source,'destination':target,
              'bytes':sum(t.numel()*t.element_size() for t in tensors),'ok':True,
              'asynchronous_prefetch':threading.current_thread().name=='fastsd-kv-prefetch'})
        return out
    return wrapped


def trace_verify_demand(proc_ids,resident):
    emit({'event':'verify_demand','start_epoch':time.time(),'requests':len(proc_ids),
          'prefetch_demand_hits':len(set(proc_ids)&set(resident))})


def install_runtime_metrics():
    import src.scheduler_runtime as runtime
    original=runtime.KVPrefetchWorker.__init__
    def init(self,**kwargs):
        global METRICS
        METRICS=kwargs['metrics']
        return original(self,**kwargs)
    runtime.KVPrefetchWorker.__init__=init
    def sample():
        from common import write_json
        while True:
            if METRICS is not None:
                write_json(Path(os.environ['EXP_TARGET_OUTPUT'])/'runtime_metrics.json',dict(METRICS))
            time.sleep(10)
    threading.Thread(target=sample,name='campaign-metrics',daemon=True).start()
