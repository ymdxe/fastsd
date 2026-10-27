"""Persistent native draft, global Poisson timeline and per-worker session reuse."""
import json
import os
from pathlib import Path
import sys
import time
from common import ROOT, annotate, barrier, control, origin_monotonic, write_json
from clock_runtime import ClockRuntimeMixin

sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'specedge'))
import torch
import edge.edge as edge

CLOCK_OFFSET = 0
_BaseEdgeClient=edge.EdgeClient


def bind(cpu,numa):
    # Placement is shared with SpecEdge's already reviewed CPU binding helper.
    import ctypes
    os.sched_setaffinity(0,set(cpu if isinstance(cpu,list) else [cpu]))
    lib=ctypes.CDLL('libnuma.so.1');lib.numa_allocate_nodemask.restype=ctypes.c_void_p
    lib.numa_bitmask_setbit.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    lib.numa_set_membind.argtypes=[ctypes.c_void_p];lib.numa_bitmask_free.argtypes=[ctypes.c_void_p]
    mask=lib.numa_allocate_nodemask()
    if not mask:raise RuntimeError('NUMA allocation failed')
    try:lib.numa_bitmask_setbit(mask,numa);lib.numa_set_membind(mask)
    finally:lib.numa_bitmask_free(mask)


class Client(ClockRuntimeMixin,edge.EdgeClient):
    def _synchronize_runtime_clock(self,samples=5):
        return _BaseEdgeClient.synchronize_clock(self,samples)

    def synchronize_clock(self,samples=5):
        global CLOCK_OFFSET
        barrier('clock_acquire')
        try:r=super().synchronize_clock(samples)
        finally:control('clock_release',run=os.environ['EXP_RUN'],worker=os.environ['EXP_WORKER'])
        CLOCK_OFFSET=r['clock_offset_ns']
        self._clock_refresh_at=time.monotonic()
        write_json(Path(os.environ['EXP_OUTPUT'])/'clock.json',r)
        barrier('clock_done')
        return r

    def health(self):
        r=super().health()
        if not self.timing_priority:self.synchronize_clock()
        return r

    def init_session(self):
        if not hasattr(self,'_campaign_session'):self._campaign_session=super().init_session()
        return self._campaign_session

    def _attach_pull(self,response,when):
        # Both methods calibrate clocks; retain baseline scheduler protocol.
        return super()._attach_pull(response,when)

    def _post(self,path,payload):
        start=time.monotonic()
        r=super()._post(path,payload)
        if path=='/verify':
            row={'endpoint':path,'task_id':payload.get('task_id'),'start_epoch':time.time()-(time.monotonic()-start),
                 'http_ms':(time.monotonic()-start)*1000,'cloud_total_ms':r.get('cloud_total_ms'),'verify_ms':r.get('verify_ms')}
            with (Path(os.environ['EXP_OUTPUT'])/'rpc.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return r


edge.EdgeClient=Client


class Barrier:
    def wait(self):pass


class Origin:
    @property
    def value(self):return origin_monotonic(CLOCK_OFFSET)
    @value.setter
    def value(self,v):pass


class Runner(edge.EdgeRunner):
    def color_print(self,message,level=1):
        if level<3:super().color_print(message,level)

    def _encode_input_ids(self,tokenizer,text):
        return tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')

    def _load_draft_model(self,path,device):
        m=super()._load_draft_model(path,device)
        if m.device.type!='cpu' or {p.dtype for p in m.parameters()}!={torch.float32}:raise RuntimeError('draft device/dtype')
        write_json(Path(os.environ['EXP_OUTPUT'])/'ready.json',{'pid':os.getpid(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'dtype':str(m.dtype),'torch':torch.__version__,'affinity':sorted(os.sched_getaffinity(0)),'epoch':time.time()})
        barrier('model_ready')
        return m


if __name__=='__main__':
    worker=int(os.environ['EXP_LOCAL_WORKER'])
    placement=json.loads((ROOT/(os.environ['EXP_NODE']+'-worker-map.json')).read_text())[worker]
    bind(placement['cpus'],placement['numa'])
    args=edge.parse_edge_arguments()
    args.max_tasks_per_draft=0
    Path(args.exp_name).mkdir(parents=True,exist_ok=True)
    torch.set_num_interop_threads(1)
    r=Runner(args)
    r.run_draft_process_http(r.tokenizer,int(os.environ['EXP_GLOBAL_WORKER']),Barrier(),Origin())
