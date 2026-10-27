"""Experiment adapter: real EdgeRunner protocol with explicit CPU/NUMA mapping and RPC logs."""
from pathlib import Path
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import torch
import edge.edge as edge_module


def bind_node(cpu, node):
    os.sched_setaffinity(0,{cpu})
    lib=ctypes.CDLL('libnuma.so.1',use_errno=True)
    lib.numa_allocate_nodemask.restype=ctypes.c_void_p
    lib.numa_bitmask_setbit.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    lib.numa_bitmask_setbit.restype=ctypes.c_void_p
    lib.numa_set_membind.argtypes=[ctypes.c_void_p]
    lib.numa_bitmask_free.argtypes=[ctypes.c_void_p]
    mask=lib.numa_allocate_nodemask()
    if not mask: raise RuntimeError('Could not allocate NUMA mask')
    try:
        lib.numa_bitmask_setbit(mask,node)
        lib.numa_set_membind(mask)
    finally: lib.numa_bitmask_free(mask)


class TracedClient(edge_module.EdgeClient):
    def synchronize_clock(self,samples=5):
        info=super().synchronize_clock(samples)
        Path(os.environ['COMM_RPC_LOG'].replace('rpc-','clock-').replace('.jsonl','.json')).write_text(json.dumps(info,indent=2)+'\n')
        return info
    def _post(self,path,payload):
        start=time.monotonic()
        record={'endpoint':path,'started_monotonic':start,'started_epoch':time.time(),'task_id':payload.get('task_id'),'pid':os.getpid()}
        try:
            result=super()._post(path,payload)
            record.update(ok=True,cloud_total_ms=result.get('cloud_total_ms'),verify_ms=result.get('verify_ms'))
            return result
        except Exception as exc:
            record.update(ok=False,error=repr(exc)); raise
        finally:
            record['finished_monotonic']=time.monotonic()
            record['http_ms']=(record['finished_monotonic']-start)*1000
            with open(os.environ['COMM_RPC_LOG'],'a') as out: out.write(json.dumps(record)+'\n')
    def _attach_pull(self,response,edge_receive_time_ns):
        if self.timing_priority:
            return super()._attach_pull(response,edge_receive_time_ns)
        # Vanilla protocol has no calibrated cross-host clock. Do not report one-way delay.
        response.setdefault('pull_s',0.0)
        return response

edge_module.EdgeClient=TracedClient


class BoundRunner(edge_module.EdgeRunner):
    def _encode_input_ids(self,tokenizer,input_text):
        value=tokenizer.apply_chat_template([{'role':'user','content':input_text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
        return value if torch.is_tensor(value) else torch.tensor([value],dtype=torch.long)
    def eval(self):
        self.model_barrier=mp.get_context('spawn').Barrier(self.args.num_drafts)
        return super().eval()
    def run_draft_process_http(self,tokenizer,proc_id,arrival_barrier=None,arrival_start_time=None):
        self.worker_id=proc_id
        mapping=json.loads((ROOT/'worker_map.json').read_text())
        placement=mapping[proc_id]
        bind_node(placement['cpu'],placement['numa'])
        os.environ['COMM_RPC_LOG']=str(Path(self.args.exp_name)/f'rpc-{proc_id:02d}.jsonl')
        return super().run_draft_process_http(tokenizer,proc_id,arrival_barrier,arrival_start_time)
    def _load_draft_model(self,model_path,device):
        started=time.monotonic()
        model=super()._load_draft_model(model_path,device)
        assert {p.dtype for p in model.parameters()}=={torch.float16}
        assert all(p.device.type=='cpu' for p in model.parameters())
        status={line.split(':')[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('Cpus_allowed_list:','Mems_allowed_list:','VmRSS:','Threads:'))}
        record={'worker_id':self.worker_id,'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),'load_seconds':time.monotonic()-started,'dtype':'float16','device':'cpu','proc_status':status,'numa_policies':sorted({line.split()[1] for line in Path('/proc/self/numa_maps').read_text().splitlines() if len(line.split())>1})}
        (Path(self.args.exp_name)/f'worker-{self.worker_id:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'WORKER_READY rank={self.worker_id} pid={os.getpid()} cpus={record["cpu_affinity"]}',flush=True)
        self.model_barrier.wait(timeout=600)
        record['barrier_release_monotonic']=time.monotonic()
        record['barrier_release_epoch']=time.time()
        (Path(self.args.exp_name)/f'worker-{self.worker_id:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
        return model

if __name__=='__main__':
    args=edge_module.parse_edge_arguments()
    output=Path(args.exp_name)
    output.mkdir(parents=True,exist_ok=True)
    args.max_tasks_per_draft=1
    BoundRunner(args).eval()
