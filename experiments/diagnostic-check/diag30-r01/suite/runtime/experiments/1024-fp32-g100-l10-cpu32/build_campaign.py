"""Create a new immutable FP32/multicore experiment from the current kvpre code."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from allocation import allocate

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OLD=ROOT.parent/'1024-poisson32-qwen06-17'


def patch(text,old,new,count=None):
    assert old in text,old
    if count is not None:assert text.count(old)==count,(old,text.count(old))
    return text.replace(old,new)


def main():
    assert not (ROOT/'source').exists(),'Do not overwrite a snapshot'
    for node in ('node1','node3'):
        data=json.loads((ROOT/'preflight'/(node+'.json')).read_text());mapping=allocate(data)
        (ROOT/(node+'-worker-map.json')).write_text(json.dumps(mapping,indent=2)+'\n')
    files=['common.py','coordinator.py','campaign.py','report.py','target.py','telemetry.py','node_agent.py','native_worker.py','finalize.py','start_control.py','bundle.py']
    for name in files:shutil.copy2(OLD/name,ROOT/name)
    for directory in ('src','edge','cloud'):
        shutil.copytree(REPO/directory,ROOT/'source'/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    engine=ROOT/'source/src/engine.py';text=engine.read_text()
    text='from telemetry import trace_cache_move, trace_verify_demand\n'+text
    text=patch(text,'        def move_dynamic_cache_to(cache, device):','        @trace_cache_move\n        def move_dynamic_cache_to(cache, device):',2)
    text=patch(text,'                    preloaded_gpu_pids.difference_update(proc_ids)','                    trace_verify_demand(proc_ids, preloaded_gpu_pids)\n                    preloaded_gpu_pids.difference_update(proc_ids)',1)
    engine.write_text(text)
    shutil.copytree(OLD/'specedge',ROOT/'specedge',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Only runtime precision, CPU placement and experiment controls change.
    native=ROOT/'native_worker.py';text=native.read_text().replace('torch.float16','torch.float32')
    text=patch(text,'os.sched_setaffinity(0,{cpu})','os.sched_setaffinity(0,set(cpu if isinstance(cpu,list) else [cpu]))')
    text=text.replace("placement['cpu']","placement['cpus']").replace("placement[\"cpu\"]","placement[\"cpus\"]")
    text=patch(text,"    r=Runner(args)","    torch.set_num_interop_threads(1)\n    r=Runner(args)")
    text=patch(text,"'threads':torch.get_num_threads(),'affinity'","'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'dtype':str(m.dtype),'torch':torch.__version__,'affinity'")
    native.write_text(text)
    agent=ROOT/'node_agent.py';text=agent.read_text();start=text.index('def placements():');end=text.index('\n\ndef environment(',start)
    text=text[:start]+'''def placements():
    data=json.loads((ROOT/(NODE+'-worker-map.json')).read_text())
    assert len(data)==16
    cpus=[cpu for row in data for cpu in row['cpus']]
    assert len(cpus)==len(set(cpus)) and set(cpus)<=os.sched_getaffinity(0)
    return data
'''+text[end:]
    text=patch(text,"    e=os.environ.copy();global_worker=worker+(16 if NODE=='node3' else 0)","    e=os.environ.copy();global_worker=worker+(16 if NODE=='node3' else 0)\n    threads=placements()[worker]['threads']")
    text=patch(text,"'OMP_NUM_THREADS':1,'MKL_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1","'OMP_NUM_THREADS':threads,'MKL_NUM_THREADS':threads,'OPENBLAS_NUM_THREADS':threads,'OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE'")
    text=patch(text,'def native_command(job,output):','def native_command(job,output,worker):')
    text=text.replace("'--draft_dtype','float16','--target_dtype','float16'","'--draft_dtype','float32','--target_dtype','float32'")
    text=patch(text,"'--edge_use_cpu','--edge_threads','1'","'--edge_use_cpu','--edge_threads',str(placements()[worker]['threads'])")
    text=text.replace("'SPECEDGE_DTYPE':'fp16'","'SPECEDGE_DTYPE':'fp32'").replace("'SPECEDGE_THREADS':1","'SPECEDGE_THREADS':placements()[worker]['threads']")
    text=patch(text,'else native_command(job,directory)','else native_command(job,directory,worker)')
    text=patch(text,"            log=(directory/'run.log').open('w')","            cmd=['taskset','--cpu-list',','.join(map(str,place[worker]['cpus'])),*cmd]\n            log=(directory/'run.log').open('w')")
    text=patch(text,"            write_json(out/'status.json',status);control('node_status',payload=status)","            write_json(out/'status.json',status);control('node_status',payload=status)\n            print(job['run'],NODE,'completed',status['completed'],'ready',status['ready'],'load',status['loadavg'],flush=True)")
    agent.write_text(text)
    for name in ('client_adapter.py','server_adapter.py'):
        file=ROOT/'specedge'/name;text=file.read_text().replace('torch.float16','torch.float32').replace('"float16"','"float32"').replace('"fp16"','"fp32"').replace('FP16','FP32')
        if name=='client_adapter.py':
            text=patch(text,'os.sched_setaffinity(0, {int(cpu)})','os.sched_setaffinity(0, set(cpu if isinstance(cpu,list) else [int(cpu)]))')
            text=patch(text,"placement['cpu']","placement['cpus']")
            text=patch(text,'torch.set_num_threads(1);torch.set_num_interop_threads(1)',"torch.set_num_threads(int(os.environ['SPECEDGE_THREADS']));torch.set_num_interop_threads(1)")
            text=patch(text,"'threads':torch.get_num_threads(),'affinity'","'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'dtype':str(model.dtype),'torch':torch.__version__,'affinity'")
        else:
            text=patch(text,'import torch\n','import torch\ntorch.backends.cuda.matmul.allow_tf32=False\ntorch.backends.cudnn.allow_tf32=False\ntorch.set_float32_matmul_precision("highest")\n',1)
        file.write_text(text)
    target=ROOT/'target.py';text=target.read_text()
    text=patch(text,'from src.engine import Decoding','import torch\ntorch.backends.cuda.matmul.allow_tf32=False\ntorch.backends.cudnn.allow_tf32=False\ntorch.set_float32_matmul_precision("highest")\nfrom src.engine import Decoding')
    text=patch(text,'    model=original(self,path,device)','    model=original(self,path,device)\n    assert model.dtype==torch.float32')
    text=patch(text,'    import torch\n','',1)
    target.write_text(text)
    gpu=json.loads((ROOT/'preflight/gpunode2.json').read_text())['selected_gpu']
    campaign=ROOT/'campaign.py';text=campaign.read_text()
    text=patch(text,'GPU=1',f'GPU={gpu["index"]}',1)
    text=patch(text,"GPU_UUID='GPU-422501a2-5189-7f5b-5ce6-95464276b27f'",'GPU_UUID='+repr(gpu['uuid']),1)
    text=text.replace("'--target_dtype','float16','--draft_dtype','float16'","'--target_dtype','float32','--draft_dtype','float32'")
    campaign.write_text(text)
    report=ROOT/'report.py';text=report.read_text().replace('Qwen3-1.7B FP16','Qwen3-1.7B FP32')
    text=text.replace('CPU draft：node1/node3 各16个','CPU FP32 draft：node1/node3 各16个')
    text=text.replace('gpunode2 GPU1',f'gpunode2 GPU{gpu["index"]}')
    report.write_text(text)
    final=ROOT/'finalize.py';text=final.read_text().replace("'GPU-422501a2-5189-7f5b-5ce6-95464276b27f'",repr(gpu['uuid']))
    text=text.replace('CPU32 Qwen3-0.6B / GPU Qwen3-1.7B','CPU32 FP32 Qwen3-0.6B / GPU FP32 Qwen3-1.7B')
    final.write_text(text)
    (ROOT/'workloads').mkdir();shutil.copy2(OLD/'workloads/lambda-10.jsonl',ROOT/'workloads/lambda-10.jsonl')
    matrix=[{'id':f'{method}-g100-l10','method':method,'gamma':100,'rate':10.,'cap':1024,'expected_requests':1319} for method in ('fastsd','standard','specedge')]
    (ROOT/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    manifest={'status':'prepared','source_branch':'kvpre','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'upstream_main':subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=REPO,text=True).strip(),'source_uncommitted_kvpre':True,
        'draft_model':'Qwen3-0.6B','target_model':'Qwen3-1.7B','dtype':'float32','draft_dtype':'float32','target_dtype':'float32',
        'workers_per_node':16,'planned_runs':3,'planned_requests':3957,'dataset':'GSM8K test full 1319','max_new_tokens':1024,
        'gamma':[100],'rates_global_rps':[10.],'arrival_seed':20261024,'batch_size':8,'token_budget':2048,'kv_batch_mode':'padded',
        'max_len_specedge':3072,'stop':'eos','greedy':True,'non_thinking':True,'physical_gpu':gpu['index'],'gpu_uuid':gpu['uuid'],
        'tf32':False,'cpu_allocation':{node:{'threads_per_model':json.loads((ROOT/(node+'-worker-map.json')).read_text())[0]['threads'],
        'physical_cores_used':sum(r['threads'] for r in json.loads((ROOT/(node+'-worker-map.json')).read_text()))} for node in ('node1','node3')},
        'arrival_policy':'Global Poisson schedule,32 static shards; scheduled queueing included in arrival TTFT/E2E and service metrics also reported.',
        'energy_policy':'100ms physical GPU UUID board power sampling, first scheduled arrival through last completion; loading excluded.',
        'transport':'Existing loopback SSH reverse forwards28634/28635; authenticated loopback controller18634 and target18635.',
        'notes':['Both draft and Target execute FP32. TF32 disabled.','Each model receives a disjoint multicore CPU set within one NUMA node;16 instances/node.',
        'Inference core unchanged except observation hooks; probability-history behavior preserved.','Only one formal complete run per method; no additional model smoke.']}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
