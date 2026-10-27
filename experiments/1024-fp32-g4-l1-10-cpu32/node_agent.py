"""One CPU node agent: 16 independent workers, streamed complete request records."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import ROOT,annotate,control,write_json

NODE=sys.argv[1]
assert NODE in ('node1','node3')


def placements():
    data=json.loads((ROOT/(NODE+'-worker-map.json')).read_text())
    assert len(data)==16
    cpus=[cpu for row in data for cpu in row['cpus']]
    assert len(cpus)==len(set(cpus)) and set(cpus)<=os.sched_getaffinity(0)
    return data


def environment(job,worker,out):
    e=os.environ.copy();global_worker=worker+(16 if NODE=='node3' else 0)
    threads=placements()[worker]['threads']
    vals={'OMP_NUM_THREADS':threads,'MKL_NUM_THREADS':threads,'OPENBLAS_NUM_THREADS':threads,'OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE','CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':1,'TRANSFORMERS_OFFLINE':1,
          'TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':1,'EXP_RUN':job['run'],'EXP_NODE':NODE,'EXP_LOCAL_WORKER':worker,
          'EXP_WORKER':f'{NODE}-{worker}','EXP_GLOBAL_WORKER':global_worker,'EXP_GAMMA':job['gamma'],'EXP_RATE':f'{job["rate"]:g}',
          'EXP_OUTPUT':out,'CAMPAIGN_CONTROL':'http://127.0.0.1:28634'}
    e.update({k:str(v) for k,v in vals.items()})
    if NODE=='node1':e['PYTHONPATH']='/home/zhangh/venvs/fastsd_cpu/lib/python3.11/site-packages:/home/zhangh/miniconda3/envs/fastsd/lib/python3.11/site-packages'
    return e


def native_command(job,output,worker):
    fast=job['method']=='fastsd';model='/mnt/nfs0/zhangh/models/Qwen3-0.6B';gamma=job['gamma']
    cmd=[sys.executable,'-u',str(ROOT/'native_worker.py'),'--draft_model',model,'--target_model',model,'--draft_dtype','float32','--target_dtype','float32',
         '--edge_use_cpu','--edge_threads',str(placements()[worker]['threads']),'--num_drafts','32','--server_url','http://127.0.0.1:28635','--request_timeout','7200',
         '--dataset','gsm8k','--dataset_file',str(ROOT/'workloads'/f'lambda-{job["rate"]:g}.jsonl'),'--arrival_distribution','poisson','--arrival_rate',str(job['rate']),
         '--arrival_seed','20261024','--profile','custom' if fast else 'vanilla','--server_sched_mode','fastsd' if fast else 'vanilla',
         '--no-pipeline_gamma_adapt','--pipeline_gamma_min',str(gamma),'--pipeline_gamma_max',str(gamma),'--gamma',str(gamma),'--max_tokens',str(job['cap']),
         '--verify_method','greedy','--temp','0','--top_k','0','--top_p','1','--stop_policy','eos','--warmup_requests','0','--exp_name',str(output)]
    for feature in ('enable_pipeline','enable_proactive_draft','enable_latency_priority'):cmd.append('--'+('' if fast else 'no-')+feature)
    return cmd


def specedge_command(job,worker,out,e):
    gamma=job['gamma'];gid=worker+(16 if NODE=='node3' else 0)
    vals={'SPECEDGE_OPTIMIZATION':2,'SPECEDGE_RESULT_PATH':str(out.parent),'SPECEDGE_EXP_NAME':'run','SPECEDGE_PROCESS_NAME':f'client_{worker}',
          'SPECEDGE_SEED':1234+gid,'SPECEDGE_MAX_LEN':3072,'SPECEDGE_DRAFT_MODEL':'/mnt/nfs0/zhangh/models/Qwen3-0.6B','SPECEDGE_DEVICE':'cpu','SPECEDGE_DTYPE':'fp32',
          'SPECEDGE_THREADS':placements()[worker]['threads'],'SPECEDGE_DATASET':'fastsd_external','SPECEDGE_REASONING':'False','SPECEDGE_MAX_N_BEAMS':1,'SPECEDGE_MAX_BEAM_LEN':gamma,
          'SPECEDGE_MAX_BRANCH_WIDTH':1,'SPECEDGE_MAX_BUDGET':gamma,'SPECEDGE_PROACTIVE_TYPE':'included','SPECEDGE_PROACTIVE_MAX_N_BEAMS':1,
          'SPECEDGE_PROACTIVE_MAX_BEAM_LEN':gamma,'SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH':1,'SPECEDGE_PROACTIVE_MAX_BUDGET':gamma+1,
          'SPECEDGE_MAX_NEW_TOKENS':job['cap'],'SPECEDGE_MAX_REQUEST_NUM':42,'SPECEDGE_REQ_OFFSET':0,'SPECEDGE_SAMPLE_REQ_CNT':1319,
          'SPECEDGE_HOST':'127.0.0.1:28635','SPECEDGE_CLIENT_IDX':gid,'EXP_REQUEST_TIMEOUT':7200,
          'EXP_RPC_LOG':str(out/f'rpc-{worker:02d}.jsonl'),'EXP_CYCLE_LOG':str(out/f'cycle-{worker:02d}.jsonl')}
    e.update({k:str(v) for k,v in vals.items()})
    return [sys.executable,'-u',str(ROOT/'specedge/client_adapter.py'),'--node',NODE,'--worker',str(worker),'--run',job['run'],'--expected','32','--output',str(out)]


def run(job):
    os.environ.update(EXP_RUN=job['run'],EXP_NODE=NODE,EXP_GAMMA=str(job['gamma']))
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024**2
    if available<80:raise RuntimeError(f'Insufficient memory for16 FP32 models: {available:.1f}GiB available,80GiB required')
    out=ROOT/job['run']/NODE
    if out.exists():raise FileExistsError('Refusing existing run '+str(out))
    out.mkdir(parents=True)
    place=placements();write_json(out/'placement.json',place)
    write_json(out/'preflight.json',{'host':NODE,'loadavg':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'python':sys.executable,'epoch':time.time()})
    procs=[];logs=[];offsets={};launches=[];metadata_sent=False
    try:
        for worker in range(16):
            directory=out/f'worker-{worker:02d}';directory.mkdir()
            e=environment(job,worker,directory)
            cmd=specedge_command(job,worker,directory,e) if job['method']=='specedge' else native_command(job,directory,worker)
            cmd=['taskset','--cpu-list',','.join(map(str,place[worker]['cpus'])),*cmd]
            log=(directory/'run.log').open('w');logs.append(log)
            proc=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append(proc)
            launches.append({'pid':proc.pid,'worker':worker,'command':cmd})
        write_json(out/'launch.json',{'job':job,'workers':launches})
        while True:
            # Read complete lines only. A partially written record is retried.
            rows=[]
            for f in sorted(out.glob('worker-*/edge_metrics_proc*.jsonl')):
                with f.open('rb') as stream:
                    stream.seek(offsets.get(str(f),0))
                    while True:
                        pos=stream.tell();line=stream.readline()
                        if not line or not line.endswith(b'\n'):break
                        row=json.loads(line)
                        if 'arrival_ttft_ms' not in row:
                            os.environ.update(EXP_NODE=NODE,EXP_GAMMA=str(job['gamma']))
                            row=annotate(row,run=job['run'],node=NODE,gamma=job['gamma'])
                        rows.append(row);offsets[str(f)]=stream.tell()
            if rows:control('results',payload={'run':job['run'],'node':NODE,'rows':rows})
            codes=[p.poll() for p in procs]
            failed=any(c is not None and c!=0 for c in codes)
            finished=all(c is not None for c in codes)
            status={'run':job['run'],'node':NODE,'epoch':time.time(),'exit_codes':codes,'finished':finished,'failed':failed,
                    'completed':sum(len(p.read_text().splitlines()) for p in out.glob('worker-*/edge_metrics_proc*.jsonl')),
                    'ready':len(list(out.glob('worker-*/ready.json')))+len(list(out.glob('worker-*/worker-*.json'))),'loadavg':os.getloadavg()}
            timing=[json.loads(p.read_text()) for p in out.glob('worker-*/timing-runtime-status.json')]
            status['clock_timing']={key:sum(item.get(key,0) for item in timing) for key in ('recalibrations','valid_pull_measurements','unreliable_pull_measurements','upload_pre_admission_retries')}
            write_json(out/'status.json',status);control('node_status',payload=status)
            print(job['run'],NODE,'completed',status['completed'],'ready',status['ready'],'load',status['loadavg'],flush=True)
            model_paths=list(out.glob('worker-*/ready.json'))+list(out.glob('worker-*/worker-*.json'))
            clock_paths=list(out.glob('worker-*/clock*.json'))
            if not metadata_sent and len(model_paths)==len(clock_paths)==16:
                workers={p.parent.name:json.loads(p.read_text()) for p in model_paths}
                for key,value in workers.items():
                    index=int(key.split('-')[1]);expected=place[index]
                    if value['dtype']!='torch.float32' or value['threads']!=expected['threads'] or value['affinity']!=expected['cpus']:raise RuntimeError('Actual FP32/thread/affinity does not match placement')
                files={name:json.loads((out/name).read_text()) for name in ('launch.json','preflight.json','placement.json')}
                files.update({'workers.json':workers,'clocks.json':{p.parent.name:json.loads(p.read_text()) for p in clock_paths},
                    'code-verification.json':{'hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('source/src/engine.py','source/src/scheduler_runtime.py','source/cloud/cloud_service.py','node_agent.py','common.py','native_worker.py','clock_runtime.py')}}})
                control('metadata',payload={'run':job['run'],'node':NODE,'files':files});metadata_sent=True
            if failed:raise RuntimeError('Worker failure '+str(codes))
            if finished:return
            if control('state',run=job['run']).get('abort'):raise RuntimeError('Controller aborted run')
            time.sleep(10)
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def main():
    placements();seen=set()
    write_json(ROOT/(NODE+'-agent.json'),{'pid':os.getpid(),'python':sys.executable,'epoch':time.time()})
    while True:
        try:
            job=control('job',node=NODE)
            if job.get('action')=='stop':return
            if job.get('run') and job['run'] not in seen:
                seen.add(job['run'])
                try:run(job)
                except BaseException as exc:
                    error={'run':job['run'],'node':NODE,'finished':True,'failed':True,'error':repr(exc),'traceback':traceback.format_exc(),'epoch':time.time()}
                    write_json(ROOT/job['run']/NODE/'error.json',error)
                    control('node_status',payload=error);control('abort',payload={'run':job['run'],'error':repr(exc)})
            time.sleep(3)
        except Exception as exc:
            print('CONTROL_RETRY',repr(exc),flush=True);time.sleep(5)


if __name__=='__main__':main()
