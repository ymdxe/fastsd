"""One CPU node agent: 16 independent workers, streamed complete request records."""
import collections
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
    p=ROOT/(NODE+'-worker-map.json')
    if p.exists():return json.loads(p.read_text())
    def read():
        return {int(a[0][3:]):list(map(int,a[1:9])) for s in Path('/proc/stat').read_text().splitlines() if (a:=s.split()) and a[0].startswith('cpu') and a[0][3:].isdigit()}
    before=read();time.sleep(1);after=read();busy={}
    for cpu in before:
        delta=[b-a for a,b in zip(before[cpu],after[cpu])]
        busy[cpu]=1-(delta[3]+delta[4])/max(1,sum(delta))
    rows=collections.defaultdict(list);seen=set()
    topology=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,NODE'],text=True)
    for line in topology.splitlines():
        if line.startswith('#'):continue
        cpu,core,socket,numa=map(int,line.split(','))
        if cpu not in os.sched_getaffinity(0) or (socket,core) in seen:continue
        seen.add((socket,core));rows[numa].append({'cpu':cpu,'numa':numa,'core':core,'socket':socket,'initial_busy':busy[cpu]})
    for row in rows.values():row.sort(key=lambda x:(x['initial_busy'],x['cpu']))
    selected=[]
    while len(selected)<16:
        for numa in sorted(rows):
            if rows[numa] and len(selected)<16:selected.append(rows[numa].pop(0))
        if not any(rows.values()) and len(selected)<16:raise RuntimeError('Fewer than16 physical cores')
    write_json(p,selected);return selected


def environment(job,worker,out):
    e=os.environ.copy();global_worker=worker+(16 if NODE=='node3' else 0)
    vals={'OMP_NUM_THREADS':1,'MKL_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':1,'TRANSFORMERS_OFFLINE':1,
          'TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':1,'EXP_RUN':job['run'],'EXP_NODE':NODE,'EXP_LOCAL_WORKER':worker,
          'EXP_WORKER':f'{NODE}-{worker}','EXP_GLOBAL_WORKER':global_worker,'EXP_GAMMA':job['gamma'],'EXP_RATE':f'{job["rate"]:g}',
          'EXP_OUTPUT':out,'CAMPAIGN_CONTROL':'http://127.0.0.1:28634'}
    e.update({k:str(v) for k,v in vals.items()})
    if NODE=='node1':e['PYTHONPATH']='/home/zhangh/venvs/fastsd_cpu/lib/python3.11/site-packages:/home/zhangh/miniconda3/envs/fastsd/lib/python3.11/site-packages'
    return e


def native_command(job,output):
    fast=job['method']=='fastsd';model='/mnt/nfs0/zhangh/models/Qwen3-0.6B';gamma=job['gamma']
    cmd=[sys.executable,'-u',str(ROOT/'native_worker.py'),'--draft_model',model,'--target_model',model,'--draft_dtype','float16','--target_dtype','float16',
         '--edge_use_cpu','--edge_threads','1','--num_drafts','32','--server_url','http://127.0.0.1:28635','--request_timeout','7200',
         '--dataset','gsm8k','--dataset_file',str(ROOT/'workloads'/f'lambda-{job["rate"]:g}.jsonl'),'--arrival_distribution','poisson','--arrival_rate',str(job['rate']),
         '--arrival_seed','20261024','--profile','custom' if fast else 'vanilla','--server_sched_mode','fastsd' if fast else 'vanilla',
         '--no-pipeline_gamma_adapt','--pipeline_gamma_min',str(gamma),'--pipeline_gamma_max',str(gamma),'--gamma',str(gamma),'--max_tokens',str(job['cap']),
         '--verify_method','greedy','--temp','0','--top_k','0','--top_p','1','--stop_policy','eos','--warmup_requests','0','--exp_name',str(output)]
    for feature in ('enable_pipeline','enable_proactive_draft','enable_latency_priority'):cmd.append('--'+('' if fast else 'no-')+feature)
    return cmd


def specedge_command(job,worker,out,e):
    gamma=job['gamma'];gid=worker+(16 if NODE=='node3' else 0)
    vals={'SPECEDGE_OPTIMIZATION':2,'SPECEDGE_RESULT_PATH':str(out.parent),'SPECEDGE_EXP_NAME':'run','SPECEDGE_PROCESS_NAME':f'client_{worker}',
          'SPECEDGE_SEED':1234+gid,'SPECEDGE_MAX_LEN':3072,'SPECEDGE_DRAFT_MODEL':'/mnt/nfs0/zhangh/models/Qwen3-0.6B','SPECEDGE_DEVICE':'cpu','SPECEDGE_DTYPE':'fp16',
          'SPECEDGE_THREADS':1,'SPECEDGE_DATASET':'fastsd_external','SPECEDGE_REASONING':'False','SPECEDGE_MAX_N_BEAMS':1,'SPECEDGE_MAX_BEAM_LEN':gamma,
          'SPECEDGE_MAX_BRANCH_WIDTH':1,'SPECEDGE_MAX_BUDGET':gamma,'SPECEDGE_PROACTIVE_TYPE':'included','SPECEDGE_PROACTIVE_MAX_N_BEAMS':1,
          'SPECEDGE_PROACTIVE_MAX_BEAM_LEN':gamma,'SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH':1,'SPECEDGE_PROACTIVE_MAX_BUDGET':gamma+1,
          'SPECEDGE_MAX_NEW_TOKENS':job['cap'],'SPECEDGE_MAX_REQUEST_NUM':42,'SPECEDGE_REQ_OFFSET':0,'SPECEDGE_SAMPLE_REQ_CNT':1319,
          'SPECEDGE_HOST':'127.0.0.1:28635','SPECEDGE_CLIENT_IDX':gid,'EXP_REQUEST_TIMEOUT':7200,
          'EXP_RPC_LOG':str(out/f'rpc-{worker:02d}.jsonl'),'EXP_CYCLE_LOG':str(out/f'cycle-{worker:02d}.jsonl')}
    e.update({k:str(v) for k,v in vals.items()})
    return [sys.executable,'-u',str(ROOT/'specedge/client_adapter.py'),'--node',NODE,'--worker',str(worker),'--run',job['run'],'--expected','32','--output',str(out)]


def run(job):
    os.environ.update(EXP_RUN=job['run'],EXP_NODE=NODE,EXP_GAMMA=str(job['gamma']))
    out=ROOT/job['run']/NODE
    if out.exists():raise FileExistsError('Refusing existing run '+str(out))
    out.mkdir(parents=True)
    place=placements();write_json(out/'placement.json',place)
    write_json(out/'preflight.json',{'host':NODE,'loadavg':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'python':sys.executable,'epoch':time.time()})
    procs=[];logs=[];offsets={};launches=[]
    try:
        for worker in range(16):
            directory=out/f'worker-{worker:02d}';directory.mkdir()
            e=environment(job,worker,directory)
            cmd=specedge_command(job,worker,directory,e) if job['method']=='specedge' else native_command(job,directory)
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
            write_json(out/'status.json',status);control('node_status',payload=status)
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
