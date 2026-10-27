"""Serial full-dataset campaign; stop on failure, skip only verified completed cells."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from common import ROOT,OPENER,control,write_json
from report import load_rows,main as report

GPU=1
GPU_UUID='GPU-422501a2-5189-7f5b-5ce6-95464276b27f'
os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'


def http(url,data=None):
    request=urllib.request.Request(url,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
    with OPENER.open(request,timeout=30) as response:return json.loads(response.read())


def native_args(cell,out):
    fast=cell['method']=='fastsd';model='/mnt/nfs0/zhangh/models/Qwen3-1.7B'
    cmd=[sys.executable,'-u',str(ROOT/'target.py'),'--draft_model',model,'--target_model',model,'--target_device','cuda:0','--target_dtype','float16','--draft_dtype','float16',
         '--kv_batch_mode','padded','--dataset','gsm8k','--profile','custom' if fast else 'vanilla','--server_sched_mode','fastsd' if fast else 'vanilla',
         '--no-pipeline_gamma_adapt','--gamma',str(cell['gamma']),'--max_tokens',str(cell['cap']),'--batch_size','8','--token_budget','2048','--num_drafts','32',
         '--verify_method','greedy','--temp','0','--top_k','0','--top_p','1','--no-measure_energy','--exp_name',str(out)]
    for feature in ('enable_pipeline','enable_proactive_draft','enable_latency_priority'):cmd.append('--'+('' if fast else 'no-')+feature)
    return cmd


def specedge_args(cell,out):
    return [sys.executable,'-u',str(ROOT/'specedge/server_adapter.py'),'--model','/mnt/nfs0/zhangh/models/Qwen3-1.7B',
            '--output',str(out),'--host','127.0.0.1','--port','18635','--management-port','18636','--batch-size','8','--max-len','3072',
            '--phase','formal','--sync-clients','32','--node1-gamma',str(cell['gamma']),'--node3-gamma',str(cell['gamma'])]


class Sampler:
    def __init__(self,out,handle,nvml):
        self.out=out;self.handle=handle;self.nvml=nvml;self.stop=threading.Event();self.samples=[];self.error=None
        self.thread=threading.Thread(target=self.loop,name='gpu-power',daemon=True)
    def loop(self):
        try:
            with (self.out/'power.jsonl').open('a',buffering=1) as f:
                while not self.stop.is_set():
                    n=self.nvml;row={'epoch':time.time(),'power_w':n.nvmlDeviceGetPowerUsage(self.handle)/1000,
                                    'memory_used_bytes':n.nvmlDeviceGetMemoryInfo(self.handle).used,'gpu_utilization_pct':n.nvmlDeviceGetUtilizationRates(self.handle).gpu}
                    try:row['total_energy_mj']=n.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                    except (n.NVMLError,AttributeError):pass
                    self.samples.append((row['epoch'],row['power_w']));f.write(json.dumps(row)+'\n')
                    self.stop.wait(.1)
        except BaseException as exc:self.error=repr(exc)
    def finish(self,start,end,tokens,requests):
        self.stop.set();self.thread.join(timeout=5)
        if self.error:raise RuntimeError('Power sampling failure '+self.error)
        energy=0.;covered=0.
        for (t0,p0),(t1,p1) in zip(self.samples,self.samples[1:]):
            a=max(t0,start);b=min(t1,end)
            if b<=a:continue
            # Linear interpolation and trapezoid clipping at exact boundaries.
            pa=p0+(p1-p0)*(a-t0)/(t1-t0);pb=p0+(p1-p0)*(b-t0)/(t1-t0)
            energy+=(pa+pb)*.5*(b-a);covered+=b-a
        duration=end-start
        if covered<duration-.5:raise RuntimeError('Incomplete GPU energy sampling coverage')
        write_json(self.out/'energy.json',{'physical_gpu':GPU,'gpu_uuid':GPU_UUID,'energy_j':energy,'energy_Wh':energy/3600,
                    'duration_s':duration,'sampled_duration_s':covered,'average_power_w':energy/covered if covered else None,
                    'joules_per_token':energy/tokens if tokens else None,'joules_per_request':energy/requests if requests else None,
                    'start_epoch':start,'end_epoch':end,'sample_interval_ms':100,'samples':len(self.samples),
                    'measurement':'physical GPU total board power, including its idle intervals in the campaign window; loading excluded'})


def gpu_processes(n,handle):
    return n.nvmlDeviceGetComputeRunningProcesses(handle)


def is_descendant(pid,parent):
    for _ in range(12):
        if pid==parent:return True
        try:
            status=Path(f'/proc/{pid}/status').read_text()
            pid=int(next(s for s in status.splitlines() if s.startswith('PPid:')).split()[1])
        except (OSError,StopIteration,ValueError):return False
        if pid<=1:return False
    return False


def run(cell,n,handle):
    directory=ROOT/cell['id'];out=directory/'target'
    if (directory/'status.json').exists():raise FileExistsError('Existing unfinished run requires explicit review: '+str(directory))
    out.mkdir(parents=True,exist_ok=True)
    # No other GPU process is allowed during board-energy comparisons.
    # The installed NVML v1 reports ~615MiB including driver reservation on
    # an otherwise idle A6000 (nvidia-smi v2 used=15MiB). Check processes too.
    while gpu_processes(n,handle) or n.nvmlDeviceGetMemoryInfo(handle).used>1024*1024**2:
        write_json(ROOT/'status.json',{'status':'waiting_for_gpu','cell':cell,'epoch':time.time()});time.sleep(30)
    cmd=specedge_args(cell,out) if cell['method']=='specedge' else native_args(cell,out)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(GPU),EXP_GPU_INDEX=str(GPU),EXP_TARGET_OUTPUT=str(out),CLOUD_SERVICE_HOST='127.0.0.1',CLOUD_SERVICE_PORT='18635',
                                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',
                                    OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    write_json(out/'launch.json',{'command':cmd,'physical_gpu':GPU,'gpu_uuid':GPU_UUID,'source_manifest':'../../manifest.json','epoch':time.time()})
    log=(out/'target.log').open('w');proc=None;sampler=None;success=False
    try:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(out/'process.json',{'pid':proc.pid,'epoch':time.time()})
        started=time.monotonic()
        while True:
            if proc.poll() is not None:raise RuntimeError('Target exited during loading')
            ready=out/'target_model_ready.json'
            if ready.exists() and json.loads(ready.read_text()).get('ready'):
                try:
                    health=http('http://127.0.0.1:'+('18636' if cell['method']=='specedge' else '18635')+'/health')
                    if health.get('ready',health.get('status') in ('ok','healthy')):break
                except Exception:pass
            if time.monotonic()-started>900:raise TimeoutError('Target loading')
            time.sleep(1)
        sampler=Sampler(directory,handle,n);sampler.thread.start()
        for node in ('node1','node3'):control('set_job',payload={'node':node,'job':dict(cell,run=cell['id'])})
        write_json(directory/'status.json',{'status':'running','cell':cell,'started_epoch':time.time()})
        last_report=0.;last_check=0.;last_completion=0;progress_at=time.monotonic()
        while True:
            if proc.poll() is not None:raise RuntimeError('Target exited')
            state=control('state',run=cell['id'])
            if state['abort']:raise RuntimeError('CPU worker failure: '+str(state['abort']))
            count=len(state['results'])
            if count!=last_completion:last_completion=count;progress_at=time.monotonic()
            if time.monotonic()-progress_at>4*3600:raise TimeoutError('No request completed for4hours')
            now=time.monotonic()
            if now-last_check>10:
                foreign=[p.pid for p in gpu_processes(n,handle) if not is_descendant(p.pid,proc.pid)]
                if foreign:raise RuntimeError('GPU collision with other processes '+str(foreign))
                if sampler.error:raise RuntimeError(sampler.error)
                last_check=now
            write_json(ROOT/'status.json',{'status':'running','cell':cell,'completed':count,'expected':1319,'workers_ready':len(state['ready']),
                      'clock_ready':len(state['clock']),'release_epoch':state['release_epoch'],'epoch':time.time()})
            if now-last_report>60:
                print(cell['id'],'completed',count,'ready',len(state['ready']),'done',list(state['done']),flush=True);report();last_report=now
            if len(state['done'])==2:
                if any(d.get('failed') for d in state['done'].values()):raise RuntimeError('CPU node failed')
                if set(map(int,state['results']))!=set(range(1319)):raise RuntimeError('Incomplete or duplicate dataset coverage')
                rows=load_rows(directory)
                samples=[json.loads(s) for s in (ROOT/'workloads'/f'lambda-{cell["rate"]:g}.jsonl').read_text().splitlines() if s.strip()]
                start=state['release_epoch']+samples[0]['scheduled_arrival_s']
                end=max(r['completion_epoch'] for r in rows)
                sampler.finish(start,end,sum(r['generated_tokens'] for r in rows),len(rows));sampler=None
                success=True;break
            time.sleep(2)
    finally:
        if sampler is not None:sampler.stop.set();sampler.thread.join(timeout=5)
        if proc is not None and proc.poll() is None:
            try:http('http://127.0.0.1:'+('18636/shutdown' if cell['method']=='specedge' else '18635/exit'),{})
            except Exception:pass
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Only processes created by this campaign are stopped.
                os.killpg(proc.pid,15)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,9);proc.wait()
        log.close()
        write_json(out/'exit.json',{'exit_status':proc.returncode if proc is not None else None,'epoch':time.time()})
        if success:
            if proc.returncode!=0:raise RuntimeError('Target nonzero exit after completed requests')
            write_json(directory/'status.json',{'status':'completed','cell':cell,'epoch':time.time()})
            report()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    import pynvml as n
    n.nvmlInit();handle=n.nvmlDeviceGetHandleByUUID(GPU_UUID.encode())
    current=None
    try:
        processed=set()
        while True:
            # Re-read the pending matrix at run boundaries, preserving the
            # active run and completed cells when arrival rates are updated.
            matrix=json.loads((ROOT/'matrix.json').read_text())
            cell=next((c for c in matrix if c['id'] not in processed),None)
            if cell is None:break
            current=cell
            p=ROOT/cell['id']/'status.json'
            if p.exists() and json.loads(p.read_text()).get('status')=='completed':
                if args.resume:
                    processed.add(cell['id']);continue
                raise FileExistsError('Use --resume to skip completed runs')
            run(cell,n,handle)
            processed.add(cell['id'])
        write_json(ROOT/'status.json',{'status':'completed','runs':len(matrix),'requests':sum(c['expected_requests'] for c in matrix),'epoch':time.time()})
        for node in ('node1','node3'):control('set_job',payload={'node':node,'job':{'run':current['id'],'action':'stop'}})
        report()
    except BaseException as exc:
        error={'status':'failed','cell':current,'error':repr(exc),'traceback':traceback.format_exc(),'epoch':time.time()}
        write_json(ROOT/'status.json',error)
        if current:
            write_json(ROOT/current['id']/'status.json',error)
            try:control('abort',payload={'run':current['id'],'error':repr(exc)})
            except Exception:pass
        report();raise
    finally:n.nvmlShutdown()


if __name__=='__main__':main()
