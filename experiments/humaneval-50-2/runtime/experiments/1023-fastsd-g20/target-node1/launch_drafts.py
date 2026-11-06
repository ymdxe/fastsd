from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import time
import threading
root=Path('/home/zhangh/workspace/fastsd/experiments/1023-fastsd-g20')
count=int(os.environ.get('COMM_COUNT','32'))
label='smoke1' if count==1 else 'run32'
output=root/label
assert not output.exists(),str(output)
output.mkdir()
model='/mnt/nfs0/zhangh/models/Qwen3-1.7B'
args=[sys.executable,'-u',str(root/'edge_comm.py'),'--draft_model',model,'--target_model',model,'--draft_dtype','float16','--target_dtype','float16','--edge_use_cpu','--edge_threads','1','--num_drafts',str(count),'--server_url','http://10.66.0.3:18601','--request_timeout','1200','--dataset','gsm8k','--dataset_file',str(root/('workload1.jsonl' if count==1 else 'workload32.jsonl')),'--profile','custom','--server_sched_mode','fastsd','--enable_pipeline','--enable_proactive_draft','--enable_latency_priority','--no-pipeline_gamma_adapt','--pipeline_gamma_min','20','--pipeline_gamma_max','20','--verify_method','greedy','--temp','0','--top_k','0','--top_p','1','--gamma','20','--max_tokens',str(64),'--stop_policy','eos','--exp_name',str(output),'--warmup_requests','0']
# The experiment adapter applies the same Qwen3 non-thinking template to all GSM8K questions.
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1')
(root/(label+'_launch.json')).write_text(json.dumps({'command':args,'dataset':'gsm8k','encoder_flag':'experiment adapter applies Target non-thinking chat formatting to canonical GSM8K questions','workers':count},indent=2)+'\n')
with (root/'commands.txt').open('a') as out:out.write(label+' '+shlex.join(args)+'\n')
print('=== START '+label+' ===',shlex.join(args),flush=True)
started=time.monotonic()
with (root/(label+'.log')).open('w') as log:
    process=subprocess.Popen(args,env=env,cwd=str(root/'source'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    (root/(label+'.pid')).write_text(str(process.pid)+'\n')
    def monitor():
        while process.poll() is None:
            ready=len(list(output.glob('worker-*.json')))
            metrics=list(output.glob('edge_metrics_proc*.jsonl'))
            completed=sum(len(p.read_text().splitlines()) for p in metrics)
            rpc_count=sum(len(p.read_text().splitlines()) for p in output.glob('rpc-*.jsonl'))
            elapsed=time.monotonic()-started
            print(f'[PROGRESS {label}] elapsed={elapsed:.0f}s model_ready={ready}/{count} requests_completed={completed}/{count} completed_RPCs={rpc_count}',flush=True)
            with (root/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'time':time.time(),'elapsed':elapsed,'ready':ready,'completed':completed,'rpc_count':rpc_count,'loadavg':os.getloadavg(),'mem_available_kb':next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))})+'\n')
            time.sleep(30)
    threading.Thread(target=monitor,daemon=True).start()
    for line in process.stdout:
        log.write(line);log.flush();print(line,end='',flush=True)
    status=process.wait()
(root/(label+'_status.json')).write_text(json.dumps({'exit_status':status,'duration_seconds':time.monotonic()-started,'workers':count},indent=2)+'\n')
print('=== '+label+' EXIT ===',status,flush=True)
