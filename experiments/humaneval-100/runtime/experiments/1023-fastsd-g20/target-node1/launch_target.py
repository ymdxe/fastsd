from pathlib import Path
import json
import os
import shlex
import subprocess
root=Path('/home/zhangh/workspace/fastsd/experiments/1023-fastsd-g20')
venv='/home/zhangh/venvs/fastsd_cpu'
conda='/home/zhangh/miniconda3/envs/fastsd'
env=os.environ.copy()
env.update(PYTHONPATH=str(root/'source')+':'+venv+'/lib/python3.11/site-packages:'+conda+'/lib/python3.11/site-packages',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CLOUD_SERVICE_HOST='10.66.0.3',CLOUD_SERVICE_PORT='18601')
code='import torch,transformers,fastapi,uvicorn,cloud.cloud_service; print("IMPORT OK",torch.__version__,transformers.__version__,fastapi.__version__); print("THREADS",torch.get_num_threads())'
subprocess.run([venv+'/bin/python','-c',code],env=env,check=True,timeout=40)
subprocess.run(['numactl','--physcpubind=0,2,4,6,8,10,12,14','--membind=0','true'],check=True)
model='/mnt/nfs0/zhangh/models/Qwen3-1.7B'
args=['numactl','--physcpubind=0,2,4,6,8,10,12,14','--membind=0',venv+'/bin/python','-u',str(root/'source/cloud/cloud_service.py'),'--draft_model',model,'--target_model',model,'--target_device','cpu','--target_dtype','float16','--draft_dtype','float16','--dataset','gsm8k','--profile','custom','--server_sched_mode','fastsd','--enable_pipeline','--enable_proactive_draft','--enable_latency_priority','--no-pipeline_gamma_adapt','--pipeline_gamma_min','20','--pipeline_gamma_max','20','--verify_method','greedy','--temp','0','--top_k','0','--top_p','1','--gamma','20','--batch_size','8','--token_budget','256','--max_tokens','64','--num_drafts','32','--no-measure_energy','--exp_name',str(root/'target')]
(root/'target').mkdir(exist_ok=True)
configuration={'command':args,'environment':{key:env[key] for key in ['PYTHONPATH','PYTHONNOUSERSITE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','CLOUD_SERVICE_HOST','CLOUD_SERVICE_PORT']}}
(root/'target_launch.json').write_text(json.dumps(configuration,indent=2)+'\n')
with (root/'commands.txt').open('a') as f:f.write('TARGET '+shlex.join(args)+'\nEnvironment: '+json.dumps(configuration['environment'])+'\n')
print('=== START TARGET ===',shlex.join(args),flush=True)
with (root/'target.log').open('w') as log:
    child=subprocess.Popen(args,env=env,cwd=str(root/'source'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (root/'target.pid').write_text(str(child.pid)+'\n')
    for line in child.stdout:
        log.write(line);log.flush();print(line,end='',flush=True)
    status=child.wait()
(root/'target_exit.txt').write_text(str(status)+'\n')
print('=== TARGET EXIT ===',status,flush=True)
