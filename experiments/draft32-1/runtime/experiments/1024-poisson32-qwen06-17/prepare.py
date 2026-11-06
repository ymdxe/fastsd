"""Freeze current kvpre code, full GSM8K and the reviewed SpecEdge implementation."""
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import subprocess
from common import ROOT,write_json

REPO=ROOT.parents[1]
OLD=REPO/'experiments/1024-hetero-16draft'


def replace(text,old,new,count=None):
    if count is not None and text.count(old)!=count:raise ValueError(('patch anchor',old,text.count(old),count))
    if old not in text:raise ValueError(('missing anchor',old))
    return text.replace(old,new)


def main():
    if (ROOT/'source').exists():raise FileExistsError('Snapshot exists; do not overwrite')
    for directory in ('src','edge','cloud'):
        shutil.copytree(REPO/directory,ROOT/'source'/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    engine=ROOT/'source/src/engine.py'
    text=engine.read_text()
    text='from telemetry import trace_cache_move, trace_verify_demand\n'+text
    text=replace(text,'        def move_dynamic_cache_to(cache, device):','        @trace_cache_move\n        def move_dynamic_cache_to(cache, device):',2)
    text=replace(text,'                    preloaded_gpu_pids.difference_update(proc_ids)','                    trace_verify_demand(proc_ids, preloaded_gpu_pids)\n                    preloaded_gpu_pids.difference_update(proc_ids)',1)
    engine.write_text(text)
    sp=ROOT/'specedge';sp.mkdir()
    shutil.copytree(OLD/'specedge/vendor',sp/'vendor',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    for file in ('bootstrap.py','vendor_manifest.json'):
        shutil.copy2(OLD/'specedge'/file,sp/file)
    # Keep routine per-token DEBUG output out of large dataset logs.
    bootstrap=(sp/'bootstrap.py').read_text().replace('logger.setLevel(logging.DEBUG if logger is logging.getLogger() else logging.INFO)','logger.setLevel(logging.INFO)')
    (sp/'bootstrap.py').write_text(bootstrap)
    text=(OLD/'specedge/client_adapter.py').read_text()
    text=text.replace('EXPERIMENT = ROOT.parent','EXPERIMENT = ROOT.parent\nsys.path.insert(0,str(EXPERIMENT))',1)
    text=text.replace('".coordinator-token"','".control-token"').replace('http://127.0.0.1:28624','http://127.0.0.1:28634')
    text=text.replace('choices=("specedge", "smoke-specedge")','type=str')
    start=text.index('async def run(args, rendezvous):');end=text.index('\n\ndef main():',start)
    text=text[:start]+(ROOT/'specedge_run_fragment.py.txt').read_text()+text[end:]
    text=text.replace('rendezvous.call("done", success=int(success))','None  # Node agent reports completion and full streamed results.')
    # The gRPC body at gamma500 stays below its 4 MiB default (FP16 masks, max_len3072).
    (sp/'client_adapter.py').write_text(text)
    text=(OLD/'specedge/server_adapter.py').read_text()
    text=text.replace('from bootstrap import install','import sys\nsys.path.insert(0,str(Path(__file__).resolve().parent.parent))\nfrom bootstrap import install',1)
    text=text.replace('!= "3"','!= os.environ["EXP_GPU_INDEX"]').replace('physical_gpu": 3','physical_gpu": int(os.environ["EXP_GPU_INDEX"])')
    text=text.replace('device_map="auto"','device_map={"": "cuda:0"}').replace('            max_memory={0: "34GiB", "cpu": "96GiB"},\n','')
    text=text.replace('max_n_beams=101','max_n_beams=max(self.gammas.values())+1')
    text=text.replace('request.client_idx < 8','request.client_idx < 16').replace('self.max_len - 101','self.max_len - max(self.gammas.values()) - 1')
    text=text.replace('1 <= args.node1_gamma <= 100 and 1 <= args.node3_gamma <= 100','1 <= args.node1_gamma <= 500 and 1 <= args.node3_gamma <= 500')
    text=text.replace('"SPECEDGE_MAX_BUDGET": "100", "SPECEDGE_NUM_CLIENTS": "16"','"SPECEDGE_MAX_BUDGET": str(max(args.node1_gamma,args.node3_gamma)), "SPECEDGE_NUM_CLIENTS": "32"')
    # No method-specific fixed GPU3 or weight offload assumptions survive.
    text=text.replace('"max_memory": {"0": "34GiB", "cpu": "96GiB"}','"weight_offload": False')
    (sp/'server_adapter.py').write_text(text)
    (ROOT/'workloads').mkdir()
    samples=[json.loads(s) for s in (REPO/'data/gsm8k.jsonl').read_text().splitlines() if s.strip()]
    assert len(samples)==1319
    rng=random.Random(20261024);intervals=[rng.expovariate(1) for _ in samples]
    for rate in (50.,):
        elapsed=0.;rows=[]
        for i,(sample,interval) in enumerate(zip(samples,intervals)):
            elapsed+=interval/rate
            rows.append(dict(sample,prompt=sample['question'],sample_id=f'gsm8k-{i:04d}',source_index=i,scheduled_arrival_s=elapsed))
        (ROOT/'workloads'/f'lambda-{rate:g}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    matrix=[]
    for i,(gamma,rate) in enumerate((g,r) for g in (200,500) for r in (50.,)):
        methods=('fastsd','standard','specedge') if i%3==0 else (('standard','specedge','fastsd') if i%3==1 else ('specedge','fastsd','standard'))
        for method in methods:
            matrix.append({'id':f'{method}-g{gamma}-l{rate:g}','method':method,'gamma':gamma,'rate':rate,'cap':1024,'expected_requests':1319})
    write_json(ROOT/'matrix.json',matrix)
    token=ROOT/'.control-token'
    fd=os.open(token,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(secrets.token_hex(32))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('source','specedge','workloads') for p in sorted((ROOT/directory).rglob('*')) if p.is_file()}
    write_json(ROOT/'manifest.json',{'status':'prepared','source_branch':'kvpre','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'upstream_main':subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=REPO,text=True).strip(),'source_uncommitted_kvpre':True,
        'dataset':'GSM8K test full 1319','draft_model':'Qwen3-0.6B','target_model':'Qwen3-1.7B','dtype':'float16','workers_per_node':16,
        'physical_gpu':1,'batch_size':8,'token_budget':2048,'max_new_tokens':1024,'max_len_specedge':3072,'arrival_seed':20261024,
        'rates_global_rps':[50.],'gamma':[200,500],'stop':'eos','greedy':True,'non_thinking':True,'hashes':hashes,
        'arrival_policy':'One shared global schedule; deterministic round-robin assignment to 32 persistent drafts; scheduled-to-start queueing included in arrival TTFT/E2E; service TTFT also reported.',
        'energy_policy':'External physical GPU UUID power sampling, 100ms; first scheduled arrival to last request completion; built-in NVML index0 service disabled.',
        'kv_policy':'Existing target cache offload enabled; actual move traces and kvpre counters; no artificial additional cache pressure.',
        'transport':'Authenticated loopback control and loopback model endpoint over existing user SSH reverse forwards; no IB performance claim.',
        'source_observation_patch':'Only a timing decorator around existing cache moves in the frozen snapshot; no added CUDA synchronization.'})


if __name__=='__main__':main()
