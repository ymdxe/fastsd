"""Read-only environment and file preflight. Does not load model weights."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

root=Path(__file__).resolve().parent
sys.path[:0]=[str(root),str(root/'runtime')]
from suite import NODES, TARGET, TOKENIZER, manifest
import torch
import transformers
from transformers import AutoTokenizer

node=platform.node().split('.')[0]
is_gpu=node.startswith('gpunode2')
model=TARGET if is_gpu else NODES[node]['model']
model_root=Path(model)
config=json.loads((model_root/'config.json').read_text())
index=model_root/'model.safetensors.index.json'
shards=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
missing=[name for name in shards if not (model_root/name).is_file()]
if missing:
    raise RuntimeError('Missing model shards: '+str(missing))
tokenizer=AutoTokenizer.from_pretrained(TOKENIZER if is_gpu else model,local_files_only=True)
vocab=json.dumps(tokenizer.get_vocab(),sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()
print(json.dumps({'host':node,'bundle_id':manifest(),'python':sys.executable,
    'torch':torch.__version__,'transformers':transformers.__version__,
    'allowed_logical_cpus':len(os.sched_getaffinity(0)),'model':model,
    'model_shards_present':len(shards),'vocab_size':config.get('vocab_size'),
    'tokenizer_vocab_sha256':hashlib.sha256(vocab).hexdigest(),
    'special_ids':{'eos':tokenizer.eos_token_id,'pad':tokenizer.pad_token_id,'bos':tokenizer.bos_token_id},
    'chat_template_sha256':hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
    'role':'Target' if is_gpu else ('coordinator/backup' if node=='node2' else '8 Draft clients'),
    'model_weights_loaded':False},ensure_ascii=False,indent=2))
if is_gpu:
    from cloud import cloud_service
    from fastapi import FastAPI
    from ar_backend import install_ar
    app=FastAPI()
    install_ar(app,{'model':TARGET,'tokenizer':TOKENIZER,'output_dir':'unused-preflight'})
    print('PASS cloud and AR imports/routes; no startup events or model loading')
else:
    result=subprocess.run([sys.executable,'-B',str(root/'runtime/edge/edge.py'),'--help'],capture_output=True,text=True)
    if result.returncode:
        print(result.stdout[-2000:],result.stderr[-4000:])
        raise SystemExit(result.returncode)
    print('PASS actual Edge imports/CLI; no model loading')