"""Only essential configuration/wire checks; no model load or numerical smoke."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from common import ROOT,write_json


def main():
    for p in list(ROOT.glob('*.py'))+list((ROOT/'specedge').glob('*.py'))+[ROOT/'source/src/engine.py']:ast.parse(p.read_text(),filename=str(p))
    manifest=json.loads((ROOT/'manifest.json').read_text());matrix=json.loads((ROOT/'matrix.json').read_text())
    assert manifest['draft_dtype']==manifest['target_dtype']=='float32' and manifest['tf32'] is False
    assert len(matrix)==6 and {(r['method'],r['gamma'],r['rate']) for r in matrix}=={(m,4,rate) for m in ('standard','fastsd','specedge') for rate in (1.,10.)}
    allocation={}
    for node in ('node1','node3'):
        resource=json.loads((ROOT/'preflight'/(node+'.json')).read_text());mapping=json.loads((ROOT/(node+'-worker-map.json')).read_text())
        assert len(mapping)==16 and resource['mem_available_gib']>=80
        topo={r['cpu']:r for r in resource['topology']};all_cpus=[];cores=[]
        for item in mapping:
            assert len(item['cpus'])==item['threads'] and item['threads']>1
            assert all(topo[c]['numa']==item['numa'] for c in item['cpus'])
            all_cpus+=item['cpus'];cores += [(topo[c]['socket'],topo[c]['core']) for c in item['cpus']]
        assert len(all_cpus)==len(set(all_cpus))==len(set(cores))
        assert len(all_cpus)<=resource['physical_cores']
        if resource['quota_cores'] is not None:assert len(all_cpus)<=resource['quota_cores']
        allocation[node]={'models':16,'threads_per_model':mapping[0]['threads'],'physical_cores_used':len(all_cpus),'physical_cores_available':resource['physical_cores']}
    old=ROOT.parent/'1024-poisson32-qwen06-17'
    schedules={}
    for rate in (1,10):
        path=ROOT/'workloads'/f'lambda-{rate}.jsonl'
        samples=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len(samples)==1319 and {r['source_index'] for r in samples}==set(range(1319))
        assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(samples,samples[1:]))
        assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((old/'workloads'/path.name).read_bytes()).digest()
        schedules[str(rate)]=samples[-1]['scheduled_arrival_s']
    agent=(ROOT/'node_agent.py').read_text()
    for setting in ("'SPECEDGE_MAX_N_BEAMS':1", "'SPECEDGE_MAX_BRANCH_WIDTH':1", "'SPECEDGE_PROACTIVE_MAX_N_BEAMS':1", "'SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH':1"):
        assert setting in agent
    assert json.loads((old/'validation.json').read_text())['prompt_parity_all']
    sys.path.insert(0,str(ROOT/'specedge'))
    from bootstrap import install
    install()
    import torch
    import util
    from specedge_grpc import specedge_pb2
    assert util.convert_dtype('fp32')==torch.float32
    width=5;max_len=3072
    mask=(torch.arange(max_len)[None,:]<=torch.arange(width)[:,None]).float()
    payload=mask.numpy().tobytes()
    restored=torch.frombuffer(bytearray(payload),dtype=torch.float32).reshape(width,max_len)
    assert torch.equal(mask,restored)
    request=specedge_pb2.ValidateRequest(client_idx=31,req_idx=1318,input_ids=bytes(width*8),position_ids=bytes(width*8),cache_seq_indices=bytes(width*8),
        parent_indices=bytes(4*8),attention_mask=payload,prefill=True,prefix='x'*2000)
    assert request.ByteSize()<4*1024**2
    from campaign import Sampler
    check_dir=ROOT/'checks';check_dir.mkdir(exist_ok=True)
    sample=Sampler(check_dir,None,None);sample.samples=[(float(t),100.) for t in range(11)];sample.thread.join=lambda **kw:None
    sample.finish(2.,8.,60,6)
    assert json.loads((check_dir/'energy.json').read_text())['energy_j']==600
    write_json(ROOT/'validation.json',{'ok':True,'epoch':time.time(),'model_inference_smokes':0,'matrix_cells':6,'dataset_requests':1319,'total_requests':7914,
        'draft_dtype':'float32','target_dtype':'float32','tf32':False,'cpu_allocation':allocation,'wire_mask_dtype':'float32',
        'wire_roundtrip_exact':True,'largest_grpc_body_bytes':request.ByteSize(),'prompt_parity_inherited_for_identical_tokenizers_and_workload':True,
        'last_scheduled_arrival_s_by_rate':schedules,'specedge_single_chain':True,'energy_integration_check':'600J for100W over6s'})


if __name__=='__main__':main()
