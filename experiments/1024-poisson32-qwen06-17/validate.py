"""Key checks only: matrix, full coverage, prompt parity, largest wire shape and energy."""
import ast
import hashlib
import json
from pathlib import Path
import time
from common import ROOT,write_json


def main():
    for p in list(ROOT.glob('*.py'))+list((ROOT/'specedge').glob('*.py'))+[ROOT/'source/src/engine.py']:
        ast.parse(p.read_text(),filename=str(p))
    matrix=json.loads((ROOT/'matrix.json').read_text())
    rates=sorted({c['rate'] for c in matrix})
    gammas=sorted({c['gamma'] for c in matrix})
    assert len(matrix)==3*len(gammas)*len(rates)
    assert {(x['method'],x['gamma'],x['rate']) for x in matrix}=={(m,g,r) for m in ('standard','fastsd','specedge') for g in gammas for r in rates}
    from transformers import AutoTokenizer
    draft=AutoTokenizer.from_pretrained('/mnt/nfs0/zhangh/models/Qwen3-0.6B',local_files_only=True)
    target=AutoTokenizer.from_pretrained('/mnt/nfs0/zhangh/models/Qwen3-1.7B',local_files_only=True)
    lengths=[];arrivals={}
    for rate in rates:
        rows=[json.loads(s) for s in (ROOT/'workloads'/f'lambda-{rate:g}.jsonl').read_text().splitlines()]
        assert len(rows)==1319 and {r['source_index'] for r in rows}==set(range(1319))
        assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(rows,rows[1:]))
        assert sum(sum(s['source_index']%32==worker for s in rows) for worker in range(32))==1319
        arrivals[str(rate)]=rows[-1]['scheduled_arrival_s']
        if rate==rates[0]:
            for row in rows:
                messages=[{'role':'user','content':row['prompt']}]
                kw={'tokenize':True,'add_generation_prompt':True,'enable_thinking':False}
                a=draft.apply_chat_template(messages,**kw);b=target.apply_chat_template(messages,**kw)
                assert a==b
                lengths.append(len(a))
                assert len(a)+1024+2*500+4<3072
    # Official protobuf body size, generated at the largest chain width.
    import sys
    sys.path.insert(0,str(ROOT/'specedge'))
    from bootstrap import install
    install()
    from specedge_grpc import specedge_pb2
    gamma=500;width=gamma+1
    msg=specedge_pb2.ValidateRequest(client_idx=31,req_idx=1318,input_ids=bytes(width*8),position_ids=bytes(width*8),cache_seq_indices=bytes(width*8),
        parent_indices=bytes(gamma*8),attention_mask=bytes(width*3072*2),prefill=True,prefix='x'*2000)
    assert msg.ByteSize()<4*1024**2
    from campaign import Sampler
    sample=Sampler(ROOT/'checks',None,None)
    sample.samples=[(float(t),100.) for t in range(11)];sample.thread.start=lambda:None;sample.thread.join=lambda **kw:None
    (ROOT/'checks').mkdir(exist_ok=True)
    sample.finish(2.,8.,60,6)
    energy=json.loads((ROOT/'checks/energy.json').read_text())
    assert energy['energy_j']==600 and energy['joules_per_token']==10
    import pynvml as nvml
    nvml.nvmlInit();handle=nvml.nvmlDeviceGetHandleByIndex(1)
    uuid=nvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid,bytes):uuid=uuid.decode()
    assert uuid=='GPU-422501a2-5189-7f5b-5ce6-95464276b27f'
    power=nvml.nvmlDeviceGetPowerUsage(handle)/1000
    nvml.nvmlShutdown()
    write_json(ROOT/'validation.json',{'ok':True,'epoch':time.time(),'model_inference_smokes':0,'matrix_cells':len(matrix),'dataset_requests':1319,
        'prompt_parity_all':True,'prompt_token_range':[min(lengths),max(lengths)],'largest_grpc_body_bytes':msg.ByteSize(),
        'scheduled_last_arrival_s':arrivals,'minimum_serial_schedule_hours':sum(arrivals.values())*3*len(gammas)/3600,
        'energy_integration_check':'100W over clipped6s=600J','gpu_uuid':uuid,'observed_idle_power_w':power})


if __name__=='__main__':main()
