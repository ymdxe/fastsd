"""Read-only model diagnostic: common prefixes and same-QKV kernel comparisons."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

SUITE=Path(sys.argv[1]).resolve()
OUT=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(SUITE),str(SUITE/'runtime')]
config=json.loads((SUITE/'config.json').read_text())
state={'state':'starting','started_epoch_s':time.time(),'formal_launch_allowed':False}
def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def status(**fields):
    state.update(fields,updated_epoch_s=time.time())
    write('status.json',state)
    print(json.dumps(fields),flush=True)

try:
    from campaign import wait_gpu_memory
    status(state='waiting_for_fixed_gpu')
    wait_gpu_memory({**config,'gpu_start_wait_timeout_s':300},OUT,'target_only')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import varlen_generate
    from src.attention_backend import PackedAttentionWorkspace
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(config['seed'])
    status(state='loading_model')
    model=AutoModelForCausalLM.from_pretrained(config['target_model'],torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'},local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(config['tokenizer_model'],local_files_only=True)
    cases=json.loads((SUITE.parent/'greedy-comparison.json').read_text())['resident_fa2']['differences']
    rows={r['source_id']:r for r in map(json.loads,(SUITE/'validation/humaneval-164.jsonl').read_text().splitlines())}
    baseline={r['source_id']:r for r in map(json.loads,(SUITE.parent/'greedy-legacy.jsonl').read_text().splitlines())}
    fa2={r['source_id']:r for r in map(json.loads,(SUITE.parent/'greedy-resident_fa2.jsonl').read_text().splitlines())}
    decoded=[{'source_id':c['source_id'],'first_token_index':c['first_token_index'],
        'legacy':tokenizer.decode(baseline[c['source_id']]['output_token_ids']),
        'fa2':tokenizer.decode(fa2[c['source_id']]['output_token_ids'])} for c in cases]
    write('decoded-differences.json',decoded)
    original=PackedAttentionWorkspace.attend
    capture={'enabled':False,'layers':[]}
    def inspect(workspace,q,caches,*,scale,timer=None):
        actual=original(workspace,q,caches,scale=scale,timer=timer)
        if not capture['enabled']:return actual
        assert len(caches)==1
        k,v=caches[0]
        qlen=q.shape[-2]
        hist=k.shape[-2]-qlen
        mask=(torch.arange(qlen,device='cuda')[:,None]+hist>=torch.arange(k.shape[-2],device='cuda')[None,:])
        sdpa=torch.nn.functional.scaled_dot_product_attention(q,k,v,
            attn_mask=mask if hist and qlen>1 else None,is_causal=hist==0,scale=scale,enable_gqa=True)
        heads=q.shape[1]//k.shape[1]
        k32=k.float().repeat_interleave(heads,1)
        v32=v.float().repeat_interleave(heads,1)
        reference=(((q.float()@k32.transpose(-2,-1))*scale).masked_fill(~mask,float('-inf')).softmax(-1)@v32).transpose(1,2).reshape(qlen,-1)
        s=sdpa.transpose(1,2).reshape(qlen,-1).float()
        a=actual.float()
        error=(a-reference).abs()
        capture['layers'].append({'layer':len(capture['layers']),
            'fa2_vs_sdpa_max_abs':(a-s).abs().max().item(),
            'fa2_vs_fp32_max_abs':error.max().item(),
            'sdpa_vs_fp32_max_abs':(s-reference).abs().max().item(),
            'fa2_fp32_outside_tolerance':(error>(.02+.02*reference.abs())).sum().item(),
            'elements':a.numel(),'query_tokens':qlen,'kv_tokens':k.shape[-2]})
        return actual
    PackedAttentionWorkspace.attend=inspect
    logits={}
    hook=model.lm_head.register_forward_hook(lambda module,inputs,output:logits.update(value=output[-1].detach().float().clone()))
    results=[]
    with torch.inference_mode():
        for case in [c for c in cases if c['source_id'] in ('HumanEval/1','HumanEval/39','HumanEval/147')]:
            source=case['source_id'];offset=case['first_token_index']
            common=baseline[source]['output_token_ids'][:offset]
            out={'case':case,'replay':'batch1, prompt chunks32, teacher-forced identical legacy prefix; differs from original batch scheduling','variants':{}}
            for backend in ('sdpa_per_sequence','flash_attn_varlen'):
                manager=KVCacheModel_batching(model,temperature=0,top_k=0,top_p=1,attention_backend=backend)
                manager.vocab_size=model.config.vocab_size
                manager.bounded_probabilities=True
                prompt=rows[source]['input_token_ids']
                segments=[prompt[i:i+32] for i in range(0,len(prompt),32)]+[[x] for x in common]
                capture['layers']=[]
                for i,part in enumerate(segments):
                    capture['enabled']=backend=='flash_attn_varlen' and i==len(segments)-1
                    varlen_generate(manager,model,[torch.tensor([part],device='cuda')],[source],config['pad_token_id'],is_prefill=True,sample_next=False)
                values,indices=logits['value'].topk(5)
                out['variants'][backend]={'top_tokens':[{'token':int(t),'text':tokenizer.decode([int(t)]),'logit':float(v)} for t,v in zip(indices,values)],
                    'top2_margin':float(values[0]-values[1]),'predicted':int(logits['value'].argmax()),'layers':list(capture['layers'])}
                manager.reset(source)
                del manager
            out['same_argmax']=out['variants']['sdpa_per_sequence']['predicted']==out['variants']['flash_attn_varlen']['predicted']
            results.append(out)
            write('kernel-replay.json',results)
            status(state='replay',source_id=source,same_argmax=out['same_argmax'])
    hook.remove()
    status(state='complete',cases=len(results),formal_launch_allowed=False,
        note='Diagnostic only; no gate relaxation or change to frozen runtime. Replays use batch1; original full164 mismatch remains authoritative.')
except BaseException as exc:
    status(state='failed',error=repr(exc),traceback=traceback.format_exc())
    raise
