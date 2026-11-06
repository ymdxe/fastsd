"""Keep both MT-Bench turns on one Draft and preserve actual dialogue history."""
import hashlib
import json
from pathlib import Path
from experiment_hooks import rpc

def resolve_sample(sample,url,output):
    if sample['dataset']!='mt_bench' or sample['turn']==1:
        return sample
    records=[json.loads(line) for line in (Path(output)/'edge_metrics_proc0.jsonl').read_text(encoding='utf-8').splitlines()]
    previous=next(row for row in records if row['task_id']==sample['previous_sample_id'])
    used=rpc(url,'/experiment/resolve-input',{'sample_id':sample['sample_id'],
        'previous_output_token_ids':previous['output_token_ids']})
    directory=Path(output)/'effective-inputs'
    directory.mkdir(exist_ok=True)
    with (directory/(str(sample['global_index'])+'.json')).open('x',encoding='utf-8') as handle:
        json.dump(used,handle,ensure_ascii=False)
    return used

def install_inputs(app,config,output,ar_engine):
    if config['dataset']!='mt_bench': return
    from fastapi import HTTPException
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['tokenizer_model'],local_files_only=True)
    samples={row['sample_id']:row for row in config['trace_rows']}
    directory=Path(output)/'effective-inputs'
    directory.mkdir()
    @app.post('/experiment/resolve-input')
    def resolve(body:dict):
        try:
            sample=samples[body['sample_id']]
            tokens=body['previous_output_token_ids']
            if (sample['turn']!=2 or not isinstance(tokens,list) or not 0<len(tokens)<=config['max_tokens']
                    or any(type(x) is not int or not 0<=x<len(tokenizer) for x in tokens)
                    or any(x in config['eos_token_ids'] for x in tokens[:-1])
                    or (len(tokens)<config['max_tokens'] and tokens[-1] not in config['eos_token_ids'])):
                raise ValueError('Invalid MT-Bench prior output')
            prompt=tokenizer.apply_chat_template([
                {'role':'user','content':sample['first_user_text']},
                {'role':'assistant','content':tokenizer.decode(tokens,skip_special_tokens=True)},
                {'role':'user','content':sample['user_text']}],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)+config['max_tokens']>config['max_context_length']:
                raise ValueError('Complete follow-up exceeds frozen context budget')
            used={**sample,'prompt':prompt,'input_token_ids':ids,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'input_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                'previous_output_token_ids':tokens,'conversation_history':'actual_same_method_first_reply'}
            path=directory/(str(sample['global_index'])+'.json')
            with path.open('x',encoding='utf-8') as handle:
                json.dump(used,handle,ensure_ascii=False)
            if ar_engine is not None:
                with ar_engine.lock:
                    ar_engine._expected[sample['sample_id']]=used
            return used
        except (KeyError,ValueError,FileExistsError,TypeError) as exc:
            raise HTTPException(409,str(exc)) from exc
