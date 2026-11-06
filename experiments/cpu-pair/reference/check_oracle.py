#!/usr/bin/env python3
"""User-run correctness gate. Loads one 8B model; not a throughput benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(reference, candidates):
    result = {}
    for label, records in candidates.items():
        missing = sorted(set(reference) - set(records))
        mismatches = []
        for key, expected in reference.items():
            if key not in records:
                continue
            actual = records[key]
            if actual != expected:
                pos = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
                           min(len(expected), len(actual)))
                mismatches.append({'source_index': int(key), 'first_difference_0based': pos,
                                   'reference_token': expected[pos] if pos < len(expected) else None,
                                   'candidate_token': actual[pos] if pos < len(actual) else None})
        result[label] = {'matched': len(reference) - len(missing) - len(mismatches),
                         'missing': missing, 'mismatches': mismatches}
    return result


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    inputs = [json.loads(s) for s in (source / 'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    historical = json.loads((source / 'historical_outputs.json').read_text(encoding='utf-8'))
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[args.dtype]
    metadata = {
        'purpose': 'correctness_diagnostic_not_performance_baseline',
        'python': sys.executable, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'host': platform.node(), 'model': args.model, 'tokenizer': args.tokenizer,
        'dtype': args.dtype, 'max_tokens': 32, 'warmup_requests': 1,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'inputs_sha256': digest(source / 'inputs.jsonl'),
        'historical_outputs_sha256': digest(source / 'historical_outputs.json'),
        'script_sha256': digest(__file__),
        'source_sha256': {name: digest(repo / name) for name in
                          ['src/kvcache_varlen.py', 'src/kvcache_batching.py', 'src/util.py']},
        'model_config_sha256': digest(Path(args.model) / 'config.json'),
    }
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (root / 'gpu-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True), encoding='utf-8')
    print('Loading Qwen3-8B for correctness checks...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                device_map={'': 'cuda:0'}, local_files_only=True).eval()
    metadata['attention_implementation'] = getattr(model.config, '_attn_implementation', None)
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def encode(prompt):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_tensors='pt').to('cuda:0')

    @torch.inference_mode()
    def hf_generate(ids, historical_tokens=None):
        past = None
        residual = ids
        output_ids, margins, candidate_scores = [], [], []
        for step in range(32):
            outputs = model(residual, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[0, -1].float()
            token = int(logits.argmax().item())
            top = torch.topk(logits, 2)
            margins.append({'position': step, 'top2_ids': top.indices.tolist(),
                            'top2_logits': top.values.tolist(),
                            'margin': float((top.values[0] - top.values[1]).item())})
            if historical_tokens:
                for label, sequence in historical_tokens.items():
                    if sequence[:step] == output_ids and step < len(sequence) and sequence[step] != token:
                        candidate_scores.append({'candidate': label, 'position': step,
                            'candidate_token': sequence[step], 'reference_token': token,
                            'logit_gap_on_shared_prefix': float((logits[token] - logits[sequence[step]]).item())})
            output_ids.append(token)
            if token == tokenizer.eos_token_id:
                break
            residual = torch.tensor([[token]], device='cuda:0', dtype=torch.long)
        return output_ids, margins, candidate_scores

    @torch.inference_mode()
    def varlen_generate_ids(ids):
        from src.kvcache_batching import KVCacheModel_batching
        from src.kvcache_varlen import varlen_generate
        cache = KVCacheModel_batching(model, temperature=0, top_k=0, top_p=1)
        cache.vocab_size = model.config.vocab_size
        residual = ids
        tokens = []
        for step in range(32):
            x = varlen_generate(cache, model, [residual], ['oracle'], tokenizer.pad_token_id,
                               is_prefill=(step == 0), sample_next=False)
            token = int(x[0][-1].item())
            tokens.append(token)
            if token == tokenizer.eos_token_id:
                break
            residual = torch.tensor([[token]], device='cuda:0', dtype=torch.long)
        return tokens

    hf_generate(encode(inputs[0]['prompt']))
    outputs = {'hf': {}, 'varlen_single': {}, 'hf_repeat': {}}
    with (root / 'requests.jsonl').open('x', encoding='utf-8') as f:
        for record in inputs:
            key = str(record['source_index'])
            ids = encode(record['prompt'])
            old = {label: rows[key] for label, rows in historical.items()}
            start = time.monotonic()
            reference, margins, scores = hf_generate(ids, old)
            varlen = varlen_generate_ids(ids)
            # Repeat the HF path only when any compared result differs.
            repeat = hf_generate(ids)[0] if varlen != reference or any(x != reference for x in old.values()) else reference
            outputs['hf'][key], outputs['varlen_single'][key], outputs['hf_repeat'][key] = reference, varlen, repeat
            row = {'source_index': int(key), 'input_token_ids': ids[0].tolist(),
                   'hf_token_ids': reference, 'varlen_single_token_ids': varlen,
                   'hf_repeat_token_ids': repeat, 'hf_top2_margins': margins,
                   'first_difference_candidate_scores': scores,
                   'hf_output_text': tokenizer.decode(reference, skip_special_tokens=True),
                   'diagnostic_wall_s': time.monotonic() - start}
            f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
            print(f'checked {int(key)+1}/{len(inputs)}: node1={old["node1"] == reference}, '
                  f'node3={old["node3"] == reference}, varlen={varlen == reference}', flush=True)
    comparisons = compare(outputs['hf'], {**historical, 'varlen_single': outputs['varlen_single'],
                                         'hf_repeat': outputs['hf_repeat']})
    summary = {'total_prompts': len(inputs), 'comparisons': comparisons,
               'all_compared_outputs_match': all(not r['missing'] and not r['mismatches'] for r in comparisons.values()),
               'formal_performance_ready': False,
               'note': 'This checks historical outputs plus single-sequence HF/varlen AR. '
                       'It does not validate full FastSD or a batched HTTP AR performance baseline. '
                       'Logit gaps can suggest numerical sensitivity but do not prove the historical cause.'}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('Results:', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='/home/hdd3/zhangh/workspace/fastsd')
    parser.add_argument('--model', default='/home/hdd/zhangh/models/Qwen3-8B')
    parser.add_argument('--tokenizer', default='/home/hdd/zhangh/models/Qwen3-0.6B')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
