"""CPU draft-model-only baseline; its outputs are not equivalent to the 8B target."""
import json
import math
import os
from pathlib import Path
import platform
import time

from experiment_hooks import finish_worker, wait_for_start

MODELS = {
    'node1': '/mnt/nfs0/zhangh/models/Qwen3-0.6B',
    'node3': '/mnt/nfs0/zhangh/models/Qwen3-1.7B',
}


def _rss_snapshot(output, phase, completed_requests=0):
    """Record current/peak RSS without requiring an optional monitoring package."""
    snapshot = {'epoch_s': time.time(), 'phase': phase,
        'completed_requests': completed_requests, 'pid': os.getpid(),
        'rss_kib': None, 'rss_peak_kib': None}
    status_path = Path('/proc/self/status')
    if status_path.exists():
        for line in status_path.read_text(encoding='utf-8').splitlines():
            if line.startswith('VmRSS:'):
                snapshot['rss_kib'] = int(line.split()[1])
            elif line.startswith('VmHWM:'):
                snapshot['rss_peak_kib'] = int(line.split()[1])
    with (output / 'worker_resources.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(snapshot) + '\n')
    return snapshot


def _eos_ids(model, tokenizer):
    # Match the SD client's stopping rule; generation_config may add other IDs.
    del model
    value = getattr(tokenizer, 'eos_token_id', None)
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {int(token_id) for token_id in values}


def _generate(model, tokenizer, torch_module, prompt, max_tokens, eos_ids):
    """Greedy cached decoding with token visibility measured after CPU item()."""
    input_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt.strip()}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors='pt')
    if not hasattr(input_ids, 'ndim'):
        input_ids = torch_module.tensor(input_ids, dtype=torch_module.long, device='cpu')
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to('cpu')
    tokens, first = [], None
    past = None
    outputs = None
    try:
        with torch_module.inference_mode():
            for index in range(max_tokens):
                outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
                token = int(outputs.logits[0, -1, :].argmax().item())
                visible = time.monotonic()
                if first is None:
                    first = visible
                tokens.append(token)
                past = outputs.past_key_values
                # Drop the full logits before the next forward; only KV survives.
                outputs = None
                if token in eos_ids or index + 1 == max_tokens:
                    break
                input_ids = torch_module.tensor([[token]], dtype=torch_module.long, device='cpu')
            completed = time.monotonic()
        return tokens, first, completed
    finally:
        # A new request always starts with a complete prompt and an empty cache.
        outputs = None
        past = None
        input_ids = None


def run(args):
    """Run one CPU worker using args.inputs, args.exp_name and args.server_url."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.runtime import configure_torch_threads

    identity = os.environ['FASTSD_SUITE_WORKER']
    node = identity.split('-w', 1)[0]
    if node not in MODELS:
        raise ValueError('Draft-only baseline supports node1/node3 workers: ' + identity)
    output = Path(args.exp_name)
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / 'edge_metrics_proc0.jsonl'
    if metric_path.exists():
        raise FileExistsError('Refusing to overwrite existing draft-only metrics: ' + str(metric_path))
    samples = [json.loads(line) for line in Path(args.inputs).read_text(encoding='utf-8').splitlines()
        if line.strip()]
    previous = -1.0
    seen = set()
    for sample in samples:
        scheduled = float(sample['scheduled_arrival_s'])
        if not math.isfinite(scheduled) or scheduled < 0 or scheduled < previous:
            raise ValueError('Worker trace must have finite, nonnegative ordered arrival times')
        if not sample['sample_id'].startswith(identity + '-') or sample['sample_id'] in seen:
            raise ValueError('Wrong-worker or duplicate sample ID: ' + sample['sample_id'])
        previous = scheduled
        seen.add(sample['sample_id'])
    if samples:
        warmup_prompt = samples[0]['prompt']
    else:
        fixture = Path(os.environ['FASTSD_SUITE_ROOT']) / 'reference' / 'inputs.jsonl'
        with fixture.open(encoding='utf-8') as handle:
            warmup_prompt = json.loads(next(line for line in handle if line.strip()))['prompt']

    configure_torch_threads(8)
    torch.manual_seed(1234)
    _rss_snapshot(output, 'before_model_load')
    tokenizer = AutoTokenizer.from_pretrained(MODELS[node], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODELS[node], local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32).to('cpu').eval()
    eos_ids = _eos_ids(model, tokenizer)
    metadata = {'method': 'draft_only', 'worker': identity, 'hostname': platform.node(),
        'model': MODELS[node], 'device': 'cpu', 'dtype': 'float32', 'threads': 8,
        'attn_implementation': getattr(model.config, '_attn_implementation', None),
        'decode': 'greedy', 'temperature': 0, 'max_tokens': 32, 'enable_thinking': False,
        'eos_inclusive': True, 'eos_token_ids': sorted(eos_ids),
        'warmup_requests': 1, 'warmup_max_tokens': 2, 'warmup_included_in_metrics': False,
        'output_equivalent_to_8b': False,
        'comparison_note': 'Draft-model generation uses different weights from the 8B target; speed is not an equivalent-output 8B acceleration result.'}
    (output / 'draft_only_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    _rss_snapshot(output, 'model_loaded')
    _generate(model, tokenizer, torch, warmup_prompt, 2, eos_ids)
    _rss_snapshot(output, 'warmup_complete')
    origin = float(wait_for_start(args))
    with metric_path.open('x', encoding='utf-8') as handle:
        for index, sample in enumerate(samples):
            scheduled = float(sample['scheduled_arrival_s'])
            remaining = origin + scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            # Match the other clients: include tokenization in request service time.
            start = time.monotonic()
            tokens, first, completed = _generate(model, tokenizer, torch, sample['prompt'], 32, eos_ids)
            if not tokens or first is None:
                raise RuntimeError('Draft-only request produced no visible output token')
            row = {'task_id': sample['sample_id'], 'method': 'draft_only',
                'output_token_ids': tokens, 'output_text': tokenizer.decode(tokens, skip_special_tokens=True),
                'generated_tokens': len(tokens), 'scheduled_arrival_s': scheduled,
                'arrival_lag_ms': max(0.0, start - origin - scheduled) * 1000.0,
                'request_e2e_ms': (completed - start) * 1000.0,
                'ttft_ms': (first - start) * 1000.0,
                'tpot_ms': (completed - first) * 1000.0 / (len(tokens) - 1) if len(tokens) > 1 else None,
                'completion_s': completed - origin,
                'request_start_monotonic_s': start, 'completion_monotonic_s': completed,
                'experiment_t0_monotonic_s': origin, 'accepted_total': 0, 'drafted_total': 0,
                'output_equivalent_to_8b': False}
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            _rss_snapshot(output, 'request_complete', index + 1)
    _rss_snapshot(output, 'worker_complete', len(samples))
    finish_worker(args)