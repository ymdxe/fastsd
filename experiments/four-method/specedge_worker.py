"""Run one CPU-adapted official SpecEdge stream in the common suite protocol."""
import asyncio
import json
import os
from pathlib import Path
import platform
import sys
import time

from experiment_hooks import rpc, wait_for_start, finish_worker


def configure_environment(args, info, samples):
    identity = os.environ['FASTSD_SUITE_WORKER']
    node, worker = identity.rsplit('-w', 1)
    expected = info.get('expected_workers', [f'{n}-w{w}' for n in ('node1','node3') for w in range(8)])
    client_idx = expected.index(identity)
    if node not in ('node1', 'node3') or not 0 <= int(worker) < 8:
        raise ValueError('SpecEdge comparison requires node1/node3 worker IDs 0..7')
    grpc_target = info.get('specedge_grpc_target', '127.0.0.1:29636')
    env = {
        'SPECEDGE_OPTIMIZATION': 2,
        'SPECEDGE_RESULT_PATH': str(Path(args.exp_name) / 'specedge-raw'),
        'SPECEDGE_EXP_NAME': os.environ['FASTSD_SUITE_RUN_ID'],
        'SPECEDGE_PROCESS_NAME': identity,
        'SPECEDGE_SEED': 1234,
        'SPECEDGE_MAX_LEN': int(info.get('specedge_max_len', 2048)),
        'SPECEDGE_DRAFT_MODEL': args.draft_model,
        'SPECEDGE_DEVICE': 'cpu', 'SPECEDGE_DTYPE': 'fp32',
        'SPECEDGE_THREADS': int(args.edge_threads),
        'SPECEDGE_DATASET': 'fastsd_external', 'SPECEDGE_REASONING': 'False',
        'SPECEDGE_MAX_N_BEAMS': 32, 'SPECEDGE_MAX_BEAM_LEN': int(args.gamma),
        'SPECEDGE_MAX_BRANCH_WIDTH': 16, 'SPECEDGE_MAX_BUDGET': 32,
        'SPECEDGE_PROACTIVE_TYPE': 'included',
        'SPECEDGE_PROACTIVE_MAX_N_BEAMS': 32,
        'SPECEDGE_PROACTIVE_MAX_BEAM_LEN': 3,
        'SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH': 16,
        'SPECEDGE_PROACTIVE_MAX_BUDGET': 32,
        'SPECEDGE_MAX_NEW_TOKENS': int(args.max_tokens),
        'SPECEDGE_MAX_REQUEST_NUM': -1, 'SPECEDGE_REQ_OFFSET': 0,
        'SPECEDGE_SAMPLE_REQ_CNT': 1, 'SPECEDGE_HOST': grpc_target,
        'SPECEDGE_CLIENT_IDX': client_idx,
    }
    os.environ.update({key: str(value) for key, value in env.items()})
    return client_idx


async def _run(args):
    info = rpc(args.server_url, '/experiment/info')
    samples = [json.loads(line) for line in Path(args.inputs).read_text(encoding='utf-8').splitlines() if line.strip()]
    client_idx = configure_environment(args, info, samples)
    from specedge.bootstrap import prepare_imports
    provenance = prepare_imports()
    import grpc
    import torch
    import transformers
    import log
    import util
    from config import SpecEdgeClientConfig as config
    from client import IntegratedSpecExecClient
    from cpu_adapter import CPUCompatibleSpecEdgeEngine
    from src.runtime import configure_torch_threads
    from specedge_grpc import specedge_pb2, specedge_pb2_grpc

    configure_torch_threads(int(args.edge_threads))
    result_path = Path(args.exp_name) / 'specedge-raw'
    result_path.mkdir(parents=True, exist_ok=True)
    log.configure_logging(log.get_default_log_config(result_path, os.environ['FASTSD_SUITE_WORKER']))
    model = util.load_graph_model(args.draft_model, torch.device('cpu'), torch.float32)
    tokenizer = util.load_tokenizer(args.draft_model)
    engine = CPUCompatibleSpecEdgeEngine(model, config.max_len, config.max_n_beams)

    class MeasuredClient(IntegratedSpecExecClient):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            self.cycles = 0
            self.visible_token_times = []
        async def _cycle(self, req_idx, step_idx, prefill=False):
            fresh = await super()._cycle(req_idx, step_idx, prefill=prefill)
            self.cycles += 1
            self.visible_token_times.extend([time.monotonic()] * int(fresh.numel()))
            return fresh

    def prompt_text(record):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': record['prompt'].strip()}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    async def generate(record, req_idx, max_tokens):
        prompt = prompt_text(record)
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(input_ids) + max_tokens + 2 * int(config.max_budget) >= int(config.max_len):
            raise ValueError('Prompt/output/tree exceeds fixed SpecEdge max_len')
        start = time.monotonic()
        client = MeasuredClient(engine=engine, tokenizer=tokenizer, prompt=prompt, max_len=config.max_len)
        original = config.max_new_tokens
        config.max_new_tokens = int(max_tokens)
        try:
            await client.generate_exact(req_idx)
            end = time.monotonic()
            tokens = [int(token) for token in client._prefix_tokens[0, client._num_original_tokens:].tolist()]
            stats = client._validator.snapshot_stats()
            if not tokens or client.first_token_time is None:
                raise RuntimeError('SpecEdge produced an empty or incomplete result')
            first = client.visible_token_times[0]
            visible = client.visible_token_times[:len(tokens)]
            return start, first, end, tokens, client.cycles, stats, visible
        finally:
            config.max_new_tokens = original
            await client._validator._channel.close()
            engine.reset()

    # Reuse the official 16-client synchronization before warmup, then the
    # suite READY/T0 barrier after every client has really loaded and warmed.
    channel = grpc.aio.insecure_channel(config.host)
    try:
        stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
        await stub.Sync(specedge_pb2.SyncRequest(), timeout=1800)
    finally:
        await channel.close()
    fixture = samples[0] if samples else json.loads((Path(__file__).parent / 'reference/inputs.jsonl').read_text(encoding='utf-8').splitlines()[0])
    await generate(fixture, -1 - client_idx, 1)
    Path(args.exp_name, 'specedge-runtime.json').write_text(json.dumps({
        **provenance, 'python': sys.executable, 'python_version': sys.version,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'grpc': grpc.__version__, 'method': 'specedge_cpu_adapted',
        'draft_model': args.draft_model, 'draft_dtype': 'float32',
        'threads': int(args.edge_threads), 'max_beam_len': int(args.gamma),
        'tree_budget': 32, 'max_n_beams': 32, 'max_branch_width': 16,
        'proactive': 'included', 'proactive_max_beam_len': 3,
        'max_len': config.max_len, 'grpc_target': config.host,
        'client_idx': client_idx, 'warmup_requests': 1,
    }, indent=2), encoding='utf-8')
    origin = wait_for_start(args)
    with Path(args.exp_name, 'edge_metrics_proc0.jsonl').open('x', encoding='utf-8') as handle:
        for index, sample in enumerate(samples):
            scheduled = float(sample.get('scheduled_arrival_s', 0))
            remaining = origin + scheduled - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            started = time.monotonic()
            start, first, end, tokens, rounds, stats, visible = await generate(sample, client_idx * 1000000 + index, args.max_tokens)
            actual = started - origin
            request_ms = (end - started) * 1000
            ttft_ms = (first - started) * 1000
            row = dict(sample)
            row.update(task_id=sample['sample_id'], method='specedge_cpu_adapted',
                engine_adapter='specedge_cpu_adapted', client_idx=client_idx,
                gamma=int(args.gamma), tree_budget=32, verify_method='greedy',
                generated_tokens=len(tokens), output_token_ids=tokens,
                output_text=tokenizer.decode(tokens, skip_special_tokens=True),
                scheduled_arrival_s=scheduled, actual_arrival_s=actual,
                arrival_lag_ms=max(0, actual-scheduled) * 1000,
                request_e2e_ms=request_ms, ttft_ms=ttft_ms,
                tpot_ms=(request_ms-ttft_ms)/(len(tokens)-1) if len(tokens)>1 else None,
                completion_s=end-origin, request_start_monotonic_s=started,
                completion_monotonic_s=end, first_token_monotonic_s=first,
                token_visibility_s=[value-origin for value in visible],
                rounds=rounds, request_bytes=stats['request_bytes'],
                response_bytes=stats['response_bytes'], rpc_count=stats['rpc_count'],
                warmup_requests_per_client=1, warmup_included_in_metrics=False,
                workload_hash=os.environ.get('FASTSD_SUITE_TRACE_SHA256'))
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            handle.flush()
            print('SpecEdge completed '+sample['sample_id'], flush=True)
    finish_worker(args)


def run(args):
    return asyncio.run(_run(args))