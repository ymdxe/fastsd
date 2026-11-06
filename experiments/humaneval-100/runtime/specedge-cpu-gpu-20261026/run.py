"""Small manual CPU Draft / GPU Target harness for pinned SpecEdge."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def command_output(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    return {'exit_code': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def verify_bundle():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    bad = [name for name, digest in manifest['files'].items()
           if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if bad:
        raise RuntimeError('Bundle mismatch: ' + ', '.join(bad))
    return manifest


def check():
    manifest = verify_bundle()
    # Cache the bundled runtime before the existing integration adjusts sys.path.
    from src.runtime import configure_torch_threads
    from specedge.validate import source_check, dependency_check
    result = {'bundle_id': manifest['bundle_id'], 'hostname': socket.gethostname(),
              'source': source_check(), 'dependencies': dependency_check()}
    model = Path(os.environ['SPECEDGE_DRAFT_MODEL_PATH'] if os.environ['SPECEDGE_ROLE'] == 'cpu'
                 else '/home/hdd/zhangh/models/Qwen3-8B')
    if not (model / 'config.json').is_file():
        raise FileNotFoundError(f'Missing local model config: {model}')
    if not any(model.glob('*.safetensors')) and not any(model.glob('*.bin')):
        raise FileNotFoundError(f'Missing local model weights: {model}')
    result['local_model'] = str(model)
    result['model_weights_found'] = True
    result['runtime_module'] = sys.modules['src.runtime'].__file__
    return result


def run_root(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise ValueError('Use a new run ID containing only letters, digits, dot, dash, underscore')
    return Path(os.environ['SPECEDGE_RESULT_ROOT']) / run_id


def create_output(args, leaf):
    manifest = verify_bundle()
    output = run_root(args.run_id) / leaf
    output.mkdir(parents=True, exist_ok=False)
    repo = ROOT.parent
    metadata = {'bundle_id': manifest['bundle_id'], 'args': vars(args), 'python': sys.executable,
                'hostname': socket.gethostname(), 'created_epoch_s': time.time(),
                'method': 'specedge_cpu_adapted',
                'git_head': command_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
                'git_status': command_output(['git', '-C', str(repo), 'status', '--short', '--branch']),
                'disk': command_output(['df', '-h', str(output)]),
                'network': 'SSH forwarding through the Windows desktop; functional test',
                'upstream_commit': manifest['upstream_commit']}
    write_json(output / 'launch.json', metadata)
    (output / 'commands.txt').write_text(shlex.join([sys.executable, '-B', str(ROOT / 'run.py'), *sys.argv[1:]]) + '\n', encoding='utf-8')
    print(f'Output: {output}', flush=True)
    return output


def server(args):
    if os.environ['SPECEDGE_ROLE'] != 'gpu':
        raise RuntimeError('Run server on gpunode2')
    if not 1 <= args.num_clients <= 16 or not 1 <= args.batch_size <= args.num_clients:
        raise ValueError('Require 1 <= batch-size <= num-clients <= 16')
    # Read the requested physical GPU before importing torch or selecting CUDA.
    gpu_state = command_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=index,uuid,name,memory.total,memory.used', '--format=csv,noheader'])
    if gpu_state['exit_code']:
        raise RuntimeError(gpu_state)
    gpu_uuid = gpu_state['stdout'].split(',')[1].strip()
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', gpu_uuid):
        raise RuntimeError(f'Cannot identify the selected physical GPU: {gpu_state}')
    compute = command_output(['nvidia-smi', '-i', gpu_uuid, '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if compute['exit_code'] or compute['stdout']:
        raise RuntimeError(f'Physical GPU {args.gpu} has compute processes or cannot be inspected: {compute}')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    output = create_output(args, 'server')
    write_json(output / 'gpu.json', {'before': gpu_state, 'physical_gpu': args.gpu,
               'gpu_uuid': gpu_uuid, 'cuda_visible_devices': gpu_uuid, 'logical_gpu': 'cuda:0'})
    # CUDA and nvidia-smi may enumerate these cards differently. Pin the UUID
    # of the card that passed the occupancy check rather than reusing its index.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    config = {'run_id': args.run_id, 'output_dir': str(output),
              'model': '/home/hdd/zhangh/models/Qwen3-8B', 'target_device': 'cuda:0',
              'num_clients': args.num_clients, 'batch_size': args.batch_size,
              'specedge_max_len': 2048, 'tree_budget': 32, 'request_timeout': 1800}
    write_json(output / 'specedge-config.json', config)
    cmd = [sys.executable, '-B', '-u', str(ROOT / 'specedge_server.py'), '--config',
           str(output / 'specedge-config.json'), '--host', '127.0.0.1', '--port', str(args.port)]
    process = subprocess.Popen(cmd, start_new_session=True)
    announced = False
    exit_code = None
    try:
        while process.poll() is None:
            if not announced and (output / 'specedge-ready.json').is_file():
                print(f'TARGET READY: {args.run_id}; clients={args.num_clients}; start the CPU client(s) now.', flush=True)
                announced = True
            time.sleep(0.5)
        exit_code = process.returncode
    except KeyboardInterrupt:
        print('Stopping this Target and its child process.', flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            exit_code = process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=25)
        write_json(output / 'exit-status.json', {'exit_code': exit_code, 'target_ready': announced})
    return exit_code if exit_code is not None else 1


def client_environment(args, output):
    values = {
        'SPECEDGE_OPTIMIZATION': 2, 'SPECEDGE_RESULT_PATH': str(output / 'specedge-raw'),
        'SPECEDGE_EXP_NAME': args.run_id, 'SPECEDGE_PROCESS_NAME': f'client-{args.client_idx}',
        'SPECEDGE_SEED': 1234, 'SPECEDGE_MAX_LEN': 2048,
        'SPECEDGE_DRAFT_MODEL': os.environ['SPECEDGE_DRAFT_MODEL_PATH'],
        'SPECEDGE_DEVICE': 'cpu', 'SPECEDGE_DTYPE': 'fp32', 'SPECEDGE_THREADS': args.threads,
        'SPECEDGE_DATASET': 'fastsd_external', 'SPECEDGE_REASONING': 'False',
        'SPECEDGE_MAX_N_BEAMS': 32, 'SPECEDGE_MAX_BEAM_LEN': args.gamma,
        'SPECEDGE_MAX_BRANCH_WIDTH': 16, 'SPECEDGE_MAX_BUDGET': 32,
        'SPECEDGE_PROACTIVE_TYPE': 'included', 'SPECEDGE_PROACTIVE_MAX_N_BEAMS': 32,
        'SPECEDGE_PROACTIVE_MAX_BEAM_LEN': 3, 'SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH': 16,
        'SPECEDGE_PROACTIVE_MAX_BUDGET': 32, 'SPECEDGE_MAX_NEW_TOKENS': args.max_tokens,
        'SPECEDGE_MAX_REQUEST_NUM': -1, 'SPECEDGE_REQ_OFFSET': 0,
        'SPECEDGE_SAMPLE_REQ_CNT': args.requests, 'SPECEDGE_HOST': args.target,
        'SPECEDGE_CLIENT_IDX': args.client_idx}
    os.environ.update({key: str(value) for key, value in values.items()})
    write_json(output / 'config.json', values)


async def client(args, output):
    client_environment(args, output)
    from src.runtime import configure_torch_threads
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
    from specedge_grpc import specedge_pb2, specedge_pb2_grpc
    configure_torch_threads(args.threads)
    log.configure_logging(log.get_default_log_config(output / 'specedge-raw', f'client-{args.client_idx}'))
    util.set_seed(1234)
    records = [json.loads(line) for line in (ROOT / 'reference' / 'inputs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    # Different clients receive different prompts; a single client starts at zero.
    selected = records[args.client_idx::args.num_clients][:args.requests]
    if len(selected) != args.requests:
        raise ValueError('Requested more prompts than available in this client shard')
    write_json(output / 'inputs.json', selected)
    print(f'Loading CPU Draft: {config.draft_model}; threads={args.threads}', flush=True)
    model = util.load_graph_model(config.draft_model, torch.device('cpu'), torch.float32)
    tokenizer = util.load_tokenizer(config.draft_model)
    engine = CPUCompatibleSpecEdgeEngine(model, config.max_len, config.max_n_beams)
    write_json(output / 'runtime.json', dict(provenance, python=sys.version,
               torch=torch.__version__, transformers=transformers.__version__, grpc=grpc.__version__,
               threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
               draft_dtype='float32', model=config.draft_model, target=args.target,
               cuda_initialized=torch.cuda.is_initialized()))
    channel = grpc.aio.insecure_channel(args.target)
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=30)
        print(f'Connected. Waiting for all {args.num_clients} configured client(s) at SpecEdge Sync.', flush=True)
        await specedge_pb2_grpc.SpecEdgeServiceStub(channel).Sync(specedge_pb2.SyncRequest(), timeout=300)
    finally:
        await channel.close()

    async def generate(record, request_idx, token_limit):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': record['prompt'].strip()}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer.encode(prompt)) + token_limit + 2 * config.max_budget >= config.max_len:
            raise ValueError('Prompt + output + tree exceeds max_len=2048')
        original = config.max_new_tokens
        config.max_new_tokens = token_limit
        instance = None
        started = time.perf_counter()
        try:
            instance = IntegratedSpecExecClient(engine=engine, tokenizer=tokenizer, prompt=prompt, max_len=config.max_len)
            await asyncio.wait_for(instance.generate_exact(request_idx), timeout=1800)
            duration = time.perf_counter() - started
            tokens = [int(value) for value in instance._prefix_tokens[0, instance._num_original_tokens:].tolist()]
            if not tokens or instance.first_token_time is None:
                raise RuntimeError('No completed output tokens')
            return {'source_index': record.get('source_index'), 'output_token_ids': tokens,
                    'output_text': tokenizer.decode(tokens, skip_special_tokens=True),
                    'generated_tokens': len(tokens), 'request_e2e_ms': duration * 1000,
                    'ttft_ms': (instance.first_token_time - started) * 1000,
                    'transport': instance._validator.snapshot_stats()}
        finally:
            config.max_new_tokens = original
            if instance is not None:
                await instance._validator._channel.close()
            engine.reset()

    print('Running one untimed warmup request (1 output token).', flush=True)
    await generate(selected[0], -1 - args.client_idx, 1)
    rows = []
    with (output / 'requests.jsonl').open('x', encoding='utf-8') as handle:
        for index, record in enumerate(selected):
            row = await generate(record, args.client_idx * 1000000 + index, args.max_tokens)
            row.update(method='specedge_cpu_adapted', client_idx=args.client_idx, gamma=args.gamma,
                       tree_budget=32, warmup_included=False)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            rows.append(row)
            print(f'COMPLETED {index + 1}/{args.requests}: {row["generated_tokens"]} tokens, {row["request_e2e_ms"] / 1000:.2f}s', flush=True)
    summary = {'method': 'specedge_cpu_adapted', 'all_requests_completed': len(rows) == args.requests,
               'requests': len(rows), 'generated_tokens': sum(row['generated_tokens'] for row in rows),
               'mean_request_e2e_ms': sum(row['request_e2e_ms'] for row in rows) / len(rows),
               'mean_ttft_ms': sum(row['ttft_ms'] for row in rows) / len(rows),
               'network': 'SSH via desktop; functional test; not an IB performance measurement',
               'output_equivalence_checked': False, 'warmup_requests': 1, 'warmup_included': False}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('check')
    srv = sub.add_parser('server')
    srv.add_argument('run_id')
    srv.add_argument('--gpu', type=int, required=True, help='physical GPU index, required')
    srv.add_argument('--num-clients', type=int, default=1)
    srv.add_argument('--batch-size', type=int, default=1)
    srv.add_argument('--port', type=int, default=19736)
    cli = sub.add_parser('client')
    cli.add_argument('run_id')
    cli.add_argument('--client-idx', type=int, default=0)
    cli.add_argument('--num-clients', type=int, default=1)
    cli.add_argument('--requests', type=int, default=4)
    cli.add_argument('--max-tokens', type=int, default=32)
    cli.add_argument('--threads', type=int, default=8)
    cli.add_argument('--gamma', type=int, default=4)
    cli.add_argument('--target', default='127.0.0.1:29736')
    status = sub.add_parser('status')
    status.add_argument('run_id')
    args = parser.parse_args()
    if args.action == 'check':
        print(json.dumps(check(), indent=2))
        return 0
    if args.action == 'server':
        return server(args)
    if args.action == 'status':
        directory = run_root(args.run_id) / 'server'
        for name in ('specedge-failed.json', 'specedge-ready.json', 'exit-status.json'):
            path = directory / name
            if path.is_file():
                print(name + ': ' + path.read_text(encoding='utf-8'))
        if not (directory / 'specedge-ready.json').exists():
            print('Target is not ready; inspect the server console.')
        return 0
    if os.environ['SPECEDGE_ROLE'] != 'cpu':
        raise RuntimeError('Run client on a CPU node')
    if not 0 <= args.client_idx < args.num_clients <= 16:
        raise ValueError('Require 0 <= client-idx < num-clients <= 16')
    if min(args.requests, args.max_tokens, args.threads, args.gamma) < 1:
        raise ValueError('requests, max-tokens, threads, gamma must be positive')
    output = create_output(args, f'{socket.gethostname().split(".")[0]}-client{args.client_idx}')
    try:
        asyncio.run(client(args, output))
    except BaseException as exc:
        write_json(output / 'exit-status.json', {'exit_code': 1, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    else:
        write_json(output / 'exit-status.json', {'exit_code': 0})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
