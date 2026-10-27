"""One target and one independent CPU draft per node; manual functional check."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['cloud', 'edge', 'summary'])
    p.add_argument('--method', choices=['standard', 'fastsd'], default='standard')
    p.add_argument('--run', default='trial1', help='Use a new name for every rerun')
    p.add_argument('--node', choices=['node1', 'node3'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cpus', help='Optional comma-separated CPU IDs; inspect lscpu first')
    p.add_argument('--port', type=int, default=19635)
    p.add_argument('--server-url', default='http://127.0.0.1:29635')
    p.add_argument('--draft-model', default='/mnt/nfs0/zhangh/models/Qwen3-0.6B')
    p.add_argument('--target-model', default='/mnt/nfs0/zhangh/models/Qwen3-1.7B')
    args = p.parse_args()
    if args.role == 'edge' and args.node is None:
        p.error('edge requires --node node1 or --node node3')
    if args.threads < 1:
        p.error('--threads must be positive')
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        p.error('--run must be a simple directory name')
    return args


def native_arguments(args, output):
    fast = args.method == 'fastsd'
    result = [
        '--draft_model', args.draft_model, '--target_model', args.target_model,
        '--draft_dtype', 'float32', '--target_dtype', 'float32',
        '--target_device', 'cuda:0', '--kv_batch_mode', 'padded',
        '--profile', 'custom' if fast else 'vanilla',
        '--server_sched_mode', 'fastsd' if fast else 'vanilla',
        '--gamma', '4', '--no-pipeline_gamma_adapt',
        '--pipeline_gamma_min', '4', '--pipeline_gamma_max', '4',
        '--max_tokens', '32', '--batch_size', '2', '--token_budget', '256',
        '--dataset', 'mt_bench', '--verify_method', 'greedy',
        '--temp', '0', '--top_k', '0', '--top_p', '1', '--seed', '1234',
        '--stop_policy', 'eos', '--warmup_requests', '0',
        '--arrival_distribution', 'immediate', '--no-measure_energy',
        '--exp_name', str(output),
    ]
    for feature in ('enable_pipeline', 'enable_proactive_draft', 'enable_latency_priority'):
        result.append('--' + ('' if fast else 'no-') + feature)
    return result


def summary(args):
    directory = ROOT / 'runs' / args.run / args.method
    paths = sorted(directory.glob('node*/edge_metrics_proc0.jsonl'))
    if args.node:
        paths = [p for p in paths if p.parent.name == args.node]
    if not paths:
        raise SystemExit('No completed requests found: ' + str(directory))
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        values = {'node': path.parent.name, 'completed': len(rows), 'expected': 2}
        for key in ('ttft_ms', 'tpot_ms', 'request_e2e_ms', 'avg_http_total_ms'):
            samples = [r[key] for r in rows if r.get(key) is not None]
            values[key + '_mean'] = statistics.mean(samples) if samples else None
        values['generated_tokens'] = [r['generated_tokens'] for r in rows]
        print(json.dumps(values, ensure_ascii=False, indent=2))
        for row in rows:
            print(row['task_id'], repr(row.get('output_text', '')))


def main():
    args = options()
    if args.role == 'summary':
        return summary(args)
    if args.role == 'cloud' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('Set CUDA_VISIBLE_DEVICES to an idle physical GPU before starting cloud')
    if args.role == 'edge':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if args.cpus:
            cpus = {int(item) for item in args.cpus.split(',')}
            if not cpus <= os.sched_getaffinity(0) or len(cpus) < args.threads:
                raise SystemExit('--cpus must be allowed and provide at least --threads CPUs')
            os.sched_setaffinity(0, cpus)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[key] = str(args.threads)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1')
    output = ROOT / 'runs' / args.run / args.method / ('cloud' if args.role == 'cloud' else args.node)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launch.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    cli = native_arguments(args, output)
    if args.role == 'cloud':
        os.environ.update(CLOUD_SERVICE_HOST='127.0.0.1', CLOUD_SERVICE_PORT=str(args.port))
        # Separate file lets multiprocessing spawn import the readiness patch.
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT / 'target.py'), *cli])
    import torch
    torch.set_num_interop_threads(1)
    import edge.edge as edge
    from clock_runtime import ClockRuntimeMixin
    base_client = edge.EdgeClient

    class Client(ClockRuntimeMixin, base_client):
        def _synchronize_runtime_clock(self, samples=5):
            return base_client.synchronize_clock(self, samples)

    edge.EdgeClient = Client
    os.environ['EXP_OUTPUT'] = str(output)
    cli += ['--edge_use_cpu', '--draft_device', 'cpu', '--edge_threads', str(args.threads),
            '--num_drafts', '1', '--max_tasks_per_draft', '2',
            '--dataset_file', str(ROOT / 'requests.jsonl'), '--server_url', args.server_url,
            '--request_timeout', '300']
    sys.argv = [sys.argv[0], *cli]
    parsed = edge.parse_edge_arguments()
    runner = edge.EdgeRunner(parsed)
    start = time.monotonic()
    runner.run_draft_process_http(runner.tokenizer, 0)
    runner._write_summary(time.monotonic() - start)
    summary(args)


if __name__ == '__main__':
    main()
