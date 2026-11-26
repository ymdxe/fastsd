"""Launch the existing cloud or edge entry point for a matched comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('specedge_sched', 'fastsd_full', 'fastsd_sched_control')
REFERENCE = '1edcaf02ffc41a7b57726450c5357ed216a3b9bc'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['cloud', 'edge'], required=True)
    p.add_argument('--method', choices=METHODS, default=METHODS[0])
    p.add_argument('--config', type=Path, default=ROOT/'configs/baselines/specedge_sched.example.json')
    p.add_argument('--target-model', type=Path, required=True)
    p.add_argument('--draft-model', type=Path, required=True)
    p.add_argument('--workload', type=Path, required=True, help='same canonical JSONL on both machines')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--server-url', default='http://127.0.0.1:8001')
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--gamma', type=int, choices=[4, 8])
    p.add_argument('--clients', type=int)
    p.add_argument('--kv-budget-mb', type=int)
    p.add_argument('--print-only', action='store_true')
    a = p.parse_args()
    opts = json.loads(a.config.read_text(encoding='utf-8'))
    baseline = a.method == 'specedge_sched'
    full = a.method == 'fastsd_full'
    directory = a.run_dir.resolve()/a.method/a.role
    workload = a.workload.resolve()
    data = workload.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    if not rows or any('sample_id' not in r or 'prompt' not in r for r in rows):
        p.error('supply canonical JSONL containing sample_id and prompt for every request')
    if len({r['sample_id'] for r in rows}) != len(rows):
        p.error('sample_id values must be unique')
    for key, override in [('gamma', a.gamma), ('num_drafts', a.clients), ('kv_residency_budget_mb', a.kv_budget_mb)]:
        if override is not None:
            opts[key] = override
    if opts['gamma'] not in (4, 8) or opts['num_drafts'] < 1:
        p.error('comparison requires gamma 4 or 8 and a positive client count')
    if opts['kv_residency_budget_mb'] > 0 and opts['kv_residency_activation'] == 'bypass':
        p.error('positive KV budgets require lazy or eager activation')
    opts.update(target_model=str(a.target_model.resolve()), draft_model=str(a.draft_model.resolve()),
        dataset_file=str(workload), workload_hash=hashlib.sha256(data).hexdigest(), exp_name=str(directory),
        profile='custom', server_sched_mode='specedge' if baseline else 'fastsd',
        enable_latency_priority=not baseline, enable_pipeline=True, enable_proactive_draft=False,
        overlap_prefill_first_draft=True, pipeline_gamma_adapt=False, kv_prefetch=full,
        kv_residency_victim_policy='cost_minus_idle_descending' if full else 'lru')
    if a.role == 'edge':
        opts.update(server_url=a.server_url, request_timeout=120.0, case='sd')
    module = 'cloud.cloud_service' if a.role == 'cloud' else 'edge.edge'
    command = [sys.executable, '-m', module]
    for key, value in opts.items():
        if isinstance(value, bool):
            command.append(('--' if value else '--no-') + key)
        else:
            command.extend(['--' + key, str(value)])
    env = dict(os.environ, CLOUD_SERVICE_HOST=a.bind, CLOUD_SERVICE_PORT=str(a.port),
               FASTSD_TARGET_MODE='speculative', FASTSD_COMPARISON_METHOD=a.method)
    print(json.dumps({'argv': command, 'bind': a.bind, 'port': a.port, 'method': a.method}, indent=2), flush=True)
    if a.print_only:
        return 0
    import torch
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(method=a.method, role=a.role, git_sha=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), implementation='fastsd_shared_backend',
        official_reference_commit=REFERENCE, draft_structure='single_chain', gamma_adaptation=False,
        prefill_protocol='fastsd_separate_rpc', predictive_kv_prefetch=full and opts['kv_residency_budget_mb'] > 0,
        effective_token_budget=None if baseline else opts['token_budget'], tokenizer=opts['draft_model'],
        expected_requests=len(rows), effective_options=opts, argv=command, bind=a.bind, port=a.port,
        host=platform.node(), cpu=platform.processor(), torch_version=torch.__version__,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        diagnostics_dir=env.get('FASTSD_DIAGNOSTIC_DIR'), workload_sha256=opts['workload_hash'],
        arrival_seed=opts['arrival_seed'])
    if baseline:
        manifest.update(baseline_scope='server_scheduler_only', scheduler='dynamic_fifo', batch_wait_ms=0,
            admission_limit='batch_size_only', phase_order='serial_prefill_then_batched_verify',
            memory_adaptation='lru_demand' if opts['kv_residency_budget_mb'] else 'resident')
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    result = subprocess.run(command, cwd=ROOT, env=env)
    if a.role == 'edge':
        def indices(pattern):
            return {json.loads(line)['global_sample_index'] for f in directory.glob(pattern)
                    for line in f.read_text(encoding='utf-8').splitlines()}
        completed = indices('edge_metrics_proc*.jsonl')
        failed = indices('failed_requests_proc*.jsonl')
        missing = sorted(set(range(len(rows))) - completed - failed)
        coverage = dict(expected=len(rows), completed=len(completed), failed=len(failed),
            unfinished_sample_ids=[rows[i]['sample_id'] for i in missing], exit_code=result.returncode)
        (directory/'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
        if failed or missing:
            return result.returncode or 1
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
