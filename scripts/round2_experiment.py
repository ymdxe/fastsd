"""Freeze a versioned runtime and run the second-review ablation matrix.

Prepare: python scripts/round2_experiment.py prepare /absolute/new/package
Run:     python /package/runtime/scripts/round2_experiment.py run /package protect_forward cloud -- [model/options]
Run Edge in a second process with the same variant and common options.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
COMMON = ['--server_sched_mode', 'fastsd', '--kv_batch_mode', 'varlen', '--gamma', '4',
          '--batch_size', '4', '--token_budget', '512', '--min_prefill_chunk_tokens', '16',
          '--prefill_chunk_quantum', '128', '--seed', '1234']
MATRIX = {
    'off': ['--no-kv_prefetch', '--no-enable_proactive_draft', '--no-pipeline_gamma_adapt', '--scheduler_policy', 'fcfs'],
    'protect_forward': ['--kv_prefetch', '--kv_prefetch_mode', 'protect_forward', '--no-enable_proactive_draft', '--no-pipeline_gamma_adapt', '--scheduler_policy', 'fcfs'],
    'overlap': ['--kv_prefetch', '--kv_prefetch_mode', 'overlap', '--no-enable_proactive_draft', '--no-pipeline_gamma_adapt', '--scheduler_policy', 'fcfs'],
    'cpu_prepare': ['--kv_prefetch', '--kv_prefetch_mode', 'cpu_prepare', '--no-enable_proactive_draft', '--no-pipeline_gamma_adapt', '--scheduler_policy', 'fcfs'],
    'ahead': ['--kv_prefetch', '--kv_prefetch_mode', 'protect_forward', '--enable_proactive_draft', '--no-pipeline_gamma_adapt', '--scheduler_policy', 'fcfs'],
    'gamma': ['--kv_prefetch', '--kv_prefetch_mode', 'protect_forward', '--enable_proactive_draft', '--pipeline_gamma_adapt', '--gamma_policy', 'progress', '--scheduler_policy', 'fcfs'],
    'progress': ['--kv_prefetch', '--kv_prefetch_mode', 'protect_forward', '--enable_proactive_draft', '--pipeline_gamma_adapt', '--gamma_policy', 'progress', '--scheduler_policy', 'progress'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()


def prepare(output, common=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / 'runtime'
    for name in ('src', 'edge', 'cloud'):
        shutil.copytree(ROOT / name, runtime / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (runtime / 'scripts').mkdir()
    shutil.copyfile(__file__, runtime / 'scripts' / Path(__file__).name)
    try:
        sha, dirty = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--', 'src', 'edge', 'cloud', 'scripts/round2_experiment.py'))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, True
    hashes = {p.relative_to(runtime).as_posix(): digest(p) for p in runtime.rglob('*') if p.is_file()}
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha': sha,
                'source_dirty': dirty, 'source_root': str(ROOT), 'files': hashes,
                'common_argv': COMMON + list(common or []), 'variants': MATRIX,
                'comparison_order': list(MATRIX),
                'metrics': ['forward_gpu_ms', 'batch_start_delay_ms', 'exposed_h2d_stall_ms',
                            'prefetch_ready_hit', 'prefetch_pending_hit', 'prefetch_demand_miss',
                            'prefetch_wasted_bytes', 'prefetch_inflight_bytes', 'request_e2e_ms']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def verify_runtime(package):
    package = Path(package).resolve()
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    runtime = package / 'runtime'
    for name, expected in manifest['files'].items():
        path = (runtime / name).resolve()
        if not path.is_relative_to(runtime) or not path.is_file() or digest(path) != expected:
            raise ValueError('frozen runtime changed: ' + name)
    return manifest, runtime


def run(package, variant, role, extra):
    manifest, runtime = verify_runtime(package)
    extra = extra[1:] if extra and extra[0] == '--' else extra
    controlled = {'--kv_prefetch', '--no-kv_prefetch', '--kv_prefetch_mode',
                  '--enable_proactive_draft', '--no-enable_proactive_draft',
                  '--pipeline_gamma_adapt', '--no-pipeline_gamma_adapt',
                  '--gamma_policy', '--scheduler_policy'}
    if any(arg.split('=')[0] in controlled for arg in extra):
        raise ValueError('choose a matrix variant instead of overriding its switches')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    result = Path(package).resolve() / 'results' / variant / (stamp + '-' + role)
    result.mkdir(parents=True, exist_ok=False)
    argv = manifest['common_argv'] + manifest['variants'][variant] + extra + ['--exp_name', str(result)]
    sys.path.insert(0, str(runtime))
    modules = {name: importlib.import_module(name) for name in
               ('src.engine', 'src.cache_transfer', 'src.progress_control',
                'cloud.cloud_service' if role == 'cloud' else 'edge.edge')}
    paths = {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}
    if not all(Path(path).is_relative_to(runtime) for path in paths.values()):
        raise RuntimeError('import resolved outside the frozen runtime')
    sys.argv = [role] + argv
    entry = modules['cloud.cloud_service' if role == 'cloud' else 'edge.edge']
    args = entry.parse_arguments() if role == 'cloud' else entry.parse_edge_arguments()
    metadata = {'source_sha': manifest['source_sha'], 'source_dirty': manifest['source_dirty'],
                'runtime_manifest_sha256': digest(Path(package) / 'manifest.json'),
                'actual_import_paths': paths, 'argv': argv, 'resolved_parameters': vars(args),
                'python': sys.executable, 'variant': variant, 'role': role,
                'environment': {k: os.environ.get(k) for k in (
                    'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'FASTSD_KV_BLOCK_TOKENS', 'FASTSD_NATIVE_GQA', 'FASTSD_DISABLE_TARGET_CACHE_OFFLOAD')},
                'versions': {name: importlib.import_module(name).__version__ for name in ('torch', 'transformers')}}
    (result / 'runtime.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    if role == 'cloud':
        entry.main()
    else:
        entry.EdgeRunner(args).eval()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('output')
    prep.add_argument('--common-argv-json', help='JSON array of common model/data/runtime arguments')
    execute = commands.add_parser('run')
    execute.add_argument('package')
    execute.add_argument('variant', choices=list(MATRIX))
    execute.add_argument('role', choices=['cloud', 'edge'])
    execute.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'prepare':
        common = json.loads(Path(args.common_argv_json).read_text()) if args.common_argv_json else []
        prepare(args.output, common)
        print(str(Path(args.output).resolve() / 'manifest.json'))
    else:
        run(args.package, args.variant, args.role, args.extra)


if __name__ == '__main__':
    main()
