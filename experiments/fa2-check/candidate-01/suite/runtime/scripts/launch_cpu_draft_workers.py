#!/usr/bin/env python3
"""Launch multiple CPU-bound draft workers with explicit core affinity.

This script starts N independent draft worker processes, each:
- Pinned to a specific CPU core
- Loading its own copy of the draft model
- Setting OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1
- Listening on a unique port

Designed for node3 with 32 physical CPU cores running Qwen3-1.7B draft workers.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def get_cpu_core_list(num_cores: int, start_core: int = 0) -> list[int]:
    """Return a list of physical CPU core IDs."""
    return list(range(start_core, start_core + num_cores))


def launch_worker(
    worker_id: int,
    cpu_core: int,
    draft_model: str,
    base_port: int,
    log_dir: Path,
    python_bin: str,
    worker_script: Path,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    dataset: str = "humaneval",
) -> subprocess.Popen:
    """Launch one draft worker pinned to a specific CPU core."""

    port = base_port + worker_id
    worker_name = f"draft_worker_{worker_id}"

    # Environment: disable threading oversubscription
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "DRAFT_WORKER_ID": str(worker_id),
        "DRAFT_WORKER_PORT": str(port),
        "DRAFT_MODEL_PATH": draft_model,
        "CUDA_VISIBLE_DEVICES": "",  # Disable GPU for this worker
    })

    # Command: use taskset for CPU affinity
    command = [
        "taskset",
        "-c",
        str(cpu_core),
        python_bin,
        str(worker_script),
        "--draft_model", draft_model,
        "--port", str(port),
        "--worker_id", str(worker_id),
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--top_p", str(top_p),
        "--dataset", dataset,
        "--device", "cpu",
    ]

    stdout_path = log_dir / f"{worker_name}.out"
    stderr_path = log_dir / f"{worker_name}.err"

    with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stdout,
            stderr=stderr,
            cwd=Path(__file__).resolve().parents[1],
        )

    print(f"[LAUNCH] Worker {worker_id} on core {cpu_core}, port {port}, PID {process.pid}")
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_workers", type=int, default=32, help="Number of draft workers")
    parser.add_argument("--start_core", type=int, default=0, help="First CPU core ID")
    parser.add_argument("--draft_model", required=True, help="Path to draft model (e.g., Qwen3-1.7B)")
    parser.add_argument("--base_port", type=int, default=19000, help="Base port for workers")
    parser.add_argument("--log_dir", required=True, help="Directory for worker logs")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    parser.add_argument("--worker_script", required=True, help="Draft worker entry script")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--manifest_out", help="Write worker manifest JSON")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cpu_cores = get_cpu_core_list(args.num_workers, args.start_core)

    if len(cpu_cores) != args.num_workers:
        print(f"ERROR: Requested {args.num_workers} workers but only {len(cpu_cores)} cores available")
        return 1

    print(f"[SETUP] Launching {args.num_workers} draft workers on cores {cpu_cores[0]}-{cpu_cores[-1]}")
    print(f"[SETUP] Draft model: {args.draft_model}")
    print(f"[SETUP] Base port: {args.base_port}")
    print(f"[SETUP] Logs: {log_dir}")

    worker_script = Path(args.worker_script)
    if not worker_script.exists():
        print(f"ERROR: Worker script not found: {worker_script}")
        return 1

    processes = []
    manifest = {
        "num_workers": args.num_workers,
        "draft_model": args.draft_model,
        "base_port": args.base_port,
        "cpu_cores": cpu_cores,
        "workers": [],
    }

    for worker_id, cpu_core in enumerate(cpu_cores):
        try:
            process = launch_worker(
                worker_id=worker_id,
                cpu_core=cpu_core,
                draft_model=args.draft_model,
                base_port=args.base_port,
                log_dir=log_dir,
                python_bin=args.python,
                worker_script=worker_script,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                dataset=args.dataset,
            )
            processes.append(process)
            manifest["workers"].append({
                "worker_id": worker_id,
                "cpu_core": cpu_core,
                "port": args.base_port + worker_id,
                "pid": process.pid,
            })
            time.sleep(0.5)  # Stagger startup
        except Exception as e:
            print(f"ERROR: Failed to launch worker {worker_id}: {e}")
            # Cleanup already-started workers
            for p in processes:
                p.terminate()
            return 1

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[MANIFEST] Written to {manifest_path}")

    print(f"\n[SUCCESS] All {args.num_workers} workers started")
    print("[INFO] Press Ctrl+C to terminate all workers")

    try:
        # Wait for any worker to exit
        while True:
            for i, process in enumerate(processes):
                retcode = process.poll()
                if retcode is not None:
                    print(f"\n[EXIT] Worker {i} exited with code {retcode}")
                    # Terminate all others
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    return retcode
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Terminating all workers...")
        for process in processes:
            process.terminate()
        time.sleep(2)
        for process in processes:
            if process.poll() is None:
                process.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
