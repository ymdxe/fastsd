#!/usr/bin/env python3
"""Prepare configuration and directory structure for 32-worker CPU+GPU experiment.

This script:
1. Creates isolated experiment directory
2. Generates config files for FastSD and SpecEdge
3. Records Git SHA, model paths, and environment info
4. Outputs commands for smoke test (2 workers) and full run (32 workers)
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

INCOMPATIBLE_WITH_STATEFUL_FASTSD = (
    "This legacy preparer is deprecated for the four-method experiment: it "
    "launches one-shot /generate workers and multiple model copies. It does "
    "not implement EdgeClient /session/init + /prefill + /verify + rollback. "
    "Use scripts/eval_suite.py prepare/plan with the latency (1 process x 32 "
    "threads) or throughput (4 processes x 8 threads) track."
)


def get_git_sha(repo_path: Path) -> str:
    """Get current commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_file_sha256(path: Path) -> str:
    """Compute SHA256 of a file (for model checkpoints)."""
    if not path.exists():
        return "NOT_FOUND"
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_model_fingerprint(model_dir: Path) -> dict:
    """Get model fingerprint (config + first checkpoint file SHA)."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {"error": "config.json not found"}

    # Find first .safetensors or .bin file
    checkpoint = None
    for pattern in ["*.safetensors", "*.bin"]:
        files = list(model_dir.glob(pattern))
        if files:
            checkpoint = sorted(files)[0]
            break

    return {
        "config_sha256": get_file_sha256(config_path),
        "checkpoint": str(checkpoint.name) if checkpoint else None,
        "checkpoint_sha256": get_file_sha256(checkpoint) if checkpoint else None,
    }


def create_experiment_manifest(
    exp_dir: Path,
    run_id: str,
    draft_model: str,
    target_model: str,
    num_workers: int,
    dataset: str,
    node2_gpu: str,
    node3_cores: str,
) -> dict:
    """Create experiment manifest with all metadata."""

    fastsd_sha = get_git_sha(REPO_ROOT)
    specedge_sha = get_git_sha(REPO_ROOT / "baselines" / "specedge" / "official")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": {
            "fastsd": fastsd_sha,
            "specedge": specedge_sha,
        },
        "models": {
            "draft": {
                "path": draft_model,
                "fingerprint": get_model_fingerprint(Path(draft_model)),
            },
            "target": {
                "path": target_model,
                "fingerprint": get_model_fingerprint(Path(target_model)),
            },
        },
        "topology": {
            "node2": {
                "role": "target",
                "gpu": node2_gpu,
            },
            "node3": {
                "role": "draft",
                "device": "cpu",
                "cores": node3_cores,
                "num_workers": num_workers,
            },
        },
        "dataset": dataset,
        "experiment_dir": str(exp_dir),
    }

    return manifest


def write_fastsd_commands(
    exp_dir: Path,
    manifest: dict,
    gamma: int = 4,
    max_tokens: int = 256,
    temperature: float = 0.0,
    arrival_rate: float = 1.0,
    arrival_seed: int = 1234,
) -> None:
    """Write FastSD experiment commands."""

    commands_file = exp_dir / "fastsd" / "commands.txt"
    commands_file.parent.mkdir(parents=True, exist_ok=True)

    num_workers = manifest["topology"]["node3"]["num_workers"]
    draft_model = manifest["models"]["draft"]["path"]
    target_model = manifest["models"]["target"]["path"]
    node2_gpu = manifest["topology"]["node2"]["gpu"]

    commands = f"""# FastSD 32-Worker Experiment
# Generated: {datetime.now(timezone.utc).isoformat()}
# FastSD SHA: {manifest['git_sha']['fastsd']}

# ========== NODE3: Launch 32 CPU draft workers ==========
# SSH to node3
ssh node3

cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

# Check CPU topology
lscpu | grep -E "CPU\\(s\\)|Thread|Core|Socket|NUMA"
numactl --hardware

# Launch workers (one per core)
python scripts/launch_cpu_draft_workers.py \\
  --num_workers {num_workers} \\
  --start_core 0 \\
  --draft_model {draft_model} \\
  --base_port 19000 \\
  --log_dir {exp_dir}/fastsd/draft_logs \\
  --worker_script scripts/cpu_draft_worker.py \\
  --max_tokens {max_tokens} \\
  --temperature {temperature} \\
  --manifest_out {exp_dir}/fastsd/draft_workers_manifest.json

# Workers will listen on ports 19000-19031

# ========== NODE2: Launch target service ==========
# SSH to node2
ssh node2

cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

# Check GPU
nvidia-smi

FASTSD_TARGET_DEVICE={node2_gpu} \\
python cloud/cloud_service.py \\
  --target_model {target_model} \\
  --draft_model {draft_model} \\
  --dataset {manifest['dataset']} \\
  --server_sched_mode fastsd \\
  --batch_size {num_workers} \\
  --token_budget 512 \\
  > {exp_dir}/fastsd/target.out 2> {exp_dir}/fastsd/target.err &

# Target listens on http://127.0.0.1:8001

# ========== NODE1 (or local): Run coordinator ==========
# This machine coordinates draft workers and target

cd /home/hdd/zhangh/workspace/new_fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate

# TODO: Implement coordinator that:
# 1. Reads canonical.jsonl workload
# 2. Distributes requests to 32 draft workers on node3:19000-19031
# 3. Collects draft sequences
# 4. Sends verify requests to node2:8001
# 5. Records per-request metrics

# ========== Smoke Test (2 workers) ==========
# Use --num_workers 2 and verify:
# - Model loading
# - Network connectivity
# - KV cache offload
# - Metrics recording
"""

    with open(commands_file, "w") as f:
        f.write(commands)

    print(f"[FASTSD] Commands written to {commands_file}")


def write_specedge_commands(
    exp_dir: Path,
    manifest: dict,
    gamma: int = 4,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> None:
    """Write SpecEdge experiment commands with CPU KV offload."""

    commands_file = exp_dir / "specedge" / "commands.txt"
    commands_file.parent.mkdir(parents=True, exist_ok=True)

    num_workers = manifest["topology"]["node3"]["num_workers"]
    draft_model = manifest["models"]["draft"]["path"]
    target_model = manifest["models"]["target"]["path"]
    node2_gpu = manifest["topology"]["node2"]["gpu"]

    commands = f"""# SpecEdge 32-Client Experiment with CPU KV Offload
# Generated: {datetime.now(timezone.utc).isoformat()}
# SpecEdge SHA: {manifest['git_sha']['specedge']}

# ========== NODE2: Patched SpecEdge server with idle_kv_device=cpu ==========
ssh node2

cd /home/hdd/zhangh/workspace/new_fastsd/baselines/specedge/official
source /home/hdd/zhangh/envs/specedge/bin/activate

# Apply KV offload patch (see specedge_kv_offload_patch.py)
# This adds:
# - idle_kv_device config option (cpu/cuda)
# - CPU→GPU copy before batch execution
# - GPU→CPU copy after batch completion
# - Timing instrumentation for copy overhead

SPECEDGE_IDLE_KV_DEVICE=cpu \\
SPECEDGE_TARGET_DEVICE={node2_gpu} \\
python src/script/batch_server.py \\
  --config {exp_dir}/specedge/specedge_config.yaml \\
  > {exp_dir}/specedge/server.out 2> {exp_dir}/specedge/server.err &

# ========== NODE3: 32 SpecEdge clients on CPU ==========
ssh node3

cd /home/hdd/zhangh/workspace/new_fastsd/baselines/specedge/official
source /home/hdd/zhangh/envs/specedge/bin/activate

# Launch clients via integration adapter
python ../../integration/client_host.py \\
  --config {exp_dir}/specedge/specedge_config.yaml

# ========== Verify KV Offload Implementation ==========
# The patched server should log:
# - "KV offload enabled: idle_kv_device=cpu"
# - Per-batch CPU→GPU and GPU→CPU copy times
# - Total bytes transferred

# Check server stderr for these log lines
grep "KV offload" {exp_dir}/specedge/server.err
grep "copy_to_gpu_ms" {exp_dir}/specedge/server.err
grep "copy_to_cpu_ms" {exp_dir}/specedge/server.err
"""

    with open(commands_file, "w") as f:
        f.write(commands)

    print(f"[SPECEDGE] Commands written to {commands_file}")


def write_specedge_kv_offload_patch(exp_dir: Path) -> None:
    """Write the SpecEdge KV offload patch code."""

    patch_file = exp_dir / "specedge" / "specedge_kv_offload_patch.py"
    patch_file.parent.mkdir(parents=True, exist_ok=True)

    patch_code = '''"""Patch for SpecEdge batch_server.py to add CPU KV cache offloading.

This patch modifies the official SpecEdge server to:
1. Store idle request KV caches on CPU
2. Copy caches to GPU before batch execution
3. Copy caches back to CPU after batch completion
4. Instrument copy timing

Apply to: baselines/specedge/official/src/script/batch_server.py
"""

import time
import torch


# ========== Add to SpecExecBatchServer.__init__ ==========
def patch_init(self):
    # Original init code...

    # NEW: KV offload configuration
    self.idle_kv_device = os.environ.get("SPECEDGE_IDLE_KV_DEVICE", "cuda")
    if self.idle_kv_device not in {"cpu", "cuda"}:
        raise ValueError(f"Invalid idle_kv_device: {self.idle_kv_device}")

    self.kv_offload_enabled = self.idle_kv_device == "cpu"
    if self.kv_offload_enabled:
        print(f"[KV-OFFLOAD] Enabled: idle_kv_device={self.idle_kv_device}")

    self.kv_offload_stats = {
        "copy_to_gpu_ms": [],
        "copy_to_cpu_ms": [],
        "copy_to_gpu_bytes": [],
        "copy_to_cpu_bytes": [],
    }


# ========== Add helper methods ==========
def move_kv_cache_to_device(kv_cache, target_device: str):
    """Move KV cache tuple to target device."""
    if kv_cache is None:
        return None

    moved = []
    total_bytes = 0
    for layer_kv in kv_cache:
        if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
            k, v = layer_kv
            k_moved = k.to(target_device)
            v_moved = v.to(target_device)
            moved.append((k_moved, v_moved))
            total_bytes += k.numel() * k.element_size()
            total_bytes += v.numel() * v.element_size()
        else:
            moved.append(layer_kv)

    return tuple(moved), total_bytes


def patch_prepare_batch_kv_caches(self, request_ids):
    """Move KV caches from CPU to GPU before batch execution."""
    if not self.kv_offload_enabled:
        return

    start = time.time()
    total_bytes = 0

    for req_id in request_ids:
        if req_id in self.kv_cache_store:
            cache = self.kv_cache_store[req_id]
            if cache is not None:
                moved_cache, bytes_moved = move_kv_cache_to_device(
                    cache, f"cuda:{self.config.device}"
                )
                self.kv_cache_store[req_id] = moved_cache
                total_bytes += bytes_moved

    elapsed_ms = (time.time() - start) * 1000
    self.kv_offload_stats["copy_to_gpu_ms"].append(elapsed_ms)
    self.kv_offload_stats["copy_to_gpu_bytes"].append(total_bytes)

    print(f"[KV-OFFLOAD] copy_to_gpu: {elapsed_ms:.2f}ms, {total_bytes/1e6:.2f}MB")


def patch_offload_batch_kv_caches(self, request_ids):
    """Move KV caches from GPU to CPU after batch completion."""
    if not self.kv_offload_enabled:
        return

    start = time.time()
    total_bytes = 0

    for req_id in request_ids:
        if req_id in self.kv_cache_store:
            cache = self.kv_cache_store[req_id]
            if cache is not None:
                moved_cache, bytes_moved = move_kv_cache_to_device(cache, "cpu")
                self.kv_cache_store[req_id] = moved_cache
                total_bytes += bytes_moved

    elapsed_ms = (time.time() - start) * 1000
    self.kv_offload_stats["copy_to_cpu_ms"].append(elapsed_ms)
    self.kv_offload_stats["copy_to_cpu_bytes"].append(total_bytes)

    print(f"[KV-OFFLOAD] copy_to_cpu: {elapsed_ms:.2f}ms, {total_bytes/1e6:.2f}MB")


# ========== Modify batch execution ==========
def patch_execute_batch(self, batch_requests):
    """Execute batch with KV cache GPU residency."""

    request_ids = [req.request_id for req in batch_requests]

    # NEW: Move caches to GPU before execution
    self.prepare_batch_kv_caches(request_ids)

    # Original batch execution code...
    # result = self.model.forward(...)

    # NEW: Move caches back to CPU after execution
    self.offload_batch_kv_caches(request_ids)

    return result


# ========== Summary at shutdown ==========
def patch_print_kv_offload_summary(self):
    if not self.kv_offload_enabled:
        return

    stats = self.kv_offload_stats
    print("\\n[KV-OFFLOAD] Summary:")
    print(f"  Total copy_to_gpu: {len(stats['copy_to_gpu_ms'])} times")
    if stats['copy_to_gpu_ms']:
        print(f"    Mean: {sum(stats['copy_to_gpu_ms'])/len(stats['copy_to_gpu_ms']):.2f}ms")
        print(f"    Total bytes: {sum(stats['copy_to_gpu_bytes'])/1e9:.2f}GB")
    print(f"  Total copy_to_cpu: {len(stats['copy_to_cpu_ms'])} times")
    if stats['copy_to_cpu_ms']:
        print(f"    Mean: {sum(stats['copy_to_cpu_ms'])/len(stats['copy_to_cpu_ms']):.2f}ms")
        print(f"    Total bytes: {sum(stats['copy_to_cpu_bytes'])/1e9:.2f}GB")


# ========== MANUAL APPLICATION REQUIRED ==========
# This is a reference implementation. To apply:
# 1. Copy these methods into batch_server.py
# 2. Call prepare_batch_kv_caches before forward
# 3. Call offload_batch_kv_caches after forward
# 4. Call print_kv_offload_summary at shutdown
'''

    with open(patch_file, "w") as f:
        f.write(patch_code)

    print(f"[PATCH] SpecEdge KV offload reference written to {patch_file}")
    print(f"[PATCH] MANUAL APPLICATION REQUIRED - see file for instructions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True, help="Unique experiment ID")
    parser.add_argument("--exp_root", required=True, help="Root directory for experiments")
    parser.add_argument("--draft_model", required=True, help="Path to draft model (Qwen3-1.7B)")
    parser.add_argument("--target_model", required=True, help="Path to target model (Qwen3-8B)")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--dataset", default="humaneval", help="Dataset name")
    parser.add_argument("--node2_gpu", default="cuda:0", help="Node2 GPU (e.g., cuda:0)")
    parser.add_argument("--node3_cores", default="0-31", help="Node3 CPU cores")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma parameter")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--arrival_rate", type=float, default=1.0, help="Poisson arrival rate")
    parser.add_argument("--arrival_seed", type=int, default=1234)

    args = parser.parse_args()

    print(f"[INCOMPATIBLE] {INCOMPATIBLE_WITH_STATEFUL_FASTSD}", file=sys.stderr)
    return 2

    # Create experiment directory
    exp_dir = Path(args.exp_root) / args.run_id
    if exp_dir.exists():
        print(f"ERROR: Experiment directory already exists: {exp_dir}")
        print("Use a different --run_id or remove the existing directory")
        return 1

    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Created experiment directory: {exp_dir}")

    # Create manifest
    manifest = create_experiment_manifest(
        exp_dir=exp_dir,
        run_id=args.run_id,
        draft_model=args.draft_model,
        target_model=args.target_model,
        num_workers=args.num_workers,
        dataset=args.dataset,
        node2_gpu=args.node2_gpu,
        node3_cores=args.node3_cores,
    )

    manifest_path = exp_dir / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Written to {manifest_path}")

    # Write commands
    write_fastsd_commands(
        exp_dir=exp_dir,
        manifest=manifest,
        gamma=args.gamma,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        arrival_rate=args.arrival_rate,
        arrival_seed=args.arrival_seed,
    )

    write_specedge_commands(
        exp_dir=exp_dir,
        manifest=manifest,
        gamma=args.gamma,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    write_specedge_kv_offload_patch(exp_dir)

    print(f"\n[SUCCESS] Experiment prepared: {exp_dir}")
    print(f"\nNext steps:")
    print(f"1. Review commands in:")
    print(f"   - {exp_dir}/fastsd/commands.txt")
    print(f"   - {exp_dir}/specedge/commands.txt")
    print(f"2. Apply SpecEdge KV offload patch (see {exp_dir}/specedge/specedge_kv_offload_patch.py)")
    print(f"3. Run smoke test (2 workers)")
    print(f"4. Run full experiment (32 workers)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
