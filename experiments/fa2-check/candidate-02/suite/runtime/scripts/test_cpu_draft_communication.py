#!/usr/bin/env python3
"""Test script to verify CPU draft worker + GPU target communication.

This script:
1. Starts 1 CPU draft worker on node3
2. Connects to GPU target on node2
3. Sends a simple request
4. Reports the communication result
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def create_test_workload(output_path: Path) -> None:
    """Create a simple 1-request test workload."""
    workload = [
        {
            "task_id": "test_cpu_draft_1",
            "prompt": "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n",
            "sample_id": 0,
        }
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in workload:
            f.write(json.dumps(item) + "\n")

    print(f"[TEST] Created test workload: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft_model", required=True, help="Path to draft model (e.g., Qwen3-1.7B)")
    parser.add_argument("--target_url", default="http://127.0.0.1:8001", help="Target service URL")
    parser.add_argument("--exp_dir", required=True, help="Experiment output directory")
    parser.add_argument("--dataset", default="humaneval", help="Dataset name")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma parameter")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max new tokens (use small value for test)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create test workload
    workload_path = exp_dir / "test_workload.jsonl"
    create_test_workload(workload_path)

    # Prepare edge.py arguments
    edge_args = [
        "python", str(REPO_ROOT / "edge" / "edge.py"),
        "--draft_model", args.draft_model,
        "--server_url", args.target_url,
        "--profile", "vanilla",  # Use vanilla profile for simplicity
        "--exp_name", str(exp_dir),
        "--dataset", args.dataset,
        "--data_path", str(workload_path),
        "--num_drafts", "1",  # Single worker for test
        "--edge_use_cpu",  # NEW: Enable CPU mode
        "--gamma", str(args.gamma),
        "--max_tokens", str(args.max_tokens),
        "--temp", str(args.temperature),
        "--top_k", str(args.top_k),
        "--top_p", str(args.top_p),
        "--request_timeout", "60.0",
    ]

    print("\n" + "=" * 80)
    print("CPU Draft Worker + GPU Target Communication Test")
    print("=" * 80)
    print(f"\nDraft model: {args.draft_model}")
    print(f"Target URL: {args.target_url}")
    print(f"Device: CPU (edge_use_cpu=True)")
    print(f"Workload: {workload_path}")
    print(f"Output: {exp_dir}")
    print(f"\nCommand:")
    print(" ".join(edge_args))
    print("\n" + "=" * 80)

    # Run edge.py
    print("\n[TEST] Starting edge worker...")
    start_time = time.time()

    import subprocess
    result = subprocess.run(
        edge_args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"\nElapsed time: {elapsed:.2f}s")
    print(f"Return code: {result.returncode}")

    if result.returncode == 0:
        print("\n✓ SUCCESS: Communication test passed")
    else:
        print("\n✗ FAILED: Communication test failed")

    # Show stdout
    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    # Show stderr
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    # Check for output metrics
    metrics_files = list(exp_dir.glob("edge_metrics_proc*.jsonl"))
    if metrics_files:
        print(f"\n--- METRICS ---")
        for metrics_file in metrics_files:
            print(f"\nFile: {metrics_file.name}")
            with open(metrics_file) as f:
                for line in f:
                    record = json.loads(line)
                    print(json.dumps(record, indent=2))
    else:
        print("\n⚠ WARNING: No metrics files found")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if result.returncode == 0 and metrics_files:
        print("\n✓ Test PASSED:")
        print("  - CPU draft worker started successfully")
        print("  - Connected to GPU target")
        print("  - Completed request processing")
        print("  - Generated metrics")

        # Parse metrics for key stats
        for metrics_file in metrics_files:
            with open(metrics_file) as f:
                for line in f:
                    record = json.loads(line)
                    if "ttft_ms" in record and "e2e_ms" in record:
                        print(f"\nKey Metrics:")
                        print(f"  TTFT: {record['ttft_ms']:.2f} ms")
                        print(f"  E2E: {record['e2e_ms']:.2f} ms")
                        print(f"  TPOT: {record.get('tpot_ms', 0):.2f} ms")
                        if "accepted_tokens" in record:
                            print(f"  Accepted tokens: {record['accepted_tokens']}")
                        if "output_tokens" in record:
                            print(f"  Output tokens: {record['output_tokens']}")
    else:
        print("\n✗ Test FAILED:")
        if result.returncode != 0:
            print(f"  - Process exited with code {result.returncode}")
        if not metrics_files:
            print("  - No metrics generated")
        print("\nCheck stderr above for error details")

    print("\n" + "=" * 80)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
