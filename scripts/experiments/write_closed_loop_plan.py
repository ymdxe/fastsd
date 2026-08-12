#!/usr/bin/env python3
"""Record a deterministic closed-loop calibration input without inventing arrivals.

Closed-loop capacity pilots deliberately do not have a Poisson trace.  This
tool writes the exact static round-robin dataset assignment used by edge.py so
the component's command, model, data checksum, and request selection remain
auditable while the pilot stays excluded from open-loop matrix analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=("mt_bench", "gsm8k", "humaneval"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-requests", required=True, type=int)
    parser.add_argument("--num-clients", required=True, type=int)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    data = Path(args.data)
    workload = run_dir / "workload"
    plan_path = workload / "closed_loop_plan.jsonl"
    manifest_path = workload / "manifest.json"
    if not str(run_dir).startswith("/home/hdd/"):
        raise SystemExit(f"ERROR: --run-dir must be under /home/hdd: {run_dir}")
    if not data.is_file():
        raise SystemExit(f"ERROR: missing dataset file: {data}")
    if args.max_requests <= 0 or args.num_clients <= 0:
        raise SystemExit("ERROR: request/client counts must be positive")
    if args.max_requests % args.num_clients:
        raise SystemExit("ERROR: closed-loop request count must divide evenly across clients")
    if not workload.is_dir():
        raise SystemExit(f"ERROR: metadata workload directory missing: {workload}")
    if plan_path.exists() or plan_path.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise SystemExit("ERROR: refusing to replace a closed-loop workload artifact")
    records = [line for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < args.max_requests:
        raise SystemExit(
            f"ERROR: dataset has {len(records)} usable rows but {args.max_requests} were requested"
        )
    assignments: list[dict[str, int]] = []
    per_client = args.max_requests // args.num_clients
    for client_id in range(args.num_clients):
        for local_index in range(per_client):
            dataset_index = client_id + local_index * args.num_clients
            assignments.append(
                {
                    "client_id": client_id,
                    "local_index": local_index,
                    "dataset_index": dataset_index,
                }
            )
    with plan_path.open("x", encoding="utf-8", newline="\n") as handle:
        for item in assignments:
            handle.write(json.dumps(item, sort_keys=True))
            handle.write("\n")
    manifest = {
        "schema_version": 1,
        "arrival_mode": "closed_loop",
        "dataset": str(data.resolve()),
        "dataset_sha256": sha256(data),
        "dataset_format": args.dataset,
        "max_requests": args.max_requests,
        "num_clients": args.num_clients,
        "plan_path": plan_path.name,
        "plan_sha256": sha256(plan_path),
        "matrix_eligible": False,
        "reason": "closed-loop capacity calibration has no shared open-loop arrival trace",
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
