#!/usr/bin/env python3
"""Derive the documented Vanilla capacity gate from immutable run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def candidate_result(run_dir: Path, pilot_p95_ms: float) -> dict[str, Any]:
    root = load(run_dir / "manifest.json")
    if root.get("method") != "vanilla" or root.get("status") != "complete":
        raise ValueError(f"candidate is not a finalized complete Vanilla run: {run_dir}")
    summary = load(run_dir / "metrics" / "summary.json")
    workload = load(run_dir / "workload" / "manifest.json")
    rate = finite(workload.get("rate_rps"), label=f"{run_dir} workload rate_rps")
    completion = finite(summary.get("completion_rate"), label=f"{run_dir} completion_rate")
    p95 = finite(summary.get("task_e2e_ms_p95"), label=f"{run_dir} task_e2e_ms_p95")
    eligible = completion == 1.0 and p95 <= 2.0 * pilot_p95_ms
    return {
        "run_dir": str(run_dir),
        "run_id": root.get("run_id"),
        "rate_rps": rate,
        "completion_rate": completion,
        "p95_e2e_ms": p95,
        "eligible": eligible,
        "rejection_reason": None
        if eligible
        else "requires completion_rate == 1.0 and p95_e2e_ms <= 2 * pilot_p95_e2e_ms",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-edge-run", required=True)
    parser.add_argument("--candidate-run", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pilot_dir = Path(args.pilot_edge_run)
    output = Path(args.output)
    if not str(pilot_dir).startswith("/home/hdd/") or not str(output).startswith("/home/hdd/"):
        raise SystemExit("ERROR: pilot and output must be under /home/hdd")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to overwrite capacity decision: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"ERROR: output parent does not exist: {output.parent}")
    try:
        pilot = load(pilot_dir / "metrics" / "summary.json")
        pilot_mu0 = finite(pilot.get("system_req_per_s"), label="pilot system_req_per_s")
        pilot_p95 = finite(pilot.get("task_e2e_ms_p95"), label="pilot task_e2e_ms_p95")
        if finite(pilot.get("completion_rate"), label="pilot completion_rate") != 1.0:
            raise ValueError("closed-loop pilot must complete every request")
        candidates = [candidate_result(Path(item), pilot_p95) for item in args.candidate_run]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    accepted = [item for item in candidates if item["eligible"]]
    capacity = max(accepted, key=lambda item: item["rate_rps"]) if accepted else None
    result = {
        "schema_version": 1,
        "definition": "highest offered Vanilla rate with completion_rate == 1.0 and p95 E2E <= 2x closed-loop pilot P95",
        "pilot_edge_run": str(pilot_dir),
        "mu0_req_per_s": pilot_mu0,
        "pilot_p95_e2e_ms": pilot_p95,
        "capacity_mu_rps": capacity["rate_rps"] if capacity else None,
        "capacity_run_id": capacity["run_id"] if capacity else None,
        "candidates": candidates,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
