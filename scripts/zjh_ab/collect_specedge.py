#!/usr/bin/env python3
"""Normalize official/CPU-adapted SpecEdge JSONL into common FastSD fields."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common_metrics import evaluate_slo


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def normalize(row: dict[str, Any], index: int) -> dict[str, Any]:
    output_tokens = int(first(row, "output_tokens", "generated_tokens", "new_tokens") or 0)
    e2e = first(row, "e2e_ms", "request_e2e_ms", "latency_ms")
    ttft = first(row, "ttft_ms", "first_token_ms")
    tpot = first(row, "tpot_ms", "decode_token_ms")
    e2e = float(e2e) if e2e is not None else None
    return {
        "sample_id": str(first(row, "sample_id", "task_id", "request_id") or index),
        "task_id": str(first(row, "task_id", "sample_id") or index),
        "success": bool(row.get("success", row.get("status", "ok") not in {"error", "failed"})),
        "ttft_ms": float(ttft) if ttft is not None else None,
        "tpot_ms": float(tpot) if tpot is not None else None,
        "e2e_ms": e2e,
        "request_e2e_ms": e2e,
        "generated_tokens": output_tokens,
        "tok_per_s": output_tokens / (e2e / 1000.0) if e2e and e2e > 0 else None,
        "output_text": first(row, "output_text", "text", "output") or "",
        "reference": row.get("reference"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "humaneval"))
    parser.add_argument("--mode", choices=["official", "cpu-adapted"], default=os.environ.get("SPECEDGE_MODE", "official"))
    parser.add_argument("--expected-requests", type=int, default=0)
    parser.add_argument("--wallclock-s", type=float, default=0.0)
    args = parser.parse_args()
    rows = [normalize(row, index) for index, row in enumerate(load_rows(args.input))]
    output = {
        "schema_version": 1,
        "method": "specedge" if args.mode == "official" else "specedge_cpu_adapted",
        "mode": args.mode,
        "official_tp": False,
        "offload": False,
        "dataset": args.dataset,
        "num_requests": len(rows),
        "expected_requests": args.expected_requests or len(rows),
        "records": rows,
        "slo": evaluate_slo(rows, wallclock_s=args.wallclock_s or 0.0, dataset=args.dataset),
        "energy": None,
        "status": "complete" if len(rows) >= (args.expected_requests or len(rows)) else "incomplete",
    }
    energy = args.input / "energy_metrics.json"
    if energy.exists():
        output["energy"] = json.loads(energy.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("method", "mode", "num_requests", "status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
