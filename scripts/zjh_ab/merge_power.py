#!/usr/bin/env python3
"""Merge target and edge power summaries for one experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.power_monitor import load_summary, merge_experiment_summaries


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--edge", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.target.is_file():
        raise SystemExit(f"target power summary missing: {args.target}")
    if not args.edge:
        raise SystemExit("edge power summary missing: no --edge path supplied")
    missing_edges = [path for path in args.edge if not path.is_file()]
    if missing_edges:
        missing = ", ".join(str(path) for path in missing_edges)
        raise SystemExit(f"edge power summary missing: {missing}")
    target = load_summary(args.target)
    edges = [load_summary(path) for path in args.edge]
    target_samples = load_jsonl(args.target.parent / "power_samples.jsonl")
    edge_samples = [load_jsonl(path.parent / "power_process_samples.jsonl") for path in args.edge]
    result = merge_experiment_summaries(
        target,
        edges,
        target_hardware_samples=target_samples,
        edge_process_samples=edge_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
