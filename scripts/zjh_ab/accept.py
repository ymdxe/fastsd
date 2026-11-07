#!/usr/bin/env python3
"""Acceptance report for the three-condition KV residency / SD-vs-FastSD A/B.

Usage:
    python3 scripts/zjh_ab/accept.py exp/zjh_ab
    python3 scripts/zjh_ab/accept.py exp/zjh_ab/sd-cpu exp/zjh_ab/sd-a5000 exp/zjh_ab/fastsd

Reads, per condition:
    <cond>/target/scheduler_metrics.json   (fastsd_scheduler.kv_residency / .cache_transfers)
    <cond>/edge/edge_metrics_proc*.jsonl   (per-request records)

Prints a comparison table, a correctness diff and a pass/fail list.  Exit code
is 1 when any required gate fails, so it can be used directly in CI or a
pre-registered acceptance script.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

CONDITIONS = ("sd-cpu", "sd-a5000", "fastsd", "pressure-fastsd")

# Metrics that must stay at zero while the residency manager is in BYPASS.
BYPASS_ZERO = (
    "h2d_bytes",
    "d2h_bytes",
    "copies",
)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_scheduler(cond_dir):
    path = os.path.join(cond_dir, "target", "scheduler_metrics.json")
    if not os.path.exists(path):
        return {}
    payload = load_json(path)
    return payload.get("fastsd_scheduler", payload) or {}


def load_records(cond_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(cond_dir, "edge", "edge_metrics_proc*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def summarize(cond_dir):
    scheduler = load_scheduler(cond_dir)
    residency = scheduler.get("kv_residency") or {}
    transfers = scheduler.get("cache_transfers") or {}
    records = load_records(cond_dir)

    # Monotonic clocks are host-local; dual-edge results must use wall-clock
    # epochs or the merged span is meaningless (and tok/s collapses to ~0).
    starts = [r["processing_start_epoch_s"] for r in records
              if r.get("processing_start_epoch_s") is not None]
    ends = [r["completion_epoch_s"] for r in records
            if r.get("completion_epoch_s") is not None]
    if not starts or not ends:
        # Backward compatibility for single-host runs written before epoch
        # timestamps existed.
        starts = [r["processing_start_monotonic_s"] for r in records
                  if r.get("processing_start_monotonic_s") is not None]
        ends = [r["completion_monotonic_s"] for r in records
                if r.get("completion_monotonic_s") is not None]
    span = (max(ends) - min(starts)) if starts and ends else 0.0
    tokens = sum(int(r.get("generated_tokens") or 0) for r in records)
    e2e = [r.get("request_e2e_ms") or r.get("e2e_ms") or 0.0 for r in records]
    ttft = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    tpot = [r["tpot_ms"] for r in records if r.get("tpot_ms") is not None]

    return {
        "dir": cond_dir,
        "requests": len(records),
        "tokens": tokens,
        "span_s": span,
        "tok_per_s": (tokens / span) if span > 0 else 0.0,
        "e2e_p50": percentile(e2e, 0.50),
        "e2e_p95": percentile(e2e, 0.95),
        "ttft_p50": percentile(ttft, 0.50),
        "tpot_p50": percentile(tpot, 0.50),
        "mode": residency.get("mode"),
        "activations": residency.get("residency_activations", 0),
        "evictions": residency.get("residency_evictions", 0),
        "restores": residency.get("residency_restores", 0),
        "used_bytes": residency.get("used_bytes", 0),
        "eviction_errors": scheduler.get("kv_residency_eviction_errors", 0),
        "h2d_bytes": transfers.get("h2d_bytes", 0.0),
        "d2h_bytes": transfers.get("d2h_bytes", 0.0),
        "copies": transfers.get("copies", 0.0),
        "spec_bytes": transfers.get("speculative_submitted_bytes", 0.0),
        "lookahead_calls": scheduler.get("lookahead_ms_count", 0) or 0,
        "ready_hit": scheduler.get("prefetch_ready_hit", 0),
        "pending_hit": scheduler.get("prefetch_pending_hit", 0),
        "demand_miss": scheduler.get("prefetch_demand_miss", 0),
        "records": records,
    }


def fmt_bytes(value):
    value = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def print_table(rows):
    headers = ("condition", "n", "tok/s", "E2E p50", "TTFT p50", "TPOT p50",
               "KV mode", "evict", "H2D", "D2H", "copies", "lookahead", "ready")
    line = "{:<34}{:>4}{:>8}{:>10}{:>10}{:>9}{:>9}{:>7}{:>9}{:>9}{:>8}{:>10}{:>7}"
    print(line.format(*headers))
    print("-" * 128)
    for row in rows:
        print(line.format(
            os.path.basename(row["dir"].rstrip("/")),
            row["requests"],
            f'{row["tok_per_s"]:.2f}',
            f'{row["e2e_p50"]:.0f}',
            f'{row["ttft_p50"]:.0f}',
            f'{row["tpot_p50"]:.0f}',
            str(row["mode"]),
            row["evictions"],
            fmt_bytes(row["h2d_bytes"]),
            fmt_bytes(row["d2h_bytes"]),
            f'{float(row["copies"]):.0f}',
            row["lookahead_calls"],
            row["ready_hit"],
        ))


def compare_outputs(rows):
    """Return per-condition task_id -> output_text maps and their diff."""

    tables = {}
    for row in rows:
        name = os.path.basename(row["dir"].rstrip("/"))
        tables[name] = {str(r.get("task_id")): r.get("output_text") for r in row["records"]
                        if r.get("task_id") is not None}
    names = list(tables)
    baseline = names[0] if names else None
    diff = {}
    for name in names[1:]:
        shared = set(tables[baseline]) & set(tables[name])
        mismatch = sorted(task for task in shared
                          if tables[baseline][task] != tables[name][task])
        diff[name] = {"comparable": len(shared), "mismatch": len(mismatch),
                      "examples": mismatch[:5]}
    return baseline, diff


def gates(rows, tolerance):
    """Evaluate the acceptance criteria that apply to this branch."""

    results = []
    by_name = {os.path.basename(r["dir"].rstrip("/")): r for r in rows}

    for name, row in by_name.items():
        if row["mode"] == "bypass":
            bad = [key for key in BYPASS_ZERO if float(row[key] or 0) > 0]
            ok = not bad and row["lookahead_calls"] == 0 and row["evictions"] == 0
            results.append((f"{name}: BYPASS performs zero cache work", ok,
                            "no tickets/copies/lookahead" if ok else f"unexpected {bad}"))
        elif row["mode"] == "active":
            ok = row["activations"] > 0 and row["evictions"] > 0
            results.append((f"{name}: ACTIVE evicts under budget", ok,
                            f"activations={row['activations']} evictions={row['evictions']}"))
            results.append((f"{name}: no eviction errors", row["eviction_errors"] == 0,
                            f"errors={row['eviction_errors']}"))

    if "sd-cpu" in by_name and "sd-a5000" in by_name:
        cpu = by_name["sd-cpu"]["tok_per_s"]
        gpu = by_name["sd-a5000"]["tok_per_s"]
        speedup = (gpu / cpu - 1.0) if cpu > 0 else 0.0
        results.append(("sd-a5000 vs sd-cpu recorded", True,
                        f"{speedup * 100:+.1f}% draft-device effect"))

    if "fastsd" in by_name and "sd-cpu" in by_name:
        # FastSD runs the CPU draft, so sd-cpu is its controlled baseline.
        base = by_name["sd-cpu"]["tok_per_s"]
        fast = by_name["fastsd"]["tok_per_s"]
        delta = (fast / base - 1.0) if base > 0 else 0.0
        results.append(("fastsd not slower than sd-cpu beyond tolerance",
                        delta >= -tolerance,
                        f"{delta * 100:+.1f}% (tolerance {tolerance * 100:.1f}%)"))

    for row in rows:
        name = os.path.basename(row["dir"].rstrip("/"))
        results.append((f"{name}: requests completed", row["requests"] > 0,
                        f"{row['requests']} records"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="condition dirs or a parent dir")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="allowed FastSD slowdown vs sd-a5000 (default 5%%)")
    args = parser.parse_args()

    dirs = []
    for path in args.paths:
        if os.path.isdir(os.path.join(path, "target")) or os.path.isdir(os.path.join(path, "edge")):
            dirs.append(path)
        else:
            dirs.extend(os.path.join(path, name) for name in CONDITIONS
                        if os.path.isdir(os.path.join(path, name)))
    if not dirs:
        print("no condition directories found", file=sys.stderr)
        return 2

    rows = [summarize(directory) for directory in dirs]
    print_table(rows)
    print()

    baseline, diff = compare_outputs(rows)
    if baseline:
        print(f"correctness baseline: {baseline}")
        for name, info in diff.items():
            print(f"  {name}: {info['comparable']} comparable, "
                  f"{info['mismatch']} mismatched"
                  + (f" e.g. {info['examples']}" if info["examples"] else ""))
        print()

    failures = 0
    for label, ok, detail in gates(rows, args.tolerance):
        print(f"[{'PASS' if ok else 'FAIL'}] {label} — {detail}")
        if not ok:
            failures += 1
    print()
    print("ACCEPTED" if failures == 0 else f"{failures} gate(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
