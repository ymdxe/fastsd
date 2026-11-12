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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.common_metrics import evaluate_slo

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


def union_seconds(intervals):
    intervals = sorted(
        (float(a), float(b)) for a, b in intervals
        if a is not None and b is not None and float(b) >= float(a)
    )
    if not intervals:
        return 0.0
    total = 0.0
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    return total + cur_b - cur_a


def summarize(cond_dir, expected_requests=164):
    scheduler = load_scheduler(cond_dir)
    residency = scheduler.get("kv_residency") or {}
    transfers = scheduler.get("cache_transfers") or {}
    records = load_records(cond_dir)
    energy = {}
    energy_path = os.path.join(cond_dir, "target", "energy_metrics.json")
    if os.path.exists(energy_path):
        energy = load_json(energy_path)

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
    request_span = (max(ends) - min(starts)) if starts and ends else 0.0
    summaries = [load_json(path) for path in sorted(glob.glob(
        os.path.join(cond_dir, "edge", "edge_metrics_summary*.json")))]
    windows = [(s.get("benchmark_start_epoch_s"), s.get("benchmark_end_epoch_s"))
               for s in summaries]
    if windows and all(a is not None and b is not None and b >= a for a, b in windows):
        span = max(b for a, b in windows) - min(a for a, b in windows)
        span_basis = "benchmark_epoch_window"
    elif summaries and any(float(s.get("wallclock_s") or 0) > 0 for s in summaries):
        # Legacy summaries have durations, but no shared start/end epochs.
        # This fallback is explicitly labelled; exact multi-host benchmark
        # boundaries cannot be recovered from historical request records.
        span = max(request_span, max(float(s.get("wallclock_s") or 0) for s in summaries))
        span_basis = "legacy_wallclock_duration"
    else:
        span, span_basis = request_span, "legacy_request_span"
    tokens = sum(int(r.get("generated_tokens") or 0) for r in records)

    def pick(record, epoch_key, mono_key):
        value = record.get(epoch_key)
        return value if value is not None else record.get(mono_key)

    decode_intervals = [
        (pick(r, "first_token_epoch_s", "first_token_monotonic_s"),
         pick(r, "completion_epoch_s", "completion_monotonic_s"))
        for r in records
        if pick(r, "first_token_epoch_s", "first_token_monotonic_s") is not None
    ]
    request_intervals = [
        (pick(r, "processing_start_epoch_s", "processing_start_monotonic_s"),
         pick(r, "completion_epoch_s", "completion_monotonic_s"))
        for r in records
    ]
    decode_busy_s = union_seconds(decode_intervals)
    request_busy_s = union_seconds(request_intervals)
    throughput_span = decode_busy_s or request_busy_s or span
    e2e = [r.get("request_e2e_ms") or r.get("e2e_ms") or 0.0 for r in records]
    ttft = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    tpot = [r["tpot_ms"] for r in records if r.get("tpot_ms") is not None]
    actual_plans = scheduler.get("actual_plans", scheduler.get("iterations"))
    ready = scheduler.get("prefetch_ready_hit", 0) or 0

    def ratio(numerator, denominator):
        return float(numerator or 0) / denominator if denominator else None

    horizons = {}
    ready_horizons = scheduler.get("prefetch_ready_by_horizon") or {}
    for horizon, bucket in (scheduler.get("prefetch_prediction_by_horizon") or {}).items():
        horizons[horizon] = dict(bucket,
            precision=ratio(bucket.get("hits"), bucket.get("predicted")),
            recall=ratio(bucket.get("hits"), bucket.get("actual")),
            coverage_per_actual_plan=ratio(bucket.get("plans"), actual_plans),
            hits_per_actual_plan=ratio(bucket.get("hits"), actual_plans),
            ready_per_actual_plan=ratio((ready_horizons.get(horizon) or {}).get("ready"), actual_plans))
    transfer_s = (transfers.get("kv_transfer_prefetch_avg_s") or
                  transfers.get("kv_transfer_h2d_avg_s") or transfers.get("kv_transfer_avg_s"))
    batch_s = scheduler.get("batch_compute_avg_s")
    transfer_batch_ratio = ratio(transfer_s, batch_s) if transfer_s is not None else None

    return {
        "dir": cond_dir,
        "requests": len(records),
        "expected_requests": expected_requests,
        "incomplete": len(records) < expected_requests,
        "tokens": tokens,
        "span_s": span,
        "span_basis": span_basis,
        "request_span_s": request_span,
        "decode_busy_s": decode_busy_s,
        "request_busy_s": request_busy_s,
        # Paper-facing throughput: whole benchmark window, including
        # arrival gaps.  Busy-window values are diagnostics below.
        "tok_per_s": (tokens / span) if span > 0 else 0.0,
        "req_per_s": (len(records) / span) if span > 0 else 0.0,
        "busy_tok_per_s": (tokens / throughput_span) if throughput_span > 0 else 0.0,
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
        "missing_transfer_counters": [key for key in ("h2d_bytes", "d2h_bytes", "copies")
                                      if key not in transfers],
        "spec_bytes": transfers.get("speculative_submitted_bytes", 0.0),
        "lookahead_calls": scheduler.get("lookahead_ms_count", 0) or 0,
        "prefetch_refresh_count": scheduler.get("prefetch_refresh_count"),
        "lookahead_build_count": scheduler.get("lookahead_build_count", scheduler.get("lookahead_ms_count")),
        "prediction_snapshots_consumed": scheduler.get("prediction_snapshots_consumed"),
        "actual_plans": actual_plans,
        "ready_per_actual_plan": ratio(ready, actual_plans),
        "h2d_bytes_per_ready": ratio(transfers.get("h2d_bytes"), ready),
        "h2d_per_ready": ratio(transfers.get("h2d_bytes"), ready),
        "copies_per_ready": ratio(transfers.get("copies"), ready),
        "prediction_by_horizon": horizons,
        "prefetch_intent_by_horizon": scheduler.get("prefetch_intent_by_horizon") or {},
        "prefetch_candidate_by_horizon": scheduler.get("prefetch_candidate_by_horizon") or {},
        "prefetch_retired_by_horizon": scheduler.get("prefetch_retired_by_horizon") or {},
        "prefetch_skipped_horizon": scheduler.get("prefetch_skipped_horizon") or {},
        "prefetch_bytes_by_horizon": transfers.get("prefetch_bytes_by_horizon", scheduler.get("prefetch_bytes_by_horizon")) or {},
        "kv_transfer_by_horizon": transfers.get("kv_transfer_by_horizon") or {},
        "batch_compute_avg_s": batch_s,
        "kv_transfer_avg_s": transfer_s,
        "transfer_per_batch_window": transfer_batch_ratio,
        "transfer_exceeds_batch_window": transfer_batch_ratio > 1 if transfer_batch_ratio is not None else None,
        "ready_hit": ready,
        "pending_hit": scheduler.get("prefetch_pending_hit", 0),
        "demand_miss": scheduler.get("prefetch_demand_miss", 0),
        "prefetch_verify_ready_hit": scheduler.get("prefetch_verify_ready_hit", 0),
        "prefetch_verify_pending_hit": scheduler.get("prefetch_verify_pending_hit", 0),
        "prefetch_verify_demand_miss": scheduler.get("prefetch_verify_demand_miss", 0),
        "max_batch_kv_bytes": scheduler.get("max_batch_kv_bytes"),
        "max_gpu_resident_kv_bytes": scheduler.get("max_gpu_resident_kv_bytes"),
        "max_total_kv_plus_prefetch_bytes": scheduler.get("max_total_kv_plus_prefetch_bytes"),
        "max_prefetch_reserved_bytes": scheduler.get("max_prefetch_reserved_bytes"),
        "residency_used_bytes_max": scheduler.get("residency_used_bytes_max"),
        "residency_used_bytes_avg": scheduler.get("residency_used_bytes_avg"),
        "prefetch_claim_lead_ms_avg": scheduler.get("prefetch_claim_lead_ms_avg"),
        "prefetch_claim_lead_ms_max": scheduler.get("prefetch_claim_lead_ms_max"),
        "slo": evaluate_slo(records, wallclock_s=span),
        "energy": energy,
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
    headers = ("condition", "n", "tok/s", "req/s", "E2E p50", "TTFT p50", "TPOT p50",
               "KV mode", "evict", "H2D", "D2H", "copies", "builds", "ready", "status")
    line = "{:<34}{:>4}{:>8}{:>7}{:>10}{:>10}{:>9}{:>9}{:>7}{:>9}{:>9}{:>8}{:>10}{:>7}{:>12}"
    print(line.format(*headers))
    print("-" * 128)
    for row in rows:
        print(line.format(
            os.path.basename(row["dir"].rstrip("/")),
            row["requests"],
            f'{row["tok_per_s"]:.2f}',
            f'{row["req_per_s"]:.2f}',
            f'{row["e2e_p50"]:.0f}',
            f'{row["ttft_p50"]:.0f}',
            f'{row["tpot_p50"]:.0f}',
            str(row["mode"]),
            row["evictions"],
            fmt_bytes(row["h2d_bytes"]) if "h2d_bytes" not in row["missing_transfer_counters"] else "n/a",
            fmt_bytes(row["d2h_bytes"]) if "d2h_bytes" not in row["missing_transfer_counters"] else "n/a",
            f'{float(row["copies"]):.0f}' if "copies" not in row["missing_transfer_counters"] else "n/a",
            str(row["lookahead_build_count"]) if row["lookahead_build_count"] is not None else "n/a",
            row["ready_hit"],
            "incomplete" if row["incomplete"] else "complete",
        ))


def print_diagnostics(rows):
    def number(value):
        return f"{value:.4f}" if value is not None else "n/a"

    for row in rows:
        print(f"{os.path.basename(row['dir'].rstrip('/'))}: "
              f"refresh={row['prefetch_refresh_count']} builds={row['lookahead_build_count']} "
              f"snapshots={row['prediction_snapshots_consumed']} actual_plans={row['actual_plans']} "
              f"ready/plan={number(row['ready_per_actual_plan'])} "
              f"H2D/ready={fmt_bytes(row['h2d_bytes_per_ready']) if row['h2d_bytes_per_ready'] is not None else 'n/a'} "
              f"copies/ready={number(row['copies_per_ready'])} "
              f"transfer/batch={number(row['transfer_per_batch_window'])} "
              f"window={row['span_basis']}")
        standard_slo = (row.get("slo", {}).get("combined", {}).get("standard") or {})
        energy = row.get("energy") or {}
        print("  pressure: "
              f"max_batch={fmt_bytes(row.get('max_batch_kv_bytes'))} "
              f"max_resident={fmt_bytes(row.get('max_gpu_resident_kv_bytes'))} "
              f"max_total+prefetch={fmt_bytes(row.get('max_total_kv_plus_prefetch_bytes'))} "
              f"claim_lead_ms={number(row.get('prefetch_claim_lead_ms_avg'))}")
        print("  SLO standard: "
              f"combined_violation={number(standard_slo.get('violation_rate'))} "
              f"goodput_tok_s={number(standard_slo.get('goodput_tok_s'))} "
              f"TTFT_violation={number((row.get('slo', {}).get('by_metric', {}).get('ttft_ms', {}).get('standard') or {}).get('violation_rate'))} "
              f"TPOT_violation={number((row.get('slo', {}).get('by_metric', {}).get('tpot_ms', {}).get('standard') or {}).get('violation_rate'))} "
              f"E2E_violation={number((row.get('slo', {}).get('by_metric', {}).get('e2e_ms', {}).get('standard') or {}).get('violation_rate'))}")
        if energy:
            print("  energy: "
                  f"J={number(energy.get('system_energy_j'))} "
                  f"avgW={number(energy.get('system_avg_power_w'))} "
                  f"peakW={number(energy.get('system_peak_power_w'))} "
                  f"p95W={number(energy.get('system_power_p95_w'))}")
        if row["missing_transfer_counters"]:
            print("  missing transfer counters (legacy sparse counters may omit zero): "
                  + ", ".join(row["missing_transfer_counters"]))
        for horizon, bucket in sorted(row["prediction_by_horizon"].items(), key=lambda pair: int(pair[0])):
            print(f"  h{horizon}: P={number(bucket['precision'])} R={number(bucket['recall'])} "
                  f"coverage/plan={number(bucket['coverage_per_actual_plan'])} "
                  f"hits/plan={number(bucket['hits_per_actual_plan'])} "
                  f"ready/plan={number(bucket['ready_per_actual_plan'])}")


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
            # ACTIVE means management is enabled, not that pressure occurred.
            # Vanilla can evict without the FastSD lazy-start activation hook.
            ok = row["evictions"] > 0
            results.append((f"{name}: KV pressure exercised (experiment coverage)", ok,
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
        results.append((f"{name}: complete benchmark", not row["incomplete"],
                        f"{row['requests']}/{row['expected_requests']} records"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="condition dirs or a parent dir")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="allowed FastSD slowdown vs sd-cpu (default 5%%)")
    parser.add_argument("--expected-requests", type=int, default=164,
                        help="required completed records per condition (default 164)")
    parser.add_argument("--json-output", help="write normalized metrics as JSON")
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

    if args.expected_requests < 1:
        parser.error("--expected-requests must be positive")
    rows = [summarize(directory, args.expected_requests) for directory in dirs]
    print_table(rows)
    print()
    print_diagnostics(rows)
    print()
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump([{k: v for k, v in row.items() if k != "records"} for row in rows], handle, indent=2)

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
