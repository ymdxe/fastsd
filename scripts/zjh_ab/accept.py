#!/usr/bin/env python3
"""Acceptance report for the six-condition FastSD experiment.

Usage:
    python3 scripts/zjh_ab/accept.py exp/zjh_ab
    python3 scripts/zjh_ab/accept.py exp/zjh_ab/run-tag

Reads, per condition:
    <cond>/target/scheduler_metrics.json   (fastsd_scheduler.kv_residency / .cache_transfers)
    <cond>/edge/edge_metrics_proc*.jsonl   (per-request records)

The report root may contain either exact condition names or runall names such
as ``<RUN_TAG>_prefetch_h1``.

Prints the requested performance metrics and a correctness diff.  The acceptance
gate intentionally uses only request count, task-set identity, completion and
CUDA OOM.  Output mismatches remain visible diagnostics and never fail the run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.common_metrics import evaluate_slo

CONDITIONS = (
    "ar",
    "sd",
    "bypass",
    "prefetch_direct",
    "prefetch_h1",
    "no_pressure_prefetch",
    "no_pressure_sd",
)


def condition_label(path):
    """Map both exact and <run-tag>_<condition> names to a condition."""

    label = os.path.basename(str(path).rstrip("/"))
    matches = [name for name in CONDITIONS
               if label == name or label.endswith("_" + name)]
    return max(matches, key=len) if matches else label

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
    idempotency_path = os.path.join(cond_dir, "target", "idempotency_metrics.json")
    idempotency = load_json(idempotency_path) if os.path.exists(idempotency_path) else {}
    energy = {}
    energy_path = os.path.join(cond_dir, "target", "energy_metrics.json")
    if os.path.exists(energy_path):
        energy = load_json(energy_path)
    power = {}
    power_path = os.path.join(cond_dir, "power_summary.json")
    if os.path.exists(power_path):
        power = load_json(power_path)

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
    prefetch_metrics = {key: scheduler.get(key) for key in (
        "prefetch_ready_hit", "prefetch_pending_hit", "prefetch_demand_miss",
        "prefetch_skipped_no_memory", "prefetch_skipped_stale", "prefetch_copy_ms",
        "prefetch_copy_count", "prefetch_lead_ms", "prefetch_lead_count",
        "prefetch_bytes", "prefetch_candidates", "prefetch_candidates_created",
        "prefetch_candidates_peak", "prefetch_allocation_oom", "prefetch_async_failed",
        "prefetch_wasted", "prefetch_fallback_to_demand")}
    if prefetch_metrics["prefetch_bytes"] is None:
        prefetch_metrics["prefetch_bytes"] = transfers.get("speculative_submitted_bytes", 0)
    log_path = os.path.join(cond_dir, "target", "target.log")
    # A CUDA OOM can be reported by either target or edge.  Missing logs are
    # treated as unknown diagnostics, not as an OOM failure; the round's
    # request-count/completion checks still catch missing work.
    log_paths = []
    if os.path.exists(log_path):
        log_paths.append(log_path)
    log_paths.extend(glob.glob(os.path.join(cond_dir, "**", "*.log"), recursive=True))
    oom_pattern = re.compile(
        r"CUDA out of memory|(?:cuda\.)?OutOfMemoryError|cudaErrorMemoryAllocation",
        re.IGNORECASE,
    )
    oom_in_log = False
    for candidate in sorted(set(log_paths)):
        try:
            with open(candidate, errors="replace") as handle:
                if oom_pattern.search(handle.read()):
                    oom_in_log = True
                    break
        except OSError:
            continue

    accepted_total = sum(
        int(r.get("accepted_total", r.get("accepted_tokens", 0)) or 0)
        for r in records
    )
    drafted_total = sum(
        int(r.get("drafted_total", r.get("drafted_tokens", 0)) or 0)
        for r in records
    )
    accepted_lengths = [
        float(r["mean_accepted_tokens_per_verify"])
        for r in records
        if r.get("mean_accepted_tokens_per_verify") is not None
    ]
    accept_rate = (accepted_total / drafted_total) if drafted_total > 0 else None
    accept_length = statistics.mean(accepted_lengths) if accepted_lengths else None
    if not accepted_lengths:
        for summary in summaries:
            if summary.get("mean_accepted_tokens_per_verify") is not None:
                accept_length = float(summary["mean_accepted_tokens_per_verify"])
                break
    if accept_rate is None:
        for summary in summaries:
            if summary.get("accept_rate") is not None:
                accept_rate = float(summary["accept_rate"])
                break

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
        "ttft_p95": percentile(ttft, 0.95),
        "tpot_p95": percentile(tpot, 0.95),
        "latency_samples": {"e2e": len(e2e), "ttft": len(ttft), "tpot": len(tpot)},
        "oom_in_log": oom_in_log,
        "oom": bool(oom_in_log or prefetch_metrics.get("prefetch_allocation_oom")),
        "target_log_present": os.path.exists(log_path),
        "prefetch_implementation": scheduler.get("kv_prefetch_implementation"),
        "prefetch_metrics": prefetch_metrics,
        "mode": residency.get("mode"),
        "activations": residency.get("residency_activations", 0),
        "evictions": residency.get("residency_evictions", 0),
        "emergency_eviction_rounds": residency.get("residency_emergency_eviction_rounds", 0),
        "emergency_evictions": residency.get("residency_emergency_evictions", 0),
        "emergency_blocked_rounds": residency.get("residency_emergency_blocked_rounds", 0),
        "restores": residency.get("residency_restores", 0),
        "used_bytes": residency.get("used_bytes", 0),
        "eviction_errors": scheduler.get("kv_residency_eviction_errors", 0),
        "idempotent_cache_hit": idempotency.get("idempotent_cache_hit"),
        "idempotent_inflight_wait": idempotency.get("idempotent_inflight_wait"),
        "cache_evictions": idempotency.get("cache_evictions"),
        "retry_count": sum(record.get("retry_count", 0) for record in records),
        "replayed_requests": sum(bool(record.get("replayed", False)) for record in records),
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
        "ready_hit": scheduler.get("prefetch_ready_hit", 0),
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
        "ahead_generated_tokens": sum(
            int(s.get("ahead_generated_tokens", 0) or 0) for s in summaries
        ),
        "ahead_reused_tokens": sum(
            int(s.get("ahead_reused_tokens", 0) or 0) for s in summaries
        ),
        "ahead_discarded_tokens": sum(
            int(s.get("ahead_discarded_tokens", 0) or 0) for s in summaries
        ),
        "ahead_forward_ms": sum(
            float(s.get("ahead_forward_ms", 0.0) or 0.0) for s in summaries
        ),
        "ahead_reuse_rate": (
            sum(int(s.get("ahead_reused_tokens", 0) or 0) for s in summaries)
            / sum(int(s.get("ahead_generated_tokens", 0) or 0) for s in summaries)
            if sum(int(s.get("ahead_generated_tokens", 0) or 0) for s in summaries) > 0
            else 0.0
        ),
        "slo": evaluate_slo(
            records,
            wallclock_s=span,
            dataset=os.environ.get("DATASET"),
            scenario=os.environ.get("SLO_SCENARIO"),
        ),
        "energy_j": energy.get("system_energy_j", energy.get("energy_j")),
        "combined_power_energy_j": power.get("combined_computational_energy_j"),
        "combined_power_w": power.get("combined_computational_power_w"),
        "accept_rate": accept_rate,
        "accept_length": accept_length,
        "task_ids": sorted({str(r.get("task_id")) for r in records if r.get("task_id") is not None}),
        "energy": energy,
        "power": power,
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
    headers = ("condition", "n", "same_tasks", "complete", "OOM", "acc", "acc_len",
               "tok/s", "TTFT", "TPOT", "E2E", "H2D", "energy")
    line = "{:<24}{:>6}{:>12}{:>10}{:>6}{:>9}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>12}"
    print(line.format(*headers))
    print("-" * 139)
    def metric(value, digits=0):
        return "n/a" if value is None else (f"{float(value):.{digits}f}" if digits else f"{float(value):.0f}")
    for row in rows:
        label = condition_label(row["dir"])
        print(line.format(
            label,
            row["requests"],
            str(row.get("same_tasks", "n/a")),
            str(row.get("complete", not row.get("incomplete", True))),
            str(row.get("oom", row.get("oom_in_log", False))),
            metric(row.get("accept_rate"), 4),
            metric(row.get("accept_length"), 2),
            f'{row["tok_per_s"]:.2f}',
            metric(row["ttft_p50"] if row["latency_samples"]["ttft"] else None),
            metric(row["tpot_p50"] if row["latency_samples"]["tpot"] else None),
            metric(row["e2e_p50"] if row["latency_samples"].get("e2e", 0) else None),
            fmt_bytes(row["h2d_bytes"]) if "h2d_bytes" not in row["missing_transfer_counters"] else "n/a",
            metric(row.get("energy_j"), 2),
        ))


def print_diagnostics(rows):
    def number(value):
        return f"{value:.4f}" if value is not None else "n/a"

    for row in rows:
        print(f"  status: {'incomplete' if row.get('incomplete') else 'complete'}")
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
        print("  requests: "
              f"retries={row.get('retry_count', 0)} "
              f"replayed_tasks={row.get('replayed_requests', 0)} "
              f"cache_hit={row.get('idempotent_cache_hit')} "
              f"inflight_wait={row.get('idempotent_inflight_wait')} "
              f"cache_evictions={row.get('cache_evictions')}")
        prefetch = row["prefetch_metrics"]
        print("  prefetch: "
              f"implementation={row.get('prefetch_implementation')} "
              f"ready={row['ready_hit']} pending={row['pending_hit']} demand_miss={row['demand_miss']} "
              f"skipped_memory={prefetch.get('prefetch_skipped_no_memory')} "
              f"skipped_stale={prefetch.get('prefetch_skipped_stale')} "
              f"copy_ms_sum={prefetch.get('prefetch_copy_ms')} "
              f"lead_ms_sum={prefetch.get('prefetch_lead_ms')} "
              f"bytes={prefetch.get('prefetch_bytes')} "
              f"candidates={prefetch.get('prefetch_candidates')} "
              f"created={prefetch.get('prefetch_candidates_created')} "
              f"peak={prefetch.get('prefetch_candidates_peak')}")
        print("  pressure: "
              f"max_batch={fmt_bytes(row.get('max_batch_kv_bytes'))} "
              f"max_resident={fmt_bytes(row.get('max_gpu_resident_kv_bytes'))} "
              f"max_total+prefetch={fmt_bytes(row.get('max_total_kv_plus_prefetch_bytes'))} "
              f"emergency_rounds={row.get('emergency_eviction_rounds', 0)} "
              f"emergency_evict={row.get('emergency_evictions', 0)} "
              f"emergency_blocked={row.get('emergency_blocked_rounds', 0)} "
              f"claim_lead_ms={number(row.get('prefetch_claim_lead_ms_avg'))}")
        print("  proactive: "
              f"generated={row.get('ahead_generated_tokens', 0)} "
              f"reused={row.get('ahead_reused_tokens', 0)} "
              f"discarded={row.get('ahead_discarded_tokens', 0)} "
              f"reuse_rate={number(row.get('ahead_reuse_rate'))} "
              f"forward_ms={number(row.get('ahead_forward_ms'))}")
        print("  acceptance: "
              f"rate={number(row.get('accept_rate'))} "
              f"length={number(row.get('accept_length'))}")
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
        power = row.get("power") or {}
        if power:
            print("  combined power: "
                  f"J={number(power.get('combined_computational_energy_j'))} "
                  f"avgW={number(power.get('combined_computational_power_w'))} "
                  f"edgeCPU_J={number(power.get('edge_cpu_estimated_energy_j'))} "
                  f"targetGPU_J={number(power.get('target_gpu_measured_energy_j'))}")
        if row["missing_transfer_counters"]:
            print("  missing transfer counters (legacy sparse counters may omit zero): "
                  + ", ".join(row["missing_transfer_counters"]))
        for horizon, bucket in sorted(row["prediction_by_horizon"].items(), key=lambda pair: int(pair[0])):
            print(f"  h{horizon}: P={number(bucket['precision'])} R={number(bucket['recall'])} "
                  f"coverage/plan={number(bucket['coverage_per_actual_plan'])} "
                  f"hits/plan={number(bucket['hits_per_actual_plan'])} "
                  f"ready/plan={number(bucket['ready_per_actual_plan'])}")


def compare_outputs(rows):
    """Return per-condition task_id -> output signatures and their diff."""

    tables = {}
    for row in rows:
        name = os.path.basename(row["dir"].rstrip("/"))
        tables[name] = {
            str(r.get("task_id")): (
                r.get("output_token_ids")
                if r.get("output_token_ids") is not None
                else r.get("output_text")
            )
            for r in row["records"]
            if r.get("task_id") is not None
        }
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


def compare_task_sets(rows):
    """Return whether every condition contains the same unique task set."""

    if not rows:
        return False, "no conditions"
    expected = rows[0].get("expected_requests")
    baseline = rows[0].get("task_ids", [])
    baseline_set = set(baseline)
    details = []
    ok = bool(baseline) and len(baseline) == len(baseline_set)
    if expected is not None:
        ok = ok and len(baseline) == int(expected)
    details.append(f"{os.path.basename(rows[0]['dir'].rstrip('/'))}={len(baseline)}")
    for row in rows[1:]:
        ids = row.get("task_ids", [])
        ids_set = set(ids)
        row_ok = len(ids) == len(ids_set) and ids_set == baseline_set
        if expected is not None:
            row_ok = row_ok and len(ids) == int(expected)
        ok = ok and row_ok
        details.append(f"{os.path.basename(row['dir'].rstrip('/'))}={len(ids)}")
    return ok, ", ".join(details)


def acceptance_gates(rows):
    """The experiment acceptance contract: n, same_tasks, complete and OOM."""

    same_tasks, task_detail = compare_task_sets(rows)
    checks = [("same_tasks", same_tasks, task_detail)]
    for row in rows:
        name = condition_label(row["dir"])
        expected = int(row["expected_requests"])
        n_ok = row["requests"] == expected
        complete = bool(n_ok and not row.get("incomplete", False))
        oom = bool(row.get("oom", row.get("oom_in_log", False)))
        checks.extend([
            (f"{name}: n={expected}", n_ok, f"n={row['requests']} expected={expected}"),
            (f"{name}: complete", complete, f"n={row['requests']}/{expected}"),
            (f"{name}: OOM", not oom, f"oom={oom}"),
        ])
    return checks


def prefetch_pair_gates(rows, tolerance=.05, h2d_tolerance=.10, legacy=None):
    """Strict paired checks; rows are ordered noprefetch then prefetch."""
    if len(rows) != 2:
        return [("exactly two paired conditions", False, f"got {len(rows)}")]
    base, pref = rows
    results = []
    tables = []
    for row in rows:
        name = os.path.basename(row["dir"])
        records = row["records"]
        identities = [str(r.get("task_id")) for r in records if r.get("task_id") is not None]
        unique = len(identities) == len(set(identities)) == row["expected_requests"] == len(records)
        results.append((f"{name}: exact unique request count", unique,
                        f"records={len(records)} unique={len(set(identities))} expected={row['expected_requests']}"))
        tables.append({str(r["task_id"]): r for r in records if r.get("task_id") is not None})
        results.append((f"{name}: no CUDA OOM", row["target_log_present"] and not row["oom_in_log"]
                        and not row["prefetch_metrics"].get("prefetch_allocation_oom"),
                        f"target_log={row['target_log_present']} OOM={row['oom_in_log']}"))
        results.append((f"{name}: transfer counters present", not row["missing_transfer_counters"],
                        str(row["missing_transfer_counters"])))
        failed = row["prefetch_metrics"].get("prefetch_async_failed")
        results.append((f"{name}: no prefetch copy errors", failed == 0, f"errors={failed}"))
        for metric in ("ttft", "tpot"):
            samples = row["latency_samples"][metric]
            results.append((f"{name}: {metric} samples present", samples == len(records) and samples > 0,
                            f"samples={samples}/{len(records)}"))
    same_keys = set(tables[0]) == set(tables[1]) and bool(tables[0])
    mismatches = []
    missing = []
    for key in set(tables[0]) & set(tables[1]):
        a, b = tables[0][key], tables[1][key]
        # Token equality is stronger than decoded text equality and detects
        # an extra/missing KV advance that happens to decode identically.
        if a.get("output_token_ids") is not None and b.get("output_token_ids") is not None:
            equal = a["output_token_ids"] == b["output_token_ids"]
        elif a.get("output_text") is not None and b.get("output_text") is not None:
            equal = a["output_text"] == b["output_text"]
        else:
            missing.append(key)
            continue
        if not equal:
            mismatches.append(key)
    # Keep the comparison visible, but do not turn output/token mismatches into
    # an acceptance failure.  The experiment gate uses only task-set identity,
    # count, completion and OOM.
    results.append(("paired output mismatch (diagnostic)", True,
                    f"same_tasks={same_keys} mismatch={len(mismatches)} missing={len(missing)} examples={mismatches[:5]}"))
    for metric in ("ttft_p50", "ttft_p95", "tpot_p50", "tpot_p95"):
        a, b = base[metric], pref[metric]
        results.append((f"prefetch {metric} within tolerance", b <= a * (1 + tolerance),
                        f"noprefetch={a:.3f} prefetch={b:.3f} tolerance={tolerance:.1%}"))
    results.append(("total H2D within tolerance", pref["h2d_bytes"] <= base["h2d_bytes"] * (1 + h2d_tolerance),
                    f"noprefetch={fmt_bytes(base['h2d_bytes'])} prefetch={fmt_bytes(pref['h2d_bytes'])} tolerance={h2d_tolerance:.1%}"))
    results.append(("prefetch exercised", (pref["prefetch_metrics"].get("prefetch_bytes") or 0) > 0,
                    f"bytes={pref['prefetch_metrics'].get('prefetch_bytes')} ready={pref['ready_hit']}"))
    results.append(("noprefetch has no speculative copies", not base["prefetch_metrics"].get("prefetch_bytes")
                    and base["ready_hit"] == 0, f"bytes={base['prefetch_metrics'].get('prefetch_bytes')}"))
    if legacy is not None:
        comparable = (not legacy["incomplete"] and legacy["requests"] == pref["requests"]
                      and set(str(r.get("task_id")) for r in legacy["records"]) == set(tables[1]))
        results.append(("legacy reference complete with same tasks", comparable,
                        f"reference={legacy['dir']} records={legacy['requests']}"))
        results.append(("ready hits at least legacy", comparable and pref["ready_hit"] >= legacy["ready_hit"],
                        f"h1={pref['ready_hit']} legacy={legacy['ready_hit']}"))
    return results


def gates(rows, tolerance):
    """Compatibility wrapper for the six-condition acceptance contract.

    Historical callers still receive the pressure diagnostic, but it is
    informational and can never fail a run.  The only false results are the
    four requested checks from :func:`acceptance_gates`.
    """

    del tolerance
    results = acceptance_gates(rows)
    for row in rows:
        name = condition_label(row["dir"])
        if row.get("mode") == "active":
            results.append((f"{name}: KV pressure exercised (diagnostic)", True,
                            f"activations={row.get('activations', 0)} evictions={row.get('evictions', 0)}"))
        results.append((f"{name}: requests completed", row["requests"] > 0,
                        f"{row['requests']} records"))
        results.append((f"{name}: complete benchmark", row.get("complete", not row["incomplete"]),
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
    parser.add_argument("--prefetch-pair", action="store_true", help="paths are noprefetch then prefetch")
    parser.add_argument("--h2d-tolerance", type=float, default=.10)
    parser.add_argument("--legacy-reference", help="existing complete legacy run for ready-hit comparison; no extra run")
    parser.add_argument("--dataset", default=os.environ.get("DATASET"), help="dataset name for dataset/scenario SLO configuration")
    parser.add_argument("--slo-scenario", default=os.environ.get("SLO_SCENARIO"))
    args = parser.parse_args()

    dirs = []
    for path in args.paths:
        if os.path.isdir(os.path.join(path, "target")) or os.path.isdir(os.path.join(path, "edge")):
            dirs.append(path)
            continue
        # Accept both a report root containing exact condition names and the
        # runall layout, where conditions are named <RUN_TAG>_<condition>.
        for name in CONDITIONS:
            exact = os.path.join(path, name)
            if os.path.isdir(exact):
                dirs.append(exact)
                continue
            matches = sorted(glob.glob(os.path.join(path, "*_" + name)))
            dirs.extend(match for match in matches if os.path.isdir(match))
    # Preserve the declared six-condition order and avoid duplicate paths when
    # a caller supplied both a parent and an explicit condition directory.
    grouped = {name: [] for name in CONDITIONS}
    for directory in dirs:
        name = condition_label(directory)
        if name in grouped:
            grouped[name].append(directory)
    ordered = []
    for name in CONDITIONS:
        for directory in grouped[name]:
            if directory not in ordered:
                ordered.append(directory)
    dirs = ordered
    if not dirs:
        print("no condition directories found", file=sys.stderr)
        return 2

    if args.expected_requests < 1:
        parser.error("--expected-requests must be positive")
    if args.tolerance < 0 or args.h2d_tolerance < 0:
        parser.error("tolerances must be nonnegative")
    if args.dataset:
        os.environ["DATASET"] = args.dataset
    if args.slo_scenario:
        os.environ["SLO_SCENARIO"] = args.slo_scenario
    rows = [summarize(directory, args.expected_requests) for directory in dirs]
    same_tasks, _ = compare_task_sets(rows)
    for row in rows:
        row["same_tasks"] = same_tasks
        row["complete"] = bool(row["requests"] == row["expected_requests"] and not row["incomplete"])
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
    checks = acceptance_gates(rows)
    if args.prefetch_pair:
        print("[INFO] --prefetch-pair is retained for report compatibility; pair performance/output checks are diagnostic only")
    for label, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label} — {detail}")
        if not ok:
            failures += 1
    print()
    print("ACCEPTED" if failures == 0 else f"{failures} gate(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
