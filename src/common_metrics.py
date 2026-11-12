"""Common request-level metric schema for FastSD comparison methods."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1

# Default SLO tiers are deliberately explicit and overrideable.  Each metric
# uses milliseconds, and a request passes a tier only when its observed value
# is at or below that tier's threshold.
DEFAULT_SLO_THRESHOLDS: dict[str, dict[str, float]] = {
    "ttft_ms": {
        "strict": 250.0,
        "standard": 500.0,
        "relaxed": 1000.0,
        "tail": 2000.0,
    },
    "tpot_ms": {
        "strict": 50.0,
        "standard": 100.0,
        "relaxed": 200.0,
        "tail": 500.0,
    },
    "e2e_ms": {
        "strict": 1000.0,
        "standard": 2000.0,
        "relaxed": 5000.0,
        "tail": 10000.0,
    },
}


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _metric_stats(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = _numeric(records, key)
    return {
        "avg": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def evaluate_slo(
    records: Iterable[dict[str, Any]],
    *,
    wallclock_s: float | None,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate multi-tier latency SLOs and retain request-level evidence.

    Violation rate is computed over all completed records. Failed requests and
    missing metrics are retained as violations so incomplete telemetry cannot
    silently improve an SLO result. Goodput counts successful requests/tokens that satisfy the
    selected metric threshold and is normalized by the measured run wallclock.
    ``combined`` additionally requires TTFT, TPOT, and E2E to pass together.
    """

    rows = [dict(record) for record in records]
    configured = thresholds or DEFAULT_SLO_THRESHOLDS
    normalized: dict[str, dict[str, float]] = {}
    for metric, tiers in configured.items():
        normalized[metric] = {
            str(tier): float(limit) for tier, limit in tiers.items()
        }
    duration = float(wallclock_s or 0.0)

    def successful(record: dict[str, Any]) -> bool:
        return bool(record.get("success", record.get("status", "ok") not in {"error", "failed"}))

    def value_for(record: dict[str, Any], metric: str) -> float | None:
        value = record.get(metric)
        if value is None and metric == "e2e_ms":
            value = record.get("task_e2e_ms", record.get("request_e2e_ms"))
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    def tier_result(metric: str, tier: str, limit: float) -> dict[str, Any]:
        details = []
        passing_records = []
        violations = 0
        for index, record in enumerate(rows):
            observed = value_for(record, metric)
            passed = observed is not None and observed <= limit and successful(record)
            violation = not passed
            violations += int(violation)
            detail = {
                "index": index,
                "sample_id": record.get("sample_id", record.get("task_id", index)),
                "observed_ms": observed,
                "threshold_ms": limit,
                "violation": violation,
                "success": successful(record),
                "generated_tokens": int(record.get("generated_tokens", 0) or 0),
            }
            details.append(detail)
            if passed:
                passing_records.append(record)
        count = len(rows)
        passing_tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in passing_records)
        return {
            "metric": metric,
            "tier": tier,
            "threshold_ms": limit,
            "sample_count": count,
            "violation_count": violations,
            "violation_rate": violations / count if count else None,
            "passing_request_count": len(passing_records),
            "goodput_req_s": len(passing_records) / duration if duration > 0 else None,
            "goodput_tok_s": passing_tokens / duration if duration > 0 else None,
            "details": details,
        }

    by_metric: dict[str, dict[str, dict[str, Any]]] = {}
    for metric, tiers in normalized.items():
        by_metric[metric] = {
            tier: tier_result(metric, tier, limit) for tier, limit in tiers.items()
        }

    combined: dict[str, dict[str, Any]] = {}
    tier_names = sorted({tier for tiers in normalized.values() for tier in tiers})
    for tier in tier_names:
        limits = {metric: normalized[metric].get(tier) for metric in normalized}
        details = []
        passing_records = []
        violations = 0
        for index, record in enumerate(rows):
            observed = {metric: value_for(record, metric) for metric in normalized}
            passed = successful(record) and all(
                observed[metric] is not None and limits[metric] is not None
                and observed[metric] <= limits[metric]
                for metric in normalized
            )
            violation = not passed
            violations += int(violation)
            details.append({
                "index": index,
                "sample_id": record.get("sample_id", record.get("task_id", index)),
                "observed_ms": observed,
                "threshold_ms": limits,
                "violation": violation,
                "success": successful(record),
                "generated_tokens": int(record.get("generated_tokens", 0) or 0),
            })
            if passed and successful(record):
                passing_records.append(record)
        passing_tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in passing_records)
        count = len(rows)
        combined[tier] = {
            "tier": tier,
            "threshold_ms": limits,
            "sample_count": count,
            "violation_count": violations,
            "violation_rate": violations / count if count else None,
            "passing_request_count": len(passing_records),
            "goodput_req_s": len(passing_records) / duration if duration > 0 else None,
            "goodput_tok_s": passing_tokens / duration if duration > 0 else None,
            "details": details,
        }
    return {
        "thresholds_ms": normalized,
        "wallclock_s": duration,
        "by_metric": by_metric,
        "combined": combined,
        "definitions": {
            "violation_rate": "failed requests or requests exceeding the threshold/missing the metric divided by all completed requests",
            "goodput_req_s": "successful requests satisfying the selected SLO divided by measured run wallclock",
            "goodput_tok_s": "generated tokens from successful requests satisfying the selected SLO divided by measured run wallclock",
            "combined": "a request passes only when TTFT, TPOT, and E2E all pass the same tier",
        },
    }


def paired_bootstrap_ci(
    records: Iterable[dict[str, Any]],
    baseline_records: Iterable[dict[str, Any]],
    *,
    key: str,
    sample_key: str = "sample_id",
    seed: int = 42,
    iterations: int = 2000,
) -> dict[str, Any]:
    """Return a deterministic paired bootstrap CI for ``method - baseline``.

    Pairing is by ``sample_id`` rather than file order so a worker completion
    order cannot silently change the comparison.  This is intentionally a
    dependency-free implementation suitable for the CPU/static test runtime.
    """

    left = {str(item.get(sample_key)): item for item in records}
    right = {str(item.get(sample_key)): item for item in baseline_records}
    pairs = [
        (float(left[sample_id][key]), float(right[sample_id][key]))
        for sample_id in sorted(left.keys() & right.keys())
        if left[sample_id].get(key) is not None and right[sample_id].get(key) is not None
    ]
    if not pairs:
        return {
            "n": 0,
            "mean_delta": None,
            "ci95": None,
            "seed": int(seed),
            "iterations": int(iterations),
            "definition": f"{key} method-baseline",
        }
    deltas = [left_value - right_value for left_value, right_value in pairs]
    mean_delta = statistics.mean(deltas)
    if len(deltas) == 1:
        interval = [deltas[0], deltas[0]]
    else:
        rng = random.Random(seed)
        means = []
        for _ in range(max(1, int(iterations))):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            means.append(statistics.mean(sample))
        interval = [percentile(means, 0.025), percentile(means, 0.975)]
    return {
        "n": len(deltas),
        "mean_delta": mean_delta,
        "ci95": interval,
        "seed": int(seed),
        "iterations": int(iterations),
        "definition": f"{key} method-baseline, paired by {sample_key}",
    }


def _with_derived_throughput(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy records and expose per-request output throughput for paired analysis."""

    result: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if item.get("throughput_tok_s") is None:
            e2e_ms = item.get("e2e_ms")
            generated = item.get("generated_tokens")
            if e2e_ms is not None and generated is not None and float(e2e_ms) > 0:
                item["throughput_tok_s"] = float(generated) / (float(e2e_ms) / 1000.0)
        result.append(item)
    return result


def paired_method_analysis(
    method_records: Iterable[dict[str, Any]],
    baseline_records: Iterable[dict[str, Any]],
    *,
    method: str = "fastsd",
    baseline: str = "specedge_cpu_adapted",
    sample_key: str = "sample_id",
    seed: int = 42,
    iterations: int = 10_000,
) -> dict[str, Any]:
    """Compare two methods directly with deterministic sample-id pairing.

    Latency metrics use ``baseline - method`` as improvement, while throughput
    uses ``method - baseline``.  The bootstrap is performed on paired deltas;
    the reported percentage CI scales that delta interval by the baseline
    mean, which keeps the output dependency-free and auditable.
    """

    left = _with_derived_throughput(method_records)
    right = _with_derived_throughput(baseline_records)
    metric_directions = {
        "ttft_ms": "lower_is_better",
        "e2e_ms": "lower_is_better",
        "tpot_ms": "lower_is_better",
        "throughput_tok_s": "higher_is_better",
    }
    metrics: dict[str, Any] = {}
    for key, direction in metric_directions.items():
        ci = paired_bootstrap_ci(
            left,
            right,
            key=key,
            sample_key=sample_key,
            seed=seed,
            iterations=iterations,
        )
        left_by_id = {str(item.get(sample_key)): item for item in left}
        baseline_values = [
            float(item[key])
            for item in right
            if item.get(sample_key) is not None
            and item.get(key) is not None
            and left_by_id.get(str(item.get(sample_key)), {}).get(key) is not None
        ]
        baseline_mean = statistics.mean(baseline_values) if baseline_values else None
        sign = -1.0 if direction == "lower_is_better" else 1.0
        if baseline_mean and ci["mean_delta"] is not None:
            improvement_pct = sign * float(ci["mean_delta"]) / baseline_mean * 100.0
            ci_pct = [
                sign * float(ci["ci95"][0]) / baseline_mean * 100.0,
                sign * float(ci["ci95"][1]) / baseline_mean * 100.0,
            ]
            ci_pct.sort()
        else:
            improvement_pct = None
            ci_pct = None
        metrics[key] = {
            **ci,
            "baseline_mean": baseline_mean,
            "improvement_pct": improvement_pct,
            "improvement_ci95_pct": ci_pct,
            "direction": direction,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "baseline": baseline,
        "paired_by": sample_key,
        "bootstrap_seed": int(seed),
        "bootstrap_iterations": int(iterations),
        "metrics": metrics,
    }


def _resource_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "network_rtt_ms",
        "network_bytes",
        "request_bytes",
        "response_bytes",
        "rpc_count",
        "kv_copy_ms",
        "kv_copy_bytes",
        "cpu_time_ms",
    )
    result: dict[str, Any] = {}
    for key in keys:
        values = _numeric(records, key)
        if values:
            result[key] = {
                "total": sum(values),
                "avg": statistics.mean(values),
                "p95": percentile(values, 0.95),
            }
    return result


def _last_number(value: Any) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return matches[-1] if matches else None


def exact_match_quality(records: list[dict[str, Any]], dataset: str) -> float | None:
    if dataset not in {"gsm8k", "mgsm"}:
        return None
    scored = 0
    correct = 0
    for record in records:
        if record.get("reference") is None:
            continue
        predicted = _last_number(record.get("output_text"))
        expected = _last_number(record["reference"])
        if expected is None:
            continue
        scored += 1
        correct += int(predicted == expected)
    return correct / scored if scored else None


def summarize_requests(
    records: Iterable[dict[str, Any]],
    *,
    method: str,
    dataset: str,
    workload_hash: str,
    run_id: str,
    wallclock_s: float | None = None,
    evaluation_scope: str = "quality",
    paired_records: Iterable[dict[str, Any]] | None = None,
    slo_thresholds: dict[str, dict[str, float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_records = list(records)
    scheduled_ttft_records = []
    for record in request_records:
        scheduled_ttft = record.get("scheduled_ttft_ms")
        if (
            scheduled_ttft is None
            and record.get("ttft_ms") is not None
            and record.get("arrival_lag_ms") is not None
        ):
            scheduled_ttft = float(record["ttft_ms"]) + float(record["arrival_lag_ms"])
        if scheduled_ttft is not None:
            scheduled_ttft_records.append({"scheduled_ttft_ms": scheduled_ttft})
    generated = sum(int(record.get("generated_tokens", 0)) for record in request_records)
    if wallclock_s is None:
        completions = _numeric(request_records, "completion_s")
        arrivals = _numeric(request_records, "actual_arrival_s")
        if completions and arrivals:
            wallclock_s = max(completions) - min(arrivals)
        else:
            wallclock_s = sum(_numeric(request_records, "e2e_ms")) / 1000.0

    accepted_per_verify = _numeric(request_records, "mean_accepted_tokens_per_verify")
    successful_records = [
        record
        for record in request_records
        if record.get("success", record.get("status", "ok") not in {"error", "failed"})
    ]
    successful_tokens = sum(int(record.get("generated_tokens", 0)) for record in successful_records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "dataset": dataset,
        "workload_hash": workload_hash,
        "num_requests": len(request_records),
        "num_successful_requests": len(successful_records),
        "evaluation_scope": evaluation_scope,
        "quality_claim": (
            "communication_smoke_only; max_new_tokens=16 is not a full quality evaluation"
            if evaluation_scope == "communication_smoke"
            else "quality metrics are reported for the configured workload"
        ),
        "total_generated_tokens": generated,
        "latency_basis": "request_processing_start",
        "wallclock_s": float(wallclock_s or 0.0),
        "throughput_tok_s": generated / wallclock_s if wallclock_s and wallclock_s > 0 else None,
        "goodput_req_s": (
            len(successful_records) / wallclock_s if wallclock_s and wallclock_s > 0 else None
        ),
        "goodput_tok_s": (
            successful_tokens / wallclock_s if wallclock_s and wallclock_s > 0 else None
        ),
        "ttft_ms": _metric_stats(request_records, "ttft_ms"),
        "scheduled_ttft_ms": _metric_stats(scheduled_ttft_records, "scheduled_ttft_ms"),
        "arrival_lag_ms": _metric_stats(request_records, "arrival_lag_ms"),
        "tpot_ms": _metric_stats(request_records, "tpot_ms"),
        "e2e_ms": _metric_stats(request_records, "e2e_ms"),
        "accept_rate": (
            sum(int(record.get("accepted_tokens", 0)) for record in request_records)
            / sum(int(record.get("drafted_tokens", 0)) for record in request_records)
            if sum(int(record.get("drafted_tokens", 0)) for record in request_records) > 0
            else None
        ),
        "mean_accepted_tokens_per_verify": (
            statistics.mean(accepted_per_verify) if accepted_per_verify else None
        ),
        "quality_exact_match": exact_match_quality(request_records, dataset),
        "metric_definitions": {
            "ttft_ms": (
                "CPU worker request processing start before session/input preparation to first "
                "visible output token; excludes scheduled wait and previous-request queue"
            ),
            "e2e_ms": "same CPU worker request processing start as TTFT to request completion",
            "scheduled_ttft_ms": "diagnostic only: scheduled workload release to first output token, including arrival lag",
            "arrival_lag_ms": "diagnostic only: delay from scheduled release to worker processing start",
            "tpot_ms": "post-first-output generation time divided by remaining output tokens",
            "request_throughput_tok_s": "request generated output tokens divided by processing-start E2E seconds",
            "throughput_tok_s": "all generated output tokens divided by measured run wallclock, retaining scheduled load gaps",
            "goodput_req_s": "successful completed requests divided by measured run wallclock",
            "goodput_tok_s": "tokens from successful completed requests divided by measured run wallclock",
            "request_bytes": (
                "serialized application-layer HTTP JSON body or protobuf request ByteSize; "
                "excludes HTTP/gRPC framing, TCP/IP, and SSH encapsulation"
            ),
            "response_bytes": (
                "serialized application-layer HTTP response body or protobuf response ByteSize; "
                "excludes HTTP/gRPC framing, TCP/IP, and SSH encapsulation"
            ),
            "rpc_count": (
                "count of measured application RPCs: HTTP POST calls for FastSD or Validate "
                "calls for SpecEdge"
            ),
        },
        "resource_metrics": _resource_stats(request_records),
        "slo": evaluate_slo(
            request_records,
            wallclock_s=wallclock_s,
            thresholds=slo_thresholds,
        ),
    }
    if paired_records is not None:
        baseline = list(paired_records)
        summary["paired_ci_vs_baseline"] = {
            key: paired_bootstrap_ci(request_records, baseline, key=key)
            for key in ("ttft_ms", "e2e_ms", "generated_tokens")
        }
    if extra:
        summary.update(extra)
    return summary


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    # Summary artifacts are evidence.  Do not replace a previous failed or
    # partial result on a rerun; identical content is accepted as idempotent.
    from .run_artifacts import write_json_once

    return write_json_once(path, payload)
