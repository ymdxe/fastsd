"""Common request-level metric schema for FastSD comparison methods."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1

_IO_STALL_METRICS = ("io_stall_ms_total", "io_stall_count", "exposed_h2d_stall_ms")
IO_STALL_NOT_APPLICABLE = {
    "io_stall_ms_total": 0.0,
    "io_stall_count": 0,
    "exposed_h2d_stall_ms": 0.0,
    "io_stall_available": False,
    "io_stall_restore_count": 0,
}


def io_stall_fields(record: dict[str, Any], *, allow_unmarked: bool = False) -> dict[str, Any]:
    """Normalize a request record; only live target responses can infer availability."""
    complete = all(record.get(key) is not None for key in _IO_STALL_METRICS)
    marker = record.get("io_stall_available", allow_unmarked)
    available = complete and marker is True
    preserve = complete and (available or record.get("io_stall_available") is False)
    fields = {key: record.get(key) if preserve else None for key in _IO_STALL_METRICS}
    restores = record.get("io_stall_restore_count")
    fields["io_stall_restore_count"] = (
        restores if restores is not None else fields["io_stall_count"]
    ) if preserve else None
    fields["io_stall_available"] = available
    return fields


@dataclass
class IOStall:
    """Accumulate restore waits across phases of one request."""

    total_ms: float = 0.0
    count: int = 0
    exposed_h2d_ms: float = 0.0
    restore_count: int = 0
    available: bool = True

    def record(self, milliseconds: float | None, *, restored: bool = False) -> None:
        if milliseconds is None:
            self.available = False
            return
        self.total_ms += milliseconds
        self.exposed_h2d_ms += milliseconds
        self.count += int(milliseconds > 0)
        self.restore_count += int(restored or milliseconds > 0)

    def add(self, record: dict[str, Any], *, allow_unmarked: bool = False) -> None:
        fields = io_stall_fields(record, allow_unmarked=allow_unmarked)
        if not fields["io_stall_available"]:
            self.available = False
            return
        self.total_ms += fields["io_stall_ms_total"]
        self.count += fields["io_stall_count"]
        self.exposed_h2d_ms += fields["exposed_h2d_stall_ms"]
        self.restore_count += fields["io_stall_restore_count"]

    def fields(self) -> dict[str, Any]:
        return {
            "io_stall_ms_total": self.total_ms if self.available else None,
            "io_stall_count": self.count if self.available else None,
            "exposed_h2d_stall_ms": self.exposed_h2d_ms if self.available else None,
            "io_stall_restore_count": self.restore_count if self.available else None,
            "io_stall_available": self.available,
        }

    def summary(self) -> dict[str, float | int | None]:
        return {
            "total_ms": self.total_ms if self.available else None,
            "count": self.count if self.available else None,
            "avg_ms": self.total_ms / self.restore_count if self.available and self.restore_count else None,
        }


def summarize_io_stall(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    measured = [record for record in records if record.get("io_stall_available") is True
                and all(record.get(key) is not None for key in _IO_STALL_METRICS)]
    stall = IOStall(available=bool(measured))
    for record in measured:
        stall.add(record)
    return {"available_count": len(measured), "unavailable_count": len(records) - len(measured),
            "restore_count": stall.restore_count if measured else None,
            "exposed_h2d_stall_ms": stall.exposed_h2d_ms if measured else None,
            **stall.summary()}

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


def _successful(record: dict[str, Any]) -> bool:
    """Whether a request produced a completed result for system throughput."""

    status = str(record.get("status", "ok")).lower()
    value = record.get("success")
    if value is not None:
        return bool(value) and status not in {"error", "failed", "timeout"}
    return status not in {"error", "failed", "timeout", "cancelled", "canceled"}


def derive_measurement_window(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Derive the common workload window without summing concurrent requests.

    The preferred fields are workload-relative ``actual_arrival_s`` and
    ``completion_s``.  Epoch and monotonic pairs are accepted only when every
    record uses the same declared clock domain.  Missing or mixed domains are
    reported as unavailable instead of manufacturing a duration from the sum
    of request latencies.
    """

    rows = [dict(record) for record in records]
    if not rows:
        return {
            "wallclock_s": None,
            "status": "missing_records",
            "definition": "common workload window from release/arrival to last completion",
        }

    domains = {
        str(row["clock_domain"])
        for row in rows
        if row.get("clock_domain") not in (None, "")
    }
    if len(domains) > 1:
        return {
            "wallclock_s": None,
            "status": "incompatible_clock_domains",
            "clock_domains": sorted(domains),
            "definition": "common workload window from release/arrival to last completion",
        }

    pairs = (
        ("actual_arrival_s", "completion_s", "workload_relative_s"),
        ("scheduled_arrival_s", "completion_s", "workload_relative_s"),
        ("processing_start_s", "completion_s", "workload_relative_s"),
        ("processing_start_epoch_s", "completion_epoch_s", "epoch_s"),
        ("processing_start_monotonic_s", "completion_monotonic_s", "monotonic_s"),
    )
    for start_key, end_key, inferred_domain in pairs:
        starts = _numeric(rows, start_key)
        ends = _numeric(rows, end_key)
        if not starts or not ends:
            continue
        start = min(starts)
        end = max(ends)
        duration = end - start
        if not math.isfinite(duration) or duration < 0:
            return {
                "wallclock_s": None,
                "status": "invalid_window",
                "clock_domain": next(iter(domains), inferred_domain),
                "definition": "common workload window from release/arrival to last completion",
            }
        return {
            "wallclock_s": float(duration),
            "status": "derived",
            "clock_domain": next(iter(domains), inferred_domain),
            "start": float(start),
            "end": float(end),
            "definition": "min release/arrival to max completion over the measured workload",
        }
    return {
        "wallclock_s": None,
        "status": "missing_window_timestamps",
        "clock_domains": sorted(domains),
        "definition": "common workload window from release/arrival to last completion",
    }


def _metric_stats(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = _numeric(records, key)
    return {
        "avg": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def resolve_slo_thresholds(
    thresholds: dict[str, dict[str, float]] | None = None,
    *,
    dataset: str | None = None,
    scenario: str | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Resolve request-level SLO limits without inventing defaults.

    Explicit ``thresholds`` wins.  Otherwise a JSON object in
    ``SLO_CONFIG_JSON`` may contain ``default``, dataset, or scenario entries,
    and the three ``SLO_*_MS`` variables provide a small command-line-friendly
    override.  An empty result means SLO is unset and must be reported as such.
    """

    if thresholds is not None:
        return thresholds
    env = environ if environ is not None else os.environ
    config: Any = {}
    raw_config = env.get("SLO_CONFIG_JSON")
    if raw_config:
        try:
            config = json.loads(raw_config)
        except (TypeError, ValueError):
            config = {}
    if isinstance(config, dict):
        selected: Any = config.get("default", config)
        if dataset and isinstance(config.get(dataset), dict):
            selected = config[dataset]
        if scenario and isinstance(config.get(scenario), dict):
            selected = config[scenario]
        if isinstance(selected, dict) and selected:
            normalized: dict[str, dict[str, float]] = {}
            for metric, value in selected.items():
                key = str(metric).lower()
                if key in {"ttft", "ttft_ms"}:
                    key = "ttft_ms"
                elif key in {"tpot", "tpot_ms"}:
                    key = "tpot_ms"
                elif key in {"e2e", "e2e_ms"}:
                    key = "e2e_ms"
                else:
                    continue
                if isinstance(value, dict):
                    normalized[key] = {str(tier): float(limit) for tier, limit in value.items()}
                else:
                    normalized[key] = {"configured": float(value)}
            if normalized:
                return normalized

    def read_metric(metric: str) -> float | None:
        suffix = metric.upper().replace("_MS", "") + "_MS"
        names = []
        if scenario:
            names.append(f"SLO_{str(scenario).upper()}_{suffix}")
        if dataset:
            names.append(f"SLO_{str(dataset).upper()}_{suffix}")
        names.append(f"SLO_{suffix}")
        for name in names:
            raw = env.get(name)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                return value
        return None

    resolved: dict[str, dict[str, float]] = {}
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        value = read_metric(metric)
        if value is not None:
            resolved[metric] = {"configured": value}
    return resolved


def evaluate_slo(
    records: Iterable[dict[str, Any]],
    *,
    wallclock_s: float | None,
    thresholds: dict[str, dict[str, float]] | None = None,
    dataset: str | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Evaluate multi-tier latency SLOs and retain request-level evidence.

    Violation rate is computed over all completed records. Failed requests and
    missing metrics are retained as violations so incomplete telemetry cannot
    silently improve an SLO result. Goodput counts successful requests/tokens that satisfy the
    selected metric threshold and is normalized by the measured run wallclock.
    ``combined`` additionally requires TTFT, TPOT, and E2E to pass together.
    """

    rows = [dict(record) for record in records]
    configured = resolve_slo_thresholds(
        thresholds, dataset=dataset, scenario=scenario
    )
    if not configured:
        return {
            "status": "unset",
            "thresholds_ms": {},
            "by_metric": {},
            "combined": {},
            "TTFT_violation_rate": "unset",
            "TPOT_violation_rate": "unset",
            "E2E_violation_rate": "unset",
            "combined_violation_rate": "unset",
            "goodput_tok_s": "unset",
            "definitions": {
                "status": "No SLO_TTFT_MS/SLO_TPOT_MS/SLO_E2E_MS or explicit threshold was configured."
            },
        }
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
    # A combined SLO is meaningful only when all three request-level limits
    # are configured. Individual metric rates remain available for partial
    # configurations.
    combined_enabled = set(normalized) == {"ttft_ms", "tpot_ms", "e2e_ms"}
    tier_names = sorted({tier for tiers in normalized.values() for tier in tiers}) if combined_enabled else []
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
    # Keep a flat request-level view for reports and shell collectors.  The
    # selected tier is ``standard`` when present, otherwise the first tier.
    def selected(metric: str) -> dict[str, Any] | None:
        tiers = by_metric.get(metric) or {}
        return tiers.get("standard") or next(iter(tiers.values()), None)

    selected_rows = {metric: selected(metric) for metric in ("ttft_ms", "tpot_ms", "e2e_ms")}
    combined_row = combined.get("standard") or next(iter(combined.values()), None)
    return {
        "status": "configured" if combined_enabled else "partial",
        "thresholds_ms": normalized,
        "wallclock_s": duration,
        "by_metric": by_metric,
        "combined": combined,
        "TTFT_violation_rate": (selected_rows["ttft_ms"] or {}).get("violation_rate", "unset"),
        "TPOT_violation_rate": (selected_rows["tpot_ms"] or {}).get("violation_rate", "unset"),
        "E2E_violation_rate": (selected_rows["e2e_ms"] or {}).get("violation_rate", "unset"),
        "combined_violation_rate": (combined_row or {}).get("violation_rate", "unset"),
        "goodput_tok_s": (combined_row or {}).get("goodput_tok_s", "unset"),
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


def _with_derived_request_throughput(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy records and expose explicitly named request output speed.

    Older artifacts called this field ``throughput_tok_s``.  That name is now
    reserved for run-level system throughput; the compatibility read below
    keeps old JSONL usable without allowing it to masquerade as a system
    metric in new reports.
    """

    result: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if item.get("request_throughput_tok_s") is None:
            legacy = item.get("throughput_tok_s")
            if legacy is not None:
                try:
                    item["request_throughput_tok_s"] = float(legacy)
                except (TypeError, ValueError):
                    pass
        if item.get("request_throughput_tok_s") is None:
            e2e_ms = item.get("e2e_ms")
            generated = item.get("generated_tokens")
            if e2e_ms is not None and generated is not None and float(e2e_ms) > 0:
                item["request_throughput_tok_s"] = float(generated) / (float(e2e_ms) / 1000.0)
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

    left = _with_derived_request_throughput(method_records)
    right = _with_derived_request_throughput(baseline_records)
    metric_directions = {
        "ttft_ms": "lower_is_better",
        "e2e_ms": "lower_is_better",
        "tpot_ms": "lower_is_better",
        "request_throughput_tok_s": "higher_is_better",
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
    # Keep the historical key as an explicit N/A compatibility entry.  A
    # system throughput CI requires one value per run/repeat, not one value per
    # request, so it must never be computed by this paired request bootstrap.
    metrics["throughput_tok_s"] = {
        "n": 0,
        "mean_delta": None,
        "ci95": None,
        "baseline_mean": None,
        "improvement_pct": None,
        "improvement_ci95_pct": None,
        "direction": "higher_is_better",
        "status": "run_level_metric_requires_repeat_aggregation",
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
        "h2d_bytes",
        "d2h_bytes",
        "evictions",
        "restores",
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


_T_CRITICAL_95 = {
    1: None,
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
    21: 2.086,
    22: 2.080,
    23: 2.074,
    24: 2.069,
    25: 2.064,
    26: 2.060,
    27: 2.056,
    28: 2.052,
    29: 2.048,
    30: 2.045,
}


def aggregate_repeat_summaries(
    summaries: Iterable[dict[str, Any]],
    *,
    method: str | None = None,
    dataset: str | None = None,
    metrics: Iterable[str] = (
        "throughput_tok_s",
        "goodput_tok_s",
        "goodput_req_s",
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
    ),
) -> dict[str, Any]:
    """Aggregate independent run summaries with run-level 95% intervals."""

    rows = [dict(item) for item in summaries]
    if method is not None:
        rows = [item for item in rows if item.get("method") == method]
    if dataset is not None:
        rows = [item for item in rows if item.get("dataset") == dataset]
    repeat_indices = []
    for item in rows:
        value = item.get("repeat_index")
        if value is None and isinstance(item.get("topology"), dict):
            value = item["topology"].get("repeat_index")
        repeat_indices.append(None if value is None else int(value))
    known_indices = [value for value in repeat_indices if value is not None]
    if len(known_indices) != len(set(known_indices)):
        raise ValueError("repeat_index must be unique within a run-level aggregation")
    hashes = {str(item.get("workload_hash")) for item in rows if item.get("workload_hash")}
    if len(hashes) > 1:
        raise ValueError("workload_hash differs across repeats")

    def scalar(item: dict[str, Any], key: str) -> float | None:
        value = item.get(key)
        if isinstance(value, dict):
            value = value.get("avg", value.get("mean"))
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "median": None, "std": None, "ci95": None, "iqr": None}
        mean = statistics.mean(values)
        median = statistics.median(values)
        if len(values) < 2:
            std = 0.0
            interval = [mean, mean]
        else:
            std = statistics.stdev(values)
            critical = _T_CRITICAL_95.get(len(values), 1.96)
            half = critical * std / math.sqrt(len(values))
            interval = [mean - half, mean + half]
        return {
            "n": len(values),
            "mean": mean,
            "median": median,
            "std": std,
            "ci95": interval,
            "iqr": [percentile(values, 0.25), percentile(values, 0.75)],
        }

    metric_rows = {
        key: stats([value for item in rows if (value := scalar(item, key)) is not None])
        for key in metrics
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method if method is not None else (rows[0].get("method") if rows else None),
        "dataset": dataset if dataset is not None else (rows[0].get("dataset") if rows else None),
        "workload_hash": next(iter(hashes), None),
        "repeat_count": len(rows),
        "repeat_indices": known_indices,
        "metrics": metric_rows,
        "ci_definition": "95% two-sided t interval across independent run summaries; median/IQR also reported",
    }


def _last_number(value: Any) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return matches[-1] if matches else None


def exact_match_quality(records: list[dict[str, Any]], dataset: str) -> float | None:
    if dataset not in {"gsm8k", "mgsm"}:
        return None
    scored = 0
    correct = 0
    for record in records:
        if "numeric_exact_match" in record:
            value = record["numeric_exact_match"]
            if value is not None:
                scored += 1
                correct += int(value)
            continue
        if record.get("reference") is None:
            continue
        predicted = _last_number(record.get("output_text"))
        expected = _last_number(record["reference"])
        if expected is None:
            continue
        scored += 1
        correct += int(predicted == expected)
    return correct / scored if scored else None


def pass_at_1_quality(records: list[dict[str, Any]], dataset: str) -> float | None:
    """Use explicit isolated-executor evidence when available; never guess."""
    if dataset not in {"humaneval", "mbpp"}:
        return None
    values = []
    for record in records:
        value = record.get("pass_at_1", record.get("passed"))
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def mt_bench_judge_quality(records: list[dict[str, Any]], dataset: str) -> float | None:
    """Return a score only when an external judge explicitly supplied one."""
    if dataset != "mt_bench":
        return None
    values = []
    for record in records:
        value = record.get("judge_score", record.get("mt_bench_score"))
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


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
    track: str | None = None,
    concurrency: int | None = None,
    arrival_distribution: str | None = None,
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
    attempted_tokens = sum(int(record.get("generated_tokens", 0)) for record in request_records)
    successful_records = [record for record in request_records if _successful(record)]
    generated = sum(int(record.get("generated_tokens", 0)) for record in successful_records)
    window = derive_measurement_window(request_records)
    if wallclock_s is None:
        wallclock_s = window.get("wallclock_s")
    elif wallclock_s < 0 or not math.isfinite(float(wallclock_s)):
        raise ValueError("wallclock_s must be a finite non-negative duration")
    latency_track = str(track or "").lower() == "latency"
    suppress_system_throughput = latency_track

    accepted_per_verify = _numeric(request_records, "mean_accepted_tokens_per_verify")
    accepted_lengths = _numeric(request_records, "accept_length") or accepted_per_verify
    total_verify_steps = sum(int(record.get("verify_steps") or 0) for record in request_records)
    successful_tokens = sum(int(record.get("generated_tokens", 0)) for record in successful_records)
    throughput = (
        generated / float(wallclock_s)
        if wallclock_s is not None and wallclock_s > 0 and not suppress_system_throughput
        else None
    )
    attempted_throughput = (
        attempted_tokens / float(wallclock_s)
        if wallclock_s is not None and wallclock_s > 0 and not suppress_system_throughput
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "dataset": dataset,
        "workload_hash": workload_hash,
        "num_requests": len(request_records),
        "num_successful_requests": len(successful_records),
        "evaluation_scope": evaluation_scope,
        "track": track,
        "concurrency": concurrency,
        "arrival_distribution": arrival_distribution,
        "quality_claim": (
            "communication_smoke_only; max_new_tokens=16 is not a full quality evaluation"
            if evaluation_scope == "communication_smoke"
            else "quality metrics are reported for the configured workload"
        ),
        "total_generated_tokens": generated,
        "successful_generated_tokens": generated,
        "attempted_generated_tokens": attempted_tokens,
        "latency_basis": "request_processing_start",
        "wallclock_s": float(wallclock_s) if wallclock_s is not None else None,
        "measurement_window_s": float(wallclock_s) if wallclock_s is not None else None,
        "measurement_window_status": "suppressed_for_latency_track" if suppress_system_throughput else (
            "explicit" if wallclock_s is not None and window.get("status") != "derived" else window.get("status")
        ),
        "measurement_window_definition": window.get("definition"),
        "clock_domain": window.get("clock_domain"),
        "throughput_tok_s": throughput,
        "attempted_throughput_tok_s": attempted_throughput,
        "goodput_req_s": (
            len(successful_records) / wallclock_s if wallclock_s and wallclock_s > 0 else None
        ),
        "goodput_tok_s": (
            successful_tokens / wallclock_s if wallclock_s and wallclock_s > 0 else None
        ),
        "ttft_ms": _metric_stats(request_records, "ttft_ms"),
        "ttft_fallback_to_e2e": any(record.get("ttft_fallback_to_e2e", False) for record in request_records),
        "ttft_fallback_to_e2e_count": sum(bool(record.get("ttft_fallback_to_e2e")) for record in request_records),
        "scheduled_ttft_ms": _metric_stats(scheduled_ttft_records, "scheduled_ttft_ms"),
        "arrival_lag_ms": _metric_stats(request_records, "arrival_lag_ms"),
        "tpot_ms": _metric_stats(request_records, "tpot_ms"),
        "e2e_ms": _metric_stats(request_records, "e2e_ms"),
        "queue_wait_ms": _metric_stats(request_records, "queue_wait_ms"),
        "io_stall": summarize_io_stall(request_records),
        "accept_rate": (
            sum(int(record.get("accepted_tokens", 0)) for record in request_records)
            / sum(int(record.get("drafted_tokens", 0)) for record in request_records)
            if sum(int(record.get("drafted_tokens", 0)) for record in request_records) > 0
            else None
        ),
        "mean_accepted_tokens_per_verify": (
            statistics.mean(accepted_per_verify) if accepted_per_verify else None
        ),
        "accept_length": (
            sum(int(record.get("accepted_tokens") or 0) for record in request_records) / total_verify_steps
            if total_verify_steps else (statistics.mean(accepted_lengths) if accepted_lengths else None)
        ),
        "quality_exact_match": exact_match_quality(request_records, dataset),
        "quality_pass_at_1": pass_at_1_quality(request_records, dataset),
        "humaneval_pass_at_1": pass_at_1_quality(request_records, dataset) if dataset == "humaneval" else None,
        "mt_bench_judge_score": mt_bench_judge_quality(request_records, dataset),
        "quality_metric": (
            "numeric_answer_accuracy" if dataset in {"gsm8k", "mgsm"}
            else "pass@1" if dataset in {"humaneval", "mbpp"}
            else "judge_score" if dataset == "mt_bench"
            else "unset"
        ),
        "metric_definitions": {
            "ttft_ms": (
                "CPU worker request processing start before session/input preparation to first "
                "visible output token; includes target queue wait, excludes scheduled arrival wait "
                "and waiting for the same edge worker to finish its previous request"
            ),
            "e2e_ms": "same CPU worker request processing start as TTFT to request completion",
            "scheduled_ttft_ms": "diagnostic only: scheduled workload release to first output token, including arrival lag",
            "arrival_lag_ms": "diagnostic only: delay from scheduled release to worker processing start",
            "tpot_ms": "post-first-output generation time divided by remaining output tokens",
            "queue_wait_ms": "AR request enqueue to dequeue on the target host; already included in edge TTFT",
            "io_stall": (
                "time target computation is blocked waiting for a required KV restore, "
                "from instrumented requests only; count is blocking restores and the average uses all "
                "consumed restores, including fully covered prefetches as zero"
            ),
            "request_throughput_tok_s": "request generated output tokens divided by processing-start E2E seconds; a latency secondary metric",
            "throughput_tok_s": "successful completed output tokens divided by the common workload window; unavailable on the latency track",
            "attempted_throughput_tok_s": "all generated tokens, including failed/partial attempts, divided by the common workload window",
            "goodput_req_s": "successful completed requests divided by measured run wallclock; SLO-qualified goodput is reported under slo",
            "goodput_tok_s": "tokens from successful completed requests divided by measured run wallclock; SLO-qualified goodput is reported under slo",
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
            dataset=dataset,
        ),
    }
    summary["TTFT_violation_rate"] = summary["slo"].get("TTFT_violation_rate", "unset")
    summary["TPOT_violation_rate"] = summary["slo"].get("TPOT_violation_rate", "unset")
    summary["E2E_violation_rate"] = summary["slo"].get("E2E_violation_rate", "unset")
    summary["combined_violation_rate"] = summary["slo"].get("combined_violation_rate", "unset")
    summary["goodput_tok_s_slo"] = summary["slo"].get("goodput_tok_s", "unset")
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
