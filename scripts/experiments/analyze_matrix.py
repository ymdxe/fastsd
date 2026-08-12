#!/usr/bin/env python3
"""Create a fresh, dependency-free summary of finalized experiment runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            output[prefix] = numeric
    return output


def as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def integrate_gpu_energy_j(path: Path) -> float | None:
    """Trapezoid-integrate one sampler CSV, separately per physical GPU."""
    if not path.is_file():
        return None
    samples: dict[str, list[tuple[float, float]]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp = as_finite_number(row.get("monotonic_s"))
                power = as_finite_number(row.get("power_draw_w"))
                gpu = row.get("physical_gpu")
                if timestamp is None or power is None or not gpu:
                    continue
                samples.setdefault(gpu, []).append((timestamp, power))
    except OSError:
        return None
    energy = 0.0
    usable = False
    for rows in samples.values():
        rows.sort()
        for (left_t, left_w), (right_t, right_w) in zip(rows, rows[1:]):
            delta_s = right_t - left_t
            if 0.0 <= delta_s <= 10.0:
                energy += (left_w + right_w) * 0.5 * delta_s
                usable = True
    return energy if usable else None


def slo_goodput_req_per_s(
    run_dir: Path, summary: dict[str, Any], *, slo_e2e_ms: float
) -> float | None:
    """Count completed requests meeting a caller-recorded E2E SLO."""
    window = as_finite_number(summary.get("measurement_window_s"))
    if window is None:
        window = as_finite_number(summary.get("wallclock_s"))
    if window is None or window <= 0:
        return None
    path = run_dir / "metrics" / "requests.jsonl"
    if not path.is_file():
        return None
    good = 0
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
        for raw in rows:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or row.get("status") != "completed":
                continue
            e2e_ms = as_finite_number(row.get("task_e2e_ms"))
            if e2e_ms is None:
                e2e_s = as_finite_number(row.get("e2e_from_arrival_s"))
                e2e_ms = e2e_s * 1000.0 if e2e_s is not None else None
            if e2e_ms is not None and e2e_ms <= slo_e2e_ms:
                good += 1
    except (OSError, json.JSONDecodeError):
        return None
    return good / window


def common_metrics(
    summary: dict[str, Any], run_dir: Path, *, slo_e2e_ms: float | None = None
) -> dict[str, float]:
    """Normalize the fields available in both FastSD and adapter summaries.

    Missing values remain absent rather than being invented as zero.  This lets
    a final report distinguish an unsupported SpecEdge measurement from an
    actual zero result.
    """
    normalized: dict[str, float] = {}
    if "task_e2e_ms_p50" in summary:
        mappings = {
            "completion_rate": "completion_rate",
            "e2e_p50_ms": "task_e2e_ms_p50",
            "e2e_p95_ms": "task_e2e_ms_p95",
            "e2e_p99_ms": "task_e2e_ms_p99",
            "arrival_lag_p95_ms": "arrival_lag_ms_p95",
            "system_tok_per_s": "system_tok_per_s",
            "generated_tokens": "total_generated_tokens",
        }
        for destination, source in mappings.items():
            value = as_finite_number(summary.get(source))
            if value is not None:
                normalized[destination] = value
        completed = as_finite_number(summary.get("completed_requests"))
        wallclock = as_finite_number(summary.get("wallclock_s"))
        if completed is not None and wallclock and wallclock > 0:
            normalized["system_req_per_s"] = completed / wallclock
    else:
        mappings = {
            "completion_rate": "completion_rate",
            "system_tok_per_s": "system_tok_per_s",
            "generated_tokens": "total_generated_tokens",
        }
        for destination, source in mappings.items():
            value = as_finite_number(summary.get(source))
            if value is not None:
                normalized[destination] = value
        # Merged partitioned SpecEdge replays use the planned measurement
        # window (first actual arrival through last completion).  A standalone
        # adapter summary has only elapsed_s, so preserve that fallback for
        # dry runs and custom factories.
        elapsed = as_finite_number(summary.get("measurement_window_s"))
        if elapsed is None:
            elapsed = as_finite_number(summary.get("elapsed_s"))
        if elapsed is not None:
            normalized["elapsed_s"] = elapsed
        for destination, source in (
            ("e2e_p50_ms", "p50_e2e_from_arrival_s"),
            ("e2e_p95_ms", "p95_e2e_from_arrival_s"),
            ("e2e_p99_ms", "p99_e2e_from_arrival_s"),
            ("arrival_lag_p95_ms", "p95_arrival_lag_s"),
        ):
            value = as_finite_number(summary.get(source))
            if value is not None:
                normalized[destination] = value * 1000.0
        completed = as_finite_number(summary.get("completed_count"))
        if completed is not None and elapsed and elapsed > 0:
            normalized["system_req_per_s"] = completed / elapsed

    edge_energy = integrate_gpu_energy_j(run_dir / "edge" / "metrics" / "gpu_samples.csv")
    cloud_energy = integrate_gpu_energy_j(run_dir / "cloud" / "metrics" / "gpu_samples.csv")
    if edge_energy is not None:
        normalized["edge_energy_j"] = edge_energy
    if cloud_energy is not None:
        normalized["cloud_energy_j"] = cloud_energy
    if edge_energy is not None and cloud_energy is not None:
        normalized["total_energy_j"] = edge_energy + cloud_energy
        tokens = normalized.get("generated_tokens")
        if tokens and tokens > 0:
            normalized["j_per_generated_token"] = (edge_energy + cloud_energy) / tokens
    if slo_e2e_ms is not None:
        goodput = slo_goodput_req_per_s(run_dir, summary, slo_e2e_ms=slo_e2e_ms)
        if goodput is not None:
            normalized["slo_goodput_req_per_s"] = goodput
    return normalized


def load_record(
    run_dir: Path, *, slo_e2e_ms: float | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, "missing root manifest"
    try:
        manifest = load_mapping(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid root manifest: {exc}"
    if manifest.get("status") != "complete":
        return None, f"root status is {manifest.get('status')!r}"
    method = manifest.get("method")
    run_id = manifest.get("run_id")
    if not isinstance(method, str) or not isinstance(run_id, str):
        return None, "root manifest lacks method or run_id"
    summary_path = run_dir / "edge" / "metrics" / "summary.json"
    if not summary_path.is_file():
        return None, "missing edge metrics summary"
    try:
        summary = load_mapping(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid edge metrics summary: {exc}"
    trace_path = run_dir / "workload" / "arrival_trace.jsonl"
    record: dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        "run_dir": str(run_dir),
        "root_manifest_sha256": sha256(manifest_path),
        "trace_sha256": sha256(trace_path) if trace_path.is_file() else "",
    }
    record.update({f"metric.{key}": value for key, value in flatten_numeric(summary).items()})
    record.update(
        {
            f"common.{key}": value
            for key, value in common_metrics(
                summary, run_dir, slo_e2e_ms=slo_e2e_ms
            ).items()
        }
    )
    return record, None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["method"]), []).append(record)
    result: dict[str, Any] = {}
    for method, rows in sorted(groups.items()):
        metric_keys = sorted({key for row in rows for key in row if key.startswith("metric.")})
        metrics: dict[str, dict[str, float | int]] = {}
        for key in metric_keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            metrics[key] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
        result[method] = {"runs": len(rows), "metrics": metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--slo-e2e-ms",
        type=float,
        help="optional positive E2E SLO; records goodput using canonical request metrics",
    )
    args = parser.parse_args()
    run_root = Path(args.run_root)
    output = Path(args.output)
    if str(run_root) != "/home/hdd" and not str(run_root).startswith("/home/hdd/"):
        raise SystemExit(f"ERROR: --run-root must be under /home/hdd: {run_root}")
    if str(output) != "/home/hdd" and not str(output).startswith("/home/hdd/"):
        raise SystemExit(f"ERROR: --output must be under /home/hdd: {output}")
    if not run_root.is_dir():
        raise SystemExit(f"ERROR: run root does not exist: {run_root}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to reuse analysis output: {output}")
    if args.slo_e2e_ms is not None and (
        not math.isfinite(args.slo_e2e_ms) or args.slo_e2e_ms <= 0
    ):
        raise SystemExit("ERROR: --slo-e2e-ms must be finite and positive")

    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for candidate in sorted(run_root.iterdir(), key=lambda item: item.name):
        if not candidate.is_dir() or candidate == output:
            continue
        record, reason = load_record(candidate, slo_e2e_ms=args.slo_e2e_ms)
        if record is None:
            if (candidate / "manifest.json").exists():
                excluded.append({"run_dir": str(candidate), "reason": reason or "unknown"})
            continue
        records.append(record)
    if not records:
        raise SystemExit("ERROR: no complete runs with canonical edge metrics were found")

    output.mkdir(parents=True, exist_ok=False)
    fields = sorted({key for record in records for key in record})
    with (output / "runs.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "schema_version": 1,
        "run_root": str(run_root),
        "included_runs": len(records),
        "excluded_runs": excluded,
        "slo_e2e_ms": args.slo_e2e_ms,
        "by_method": summarize(records),
    }
    (output / "matrix_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
