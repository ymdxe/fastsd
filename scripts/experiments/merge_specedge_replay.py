#!/usr/bin/env python3
"""Merge two partitioned Official SpecEdge replay artifacts without overwriting.

The pinned Official client configuration is process-global, so the experiment
starts one replay process per edge GPU.  This tool restores the canonical
arrival-trace order only after both processes have finished.  It validates
that every selected trace row appears exactly once rather than silently
concatenating partial output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON artifact {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSONL {path}:{line_number}: {exc}")
        if not isinstance(value, dict):
            fail(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    if not rows:
        fail(f"no request records in {path}")
    return rows


def integer(value: Any, *, field: str, source: Path | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        location = f" in {source}" if source else ""
        fail(f"{field} must be a non-negative integer{location}")
    return value


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def completion_token_count(record: dict[str, Any]) -> int | None:
    """Read an explicit adapter completion count without guessing a token API."""

    result = record.get("client_result")
    if not isinstance(result, dict) or "generated_token_count" not in result:
        return None
    trace_index = record.get("trace_index")
    return integer(
        result["generated_token_count"],
        field=f"client_result.generated_token_count at trace_index {trace_index}",
    )


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def selected_trace(trace_path: Path, max_requests: int | None) -> list[dict[str, Any]]:
    rows = load_jsonl(trace_path)
    if max_requests is not None:
        if max_requests <= 0:
            fail("--max-requests must be positive when supplied")
        rows = rows[:max_requests]
    expected: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        arrival_index = integer(row.get("arrival_index"), field="arrival_index", source=trace_path)
        if arrival_index != expected_index:
            fail(
                f"trace arrival_index must be contiguous at {trace_path}: "
                f"expected {expected_index}, got {arrival_index}"
            )
        integer(row.get("client_id"), field="client_id", source=trace_path)
        expected.append(row)
    if not expected:
        fail(f"trace has no selected requests: {trace_path}")
    return expected


def merge(
    *,
    trace_path: Path,
    request_paths: list[Path],
    summary_paths: list[Path],
    output_path: Path,
    summary_output_path: Path,
    run_id: str,
    max_requests: int | None,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        fail(f"refusing to overwrite request output: {output_path}")
    if summary_output_path.exists() or summary_output_path.is_symlink():
        fail(f"refusing to overwrite summary output: {summary_output_path}")
    if output_path.parent != summary_output_path.parent:
        fail("request output and summary output must share a parent directory")
    if not output_path.parent.is_dir():
        fail(f"output parent does not exist: {output_path.parent}")

    trace = selected_trace(trace_path, max_requests)
    trace_sha = sha256_file(trace_path)
    expected_by_index = {index: row for index, row in enumerate(trace)}
    merged: dict[int, dict[str, Any]] = {}

    if len(request_paths) != len(summary_paths):
        fail("request and summary artifact counts must match")
    client_summaries: list[dict[str, Any]] = []
    for request_path, summary_path in zip(request_paths, summary_paths):
        summary = load_json(summary_path)
        if summary.get("run_id") != run_id:
            fail(f"summary run_id mismatch: {summary_path}")
        if summary.get("trace_sha256") != trace_sha:
            fail(f"summary trace SHA mismatch: {summary_path}")
        client_summaries.append(summary)
        for record in load_jsonl(request_path):
            if record.get("run_id") != run_id:
                fail(f"request run_id mismatch in {request_path}")
            trace_index = integer(record.get("trace_index"), field="trace_index", source=request_path)
            trace_row = expected_by_index.get(trace_index)
            if trace_row is None:
                fail(f"request trace_index {trace_index} is absent from selected trace")
            if trace_index in merged:
                fail(f"duplicate request record for trace_index {trace_index}")
            if record.get("client_id") != trace_row.get("client_id"):
                fail(f"request client_id disagrees with trace at trace_index {trace_index}")
            merged[trace_index] = record

    if set(merged) != set(expected_by_index):
        missing = sorted(set(expected_by_index) - set(merged))
        extra = sorted(set(merged) - set(expected_by_index))
        fail(f"partitioned replay does not cover the selected trace (missing={missing[:5]}, extra={extra[:5]})")

    ordered = [merged[index] for index in sorted(merged)]
    invalid_statuses = [
        f"{record.get('trace_index')}:{record.get('status')!r}"
        for record in ordered
        if record.get("status") != "completed"
    ]
    if invalid_statuses:
        fail(
            "formal SpecEdge replay contains non-completed requests; "
            "refusing canonical merge (examples="
            f"{invalid_statuses[:5]})"
        )
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    completed = [record for record in ordered if record.get("status") == "completed"]
    e2e = [number for record in completed if (number := finite(record.get("e2e_from_arrival_s"))) is not None]
    arrival_lag = [number for record in ordered if (number := finite(record.get("arrival_lag_s"))) is not None]
    queue_wait = [number for record in ordered if (number := finite(record.get("queue_wait_s"))) is not None]
    arrivals = [number for record in ordered if (number := finite(record.get("actual_arrival_monotonic_s"))) is not None]
    completions = [number for record in ordered if (number := finite(record.get("completion_monotonic_s"))) is not None]
    elapsed_s = max(completions) - min(arrivals) if arrivals and completions else None
    if elapsed_s is not None:
        elapsed_s = max(0.0, elapsed_s)
    generated_token_counts = [
        count
        for record in completed
        if (count := completion_token_count(record)) is not None
    ]
    generated_token_count_available = bool(completed) and len(
        generated_token_counts
    ) == len(completed)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "specedge",
        "run_id": run_id,
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha,
        "request_count": len(ordered),
        "completed_count": len(completed),
        "error_count": sum(record.get("status") == "error" for record in ordered),
        "cancelled_count": sum(record.get("status") == "cancelled" for record in ordered),
        "completion_rate": len(completed) / len(ordered),
        "measurement_window_s": elapsed_s,
        "generated_token_count_available": generated_token_count_available,
        "generated_token_count_record_count": len(generated_token_counts),
        "p50_e2e_from_arrival_s": percentile(e2e, 0.50),
        "p95_e2e_from_arrival_s": percentile(e2e, 0.95),
        "p99_e2e_from_arrival_s": percentile(e2e, 0.99),
        "p95_arrival_lag_s": percentile(arrival_lag, 0.95),
        "p95_edge_queue_wait_s": percentile(queue_wait, 0.95),
        "client_summaries": [
            {
                "trace_client_ids": item.get("trace_client_ids"),
                "metrics_output": item.get("metrics_output"),
                "elapsed_s": item.get("elapsed_s"),
            }
            for item in client_summaries
        ],
    }
    if generated_token_count_available:
        total_generated_tokens = sum(generated_token_counts)
        payload["total_generated_tokens"] = total_generated_tokens
        if elapsed_s is not None and elapsed_s > 0:
            payload["system_tok_per_s"] = total_generated_tokens / elapsed_s
    with summary_output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--requests", required=True, nargs="+")
    parser.add_argument("--summaries", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    summary = merge(
        trace_path=Path(args.trace),
        request_paths=[Path(value) for value in args.requests],
        summary_paths=[Path(value) for value in args.summaries],
        output_path=Path(args.output),
        summary_output_path=Path(args.summary_output),
        run_id=args.run_id,
        max_requests=args.max_requests,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
