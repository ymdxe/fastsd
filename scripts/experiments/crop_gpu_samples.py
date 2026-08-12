#!/usr/bin/env python3
"""Create a fresh GPU-sample CSV clipped to a host-local request window.

Raw samples intentionally begin before model loading so that provenance can
show cold-start behavior.  This tool creates the canonical samples used by
energy analysis: from first real request arrival to final completion only.
It never overwrites the raw input or an existing output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite: {value!r}")
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing window records: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{lineno}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{lineno}")
        rows.append(value)
    if not rows:
        raise ValueError(f"no window records in {path}")
    return rows


def resolve_window(records: list[dict[str, Any]], start_field: str, end_field: str) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for row in records:
        if row.get("status", "completed") != "completed":
            continue
        start = row.get(start_field)
        end = row.get(end_field)
        if start is None or end is None:
            continue
        starts.append(number(start, label=start_field))
        ends.append(number(end, label=end_field))
    if not starts or not ends:
        raise ValueError(
            "no completed records contain both requested window fields: "
            f"{start_field}, {end_field}"
        )
    window = (min(starts), max(ends))
    if window[1] <= window[0]:
        raise ValueError(f"non-positive measurement window: {window[0]}..{window[1]}")
    return window


def resolve_event_window(
    records: list[dict[str, Any]], *, timestamp_field: str, method: str
) -> tuple[float, float]:
    """Resolve a server-side RPC window from transparent adapter events."""
    entered: list[float] = []
    left: list[float] = []
    for row in records:
        if row.get("method") != method:
            continue
        timestamp = row.get(timestamp_field)
        if timestamp is None:
            continue
        # The transport adapter calls this field ``phase`` to avoid colliding
        # with its human-readable lifecycle ``event`` messages.  Accept the
        # generic name too so reviewed custom adapters can use the same tool.
        event = row.get("phase", row.get("event"))
        if event == "enter":
            entered.append(number(timestamp, label=timestamp_field))
        elif event == "leave":
            left.append(number(timestamp, label=timestamp_field))
    if not entered or not left:
        raise ValueError(
            "no complete server RPC event window for "
            f"method={method!r} and timestamp field {timestamp_field!r}"
        )
    window = (min(entered), max(left))
    if window[1] <= window[0]:
        raise ValueError(f"non-positive measurement window: {window[0]}..{window[1]}")
    return window


def read_samples(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"missing raw sampler CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "monotonic_s" not in reader.fieldnames:
            raise ValueError(f"sampler CSV needs monotonic_s header: {path}")
        rows = list(reader)
        fields = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"no raw GPU samples in {path}")
    return fields, rows


def interpolate(left: dict[str, str], right: dict[str, str], timestamp: float) -> dict[str, str]:
    left_t = number(left["monotonic_s"], label="sample monotonic_s")
    right_t = number(right["monotonic_s"], label="sample monotonic_s")
    if right_t <= left_t:
        raise ValueError("raw samples are not strictly ordered for interpolation")
    fraction = (timestamp - left_t) / (right_t - left_t)
    result = dict(left)
    result["monotonic_s"] = f"{timestamp:.9f}"
    # GPU power is the only continuous input to trapezoid integration.  Other
    # fields remain the left observation and are still useful provenance.
    for key in ("power_draw_w", "memory_used_mib", "utilization_gpu_pct"):
        if key not in left or key not in right:
            continue
        try:
            value = number(left[key], label=key) + fraction * (
                number(right[key], label=key) - number(left[key], label=key)
            )
        except ValueError:
            continue
        result[key] = f"{value:.9f}"
    return result


def boundary_row(rows: list[dict[str, str]], timestamp: float, *, is_start: bool) -> dict[str, str]:
    keyed = [(number(row["monotonic_s"], label="sample monotonic_s"), row) for row in rows]
    keyed.sort(key=lambda item: item[0])
    if timestamp < keyed[0][0] or timestamp > keyed[-1][0]:
        raise ValueError(
            "raw sampler does not cover the measurement boundary; do not "
            "estimate energy from an incomplete interval"
        )
    for index, (sample_t, row) in enumerate(keyed):
        if sample_t == timestamp:
            copied = dict(row)
            copied["monotonic_s"] = f"{timestamp:.9f}"
            return copied
        if sample_t > timestamp:
            return interpolate(keyed[index - 1][1], row, timestamp)
    # Exact final sample was handled above.
    raise AssertionError("unreachable boundary selection")


def crop_rows(rows: list[dict[str, str]], start: float, end: float) -> list[dict[str, str]]:
    by_gpu: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        gpu = row.get("physical_gpu")
        if not gpu:
            raise ValueError("sampler row lacks physical_gpu")
        by_gpu.setdefault(gpu, []).append(row)
    selected: list[dict[str, str]] = []
    for gpu, gpu_rows in sorted(by_gpu.items()):
        del gpu  # keeps ordering explicit without carrying it into payloads
        start_row = boundary_row(gpu_rows, start, is_start=True)
        end_row = boundary_row(gpu_rows, end, is_start=False)
        interior = [
            dict(row)
            for row in gpu_rows
            if start < number(row["monotonic_s"], label="sample monotonic_s") < end
        ]
        selected.extend([start_row, *interior, end_row])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="raw gpu_samples_raw.csv")
    parser.add_argument("--records", required=True, help="completed request/event JSONL")
    parser.add_argument("--start-field")
    parser.add_argument("--end-field")
    parser.add_argument(
        "--event-timestamp-field",
        help="adapter event timestamp field; uses enter/leave rows instead of request rows",
    )
    parser.add_argument("--event-method", default="Validate")
    parser.add_argument("--output", required=True, help="new canonical gpu_samples.csv")
    args = parser.parse_args()

    raw = Path(args.raw)
    records = Path(args.records)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to overwrite cropped sampler CSV: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"ERROR: output parent does not exist: {output.parent}")
    try:
        window_records = read_jsonl(records)
        if args.event_timestamp_field:
            if args.start_field or args.end_field:
                raise ValueError(
                    "--event-timestamp-field cannot be combined with --start-field/--end-field"
                )
            start, end = resolve_event_window(
                window_records,
                timestamp_field=args.event_timestamp_field,
                method=args.event_method,
            )
        else:
            if not args.start_field or not args.end_field:
                raise ValueError(
                    "supply both --start-field/--end-field or --event-timestamp-field"
                )
            start, end = resolve_window(window_records, args.start_field, args.end_field)
        fields, samples = read_samples(raw)
        cropped = crop_rows(samples, start, end)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cropped)
    print(json.dumps({"start_monotonic_s": start, "end_monotonic_s": end, "samples": len(cropped)}))


if __name__ == "__main__":
    main()
