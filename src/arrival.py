"""Deterministic workload traces for open-loop experiments.

The generator in this module deliberately uses a private ``random.Random``
instance.  It therefore does not alter process-global random state used by
model sampling, and a trace can be regenerated solely from its input dataset,
rate, and seed.

An arrival trace assigns each selected dataset row at most once.  It preserves
the first sampled exponential inter-arrival time: the first request is
scheduled at that offset rather than being silently moved to time zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


TRACE_SCHEMA_VERSION = 1
REQUIRED_TRACE_FIELDS = (
    "arrival_index",
    "dataset_index",
    "task_id",
    "client_id",
    "interarrival_s",
    "scheduled_offset_s",
)


def _positive_finite_rate(rate_rps: float) -> float:
    """Return a validated arrival rate in requests per second."""

    try:
        rate = float(rate_rps)
    except (TypeError, ValueError) as exc:
        raise ValueError("rate_rps must be a positive finite number") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_rps must be a positive finite number")
    return rate


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _task_id_for(record: Any, dataset_index: int, task_id_key: str) -> Any:
    """Use the dataset task id when present, otherwise use its stable index."""

    if isinstance(record, Mapping):
        task_id = record.get(task_id_key)
        if task_id is not None:
            return task_id
    return dataset_index


def generate_poisson_trace(
    dataset_records: Sequence[Any],
    *,
    rate_rps: float,
    seed: int,
    num_clients: int = 1,
    max_requests: int | None = None,
    task_id_key: str = "task_id",
) -> list[dict[str, Any]]:
    """Build a deterministic, open-loop Poisson arrival trace.

    Args:
        dataset_records: Source records in the exact order they should be
            assigned.  Each selected record is assigned once; the generator
            never cycles a dataset to manufacture extra requests.
        rate_rps: Global Poisson arrival rate in requests per second.
        seed: Seed used only by a local :class:`random.Random` instance.
        num_clients: Number of serial replay clients.  Client assignment is
            stable round-robin by ``arrival_index``.
        max_requests: Number of leading dataset records to include.  ``None``
            includes the whole dataset and a value greater than the dataset
            length is rejected.
        task_id_key: Mapping key used for a stable task id when available.

    Returns:
        JSON-serialisable dictionaries containing ``arrival_index``,
        ``dataset_index``, ``task_id``, ``client_id``, ``interarrival_s``, and
        ``scheduled_offset_s``.  The first offset equals the first sampled
        inter-arrival time, rather than being forced to zero.
    """

    rate = _positive_finite_rate(rate_rps)
    clients = _positive_int(num_clients, "num_clients")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(task_id_key, str) or not task_id_key:
        raise ValueError("task_id_key must be a non-empty string")

    record_count = len(dataset_records)
    if record_count == 0:
        raise ValueError("dataset_records must not be empty")

    request_count = record_count if max_requests is None else _positive_int(
        max_requests, "max_requests"
    )
    if request_count > record_count:
        raise ValueError(
            "max_requests cannot exceed dataset_records length; "
            "cycling would duplicate dataset assignments"
        )

    rng = random.Random(seed)
    scheduled_offset_s = 0.0
    trace: list[dict[str, Any]] = []
    for arrival_index in range(request_count):
        interarrival_s = rng.expovariate(rate)
        scheduled_offset_s += interarrival_s
        record = dataset_records[arrival_index]
        trace.append(
            {
                "arrival_index": arrival_index,
                "dataset_index": arrival_index,
                "task_id": _task_id_for(record, arrival_index, task_id_key),
                "client_id": arrival_index % clients,
                "interarrival_s": interarrival_s,
                "scheduled_offset_s": scheduled_offset_s,
            }
        )
    return trace


def validate_trace_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate trace schema and return rows as ordinary dictionaries.

    The checks make malformed or accidentally duplicated workload assignments
    fail before an experiment starts.  Extra fields are retained to allow
    callers to attach prompt metadata without changing this core schema.
    """

    validated: list[dict[str, Any]] = []
    dataset_indices: set[int] = set()
    previous_offset = -1.0

    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"trace row {expected_index} must be a mapping")
        missing = [field for field in REQUIRED_TRACE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"trace row {expected_index} is missing fields: {missing}")

        arrival_index = row["arrival_index"]
        dataset_index = row["dataset_index"]
        client_id = row["client_id"]
        if (
            isinstance(arrival_index, bool)
            or not isinstance(arrival_index, int)
            or arrival_index != expected_index
        ):
            raise ValueError("arrival_index values must be contiguous and start at zero")
        if (
            isinstance(dataset_index, bool)
            or not isinstance(dataset_index, int)
            or dataset_index < 0
        ):
            raise ValueError("dataset_index must be a non-negative integer")
        if dataset_index in dataset_indices:
            raise ValueError(f"duplicate dataset assignment: {dataset_index}")
        if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
            raise ValueError("client_id must be a non-negative integer")

        interarrival_s = _finite_float(row["interarrival_s"], "interarrival_s")
        scheduled_offset_s = _finite_float(
            row["scheduled_offset_s"], "scheduled_offset_s"
        )
        if interarrival_s <= 0.0:
            raise ValueError("interarrival_s must be positive")
        if scheduled_offset_s < 0.0:
            raise ValueError("scheduled_offset_s must be non-negative")
        if scheduled_offset_s < previous_offset:
            raise ValueError("scheduled_offset_s values must be monotonic")

        dataset_indices.add(dataset_index)
        previous_offset = scheduled_offset_s
        validated.append(dict(row))

    return validated


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def write_trace_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and write a trace as deterministic UTF-8 JSONL.

    Existing trace files are protected by default.  This keeps a trace SHA
    meaningful across methods and prevents a replay from silently replacing a
    workload used by an earlier run.
    """

    output_path = Path(path)
    validated = validate_trace_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in validated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def read_trace_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL trace without changing it."""

    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in trace {input_path} line {line_number}"
                ) from exc
            rows.append(row)
    return validate_trace_rows(rows)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_trace_manifest(path: str | Path) -> dict[str, Any]:
    """Return lightweight, JSON-serialisable provenance for a trace file."""

    trace_path = Path(path)
    rows = read_trace_jsonl(trace_path)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_file": trace_path.name,
        "trace_sha256": sha256_file(trace_path),
        "request_count": len(rows),
    }


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "REQUIRED_TRACE_FIELDS",
    "build_trace_manifest",
    "generate_poisson_trace",
    "read_trace_jsonl",
    "sha256_file",
    "validate_trace_rows",
    "write_trace_jsonl",
]
