#!/usr/bin/env python3
"""Replay a shared Poisson arrival trace against an explicitly supplied client.

This module is intentionally independent of ``baselines/specedge/official``.
Official SpecEdge's model/client construction is environment-specific and
imports CUDA, torch, grpc, and generated protobuf modules.  Guessing that
construction here would make a comparison less reproducible.  A live run must
therefore supply a small, reviewed client factory with ``--client-factory``.

The factory contract is deliberately narrow::

    def create_client(request: Mapping[str, Any], context: ReplayContext):
        async def invoke() -> Mapping[str, Any] | None:
            ...  # make one known Official SpecEdge request
        return invoke

The returned object must be callable with no arguments.  Its return value may
be synchronous or awaitable.  The adapter never imports or calls an Official
SpecEdge API by itself.  ``--dry-run`` uses a standard-library-only client so
trace timing and result collection can be validated before a GPU run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# When this file is executed by path, an explicit factory imported through the
# package name must still share these exception/context classes with __main__.
if __name__ == "__main__":
    sys.modules.setdefault("baselines.specedge.adapter.poisson_client", sys.modules[__name__])


SCHEMA_VERSION = 1
_MAX_RESULT_REPR_CHARS = 4096


class AdapterConfigurationError(RuntimeError):
    """Raised when replay cannot safely start with the supplied arguments."""


class TraceFormatError(ValueError):
    """Raised when a shared arrival trace is malformed or non-replayable."""


@dataclass(frozen=True)
class TraceRequest:
    """One immutable trace record with validated timing information."""

    request_id: str
    trace_index: int
    trace_line: int
    scheduled_offset_s: float
    client_id: int | None
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ReplayContext:
    """Timing and provenance passed to a user-supplied client factory."""

    run_id: str
    trace_path: str
    trace_sha256: str
    request_id: str
    trace_index: int
    client_id: int | None
    scheduled_offset_s: float
    run_started_monotonic_s: float
    scheduled_deadline_monotonic_s: float
    actual_arrival_monotonic_s: float
    dispatch_monotonic_s: float


ClientInvoker = Callable[[], Any]
ClientFactory = Callable[[Mapping[str, Any], ReplayContext], ClientInvoker]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_scheduled_offset(value: Any, *, trace_line: int) -> float:
    if isinstance(value, bool):
        raise TraceFormatError(
            f"trace line {trace_line}: scheduled_offset_s must be a number, not bool"
        )
    try:
        offset = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceFormatError(
            f"trace line {trace_line}: scheduled_offset_s must be numeric"
        ) from exc
    if not math.isfinite(offset) or offset < 0:
        raise TraceFormatError(
            f"trace line {trace_line}: scheduled_offset_s must be finite and >= 0"
        )
    return offset


def _as_non_negative_int(value: Any, *, field: str, trace_line: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceFormatError(
            f"trace line {trace_line}: {field} must be a non-negative integer"
        )
    return value


def load_trace(
    trace_path: str | Path,
    *,
    max_requests: Optional[int] = None,
    client_id: Optional[int] = None,
) -> tuple[list[TraceRequest], str]:
    """Load a canonical JSONL trace without importing Official SpecEdge.

    Each non-empty line must be a JSON object containing
    ``scheduled_offset_s``.  The values must be non-decreasing so the shared
    trace remains a faithful Poisson schedule rather than an implicitly sorted
    workload.  ``request_id`` is optional; a stable ``trace-<index>`` ID is
    added when it is absent.  ``client_id`` selects the canonical trace
    partition for one Official SpecEdge process.  The leading
    ``max_requests`` rows are selected before that partition filter, so two
    client processes replay the same global workload prefix rather than
    independently expanding it.
    """

    path = Path(trace_path)
    if not path.is_file():
        raise TraceFormatError(f"trace file does not exist: {path}")
    if max_requests is not None and max_requests <= 0:
        raise TraceFormatError("max_requests must be > 0 when supplied")
    if client_id is not None and (
        isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0
    ):
        raise TraceFormatError("client_id must be a non-negative integer when supplied")

    requests: list[TraceRequest] = []
    seen_request_ids: set[str] = set()
    previous_offset = -1.0
    trace_index = 0

    with path.open("r", encoding="utf-8") as handle:
        for trace_line, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TraceFormatError(
                    f"trace line {trace_line}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise TraceFormatError(f"trace line {trace_line}: expected a JSON object")
            if "scheduled_offset_s" not in payload:
                raise TraceFormatError(
                    f"trace line {trace_line}: missing required scheduled_offset_s"
                )

            scheduled_offset_s = _as_scheduled_offset(
                payload["scheduled_offset_s"], trace_line=trace_line
            )
            if scheduled_offset_s < previous_offset:
                raise TraceFormatError(
                    f"trace line {trace_line}: scheduled_offset_s must be non-decreasing"
                )
            previous_offset = scheduled_offset_s

            supplied_arrival_index = payload.get("arrival_index", trace_index)
            if "arrival_index" in payload:
                supplied_arrival_index = _as_non_negative_int(
                    supplied_arrival_index,
                    field="arrival_index",
                    trace_line=trace_line,
                )
                if supplied_arrival_index != trace_index:
                    raise TraceFormatError(
                        f"trace line {trace_line}: arrival_index must match canonical "
                        f"trace order ({trace_index})"
                    )

            trace_client_id: int | None = None
            if "client_id" in payload:
                trace_client_id = _as_non_negative_int(
                    payload["client_id"], field="client_id", trace_line=trace_line
                )
            if client_id is not None and trace_client_id is None:
                raise TraceFormatError(
                    f"trace line {trace_line}: client_id is required when --client-id is used"
                )

            request_id = str(payload.get("request_id", f"trace-{trace_index}"))
            if not request_id:
                raise TraceFormatError(f"trace line {trace_line}: request_id must not be empty")
            if request_id in seen_request_ids:
                raise TraceFormatError(
                    f"trace line {trace_line}: duplicate request_id {request_id!r}"
                )
            seen_request_ids.add(request_id)

            if client_id is None or trace_client_id == client_id:
                requests.append(
                    TraceRequest(
                        request_id=request_id,
                        trace_index=supplied_arrival_index,
                        trace_line=trace_line,
                        scheduled_offset_s=scheduled_offset_s,
                        client_id=trace_client_id,
                        payload=payload,
                    )
                )
            trace_index += 1
            if max_requests is not None and trace_index >= max_requests:
                break

    if not requests:
        suffix = "" if client_id is None else f" matching client_id={client_id}"
        raise TraceFormatError(f"trace has no requests{suffix}: {path}")
    return requests, _sha256_file(path)


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Keep factory metadata JSONL-safe without invoking arbitrary methods."""

    if depth > 8:
        return "<max metadata depth reached>"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    rendered = repr(value)
    if len(rendered) > _MAX_RESULT_REPR_CHARS:
        rendered = rendered[:_MAX_RESULT_REPR_CHARS] + "...<truncated>"
    return {"repr": rendered, "type": type(value).__name__}


async def _sleep_until(deadline_monotonic_s: float) -> None:
    """Wait for an absolute deadline; never derive timing from a prior request."""

    while True:
        remaining_s = deadline_monotonic_s - time.monotonic()
        if remaining_s <= 0:
            return
        await asyncio.sleep(remaining_s)


async def _await_result(value: Any, timeout_s: Optional[float]) -> Any:
    async def resolve() -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    if timeout_s is None:
        return await resolve()
    return await asyncio.wait_for(resolve(), timeout=timeout_s)


def _quantile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _generated_token_count(record: Mapping[str, Any]) -> int | None:
    """Return one explicit factory-reported completion token count, if valid.

    Generic/dry-run factories are not required to expose this optional field.
    A final merged SpecEdge summary only reports a total when every completed
    request supplied a valid count, so partial values can never masquerade as
    a system-wide tokens-per-second or joules-per-token measurement.
    """

    result = record.get("client_result")
    if not isinstance(result, Mapping) or "generated_token_count" not in result:
        return None
    value = result["generated_token_count"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _derive_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.summary.json")


def _assert_new_output_paths(output_path: Path, summary_path: Path) -> None:
    if output_path.resolve() == summary_path.resolve():
        raise AdapterConfigurationError("output and summary paths must differ")
    for path in (output_path, summary_path):
        if path.exists():
            raise AdapterConfigurationError(
                f"refusing to overwrite existing experiment artifact: {path}"
            )


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


async def replay_trace(
    requests: Sequence[TraceRequest],
    *,
    trace_path: str | Path,
    trace_sha256: str,
    run_id: str,
    client_factory: ClientFactory,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    max_concurrency: int = 1,
    request_timeout_s: Optional[float] = None,
    run_started_monotonic_s: Optional[float] = None,
) -> dict[str, Any]:
    """Run a trace with absolute monotonic deadlines and write JSONL metadata.

    Arrival is recorded as soon as each trace deadline is reached.  A semaphore
    then bounds the number of active clients, yielding a separately reported
    ``queue_wait_s``.  Consequently a slow request cannot shift later Poisson
    arrivals; it can only create measurable client-side queueing.
    """

    if not requests:
        raise AdapterConfigurationError("cannot replay an empty request list")
    if max_concurrency <= 0:
        raise AdapterConfigurationError("max_concurrency must be > 0")
    if request_timeout_s is not None and request_timeout_s <= 0:
        raise AdapterConfigurationError("request_timeout_s must be > 0 when supplied")
    if run_started_monotonic_s is not None and (
        isinstance(run_started_monotonic_s, bool)
        or not math.isfinite(float(run_started_monotonic_s))
    ):
        raise AdapterConfigurationError(
            "run_started_monotonic_s must be a finite monotonic timestamp"
        )
    if not callable(client_factory):
        raise AdapterConfigurationError("client_factory must be callable")

    output = Path(output_path)
    summary = Path(summary_path) if summary_path is not None else _derive_summary_path(output)
    _assert_new_output_paths(output, summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)
    observed_wall_time = time.time()
    observed_monotonic_s = time.monotonic()
    started_monotonic_s = (
        observed_monotonic_s
        if run_started_monotonic_s is None
        else float(run_started_monotonic_s)
    )
    # Preserve a best-effort wall-clock projection while making monotonic time
    # authoritative for all scheduling and latency calculations.
    started_wall_time = observed_wall_time + (
        started_monotonic_s - observed_monotonic_s
    )
    records: list[dict[str, Any]] = []
    record_lock = asyncio.Lock()

    # Exclusive creation prevents accidental replacement of a previous run.
    with output.open("x", encoding="utf-8") as output_handle:

        async def emit(record: dict[str, Any]) -> None:
            encoded = json.dumps(record, sort_keys=True, ensure_ascii=False)
            async with record_lock:
                output_handle.write(encoded + "\n")
                output_handle.flush()
                records.append(record)

        async def execute(request: TraceRequest) -> None:
            scheduled_deadline_monotonic_s = (
                started_monotonic_s + request.scheduled_offset_s
            )
            await _sleep_until(scheduled_deadline_monotonic_s)
            actual_arrival_monotonic_s = time.monotonic()
            await semaphore.acquire()
            dispatch_monotonic_s = time.monotonic()

            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "request",
                "run_id": run_id,
                "request_id": request.request_id,
                "trace_index": request.trace_index,
                "trace_line": request.trace_line,
                "client_id": request.client_id,
                "scheduled_offset_s": request.scheduled_offset_s,
                "scheduled_deadline_monotonic_s": scheduled_deadline_monotonic_s,
                "actual_arrival_monotonic_s": actual_arrival_monotonic_s,
                "actual_arrival_offset_s": actual_arrival_monotonic_s
                - started_monotonic_s,
                "arrival_lag_s": actual_arrival_monotonic_s
                - scheduled_deadline_monotonic_s,
                "dispatch_monotonic_s": dispatch_monotonic_s,
                "dispatch_offset_s": dispatch_monotonic_s - started_monotonic_s,
                "queue_wait_s": dispatch_monotonic_s - actual_arrival_monotonic_s,
                "status": "error",
                "success": False,
            }
            try:
                context = ReplayContext(
                    run_id=run_id,
                    trace_path=str(Path(trace_path)),
                    trace_sha256=trace_sha256,
                    request_id=request.request_id,
                    trace_index=request.trace_index,
                    client_id=request.client_id,
                    scheduled_offset_s=request.scheduled_offset_s,
                    run_started_monotonic_s=started_monotonic_s,
                    scheduled_deadline_monotonic_s=scheduled_deadline_monotonic_s,
                    actual_arrival_monotonic_s=actual_arrival_monotonic_s,
                    dispatch_monotonic_s=dispatch_monotonic_s,
                )
                client = client_factory(request.payload, context)
                if not callable(client):
                    raise AdapterConfigurationError(
                        "client factory must return a zero-argument callable; "
                        "the adapter will not guess an Official SpecEdge API"
                    )
                result = await _await_result(client(), request_timeout_s)
                completed_monotonic_s = time.monotonic()
                record.update(
                    {
                        "status": "completed",
                        "success": True,
                        "completion_monotonic_s": completed_monotonic_s,
                        "completion_offset_s": completed_monotonic_s
                        - started_monotonic_s,
                        "e2e_from_arrival_s": completed_monotonic_s
                        - actual_arrival_monotonic_s,
                        "service_from_dispatch_s": completed_monotonic_s
                        - dispatch_monotonic_s,
                        "client_result": _json_safe(result),
                    }
                )
            except asyncio.CancelledError:
                cancelled_monotonic_s = time.monotonic()
                record.update(
                    {
                        "status": "cancelled",
                        "error_type": "CancelledError",
                        "error_message": "replay task was cancelled",
                        "completion_monotonic_s": cancelled_monotonic_s,
                        "completion_offset_s": cancelled_monotonic_s
                        - started_monotonic_s,
                    }
                )
                await emit(record)
                raise
            except Exception as exc:  # Request-level failures belong in the artifact.
                failed_monotonic_s = time.monotonic()
                record.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "completion_monotonic_s": failed_monotonic_s,
                        "completion_offset_s": failed_monotonic_s
                        - started_monotonic_s,
                        "e2e_from_arrival_s": failed_monotonic_s
                        - actual_arrival_monotonic_s,
                        "service_from_dispatch_s": failed_monotonic_s
                        - dispatch_monotonic_s,
                    }
                )
            finally:
                semaphore.release()
            await emit(record)

        tasks = [asyncio.create_task(execute(request)) for request in requests]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    finished_wall_time = time.time()
    finished_monotonic_s = time.monotonic()
    successful = [record for record in records if record["success"]]
    durations = [record["e2e_from_arrival_s"] for record in successful]
    arrival_lags = [record["arrival_lag_s"] for record in records]
    error_count = sum(record["status"] == "error" for record in records)
    cancelled_count = sum(record["status"] == "cancelled" for record in records)
    generated_token_counts = [
        count
        for record in successful
        if (count := _generated_token_count(record)) is not None
    ]
    generated_token_count_available = bool(successful) and len(
        generated_token_counts
    ) == len(successful)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "run_id": run_id,
        "trace_path": str(Path(trace_path)),
        "trace_sha256": trace_sha256,
        "request_count": len(requests),
        "trace_client_ids": sorted(
            {request.client_id for request in requests if request.client_id is not None}
        ),
        "max_concurrency": max_concurrency,
        "request_timeout_s": request_timeout_s,
        "started_wall_time_unix_s": started_wall_time,
        "finished_wall_time_unix_s": finished_wall_time,
        "started_monotonic_s": started_monotonic_s,
        "finished_monotonic_s": finished_monotonic_s,
        "elapsed_s": finished_monotonic_s - started_monotonic_s,
        "completed_count": len(successful),
        "error_count": error_count,
        "cancelled_count": cancelled_count,
        # A replay library caller may inspect partial artifacts, but the CLI
        # maps this explicit state to a non-zero process exit.  That lets the
        # two-client formal launcher skip merge/finalization rather than
        # treating request-level RPC failures as a valid measurement.
        "replay_status": (
            "complete" if error_count == 0 and cancelled_count == 0 else "failed"
        ),
        "completion_rate": len(successful) / len(requests),
        "generated_token_count_available": generated_token_count_available,
        "generated_token_count_record_count": len(generated_token_counts),
        "metrics_output": str(output),
        "p50_e2e_from_arrival_s": _quantile(durations, 0.50),
        "p95_e2e_from_arrival_s": _quantile(durations, 0.95),
        "p99_e2e_from_arrival_s": _quantile(durations, 0.99),
        "p95_arrival_lag_s": _quantile(arrival_lags, 0.95),
    }
    if generated_token_count_available:
        metadata["total_generated_tokens"] = sum(generated_token_counts)
    _write_new_json(summary, metadata)
    return metadata


def make_dry_run_factory(latency_s: float = 0.0) -> ClientFactory:
    """Return a dependency-free client factory for timing/metadata smoke tests."""

    if latency_s < 0:
        raise AdapterConfigurationError("dry_run latency must be >= 0")

    def factory(request: Mapping[str, Any], context: ReplayContext) -> ClientInvoker:
        async def invoke() -> Mapping[str, Any]:
            if latency_s:
                await asyncio.sleep(latency_s)
            return {
                "mode": "dry_run",
                "request_id": context.request_id,
                "trace_index": context.trace_index,
                "payload_keys": sorted(str(key) for key in request.keys()),
            }

        return invoke

    return factory


def load_client_factory(specification: str) -> ClientFactory:
    """Load only an explicitly named factory; no Official API is inferred."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise AdapterConfigurationError(
            "client factory must use the form 'module.path:callable_name'"
        )
    # A path-executed adapter has its own directory on sys.path, not the repo
    # root.  Add the root only to make the explicit package name importable;
    # this does not import Official SpecEdge or any optional dependency.
    repository_root = str(Path(__file__).resolve().parents[3])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise AdapterConfigurationError(
            f"could not import explicit client factory module {module_name!r}: {exc}"
        ) from exc
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as exc:
        raise AdapterConfigurationError(
            f"client factory {specification!r} was not found"
        ) from exc
    if not callable(factory):
        raise AdapterConfigurationError(f"client factory {specification!r} is not callable")
    return factory


def prepare_client_factory(factory: ClientFactory) -> None:
    """Optionally warm an explicit factory before the measured replay window.

    A factory may expose a zero-argument ``prepare`` attribute to load a model
    or verify a known transport before ``replay_trace`` captures its monotonic
    start time.  Ordinary factories need not provide it.  Preparation stays
    explicit: this adapter never imports an Official SpecEdge implementation.
    """

    prepare = getattr(factory, "prepare", None)
    if prepare is None:
        return
    if not callable(prepare):
        raise AdapterConfigurationError("client factory prepare attribute must be callable")
    result = prepare()
    if inspect.isawaitable(result):
        asyncio.run(result)


def _finite_monotonic_timestamp(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise AdapterConfigurationError(f"{name} must be a finite monotonic timestamp")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterConfigurationError(
            f"{name} must be a finite monotonic timestamp"
        ) from exc
    if not math.isfinite(timestamp):
        raise AdapterConfigurationError(f"{name} must be a finite monotonic timestamp")
    return timestamp


def _publish_ready_file(path: str | Path) -> None:
    """Publish a new per-client readiness artifact without overwriting one."""

    ready_path = Path(path)
    if ready_path.exists() or ready_path.is_symlink():
        raise AdapterConfigurationError(
            f"refusing to overwrite existing replay ready artifact: {ready_path}"
        )
    if not ready_path.parent.is_dir():
        raise AdapterConfigurationError(
            f"replay ready-file parent must already exist: {ready_path.parent}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "event": "factory_prepared",
        "pid": os.getpid(),
        "ready_monotonic_s": time.monotonic(),
    }
    _write_new_json(ready_path, payload)


def _read_start_file(path: str | Path) -> float:
    start_path = Path(path)
    if not start_path.is_file():
        raise AdapterConfigurationError(f"replay start file is not a file: {start_path}")
    try:
        document = json.loads(start_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterConfigurationError(
            f"replay start file is not valid JSON: {start_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise AdapterConfigurationError("replay start file must contain a JSON object")
    if "run_started_monotonic_s" not in document:
        raise AdapterConfigurationError(
            "replay start file is missing run_started_monotonic_s"
        )
    return _finite_monotonic_timestamp(
        document["run_started_monotonic_s"], name="run_started_monotonic_s"
    )


def wait_for_start_file(path: str | Path, *, timeout_s: float) -> float:
    """Wait for a wrapper-created common start barrier on this same host."""

    if timeout_s <= 0 or not math.isfinite(timeout_s):
        raise AdapterConfigurationError("start-barrier-timeout-s must be finite and > 0")
    start_path = Path(path)
    deadline = time.monotonic() + timeout_s
    last_read_error: AdapterConfigurationError | None = None
    while True:
        if start_path.exists():
            try:
                start = _read_start_file(start_path)
            except AdapterConfigurationError as exc:
                # The coordinating wrapper creates the JSON with exclusive
                # creation.  A reader can observe the inode between create
                # and close, so retry a transient partial write until the
                # bounded barrier timeout rather than desynchronizing one
                # client while the other waits.
                last_read_error = exc
            else:
                if start <= time.monotonic():
                    raise AdapterConfigurationError(
                        "replay start barrier is already in the past; do not measure a "
                        "desynchronized multi-client trace"
                    )
                return start
        if time.monotonic() >= deadline:
            if last_read_error is not None:
                raise AdapterConfigurationError(
                    f"timed out waiting for a complete replay start file: {start_path}; "
                    f"last error: {last_read_error}"
                ) from last_read_error
            raise AdapterConfigurationError(
                f"timed out waiting for replay start file: {start_path}"
            )
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a shared Poisson trace through an explicit SpecEdge client factory."
    )
    parser.add_argument("--trace", required=True, help="Canonical JSONL trace path")
    parser.add_argument(
        "--output", required=True, help="New request-metrics JSONL path (must not exist)"
    )
    parser.add_argument(
        "--summary-output", help="New summary JSON path (default: beside --output)"
    )
    parser.add_argument("--run-id", required=True, help="Unique experiment run identifier")
    parser.add_argument(
        "--max-requests", type=int, help="Replay only the leading N trace records"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        help="Replay only canonical trace rows assigned to this edge client",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1, help="Maximum active client calls"
    )
    parser.add_argument(
        "--request-timeout-s", type=float, help="Per-request invocation timeout"
    )
    parser.add_argument(
        "--start-at-monotonic-s",
        type=float,
        help="Explicit common monotonic origin for a same-host replay partition",
    )
    parser.add_argument(
        "--ready-file",
        help="New per-client readiness JSON written after factory preparation",
    )
    parser.add_argument(
        "--start-file",
        help="Wrapper-created JSON containing shared run_started_monotonic_s",
    )
    parser.add_argument(
        "--start-barrier-timeout-s",
        type=float,
        default=180.0,
        help="Maximum wait for --start-file after publishing --ready-file",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a dependency-free no-op client; sends no request to SpecEdge",
    )
    mode.add_argument(
        "--client-factory",
        help="Explicit factory module.path:callable_name; no official API is guessed",
    )
    parser.add_argument(
        "--dry-run-latency-s",
        type=float,
        default=0.0,
        help="Optional simulated service latency used only with --dry-run",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.dry_run_latency_s < 0:
        parser.error("--dry-run-latency-s must be >= 0")
    if not args.dry_run and args.dry_run_latency_s != 0:
        parser.error("--dry-run-latency-s may only be used with --dry-run")
    if bool(args.ready_file) != bool(args.start_file):
        parser.error("--ready-file and --start-file must be supplied together")
    if args.start_file and args.start_at_monotonic_s is not None:
        parser.error("--start-file cannot be combined with --start-at-monotonic-s")

    try:
        requests, trace_sha256 = load_trace(
            args.trace,
            max_requests=args.max_requests,
            client_id=args.client_id,
        )
        factory = (
            make_dry_run_factory(args.dry_run_latency_s)
            if args.dry_run
            else load_client_factory(args.client_factory)
        )
        if getattr(factory, "requires_client_id", False):
            if args.client_id is None:
                raise AdapterConfigurationError(
                    "this client factory requires --client-id so one Official "
                    "SpecEdge process replays exactly one trace partition"
                )
            if args.max_concurrency != 1:
                raise AdapterConfigurationError(
                    "this client factory requires --max-concurrency 1; use one "
                    "process per logical Official SpecEdge client"
                )
        prepare_client_factory(factory)
        shared_start_monotonic_s: Optional[float] = None
        if args.ready_file:
            _publish_ready_file(args.ready_file)
            shared_start_monotonic_s = wait_for_start_file(
                args.start_file, timeout_s=args.start_barrier_timeout_s
            )
        elif args.start_at_monotonic_s is not None:
            shared_start_monotonic_s = _finite_monotonic_timestamp(
                args.start_at_monotonic_s, name="start-at-monotonic-s"
            )
            if shared_start_monotonic_s <= time.monotonic():
                raise AdapterConfigurationError(
                    "start-at-monotonic-s is already in the past"
                )
        summary = asyncio.run(
            replay_trace(
                requests,
                trace_path=args.trace,
                trace_sha256=trace_sha256,
                run_id=args.run_id,
                client_factory=factory,
                output_path=args.output,
                summary_path=args.summary_output,
                max_concurrency=args.max_concurrency,
                request_timeout_s=args.request_timeout_s,
                run_started_monotonic_s=shared_start_monotonic_s,
            )
        )
    except (AdapterConfigurationError, TraceFormatError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: replay interrupted", file=sys.stderr)
        return 130

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["replay_status"] != "complete":
        print(
            "error: replay recorded request failures "
            f"(error_count={summary['error_count']}, "
            f"cancelled_count={summary['cancelled_count']}); "
            "refusing to report a successful formal replay",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
