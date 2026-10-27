import argparse
import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import re
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, List

import requests
import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(sys.path[0], "../"))

from src.arrival import poisson_arrival_offsets, shard_samples
from src.engine import Decoding
from src.kvcache import KVCacheModel
from src.metrics import elapsed_ms, tpot_ms
try:
    from src.speculative_sampling import (
        encode_sparse_block,
        make_generator,
        sparse_block_from_dense,
    )
except ImportError:  # lightweight entrypoint tests stub the src package
    def _missing_sampling_dependency(*_args, **_kwargs):
        raise RuntimeError("src.speculative_sampling is required for rejection mode")

    encode_sparse_block = _missing_sampling_dependency
    make_generator = _missing_sampling_dependency
    sparse_block_from_dense = _missing_sampling_dependency
try:
    from src.runtime import configure_torch_threads, resolve_dtype, synchronize
except ImportError:  # lightweight entrypoint tests may stub old runtime helpers
    from src.runtime import configure_torch_threads, resolve_dtype

    def synchronize(device: Any) -> None:
        del device
from src.util import parse_arguments, seed_everything
try:
    from src.progress_control import AheadController
except ImportError:  # entrypoint tests replace the src package with stubs
    AheadController = None


def _verify_with_ready_time(verify_call, **kwargs):
    response, elapsed = verify_call(**kwargs)
    response = dict(response)
    response["_response_ready_monotonic"] = time.monotonic()
    return response, elapsed



def _validate_prefill_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the state-establishing RPC before a verify can be submitted."""

    if not isinstance(response, dict) or response.get("status") != "prefill_ok":
        raise RuntimeError(f"prefill failed: {response}")
    return response


def _validate_prefill_arrival_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the lightweight arrival acknowledgement."""

    if not isinstance(response, dict) or response.get("status") != "arrival_recorded":
        raise RuntimeError(f"prefill arrival failed: {response}")
    return response


def _shutdown_executor(executor: concurrent.futures.Executor, *, wait: bool = True) -> None:
    """Shutdown an HTTP executor across Python versions with cancel support."""

    try:
        executor.shutdown(wait=wait, cancel_futures=True)
    except TypeError:
        # ``cancel_futures`` was added in Python 3.9.
        executor.shutdown(wait=wait)


def _submit_http_call(
    executor: concurrent.futures.Executor, function: Callable[..., Any], *args, **kwargs
) -> concurrent.futures.Future:
    """Submit one HTTP call and close the executor if that call fails."""

    future = executor.submit(function, *args, **kwargs)

    def close_after_failure(done_future: concurrent.futures.Future) -> None:
        if done_future.cancelled():
            return
        try:
            failed = done_future.exception() is not None
        except BaseException:
            failed = True
        if failed:
            # ``wait=False`` is intentional here: this callback can execute on
            # the executor's worker thread.  The caller still joins it in the
            # normal/failure path via ``_shutdown_executor(..., wait=True)``.
            _shutdown_executor(executor, wait=False)

    future.add_done_callback(close_after_failure)
    return future


def _run_first_draft_with_prefill(
    executor: concurrent.futures.Executor,
    prefill_call: Callable[[], Dict[str, Any]],
    draft_call: Callable[[], Any],
    *,
    overlap: bool,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    timing_call: Callable[[Dict[str, float]], Dict[str, Any]] | None = None,
    cancel_call: Callable[[], Any] | None = None,
) -> tuple[Any, Dict[str, Any], Dict[str, float]]:
    """Start prompt Prefill independently from local first-round drafting.

    The returned timestamps are monotonic and describe the two operations' actual
    intervals.  Prefill validation happens before this helper returns, which gives
    the caller a hard gate before it submits the first Verify.  ``draft_call`` is
    deliberately invoked exactly once so the resulting KV/probability state can be
    reused by the normal decoding loop.
    """

    # A timing-required Prefill is a long poll that cannot complete until the
    # timing update arrives.  Treat the presence of ``timing_call`` as an
    # explicit requirement to overlap, even if an older caller omitted the
    # overlap flag.
    overlap = bool(overlap or timing_call is not None)
    prefill_started_at = clock()
    prefill_future = None

    def run_prefill() -> tuple[Dict[str, Any], float]:
        response = prefill_call()
        completed_at = clock()
        _validate_prefill_response(response)
        return response, completed_at

    def unpack_draft_result(result: Any) -> tuple[Any, Dict[str, Any]]:
        # The timing-aware Edge generator may return ``(tokens, timing_dict)``;
        # retain the original plain-output contract for existing callers/tests.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        return result, {}

    def upload_timing(local_timing: Dict[str, Any], first_draft_ms: float):
        if timing_call is None:
            return None
        response = timing_call(
            {
                "local_prefill_s": float(
                    local_timing.get("local_prefill_s", first_draft_ms / 1000.0)
                ),
                "local_decode_per_token_s": float(
                    local_timing.get("local_decode_per_token_s", first_draft_ms / 1000.0)
                ),
            }
        )
        if not isinstance(response, dict) or response.get("status") != "timing_ready":
            raise RuntimeError(f"prefill timing update failed: {response}")
        return response

    try:
        if overlap:
            prefill_future = _submit_http_call(executor, run_prefill)
            draft_started_at = clock()
            draft_result = draft_call()
            draft_completed_at = clock()
            draft_output, local_timing = unpack_draft_result(draft_result)
            # This call must happen before waiting for the long-poll Prefill.
            # Otherwise the cloud cannot mark its WorkItem timing-ready and the
            # future would wait forever.
            timing_response = upload_timing(
                local_timing,
                max(0.0, (draft_completed_at - draft_started_at) * 1000.0),
            )

            wait_timeout = None
            if timeout is not None:
                wait_timeout = max(0.0, timeout - (clock() - prefill_started_at))
            try:
                prefill_response, prefill_completed_at = prefill_future.result(
                    timeout=wait_timeout
                )
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError("prefill timed out before first verify") from exc
        else:
            prefill_response, prefill_completed_at = run_prefill()
            draft_started_at = clock()
            draft_result = draft_call()
            draft_completed_at = clock()
            draft_output, local_timing = unpack_draft_result(draft_result)
            timing_response = upload_timing(
                local_timing,
                max(0.0, (draft_completed_at - draft_started_at) * 1000.0),
            )
    except BaseException:
        # No Verify may be queued after a Prefill/draft/timing failure.  Also
        # release a cloud long-poll before joining the executor; otherwise a
        # local draft exception would leave the timing-gated Prefill blocked.
        if prefill_future is not None and not prefill_future.done() and cancel_call is not None:
            try:
                cancel_call()
            except BaseException:
                # Preserve the original local error; the cloud endpoint has
                # its own timeout/worker-death cleanup path.
                pass
        _shutdown_executor(executor, wait=True)
        raise

    timings = {
        "prefill_started_at": float(prefill_started_at),
        "prefill_completed_at": float(prefill_completed_at),
        "draft_started_at": float(draft_started_at),
        "draft_completed_at": float(draft_completed_at),
        "prefill_ms": max(0.0, (prefill_completed_at - prefill_started_at) * 1000.0),
        "first_draft_ms": max(0.0, (draft_completed_at - draft_started_at) * 1000.0),
    }
    for field in ("local_prefill_s", "local_decode_per_token_s", "local_prefill_ms", "local_decode_per_token_ms"):
        if field in local_timing:
            timings[field] = float(local_timing[field])
    if isinstance(timing_response, dict):
        for field in ("push_s", "push_ms", "timing_upload_s", "timing_upload_ms"):
            if field in timing_response:
                timings[field] = float(timing_response[field])
    timings["prefill_first_draft_overlap_ms"] = max(
        0.0,
        (
            min(prefill_completed_at, draft_completed_at)
            - max(prefill_started_at, draft_started_at)
        )
        * 1000.0,
    ) if overlap else 0.0
    timings["prefill_wait_after_first_draft_ms"] = (
        max(0.0, (prefill_completed_at - draft_completed_at) * 1000.0)
        if overlap
        else 0.0
    )
    return draft_output, prefill_response, timings


def _run_first_draft_after_arrival(
    executor: concurrent.futures.Executor,
    arrival_call: Callable[[], Dict[str, Any]],
    draft_call: Callable[[], Any],
    prefill_upload_call: Callable[[Any], Dict[str, Any]],
    *,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    cancel_call: Callable[[], Any] | None = None,
) -> tuple[Any, Dict[str, Any], Dict[str, float]]:
    """Record arrival, draft locally, then upload tokens to start Prefill.

    The arrival RPC carries no prompt or draft tokens.  It is issued in the
    background so local first-round generation can proceed while the cloud is
    idle.  Only after that round completes are the generated tokens uploaded;
    the upload response is the state-establishing Prefill response.
    """

    arrival_started_at = clock()
    arrival_future = _submit_http_call(executor, arrival_call)

    def unpack_draft_result(result: Any) -> tuple[Any, Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        return result, {}

    try:
        draft_started_at = clock()
        draft_result = draft_call()
        draft_completed_at = clock()
        draft_output, local_timing = unpack_draft_result(draft_result)

        wait_timeout = None
        if timeout is not None:
            wait_timeout = max(0.0, timeout - (clock() - arrival_started_at))
        try:
            arrival_response = arrival_future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError("prefill arrival acknowledgement timed out") from exc
        arrival_completed_at = clock()
        _validate_prefill_arrival_response(arrival_response)

        upload_started_at = clock()
        prefill_response = prefill_upload_call(draft_output)
        prefill_completed_at = clock()
        _validate_prefill_response(prefill_response)
    except BaseException:
        if cancel_call is not None:
            try:
                cancel_call()
            except BaseException:
                pass
        _shutdown_executor(executor, wait=True)
        raise

    timings = {
        "arrival_started_at": float(arrival_started_at),
        "arrival_completed_at": float(arrival_completed_at),
        "prefill_started_at": float(upload_started_at),
        "prefill_completed_at": float(prefill_completed_at),
        "draft_started_at": float(draft_started_at),
        "draft_completed_at": float(draft_completed_at),
        "prefill_ms": max(0.0, (prefill_completed_at - upload_started_at) * 1000.0),
        "first_draft_ms": max(0.0, (draft_completed_at - draft_started_at) * 1000.0),
        # The cloud Prefill starts only after the local round has been sent.
        "prefill_first_draft_overlap_ms": 0.0,
        "prefill_wait_after_first_draft_ms": 0.0,
    }
    for field in (
        "local_prefill_s",
        "local_decode_per_token_s",
        "local_prefill_ms",
        "local_decode_per_token_ms",
    ):
        if field in local_timing:
            timings[field] = float(local_timing[field])
    return draft_output, prefill_response, timings


def _timed_local_draft_generate(approx_model_cache: Any, prefix: Any, gamma: int):
    """Generate one first Prefill token plus ``gamma-1`` Decode tokens.

    ``KVCacheModel.generate(prefix, gamma)`` performs exactly this sequence,
    but does not expose the first full-prefix forward separately.  Splitting
    the loop preserves sampling/KV semantics while making the two terms in the
    priority formula directly measurable.  CUDA synchronization brackets each
    forward so asynchronous kernel launches cannot under-report latency.
    """

    gamma = int(gamma)
    if gamma <= 0:
        return prefix, {"local_prefill_s": 0.0, "local_decode_per_token_s": 0.0}
    device = getattr(getattr(approx_model_cache, "_model", None), "device", "cpu")
    synchronize(device)
    started = time.monotonic_ns()
    output = approx_model_cache.generate(prefix, 1)
    synchronize(device)
    prefill_s = max(0.0, (time.monotonic_ns() - started) / 1_000_000_000.0)
    decode_s = 0.0
    if gamma > 1:
        synchronize(device)
        decode_started = time.monotonic_ns()
        for _ in range(gamma - 1):
            output = approx_model_cache.generate(output, 1)
            synchronize(device)
        decode_s = max(0.0, (time.monotonic_ns() - decode_started) / 1_000_000_000.0)
        decode_s /= float(gamma - 1)
    return output, {
        "local_prefill_s": prefill_s,
        "local_decode_per_token_s": decode_s,
        "local_prefill_ms": prefill_s * 1000.0,
        "local_decode_per_token_ms": decode_s * 1000.0,
    }


def configure_spawn_executable() -> str | None:
    """Use an explicitly configured Python wrapper for spawn children.

    Python's spawn implementation otherwise derives the child executable from
    ``sys.executable``.  On node3 that would bypass the glibc-compatible
    ``PYTHON_BIN`` wrapper and launch the incompatible interpreter directly.
    With no override, leave multiprocessing's normal interpreter selection
    untouched.
    """

    configured = os.environ.get("PYTHON_BIN")
    if configured is None:
        return None
    if not configured:
        raise ValueError("PYTHON_BIN is set but empty; provide an executable Python wrapper")

    executable = os.path.abspath(os.path.expanduser(configured))
    if not os.path.isfile(executable):
        raise FileNotFoundError(
            f"PYTHON_BIN does not point to a regular file: {executable}"
        )
    if not os.access(executable, os.X_OK):
        raise PermissionError(f"PYTHON_BIN is not executable: {executable}")

    mp.set_executable(executable)
    return executable


class EdgeClient:
    """边缘端 HTTP 客户端，负责与云端 target 服务通信。"""

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        timing_priority: bool = False,
        clock_uncertainty_limit_ms: float = 10.0,
        clock_recovery_log: str | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.timing_priority = bool(timing_priority)
        self.session = requests.Session()
        # Timing updates must not share the in-flight /prefill connection.
        self.timing_session = requests.Session()
        # 避免 127.0.0.1 请求被环境变量代理劫持到其他服务。
        self.session.trust_env = False
        self.timing_session.trust_env = False
        # Defined as cloud_epoch - edge_epoch, measured by NTP-style probes.
        self.clock_offset_ns = 0
        self.clock_uncertainty_ns = 0
        self.clock_uncertainty_limit_ms = float(clock_uncertainty_limit_ms)
        self.clock_recovery_log = clock_recovery_log
        self._clock_lock = threading.RLock()
        self._last_clock_calibration = {}
        self._clock_calibrated_at = None
        self._clock_recovery_sequence = 0
        self._stats_lock = threading.Lock()
        self._request_bytes = 0
        self._response_bytes = 0
        self._rpc_count = 0
        self._clock_sync_rpc_count = 0
        self._clock_sync_time_s = 0.0
        self._clock_recovery_count = 0
        self._clock_pull_recovery_count = 0
        self._clock_upload_recovery_count = 0
        self._clock_recovery_time_s = 0.0
        self._clock_rejected_upload_http_s = 0.0

    def snapshot_transport_stats(self) -> Dict[str, Any]:
        """Return application-layer HTTP byte/RPC counters for this client."""

        with self._stats_lock:
            return {
                "request_bytes": int(self._request_bytes),
                "response_bytes": int(self._response_bytes),
                "rpc_count": int(self._rpc_count),
                "clock_sync_rpc_count": int(self._clock_sync_rpc_count),
                "clock_sync_time_s": float(self._clock_sync_time_s),
                "clock_recovery_count": int(self._clock_recovery_count),
                "clock_pull_recovery_count": int(self._clock_pull_recovery_count),
                "clock_upload_recovery_count": int(self._clock_upload_recovery_count),
                "clock_recovery_time_s": float(self._clock_recovery_time_s),
                "clock_rejected_upload_http_s": float(self._clock_rejected_upload_http_s),
            }

    def reset_transport_stats(self) -> None:
        """Exclude health checks and stateful warmups from measured requests."""

        with self._stats_lock:
            self._request_bytes = 0
            self._response_bytes = 0
            self._rpc_count = 0
            self._clock_sync_rpc_count = 0
            self._clock_sync_time_s = 0.0
            self._clock_recovery_count = 0
            self._clock_pull_recovery_count = 0
            self._clock_upload_recovery_count = 0
            self._clock_recovery_time_s = 0.0
            self._clock_rejected_upload_http_s = 0.0

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = self.session.post(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        with self._stats_lock:
            self._request_bytes += len(body)
            self._response_bytes += len(response.content)
            self._rpc_count += 1
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Keep validation diagnostics in the worker log.  Token payloads
            # and prompts are deliberately omitted from this error context.
            context = {name: payload[name] for name in (
                "task_id", "prefix_len", "gamma", "tail_only", "has_bridge_token",
                "edge_send_time_ns", "clock_offset_ns",
            ) if name in payload}
            if "draft_output" in payload:
                context["draft_output_length"] = len(payload["draft_output"])
            raise requests.HTTPError(
                f"{exc}; response_detail={response.text[:4096]}; "
                f"request_context={json.dumps(context, sort_keys=True)}",
                request=exc.request, response=response,
            ) from exc
        return response.json()

    def health(self) -> Dict[str, Any]:
        url = f"{self.server_url}/health"
        response = self.session.get(url, timeout=self.timeout)
        with self._stats_lock:
            self._response_bytes += len(response.content)
            self._rpc_count += 1
        response.raise_for_status()
        return response.json()

    def release_session(self, session_id: str) -> Dict[str, Any]:
        # A separate connection can cancel a still-running verify long poll.
        response = self.timing_session.post(
            self.server_url + "/session/release",
            json={"session_id": session_id}, timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "released":
            raise RuntimeError(f"session release failed: {payload}")
        return payload

    def init_session(self) -> str:
        resp = self._post("/session/init", {})
        return resp["session_id"]


    def synchronize_clock(self, samples: int = 5) -> dict[str, Any]:
        """Calibrate cloud-edge epoch offset with NTP-style probes.

        The minimum-RTT sample is used as the least-contended offset estimate;
        no RTT/2 value is ever used as an application upload/pull duration.
        """

        with self._clock_lock:
            started = time.monotonic()
            measurements = []
            calibration = {"samples": measurements, "accepted": False,
                "uncertainty_limit_ms": self.clock_uncertainty_limit_ms}
            self._last_clock_calibration = calibration
            try:
                for _ in range(max(1, int(samples))):
                    edge_before = time.time_ns()
                    # GET has an empty application body. Count every attempt,
                    # including a probe which later fails at transport level.
                    with self._stats_lock:
                        self._rpc_count += 1
                        self._clock_sync_rpc_count += 1
                    response = self.session.get(
                        f"{self.server_url}/clock/sync", timeout=self.timeout
                    )
                    edge_after = time.time_ns()
                    with self._stats_lock:
                        self._response_bytes += len(response.content)
                    response.raise_for_status()
                    cloud_time = int(response.json()["cloud_time_ns"])
                    midpoint = (edge_before + edge_after) // 2
                    measurements.append({"edge_before_ns": edge_before,
                        "edge_after_ns": edge_after, "cloud_time_ns": cloud_time,
                        "rtt_ns": edge_after - edge_before,
                        "clock_offset_ns": cloud_time - midpoint})
                    if edge_after < edge_before:
                        raise RuntimeError("edge epoch clock moved backwards during calibration")
                best = min(measurements, key=lambda item: item["rtt_ns"])
                uncertainty = int((best["rtt_ns"] + 1) // 2)
                calibration.update(clock_offset_ns=int(best["clock_offset_ns"]),
                    clock_uncertainty_ns=uncertainty, sample_count=len(measurements))
                if self.clock_uncertainty_limit_ms > 0 and uncertainty > self.clock_uncertainty_limit_ms * 1_000_000:
                    raise RuntimeError(
                        "clock calibration uncertainty exceeds configured limit: "
                        f"{uncertainty / 1_000_000:.3f}ms > {self.clock_uncertainty_limit_ms:.3f}ms"
                    )
                self.clock_offset_ns = calibration["clock_offset_ns"]
                self.clock_uncertainty_ns = uncertainty
                calibration["accepted"] = True
                self._clock_calibrated_at = time.monotonic()
                return calibration
            except Exception as exc:
                calibration["error"] = repr(exc)
                raise
            finally:
                calibration["elapsed_s"] = time.monotonic() - started
                with self._stats_lock:
                    self._clock_sync_time_s += calibration["elapsed_s"]

    def _clock_snapshot(self) -> dict[str, int]:
        with self._clock_lock:
            return {"clock_offset_ns": int(self.clock_offset_ns),
                "clock_uncertainty_ns": int(self.clock_uncertainty_ns)}

    def _recalibrate_recovery(self, event: dict) -> None:
        try:
            calibration = self.synchronize_clock(samples=5)
        except Exception:
            event["calibrations"].append(dict(self._last_clock_calibration))
            raise
        event["calibrations"].append(calibration)

    def _record_clock_recovery(self, event: dict, started: float, *, elapsed_s: float | None = None) -> None:
        event["recovery_elapsed_s"] = time.monotonic() - started if elapsed_s is None else elapsed_s
        event["recorded_epoch_ns"] = time.time_ns()
        with self._stats_lock:
            self._clock_recovery_sequence += 1
            event["sequence"] = self._clock_recovery_sequence
            self._clock_recovery_count += 1
            if event["kind"] == "negative_pull":
                self._clock_pull_recovery_count += 1
            else:
                self._clock_upload_recovery_count += 1
            self._clock_recovery_time_s += event["recovery_elapsed_s"]
            self._clock_rejected_upload_http_s += sum(
                item["http_elapsed_s"] for item in event.get("rejected_uploads", [])
            )
            if self.clock_recovery_log:
                parent = os.path.dirname(self.clock_recovery_log)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.clock_recovery_log, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        brief = {name: event[name] for name in ("kind", "route", "task_id", "status",
            "sequence", "recovery_elapsed_s") if name in event}
        brief["calibration_attempts"] = len(event["calibrations"])
        brief["audit_file"] = self.clock_recovery_log
        print("[CLOCK_RECOVERY] " + json.dumps(brief if self.clock_recovery_log else event), flush=True)

    def _ensure_clock(self) -> bool:
        # Bound calibration staleness; all probe costs remain in request E2E.
        if self._clock_calibrated_at is None or time.monotonic() - self._clock_calibrated_at > 30:
            self.synchronize_clock(samples=5)
            return True
        return False

    def _attach_pull(self, response: Dict[str, Any], edge_receive_time_ns: int,
                     *, route: str = "unknown", task_id: str | None = None) -> Dict[str, Any]:
        from src.clock_bounds import duration_bounds
        cloud_send = response.get("cloud_send_time_ns")
        if cloud_send is None:
            response.setdefault("pull_s", 0.0)
            response["pull_timing_available"] = False
            return response
        receipt, cloud_send = int(edge_receive_time_ns), int(cloud_send)
        if receipt <= 0 or cloud_send <= 0:
            raise RuntimeError("invalid cloud_send_time_ns in timing response")
        state = self._clock_snapshot()
        raw = receipt - cloud_send + state["clock_offset_ns"]
        event = None
        try:
            bounds = duration_bounds(raw, state["clock_uncertainty_ns"])
        except ValueError:
            started = time.monotonic()
            event = {"kind": "negative_pull", "route": route, "task_id": task_id,
                "edge_receive_time_ns": receipt, "cloud_send_time_ns": cloud_send,
                "model_rpc_retransmitted": False, "max_calibration_attempts": 3,
                "observations": [{"raw_pull_ns": raw, **state}], "calibrations": []}
            try:
                for _ in range(3):
                    self._recalibrate_recovery(event)
                    state = self._clock_snapshot()
                    raw = receipt - cloud_send + state["clock_offset_ns"]
                    event["observations"].append({"raw_pull_ns": raw, **state})
                    try:
                        bounds = duration_bounds(raw, state["clock_uncertainty_ns"])
                        break
                    except ValueError:
                        continue
                else:
                    raise RuntimeError("pull interval remains entirely negative after 3 clock calibrations")
                event["status"] = "recovered"
            except Exception as exc:
                event.update(status="failed", error=repr(exc))
                raise
            finally:
                self._record_clock_recovery(event, started)
            response["clock_recovery_pull_s"] = event["recovery_elapsed_s"]
            response["pull_is_posthoc_calibrated"] = True
        # The scheduler consumes an explicitly labelled conservative upper
        # estimate, not the signed midpoint or a clipped point measurement.
        response.update(pull_s=bounds["scheduler_seconds"],
            pull_ms=bounds["scheduler_seconds"] * 1000,
            pull_timing_available=True, pull_timing=bounds)
        if self.clock_recovery_log:
            path = os.path.join(os.path.dirname(self.clock_recovery_log), "timing-observations.jsonl")
            record = {"route": route, "task_id": task_id, "pull": bounds,
                "push": response.get("push_timing"), "clock": state,
                "edge_receive_time_ns": receipt, "cloud_send_time_ns": cloud_send,
                "posthoc_calibration": event is not None}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with self._stats_lock, open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return response

    def close(self) -> None:
        self.session.close()
        self.timing_session.close()

    def prefill_arrival(
        self,
        session_id: str,
        task_id: str,
        edge_send_time_ns: int | None = None,
    ) -> Dict[str, Any]:
        """Tell the cloud that a Prefill task has arrived.

        This request intentionally contains no prompt or draft token payload.
        The cloud records its own arrival timestamps and keeps the task pending
        until the first local draft round is uploaded.
        """

        if edge_send_time_ns is None:
            edge_send_time_ns = time.time_ns()
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "edge_send_time_ns": int(edge_send_time_ns),
            "clock_offset_ns": int(self.clock_offset_ns),
            "clock_uncertainty_ns": int(self.clock_uncertainty_ns),
        }
        return self._post("/prefill/arrival", payload)

    def prefill(
        self,
        session_id: str,
        task_id: str,
        draft_output: List[int],
        prefix_len: int,
        lag: float,
        current_time: float,
        gamma: int = 0,
        timing_required: bool = False,
        edge_send_time_ns: int | None = None,
        prompt_len: int | None = None,
        timing_estimate: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        if self._ensure_clock():
            edge_send_time_ns = None
        if edge_send_time_ns is None:
            edge_send_time_ns = time.time_ns()
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "draft_output": draft_output,
            "prefix_len": prefix_len,
            "prompt_len": int(prompt_len if prompt_len is not None else prefix_len),
            "lag": lag,
            "current_time": current_time,
            "gamma": int(gamma),
            "prefill_gamma": int(gamma),
            "timing_required": bool(timing_required),
            # Historical local estimates allow priority scoring without
            # waiting for the current first draft to complete.
            "timing_priority": bool(self.timing_priority and (timing_required or timing_estimate)),
            **(timing_estimate or {}),
            "edge_send_time_ns": int(edge_send_time_ns),
            "clock_offset_ns": int(self.clock_offset_ns),
            "clock_uncertainty_ns": int(self.clock_uncertainty_ns),
        }
        response = self._post("/prefill", payload)
        return self._attach_pull(response, time.time_ns(), route="/prefill", task_id=task_id)

    def prefill_timing(
        self,
        session_id: str,
        task_id: str,
        gamma: int,
        local_prefill_s: float,
        local_decode_per_token_s: float,
    ) -> Dict[str, Any]:
        """Upload local first-token/Decode measurements and unlock Prefill."""

        edge_send_time_ns = time.time_ns()
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "gamma": int(gamma),
            "local_prefill_s": float(local_prefill_s),
            "local_decode_per_token_s": float(local_decode_per_token_s),
            "T_i_p": float(local_prefill_s),
            "T_i_d": float(local_decode_per_token_s),
            "edge_send_time_ns": edge_send_time_ns,
            "clock_offset_ns": int(self.clock_offset_ns),
            "clock_uncertainty_ns": int(self.clock_uncertainty_ns),
        }
        # This call intentionally uses a different Session from /prefill.
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        upload_started = time.monotonic()
        response = self.timing_session.post(
            f"{self.server_url}/prefill/timing",
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        with self._stats_lock:
            self._request_bytes += len(body)
            self._response_bytes += len(response.content)
            self._rpc_count += 1
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "timing_ready":
            raise RuntimeError(f"prefill timing update failed: {result}")
        result["timing_upload_s"] = max(0.0, time.monotonic() - upload_started)
        result["timing_upload_ms"] = result["timing_upload_s"] * 1000.0
        return result

    def prefill_cancel(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """Release a timing-gated cloud Prefill after a local failure.

        This uses the independent control connection so it can run while the
        original ``/prefill`` request is blocked in its long poll.
        """

        payload = {
            "session_id": session_id,
            "task_id": task_id,
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = self.timing_session.post(
            f"{self.server_url}/prefill/cancel",
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        with self._stats_lock:
            self._request_bytes += len(body)
            self._response_bytes += len(response.content)
            self._rpc_count += 1
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "cancelled":
            raise RuntimeError(f"prefill cancel failed: {result}")
        return result

    def verify(
        self,
        session_id: str,
        task_id: str,
        draft_output: List[int],
        prefix_len: int,
        lag: float,
        current_time: float,
        gamma: int,
        transport_rtt: float = 0.0,
        tail_only: bool = False,
        has_bridge_token: bool = False,
        local_decode_per_token_s: float | None = None,
        last_pull_s: float | None = None,
        edge_send_time_ns: int | None = None,
        verify_method: str = "rejection",
        sampling_seed: int | None = None,
        draft_prob_block: str | None = None,
    ) -> tuple[Dict[str, Any], float]:
        if self._ensure_clock():
            edge_send_time_ns = None
        if edge_send_time_ns is None:
            edge_send_time_ns = time.time_ns()
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "draft_output": draft_output,
            "prefix_len": prefix_len,
            "lag": lag,
            "current_time": current_time,
            "gamma": gamma,
            "transport_rtt": transport_rtt,
            "tail_only": tail_only,
            "has_bridge_token": has_bridge_token,
            "timing_priority": self.timing_priority,
            "local_decode_per_token_s": local_decode_per_token_s,
            "last_pull_s": last_pull_s,
            "T_i_d": local_decode_per_token_s,
            "T_i_pull": last_pull_s,
            "edge_send_time_ns": int(edge_send_time_ns),
            "clock_offset_ns": int(self.clock_offset_ns),
            "clock_uncertainty_ns": int(self.clock_uncertainty_ns),
            "verify_method": str(verify_method),
            "sampling_seed": int(sampling_seed or 0),
            "draft_prob_block": draft_prob_block,
        }
        started = time.monotonic()
        event = None
        try:
            for attempt in range(4):
                transport_start = time.monotonic()
                try:
                    resp = self._post("/verify", payload)
                except requests.HTTPError as exc:
                    rejected_elapsed = time.monotonic() - transport_start
                    response = exc.response
                    detail = None
                    if response is not None and response.status_code == 422:
                        try:
                            detail = response.json().get("detail")
                        except (ValueError, AttributeError):
                            pass
                    # This machine-readable rejection is emitted exclusively
                    # before target admission. Any other error is not retried.
                    if not (isinstance(detail, dict)
                            and detail.get("code") == "negative_upload_pre_enqueue"
                            and detail.get("route") == "/verify"
                            and detail.get("enqueued") is False):
                        raise
                    if event is None:
                        event = {"kind": "negative_upload", "route": "/verify", "task_id": task_id,
                            "max_calibration_attempts": 3, "calibrations": [], "rejected_uploads": []}
                    event["rejected_uploads"].append({"server_detail": detail,
                        "http_elapsed_s": rejected_elapsed, **self._clock_snapshot(),
                        "request_clock_offset_ns": payload["clock_offset_ns"],
                        "edge_send_time_ns": payload["edge_send_time_ns"]})
                    if attempt == 3:
                        raise RuntimeError("negative upload persists after 3 clock calibrations") from exc
                    self._recalibrate_recovery(event)
                    # All model inputs, task IDs and sampling state stay frozen.
                    payload["edge_send_time_ns"] = time.time_ns()
                    payload.update(self._clock_snapshot())
                    continue
                edge_receive_time_ns = time.time_ns()
                measured_http_total = time.monotonic() - transport_start
                break
            if event is not None:
                event.update(status="recovered", successful_http_s=measured_http_total)
                overhead = time.monotonic() - started - measured_http_total
                self._record_clock_recovery(event, started, elapsed_s=overhead)
                resp["clock_recovery_upload_s"] = overhead
        except Exception as exc:
            if event is not None:
                event.update(status="failed", error=repr(exc))
                self._record_clock_recovery(event, started)
            raise
        # Only this successful attempt feeds the scheduler's transport RTT.
        # All retries/calibration are still included in the caller's real E2E.
        self._attach_pull(resp, edge_receive_time_ns, route="/verify", task_id=task_id)
        return resp, measured_http_total


class EdgeRunner(Decoding):
    """边缘端运行器：复用 Decoding 基类并执行 draft 侧循环。"""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.load_tokenizer()
        self.answer_trigger = "The answer is"
        self.gsm8k_prompt = self._create_gsm8k_demo_text(
            n_shot=8,
            cot_flag=True,
            answer_trigger=self.answer_trigger,
        )

    def _load_draft_model(self, model_path: str, device: str):
        """Load either a GPTQ draft or a regular Transformers checkpoint."""
        quant_config = os.path.join(model_path, "quantize_config.json")
        if os.path.exists(quant_config):
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "GPTQ draft requested, but auto_gptq is unavailable. "
                    "Install auto-gptq or provide a non-GPTQ Transformers checkpoint."
                ) from exc
            return AutoGPTQForCausalLM.from_quantized(
                model_path,
                device=device,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device},
            torch_dtype=resolve_dtype(getattr(self.args, "draft_dtype", "auto"), device),
            trust_remote_code=True,
        ).eval()

    def load_data(self):
        return

    def preprocess(self, input_text):
        return input_text.strip()

    def _encode_input_ids(self, tokenizer, input_text: str):
        if self.args.dataset != "mt_bench":
            return tokenizer.encode(input_text, return_tensors="pt")

        messages = [{"role": "user", "content": input_text}]
        template_args = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_args,
            )
        except TypeError:
            # Older compatible tokenizers may not expose Qwen3's
            # enable_thinking switch, but still require the chat template.
            input_ids = tokenizer.apply_chat_template(messages, **template_args)
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids

    def postprocess(self, input_text, output_text):
        return output_text

    def _create_gsm8k_demo_text(self, n_shot: int = 8, cot_flag: bool = True, answer_trigger: str = "The answer is") -> str:
        """Build GSM8K few-shot prompt text aligned with benchmark/eval_gsm8k.py."""
        question = []
        chain = []
        answer = []

        question.append(
            "There are 15 trees in the grove. "
            "Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today?"
        )
        chain.append(
            "There are 15 trees originally. "
            "Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6."
        )
        answer.append("6")

        question.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot?"
        )
        chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        answer.append("5")

        question.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total?"
        )
        chain.append(
            "Originally, Leah had 32 chocolates. "
            "Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39."
        )
        answer.append("39")

        question.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
            "has 12 lollipops. How many lollipops did Jason give to Denny?"
        )
        chain.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some "
            "to Denny. So he gave Denny 20 - 12 = 8."
        )
        answer.append("8")

        question.append(
            "Shawn has five toys. For Christmas, he got two toys each from his "
            "mom and dad. How many toys does he have now?"
        )
        chain.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and "
            "dad, then that is 4 more toys. 5 + 4 = 9."
        )
        answer.append("9")

        question.append(
            "There were nine computers in the server room. Five more computers "
            "were installed each day, from monday to thursday. "
            "How many computers are now in the server room?"
        )
        chain.append(
            "There were originally 9 computers. For each of 4 days, 5 more "
            "computers were added. So 5 * 4 = 20 computers were added. "
            "9 + 20 is 29."
        )
        answer.append("29")

        question.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
            "wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?"
        )
        chain.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, "
            "he had 58 - 23 = 35. After losing 2 more, "
            "he had 35 - 2 = 33 golf balls."
        )
        answer.append("33")

        question.append(
            "Olivia has $23. She bought five bagels for $3 each. "
            "How much money does she have left?"
        )
        chain.append(
            "Olivia had 23 dollars. "
            "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
            "So she has 23 - 15 dollars left. 23 - 15 is 8."
        )
        answer.append("8")

        demo_text = ""
        for i in range(min(n_shot, len(question))):
            if cot_flag:
                demo_text += (
                    "Q: "
                    + question[i]
                    + "\nA: "
                    + chain[i]
                    + " "
                    + answer_trigger
                    + " "
                    + answer[i]
                    + ".\n\n"
                )
            else:
                demo_text += (
                    "Question: "
                    + question[i]
                    + "\nAnswer: "
                    + answer_trigger
                    + " "
                    + answer[i]
                    + ".\n\n"
                )
        return demo_text

    def _preprocess_gsm8k(self, question_text: str) -> str:
        """Format GSM8K input in the same few-shot style as benchmark script."""
        return self.gsm8k_prompt + "Q: " + question_text + "\nA:"

    def _sample_prompt(self, sample: dict, proc_id: int, idx: int) -> tuple[str, str]:
        """Return the canonical prompt/task id for a raw or canonical sample."""

        if "sample_id" in sample and "prompt" in sample:
            return sample["prompt"].strip(), str(sample["sample_id"])
        if self.args.dataset == "gsm8k":
            return (
                self._preprocess_gsm8k(sample["question"].strip()),
                str(sample.get("task_id", f"gsm8k-{proc_id}-{idx}")),
            )
        if self.args.dataset == "humaneval":
            return (
                sample["prompt"].strip(),
                str(sample.get("task_id", f"humaneval-{proc_id}-{idx}")),
            )
        if self.args.dataset == "mgsm":
            return (
                sample["question"].strip(),
                str(sample.get("question_id", f"mgsm-{proc_id}-{idx}")),
            )
        return (
            sample["turns"][0].strip(),
            str(sample.get("task_id", f"mtbench-{proc_id}-{idx}")),
        )

    def _warmup_stateful_requests(self, client: EdgeClient, draft_model, tokenizer, samples, proc_id: int) -> None:
        """Exercise the real session/prefill/verify path before measurement.

        Warmups deliberately do not write metric records.  They are repeated
        per draft process so every CPU model copy is warmed, while the target
        receives the same stateful protocol traffic through the configured
        topology.  Any failed warmup raises and aborts the run rather than
        silently turning a requested warmup into a no-op.
        """

        count = int(getattr(self.args, "warmup_requests", 0))
        if count <= 0 or not samples:
            return
        warmup_sample = samples[0]
        input_text, task_id = self._sample_prompt(warmup_sample, proc_id, 0)
        input_ids = self._encode_input_ids(tokenizer, input_text).to(draft_model.device)
        for warmup_idx in range(count):
            seed_everything(100000 + proc_id * count + warmup_idx)
            session_id = client.init_session()
            try:
                prefix = input_ids.clone()
                response = client.prefill(
                    session_id=session_id,
                    task_id=f"warmup-{proc_id}-{warmup_idx}-{task_id}",
                    draft_output=prefix[0].tolist(),
                    prefix_len=prefix.shape[1],
                    lag=0.0,
                    current_time=time.time(),
                )
                if response.get("status") != "prefill_ok":
                    raise RuntimeError(f"warmup prefill failed: {response}")
                verify_method = str(getattr(self.args, "verify_method", "rejection")).lower()
                draft_top_k = int(getattr(self.args, "draft_top_k", 64))
                cache = KVCacheModel(
                    draft_model,
                    self.args.temp,
                    draft_top_k if verify_method == "rejection" else self.args.top_k,
                    0.0 if verify_method == "rejection" else self.args.top_p,
                    generator=make_generator(
                        int(self.args.seed) + proc_id * count + warmup_idx,
                        draft_model.device,
                    ),
                    exact_top_k=(verify_method == "rejection"),
                    bounded_probabilities=True,
                )
                cache.vocab_size = self.args.vocab_size
                if verify_method == "rejection":
                    draft, draft_rows = cache.generate_with_probs(prefix, 1)
                    sparse_ids, sparse_probs = sparse_block_from_dense(draft_rows, draft_top_k)
                    draft_prob_block = encode_sparse_block(sparse_ids, sparse_probs)
                else:
                    draft = cache.generate(prefix, 1)
                    draft_prob_block = None
                response, _ = client.verify(
                    session_id=session_id,
                    task_id=f"warmup-{proc_id}-{warmup_idx}-{task_id}",
                    draft_output=draft[0].tolist(),
                    prefix_len=prefix.shape[1],
                    lag=0.0,
                    current_time=time.time(),
                    gamma=1,
                    tail_only=False,
                    has_bridge_token=False,
                    verify_method=verify_method,
                    sampling_seed=int(self.args.seed) + proc_id * count + warmup_idx,
                    draft_prob_block=draft_prob_block,
                )
                if "final_token" not in response or "accepted" not in response:
                    raise RuntimeError(f"warmup verify failed: {response}")
            finally:
                client.release_session(session_id)


    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        ordered = sorted(float(v) for v in values)
        idx = int(round((len(ordered) - 1) * q))
        idx = max(0, min(len(ordered) - 1, idx))
        return ordered[idx]

    def _metrics_path(self, proc_id: int) -> str:
        return os.path.join(self.args.exp_name, f"edge_metrics_proc{proc_id}.jsonl")

    @staticmethod
    def _truncate_at_eos(prefix: torch.Tensor, eos_token_id: int, start_idx: int) -> tuple[torch.Tensor, bool]:
        """Truncate newly generated suffix at the first EOS token if present."""
        if eos_token_id is None:
            return prefix, False
        if start_idx >= prefix.shape[1]:
            return prefix, False
        suffix = prefix[0, start_idx:].tolist()
        if eos_token_id not in suffix:
            return prefix, False
        eos_offset = suffix.index(eos_token_id)
        keep_len = start_idx + eos_offset + 1
        return prefix[:, :keep_len], True

    @staticmethod
    def _truncate_on_humaneval_markers(generated_text: str) -> tuple[str, bool]:
        """
        HumanEval samples should stop at function solution.
        If model starts generating tests/runner blocks, truncate before them.
        """
        markers = [
            "\nif __name__ == \"__main__\":",
            "\n#tests",
            "\nimport unittest",
            "\nimport pytest",
            "\nclass Test",
        ]
        cut = -1
        for marker in markers:
            idx = generated_text.find(marker)
            if idx != -1 and (cut == -1 or idx < cut):
                cut = idx
        if cut == -1:
            return generated_text, False
        return generated_text[:cut].rstrip(), True

    @staticmethod
    def _truncate_on_gsm8k_markers(generated_text: str) -> tuple[str, bool]:
        """
        GSM8K few-shot prompt uses `Q:`/`A:` pattern.
        Stop once model starts generating the next `Q:` block.
        """
        match = re.search(r"\n\s*Q:", generated_text)
        if match is None:
            return generated_text, False
        cut = match.start()
        return generated_text[:cut].rstrip(), True

    @staticmethod
    def _is_degenerate_repeat(prefix: torch.Tensor, input_len: int, window: int = 64) -> bool:
        """
        Stop on obvious collapse mode: a long run of the same token.
        This prevents runaway outputs like '111111...' or repeated whitespace.
        """
        if prefix.shape[1] - input_len < window:
            return False
        tail = prefix[0, -window:]
        return bool((tail == tail[-1]).all().item())

    def _write_summary(self, wallclock_s: float) -> None:
        records: List[Dict[str, Any]] = []
        for path in sorted(glob.glob(os.path.join(self.args.exp_name, "edge_metrics_proc*.jsonl"))):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

        task_e2e = [float(r.get("task_e2e_ms", 0.0)) for r in records]
        request_e2e = [float(r.get("request_e2e_ms", 0.0)) for r in records]
        ttft = [float(r["ttft_ms"]) for r in records if r.get("ttft_ms") is not None]
        decode_ttft = [
            float(r["decode_ttft_ms"])
            for r in records
            if r.get("decode_ttft_ms") is not None
        ]
        tpot = [
            float(r["tpot_ms"])
            for r in records
            if int(r.get("generated_tokens", 0)) > 1 and r.get("tpot_ms") is not None
        ]
        arrival_lag = [float(r.get("arrival_lag_ms", 0.0)) for r in records]
        overlap_prefill_first_draft = [
            float(bool(r.get("overlap_prefill_first_draft", False))) for r in records
        ]
        prefill_ms = [float(r.get("prefill_ms", 0.0)) for r in records]
        first_draft_ms = [float(r.get("first_draft_ms", 0.0)) for r in records]
        local_prefill_ms = [float(r.get("local_prefill_ms", 0.0)) for r in records]
        local_decode_per_token_ms = [
            float(r.get("local_decode_per_token_ms", 0.0)) for r in records
        ]
        push_ms = [float(r.get("avg_push_ms", 0.0)) for r in records]
        pull_ms = [float(r.get("avg_pull_ms", 0.0)) for r in records]
        prefill_first_draft_overlap_ms = [
            float(r.get("prefill_first_draft_overlap_ms", 0.0)) for r in records
        ]
        prefill_wait_after_first_draft_ms = [
            float(r.get("prefill_wait_after_first_draft_ms", 0.0)) for r in records
        ]
        total_tokens = sum(int(r.get("generated_tokens", 0)) for r in records)
        max_generated_tokens_observed = max(
            (int(r.get("generated_tokens", 0)) for r in records),
            default=0,
        )
        total_accepted = sum(int(r.get("accepted_total", 0)) for r in records)
        total_drafted = sum(int(r.get("drafted_total", 0)) for r in records)
        q_tail = [float(r["q_tail"]) for r in records if r.get("q_tail") is not None]
        actual_arrivals = [float(r["actual_arrival_s"]) for r in records if "actual_arrival_s" in r]
        scheduled_arrivals = [
            float(r["scheduled_arrival_s"]) for r in records if "scheduled_arrival_s" in r
        ]
        completions = [float(r["completion_s"]) for r in records if "completion_s" in r]
        active_window_s = 0.0
        actual_arrival_span_s = 0.0
        scheduled_arrival_span_s = 0.0
        if actual_arrivals and completions:
            active_window_s = max(0.0, max(completions) - min(actual_arrivals))
        if len(actual_arrivals) > 1:
            actual_arrival_span_s = max(actual_arrivals) - min(actual_arrivals)
        if len(scheduled_arrivals) > 1:
            scheduled_arrival_span_s = max(scheduled_arrivals) - min(scheduled_arrivals)
        summary = {
            "profile": self.args.profile,
            "server_sched_mode": self.args.server_sched_mode,
            "enable_pipeline": bool(getattr(self.args, "enable_pipeline", True)),
            "enable_proactive_draft": bool(getattr(self.args, "enable_proactive_draft", True)),
            "overlap_prefill_first_draft": bool(
                getattr(self.args, "overlap_prefill_first_draft", False)
            ),
            "enable_latency_priority": bool(
                getattr(self.args, "enable_latency_priority", False)
            ),
            "num_drafts": int(self.args.num_drafts),
            "warmup_requests_per_process": int(getattr(self.args, "warmup_requests", 0)),
            "warmup_included_in_metrics": False,
            "arrival_distribution": self.args.arrival_distribution,
            "arrival_rate_rps": float(self.args.arrival_rate),
            "arrival_seed": int(self.args.arrival_seed),
            "ttft_definition": "request_start_to_first_output_token_visible",
            "decode_ttft_definition": "prefill_complete_to_first_output_token_visible",
            "tpot_definition": "(request_completion-first_token_time)/(output_tokens-1)",
            "scheduled_arrival_span_s": float(scheduled_arrival_span_s),
            "realized_scheduled_arrival_rate_rps": (
                float((len(scheduled_arrivals) - 1) / scheduled_arrival_span_s)
                if scheduled_arrival_span_s > 0
                else 0.0
            ),
            "actual_arrival_span_s": float(actual_arrival_span_s),
            "actual_model_start_rate_rps": (
                float((len(actual_arrivals) - 1) / actual_arrival_span_s)
                if actual_arrival_span_s > 0
                else 0.0
            ),
            "num_tasks": len(records),
            "wallclock_s": float(wallclock_s),
            "total_generated_tokens": int(total_tokens),
            "max_generated_tokens_observed": int(max_generated_tokens_observed),
            "system_tok_per_s": float(total_tokens / wallclock_s) if wallclock_s > 0 else 0.0,
            "active_window_s": float(active_window_s),
            "active_window_tok_per_s": float(total_tokens / active_window_s) if active_window_s > 0 else 0.0,
            "accept_rate": float(total_accepted / total_drafted) if total_drafted > 0 else 0.0,
            "q_tail_avg": float(statistics.mean(q_tail)) if q_tail else None,
            "task_e2e_ms_avg": float(statistics.mean(task_e2e)) if task_e2e else 0.0,
            "task_e2e_ms_p50": self._percentile(task_e2e, 0.50),
            "task_e2e_ms_p90": self._percentile(task_e2e, 0.90),
            "task_e2e_ms_p95": self._percentile(task_e2e, 0.95),
            "request_e2e_ms_avg": float(statistics.mean(request_e2e)) if request_e2e else 0.0,
            "request_e2e_ms_p50": self._percentile(request_e2e, 0.50),
            "request_e2e_ms_p90": self._percentile(request_e2e, 0.90),
            "request_e2e_ms_p95": self._percentile(request_e2e, 0.95),
            "request_e2e_ms_p99": self._percentile(request_e2e, 0.99),
            "ttft_ms_avg": float(statistics.mean(ttft)) if ttft else 0.0,
            "ttft_ms_p50": self._percentile(ttft, 0.50),
            "ttft_ms_p90": self._percentile(ttft, 0.90),
            "ttft_ms_p95": self._percentile(ttft, 0.95),
            "ttft_ms_p99": self._percentile(ttft, 0.99),
            "decode_ttft_ms_avg": float(statistics.mean(decode_ttft)) if decode_ttft else 0.0,
            "decode_ttft_ms_p50": self._percentile(decode_ttft, 0.50),
            "decode_ttft_ms_p90": self._percentile(decode_ttft, 0.90),
            "decode_ttft_ms_p95": self._percentile(decode_ttft, 0.95),
            "decode_ttft_ms_p99": self._percentile(decode_ttft, 0.99),
            "tpot_ms_avg": float(statistics.mean(tpot)) if tpot else 0.0,
            "tpot_ms_p50": self._percentile(tpot, 0.50),
            "tpot_ms_p90": self._percentile(tpot, 0.90),
            "tpot_ms_p95": self._percentile(tpot, 0.95),
            "tpot_ms_p99": self._percentile(tpot, 0.99),
            "arrival_lag_ms_avg": float(statistics.mean(arrival_lag)) if arrival_lag else 0.0,
            "arrival_lag_ms_p95": self._percentile(arrival_lag, 0.95),
            "overlap_prefill_first_draft_avg": (
                float(statistics.mean(overlap_prefill_first_draft))
                if overlap_prefill_first_draft
                else 0.0
            ),
            "prefill_ms_avg": float(statistics.mean(prefill_ms)) if prefill_ms else 0.0,
            "first_draft_ms_avg": (
                float(statistics.mean(first_draft_ms)) if first_draft_ms else 0.0
            ),
            "local_prefill_ms_avg": (
                float(statistics.mean(local_prefill_ms)) if local_prefill_ms else 0.0
            ),
            "local_decode_per_token_ms_avg": (
                float(statistics.mean(local_decode_per_token_ms))
                if local_decode_per_token_ms
                else 0.0
            ),
            "push_ms_avg": float(statistics.mean(push_ms)) if push_ms else 0.0,
            "pull_ms_avg": float(statistics.mean(pull_ms)) if pull_ms else 0.0,
            "prefill_first_draft_overlap_ms_avg": (
                float(statistics.mean(prefill_first_draft_overlap_ms))
                if prefill_first_draft_overlap_ms
                else 0.0
            ),
            "prefill_wait_after_first_draft_ms_avg": (
                float(statistics.mean(prefill_wait_after_first_draft_ms))
                if prefill_wait_after_first_draft_ms
                else 0.0
            ),
        }
        summary_path = os.path.join(self.args.exp_name, "edge_metrics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        self.color_print(f"[METRICS] wrote summary: {summary_path}", 2)

    def eval(self):
        configure_spawn_executable()
        if torch.cuda.is_available():
            torch.cuda.init()
        torch.multiprocessing.set_start_method("spawn", force=True)
        existing_metrics = glob.glob(
            os.path.join(self.args.exp_name, "edge_metrics_proc*.jsonl")
        )
        if existing_metrics:
            raise FileExistsError(
                "refusing to overwrite existing Edge metrics; use a new exp_name: "
                + ", ".join(sorted(existing_metrics))
            )

        # 启动多个边缘 draft 进程，每个进程独立 session_id
        wallclock_start = time.monotonic()
        processes = []
        arrival_barrier = None
        arrival_start_time = None
        ctx = mp.get_context("spawn")
        if self.args.arrival_distribution == "poisson" and self.args.num_drafts > 1:
            arrival_barrier = ctx.Barrier(self.args.num_drafts)
            arrival_start_time = ctx.Value("d", 0.0)
        for proc_id in range(self.args.num_drafts):
            proc = ctx.Process(
                target=self.run_draft_process_http,
                args=(self.tokenizer, proc_id, arrival_barrier, arrival_start_time),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

        self._write_summary(max(0.0, time.monotonic() - wallclock_start))
        failed = [proc.exitcode for proc in processes if proc.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} draft process(es) failed with exit codes {failed}")

    def _resolve_data_file(self) -> str:
        if self.args.dataset_file:
            return self.args.dataset_file
        if self.args.dataset == "humaneval":
            default_file = "humaneval.jsonl"
        elif self.args.dataset == "gsm8k":
            default_file = "gsm8k.jsonl"
        elif self.args.dataset == "mgsm":
            default_file = "mgsm.jsonl"
        elif self.args.dataset == "mt_bench":
            default_file = "mt_bench.jsonl"
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        if os.path.isdir(self.args.data_path):
            return os.path.join(self.args.data_path, default_file)
        return self.args.data_path

    @torch.no_grad()
    def run_draft_process_http(
        self,
        tokenizer,
        proc_id: int,
        arrival_barrier=None,
        arrival_start_time=None,
    ):
        """Run one draft worker and always close each task's HTTP executor."""

        executor_holder: Dict[str, Any] = {}
        try:
            return self._run_draft_process_http_impl(
                tokenizer,
                proc_id,
                arrival_barrier,
                arrival_start_time,
                executor_holder,
            )
        finally:
            executor = executor_holder.get("executor")
            if executor is not None:
                _shutdown_executor(executor, wait=True)
            client = executor_holder.get("client")
            if client is not None:
                client.close()

    def _run_draft_process_http_impl(
        self,
        tokenizer,
        proc_id: int,
        arrival_barrier=None,
        arrival_start_time=None,
        executor_holder: Dict[str, Any] | None = None,
    ):
        # Support CPU device for draft workers
        use_cpu = getattr(self.args, "edge_use_cpu", False)
        if use_cpu:
            device = "cpu"
            configure_torch_threads(getattr(self.args, "edge_threads", None))
            self.color_print(f"[Edge {proc_id}] loading draft model on CPU", 3)
        else:
            gpu_id = (proc_id % max(1, self.args.edge_gpus)) + self.args.edge_gpu_start
            device = f"cuda:{gpu_id}"
            self.color_print(f"[Edge {proc_id}] loading draft model on {device}", 3)

        draft_model = self._load_draft_model(self.args.draft_model, device)

        client = EdgeClient(
            self.args.server_url,
            timeout=self.args.request_timeout,
            timing_priority=bool(
                getattr(self.args, "enable_latency_priority", False)
                and self.args.server_sched_mode == "fastsd"
            ),
            clock_uncertainty_limit_ms=float(getattr(self.args, "timing_max_clock_uncertainty_ms", 10.0)),
            clock_recovery_log=os.path.join(self.args.exp_name, "clock-recovery-proc%d.jsonl" % proc_id),
        )
        if executor_holder is not None:
            executor_holder["client"] = client
        health = client.health()
        if health.get("status") not in {"ok", "healthy"}:
            raise RuntimeError(f"Cloud service health check failed: {health}")
        timing_protocol = bool(
            getattr(self.args, "enable_latency_priority", False)
            and self.args.server_sched_mode == "fastsd"
        )
        if timing_protocol:
            clock_info = client.synchronize_clock(samples=5)
            max_uncertainty_ms = float(
                getattr(self.args, "timing_max_clock_uncertainty_ms", 10.0)
            )
            if max_uncertainty_ms > 0 and clock_info["clock_uncertainty_ns"] > max_uncertainty_ms * 1_000_000:
                raise RuntimeError(
                    "clock calibration uncertainty exceeds configured limit: "
                    f"{clock_info['clock_uncertainty_ns'] / 1_000_000:.3f}ms > {max_uncertainty_ms:.3f}ms"
                )

        data_file = self._resolve_data_file()
        with open(data_file, "r") as f:
            samples = [json.loads(line) for line in f.readlines()]

        self._warmup_stateful_requests(client, draft_model, tokenizer, samples, proc_id)
        client.reset_transport_stats()

        indexed_samples = shard_samples(
            samples,
            num_shards=self.args.num_drafts,
            shard_id=proc_id,
            max_items=self.args.max_tasks_per_draft,
        )
        arrival_offsets = None
        arrival_origin = None
        if self.args.arrival_distribution == "poisson":
            if samples and all("scheduled_arrival_s" in sample for sample in samples):
                arrival_offsets = [float(sample["scheduled_arrival_s"]) for sample in samples]
            else:
                arrival_offsets = poisson_arrival_offsets(
                    len(samples), self.args.arrival_rate, self.args.arrival_seed
                )
            if arrival_barrier is not None:
                arrival_barrier.wait()
                if proc_id == 0:
                    arrival_start_time.value = time.monotonic()
                arrival_barrier.wait()
                arrival_origin = float(arrival_start_time.value)
            else:
                arrival_origin = time.monotonic()
        seed_everything(42 + proc_id)

        prefill_timing_estimate = {}
        for idx, (global_idx, sample) in enumerate(indexed_samples):
            scheduled_arrival_s = 0.0
            if arrival_offsets is not None:
                scheduled_arrival_s = float(arrival_offsets[global_idx])
                remaining_s = scheduled_arrival_s - (time.monotonic() - arrival_origin)
                if remaining_s > 0:
                    time.sleep(remaining_s)
                actual_arrival_s = max(0.0, time.monotonic() - arrival_origin)
            else:
                actual_arrival_s = 0.0
            arrival_lag_ms = max(0.0, (actual_arrival_s - scheduled_arrival_s) * 1000.0)
            request_start = time.monotonic()
            transport_before = client.snapshot_transport_stats()
            session_id = client.init_session()
            try:
                sampling_seed = int(self.args.seed) + int(global_idx)
                verify_method = str(getattr(self.args, "verify_method", "rejection")).lower()
                draft_top_k = int(getattr(self.args, "draft_top_k", 64))
                draft_generator = make_generator(sampling_seed, draft_model.device)
                draft_cache_top_k = draft_top_k if verify_method == "rejection" else self.args.top_k
                draft_cache_top_p = 0.0 if verify_method == "rejection" else self.args.top_p
                approx_model_cache = KVCacheModel(
                    draft_model,
                    self.args.temp,
                    draft_cache_top_k,
                    draft_cache_top_p,
                    generator=draft_generator,
                    exact_top_k=(verify_method == "rejection"),
                    bounded_probabilities=True,
                )
                approx_model_cache.vocab_size = self.args.vocab_size

                input_text, task_id = self._sample_prompt(sample, proc_id, idx)

                # input_text = 'def fib(n'  # for debug
                input_ids = self._encode_input_ids(tokenizer, input_text).to(draft_model.device)
                prefix = input_ids.clone()
                max_len = input_ids.shape[1] + self.args.max_tokens
                first_token_time = None
                pipeline_enabled = bool(getattr(self.args, "enable_pipeline", True))
                proactive_enabled = bool(getattr(self.args, "enable_proactive_draft", True))
                overlap_prefill_first_draft = bool(timing_protocol or getattr(self.args, "overlap_prefill_first_draft", False))
                final_token = None
                reused_pending_tokens: List[int] = []
                reused_pending_prob_rows: List[torch.Tensor] = []
                current_gamma = min(int(self.args.gamma), int(self.args.max_tokens))
                verify_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                if executor_holder is not None:
                    executor_holder["executor"] = verify_executor
                # ``current_time`` is a cloud scheduler protocol field (epoch time);
                # all local duration/ordering measurements use ``monotonic``.
                prefill_current_time = time.time()
                prefill_send_time_ns = time.time_ns()
                prefill_call = lambda: client.prefill(
                    session_id=session_id,
                    task_id=task_id,
                    draft_output=prefix[0].tolist(),
                    prefix_len=prefix.shape[1],
                    lag=0.0,
                    current_time=prefill_current_time,
                    gamma=current_gamma,
                    timing_required=False,
                    timing_estimate=prefill_timing_estimate if timing_protocol else None,
                    edge_send_time_ns=prefill_send_time_ns,
                )
                # Upload the prompt immediately and generate the first draft
                # once. Verify waits until both independent operations finish.
                first_round_x = None
                first_round_draft_ms = 0.0
                prefill_ms = 0.0
                prefill_first_draft_overlap_ms = 0.0
                prefill_wait_after_first_draft_ms = 0.0
                task_start = None
                first_round_draft_started_at = None
                first_round_timings = {}
                if prefix.shape[1] < max_len:
                    if timing_protocol:
                        first_round_x, prefill_resp, first_round_timings = _run_first_draft_with_prefill(
                            verify_executor, prefill_call,
                            lambda: _timed_local_draft_generate(approx_model_cache, prefix, current_gamma),
                            overlap=True,
                            timeout=float(getattr(self.args, "request_timeout", 0.0)),
                            cancel_call=lambda: client.release_session(session_id),
                        )
                        prefill_timing_estimate = {
                            key: float(first_round_timings[key])
                            for key in ("local_prefill_s", "local_decode_per_token_s")
                            if key in first_round_timings
                        }
                    else:
                        first_round_x, prefill_resp, first_round_timings = _run_first_draft_with_prefill(
                            verify_executor,
                            prefill_call,
                            lambda: approx_model_cache.generate(prefix, current_gamma),
                            overlap=overlap_prefill_first_draft,
                            timeout=float(getattr(self.args, "request_timeout", 0.0)),
                        )
                    if timing_protocol and "cloud_send_time_ns" not in prefill_resp:
                        raise RuntimeError("FastSD Prefill response omitted cloud_send_time_ns")
                    prefill_ms = first_round_timings["prefill_ms"]
                    first_round_draft_ms = first_round_timings["first_draft_ms"]
                    prefill_first_draft_overlap_ms = first_round_timings[
                        "prefill_first_draft_overlap_ms"
                    ]
                    prefill_wait_after_first_draft_ms = first_round_timings[
                        "prefill_wait_after_first_draft_ms"
                    ]
                    first_round_draft_started_at = first_round_timings["draft_started_at"]
                    task_start = first_round_timings["prefill_completed_at"]
                else:
                    # Keep the session protocol valid even when no decode round is
                    # requested.  This branch normally does not occur in benchmark
                    # runs (max_tokens is positive), but it avoids an unnecessary
                    # local draft call for a zero-length output.
                    prefill_start = time.monotonic()
                    try:
                        prefill_resp = _validate_prefill_response(prefill_call())
                    except BaseException:
                        _shutdown_executor(verify_executor, wait=True)
                        raise
                    prefill_completed_at = time.monotonic()
                    prefill_ms = max(0.0, (prefill_completed_at - prefill_start) * 1000.0)
                    task_start = prefill_completed_at
                last_transport_rtt = 0.0
                last_pull_s = float(prefill_resp.get("pull_s", 0.0))
                first_prefill_pull_s = last_pull_s
                first_prefill_push_s = float(
                    first_round_timings.get("push_s", prefill_resp.get("push_s", 0.0))
                    if isinstance(first_round_timings, dict)
                    else prefill_resp.get("push_s", 0.0)
                )
                ahead_control = AheadController()
                ahead_generated_tokens = ahead_reused_tokens = 0
                ahead_forward_s = response_consumption_delay_s = 0.0
                ahead_forward_per_token_s = 0.0
                previous_verify_wait_s = 0.0
                reuse_hit_rounds = 0
                reuse_miss_rounds = 0
                reuse_miss_not_full_accept_rounds = 0
                sum_round_ms = 0.0
                sum_draft_ms = 0.0
                sum_wait_ms = 0.0
                sum_http_total_ms = 0.0
                sum_cloud_total_ms = 0.0
                sum_verify_ms = 0.0
                sum_transport_rtt_ms = 0.0
                sum_push_ms = 0.0
                sum_pull_ms = 0.0
                accepted_total = 0
                drafted_total = 0
                rounds = 0
                first_round_pending = first_round_x is not None
                while prefix.shape[1] < max_len:
                    prefix_len = prefix.shape[1]
                    approx_model_cache.retain_probabilities(prefix_len - 1)
                    round_start = time.monotonic()
                    req_gamma = min(current_gamma if pipeline_enabled else int(self.args.gamma), max_len - prefix_len)
                    local_decode_per_token_s = None

                    # 若上一轮复用 token 不足 gamma，这里补齐到 gamma 后再发起验证。
                    if first_round_pending:
                        # The helper already produced this exact first-round state;
                        # never regenerate it after waiting for Prefill.
                        x = first_round_x
                        first_round_pending = False
                        round_start = float(first_round_draft_started_at)
                        draft_elapsed = first_round_draft_ms / 1000.0
                        local_decode_per_token_s = float(
                            first_round_timings.get(
                                "local_decode_per_token_s",
                                draft_elapsed / max(1, req_gamma),
                            )
                        )
                        if verify_method == "rejection":
                            start = max(0, prefix_len - 1)
                            draft_rows = approx_model_cache.probability_rows(start, start + req_gamma)[0].detach().float()
                        else:
                            draft_rows = None
                    elif reused_pending_tokens:
                        draft_started = time.monotonic()
                        reuse_count = min(len(reused_pending_tokens), req_gamma)
                        ahead_reused_tokens += reuse_count
                        reuse_tensor = torch.tensor(
                            [reused_pending_tokens[:reuse_count]],
                            device=prefix.device,
                            dtype=prefix.dtype,
                        )
                        x = torch.cat((prefix, reuse_tensor), dim=1)
                        approx_model_cache.rollback(min(
                            approx_model_cache._past_key_values.get_seq_length(), x.shape[1] - 1
                        ))
                        if reuse_count < req_gamma:
                            if verify_method == "rejection":
                                x, generated_rows = approx_model_cache.generate_with_probs(
                                    x, req_gamma - reuse_count
                                )
                                draft_rows = torch.cat(
                                    [
                                        row.to(x.device).reshape(1, -1)
                                        for row in reused_pending_prob_rows[:reuse_count]
                                    ]
                                    + [generated_rows],
                                    dim=0,
                                )
                            else:
                                x = approx_model_cache.generate(x, req_gamma - reuse_count)
                                draft_rows = None
                        elif verify_method == "rejection":
                            draft_rows = torch.cat(
                                [
                                    row.to(x.device).reshape(1, -1)
                                    for row in reused_pending_prob_rows[:reuse_count]
                                ],
                                dim=0,
                            )
                        else:
                            draft_rows = None
                        draft_elapsed = max(0.0, time.monotonic() - draft_started)
                        local_decode_per_token_s = draft_elapsed / max(1, req_gamma - reuse_count)
                    else:
                        draft_started = time.monotonic()
                        if verify_method == "rejection":
                            x, draft_rows = approx_model_cache.generate_with_probs(
                                prefix, req_gamma
                            )
                        else:
                            x = approx_model_cache.generate(prefix, req_gamma)
                            draft_rows = None
                        synchronize(getattr(draft_model, "device", "cpu"))
                        draft_elapsed = max(0.0, time.monotonic() - draft_started)
                        local_decode_per_token_s = draft_elapsed / max(1, req_gamma)

                    draft_prob_block = None
                    if verify_method == "rejection":
                        if draft_rows is None or int(draft_rows.shape[0]) != req_gamma:
                            raise RuntimeError(
                                "draft probability rows do not match the requested gamma"
                            )
                        sparse_ids, sparse_probs = sparse_block_from_dense(
                            draft_rows, draft_top_k
                        )
                        draft_prob_block = encode_sparse_block(sparse_ids, sparse_probs)

                    has_bridge_token = pipeline_enabled and (final_token is not None)
                    if pipeline_enabled:
                        pending_tokens = x[0, prefix_len:prefix_len + req_gamma].tolist()
                        if has_bridge_token:
                            # 发送：上一轮 final_token + 本轮 gamma 个 draft token
                            payload_tokens = [final_token] + pending_tokens
                        else:
                            # 首轮无上一轮 final_token，仅发送本轮 gamma 个 draft token
                            payload_tokens = pending_tokens
                    else:
                        # vanilla verify: send full prefix + gamma draft tokens.
                        payload_tokens = x[0, :prefix_len + req_gamma].tolist()
                    if getattr(self.args, "debug_verify_tokens", False):
                        debug_tail = 16
                        self.color_print(
                            f"[VERIFY-EDGE-SEND][pid={proc_id}][session={session_id}] "
                            f"prefix_len={prefix_len} has_bridge_token={has_bridge_token} "
                            f"tail_ids={payload_tokens[-debug_tail:]}",
                            3,
                        )

                    verify_future = _submit_http_call(
                        verify_executor,
                        _verify_with_ready_time,
                        client.verify,
                        session_id=session_id,
                        task_id=task_id,
                        draft_output=payload_tokens,
                        prefix_len=prefix_len,
                        # For the overlapped first round, this is draft compute only;
                        # any wait for Prefill happened before Verify submission.
                        lag=draft_elapsed if first_round_draft_ms > 0 and rounds == 0 else max(
                            0.0, time.monotonic() - round_start
                        ),
                        current_time=time.time(),
                        gamma=req_gamma,
                        transport_rtt=last_transport_rtt,
                        tail_only=pipeline_enabled,
                        has_bridge_token=has_bridge_token,
                        local_decode_per_token_s=local_decode_per_token_s,
                        last_pull_s=last_pull_s,
                        verify_method=verify_method,
                        sampling_seed=sampling_seed,
                        draft_prob_block=draft_prob_block,
                    )
                    if rounds > 0 or first_round_draft_ms <= 0:
                        draft_elapsed = max(0.0, time.monotonic() - round_start)
                    if getattr(self.args, "debug_pipeline", False):
                        self.color_print(
                            f"[PIPELINE-EDGE][pid={proc_id}][session={session_id}] "
                            f"send req_gamma={req_gamma} draft_ms={draft_elapsed*1000:.2f} "
                            f"transport_rtt_prev={last_transport_rtt*1000:.2f}ms "
                            f"prefix_len={prefix_len} payload_tokens={len(payload_tokens)}",
                            3,
                        )

                    # Proactive length is independently capped and benefit-gated.
                    overlap_tokens: List[int] = []
                    overlap_prob_rows: List[torch.Tensor] = []
                    overlap_prefix = x
                    max_overlap_tokens = max(0, min(int(getattr(self.args, "ahead_max_tokens", 4)), max_len - x.shape[1]))
                    ahead_started = time.monotonic()
                    wait_start = time.monotonic()
                    wait_draft_steps = 0
                    if getattr(self.args, "debug_verify_tokens", False):
                        self.color_print(
                            f"[VERIFY-EDGE-WAIT-START][pid={proc_id}][session={session_id}] "
                            f"prefix_len={prefix_len} max_overlap_tokens={max_overlap_tokens}",
                            3,
                        )
                    if proactive_enabled:
                        while not verify_future.done() and ahead_control.allow(
                            steps=len(overlap_tokens), cap=max_overlap_tokens,
                            elapsed_s=time.monotonic() - ahead_started,
                            expected_wait_s=previous_verify_wait_s,
                            draft_s=ahead_forward_per_token_s,
                        ):
                            step_started = time.monotonic()
                            if verify_method == "rejection":
                                overlap_prefix, overlap_rows = approx_model_cache.generate_with_probs(
                                    overlap_prefix, 1
                                )
                                overlap_prob_rows.append(overlap_rows[0].detach().float())
                            else:
                                overlap_prefix = approx_model_cache.generate(overlap_prefix, 1)
                            overlap_tokens.append(int(overlap_prefix[0, -1].item()))
                            wait_draft_steps += 1
                            step_s = time.monotonic() - step_started
                            ahead_forward_s += step_s
                            ahead_forward_per_token_s = (0.8 * ahead_forward_per_token_s + 0.2 * step_s
                                                         if ahead_forward_per_token_s else step_s)
                            if getattr(self.args, "debug_verify_tokens", False) and (
                                wait_draft_steps == 1
                                or wait_draft_steps % 8 == 0
                                or wait_draft_steps == max_overlap_tokens
                            ):
                                self.color_print(
                                    f"[VERIFY-EDGE-WAIT-DRAFT][pid={proc_id}][session={session_id}] "
                                    f"drafted_while_wait={wait_draft_steps} latest_token={overlap_tokens[-1]} "
                                    f"verify_done={verify_future.done()}",
                                    3,
                                )
                    else:
                        while not verify_future.done():
                            time.sleep(0.0005)

                    if getattr(self.args, "debug_verify_tokens", False):
                        wait_ms = (time.monotonic() - wait_start) * 1000.0
                        debug_tail = 8
                        self.color_print(
                            f"[VERIFY-EDGE-WAIT-END][pid={proc_id}][session={session_id}] "
                            f"wait_ms={wait_ms:.2f} drafted_while_wait={wait_draft_steps} "
                            f"verify_done={verify_future.done()} overlap_tail={overlap_tokens[-debug_tail:]}",
                            3,
                        )

                    verify_resp, measured_http_total = verify_future.result()
                    response_consumed_at = time.monotonic()
                    response_delay = max(0.0, response_consumed_at - verify_resp.get("_response_ready_monotonic", response_consumed_at))
                    response_consumption_delay_s += response_delay
                    previous_verify_wait_s = measured_http_total
                    ahead_generated_tokens += len(overlap_tokens)
                    if timing_protocol and "cloud_send_time_ns" not in verify_resp:
                        raise RuntimeError("FastSD Verify response omitted cloud_send_time_ns")
                    if "pull_s" in verify_resp:
                        last_pull_s = float(verify_resp["pull_s"])
                    verify_ms = float(verify_resp.get("verify_ms", 0.0))
                    cloud_total_ms = float(verify_resp.get("cloud_total_ms", 0.0))
                    push_s = float(verify_resp.get("push_s", 0.0))
                    pull_s = float(verify_resp.get("pull_s", last_pull_s))
                    # A purer transport estimate: subtract cloud-side service time from end-to-end HTTP time.
                    last_transport_rtt = max(0.0, measured_http_total - cloud_total_ms / 1000.0)
                    round_elapsed = max(0.0, time.monotonic() - round_start)
                    wait_elapsed_ms = max(0.0, (time.monotonic() - wait_start) * 1000.0)
                    sum_round_ms += round_elapsed * 1000.0
                    sum_draft_ms += draft_elapsed * 1000.0
                    sum_wait_ms += wait_elapsed_ms
                    sum_http_total_ms += measured_http_total * 1000.0
                    sum_cloud_total_ms += cloud_total_ms
                    sum_verify_ms += verify_ms
                    sum_transport_rtt_ms += last_transport_rtt * 1000.0
                    sum_push_ms += push_s * 1000.0
                    sum_pull_ms += pull_s * 1000.0
                    rounds += 1

                    accepted = int(verify_resp["accepted"])
                    accepted_cnt = accepted - prefix_len
                    accepted_total += max(0, accepted_cnt)
                    drafted_total += max(1, req_gamma)
                    final_token = int(verify_resp["final_token"])
                    if pipeline_enabled and "suggested_gamma" in verify_resp:
                        current_gamma = int(verify_resp["suggested_gamma"])
                    if getattr(self.args, "debug_pipeline", False):
                        self.color_print(
                            f"[PIPELINE-EDGE][pid={proc_id}][session={session_id}] "
                            f"recv accepted={accepted_cnt}/{req_gamma} round_ms={round_elapsed*1000:.2f} "
                            f"transport_rtt={last_transport_rtt*1000:.2f}ms "
                            f"http_total_ms={measured_http_total*1000:.2f} cloud_total_ms={cloud_total_ms:.2f} "
                            f"verify_ms={verify_ms:.2f} "
                            f"suggested_gamma={current_gamma}",
                            3,
                        )
                    if getattr(self.args, "debug_verify_tokens", False):
                        self.color_print(
                            f"[VERIFY-EDGE-RESP][pid={proc_id}][session={session_id}] "
                            f"accepted={accepted} final_token={final_token} req_gamma={req_gamma} suggested_gamma={current_gamma} "
                            f"drafted_while_wait={wait_draft_steps}",
                            3,
                        )
                    final_token_tensor = torch.tensor([[final_token]], device=x.device, dtype=x.dtype)

                    prefix = torch.cat((x[:, :accepted], final_token_tensor), dim=1)
                    prefix, hit_eos = self._truncate_at_eos(prefix, tokenizer.eos_token_id, prefix_len)
                    prefix = prefix[:, :max_len]
                    if first_token_time is None and prefix.shape[1] > input_ids.shape[1]:
                        first_token_time = time.monotonic()
                    reused_pending_tokens = []
                    reused_pending_prob_rows = []

                    if proactive_enabled and accepted_cnt == req_gamma and overlap_tokens:
                        if overlap_tokens[0] == final_token:
                            reuse_limit = min(current_gamma, max(0, max_len - prefix.shape[1]))
                            reused_pending_tokens = overlap_tokens[1 : 1 + reuse_limit]
                            if verify_method == "rejection":
                                reused_pending_prob_rows = [
                                    row.detach().float()
                                    for row in overlap_prob_rows[1 : 1 + len(reused_pending_tokens)]
                                ]
                            reuse_hit_rounds += 1
                            if getattr(self.args, "debug_verify_tokens", False):
                                self.color_print(
                                    f"[VERIFY-EDGE-REUSE-HIT][pid={proc_id}][session={session_id}] "
                                    f"final_token={final_token} overlap_first={overlap_tokens[0]} "
                                    f"reused_count={len(reused_pending_tokens)}",
                                    2,
                                )
                        else:
                            reuse_miss_rounds += 1
                            if getattr(self.args, "debug_verify_tokens", False):
                                self.color_print(
                                    f"[VERIFY-EDGE-DROP][pid={proc_id}][session={session_id}] "
                                    f"final_token={final_token} overlap_first={overlap_tokens[0]}",
                                    3,
                                )
                    elif proactive_enabled and accepted_cnt == req_gamma:
                        reuse_miss_rounds += 1
                        if getattr(self.args, "debug_verify_tokens", False):
                            self.color_print(
                                f"[VERIFY-EDGE-REUSE-MISS][pid={proc_id}][session={session_id}] "
                                f"reason=overlap_empty final_token={final_token}",
                                3,
                            )
                    elif proactive_enabled:
                        reuse_miss_not_full_accept_rounds += 1
                        if getattr(self.args, "debug_verify_tokens", False):
                            self.color_print(
                                f"[VERIFY-EDGE-REUSE-SKIP][pid={proc_id}][session={session_id}] "
                                f"reason=not_full_accept accepted_cnt={accepted_cnt} gamma={req_gamma}",
                                3,
                            )

                    if reused_pending_tokens and not hit_eos and prefix.shape[1] < max_len:
                        # The last generated token has no KV yet. Keep exactly the
                        # prefix used for next round, including confirmed ahead KV.
                        keep = min(approx_model_cache._past_key_values.get_seq_length(),
                                   prefix.shape[1] + len(reused_pending_tokens) - 1)
                        approx_model_cache.rollback(keep)
                    else:
                        approx_model_cache.rollback(min(accepted, prefix.shape[1]))
                        reused_pending_tokens = []
                        reused_pending_prob_rows = []

                    ahead_control.observe(len(overlap_tokens), len(reused_pending_tokens),
                                          ahead_forward_per_token_s, response_delay)

                    if hit_eos:
                        if getattr(self.args, "debug_verify_tokens", False):
                            self.color_print(
                                f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=eos_token",
                                2,
                            )
                        break
                    if self._is_degenerate_repeat(prefix, input_ids.shape[1]):
                        self.color_print(
                            f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=degenerate_repeat",
                            2,
                        )
                        break
                    if self.args.stop_policy == "dataset" and self.args.dataset == "humaneval":
                        current_text = tokenizer.decode(
                            prefix[0, input_ids.shape[1]:], skip_special_tokens=True
                        )
                        truncated_text, marker_hit = self._truncate_on_humaneval_markers(current_text)
                        if marker_hit:
                            trunc_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
                            trunc_tensor = torch.tensor(
                                [trunc_ids], device=prefix.device, dtype=prefix.dtype
                            )
                            prefix = torch.cat((input_ids, trunc_tensor), dim=1)
                            self.color_print(
                                f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=humaneval_stop_marker",
                                2,
                            )
                            break
                    if self.args.stop_policy == "dataset" and self.args.dataset == "gsm8k":
                        current_text = tokenizer.decode(
                            prefix[0, input_ids.shape[1]:], skip_special_tokens=True
                        )
                        truncated_text, marker_hit = self._truncate_on_gsm8k_markers(current_text)
                        if marker_hit:
                            trunc_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
                            trunc_tensor = torch.tensor(
                                [trunc_ids], device=prefix.device, dtype=prefix.dtype
                            )
                            prefix = torch.cat((input_ids, trunc_tensor), dim=1)
                            self.color_print(
                                f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=gsm8k_next_question_marker",
                                2,
                            )
                            break

                _shutdown_executor(verify_executor, wait=True)
                if executor_holder is not None:
                    executor_holder["executor"] = None
                completion_time = time.monotonic()
                transport_after = client.snapshot_transport_stats()
                transport_delta = {
                    key: int(transport_after[key] - transport_before[key])
                    for key in ("request_bytes", "response_bytes", "rpc_count")
                }

                generated_text = tokenizer.decode(
                    prefix[0, input_ids.shape[1]:], skip_special_tokens=True
                )
                total_reuse_rounds = reuse_hit_rounds + reuse_miss_rounds
                reuse_hit_rate = (reuse_hit_rounds / total_reuse_rounds) if total_reuse_rounds > 0 else 0.0
                self.color_print(
                    f"[Edge {proc_id}] task {task_id} reuse stats: "
                    f"hit={reuse_hit_rounds} miss={reuse_miss_rounds} "
                    f"skip_not_full_accept={reuse_miss_not_full_accept_rounds} "
                    f"hit_rate={reuse_hit_rate:.3f}",
                    2,
                )
                self.color_print(
                    f"[Edge {proc_id}] finished task {task_id}, generated {prefix.shape[1] - input_ids.shape[1]} tokens",
                    2,
                )
                self.color_print(f"[Edge {proc_id}] task {task_id} output:\n{generated_text}", 2)

                generated_tokens = int(prefix.shape[1] - input_ids.shape[1])
                task_e2e_ms = elapsed_ms(task_start, completion_time)
                request_e2e_ms = elapsed_ms(request_start, completion_time)
                ttft_value = (
                    elapsed_ms(request_start, first_token_time)
                    if first_token_time is not None
                    else None
                )
                decode_ttft_value = (
                    elapsed_ms(task_start, first_token_time)
                    if first_token_time is not None
                    else None
                )
                tpot_value = (
                    tpot_ms(first_token_time, completion_time, generated_tokens)
                    if first_token_time is not None
                    else None
                )
                completion_s = (
                    max(0.0, completion_time - arrival_origin)
                    if arrival_origin is not None
                    else request_e2e_ms / 1000.0
                )
                per_task = {
                    "task_id": task_id,
                    "proc_id": int(proc_id),
                    "profile": self.args.profile,
                    "server_sched_mode": self.args.server_sched_mode,
                    "verify_method": verify_method,
                    "draft_top_k": int(draft_top_k),
                    # q_k is the exact normalized top-k draft distribution, so its
                    # transmitted sparse support has zero omitted mass.  Keep the
                    # experiment-facing q_tail name alongside the detailed field.
                    "q_tail": 0.0 if verify_method == "rejection" else None,
                    "draft_support_tail_mass": 0.0 if verify_method == "rejection" else None,
                    "enable_pipeline": pipeline_enabled,
                    "enable_proactive_draft": proactive_enabled,
                    "enable_latency_priority": timing_protocol,
                    "generated_tokens": generated_tokens,
                    "global_sample_index": int(global_idx),
                    "sampling_seed": int(sampling_seed),
                    "scheduled_arrival_s": scheduled_arrival_s,
                    "actual_arrival_s": actual_arrival_s,
                    "arrival_lag_ms": arrival_lag_ms,
                    "prefill_ms": prefill_ms,
                    "local_prefill_s": float(first_round_timings.get("local_prefill_s", 0.0)),
                    "local_prefill_ms": float(first_round_timings.get("local_prefill_ms", 0.0)),
                    "local_decode_per_token_s": float(
                        first_round_timings.get("local_decode_per_token_s", 0.0)
                    ),
                    "local_decode_per_token_ms": float(
                        first_round_timings.get("local_decode_per_token_ms", 0.0)
                    ),
                    "prefill_push_ms": first_prefill_push_s * 1000.0,
                    "prefill_timing_upload_ms": float(
                        first_round_timings.get("timing_upload_ms", 0.0)
                    ),
                    "prefill_pull_ms": first_prefill_pull_s * 1000.0,
                    "overlap_prefill_first_draft": overlap_prefill_first_draft,
                    "first_draft_ms": first_round_draft_ms,
                    "prefill_first_draft_overlap_ms": prefill_first_draft_overlap_ms,
                    "prefill_wait_after_first_draft_ms": prefill_wait_after_first_draft_ms,
                    "request_e2e_ms": request_e2e_ms,
                    "ttft_ms": ttft_value,
                    "decode_ttft_ms": decode_ttft_value,
                    "tpot_ms": tpot_value,
                    "completion_s": completion_s,
                    "task_e2e_ms": task_e2e_ms,
                    "tok_per_s_task": float(generated_tokens / (task_e2e_ms / 1000.0)) if task_e2e_ms > 0 else 0.0,
                    "rounds": int(rounds),
                    "avg_round_ms": float(sum_round_ms / rounds) if rounds > 0 else 0.0,
                    "avg_draft_ms": float(sum_draft_ms / rounds) if rounds > 0 else 0.0,
                    "avg_wait_ms": float(sum_wait_ms / rounds) if rounds > 0 else 0.0,
                    "avg_http_total_ms": float(sum_http_total_ms / rounds) if rounds > 0 else 0.0,
                    "one_way_timing_policy": "calibration_upper_bound_v1",
                    "avg_cloud_total_ms": float(sum_cloud_total_ms / rounds) if rounds > 0 else 0.0,
                    "avg_verify_ms": float(sum_verify_ms / rounds) if rounds > 0 else 0.0,
                    "avg_transport_rtt_ms": float(sum_transport_rtt_ms / rounds) if rounds > 0 else 0.0,
                    "avg_push_ms": float(sum_push_ms / rounds) if rounds > 0 else 0.0,
                    "avg_pull_ms": float(sum_pull_ms / rounds) if rounds > 0 else 0.0,
                    "accepted_total": int(accepted_total),
                    "drafted_total": int(drafted_total),
                    "accept_rate": float(accepted_total / drafted_total) if drafted_total > 0 else 0.0,
                    "mean_accepted_tokens_per_verify": (
                        float(accepted_total / rounds) if rounds > 0 else None
                    ),
                    "ahead_generated_tokens": ahead_generated_tokens,
                    "ahead_reused_tokens": ahead_reused_tokens,
                    "ahead_discarded_tokens": ahead_generated_tokens - ahead_reused_tokens,
                    "ahead_forward_ms": ahead_forward_s * 1000.0,
                    "response_ready_to_consumed_ms": response_consumption_delay_s * 1000.0,
                    "draft_forward_tokens": approx_model_cache.forward_tokens,
                    "draft_forward_cpu_ms": approx_model_cache.forward_cpu_s * 1000.0,
                    "ahead_max_tokens": int(getattr(self.args, "ahead_max_tokens", 4)),
                    "reuse_hit_rounds": int(reuse_hit_rounds),
                    "reuse_miss_rounds": int(reuse_miss_rounds),
                    "reuse_skip_not_full_accept_rounds": int(reuse_miss_not_full_accept_rounds),
                    "output_text": generated_text,
                    "output_token_ids": [
                        int(token_id)
                        for token_id in prefix[0, input_ids.shape[1] :].tolist()
                    ],
                    "reference": sample.get("reference", sample.get("answer")),
                    "workload_hash": self.args.workload_hash,
                    "request_bytes": transport_delta["request_bytes"],
                    "response_bytes": transport_delta["response_bytes"],
                    "rpc_count": transport_delta["rpc_count"],
                    "transport_bytes_definition": (
                        "application-layer HTTP JSON body/response bytes; excludes HTTP framing, "
                        "TCP/IP, and SSH encapsulation"
                    ),
                }
                with open(self._metrics_path(proc_id), "a") as f:
                    f.write(json.dumps(per_task, ensure_ascii=True) + "\n")
            finally:
                try:
                    client.release_session(session_id)
                except Exception as exc:
                    self.color_print(f"[SESSION-RELEASE-FAILED] {session_id}: {exc}", 2)



def parse_edge_arguments() -> argparse.Namespace:
    edge_parser = argparse.ArgumentParser(add_help=False)
    edge_parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8001")
    edge_parser.add_argument("--request_timeout", type=float, default=30.0)
    edge_parser.add_argument("--edge_gpu_start", type=int, default=0)
    edge_parser.add_argument("--edge_gpus", type=int, default=1)
    edge_parser.add_argument(
        "--max_tasks_per_draft",
        type=int,
        default=0,
        help="max tasks per Edge shard; <= 0 processes the full canonical shard",
    )

    edge_args, remaining = edge_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        base_args = parse_arguments()
    finally:
        sys.argv = original_argv

    base_args.server_url = edge_args.server_url
    base_args.request_timeout = edge_args.request_timeout
    base_args.edge_gpu_start = edge_args.edge_gpu_start
    base_args.edge_gpus = edge_args.edge_gpus
    # Edge is the formal canonical-request launcher.  Keep its default
    # unbounded even though legacy benchmark entrypoints retain their explicit
    # smoke cap from src.util.parse_arguments().
    base_args.max_tasks_per_draft = edge_args.max_tasks_per_draft
    return base_args


if __name__ == "__main__":
    args = parse_edge_arguments()
    runner = EdgeRunner(args)
    runner.eval()
