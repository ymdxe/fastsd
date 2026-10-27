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
    """Legacy helper for clients that send Prefill before local drafting.

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
        self._stats_lock = threading.Lock()
        self._request_bytes = 0
        self._response_bytes = 0
        self._rpc_count = 0

    def snapshot_transport_stats(self) -> Dict[str, int]:
        """Return application-layer HTTP byte/RPC counters for this client."""

        return {
            "request_bytes": int(self._request_bytes),
            "response_bytes": int(self._response_bytes),
            "rpc_count": int(self._rpc_count),
        }

    def reset_transport_stats(self) -> None:
        """Exclude health checks and stateful warmups from measured requests."""

        self._request_bytes = 0
        self._response_bytes = 0
        self._rpc_count = 0

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
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        url = f"{self.server_url}/health"
        response = self.session.get(url, timeout=self.timeout)
        with self._stats_lock:
            self._response_bytes += len(response.content)
            self._rpc_count += 1
        response.raise_for_status()
        return response.json()

    def init_session(self) -> str:
        resp = self._post("/session/init", {})
        return resp["session_id"]

    def synchronize_clock(self, samples: int = 5) -> dict[str, int]:
        """Calibrate cloud-edge epoch offset with NTP-style probes.

        The minimum-RTT sample is used as the least-contended offset estimate;
        no RTT/2 value is ever used as an application upload/pull duration.
        """

        measurements = []
        for _ in range(max(1, int(samples))):
            edge_before = time.time_ns()
            response = self.session.get(
                f"{self.server_url}/clock/sync", timeout=self.timeout
            )
            edge_after = time.time_ns()
            response.raise_for_status()
            cloud_time = int(response.json()["cloud_time_ns"])
            midpoint = (edge_before + edge_after) // 2
            measurements.append(
                (edge_after - edge_before, cloud_time - midpoint)
            )
        rtt_ns, offset_ns = min(measurements, key=lambda item: item[0])
        self.clock_offset_ns = int(offset_ns)
        self.clock_uncertainty_ns = max(0, int(rtt_ns // 2))
        return {
            "clock_offset_ns": self.clock_offset_ns,
            "clock_uncertainty_ns": self.clock_uncertainty_ns,
            "samples": len(measurements),
        }

    def _attach_pull(self, response: Dict[str, Any], edge_receive_time_ns: int) -> Dict[str, Any]:
        """Attach a synchronized cloud-to-edge duration to an RPC response."""

        cloud_send = response.get("cloud_send_time_ns")
        if cloud_send is None:
            response.setdefault("pull_s", 0.0)
            return response
        try:
            pull_ns = int(edge_receive_time_ns) - (
                int(cloud_send) - int(self.clock_offset_ns)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("invalid cloud_send_time_ns in timing response") from exc
        if pull_ns < 0:
            raise RuntimeError(
                "synchronized clocks produced a negative pull duration; retry clock calibration"
            )
        response["pull_s"] = pull_ns / 1_000_000_000.0
        response["pull_ms"] = response["pull_s"] * 1000.0
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
    ) -> Dict[str, Any]:
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
            # Arrival-gated token uploads are prioritized by the cloud arrival
            # timestamp; only the legacy timing-gated path opts into telemetry
            # based scoring.
            "timing_priority": bool(self.timing_priority and timing_required),
            "edge_send_time_ns": int(edge_send_time_ns),
            "clock_offset_ns": int(self.clock_offset_ns),
        }
        response = self._post("/prefill", payload)
        return self._attach_pull(response, time.time_ns())

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
            "verify_method": str(verify_method),
            "sampling_seed": int(sampling_seed or 0),
            "draft_prob_block": draft_prob_block,
        }
        transport_start = time.monotonic()
        resp = self._post("/verify", payload)
        edge_receive_time_ns = time.time_ns()
        measured_http_total = max(0.0, time.monotonic() - transport_start)
        self._attach_pull(resp, edge_receive_time_ns)
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
        )
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
            overlap_prefill_first_draft = bool(
                getattr(self.args, "overlap_prefill_first_draft", False)
            )
            final_token = None
            reused_pending_tokens: List[int] = []
            reused_pending_prob_rows: List[torch.Tensor] = []
            current_gamma = int(self.args.gamma)
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
                timing_required=timing_protocol,
                edge_send_time_ns=prefill_send_time_ns,
            )
            arrival_call = None
            prefill_upload_call = None
            if timing_protocol:
                arrival_call = lambda: client.prefill_arrival(
                    session_id=session_id,
                    task_id=task_id,
                    edge_send_time_ns=time.time_ns(),
                )

                def prefill_upload_call(draft_output):
                    if getattr(torch, "is_tensor", lambda value: False)(draft_output):
                        upload_tokens = draft_output[0].detach().cpu().tolist()
                    elif (
                        isinstance(draft_output, (list, tuple))
                        and draft_output
                        and isinstance(draft_output[0], (list, tuple))
                    ):
                        upload_tokens = list(draft_output[0])
                    else:
                        upload_tokens = list(draft_output)
                    return client.prefill(
                        session_id=session_id,
                        task_id=task_id,
                        draft_output=[int(token) for token in upload_tokens],
                        prefix_len=prefix.shape[1],
                        prompt_len=prefix.shape[1],
                        lag=0.0,
                        current_time=time.time(),
                        gamma=current_gamma,
                        timing_required=False,
                        edge_send_time_ns=time.time_ns(),
                    )

            # The first draft is generated exactly once.  In the timing-aware
            # FastSD profile the cloud only records arrival while this call
            # runs; the generated tokens are uploaded before Prefill/Verify
            # state is established.
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
                    first_round_x, prefill_resp, first_round_timings = _run_first_draft_after_arrival(
                        verify_executor,
                        arrival_call,
                        lambda: _timed_local_draft_generate(
                            approx_model_cache, prefix, current_gamma
                        ),
                        prefill_upload_call,
                        timeout=float(getattr(self.args, "request_timeout", 0.0)),
                        cancel_call=lambda: client.prefill_cancel(session_id, task_id),
                    )
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
                if timing_protocol:
                    _, prefill_resp, first_round_timings = _run_first_draft_after_arrival(
                        verify_executor,
                        arrival_call,
                        lambda: (
                            prefix,
                            {
                                "local_prefill_s": 0.0,
                                "local_decode_per_token_s": 0.0,
                            },
                        ),
                        prefill_upload_call,
                        timeout=float(getattr(self.args, "request_timeout", 0.0)),
                        cancel_call=lambda: client.prefill_cancel(session_id, task_id),
                    )
                    if "cloud_send_time_ns" not in prefill_resp:
                        raise RuntimeError("FastSD Prefill response omitted cloud_send_time_ns")
                    prefill_ms = first_round_timings["prefill_ms"]
                    task_start = first_round_timings["prefill_completed_at"]
                else:
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
                round_start = time.monotonic()
                req_gamma = current_gamma if pipeline_enabled else int(self.args.gamma)
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
                        draft_rows = approx_model_cache._prob_history[
                            0, start:start + req_gamma
                        ].detach().float()
                    else:
                        draft_rows = None
                elif reused_pending_tokens:
                    draft_started = time.monotonic()
                    reuse_count = min(len(reused_pending_tokens), req_gamma)
                    reuse_tensor = torch.tensor(
                        [reused_pending_tokens[:reuse_count]],
                        device=prefix.device,
                        dtype=prefix.dtype,
                    )
                    x = torch.cat((prefix, reuse_tensor), dim=1)
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

                # 验证等待期间持续 draft，最多缓存 gamma+1（bridge + gamma）个 token。
                overlap_tokens: List[int] = []
                overlap_prob_rows: List[torch.Tensor] = []
                overlap_prefix = x
                max_overlap_tokens = req_gamma + 1
                wait_start = time.monotonic()
                wait_draft_steps = 0
                if getattr(self.args, "debug_verify_tokens", False):
                    self.color_print(
                        f"[VERIFY-EDGE-WAIT-START][pid={proc_id}][session={session_id}] "
                        f"prefix_len={prefix_len} max_overlap_tokens={max_overlap_tokens}",
                        3,
                    )
                if proactive_enabled:
                    while len(overlap_tokens) < max_overlap_tokens and not verify_future.done():
                        if verify_method == "rejection":
                            overlap_prefix, overlap_rows = approx_model_cache.generate_with_probs(
                                overlap_prefix, 1
                            )
                            overlap_prob_rows.append(overlap_rows[0].detach().float())
                        else:
                            overlap_prefix = approx_model_cache.generate(overlap_prefix, 1)
                        overlap_tokens.append(int(overlap_prefix[0, -1].item()))
                        wait_draft_steps += 1
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
                approx_model_cache.rollback(accepted)
                reused_pending_tokens = []
                reused_pending_prob_rows = []

                if proactive_enabled and accepted_cnt == req_gamma and overlap_tokens:
                    if overlap_tokens[0] == final_token:
                        reused_pending_tokens = overlap_tokens[1 : 1 + req_gamma]
                        if verify_method == "rejection":
                            reused_pending_prob_rows = [
                                row.detach().float()
                                for row in overlap_prob_rows[1 : 1 + req_gamma]
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
