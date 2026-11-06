import asyncio
import multiprocessing as mp
import os
import queue as py_queue
import sys
import threading
import uuid
from typing import Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.engine import Decoding
from src.util import parse_arguments
from src.request_validation import canonicalize_request
import uvicorn
import time


app = FastAPI(title="FastSD Cloud Target Service")

# 进程内 IPC 句柄（由 main 初始化）
request_queue: Optional[mp.Queue] = None
response_queue: Optional[mp.Queue] = None
worker_proc: Optional[mp.Process] = None
_request_max_tokens: int = 400
_server_sched_mode: str = "fastsd"
_enable_latency_priority: bool = True

# FastAPI 请求等待表
_pending: Dict[str, asyncio.Future] = {}
_pending_sessions: Dict[str, str] = {}
_pending_meta: Dict[str, str] = {}
_pending_task_ids: Dict[str, str] = {}
_pending_lock = threading.Lock()
_prefill_arrivals: Dict[str, dict] = {}
_prefill_timing_updates: Dict[str, dict] = {}
_cloud_time_lock = threading.Lock()
_cloud_total_ms_sum: float = 0.0
_cloud_total_ms_count: int = 0


def _finite_nonnegative(value: float, field: str) -> float:
    """Validate a duration without hiding invalid telemetry as zero."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be finite and non-negative") from exc
    if not number == number or number in (float("inf"), float("-inf")) or number < 0.0:
        raise HTTPException(status_code=422, detail=f"{field} must be finite and non-negative")
    return number


def _compute_upload_seconds(edge_send_time_ns: int, clock_offset_ns: int, cloud_receive_time_ns: int) -> float:
    """Compute cloud-received minus Edge-sent using synchronized epoch clocks.

    ``clock_offset_ns`` is defined as cloud_time - edge_time.  A negative
    corrected duration is a protocol error; it is never replaced with RTT/2.
    """

    try:
        edge_send = int(edge_send_time_ns)
        offset = int(clock_offset_ns)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail="invalid synchronized timing timestamp") from exc
    if edge_send <= 0:
        raise HTTPException(status_code=422, detail="edge_send_time_ns must be positive")
    duration_ns = int(cloud_receive_time_ns) - (edge_send + offset)
    if duration_ns < 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "synchronized clocks produced a negative upload duration; "
                "retry clock calibration"
            ),
        )
    return duration_ns / 1_000_000_000.0


def _get_avg_cloud_total_ms() -> float:
    with _cloud_time_lock:
        if _cloud_total_ms_count <= 0:
            return 0.0
        return _cloud_total_ms_sum / float(_cloud_total_ms_count)


def _record_cloud_total_ms(value_ms: float) -> None:
    global _cloud_total_ms_sum, _cloud_total_ms_count
    with _cloud_time_lock:
        _cloud_total_ms_sum += max(0.0, float(value_ms))
        _cloud_total_ms_count += 1


class PrefillRequest(BaseModel):
    session_id: str
    task_id: str
    draft_output: list[int]
    prefix_len: int
    lag: float
    current_time: float
    prompt_len: Optional[int] = None
    gamma: int = 0
    prefill_gamma: int = 0
    timing_required: bool = False
    timing_priority: bool = False
    edge_send_time_ns: int = 0
    clock_offset_ns: int = 0


class PrefillArrivalRequest(BaseModel):
    session_id: str
    task_id: str
    edge_send_time_ns: int = 0
    clock_offset_ns: int = 0


class PrefillTimingRequest(BaseModel):
    session_id: str
    task_id: str
    gamma: int
    local_prefill_s: Optional[float] = None
    local_prefill_ms: Optional[float] = None
    local_decode_per_token_s: Optional[float] = None
    local_decode_per_token_ms: Optional[float] = None
    # Formula-name aliases accepted for replaying timing traces.
    T_i_p: Optional[float] = None
    T_i_d: Optional[float] = None
    T_i_push: Optional[float] = None
    edge_send_time_ns: int
    clock_offset_ns: int = 0


class PrefillCancelRequest(BaseModel):
    session_id: str
    task_id: str


class VerifyRequest(BaseModel):
    session_id: str
    task_id: str
    draft_output: list[int]
    prefix_len: int
    lag: float
    current_time: float
    gamma: int = 0
    transport_rtt: float = 0.0
    tail_only: bool = False
    has_bridge_token: bool = False
    timing_priority: bool = False
    local_decode_per_token_s: Optional[float] = None
    local_decode_per_token_ms: Optional[float] = None
    T_i_d: Optional[float] = None
    T_i_pull: Optional[float] = None
    last_pull_s: Optional[float] = None
    pull_s: Optional[float] = None
    verify_method: str = "rejection"
    sampling_seed: int = 0
    draft_prob_block: Optional[str] = None
    edge_send_time_ns: int = 0
    clock_offset_ns: int = 0


class CloudTargetWorker(Decoding):
    """
    云端 target 侧服务：
    仅复用 Decoding 中的 run_target_process_batching。
    """

    def load_data(self):
        return

    def preprocess(self, input_text: str) -> str:
        return input_text

    def postprocess(self, input_text: str, output_text: str) -> str:
        return output_text

    def eval(self):
        # 云端 worker 不通过 Decoding.eval 驱动，这里仅满足抽象基类要求。
        return


class _ResponseQueueProxy:
    def __init__(self, shared_queue: mp.Queue, request_id: str) -> None:
        self.shared_queue = shared_queue
        self.request_id = request_id

    def put(self, payload: dict) -> None:
        self.shared_queue.put({"request_id": self.request_id, "payload": payload})


class _ResponseQueuesProxy:
    def __init__(self, shared_queue: mp.Queue) -> None:
        self.shared_queue = shared_queue

    def __getitem__(self, request_id: str) -> _ResponseQueueProxy:
        return _ResponseQueueProxy(self.shared_queue, request_id)


def _worker_entry(args, req_q: mp.Queue, resp_q: mp.Queue) -> None:
    """
    Worker 进程入口（必须是模块级函数，便于 spawn 模式 pickle）。
    """
    worker = CloudTargetWorker(args)
    worker.load_tokenizer()
    response_queues = _ResponseQueuesProxy(resp_q)
    worker.run_target_process_batching(worker.tokenizer, req_q, response_queues)


def _start_worker(args, req_q: mp.Queue, resp_q: mp.Queue) -> mp.Process:
    """
    启动 Worker 进程，内部使用 Decoding.run_target_process_batching。
    """
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_worker_entry, args=(args, req_q, resp_q), daemon=True)
    proc.start()
    return proc


def _start_result_dispatcher(loop: asyncio.AbstractEventLoop) -> None:
    """
    后台线程：
    从 response_queue 中取结果，唤醒对应的 HTTP 请求 Future。
    """
    def _run():
        while True:
            if response_queue is None:
                time.sleep(0.01)
                continue
            try:
                msg = response_queue.get(timeout=0.5)
            except py_queue.Empty:
                # A dead worker must not leave HTTP callers awaiting a
                # Future forever.  The next request will receive a 503 from
                # ``_enqueue_and_wait`` as well.
                if worker_proc is not None and not worker_proc.is_alive():
                    with _pending_lock:
                        pending = list(_pending.values())
                        _pending.clear()
                        _pending_meta.clear()
                        _pending_sessions.clear()
                        _pending_task_ids.clear()
                        _prefill_arrivals.clear()
                        _prefill_timing_updates.clear()
                    for fut in pending:
                        if not fut.done():
                            loop.call_soon_threadsafe(
                                fut.set_exception,
                                RuntimeError("target worker exited before returning a response"),
                            )
                continue
            req_id = msg.get("request_id")
            payload = msg.get("payload", {})
            with _pending_lock:
                fut = _pending.pop(req_id, None)
                session_id = _pending_meta.pop(req_id, None)
                if session_id is not None:
                    _pending_task_ids.pop(session_id, None)
                if session_id is not None and _pending_sessions.get(session_id) == req_id:
                    _pending_sessions.pop(session_id, None)
            if fut is not None and not fut.done():
                loop.call_soon_threadsafe(fut.set_result, payload)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


@app.on_event("startup")
def startup_event() -> None:
    """
    FastAPI 启动时启动结果分发线程。
    队列与 Worker 由 main 初始化。
    """
    loop = asyncio.get_event_loop()
    _start_result_dispatcher(loop)


@app.get("/health")
def health() -> dict:
    if request_queue is None:
        status = "not_initialized"
    elif worker_proc is None or not worker_proc.is_alive():
        status = "failed"
    else:
        status = "ok"
    return {
        "status": status,
        "worker_alive": bool(worker_proc is not None and worker_proc.is_alive()),
        "model_ready": bool(worker_proc is not None and worker_proc.is_alive()),
    }


@app.get("/clock/sync")
def clock_sync() -> dict:
    """Return an epoch timestamp for the Edge NTP-style clock calibration."""

    return {"cloud_time_ns": time.time_ns()}


@app.post("/session/init")
def session_init() -> dict:
    """
    返回新的 session_id（UUID）。
    """
    print("Initializing new session...")
    return {"session_id": uuid.uuid4().hex}


async def _enqueue_and_wait(request_dict: dict, req_id: str) -> dict:
    if request_queue is None:
        return {"error": "queues_not_initialized"}
    if worker_proc is not None and not worker_proc.is_alive():
        raise HTTPException(status_code=503, detail="target worker is not alive")

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    response_key = uuid.uuid4().hex
    with _pending_lock:
        if req_id in _pending_sessions:
            raise HTTPException(status_code=409, detail="session already has an in-flight request")
        _pending[response_key] = fut
        _pending_sessions[req_id] = response_key
        _pending_meta[response_key] = req_id
        _pending_task_ids[req_id] = str(request_dict.get("task_id", ""))
    request_dict["response_key"] = response_key
    request_dict["proc_id"] = req_id
    try:
        request_queue.put(request_dict)
        return await fut
    except asyncio.CancelledError:
        with _pending_lock:
            _pending.pop(response_key, None)
            _pending_meta.pop(response_key, None)
            _pending_task_ids.pop(req_id, None)
            if _pending_sessions.get(req_id) == response_key:
                _pending_sessions.pop(req_id, None)
        raise
    except BaseException:
        with _pending_lock:
            _pending.pop(response_key, None)
            _pending_meta.pop(response_key, None)
            _pending_task_ids.pop(req_id, None)
            if _pending_sessions.get(req_id) == response_key:
                _pending_sessions.pop(req_id, None)
        raise


@app.post("/prefill/arrival")
async def prefill_arrival(req: PrefillArrivalRequest) -> dict:
    """Record a Prefill arrival without sending work to the target worker.

    The target worker must not see a Prefill until the Edge uploads its first
    local draft round.  The monotonic timestamp is kept in the cloud process
    and copied into the later token upload so scheduler wait time starts here,
    rather than when the upload reaches the worker.
    """

    if request_queue is None:
        return {"error": "queues_not_initialized"}

    cloud_arrival_time_ns = time.time_ns()
    cloud_arrival_monotonic_s = time.monotonic()
    with _pending_lock:
        pending_response = _pending_sessions.get(req.session_id)
        if pending_response is not None:
            raise HTTPException(status_code=409, detail="session has an in-flight request")
        previous = _prefill_arrivals.get(req.session_id)
        if previous is not None:
            if previous.get("task_id") != str(req.task_id):
                raise HTTPException(status_code=409, detail="session already has a different Prefill arrival")
            return {
                "session_id": req.session_id,
                "task_id": req.task_id,
                "status": "arrival_recorded",
                "cloud_arrival_time_ns": int(previous["cloud_arrival_time_ns"]),
            }
        _prefill_arrivals[req.session_id] = {
            "task_id": str(req.task_id),
            "cloud_arrival_time_ns": int(cloud_arrival_time_ns),
            "cloud_arrival_monotonic_s": float(cloud_arrival_monotonic_s),
        }
    return {
        "session_id": req.session_id,
        "task_id": req.task_id,
        "status": "arrival_recorded",
        "cloud_arrival_time_ns": int(cloud_arrival_time_ns),
    }


@app.post("/prefill")
async def prefill(req: PrefillRequest) -> dict:
    if request_queue is None:
        return {"error": "queues_not_initialized"}

    req_id = req.session_id
    with _pending_lock:
        arrival = dict(_prefill_arrivals.get(req_id) or {})
    if arrival and arrival.get("task_id") != str(req.task_id):
        raise HTTPException(status_code=409, detail="Prefill task_id does not match recorded arrival")
    arrival_recorded = bool(arrival)
    timing_required = bool(req.timing_required) and not arrival_recorded

    request_dict = {
        "task_id": req.task_id,
        # Keep multiprocessing.Queue payloads tensor-free.  Passing a torch
        # storage through the queue uses the resource_sharer FD channel and
        # eventually fails in long experiments with "received 0 items of
        # ancdata".  The target worker tensorizes this plain list on ingress.
        "draft_output": [int(token) for token in req.draft_output],
        "prefix_len": req.prefix_len,
        "prompt_len": int(req.prompt_len or req.prefix_len),
        "proc_id": req_id,
        "lag": req.lag,
        "current_time": req.current_time,
        "prefill_gamma": int(req.prefill_gamma or req.gamma or 0),
        "timing_required": timing_required,
        "timing_priority": bool(req.timing_priority),
        # Arrival-gated requests are executable only after this token upload;
        # legacy timing-gated requests retain the old control-message path.
        "timing_ready": not timing_required,
        "edge_send_time_ns": int(req.edge_send_time_ns or 0),
        "clock_offset_ns": int(req.clock_offset_ns or 0),
        "task_type": "prefill",
    }
    if arrival_recorded:
        request_dict.update(
            {
                "arrival_time_ns": int(arrival["cloud_arrival_time_ns"]),
                "arrival_monotonic_s": float(arrival["cloud_arrival_monotonic_s"]),
                "cloud_arrival_time_ns": int(arrival["cloud_arrival_time_ns"]),
                "cloud_arrival_monotonic_s": float(arrival["cloud_arrival_monotonic_s"]),
                "arrival_recorded": True,
            }
        )
    try:
        request_dict = canonicalize_request(
            request_dict,
            max_tokens=_request_max_tokens,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request_dict.get("timing_required") and (
        _server_sched_mode != "fastsd" or not _enable_latency_priority
    ):
        raise HTTPException(status_code=422, detail="timing-aware Prefill is FastSD-only")
    try:
        resp = await _enqueue_and_wait(request_dict, req_id)
    except BaseException:
        # Release request-scoped timing/arrival state when the long poll fails.
        _prefill_timing_updates.pop(req_id, None)
        if arrival_recorded:
            with _pending_lock:
                _prefill_arrivals.pop(req_id, None)
        raise
    if "error" in resp:
        _prefill_timing_updates.pop(req_id, None)
        if arrival_recorded:
            with _pending_lock:
                _prefill_arrivals.pop(req_id, None)
        raise HTTPException(status_code=422, detail=str(resp["error"]))
    if "status" in resp:
        out = {"session_id": req_id, "status": resp["status"]}
    else:
        out = {"session_id": req_id, "status": "prefill_ok"}
    timing = _prefill_timing_updates.pop(req_id, {})
    if arrival_recorded:
        with _pending_lock:
            _prefill_arrivals.pop(req_id, None)
    out["cloud_send_time_ns"] = time.time_ns()
    if arrival_recorded:
        out["cloud_arrival_time_ns"] = int(arrival["cloud_arrival_time_ns"])
    if "push_s" in timing:
        out["push_s"] = float(timing["push_s"])
    out["timing_ready"] = True
    return out


@app.post("/prefill/timing")
async def prefill_timing(req: PrefillTimingRequest) -> dict:
    """Legacy timing-control endpoint kept for older clients.

    This endpoint intentionally uses a separate HTTP connection/session on the
    Edge.  The arrival-plus-token-upload FastSD path does not call it; it only
    sends a control message for older timing-gated requests.
    """

    if request_queue is None:
        return {"error": "queues_not_initialized"}
    if _server_sched_mode != "fastsd" or not _enable_latency_priority:
        raise HTTPException(status_code=422, detail="/prefill/timing is FastSD-only")
    local_prefill_s = req.local_prefill_s
    if local_prefill_s is None and req.T_i_p is not None:
        local_prefill_s = req.T_i_p
    if local_prefill_s is None and req.local_prefill_ms is not None:
        local_prefill_s = float(req.local_prefill_ms) / 1000.0
    local_decode_s = req.local_decode_per_token_s
    if local_decode_s is None and req.T_i_d is not None:
        local_decode_s = req.T_i_d
    if local_decode_s is None and req.local_decode_per_token_ms is not None:
        local_decode_s = float(req.local_decode_per_token_ms) / 1000.0
    if local_prefill_s is None or local_decode_s is None:
        raise HTTPException(status_code=422, detail="timing update requires local Prefill and Decode durations")
    local_prefill_s = _finite_nonnegative(local_prefill_s, "local_prefill_s")
    local_decode_s = _finite_nonnegative(local_decode_s, "local_decode_per_token_s")
    if int(req.gamma) <= 0 or int(req.gamma) > _request_max_tokens:
        raise HTTPException(status_code=422, detail="gamma must be positive and within max_tokens")

    with _pending_lock:
        response_key = _pending_sessions.get(req.session_id)
        pending_task = _pending_task_ids.get(req.session_id)
    if response_key is None:
        raise HTTPException(status_code=409, detail="no in-flight Prefill for session")
    if pending_task is not None and pending_task != str(req.task_id):
        raise HTTPException(status_code=409, detail="timing task_id does not match Prefill")

    cloud_receive_time_ns = time.time_ns()
    push_s = _compute_upload_seconds(
        req.edge_send_time_ns,
        req.clock_offset_ns,
        cloud_receive_time_ns,
    )
    update = {
        "control_type": "prefill_timing",
        "task_type": "prefill_timing",
        "proc_id": req.session_id,
        "task_id": req.task_id,
        "response_key": response_key,
        "local_prefill_s": local_prefill_s,
        "local_decode_per_token_s": local_decode_s,
        "prefill_gamma": int(req.gamma),
        "m_i": int(req.gamma),
        "push_s": push_s,
        "T_i_p": local_prefill_s,
        "T_i_d": local_decode_s,
        "T_i_push": push_s,
        "timing_ready": True,
        "timing_cloud_receive_time_ns": cloud_receive_time_ns,
    }
    previous = _prefill_timing_updates.get(req.session_id)
    if previous is not None:
        comparable = (previous.get("local_prefill_s"), previous.get("local_decode_per_token_s"), previous.get("m_i"))
        current = (update["local_prefill_s"], update["local_decode_per_token_s"], update["m_i"])
        if comparable != current:
            raise HTTPException(status_code=409, detail="conflicting duplicate Prefill timing update")
        return {"session_id": req.session_id, "status": "timing_ready", "push_s": previous["push_s"]}
    _prefill_timing_updates[req.session_id] = update
    try:
        request_queue.put(update)
    except BaseException:
        _prefill_timing_updates.pop(req.session_id, None)
        raise
    return {"session_id": req.session_id, "status": "timing_ready", "push_s": push_s}


@app.post("/prefill/cancel")
async def prefill_cancel(req: PrefillCancelRequest) -> dict:
    """Cancel a timing-gated Prefill whose Edge draft failed or timed out.

    The Edge cannot join its HTTP executor while the original Prefill long poll
    is waiting for ``prefill_timing``.  Removing the pending future and
    completing it with an error lets that request unwind, while the worker
    receives a matching control message and discards a queued WorkItem (or
    remembers the cancellation if ingress has not happened yet).
    """

    if request_queue is None:
        return {"error": "queues_not_initialized"}
    with _pending_lock:
        response_key = _pending_sessions.get(req.session_id)
        pending_task = _pending_task_ids.get(req.session_id)
        fut = _pending.get(response_key) if response_key is not None else None
        arrival = _prefill_arrivals.get(req.session_id)
        if response_key is None and arrival is None:
            raise HTTPException(status_code=409, detail="no in-flight Prefill for session")
        if pending_task is not None and pending_task != str(req.task_id):
            raise HTTPException(status_code=409, detail="cancel task_id does not match Prefill")
        if arrival is not None and arrival.get("task_id") != str(req.task_id):
            raise HTTPException(status_code=409, detail="cancel task_id does not match recorded arrival")
        _prefill_arrivals.pop(req.session_id, None)
        if response_key is not None:
            _pending.pop(response_key, None)
            _pending_meta.pop(response_key, None)
            _pending_task_ids.pop(req.session_id, None)
            if _pending_sessions.get(req.session_id) == response_key:
                _pending_sessions.pop(req.session_id, None)
    _prefill_timing_updates.pop(req.session_id, None)
    if fut is not None and not fut.done():
        # A normal result keeps the cancellation as an expected HTTP 422 from
        # /prefill, rather than creating an unobserved Future exception.
        fut.set_result({"error": "prefill cancelled by Edge"})
    if response_key is not None:
        request_queue.put(
            {
                "control_type": "prefill_cancel",
                "task_type": "prefill_cancel",
                "proc_id": req.session_id,
                "task_id": req.task_id,
            }
        )
    return {"session_id": req.session_id, "status": "cancelled"}


@app.post("/verify")
async def verify(req: VerifyRequest) -> dict:
    api_start = time.time()
    cloud_receive_time_ns = time.time_ns()
    if request_queue is None:
        return {"error": "queues_not_initialized"}

    req_id = req.session_id

    avg_cloud_total_ms = _get_avg_cloud_total_ms()

    request_dict = {
        "task_id": req.task_id,
        "draft_output": [int(token) for token in req.draft_output],
        "prefix_len": req.prefix_len,
        "proc_id": req_id,
        "lag": req.lag,
        "current_time": req.current_time,
        "gamma": req.gamma,
        "transport_rtt": req.transport_rtt,
        "avg_cloud_total_ms": avg_cloud_total_ms,
        "task_type": "verify",
        "tail_only": req.tail_only,
        "has_bridge_token": req.has_bridge_token,
        "local_decode_per_token_s": req.local_decode_per_token_s,
        "local_decode_per_token_ms": req.local_decode_per_token_ms,
        "T_i_d": req.T_i_d,
        "T_i_pull": req.T_i_pull,
        "last_pull_s": req.last_pull_s,
        "pull_s": req.pull_s,
        "verify_method": req.verify_method,
        "sampling_seed": int(req.sampling_seed),
        "draft_prob_block": req.draft_prob_block,
        "edge_send_time_ns": int(req.edge_send_time_ns or 0),
        "clock_offset_ns": int(req.clock_offset_ns or 0),
    }
    if _server_sched_mode == "fastsd" and req.timing_priority and _enable_latency_priority:
        # FastSD requests must carry an auditable upload timestamp.  Legacy
        # baseline requests keep their historical RTT path untouched.
        if request_dict["edge_send_time_ns"] <= 0:
            raise HTTPException(status_code=422, detail="FastSD Verify requires edge_send_time_ns")
        request_dict["push_s"] = _compute_upload_seconds(
            request_dict["edge_send_time_ns"],
            request_dict["clock_offset_ns"],
            cloud_receive_time_ns,
        )
    for field in ("local_decode_per_token_s", "local_decode_per_token_ms", "last_pull_s", "pull_s"):
        if request_dict.get(field) is not None:
            request_dict[field] = _finite_nonnegative(request_dict[field], field)
    try:
        request_dict = canonicalize_request(
            request_dict,
            max_tokens=_request_max_tokens,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    resp = await _enqueue_and_wait(request_dict, req_id)
    if "error" in resp:
        raise HTTPException(status_code=422, detail=str(resp["error"]))
    accepted = int(resp["accepted"])
    final_token = resp["final_token"]
    final_token_id = int(final_token.item()) if hasattr(final_token, "item") else int(
        final_token
    )

    out = {"session_id": req_id, "accepted": accepted, "final_token": final_token_id}
    if "verify_ms" in resp:
        out["verify_ms"] = float(resp["verify_ms"])
    if "suggested_gamma" in resp:
        out["suggested_gamma"] = int(resp["suggested_gamma"])
    out["cloud_send_time_ns"] = time.time_ns()
    if "push_s" in request_dict:
        out["push_s"] = float(request_dict["push_s"])
    # Cloud-side end-to-end service time for this HTTP verify request.
    out["cloud_total_ms"] = (time.time() - api_start) * 1000.0
    _record_cloud_total_ms(out["cloud_total_ms"])
    return out


@app.post("/exit")
def exit_worker() -> dict:
    if request_queue is None:
        return {"status": "queues_not_initialized"}
    request_queue.put(None)
    return {"status": "sent"}


def main() -> None:
    global request_queue, response_queue, worker_proc, _request_max_tokens, _server_sched_mode, _enable_latency_priority
    args = parse_arguments()
    _request_max_tokens = int(args.max_tokens)
    _server_sched_mode = str(getattr(args, "server_sched_mode", "fastsd"))
    _enable_latency_priority = bool(getattr(args, "enable_latency_priority", False))

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()

    worker_proc = _start_worker(args, request_queue, response_queue)

    # Keep the shared-server default loopback-only.  A wider bind requires an
    # explicit CLOUD_SERVICE_HOST override in the run plan/configuration.
    host = os.environ.get("CLOUD_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("CLOUD_SERVICE_PORT", "8001"))
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
