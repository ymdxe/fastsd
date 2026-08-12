import asyncio
import multiprocessing as mp
import os
import sys
import threading
import uuid
from typing import Dict, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.engine import Decoding
from src.util import parse_arguments
import uvicorn
import time


app = FastAPI(title="FastSD Cloud Target Service")

# 进程内 IPC 句柄（由 main 初始化）
request_queue: Optional[mp.Queue] = None
response_queue: Optional[mp.Queue] = None
worker_proc: Optional[mp.Process] = None
worker_ready: Optional[mp.Event] = None
worker_failure: Optional[mp.Queue] = None
service_shutdown_requested = False
service_run_id: Optional[str] = None

# FastAPI 请求等待表
_pending: Dict[str, asyncio.Future] = {}
_pending_lock = threading.Lock()
_cloud_time_lock = threading.Lock()
_cloud_total_ms_sum: float = 0.0
_cloud_total_ms_count: int = 0


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
    # Kept for wire compatibility only.  The cloud scheduler uses its own
    # server_enqueue_monotonic value and never compares hosts' clocks.
    current_time: Optional[float] = None


class VerifyRequest(BaseModel):
    session_id: str
    task_id: str
    draft_output: list[int]
    prefix_len: int
    lag: float
    current_time: Optional[float] = None
    gamma: int = 0
    transport_rtt: float = 0.0
    tail_only: bool = False
    has_bridge_token: bool = False


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


def _worker_entry(
    args,
    req_q: mp.Queue,
    resp_q: mp.Queue,
    ready_event: mp.Event,
    failure_queue: mp.Queue,
) -> None:
    """
    Worker 进程入口（必须是模块级函数，便于 spawn 模式 pickle）。
    """
    try:
        worker = CloudTargetWorker(args)
        worker.load_tokenizer()
        response_queues = _ResponseQueuesProxy(resp_q)
        worker.run_target_process_batching(
            worker.tokenizer,
            req_q,
            response_queues,
            ready_event=ready_event,
        )
    except BaseException as exc:
        try:
            failure_queue.put(repr(exc))
        finally:
            raise


def _start_worker(
    args,
    req_q: mp.Queue,
    resp_q: mp.Queue,
    ready_event: mp.Event,
    failure_queue: mp.Queue,
) -> mp.Process:
    """
    启动 Worker 进程，内部使用 Decoding.run_target_process_batching。
    """
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_worker_entry,
        args=(args, req_q, resp_q, ready_event, failure_queue),
        daemon=True,
    )
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
            msg = response_queue.get()
            req_id = msg.get("request_id")
            payload = msg.get("payload", {})
            with _pending_lock:
                fut = _pending.pop(req_id, None)
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
    if request_queue is None or worker_proc is None or worker_ready is None:
        return JSONResponse({"status": "not_initialized"}, status_code=503)
    if worker_ready.is_set() and worker_proc.is_alive():
        return {"status": "ok", "run_id": service_run_id}
    failure = None
    if worker_failure is not None:
        try:
            failure = worker_failure.get_nowait()
        except Exception:
            pass
    return JSONResponse(
        {
            "status": "starting" if worker_proc.is_alive() else "failed",
            "worker_pid": worker_proc.pid,
            "run_id": service_run_id,
            **({"worker_error": failure} if failure else {}),
        },
        status_code=503,
    )


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

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    with _pending_lock:
        _pending[req_id] = fut

    request_dict.setdefault("server_enqueue_monotonic", time.monotonic())
    request_queue.put(request_dict)
    result = await fut
    return result


@app.post("/prefill")
async def prefill(req: PrefillRequest) -> dict:
    if request_queue is None:
        return {"error": "queues_not_initialized"}

    req_id = req.session_id

    draft_output_tensor = torch.tensor(req.draft_output, dtype=torch.long).unsqueeze(0)

    request_dict = {
        "task_id": req.task_id,
        "draft_output": draft_output_tensor,
        "prefix_len": req.prefix_len,
        "proc_id": req_id,
        "lag": req.lag,
        "current_time": req.current_time,
        "task_type": "prefill",
    }

    resp = await _enqueue_and_wait(request_dict, req_id)
    if "error" in resp:
        return {"session_id": req_id, **resp}
    if "status" in resp:
        if resp["status"] != "prefill_ok":
            return {"session_id": req_id, **resp}

        # Preserve the worker's node2-local timing bounds and prefill metrics.
        # The edge does not compare these values with its own monotonic clock;
        # it records them so finalization can crop the cloud GPU samples from
        # the first prefill enqueue through final verification completion.
        out = {"session_id": req_id, "status": "prefill_ok"}
        for key in (
            "prefill_chunks",
            "prefill_queue_ms",
            "prefill_service_ms",
            "server_enqueue_monotonic_s",
            "server_completed_monotonic_s",
        ):
            if key in resp:
                out[key] = float(resp[key]) if key != "prefill_chunks" else int(resp[key])
        return out
    return {"session_id": req_id, "error": "invalid_prefill_worker_response"}


@app.post("/verify")
async def verify(req: VerifyRequest) -> dict:
    api_start = time.monotonic()
    if request_queue is None:
        return {"error": "queues_not_initialized"}

    req_id = req.session_id

    draft_output_tensor = torch.tensor(req.draft_output, dtype=torch.long).unsqueeze(0)
    avg_cloud_total_ms = _get_avg_cloud_total_ms()

    request_dict = {
        "task_id": req.task_id,
        "draft_output": draft_output_tensor,
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
    }

    resp = await _enqueue_and_wait(request_dict, req_id)
    if "error" in resp:
        return {"session_id": req_id, **resp}
    accepted = int(resp["accepted"])
    final_token = resp["final_token"]
    final_token_id = int(final_token.item()) if hasattr(final_token, "item") else int(
        final_token
    )

    out = {"session_id": req_id, "accepted": accepted, "final_token": final_token_id}
    if "verify_ms" in resp:
        out["verify_ms"] = float(resp["verify_ms"])
    if "verify_queue_ms" in resp:
        out["verify_queue_ms"] = float(resp["verify_queue_ms"])
    if "verify_service_ms" in resp:
        out["verify_service_ms"] = float(resp["verify_service_ms"])
    for key in ("server_enqueue_monotonic_s", "server_completed_monotonic_s"):
        if key in resp:
            # This timestamp belongs to node2's monotonic domain.  The edge
            # saves it only so the finalizer can crop node2 GPU samples.
            out[key] = float(resp[key])
    if "suggested_gamma" in resp:
        out["suggested_gamma"] = int(resp["suggested_gamma"])
    # Cloud-side end-to-end service time for this HTTP verify request.
    out["cloud_total_ms"] = (time.monotonic() - api_start) * 1000.0
    _record_cloud_total_ms(out["cloud_total_ms"])
    return out


@app.post("/exit")
def exit_worker() -> dict:
    if request_queue is None:
        return {"status": "queues_not_initialized"}
    request_queue.put(None)
    return {"status": "sent"}


@app.post("/shutdown")
async def shutdown_service(
    x_fastsd_run_id: Optional[str] = Header(default=None),
) -> dict:
    """Request a graceful, local experiment-service shutdown.

    The listener is bound only to the node2 IB address by the run wrapper.  It
    receives no credentials and never discovers/kills a process: it sends the
    scheduler's documented sentinel to the exact worker created by this
    process, then asks this Uvicorn server to leave its own event loop.
    """

    global service_shutdown_requested
    if not service_run_id or x_fastsd_run_id != service_run_id:
        raise HTTPException(
            status_code=409,
            detail="shutdown requires the matching X-FastSD-Run-ID for this service",
        )
    if request_queue is not None:
        request_queue.put(None)
    service_shutdown_requested = True
    return {"status": "shutdown_requested"}


def main() -> None:
    global request_queue, response_queue, worker_proc, worker_ready, worker_failure
    global service_shutdown_requested, service_run_id
    args = parse_arguments()
    service_shutdown_requested = False
    service_run_id = args.run_id

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    worker_ready = ctx.Event()
    worker_failure = ctx.Queue()

    worker_proc = _start_worker(
        args, request_queue, response_queue, worker_ready, worker_failure
    )

    port = int(args.port or os.environ.get("CLOUD_SERVICE_PORT", "8001"))
    bind_host = args.bind_host or os.environ.get("CLOUD_BIND_HOST", "127.0.0.1")
    config = uvicorn.Config(app, host=bind_host, port=port, workers=1)
    server = uvicorn.Server(config)

    def _watch_shutdown_request() -> None:
        while not service_shutdown_requested:
            time.sleep(0.05)
        server.should_exit = True

    threading.Thread(target=_watch_shutdown_request, daemon=True).start()
    server.run()
    if worker_proc is not None:
        worker_proc.join(timeout=30.0)
        if worker_proc.is_alive():
            # This is solely the worker spawned by this server.  A normal
            # experiment shutdown sends its sentinel first; failure to drain
            # is surfaced to the wrapper rather than silently reporting a
            # complete cloud component.
            worker_proc.terminate()
            worker_proc.join(timeout=5.0)
            raise RuntimeError("cloud target worker did not drain after shutdown request")


if __name__ == "__main__":
    main()
