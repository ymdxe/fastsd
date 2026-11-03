import asyncio
import multiprocessing as mp
import os
import sys
import threading
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, NonNegativeInt

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

# FastAPI 请求等待表
_pending: Dict[str, asyncio.Future] = {}
_pending_lock = threading.Lock()
_prefill_arrivals: Dict[str, float] = {}
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


class SessionRequest(BaseModel):
    session_id: str


class TaskRequest(SessionRequest):
    task_id: str
    draft_output: list[NonNegativeInt] = Field(min_length=1)
    prefix_len: int = Field(gt=0)
    lag: float = Field(ge=0, allow_inf_nan=False)
    current_time: float = Field(allow_inf_nan=False)
    gamma: int = Field(gt=0)
    draft_time_per_token: float = Field(ge=0, allow_inf_nan=False)
    edge_send_time: float = Field(allow_inf_nan=False)
    clock_offset: float = Field(allow_inf_nan=False)


class PrefillRequest(TaskRequest):
    prefill_time: float = Field(ge=0, allow_inf_nan=False)


class VerifyRequest(TaskRequest):
    pull_time: float = Field(ge=0, allow_inf_nan=False)
    transport_rtt: float = Field(default=0, ge=0, allow_inf_nan=False)
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
            msg = response_queue.get()
            req_id = msg.get("request_id")
            payload = msg.get("payload", {})
            loop.call_soon_threadsafe(_deliver, req_id, payload)

    def _deliver(req_id, payload):
        with _pending_lock:
            fut = _pending.pop(req_id, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


@app.on_event("startup")
async def startup_event() -> None:
    """
    FastAPI 启动时启动结果分发线程。
    队列与 Worker 由 main 初始化。
    """
    loop = asyncio.get_event_loop()
    _start_result_dispatcher(loop)


@app.get("/health")
def health() -> dict:
    status = "ok" if request_queue is not None else "not_initialized"
    return {"status": status}


@app.post("/session/init")
def session_init() -> dict:
    """
    返回新的 session_id（UUID）。
    """
    print("Initializing new session...")
    return {"session_id": uuid.uuid4().hex}


@app.post("/clock/sync")
async def clock_sync() -> dict:
    received = time.time()
    return {"received": received, "sent": time.time()}


@app.post("/prefill/arrival")
async def prefill_arrival(req: SessionRequest) -> dict:
    if req.session_id in _prefill_arrivals:
        raise HTTPException(409, "prefill arrival already registered")
    _prefill_arrivals[req.session_id] = time.monotonic()
    return {"status": "registered"}


async def _enqueue_and_wait(request_dict: dict, req_id: str) -> dict:
    if request_queue is None:
        raise HTTPException(503, "queues_not_initialized")

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    with _pending_lock:
        if req_id in _pending:
            raise HTTPException(409, "session already has unfinished work")
        _pending[req_id] = fut

    request_queue.put(request_dict)
    result = await asyncio.shield(fut)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/session/release")
async def session_release(req: SessionRequest) -> dict:
    result = await _enqueue_and_wait(
        {"proc_id": req.session_id, "task_type": "release"}, req.session_id)
    _prefill_arrivals.pop(req.session_id, None)
    return result


@app.post("/prefill")
async def prefill(req: PrefillRequest) -> dict:
    req_id = req.session_id
    if req_id not in _prefill_arrivals:
        raise HTTPException(400, "register arrival before generating the first draft")
    if len(req.draft_output) != req.prefix_len:
        raise HTTPException(400, "prefill must contain exactly the prompt")
    request_dict = req.model_dump()
    request_dict.update({
        "proc_id": req_id,
        "task_type": "prefill",
        "arrival_time": _prefill_arrivals[req_id],
        "push_time": max(0.0, time.time() - req.edge_send_time - req.clock_offset),
    })
    resp = await _enqueue_and_wait(request_dict, req_id)
    _prefill_arrivals.pop(req_id)
    return {"session_id": req_id, "status": resp["status"], "cloud_send_time": time.time()}


@app.post("/verify")
async def verify(req: VerifyRequest) -> dict:
    api_start = time.monotonic()
    req_id = req.session_id
    request_dict = req.model_dump()
    request_dict.update({
        "proc_id": req_id,
        "arrival_time": api_start,
        "push_time": max(0.0, time.time() - req.edge_send_time - req.clock_offset),
        "avg_cloud_total_ms": _get_avg_cloud_total_ms(),
        "task_type": "verify",
    })

    resp = await _enqueue_and_wait(request_dict, req_id)
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
    # Cloud-side end-to-end service time for this HTTP verify request.
    out["cloud_total_ms"] = (time.monotonic() - api_start) * 1000.0
    _record_cloud_total_ms(out["cloud_total_ms"])
    out["cloud_send_time"] = time.time()
    return out


@app.post("/exit")
def exit_worker() -> dict:
    if request_queue is None:
        return {"status": "queues_not_initialized"}
    request_queue.put(None)
    return {"status": "sent"}


def main() -> None:
    global request_queue, response_queue, worker_proc
    args = parse_arguments()

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()

    worker_proc = _start_worker(args, request_queue, response_queue)

    port = int(os.environ.get("CLOUD_SERVICE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
