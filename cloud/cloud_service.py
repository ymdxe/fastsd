import asyncio
import multiprocessing as mp
import os
import sys
import threading
import uuid
from typing import Dict, Optional

import torch
from fastapi import FastAPI
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
    current_time: float


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
    status = "ok" if request_queue is not None else "not_initialized"
    return {"status": status}


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
        return {"session_id": req_id, **resp}
    return {"session_id": req_id, "status": "prefill_ok"}


@app.post("/verify")
async def verify(req: VerifyRequest) -> dict:
    api_start = time.time()
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
    if "suggested_gamma" in resp:
        out["suggested_gamma"] = int(resp["suggested_gamma"])
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
