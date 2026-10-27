"""SpecEdge gRPC Target with single-chain variable-width batches and FP16 offload.

The frozen official custom Qwen3, KVCache, BatchGraphEngine, gRPC wire messages,
client SpecExec verification, and proactive drafting are retained. This adapter
replaces fixed-width server admission/cache bookkeeping so gamma100 and gamma20
clients can share one Target. CUDA graphs are disabled for CPU weight offload.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import secrets
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from bootstrap import install
install()

import grpc
import grpc.aio
import torch

import log
import util
from config import SpecEdgeBatchServerConfig as config
from model.qwen3 import Qwen3ForCausalLM
from specedge.engine.graph import BatchGraphEngine
from specedge_grpc import specedge_pb2, specedge_pb2_grpc
from strategy.server_verify.specexec.grpc import SpecExecBatchServer

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def decode(payload: bytes, dtype: torch.dtype, shape) -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape)


def chain_accepted(tokens: list[int], choices: list[int]) -> int:
    """Count matching draft tokens, excluding the known root and bonus token."""
    accepted = 0
    for proposed, target_choice in zip(tokens[1:], choices):
        if proposed != target_choice:
            break
        accepted += 1
    return accepted


def validate_chain_layout(indices: list[int], positions: list[int], parents: list[int]) -> int:
    if len(indices) < 2 or len(parents) != len(indices) - 1:
        raise ValueError("Each request must contain one root and a nonempty draft chain")
    root = indices[0]
    if root < 0 or indices != list(range(root, root + len(indices))):
        raise ValueError("Single-chain cache indices must be consecutive and nonnegative")
    if positions != indices or parents != indices[:-1]:
        raise ValueError("Request positions/parents are not a single chronological chain")
    return root


class ChainInferenceController:
    def __init__(self, num_clients, recv_queue, resp_queue):
        self._recv_queue, self._resp_queue = recv_queue, resp_queue
        self.num_clients = num_clients
        self.batch_size = config.max_batch_size
        self.max_len = config.max_len
        self.device = config.device
        self.dtype = config.dtype
        self.output = Path(os.environ["SPECEDGE_EXPERIMENT_OUTPUT"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.iterations = 0
        self.total_requests = 0
        self.gamma_counts = {}
        self.phase = os.environ.get("SPECEDGE_PHASE", "formal")
        self.gammas = {"node1": int(os.environ.get("SPECEDGE_NODE1_GAMMA", "100")),
                       "node3": int(os.environ.get("SPECEDGE_NODE3_GAMMA", "20"))}
        if self.dtype != torch.float16 or str(self.device) != "cuda:0":
            raise ValueError("This experiment requires FP16 on logical cuda:0")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != os.environ["EXP_GPU_INDEX"]:
            raise ValueError("This experiment is restricted to physical GPU3")
        if config.temperature != 0 or config.cache_prefill:
            raise ValueError("Use greedy temperature=0 and cache_prefill=false")
        self.tokenizer = util.load_tokenizer(config.target_model)
        self.model = Qwen3ForCausalLM.from_pretrained(
            config.target_model,
            device_map={"": "cuda:0"},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_state_dict=True,
        ).eval()
        mapping = {str(k): str(v) for k, v in self.model.hf_device_map.items()}
        if "disk" in mapping.values() or str(self.model.device) != "cuda:0":
            raise RuntimeError("Expected GPU3 execution with CPU-only weight offload")
        self.engine = BatchGraphEngine(
            model=self.model,
            max_len=self.max_len,
            max_batch_size=self.batch_size,
            max_n_beams=max(self.gammas.values())+1,
            use_cuda_graph=False,
        )
        torch.cuda.synchronize(0)
        record = {
            "ready": True, "pid": os.getpid(), "loaded_epoch": time.time(),
            "model": config.target_model, "dtype": "float16", "physical_gpu": int(os.environ["EXP_GPU_INDEX"]),
            "weight_offload": False,
            "hf_device_map": mapping, "max_len": self.max_len,
            "max_batch_size": self.batch_size, "cuda_graphs": False,
            "phase": self.phase, "gammas": self.gammas,
            "token_budget": None,
            "client_kv_storage": "cpu", "active_batch_kv_storage": "cuda:0",
            "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(0),
            "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(0),
        }
        write_json(self.output / "target_device_map.json", record)
        write_json(self.output / "target_model_ready.json", record)
        print("SPECEDGE_TARGET_READY " + json.dumps(record), flush=True)

    def loop(self):
        stopping = False
        while not stopping:
            try:
                raw = self._recv_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if raw is None:
                break
            if isinstance(raw, dict):
                self.control(raw)
                continue
            batch = [specedge_pb2.ValidateRequest.FromString(raw)]
            deferred_control = None
            while len(batch) < self.batch_size:
                try:
                    raw = self._recv_queue.get_nowait()
                except queue.Empty:
                    break
                if raw is None:
                    stopping = True
                    break
                if isinstance(raw, dict):
                    deferred_control = raw
                    break
                batch.append(specedge_pb2.ValidateRequest.FromString(raw))
            self.inference(batch)
            if deferred_control is not None:
                self.control(deferred_control)
        self.save_metrics("scheduler_metrics.json")

    def save_metrics(self, filename):
        write_json(self.output / filename, {
            "batches": self.iterations, "verify_requests": self.total_requests,
            "gamma_counts": self.gamma_counts, "client_states": len(self.cache),
            "phase": self.phase, "token_budget": None,
            "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
        })

    def control(self, message):
        if message.get("operation") != "begin_formal":
            raise ValueError("Unknown internal control message")
        self.save_metrics(f"scheduler_metrics-{self.phase}.json")
        self.cache.clear()
        self.engine._past_key_values.clear()
        self.iterations = self.total_requests = 0
        self.gamma_counts = {}
        self.phase = "formal"
        self.gammas = {"node1": 100, "node3": 20}
        ready = json.loads((self.output / "target_model_ready.json").read_text())
        ready.update(phase=self.phase, gammas=self.gammas, formal_started_epoch=time.time(),
                     model_reloaded=False)
        write_json(self.output / "target_model_ready.json", ready)
        write_json(self.output / "formal_configuration.json", ready)
        write_json(Path(message["ack_path"]), {"ok": True, "phase": self.phase,
                                              "gammas": self.gammas, "client_kv_cleared": True,
                                              "model_reloaded": False})

    @torch.inference_mode()
    def inference(self, batch):
        started = time.perf_counter()
        if len({r.client_idx for r in batch}) != len(batch):
            raise ValueError("A batch cannot contain two in-flight requests for one client")
        rows = []
        for request in batch:
            ids = decode(request.input_ids, torch.long, (-1,))
            width = ids.numel()
            gamma = width - 1
            expected = self.gammas["node1" if request.client_idx < 16 else "node3"]
            if gamma != expected:
                raise ValueError(f"client {request.client_idx}: gamma {gamma}, expected {expected}")
            positions = decode(request.position_ids, torch.long, (-1,))
            indices = decode(request.cache_seq_indices, torch.long, (-1,))
            parents = decode(request.parent_indices, torch.long, (-1,))
            root = validate_chain_layout(indices.tolist(), positions.tolist(), parents.tolist())
            if indices.numel() != width or root + width > self.max_len - max(self.gammas.values()) - 1:
                raise ValueError("Request exceeds the live-token region of the static cache")
            mask = decode(request.attention_mask, self.dtype, (width, self.max_len))
            expected_mask = torch.arange(self.max_len)[None, :] <= indices[:, None]
            if not torch.equal(mask, expected_mask.to(self.dtype)):
                raise ValueError("Single-chain attention mask must exactly match causal ancestry")
            rows.append((request, ids, positions, indices, root, width, mask))
            self.gamma_counts[str(gamma)] = self.gamma_counts.get(str(gamma), 0) + 1

        width = max(row[5] for row in rows)
        batch_size = self.batch_size
        cache = self.engine._past_key_values
        cache.clear()
        # KVCache.update returns the whole fixed-size batch cache, so inactive
        # rows are explicit dummy rows. Their outputs never reach clients.
        input_ids = torch.full((batch_size, width), self.tokenizer.pad_token_id,
                               dtype=torch.long, device=self.device)
        positions = torch.zeros_like(input_ids)
        cache_indices = torch.arange(self.max_len - width, self.max_len,
                                     device=self.device).repeat(batch_size, 1)
        attention_mask = torch.zeros((batch_size, 1, width, self.max_len),
                                      device=self.device, dtype=self.dtype)
        attention_mask[..., 0] = 1
        prefill_count = 0
        for slot, (request, ids, pos, indices, root, count, mask) in enumerate(rows):
            if request.prefill:
                prompt_ids = self.tokenizer.encode(request.prefix, return_tensors="pt").to(self.device)
                if prompt_ids.numel() - 1 != root or int(prompt_ids[0, -1]) != int(ids[0]):
                    raise ValueError("Runtime prefill prompt does not match the client's chain root")
                if root:
                    prefix_positions = torch.arange(root, device=self.device).unsqueeze(0)
                    prefix_mask = torch.zeros((1, 1, root, self.max_len),
                                               device=self.device, dtype=self.dtype)
                    prefix_mask[..., :root] = torch.ones((root, root),
                                                        device=self.device, dtype=self.dtype).tril()
                    self.engine.prefill(
                        input_ids=prompt_ids[:, :-1], position_ids=prefix_positions,
                        batch_idx=slot, cache_seq_indices=prefix_positions.flatten(),
                        attention_mask=prefix_mask,
                    )
                prefill_count += 1
                self.cache.pop(request.client_idx, None)
            else:
                saved = self.cache.get(request.client_idx)
                if saved is None or saved[2] != root or saved[3] != request.req_idx:
                    raise ValueError(f"client {request.client_idx}: prefix/cache continuity mismatch")
                cache.k_cache[:, slot, :, :root, :].copy_(saved[0])
                cache.v_cache[:, slot, :, :root, :].copy_(saved[1])
            input_ids[slot, :count] = ids.to(self.device)
            positions[slot, :count] = pos.to(self.device)
            cache_indices[slot, :count] = indices.to(self.device)
            attention_mask[slot, 0, :count] = mask.to(self.device)

        # Source seq_indices select positions in this call's K/V outputs, not
        # absolute positions in the persistent tree cache.
        cache.seq_indices = torch.arange(width, device=self.device).repeat(batch_size)
        batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(width)
        torch.cuda.synchronize(0)
        forward_started = time.perf_counter()
        logits = self.engine.forward(
            input_ids=input_ids, position_ids=positions,
            cache_batch_indices=batch_indices,
            cache_seq_indices=cache_indices.flatten(), attention_mask=attention_mask,
        )
        selection = util.sampler_from_logits(logits, temperature=0)
        torch.cuda.synchronize(0)
        forward_ms = (time.perf_counter() - forward_started) * 1000
        records = []
        responses = []
        for slot, (request, ids, pos, indices, root, count, mask) in enumerate(rows):
            choices = selection[slot, :count].to("cpu", dtype=torch.long).contiguous()
            accepted = chain_accepted(ids.tolist(), choices.tolist())
            # Exactly the official single-chain gather result: retain root and
            # its accepted descendants. The bonus token is the next request's
            # uncached root. Padding queries are never admitted to this prefix.
            length = root + accepted + 1
            self.cache[request.client_idx] = (
                cache.k_cache[:, slot, :, :length, :].to("cpu").clone(),
                cache.v_cache[:, slot, :, :length, :].to("cpu").clone(),
                length, request.req_idx,
            )
            responses.append((choices.numpy().tobytes(), request.client_idx))
            records.append({
                "client_idx": request.client_idx, "req_idx": request.req_idx,
                "prefill": request.prefill, "gamma": count - 1,
                "input_tokens_including_root": count, "accepted_draft_tokens": accepted,
                "retained_cache_tokens": length,
            })
        server_ms = (time.perf_counter() - started) * 1000
        for payload, client_idx in responses:
            self._resp_queue.put(((payload, prefill_count), client_idx))
        self.iterations += 1
        self.total_requests += len(rows)
        with (self.output / f"batches-{self.phase}.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "batch": self.iterations, "phase": self.phase, "completed_epoch": time.time(),
                "active_requests": len(rows), "padded_batch_size": batch_size,
                "query_width": width, "forward_ms": forward_ms,
                "server_ms": server_ms, "requests": records,
                "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(0),
            }) + "\n")


def inference_worker(num_clients, recv_queue, resp_queue):
    output = Path(os.environ["SPECEDGE_EXPERIMENT_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    log.configure_logging(log.get_default_log_config(output, "server_worker"))
    write_json(output / "target_model_ready.json", {"ready": False, "pid": os.getpid(), "state": "loading"})
    try:
        torch.set_num_threads(8)
        controller = ChainInferenceController(num_clients, recv_queue, resp_queue)
        controller.loop()
        write_json(output / "target_model_ready.json", {"ready": False, "pid": os.getpid(), "state": "stopped"})
    except BaseException as exc:
        write_json(output / "worker_error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        write_json(output / "target_model_ready.json", {"ready": False, "pid": os.getpid(), "state": "failed", "error": repr(exc)})
        raise


class ExperimentGrpcServer(SpecExecBatchServer):
    def __init__(self, *args, **kwargs):
        self._sync_expected = int(os.environ.get("SPECEDGE_SYNC_CLIENTS", "16"))
        self.phase = os.environ.get("SPECEDGE_PHASE", "formal")
        super().__init__(*args, **kwargs)

    def _init_inference_loop(self):
        self._inference_process = mp.get_context("spawn").Process(
            target=inference_worker,
            args=(self._num_clients, self._recv_queue, self._resp_queue), daemon=False,
        )
        self._inference_process.start()

    async def Sync(self, request, context):
        expected = self._sync_expected
        async with self._all_sync:
            self._synced += 1
            if self._synced == expected:
                self._synced = 0
                self._all_sync.notify_all()
            else:
                await asyncio.wait_for(self._all_sync.wait(), timeout=7200)
        return specedge_pb2.SyncResponse()

    async def Validate(self, request, context):
        if not 0 <= request.client_idx < self._num_clients:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid client_idx")
        if not self._inference_process.is_alive():
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Target worker has exited")
        future = asyncio.get_running_loop().create_future()
        with self._resp_lock:
            previous = self._resp_futures.get(request.client_idx)
            duplicate = previous is not None and not previous.done()
            if not duplicate:
                self._resp_futures[request.client_idx] = future
        if duplicate:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "client already has an in-flight request")
        self._recv_queue.put(request.SerializeToString())
        selection, prefill_count = await asyncio.wait_for(future, timeout=7200)
        return specedge_pb2.ValidateResponse(selection=selection, prefill=prefill_count)


def start_management(controller, shutdown, loop, output, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def health(self):
            ready_path = output / "target_model_ready.json"
            record = json.loads(ready_path.read_text()) if ready_path.exists() else {"ready": False}
            alive = controller._inference_process.is_alive()
            record.update(worker_alive=alive, service_pid=os.getpid(),
                          phase=controller.phase, sync_clients=controller._sync_expected)
            record["ready"] = bool(record.get("ready") and alive)
            return record

        def do_GET(self):
            if self.path == "/health":
                return self.reply(self.health())
            return self.reply({"error": "unknown endpoint"}, 404)

        def do_POST(self):
            if self.path == "/shutdown":
                self.reply({"status": "shutting_down"})
                loop.call_soon_threadsafe(shutdown.set)
                return
            if self.path != "/phase":
                return self.reply({"error": "unknown endpoint"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (ValueError, TypeError):
                return self.reply({"error": "invalid JSON"}, 400)
            if body != {"phase": "formal"}:
                return self.reply({"error": "only phase=formal is supported"}, 400)
            if controller.phase != "smoke":
                return self.reply({"error": "formal transition is allowed once, after smoke"}, 409)
            if not self.health()["ready"]:
                return self.reply({"error": "model is not ready"}, 409)
            with controller._resp_lock:
                pending = any(not future.done() for future in controller._resp_futures.values())
            if pending or controller._synced:
                return self.reply({"error": "clients still have pending work"}, 409)
            ack = output / ("control-" + secrets.token_hex(8) + ".json")
            controller._recv_queue.put({"operation": "begin_formal", "ack_path": str(ack)})
            deadline = time.monotonic() + 60
            while not ack.exists():
                if time.monotonic() > deadline or not controller._inference_process.is_alive():
                    return self.reply({"error": "Target did not acknowledge phase transition"}, 503)
                time.sleep(0.1)
            with controller._resp_lock:
                controller._resp_futures.clear()
            controller._sync_expected = 16
            controller.phase = "formal"
            return self.reply(json.loads(ack.read_text()))

    management = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=management.serve_forever, daemon=True).start()
    return management


async def serve(host, port, output, management_port):
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)
    controller = ExperimentGrpcServer(shutdown_event=shutdown)
    management = None
    server = grpc.aio.server()
    specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(controller, server)
    try:
        if server.add_insecure_port(f"{host}:{port}") == 0:
            raise RuntimeError(f"Could not bind gRPC server on {host}:{port}")
        management = start_management(controller, shutdown, loop, output, management_port)
        await server.start()
        write_json(output / "service.json", {"pid": os.getpid(), "host": host, "port": port,
                                             "worker_pid": controller._inference_process.pid,
                                             "management_host": "127.0.0.1", "management_port": management_port})
        while not shutdown.is_set():
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=1)
            except asyncio.TimeoutError:
                if not controller._inference_process.is_alive():
                    raise RuntimeError("SpecEdge Target worker exited; inspect worker_error.json")
    finally:
        if management is not None:
            management.shutdown()
            management.server_close()
        await server.stop(grace=2)
        await controller.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT.parent / "model-view" / "Qwen3-32B")
    parser.add_argument("--output", type=Path, default=ROOT / "target")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18625)
    parser.add_argument("--management-port", type=int, default=18626)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--sync-clients", type=int, default=16)
    parser.add_argument("--node1-gamma", type=int, default=100)
    parser.add_argument("--node3-gamma", type=int, default=20)
    parser.add_argument("--phase", choices=["smoke", "formal"], default="formal")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != os.environ["EXP_GPU_INDEX"]:
        raise ValueError("Launch with CUDA_VISIBLE_DEVICES=3")
    if not (1 <= args.node1_gamma <= 500 and 1 <= args.node3_gamma <= 500):
        raise ValueError("Gamma must be in [1,100] for this experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "SPECEDGE_EXPERIMENT_OUTPUT": str(args.output), "SPECEDGE_RESULT_PATH": str(args.output),
        "SPECEDGE_EXP_NAME": "official_cycles", "SPECEDGE_PROCESS_NAME": "server",
        "SPECEDGE_SEED": "1234", "SPECEDGE_MAX_LEN": str(args.max_len),
        "SPECEDGE_BATCH_TYPE": "dynamic", "SPECEDGE_DATASET": "fastsd_external",
        "SPECEDGE_SAMPLE_REQ_CNT": "1", "SPECEDGE_REQ_OFFSET": "0",
        "SPECEDGE_TARGET_MODEL": str(args.model), "SPECEDGE_SERVER_DEVICE": "cuda:0",
        "SPECEDGE_DTYPE": "fp16", "SPECEDGE_TEMPERATURE": "0",
        "SPECEDGE_MAX_BATCH_SIZE": str(args.batch_size), "SPECEDGE_MAX_N_BEAMS": "1",
        "SPECEDGE_MAX_BUDGET": str(max(args.node1_gamma,args.node3_gamma)), "SPECEDGE_NUM_CLIENTS": "32",
        "SPECEDGE_CACHE_PREFILL": "False", "SPECEDGE_SYNC_CLIENTS": str(args.sync_clients),
        "SPECEDGE_NODE1_GAMMA": str(args.node1_gamma), "SPECEDGE_NODE3_GAMMA": str(args.node3_gamma),
        "SPECEDGE_PHASE": args.phase,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(settings)
    mp.set_start_method("spawn", force=True)
    log.configure_logging(log.get_default_log_config(args.output, "server"))
    write_json(args.output / "server_config.json", settings)
    asyncio.run(serve(args.host, args.port, args.output, args.management_port))


if __name__ == "__main__":
    main()
