"""Launch the official SpecEdge server on a configurable loopback port."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import torch
import yaml

INTEGRATION_ROOT = Path(__file__).resolve().parent
SPECEDGE_ROOT = INTEGRATION_ROOT.parent / "official"
sys.path.insert(0, str(SPECEDGE_ROOT / "src"))
sys.path.insert(0, str(SPECEDGE_ROOT / "src" / "script"))

import batch_server as official_server
import log
import util
from strategy.server_verify.specexec import grpc as verifier

from memory import BudgetExceeded, DeviceBudget, cache_bytes, plan_budget
from modeling import load_model, load_tokenizer, model_class, model_config
from sampling import probabilities, rejection_selection, select_tokens
from offload import BatchResponses, LRUSlots, SlotView
from settings import OFFICIAL, OFFICIAL_OOM, OFFICIAL_REVISION, apply_runtime_options, check_official_checkout, gpu_slots, result_metadata, validate_config, verify_method_for
from tp import ShardedSlots, TPBatchEngine, check_tp, parameter_bytes, shard_model


def budget_for(config, architecture, devices):
    server, base, client = (config[key] for key in ("server", "base", "client"))
    size = len(devices)
    check_tp(architecture, size)
    with torch.device("meta"):
        estimates = parameter_bytes(model_class(architecture)(architecture), 2, size)
    rows = []
    for rank, device in enumerate(devices):
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info(device)
            if base["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError(f"bf16 unsupported on {device}")
        logits_bytes = (int(client["max_budget"]) + 1) * architecture.vocab_size * 16 if rank == 0 else 0
        rows.append(DeviceBudget(str(device), total, free, estimates[rank], cache_bytes(architecture, int(base["max_len"]), 2, size), logits_bytes))
    return plan_budget(
        rows, instances=int(server["num_clients"]), batch_size=int(server["max_batch_size"]),
        max_len=int(base["max_len"]), max_n_beams=int(client["max_n_beams"]), verifier_width=int(client["max_budget"]) + 1,
        reserve_bytes=int(server.get("reserve_bytes", 2 * 1024**3)), element_size=2,
        allow_memory_clamp=server.get("allow_memory_clamp", False),
        resident_slots=gpu_slots(config),
    )


class AdaptedController(verifier.InferenceController):
    def __init__(self, config, recv_queue, resp_queue):
        server = config["server"]
        instances = int(server["num_clients"])
        self.verify_method = verify_method_for(config, validate_config(config)[0])
        self.sampling = (float(server["temperature"]), float(server.get("top_p", 0.8)), int(server.get("top_k", 0)))
        self.seed = int(config["base"]["seed"])
        self.generators = {}
        self.proposals = {}
        self.slots = None
        self.responses = BatchResponses(resp_queue) if server.get("offload") else None
        super().__init__(0 if self.responses else instances, recv_queue, self.responses or resp_queue)
        if self.responses:
            sample = self.k_cache.new_empty((self.k_cache.shape[0], *self.k_cache.shape[2:]))
            self.slots = LRUSlots(sample, gpu_slots(config), synchronize=lambda: torch.cuda.current_stream(self._device).synchronize())
            self.k_cache, self.v_cache = SlotView(self.slots, 0), SlotView(self.slots, 1)
            self._num_clients = instances
            self._iter_idx = torch.zeros(instances, device=self._device, dtype=torch.long)

    def _inference(self, batch):
        if self.slots is not None:
            self.slots.begin(batch)
        for row, request in enumerate(batch):
            if self.verify_method in {"rejection", "target_only"} and request.prefill:
                self.generators[request.client_idx] = torch.Generator(device=self._device).manual_seed(self.seed + request.req_idx)
            if self.verify_method == "rejection":
                parent_bytes = self._max_budget * 8
                payload = request.parent_indices[parent_bytes:]
                expected = self._max_budget * self._model.config.vocab_size * 4
                if len(payload) != expected:
                    raise ValueError("missing or incompatible single-chain proposal probabilities")
                self.proposals[row] = (request.client_idx, torch.frombuffer(bytearray(payload), dtype=torch.float32)
                                       .reshape(self._max_budget, -1).to(self._device))
                request.parent_indices = request.parent_indices[:parent_bytes]
        result = super()._inference(batch)
        self.proposals.clear()
        if self.responses is not None:
            self.responses.flush(self.slots.stats)
        return result

    def sample(self, logits, temperature):
        if self.verify_method == "rejection":
            selections = torch.zeros_like(self._input_ids)
            for row, (client, draft) in self.proposals.items():
                target = probabilities(logits[row], *self.sampling)
                selections[row] = rejection_selection(target, draft, self._input_ids[row, 1:], self.generators[client])
            return selections
        if self.verify_method == "target_only":
            selections = torch.zeros_like(self._input_ids)
            for row, client in enumerate(self._client_indices.tolist()):
                if client >= 0:
                    probs = probabilities(logits[row], *self.sampling)
                    selections[row] = torch.multinomial(probs, 1, generator=self.generators[client]).squeeze(-1)
            return selections
        return select_tokens(logits, *self.sampling)

    def _reorder_kv_cache(self, selection):
        if self.slots is not None:
            self.slots.writing = True
        if self._max_budget:
            return super()._reorder_kv_cache(selection)
        for batch_idx, client_idx in enumerate(self._client_indices.tolist()):
            if client_idx < 0:
                continue
            cache = self._engine._past_key_values
            self.k_cache[:, client_idx, ...].copy_(cache.k_cache[:, batch_idx, ...])
            self.v_cache[:, client_idx, ...].copy_(cache.v_cache[:, batch_idx, ...])


def init_adapted(config, architecture, devices, recv_queue, resp_queue):
    server = config["server"]
    dtype = official_server.config.dtype
    chat = architecture.model_type == "qwen3" or config["integration"].get("dataset") == "mt_bench"
    tokenizer, signature = load_tokenizer(server["target_model"], server.get("tokenizer", server["target_model"]), architecture, chat=chat)
    model = load_model(server["target_model"], architecture, devices[0] if len(devices) == 1 else "cpu", dtype)
    original_engine = verifier.BatchGraphEngine
    original_heads = model.config.num_key_value_heads
    if server.get("offload"):
        verifier.BatchGraphEngine = lambda **kwargs: original_engine(**kwargs, use_cuda_graph=False)
    if len(devices) == 2:
        shard_model(model, devices)

        def engine(**kwargs):
            result = TPBatchEngine(**kwargs, devices=devices, invert_mask=util.invert_mask)
            model.config.num_key_value_heads = original_heads // len(devices)
            return result

        verifier.BatchGraphEngine = engine

    original_loaders = (util.load_dataset, util.load_tokenizer, util.load_graph_model)
    util.load_dataset = lambda *_args, **_kwargs: []
    util.load_tokenizer = lambda _path: tokenizer
    util.load_graph_model = lambda **_kwargs: model
    try:
        controller = AdaptedController(config, recv_queue, resp_queue)
    finally:
        util.load_dataset, util.load_tokenizer, util.load_graph_model = original_loaders
        verifier.BatchGraphEngine = original_engine
        model.config.num_key_value_heads = original_heads

    if len(devices) == 2:
        for name in ("k_cache", "v_cache"):
            primary = getattr(controller, name)
            setattr(controller, name, ShardedSlots((primary, torch.zeros_like(primary, device=devices[1]))))
    if not config["client"]["max_budget"]:
        decode = util.decode
        util.decode = lambda payload, device, dtype, shape: (
            decode(payload, device, dtype, shape) if payload else torch.empty(0, device=device, dtype=dtype))
    util.sampler_from_logits = controller.sample
    return controller, {"token_signature": signature, "chat": chat}


def inference_worker(config_path, config, devices, method, recv_queue, resp_queue, startup):
    budget = None
    try:
        path = Path(config["base"]["result_path"]) / config["base"]["exp_name"] / "specedge_run.json"
        if path.exists():
            raise FileExistsError(f"result directory already contains {path.name}: {path.parent}")
        official_server._load_config(Path(config_path))
        os.environ["SPECEDGE_MAX_BUDGET"] = str(config["client"]["max_budget"])
        os.environ["SPECEDGE_MAX_N_BEAMS"] = str(config["client"]["max_n_beams"])
        official_server.util.set_seed(int(config["base"]["seed"]))
        logger = log.get_logger()
        server = config["server"]
        architecture = model_config(server["target_model"])
        devices = [torch.device(device) for device in devices]
        budget = budget_for(config, architecture, devices)
        logger.info("SpecEdge startup budget: %s", json.dumps(budget))
        official_server.config.max_batch_size = budget["batch_size"]
        if method == OFFICIAL:
            original_loaders = util.load_dataset, util.load_tokenizer, util.load_graph_model
            handshake = {}
            if config.get("integration"):
                chat = architecture.model_type == "qwen3" or config["integration"].get("dataset") == "mt_bench"
                tokenizer, signature = load_tokenizer(server["target_model"], server.get("tokenizer", server["target_model"]), architecture, chat=chat)
                util.load_dataset = lambda *_args, **_kwargs: []
                util.load_tokenizer = lambda _path: tokenizer
                util.load_graph_model = lambda name, device, dtype: load_model(name, architecture, device, dtype)
                handshake = {"token_signature": signature, "chat": chat}
            try:
                controller = verifier.InferenceController(int(server["num_clients"]), recv_queue, resp_queue)
            finally:
                util.load_dataset, util.load_tokenizer, util.load_graph_model = original_loaders
        else:
            controller, handshake = init_adapted(config, architecture, devices, recv_queue, resp_queue)
        metadata = result_metadata(config, budget)
        metadata["official_revision"] = OFFICIAL_REVISION
        metadata["dataset"] = config.get("integration", {}).get("dataset", config["client"]["dataset"])
        metadata["target_runtime"] = ("eager_lru" if server.get("offload") else
                                      "eager_tp" if len(devices) == 2 else "official_graph_engine")
        metadata["target_gpu_uuids"] = [str(getattr(torch.cuda.get_device_properties(device), "uuid", "")) or None for device in devices]
        logger.info("SpecEdge target: %s", json.dumps(metadata))
        with path.open("x", encoding="utf-8") as output:
            json.dump({**metadata, "memory_budget": budget}, output, indent=2)
        startup.put((None, {**handshake, "run": metadata}))
    except Exception as error:
        if method == OFFICIAL and isinstance(error, (BudgetExceeded, torch.cuda.OutOfMemoryError)):
            if budget is None:
                budget = {"devices": error.devices} if isinstance(error, BudgetExceeded) else None
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as output:
                json.dump(oom_record(config, error, budget), output, indent=2)
        startup.put((f"{type(error).__name__}: {error}", None))
        raise
    try:
        controller.loop()
    except torch.cuda.OutOfMemoryError as error:
        if method == OFFICIAL:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(oom_record(config, error, budget), indent=2), encoding="utf-8")
            temporary.replace(path)
        raise


class IntegratedServer(official_server.SpecExecBatchServer):
    def __init__(self, config_path, config, devices, method, shutdown_event):
        self.config_path = config_path
        self.config = config
        self.devices = devices
        self.method = method
        self.startup = mp.Queue()
        self.handshake = None
        self.arrival_epoch = None
        super().__init__(shutdown_event=shutdown_event)

    def _init_inference_loop(self):
        self._inference_process = mp.Process(
            target=inference_worker,
            args=(self.config_path, self.config, self.devices, self.method, self._recv_queue, self._resp_queue, self.startup),
        )
        self._inference_process.start()

    async def wait_ready(self):
        from queue import Empty

        while not self._shutdown_event.is_set():
            try:
                error, self.handshake = self.startup.get_nowait()
            except Empty:
                if not self._inference_process.is_alive():
                    raise RuntimeError("SpecEdge inference process exited before startup")
                await asyncio.sleep(0.1)
            else:
                if error:
                    raise RuntimeError(error)
                return True
        return False

    async def Sync(self, request, context):
        response = await super().Sync(request, context)
        if dict(context.invocation_metadata()).get("specedge-stage") == "arrivals":
            if self.arrival_epoch is None:
                self.arrival_epoch = time.time() + 2.0
            context.set_trailing_metadata((("specedge-arrival-epoch", str(self.arrival_epoch)),))
            return response
        if self.method != OFFICIAL or self.config.get("integration"):
            context.set_trailing_metadata((("specedge-run-bin", json.dumps(self.handshake).encode()),))
        return response

    async def Validate(self, request, context):
        if not self.config["server"].get("offload"):
            return await super().Validate(request, context)
        if not 0 <= request.client_idx < int(self.config["server"]["num_clients"]):
            await context.abort(official_server.grpc.StatusCode.INVALID_ARGUMENT, "invalid client slot")
        future = asyncio.get_running_loop().create_future()
        with self._resp_lock:
            self._resp_futures[request.client_idx] = future
        self._recv_queue.put(request.SerializeToString())
        selection, prefill, stats = await asyncio.wait_for(future, timeout=300)
        context.set_trailing_metadata((("specedge-io-bin", json.dumps(stats).encode()),))
        return verifier.specedge_pb2.ValidateResponse(selection=selection, prefill=prefill)


async def serve(config_path, config, devices, method, host: str, port: int) -> None:
    official_server.shutdown_event = asyncio.Event()
    official_server.config.num_clients = int(config["server"]["num_clients"])
    controller = IntegratedServer(config_path, config, devices, method, official_server.shutdown_event)
    server = official_server.grpc.aio.server(options=(("grpc.max_receive_message_length", 128 * 1024**2),))
    official_server.specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(
        controller, server
    )
    try:
        if not await controller.wait_ready():
            return
        if server.add_insecure_port(f"{host}:{port}") == 0:
            raise RuntimeError(f"failed to bind SpecEdge gRPC server to {host}:{port}")
        await server.start()
        while not official_server.shutdown_event.is_set():
            if not controller._inference_process.is_alive():
                raise RuntimeError("SpecEdge inference process exited during the run")
            await asyncio.sleep(0.5)
    finally:
        await server.stop(grace=2.0)
        await controller.cleanup()
        controller.startup.close()


def oom_record(config, error, budget):
    batch = int(config["server"]["max_batch_size"])
    integration = config.get("integration", {})
    server = config["server"]
    return dict(status="oom", method=OFFICIAL_OOM, attempted_method=OFFICIAL,
                requested_batch=batch, effective_batch=batch, oom=True, offload=False,
                workload_hash=integration.get("workload_hash"), dataset=integration.get("dataset"),
                client_count=int(server["num_clients"]), arrival_seed=integration.get("arrival_seed"),
                temperature=float(server["temperature"]), top_p=float(server.get("top_p", 0.8)),
                top_k=int(server.get("top_k", 0)), seed=int(config["base"]["seed"]),
                verify_method="target_sample_match", official_revision=OFFICIAL_REVISION,
                memory_budget=budget, error=str(error))


def main(config_path: str, host: str, port: int, budget_only=False) -> int:
    check_official_checkout(SPECEDGE_ROOT)
    with Path(config_path).open(encoding="utf-8") as handle:
        config = apply_runtime_options(yaml.safe_load(handle))
    method, devices = validate_config(config)
    if method == OFFICIAL and sys.version_info[:2] != (3, 14):
        raise RuntimeError("run native SpecEdge server with its Python 3.14 environment")
    if budget_only:
        architecture = model_config(config["server"]["target_model"])
        try:
            print(json.dumps(budget_for(config, architecture, devices), indent=2))
        except BudgetExceeded as error:
            if method != OFFICIAL:
                raise
            record = json.dumps(oom_record(config, error, {"devices": error.devices}), indent=2)
            path = Path(config["base"]["result_path"]) / config["base"]["exp_name"] / "specedge_run.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as output:
                output.write(record)
            print(record)
            return 1
        return 0
    mp.set_start_method("spawn")
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGINT, official_server.signal_handler)
    signal.signal(signal.SIGTERM, official_server.signal_handler)
    asyncio.run(serve(config_path, config, devices, method, host, port))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--budget-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.host, args.port, args.budget_only))
