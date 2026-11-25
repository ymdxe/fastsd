"""SpecEdge client adapter for FastSD canonical datasets and arrival schedules."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import torch

INTEGRATION_ROOT = Path(__file__).resolve().parent
SPECEDGE_ROOT = INTEGRATION_ROOT.parent / "official"
FASTSD_ROOT = INTEGRATION_ROOT.parents[2]
sys.path.insert(0, str(FASTSD_ROOT))
sys.path.insert(0, str(SPECEDGE_ROOT / "src"))

import grpc

import log
import util
from config import SpecEdgeClientConfig as config


from specedge.client.specexec import SpecExecClient
from specedge.engine.graph import GraphEngine
from specedge_grpc import specedge_pb2, specedge_pb2_grpc
from src.evaluation import token_ids_hash
from src.runtime import configure_torch_threads
from src.metrics import request_latency

from cpu_adapter import (
    CPUCompatibleSpecEdgeEngine,
    cpu_timing_adapter,
)
from dataset_adapter import ADAPTERS, DatasetAdapter
from linear import TargetOnlyClient, grow_single_chain
from modeling import compare_token_signatures, load_model, load_tokenizer, model_config
from tree_order import order_tree_compaction_indices
from wire_codec import ExplicitSpecEdgeGrpcClient
from sampling import probabilities
from offload import empty_stats


class IntegratedSpecExecClient(SpecExecClient):
    """Keep official cycles, but enforce the suite's exact output-token budget."""

    def __init__(self, *args, eos_ids, wire_dtype, sampling, seed, native=False, draft_top_k=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The official client constructs its controller internally. Replace
        # only the transport object with the explicit repository adapter; the
        # official Tree/SpecExec/proactive logic remains the execution core.
        if not native:
            self._validator = ExplicitSpecEdgeGrpcClient(
                config.host, self._device, wire_mask_dtype=wire_dtype,
                channel=self._validator._channel, verifier_budget=self._max_budget,
            )
        self.first_token_time = None
        self.eos_ids = torch.tensor(eos_ids, device=self._device)
        self.draft_structure = os.environ.get("SPECEDGE_DRAFT_STRUCTURE", "tree")
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        self.verify_steps = 0
        self.native = native
        self.sampling = sampling if draft_top_k is None else (sampling[0], 1.0, int(draft_top_k))
        self.random = torch.Generator(device=self._device).manual_seed(seed)
        if self._num_original_tokens < 2 or self._num_original_tokens + self._max_new_tokens + self._max_budget > self._max_len:
            raise ValueError(f"prompt={self._num_original_tokens}, output={self._max_new_tokens}, draft budget={self._max_budget} do not fit max_len={self._max_len} (prompt must have at least two tokens)")

    def _grow_tree(self, prefill):
        if self.draft_structure == "single_chain":
            rows = []

            def propose(logits):
                row = probabilities(logits, *self.sampling)
                rows.append(row)
                token = torch.multinomial(row, 1, generator=self.random)
                return row.gather(-1, token).log().squeeze(-1), token.squeeze(-1)

            result = grow_single_chain(self._tree, self._process_candidates, self._max_beam_len, prefill, propose=propose)
            self._validator.draft_probabilities = torch.cat(rows)
            return result
        return super()._grow_tree(prefill)

    async def _validate_tree(self, req_idx, prefill=False):
        draft_status = self._tree.status[self._tree.prefix_len : self._tree.end]
        proposed = int((draft_status >= self._tree.PROCESSED).sum().item())
        fresh, stats = await super()._validate_tree(req_idx, prefill)
        if self.first_token_time is None and fresh.numel():
            self.first_token_time = time.perf_counter()
        self.drafted_tokens += proposed
        self.accepted_tokens += fresh.numel() - 1
        self.verify_steps += 1
        return fresh, stats

    def _trim_by_budget(self):
        if self.native:
            return super()._trim_by_budget()
        """Compact selected nodes without breaking the official tree topology.

        The official selection uses ``sorted=False`` because candidate ranking
        is unrelated to storage order.  ``Tree.gather`` is topology-sensitive,
        however: it remaps parents using the source/destination pairing.  Keep
        the selected set unchanged but restore the append order before that
        remap, so official path validation remains lossless and path-ordered.
        """

        src_indices = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, sorted=False)
            .indices
            + self._tree.prefix_len
        )
        src_indices = order_tree_compaction_indices(src_indices)
        dest_indices = torch.arange(
            self._tree.prefix_len,
            self._tree.prefix_len + src_indices.size(-1),
            device=self._device,
        )

        self._tree.gather(src_indices, dest_indices)
        self._engine.gather(src_indices, dest_indices)

    async def _cycle(self, req_idx: int, step_idx: int, prefill=False):
        if self._device.type == "cpu":
            with cpu_timing_adapter():
                fresh_tokens = await super()._cycle(req_idx, step_idx, prefill=prefill)
        else:
            fresh_tokens = await super()._cycle(req_idx, step_idx, prefill=prefill)
        return fresh_tokens

    async def generate_exact(self, req_idx: int):
        generated = 0
        try:
            for step_idx in range(self._max_new_tokens):
                fresh = await self._cycle(req_idx, step_idx, prefill=step_idx == 0)
                fresh = fresh[..., : self._max_new_tokens - generated]
                eos_positions = torch.isin(fresh, self.eos_ids).nonzero()
                if eos_positions.numel():
                    fresh = fresh[..., : int(eos_positions[0, -1].item()) + 1]
                self._prefix_tokens = torch.cat((self._prefix_tokens, fresh), dim=-1)
                generated += fresh.numel()
                if eos_positions.numel() or generated >= self._max_new_tokens:
                    break
        finally:
            await self._validator._channel.close()


def load_shard() -> list[dict]:
    dataset_file = Path(os.environ["FASTSD_EVAL_DATASET_FILE"])
    with dataset_file.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    num_clients = int(os.environ["FASTSD_EVAL_NUM_CLIENTS"])
    return records[config.client_idx :: num_clients]


async def warmup_specedge(records, new_client) -> int:
    """Run official SpecExec cycles before the timed workload.

    The official result logger records these cycles with negative request IDs;
    the suite normalizer excludes those IDs.  Completion JSONL is only opened
    after this function returns, so warmups cannot enter request-level metrics.
    """

    count = int(os.environ.get("FASTSD_EVAL_WARMUP_REQUESTS", "0"))
    if count < 0:
        raise ValueError("FASTSD_EVAL_WARMUP_REQUESTS must be non-negative")
    if count == 0 or not records:
        return 0
    original_max_new_tokens = int(config.max_new_tokens)
    config.max_new_tokens = 1
    try:
        for warmup_idx in range(count):
            warmup_client = new_client(records[0], ())
            await warmup_client.generate_exact(-1 - warmup_idx)
    finally:
        config.max_new_tokens = original_max_new_tokens
    return count


async def main():
    logger = log.get_logger()
    records = load_shard()
    util.set_seed(config.seed + config.client_idx)

    if config.device.type == "cpu":
        configure_torch_threads(int(os.environ.get("SPECEDGE_THREADS", "1")))
    tokenizer_path = os.environ.get("SPECEDGE_DRAFT_TOKENIZER", config.draft_model)
    logger.info("Loading draft model=%s tokenizer=%s dtype=%s device=%s", config.draft_model, tokenizer_path, config.dtype, config.device)
    native = os.environ["FASTSD_EVAL_METHOD"] == "specedge_official"
    target_only = config.max_budget == 0
    architecture = None if target_only else model_config(config.draft_model)
    draft_model = None if target_only else load_model(config.draft_model, architecture, config.device, config.dtype)
    if target_only:
        engine = None
    elif config.device.type == "cpu":
        engine = CPUCompatibleSpecEdgeEngine(
            model=draft_model,
            max_len=config.max_len,
            max_n_beams=config.max_n_beams,
        )
    else:
        engine = GraphEngine(
            model=draft_model,
            max_len=config.max_len,
            max_n_beams=config.max_n_beams,
        )
    async with grpc.aio.insecure_channel(config.host) as channel:
        stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
        call = stub.Sync(specedge_pb2.SyncRequest(), wait_for_ready=True, timeout=300)
        await call
        handshake = json.loads(dict(await call.trailing_metadata())["specedge-run-bin"])
    run = handshake["run"]
    if run["method"] != os.environ["FASTSD_EVAL_METHOD"]:
        raise ValueError(f"client/server method mismatch: {run['method']}")
    expected = dict(instances=int(os.environ["FASTSD_EVAL_NUM_CLIENTS"]), max_len=config.max_len,
                    gamma=config.max_beam_len, max_n_beams=config.max_n_beams, seed=config.seed, tree_budget=config.max_budget,
                    workload_hash=os.environ["FASTSD_EVAL_WORKLOAD_HASH"], max_tokens=config.max_new_tokens)
    differences = [key for key, value in expected.items() if run.get(key) != value]
    if differences:
        raise ValueError(f"client/server config mismatch: {', '.join(differences)}")
    chat = handshake["chat"]
    if target_only:
        tokenizer_path = run["tokenizer"]
        architecture = model_config(run["target_model"])
    canonical_prompt = os.environ.get("FASTSD_EVAL_CANONICAL_PROMPT") == "1"
    tokenizer, signature = load_tokenizer(run["target_model"] if target_only else config.draft_model,
                                          tokenizer_path, architecture, chat=chat and (target_only or not canonical_prompt))
    compare_token_signatures(handshake["token_signature"], signature)
    if canonical_prompt and not target_only:
        tokenizer, prompt_signature = load_tokenizer(run["target_model"], run["tokenizer"], model_config(run["target_model"]), chat=chat)
        compare_token_signatures(handshake["token_signature"], prompt_signature)
    run.update(draft_model=config.draft_model, draft_tokenizer=tokenizer_path,
               draft_dtype=str(config.dtype).removeprefix("torch."), draft_device=str(config.device))

    def new_client(record, history):
        adapter: DatasetAdapter = ADAPTERS[record["dataset"]]
        prompt = adapter.build_prompt(record, tokenizer, chat=chat, history=history)
        wire_dtype = util.convert_dtype(run["target_dtype"])
        if target_only:
            validator = ExplicitSpecEdgeGrpcClient(config.host, torch.device("cpu"), wire_mask_dtype=wire_dtype, verifier_budget=0)
            return TargetOnlyClient(tokenizer, prompt, signature["eos"]["effective"], validator,
                                    client_idx=config.client_idx, max_len=config.max_len, max_new_tokens=config.max_new_tokens)
        return IntegratedSpecExecClient(
            engine=engine, tokenizer=tokenizer, prompt=prompt, native=native,
            max_len=config.max_len, eos_ids=signature["eos"]["effective"], wire_dtype=wire_dtype,
            sampling=(run["temperature"], run["top_p"], run["top_k"]),
            draft_top_k=run.get("draft_top_k"),
            seed=config.seed + int(record["global_index"]),
        )

    warmup_count = await warmup_specedge(records, new_client)

    if os.environ.get("FASTSD_EVAL_ARRIVAL_DISTRIBUTION") == "poisson":
        async with grpc.aio.insecure_channel(config.host) as channel:
            stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
            call = stub.Sync(specedge_pb2.SyncRequest(), timeout=300, metadata=(("specedge-stage", "arrivals"),))
            await call
            start_epoch = float(dict(await call.trailing_metadata())["specedge-arrival-epoch"])
    else:
        start_epoch = float(os.environ["FASTSD_EVAL_START_EPOCH"])
    # A slow model warmup must not shift the scheduled workload origin.
    delay = start_epoch - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
    completion_dir = Path(os.environ["FASTSD_EVAL_COMPLETION_DIR"])
    completion_dir.mkdir(parents=True, exist_ok=True)
    output_path = completion_dir / f"client_{config.client_idx}_requests.jsonl"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite SpecEdge completion output: {output_path}")

    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            scheduled = float(record.get("scheduled_arrival_s", 0.0))
            if os.environ.get("FASTSD_EVAL_ARRIVAL_DISTRIBUTION") == "poisson":
                remaining = start_epoch + scheduled - time.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                actual_arrival = max(0.0, time.time() - start_epoch)
            else:
                actual_arrival = 0.0

            request_start = time.perf_counter()
            processing_start_s = time.time() - start_epoch
            adapter = ADAPTERS[record["dataset"]]
            completions = []
            output_token_ids = []
            turn_results = []
            drafted_tokens = accepted_tokens = verify_steps = 0
            first_token_time = None
            transport_stats = dict(request_bytes=0, response_bytes=0, rpc_count=0)
            io_stats = empty_stats()
            turn_count = adapter.turn_count(record)
            if os.environ.get("SPECEDGE_MT_BENCH_TURN_POLICY") == "first_turn_only":
                turn_count = 1
            for turn in range(turn_count):
                client = new_client(record, completions)
                prompt_ids = client._prefix_tokens[0].tolist()
                await client.generate_exact(int(record["global_index"]))
                if first_token_time is None:
                    first_token_time = client.first_token_time
                ids = client._prefix_tokens[0, client._num_original_tokens :].tolist()
                completion = DatasetAdapter.extract_answer(adapter, tokenizer.decode(ids, skip_special_tokens=True)) if record["dataset"] == "mt_bench" else tokenizer.decode(ids, skip_special_tokens=True)
                completions.append(completion)
                prompt_hash = token_ids_hash(prompt_ids)
                turn_results.append(dict(turn_index=turn, output_text=completion, output_token_ids=ids,
                                         input_ids_sha256=prompt_hash))
                output_token_ids.extend(ids)
                drafted_tokens += client.drafted_tokens
                accepted_tokens += client.accepted_tokens
                verify_steps += client.verify_steps
                if not native:
                    for key, value in client._validator.snapshot_stats().items():
                        transport_stats[key] += value
                    if run["offload"] and not client._validator.io_stall.get("io_stall_available"):
                        raise RuntimeError("missing SpecEdge offload metrics")
                    for key, value in client._validator.io_stall.items():
                        if key != "io_stall_available":
                            io_stats[key] += value
            completion_time = time.perf_counter()
            request_e2e_ms = (completion_time - request_start) * 1000.0
            generated_tokens = len(output_token_ids)
            ttft_ms, tpot_ms, ttft_fallback = request_latency(
                request_e2e_ms, (first_token_time - request_start) * 1000.0 if first_token_time is not None else None,
                generated_tokens,
            )
            payload = {
                **run,
                "sample_id": record["sample_id"],
                "global_index": int(record["global_index"]),
                "client_idx": int(config.client_idx),
                "dataset": record["dataset"],
                "scheduled_arrival_s": scheduled,
                "actual_arrival_s": actual_arrival,
                "processing_start_s": processing_start_s,
                "completion_s": processing_start_s + request_e2e_ms / 1000.0,
                "processing_start_epoch_s": start_epoch + processing_start_s,
                "completion_epoch_s": start_epoch + processing_start_s + request_e2e_ms / 1000.0,
                "arrival_lag_ms": max(0.0, (actual_arrival - scheduled) * 1000.0),
                "latency_basis": "request_processing_start",
                "e2e_ms": request_e2e_ms,
                "request_e2e_ms": request_e2e_ms,
                "generated_tokens": generated_tokens,
                **io_stats,
                "io_stall_restore_count": io_stats["restores"],
                "request_throughput_tok_s": generated_tokens / (request_e2e_ms / 1000.0) if request_e2e_ms > 0 else None,
                "ttft_ms": ttft_ms,
                "ttft_fallback_to_e2e": ttft_fallback,
                "tpot_ms": tpot_ms,
                "output_text": completions[0] if len(completions) == 1 else "\n\n".join(completions),
                "output_token_ids": output_token_ids,
                "turns": turn_results,
                "drafted_tokens": drafted_tokens,
                "accepted_tokens": accepted_tokens,
                "verify_steps": verify_steps,
                "acceptance_rate": accepted_tokens / drafted_tokens if drafted_tokens else 0.0,
                "input_ids_sha256": turn_results[0]["input_ids_sha256"],
                "eos_token_ids": signature["eos"]["effective"],
                "stop_policy": "eos",
                "accept_rate": accepted_tokens / drafted_tokens if drafted_tokens else 0.0,
                "accept_length": accepted_tokens / verify_steps if verify_steps else 0.0,
                "reference": record.get("reference"),
                "prompt": record["prompt"],
                "workload_hash": os.environ.get("FASTSD_EVAL_WORKLOAD_HASH"),
                "engine_adapter": (
                    "specedge_cpu_adapted"
                    if config.device.type == "cpu"
                    else "official_graph_engine"
                ),
                "warmup_requests_per_client": warmup_count,
                "warmup_included_in_metrics": False,
                "request_bytes": transport_stats["request_bytes"],
                "response_bytes": transport_stats["response_bytes"],
                "rpc_count": transport_stats["rpc_count"],
                "transport_bytes_definition": (
                    "application-layer protobuf request/response ByteSize; excludes gRPC framing, "
                    "TCP/IP, and SSH encapsulation"
                ),
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    result_path = Path(config.result_path) / config.exp_name
    log_config = log.get_default_log_config(result_path, config.process_name)
    log.configure_logging(log_config)
    log.log_unexpected_exception()
    asyncio.run(main())
