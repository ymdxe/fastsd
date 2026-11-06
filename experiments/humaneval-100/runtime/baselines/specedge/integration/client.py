"""SpecEdge client adapter for FastSD canonical datasets and arrival schedules."""

from __future__ import annotations

import asyncio
import json
import os
import random
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
from src.runtime import configure_torch_threads

from cpu_adapter import (
    CPUCompatibleSpecEdgeEngine,
    CPUCompatibleTiming,
    cpu_timing_adapter,
)
from tree_order import order_tree_compaction_indices
from wire_codec import ExplicitSpecEdgeGrpcClient


class IntegratedSpecExecClient(SpecExecClient):
    """Keep official cycles, but enforce the suite's exact output-token budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The official client constructs its controller internally. Replace
        # only the transport object with the explicit repository adapter; the
        # official Tree/SpecExec/proactive logic remains the execution core.
        self._validator = ExplicitSpecEdgeGrpcClient(
            config.host,
            self._device,
            wire_mask_dtype=torch.bfloat16,
        )
        self.measurement_start = None
        self.first_token_time = None

    def _trim_by_budget(self):
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
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        return fresh_tokens

    async def generate_exact(self, req_idx: int):
        max_output_tokens = int(config.max_new_tokens)
        step_idx = 0
        fresh_tokens = await self._cycle(req_idx, step_idx, prefill=True)
        eos_positions = (fresh_tokens == self._tokenizer.eos_token_id).nonzero()
        if eos_positions.numel() > 0:
            fresh_tokens = fresh_tokens[..., : int(eos_positions[0, -1].item()) + 1]
        fresh_tokens = fresh_tokens[..., :max_output_tokens]
        self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
        generated = int(fresh_tokens.numel())
        eos_flag = bool(eos_positions.numel() > 0)

        step_idx = 1
        while generated < max_output_tokens and not eos_flag:
            fresh_tokens = await self._cycle(req_idx, step_idx)
            eos_positions = (fresh_tokens == self._tokenizer.eos_token_id).nonzero()
            if eos_positions.numel() > 0:
                eos_column = int(eos_positions[0, -1].item())
                fresh_tokens = fresh_tokens[..., : eos_column + 1]
                eos_flag = True
            remaining = max_output_tokens - generated
            fresh_tokens = fresh_tokens[..., :remaining]
            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
            generated += int(fresh_tokens.numel())
            step_idx += 1


def format_prompt(tokenizer, record: dict) -> str:
    if record["dataset"] != "mt_bench":
        return record["prompt"]
    messages = [{"role": "user", "content": record["prompt"]}]
    template_args = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **template_args,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **template_args)


def load_shard() -> list[dict]:
    dataset_file = Path(os.environ["FASTSD_EVAL_DATASET_FILE"])
    with dataset_file.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    num_clients = int(os.environ["FASTSD_EVAL_NUM_CLIENTS"])
    return records[config.client_idx :: num_clients]


async def warmup_specedge(engine, tokenizer, records: list[dict]) -> int:
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
        prompt = format_prompt(tokenizer, records[0])
        for warmup_idx in range(count):
            warmup_client = IntegratedSpecExecClient(
                engine=engine,
                tokenizer=tokenizer,
                prompt=prompt,
                max_len=config.max_len,
            )
            await warmup_client.generate_exact(-1 - warmup_idx)
    finally:
        config.max_new_tokens = original_max_new_tokens
    return count


async def main():
    logger = log.get_logger()
    records = load_shard()
    random.seed(config.seed + config.client_idx)

    if config.device.type == "cpu":
        configure_torch_threads(int(os.environ.get("SPECEDGE_THREADS", "1")))
    logger.info("Loading draft model %s on %s", config.draft_model, config.device)
    draft_model = util.load_graph_model(
        name=config.draft_model,
        device=config.device,
        dtype=config.dtype,
    )
    if config.device.type == "cpu":
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
    tokenizer = util.load_tokenizer(config.draft_model)

    with grpc.insecure_channel(config.host) as channel:
        stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
        _ = stub.Sync(specedge_pb2.SyncRequest())

    warmup_count = await warmup_specedge(engine, tokenizer, records)

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
            client = IntegratedSpecExecClient(
                engine=engine,
                tokenizer=tokenizer,
                prompt=format_prompt(tokenizer, record),
                max_len=config.max_len,
            )
            client.measurement_start = request_start
            await client.generate_exact(int(record["global_index"]))
            transport_stats = client._validator.snapshot_stats()
            request_e2e_ms = (time.perf_counter() - request_start) * 1000.0
            generated_ids = client._prefix_tokens[0, client._num_original_tokens :]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            ttft_ms = (
                (client.first_token_time - request_start) * 1000.0
                if client.first_token_time is not None
                else request_e2e_ms
            )
            generated_tokens = int(generated_ids.numel())
            payload = {
                "sample_id": record["sample_id"],
                "global_index": int(record["global_index"]),
                "client_idx": int(config.client_idx),
                "dataset": record["dataset"],
                "scheduled_arrival_s": scheduled,
                "actual_arrival_s": actual_arrival,
                "completion_s": max(0.0, time.time() - start_epoch),
                "arrival_lag_ms": max(0.0, (actual_arrival - scheduled) * 1000.0),
                "request_e2e_ms": request_e2e_ms,
                "generated_tokens": generated_tokens,
                "ttft_ms": ttft_ms,
                "tpot_ms": (
                    (request_e2e_ms - ttft_ms) / (generated_tokens - 1)
                    if generated_tokens > 1
                    else 0.0
                ),
                "output_text": completion,
                "output_token_ids": [int(token_id) for token_id in generated_ids.tolist()],
                "reference": record.get("reference"),
                "workload_hash": os.environ.get("FASTSD_EVAL_WORKLOAD_HASH"),
                "method": os.environ.get("FASTSD_EVAL_METHOD", "specedge_cpu_adapted"),
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
