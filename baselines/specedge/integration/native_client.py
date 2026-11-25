"""Observe native SpecEdge generation without replacing its prompt, search, or wire path."""

import asyncio
import json
import sys
import time
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = INTEGRATION_ROOT.parent / "official" / "src"
sys.path[:0] = [str(OFFICIAL_SRC), str(OFFICIAL_SRC / "script"), str(INTEGRATION_ROOT.parents[2])]

import client as official_client
import log
from specedge.client.specexec import SpecExecClient
from src.common_metrics import IO_STALL_NOT_APPLICABLE
from src.metrics import request_latency
from settings import OFFICIAL, OFFICIAL_REVISION


class ObservedClient(SpecExecClient):
    first_token_time = None
    drafted_tokens = 0
    accepted_tokens = 0
    verify_steps = 0

    async def _validate_tree(self, req_idx, prefill=False):
        status = self._tree.status[self._tree.prefix_len:self._tree.end]
        proposed = int((status >= self._tree.PROCESSED).sum().item())
        fresh, stats = await super()._validate_tree(req_idx, prefill=prefill)
        if self.first_token_time is None and fresh.numel():
            self.first_token_time = time.perf_counter()
        self.drafted_tokens += proposed
        self.accepted_tokens += max(0, fresh.numel() - 1)
        self.verify_steps += 1
        return fresh, stats


async def main():
    config = official_client.config
    result_dir = Path(config.result_path) / config.exp_name
    log.configure_logging(log.get_default_log_config(result_dir, config.process_name))
    output_path = result_dir / f"client_{config.client_idx}_requests.jsonl"
    with output_path.open("x", encoding="utf-8") as output:
        async def observe(engine, tokenizer, req_idx, prompt):
            start = time.perf_counter()
            epoch_start = time.time()
            generator = ObservedClient(engine=engine, tokenizer=tokenizer, prompt=prompt, max_len=config.max_len)
            try:
                await generator.generate(req_idx)
                finished = time.perf_counter()
            finally:
                await generator._validator._channel.close()
            ids = generator._prefix_tokens[0, generator._num_original_tokens:].tolist()
            elapsed = (finished - start) * 1000.0
            ttft, tpot, fallback = request_latency(
                elapsed, (generator.first_token_time - start) * 1000.0 if generator.first_token_time is not None else None, len(ids),
            )
            accepted, drafted, steps = generator.accepted_tokens, generator.drafted_tokens, generator.verify_steps
            record = dict(
                method=OFFICIAL, official_revision=OFFICIAL_REVISION, sample_id=f"{config.client_idx}:{req_idx}",
                global_index=req_idx, generated_tokens=len(ids), e2e_ms=elapsed, ttft_ms=ttft, tpot_ms=tpot,
                ttft_fallback_to_e2e=fallback, output_token_ids=ids, output_text=tokenizer.decode(ids, skip_special_tokens=True),
                processing_start_epoch_s=epoch_start, completion_epoch_s=epoch_start + elapsed / 1000,
                latency_basis="request_processing_start",
                draft_device=str(config.device), accepted_tokens=accepted, drafted_tokens=drafted, verify_steps=steps,
                accept_rate=accepted / drafted if drafted else 0.0, accept_length=accepted / steps if steps else 0.0,
                **IO_STALL_NOT_APPLICABLE,
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

        official_client.generate = observe
        await official_client.main()


if __name__ == "__main__":
    asyncio.run(main())
