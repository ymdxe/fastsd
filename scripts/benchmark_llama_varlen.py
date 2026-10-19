"""Single-GPU Llama padded/varlen microbenchmark (local checkpoints only).

Includes input packing, model forward, probability processing and KV writeback.
Excludes model loading, cache setup/rollback, networking and CPU KV offload.
No Qwen model is loaded and no weights are downloaded.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from src.kvcache_batching import KVCacheModel_batching
from src.kvcache_varlen import supports_varlen_model, varlen_generate


def manager_for(model):
    manager = KVCacheModel_batching(model, temperature=1.0)
    manager.vocab_size = model.config.vocab_size
    return manager


def forward(manager, model, residuals, pids, mode, fresh):
    if mode == "varlen":
        return varlen_generate(manager, model, residuals, pids, 0, sample_next=False)
    inputs = []
    for pid, residual in zip(pids, residuals):
        if fresh:
            inputs.append(residual[0])
        else:
            cached = manager._past_key_values[pid].get_seq_length()
            prefix = residual.new_zeros(cached)
            inputs.append(torch.cat((prefix, residual[0])))
    lengths = [x.numel() for x in inputs]
    dense = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return manager.generate(dense, 1, pids, 0, fresh, input_lens=lengths, sample_next=False)


@torch.inference_mode()
def run(args):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "llama":
        raise ValueError("This benchmark only accepts Llama checkpoints")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    ).to("cuda:0").eval()
    if not supports_varlen_model(model):
        raise ValueError("Loaded Llama decoder is unsupported by varlen")
    report = {
        "model": args.model, "model_type": config.model_type,
        "host": platform.node(), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "dtype": args.dtype, "attention": "per-sequence SDPA; packed linear layers",
        "warmup": args.warmup, "repeats": args.repeats,
        "scope": "single-GPU forward microbenchmark; no network/offload/scheduler timing",
        "cases": [],
    }
    cases = [
        ("prefill_mixed", [0, 0, 0], [16, 64, 256]),
        ("prefill_equal", [0, 0, 0], [128, 128, 128]),
        ("prefill_continuation", [64, 256, 512], [16, 64, 128]),
        ("verify_mixed", [128, 512, 1024], [4, 6, 8]),
        ("verify_equal", [512, 512, 512], [6, 6, 6]),
        ("verify_single_query", [128, 512, 1024], [1, 1, 1]),
    ]
    for name, cached_lens, new_lens in cases:
        pids = list(range(len(new_lens)))
        residuals = [torch.randint(3, config.vocab_size, (1, n), device="cuda") for n in new_lens]
        fresh = not any(cached_lens)
        managers = {mode: manager_for(model) for mode in ("padded", "varlen")}
        if not fresh:
            prefixes = [torch.randint(3, config.vocab_size, (1, n), device="cuda") for n in cached_lens]
            for manager in managers.values():
                # Identical unpadded setup isolates the measured continuation.
                varlen_generate(manager, model, prefixes, pids, 0, sample_next=False)
        timings = {mode: [] for mode in managers}
        peaks = {mode: [] for mode in managers}
        rows = {}
        for iteration in range(args.warmup + args.repeats):
            # Alternate execution order to reduce systematic warm-cache bias.
            order = ("padded", "varlen") if iteration % 2 == 0 else ("varlen", "padded")
            for mode in order:
                manager = managers[mode]
                for pid, cached in zip(pids, cached_lens):
                    if fresh:
                        manager.reset(pid)
                    else:
                        manager.rollback(pid, cached)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                forward(manager, model, residuals, pids, mode, fresh)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= args.warmup:
                    timings[mode].append(elapsed)
                    peaks[mode].append((torch.cuda.max_memory_allocated() - baseline) / (1024 ** 2))
                if iteration == args.warmup + args.repeats - 1:
                    rows[mode] = torch.cat([
                        manager._prob_history[pid][0, -n:, :].float().cpu()
                        for pid, n in zip(pids, new_lens)
                    ])
        delta = (rows["padded"] - rows["varlen"]).abs()
        # Compare each request with an independent native Llama forward.
        # In BF16, a masked padded SDPA call may choose a different kernel
        # from an unmasked single-sequence call, so record both comparisons.
        reference = []
        for pid, cached, residual in zip(pids, cached_lens, residuals):
            cache = None
            if cached:
                history = managers["varlen"]._past_key_values[pid].to_legacy_cache()
                cache = DynamicCache.from_legacy_cache([
                    (k[:, :, :cached, :].clone(), v[:, :, :cached, :].clone())
                    for k, v in history
                ])
            ref = model(residual, past_key_values=cache, use_cache=True)
            reference.append(ref.logits[0].float().softmax(-1).cpu())
        reference = torch.cat(reference)
        result = {
            "name": name, "cached_lengths": cached_lens, "new_lengths": new_lens,
            "real_new_tokens": sum(new_lens),
            "padded_new_positions": len(new_lens) * max(new_lens),
            "median_ms": {mode: statistics.median(values) for mode, values in timings.items()},
            "samples_ms": timings,
            "median_incremental_peak_mib": {mode: statistics.median(values) for mode, values in peaks.items()},
            "prob_max_abs_diff": delta.max().item(),
            "prob_mean_row_l1": delta.sum(-1).mean().item(),
            "argmax_agreement": (rows["padded"].argmax(-1) == rows["varlen"].argmax(-1)).float().mean().item(),
            "vs_native_single_sequence": {
                mode: {
                    "prob_max_abs_diff": (values - reference).abs().max().item(),
                    "prob_mean_row_l1": (values - reference).abs().sum(-1).mean().item(),
                    "argmax_agreement": (values.argmax(-1) == reference.argmax(-1)).float().mean().item(),
                }
                for mode, values in rows.items()
            },
        }
        result["speedup"] = result["median_ms"]["padded"] / result["median_ms"]["varlen"]
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
        del managers, rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    run(args)
