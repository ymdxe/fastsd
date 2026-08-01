import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = (
    "You are benchmarking autoregressive decoding throughput. "
    "Write a short explanation of speculative decoding and cloud-edge inference, "
    "then provide three concise optimization ideas."
)


def build_prompt(prompt_file: str | None, prompt_repeat: int) -> str:
    if prompt_repeat < 1:
        raise ValueError("prompt_repeat must be >= 1")

    if prompt_file is None:
        base_prompt = PROMPT
    else:
        base_prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
        if not base_prompt:
            raise ValueError(f"prompt file is empty: {prompt_file}")

    return "\n\n".join(base_prompt for _ in range(prompt_repeat))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark per-model throughput under single-GPU model concurrency.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=2)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    return parser.parse_args()


def load_model(model_path: Path, gpu_index: int, dtype_name: str, trust_remote_code: bool):
    if (model_path / "quantize_config.json").exists():
        model = AutoGPTQForCausalLM.from_quantized(
            str(model_path),
            device=f"cuda:{gpu_index}",
            use_safetensors=True,
            trust_remote_code=trust_remote_code,
        )
        return model.eval()

    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map={"": f"cuda:{gpu_index}"},
    )
    return model.eval()


def worker_main(rank, args, ready_queue, result_queue, start_barrier, measure_barrier):
    try:
        torch.cuda.set_device(args.gpu)
        model_path = Path(args.model_path)
        model = load_model(model_path, args.gpu, args.dtype, args.trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompt_text = build_prompt(args.prompt_file, args.prompt_repeat)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(f"cuda:{args.gpu}")
        attention_mask = inputs["attention_mask"].to(f"cuda:{args.gpu}")
        prompt_tokens = int(input_ids.shape[1])

        ready_queue.put({"rank": rank, "status": "loaded"})
        start_barrier.wait()

        def run_fixed_decode():
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

                start = time.perf_counter()
                for _ in range(args.max_new_tokens):
                    outputs = model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize(args.gpu)
                elapsed = time.perf_counter() - start
                return elapsed

        for _ in range(args.warmup_runs):
            _ = run_fixed_decode()

        measure_barrier.wait()

        total_new_tokens = 0
        total_elapsed = 0.0
        per_run = []
        for _ in range(args.measure_runs):
            elapsed = run_fixed_decode()
            new_tokens = args.max_new_tokens
            total_new_tokens += new_tokens
            total_elapsed += elapsed
            per_run.append(
                {
                    "elapsed_s": elapsed,
                    "new_tokens": new_tokens,
                    "tokens_per_s": (new_tokens / elapsed) if elapsed > 0 else 0.0,
                }
            )

        result_queue.put(
            {
                "rank": rank,
                "status": "ok",
                "total_new_tokens": total_new_tokens,
                "total_elapsed_s": total_elapsed,
                "tokens_per_s": (total_new_tokens / total_elapsed) if total_elapsed > 0 else 0.0,
                "per_run": per_run,
                "peak_memory_gib": torch.cuda.max_memory_allocated(args.gpu) / 1024**3,
                "prompt_tokens": prompt_tokens,
            }
        )
    except Exception as exc:  # pragma: no cover - benchmark worker
        result_queue.put(
            {
                "rank": rank,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def collect_loaded(ready_queue, result_queue, concurrency: int, timeout_s: float = 1800.0):
    loaded = 0
    early_results = []
    deadline = time.time() + timeout_s
    while loaded < concurrency and time.time() < deadline:
        try:
            msg = ready_queue.get(timeout=1.0)
        except queue.Empty:
            msg = None
        if msg and msg.get("status") == "loaded":
            loaded += 1
        while True:
            try:
                result_msg = result_queue.get_nowait()
            except queue.Empty:
                break
            early_results.append(result_msg)
    return loaded, early_results


def collect_results(result_queue, concurrency: int, timeout_s: float = 7200.0):
    results = []
    deadline = time.time() + timeout_s
    while len(results) < concurrency and time.time() < deadline:
        try:
            results.append(result_queue.get(timeout=5.0))
        except queue.Empty:
            continue
    return results


def main():
    args = parse_args()
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    else:
        mp.set_start_method("spawn", force=True)

    ready_queue = mp.Queue()
    result_queue = mp.Queue()
    start_barrier = mp.Barrier(args.concurrency + 1)
    measure_barrier = mp.Barrier(args.concurrency + 1)

    processes = []
    launch_time = time.perf_counter()
    for rank in range(args.concurrency):
        proc = mp.Process(
            target=worker_main,
            args=(rank, args, ready_queue, result_queue, start_barrier, measure_barrier),
        )
        proc.start()
        processes.append(proc)

    loaded, early_results = collect_loaded(ready_queue, result_queue, args.concurrency)
    if loaded != args.concurrency:
        for proc in processes:
            proc.terminate()
        summary = {
            "model_path": args.model_path,
            "concurrency": args.concurrency,
            "status": "load_failed",
            "loaded": loaded,
            "launch_elapsed_s": time.perf_counter() - launch_time,
            "errors": early_results,
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1

    start_barrier.wait()
    measure_barrier.wait()
    results = early_results + collect_results(result_queue, args.concurrency - len(early_results))

    for proc in processes:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

    errors = [item for item in results if item.get("status") != "ok"]
    oks = [item for item in results if item.get("status") == "ok"]

    summary = {
        "model_path": args.model_path,
        "concurrency": args.concurrency,
        "status": "ok" if len(oks) == args.concurrency else "error",
        "loaded": loaded,
        "workers_reported": len(results),
        "launch_elapsed_s": time.perf_counter() - launch_time,
        "prompt_file": args.prompt_file,
        "prompt_repeat": args.prompt_repeat,
        "prompt_tokens": oks[0]["prompt_tokens"] if oks else 0,
        "avg_tokens_per_s": (sum(item["tokens_per_s"] for item in oks) / len(oks)) if oks else 0.0,
        "min_tokens_per_s": min((item["tokens_per_s"] for item in oks), default=0.0),
        "max_tokens_per_s": max((item["tokens_per_s"] for item in oks), default=0.0),
        "avg_peak_memory_gib": (sum(item["peak_memory_gib"] for item in oks) / len(oks)) if oks else 0.0,
        "worker_results": sorted(results, key=lambda item: item["rank"]),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
