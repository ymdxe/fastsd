import argparse
import json
import multiprocessing as mp
import os
import queue
import time


def worker_main(rank, model_path, gpu, n_gpu_layers, n_threads, n_ctx, prompt, max_new_tokens, measure_runs, result_queue):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    from llama_cpp import Llama

    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            main_gpu=gpu,
            n_threads=n_threads,
            n_ctx=n_ctx,
            logits_all=True,
            verbose=False,
        )
    except Exception as exc:
        result_queue.put({"rank": rank, "status": "load_failed", "error": repr(exc)})
        return

    try:
        eos_token = model.token_eos()
        per_run = []
        total_elapsed = 0.0
        total_new_tokens = 0

        for _ in range(measure_runs):
            model.reset()
            input_tokens = model.tokenize(prompt.encode("utf-8"), add_bos=True)
            model.eval(input_tokens)

            generated = 0
            started = time.perf_counter()
            while generated < max_new_tokens:
                next_token = model.sample(top_k=1, top_p=0.95, temp=0.0)
                model.eval([next_token])
                generated += 1
                if next_token == eos_token:
                    break
            elapsed = time.perf_counter() - started

            total_elapsed += elapsed
            total_new_tokens += generated
            per_run.append(
                {
                    "elapsed_s": elapsed,
                    "new_tokens": generated,
                    "tokens_per_s": (generated / elapsed) if elapsed > 0 else 0.0,
                }
            )

        result_queue.put(
            {
                "rank": rank,
                "status": "ok",
                "total_elapsed_s": total_elapsed,
                "total_new_tokens": total_new_tokens,
                "tokens_per_s": (total_new_tokens / total_elapsed) if total_elapsed > 0 else 0.0,
                "per_run": per_run,
            }
        )
    except Exception as exc:
        result_queue.put({"rank": rank, "status": "runtime_failed", "error": repr(exc)})


def collect_results(processes, result_queue):
    results = []
    while len(results) < len(processes):
        if not any(p.is_alive() for p in processes):
            break
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            pass

    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    for process in processes:
        process.join(timeout=1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--measure-runs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument(
        "--prompt",
        default="Write a Python function that returns the Fibonacci sequence up to n elements.",
    )
    args = parser.parse_args()

    if "spawn" in mp.get_all_start_methods():
        mp.set_start_method("spawn", force=True)

    result_queue = mp.Queue()
    processes = []

    launch_started = time.perf_counter()
    for rank in range(args.concurrency):
        process = mp.Process(
            target=worker_main,
            args=(
                rank,
                args.model_path,
                args.gpu,
                args.n_gpu_layers,
                args.n_threads,
                args.n_ctx,
                args.prompt,
                args.max_new_tokens,
                args.measure_runs,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    results = collect_results(processes, result_queue)
    launch_elapsed = time.perf_counter() - launch_started

    ok_results = [r for r in results if r.get("status") == "ok"]
    response = {
        "model_path": args.model_path,
        "concurrency": args.concurrency,
        "status": "ok" if len(ok_results) == args.concurrency else "partial_failure",
        "workers_reported": len(results),
        "loaded": len(ok_results),
        "launch_elapsed_s": launch_elapsed,
        "avg_tokens_per_s": (
            sum(r["tokens_per_s"] for r in ok_results) / len(ok_results) if ok_results else 0.0
        ),
        "min_tokens_per_s": min((r["tokens_per_s"] for r in ok_results), default=0.0),
        "max_tokens_per_s": max((r["tokens_per_s"] for r in ok_results), default=0.0),
        "worker_results": sorted(ok_results, key=lambda r: r["rank"]),
        "errors": [r for r in results if r.get("status") != "ok"],
    }
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
