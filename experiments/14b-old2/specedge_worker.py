"""Run one BF16 CPU-adapted official SpecEdge stream in the suite protocol.

Gamma is both the chain depth and number of draft candidates (node3/node1/node2: 8/4/4).
Normal/proactive beams and branch width are one, with included proactive depth 3.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "runtime")]
from common import trace_hash, valid_id, write_json
from experiment_hooks import rpc, wait_for_start, finish_worker


def configure_environment(args, info, samples):
    identity = os.environ["FASTSD_SUITE_WORKER"]
    expected = info["expected_workers"]
    if len(expected) != 32 or len(set(expected)) != 32:
        raise ValueError("SpecEdge requires exactly 32 unique clients")
    client_idx = expected.index(identity)
    node, worker = identity.rsplit("-w", 1)
    topology = {"node1": 8, "node2": 8, "node3": 16}
    if node not in topology or not 0 <= int(worker) < topology[node]:
        raise ValueError("Expected node1/node2/node3 workers in the 8/8/16 topology")
    node_config = info["nodes"][node]
    if int(node_config["workers"]) != topology[node] or int(node_config["threads"]) != 6:
        raise ValueError("The configured Draft topology differs from 8/8/16 with six threads")
    expected_gamma = {"node3":8,"node1":4,"node2":4}[node]
    if int(args.gamma) != expected_gamma or int(node_config["gamma"]) != expected_gamma:
        raise ValueError("SpecEdge tree depth must be node3/node1/node2 8/4/4")
    args.draft_model = node_config["model"]
    args.tokenizer_model = node_config["tokenizer_model"]
    args.edge_threads = 6
    args.max_tokens = int(info["max_tokens"])
    args.request_timeout = float(info.get("request_timeout", 3600))
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0 or args.max_tokens <= 0:
        raise ValueError("Invalid generation cap or request timeout")
    if not samples:
        raise ValueError("A SpecEdge worker must have at least one request")
    if int(info.get("tree_budget", 8)) != 8:
        raise ValueError("Target maximum single-chain budget must be 8")
    eos_ids = list(map(int, info["eos_token_ids"]))
    if set(eos_ids) != {151645, 151643}:
        raise ValueError("Config 1 requires the common Qwen3 EOS token IDs")
    env = {
        "FASTSD_EOS_TOKEN_IDS": json.dumps(eos_ids),
        "SPECEDGE_OPTIMIZATION": 2,
        "SPECEDGE_RESULT_PATH": str(Path(args.exp_name) / "specedge-raw"),
        "SPECEDGE_EXP_NAME": os.environ["FASTSD_SUITE_RUN_ID"],
        "SPECEDGE_PROCESS_NAME": identity, "SPECEDGE_SEED": int(info.get("seed", 1234)),
        "SPECEDGE_MAX_LEN": int(info.get("specedge_max_len", 2048)),
        "SPECEDGE_DRAFT_MODEL": args.draft_model,
        "SPECEDGE_DEVICE": "cpu", "SPECEDGE_DTYPE": "bf16",
        "SPECEDGE_THREADS": 6, "SPECEDGE_ENGINE": "specedge_cpu_adapted",
        "SPECEDGE_DATASET": "fastsd_external", "SPECEDGE_REASONING": "False",
        "SPECEDGE_MAX_N_BEAMS": 1, "SPECEDGE_MAX_BEAM_LEN": int(args.gamma),
        "SPECEDGE_MAX_BRANCH_WIDTH": 1, "SPECEDGE_MAX_BUDGET": int(args.gamma),
        "SPECEDGE_PROACTIVE_TYPE": "included",
        "SPECEDGE_PROACTIVE_MAX_N_BEAMS": 1,
        "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": 3,
        "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": 1,
        "SPECEDGE_PROACTIVE_MAX_BUDGET": int(args.gamma),
        "SPECEDGE_MAX_NEW_TOKENS": args.max_tokens,
        "SPECEDGE_MAX_REQUEST_NUM": -1, "SPECEDGE_REQ_OFFSET": 0,
        "SPECEDGE_SAMPLE_REQ_CNT": 1,
        "SPECEDGE_HOST": info.get("specedge_grpc_target", info.get("specedge_server_url", "127.0.0.1:29660")),
        "SPECEDGE_CLIENT_IDX": client_idx,
    }
    os.environ.update({key: str(value) for key, value in env.items()})
    return client_idx


def _validate_inputs(args):
    from suite import verify_bundle
    bundle_id = verify_bundle()
    identity = os.environ["FASTSD_SUITE_WORKER"]
    run_id = valid_id(os.environ["FASTSD_SUITE_RUN_ID"])
    node = platform.node().split(".")[0]
    if identity != f"{node}-w{args.worker_id}":
        raise RuntimeError("Worker identity does not match this host/worker ID")
    info = rpc(args.server_url, "/experiment/info")
    if info["run_id"] != run_id or info["bundle_id"] != bundle_id:
        raise RuntimeError("Target run ID or bundle differs from this worker")
    if info["method"] != "specedge" or args.method != "specedge":
        raise RuntimeError("Target method differs from this SpecEdge worker")
    if bundle_id != os.environ["FASTSD_SUITE_BUNDLE_ID"]:
        raise RuntimeError("Launcher bundle differs from this worker")
    if info["trace_sha256"] != os.environ["FASTSD_SUITE_TRACE_SHA256"]:
        raise RuntimeError("Launcher trace differs from Target")
    if trace_hash(info["trace_rows"]) != info["trace_sha256"]:
        raise RuntimeError("Target trace hash does not match the supplied rows")
    if info.get("dtype") != "bfloat16":
        raise RuntimeError("This experiment requires BF16")
    if getattr(args, "run_info", None):
        local_info = json.loads(Path(args.run_info).read_text(encoding="utf-8"))
        for field in ("run_id", "bundle_id", "trace_sha256", "method"):
            if local_info[field] != info[field]:
                raise RuntimeError("Local run-info mismatch: " + field)
    affinity = [int(value) for value in args.affinity.split(",")]
    if len(affinity) != 6 or len(set(affinity)) != 6:
        raise RuntimeError("Expected exactly six distinct logical CPUs")
    if not set(affinity).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Assigned CPUs are outside this process's allowed affinity")
    os.sched_setaffinity(0, affinity)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU worker must be launched with CUDA_VISIBLE_DEVICES empty")
    output = Path(args.exp_name).resolve()
    if not output.is_dir():
        raise RuntimeError("Launcher must create the dedicated worker output directory")
    if (output / "worker-status.json").exists() or list(output.glob("edge_metrics_proc*.jsonl")):
        raise FileExistsError("Worker results already exist; use a new run ID")
    samples = [
        json.loads(line) for line in Path(args.inputs).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [
        row for row in info["trace_rows"]
        if row["node"] == node and int(row["worker"]) == args.worker_id
    ]
    if not samples or samples != expected:
        raise RuntimeError("Worker inputs differ from the full frozen assignment")
    return info, samples, output, identity, run_id


async def _run(args, info, samples, output):
    os.environ["FASTSD_DIAGNOSTIC_DIR"]=str(output/"diagnostics")
    client_idx = configure_environment(args, info, samples)
    from specedge.bootstrap import prepare_imports
    provenance = prepare_imports()
    import grpc
    import torch
    import transformers
    import log
    import util
    from config import SpecEdgeClientConfig as config
    from client import IntegratedSpecExecClient
    from specedge_correctness import FIX_ID, SpecEdgeCorrectnessMixin
    from single_chain_contract import SINGLE_CHAIN_ID
    from cpu_adapter import CPUCompatibleSpecEdgeEngine
    from src.runtime import configure_torch_threads
    from specedge_grpc import specedge_pb2, specedge_pb2_grpc

    if info.get("specedge_correctness_fix") != FIX_ID:
        raise RuntimeError("SpecEdge correctness fix differs from the frozen configuration")
    if (super(IntegratedSpecExecClient, IntegratedSpecExecClient)._validate_tree
            is not SpecEdgeCorrectnessMixin._validate_tree
            or IntegratedSpecExecClient._reorder_by_sequence_proactive
            is not SpecEdgeCorrectnessMixin._reorder_by_sequence_proactive):
        raise RuntimeError("SpecEdge correctness mixin is not bound below the single-chain audit")
    torch.set_num_interop_threads(1)
    configure_torch_threads(6)
    result_path = output / "specedge-raw"
    result_path.mkdir(parents=True, exist_ok=True)
    log_config = log.get_default_log_config(result_path, os.environ["FASTSD_SUITE_WORKER"])
    log_config["handlers"]["file"]["level"] = "INFO"
    log_config["loggers"]["root"]["level"] = "INFO"
    log.configure_logging(log_config)
    util.set_seed(config.seed)
    started = time.monotonic()
    model = util.load_graph_model(args.draft_model, torch.device("cpu"), torch.bfloat16)
    from measurements import observe_cpu_model
    observe_cpu_model(model)
    parameters = list(model.parameters())
    metadata = {
        "model_path": args.draft_model, "load_s": time.monotonic() - started,
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "epoch_s": time.time(),
    }
    write_json(output / "actual-model.json", metadata)
    if metadata["parameter_dtypes"] != ["torch.bfloat16"] or metadata["parameter_devices"] != ["cpu"]:
        raise RuntimeError("Actual Draft dtype/device differs from CPU BF16")
    if (metadata["num_threads"] != 6 or metadata["num_interop_threads"] != 1
            or len(metadata["affinity"]) != 6):
        raise RuntimeError("Actual Draft threads/affinity differ from six CPUs")
    if metadata["cuda_initialized"]:
        raise RuntimeError("CPU Draft unexpectedly initialized CUDA")
    tokenizer = util.load_tokenizer(args.tokenizer_model)
    engine = CPUCompatibleSpecEdgeEngine(model, config.max_len, config.max_n_beams)
    if engine._dtype != torch.bfloat16:
        raise RuntimeError("CPU SpecEdge engine did not preserve BF16")

    class MeasuredClient(IntegratedSpecExecClient):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            self.cycles = 0
            self.visible_token_times = []

        async def _cycle(self, req_idx, step_idx, prefill=False):
            fresh = await super()._cycle(req_idx, step_idx, prefill=prefill)
            self.cycles += 1
            self.visible_token_times.extend([time.monotonic()] * int(fresh.numel()))
            return fresh

    def prompt_text(record):
        return record["prompt"]

    async def generate(record, req_idx, max_tokens):
        prompt = prompt_text(record)
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if input_ids != record["input_token_ids"]:
            raise ValueError("Canonical prompt token IDs differ from the frozen workload")
        if len(input_ids) + max_tokens + 2 * int(config.max_budget) >= int(config.max_len):
            raise ValueError("Prompt/output/tree exceeds fixed SpecEdge max_len")
        client = MeasuredClient(engine=engine, tokenizer=tokenizer, prompt=prompt, max_len=config.max_len)
        original = config.max_new_tokens
        config.max_new_tokens = int(max_tokens)
        try:
            await asyncio.wait_for(client.generate_exact(req_idx), timeout=args.request_timeout)
            end = time.monotonic()
            tokens = [
                int(token) for token in client._prefix_tokens[0, client._num_original_tokens:].tolist()
            ]
            stats = client._validator.snapshot_stats()
            stats["chain_audits"] = client.chain_audits
            if not tokens or not client.visible_token_times or len(tokens) > max_tokens:
                raise RuntimeError("SpecEdge produced an empty result or exceeded the generation cap")
            first = client.visible_token_times[0]
            visible = client.visible_token_times[:len(tokens)]
            if len(visible) != len(tokens):
                raise RuntimeError("SpecEdge token visibility records are incomplete")
            return first, end, tokens, client.cycles, stats, visible
        finally:
            config.max_new_tokens = original
            await client._validator._channel.close()
            engine.reset()

    # Native client synchronization precedes warmup; suite READY follows it.
    channel = grpc.aio.insecure_channel(config.host)
    try:
        stub = specedge_pb2_grpc.SpecEdgeServiceStub(channel)
        await stub.Sync(specedge_pb2.SyncRequest(), timeout=args.request_timeout)
    finally:
        await channel.close()
    warmup = await generate(samples[0], -1 - client_idx, 1)
    write_json(output / "specedge-runtime.json", {
        **provenance, "python": sys.executable, "python_version": sys.version,
        "correctness_fix": FIX_ID, "correctness_methods_bound": True,
        "single_chain": SINGLE_CHAIN_ID, "warmup_chain_audits": warmup[4]["chain_audits"],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "grpc": grpc.__version__, "method": "specedge", "engine_adapter": "specedge_cpu_adapted",
        "draft_model": args.draft_model, "tokenizer_model": args.tokenizer_model,
        "eos_token_ids": json.loads(os.environ["FASTSD_EOS_TOKEN_IDS"]),
        "canonical_prompt_tokens_verified": True, "draft_dtype": str(model.dtype),
        "actual_model": metadata, "threads": 6, "max_beam_len": int(args.gamma),
        "gamma_semantics": "Single-chain candidate count and depth",
        "tree_budget": int(args.gamma), "max_n_beams": 1, "max_branch_width": 1,
        "proactive": "included", "proactive_max_beam_len": 3,
        "proactive_max_n_beams": 1, "proactive_max_branch_width": 1,
        "proactive_max_budget": int(args.gamma),
        "max_len": config.max_len, "max_tokens": args.max_tokens,
        "grpc_target": config.host, "client_idx": client_idx,
        "warmup_requests": 1, "request_timeout": args.request_timeout,
        "log_level": "INFO; structured cycle results retained",
        "trace_sha256": info["trace_sha256"], "worker": os.environ["FASTSD_SUITE_WORKER"],
    })
    origin = wait_for_start(args)
    with (output / "edge_metrics_proc0.jsonl").open("x", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            scheduled = float(sample.get("scheduled_arrival_s", 0))
            remaining = origin + scheduled - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            started = time.monotonic()
            from conversations import resolve_sample
            from measurements import set_task
            set_task(sample["sample_id"])
            sample=resolve_sample(sample,args.server_url,output)
            request_id = int(sample["global_index"])
            first, end, tokens, rounds, stats, visible = await generate(sample, request_id, args.max_tokens)
            actual = started - origin
            request_ms = (end - started) * 1000
            ttft_ms = (first - started) * 1000
            row = dict(sample)
            row.update(
                task_id=sample["sample_id"], method="specedge",
                correctness_fix=FIX_ID,
                engine_adapter="specedge_cpu_adapted", client_idx=client_idx,
                gamma=int(args.gamma), gamma_semantics="single_chain_candidate_count_and_depth",
                single_chain=SINGLE_CHAIN_ID, chain_audits=stats["chain_audits"],
                actual_candidates_per_cycle=int(args.gamma), wire_inputs_per_cycle=int(args.gamma) + 1,
                tree_budget=int(args.gamma), max_n_beams=1, max_branch_width=1,
                proactive_type="included", proactive_max_beam_len=3,
                proactive_max_n_beams=1, proactive_max_branch_width=1,
                proactive_max_budget=int(args.gamma),
                tokenizer_model=args.tokenizer_model, eos_token_ids=json.loads(os.environ["FASTSD_EOS_TOKEN_IDS"]),
                verify_method="greedy", draft_dtype="bfloat16", target_dtype="bfloat16",
                generated_tokens=len(tokens), output_token_ids=tokens,
                output_text=tokenizer.decode(tokens, skip_special_tokens=True),
                scheduled_arrival_s=scheduled, actual_arrival_s=actual,
                arrival_lag_ms=max(0, actual - scheduled) * 1000,
                processing_start_s=actual, latency_basis="request_processing_start",
                task_e2e_ms=request_ms, request_e2e_ms=request_ms, ttft_ms=ttft_ms,
                tpot_ms=(visible[-1] - first) * 1000 / (len(tokens) - 1) if len(tokens) > 1 else None,
                completion_s=end - origin, request_start_monotonic_s=started,
                completion_monotonic_s=end, first_token_monotonic_s=first,
                token_visibility_s=[value - origin for value in visible],
                rounds=rounds, request_bytes=stats["request_bytes"],
                response_bytes=stats["response_bytes"], rpc_count=stats["rpc_count"],
                warmup_requests_per_client=1, warmup_included_in_metrics=False,
                workload_hash=info["trace_sha256"], trace_sha256=info["trace_sha256"],
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print("SpecEdge completed " + sample["sample_id"], flush=True)
    finish_worker(args)
    return {"requests": len(samples), "local_t0_monotonic_s": origin}


def run(args):
    info, samples, output, identity, run_id = _validate_inputs(args)
    status = {
        "worker": identity, "run_id": run_id, "state": "loading",
        "started_epoch_s": time.time(),
    }
    write_json(output / "worker-status.json", status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("worker signal " + str(signum))

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        completed = asyncio.run(_run(args, info, samples, output))
        status.update(state="complete", **completed)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished_epoch_s"] = time.time()
        write_json(output / "worker-status.json", status)
        from diagnostic_trace import trace
        trace.close()
        try:
            from specedge.bootstrap import stop_legacy_logging
            stop_legacy_logging()
        except Exception:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--method", choices=("specedge",), default="specedge")
    parser.add_argument("--gamma", required=True, type=int)
    parser.add_argument("--affinity", required=True)
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--run-info")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
