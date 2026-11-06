"""CPU worker wrapper for canonical inputs and the shared generation contract."""
import argparse
import hashlib
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
RUNTIME = ROOT / "runtime"
sys.path[:0] = [str(ROOT), str(RUNTIME)]
from common import trace_hash, valid_id, write_json
from experiment_hooks import finish_worker, rpc, wait_for_start


def validate_canonical_sample(sample):
    """Validate the frozen prompt and IDs without formatting or retokenizing."""
    prompt = sample.get("prompt")
    token_ids = sample.get("input_token_ids")
    if not isinstance(sample.get("sample_id"), str) or not sample["sample_id"]:
        raise ValueError("Canonical sample requires a nonempty sample_id")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Canonical sample requires an exact rendered prompt")
    if not isinstance(token_ids, list) or not token_ids or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise ValueError("Canonical sample requires nonempty integer input_token_ids")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    ids_hash = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    if prompt_hash != sample.get("prompt_sha256") or ids_hash != sample.get("input_ids_sha256"):
        raise ValueError("Canonical prompt/token hash mismatch: " + sample["sample_id"])
    return prompt, token_ids


def validate_output_record(record, sample, max_tokens, eos_token_ids):
    token_ids = record.get("output_token_ids")
    if not isinstance(token_ids, list) or not 0 < len(token_ids) <= max_tokens:
        raise ValueError("Missing or over-budget output token IDs")
    if record.get("generated_tokens") != len(token_ids):
        raise ValueError("Output token count mismatch")
    if any(token in eos_token_ids for token in token_ids[:-1]):
        raise ValueError("Output continues after the first common EOS")
    if len(token_ids) < max_tokens and token_ids[-1] not in eos_token_ids:
        raise ValueError("Output stopped before either EOS or its token budget")
    for key in ("prompt_sha256", "input_ids_sha256"):
        if record.get(key) != sample[key]:
            raise ValueError("Output canonical input evidence mismatch: " + key)


class NoopBarrier:
    def wait(self, *args, **kwargs):
        return 0


class OriginHolder:
    """Keep the calibrated T0 when upstream tries to reset a local origin."""
    def __init__(self):
        self.origin = None

    def set_origin(self, origin):
        origin = float(origin)
        if not math.isfinite(origin):
            raise ValueError("Common T0 is not finite")
        self.origin = origin

    @property
    def value(self):
        if self.origin is None:
            raise RuntimeError("Common T0 has not been established")
        return self.origin

    @value.setter
    def value(self, ignored):
        # Upstream proc_id == 0 writes time.monotonic() after its barrier.
        # The common cross-host origin remains the one from wait_for_start.
        pass


def build_runtime_flags(options, info, node_config):
    from suite import common_flags, method_flags
    return common_flags() + method_flags(options.method) + [
        "--draft_model", node_config["model"],
        "--draft_dtype", "bfloat16", "--draft_device", "cpu",
        "--edge_use_cpu", "--edge_threads", str(node_config["threads"]),
        "--num_drafts", "1", "--gamma", str(options.gamma),
        "--server_url", options.server_url, "--request_timeout", str(info["request_timeout"]),
        "--timing_max_clock_uncertainty_ms", str(info["clock_uncertainty_limit_ms"]),
        "--dataset", info["trace_rows"][0]["dataset"], "--dataset_file", str(Path(options.inputs).resolve()),
        "--max_tasks_per_draft", "0", "--warmup_requests", "1",
        "--arrival_distribution", "poisson", "--arrival_rate", str(info["rate_per_worker"]),
        "--arrival_seed", str(info["seed"]), "--workload_hash", info["trace_sha256"],
        "--exp_name", str(Path(options.exp_name).resolve()),
    ]


def observed_runner_class(base_class, torch_module, holder, metadata_path):
    class ObservedRunner(base_class):
        def load_tokenizer(self):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.canonical_tokenizer_model, local_files_only=True, trust_remote_code=True
            )
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Canonical tokenizer requires a PAD or EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def _sample_prompt(self, sample, proc_id, idx):
            prompt, _ = validate_canonical_sample(sample)
            return prompt, sample["sample_id"]

        def _encode_input_ids(self, tokenizer, input_text, sample=None):
            if sample is None:
                raise ValueError("This experiment requires the complete frozen sample")
            prompt, token_ids = validate_canonical_sample(sample)
            if input_text != prompt:
                raise ValueError("Canonical prompt was changed before inference")
            return torch_module.tensor([token_ids], dtype=torch_module.long)

        @staticmethod
        def _is_degenerate_repeat(*args, **kwargs):
            # All methods use only the common EOS set and the output token cap.
            return False

        def _load_draft_model(self, model_path, device):
            if str(device) != "cpu":
                raise RuntimeError("This experiment requires a CPU draft")
            start = time.monotonic()
            model = super()._load_draft_model(model_path, device)
            parameters = list(model.parameters())
            dtypes = sorted({str(p.dtype) for p in parameters})
            devices = sorted({str(p.device) for p in parameters})
            metadata = {
                "model_path": model_path, "load_s": time.monotonic() - start,
                "tokenizer_path": self.args.canonical_tokenizer_model,
                "tokenizer_json_sha256": hashlib.sha256(
                    (Path(self.args.canonical_tokenizer_model) / "tokenizer.json").read_bytes()
                ).hexdigest(),
                "input_policy": "frozen_target_chat_template_token_ids",
                "eos_token_ids": self.args.eos_token_ids,
                "parameter_dtypes": dtypes, "parameter_devices": devices,
                "parameter_count": sum(p.numel() for p in parameters),
                "torch_version": torch_module.__version__,
                "python_version": platform.python_version(),
                "num_threads": torch_module.get_num_threads(),
                "num_interop_threads": torch_module.get_num_interop_threads(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "cuda_initialized": torch_module.cuda.is_initialized(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "epoch_s": time.time(),
            }
            write_json(metadata_path, metadata)
            if dtypes != ["torch.bfloat16"] or devices != ["cpu"]:
                raise RuntimeError("Actual Draft dtype/device mismatch: " + str(metadata))
            if metadata["num_threads"] != int(self.args.edge_threads):
                raise RuntimeError("Actual Draft thread count differs from configuration")
            if metadata["num_interop_threads"] != 1:
                raise RuntimeError("Actual Draft interop thread count must be one")
            if metadata["cuda_initialized"]:
                raise RuntimeError("CPU Draft unexpectedly initialized CUDA")
            return model

        def _warmup_stateful_requests(self, client, draft_model, tokenizer, samples, proc_id):
            if not samples:
                raise RuntimeError("No requests assigned to this worker")
            original_priority = client.timing_priority
            try:
                client.timing_priority = False
                super()._warmup_stateful_requests(client, draft_model, tokenizer, samples, proc_id)
            finally:
                client.timing_priority = original_priority
            holder.set_origin(wait_for_start(self.args))

    return ObservedRunner


def run(options):
    from suite import verify_bundle
    bundle_id = verify_bundle()
    run_id = valid_id(os.environ["FASTSD_SUITE_RUN_ID"])
    identity = os.environ["FASTSD_SUITE_WORKER"]
    node = platform.node().split(".")[0]
    if identity != f"{node}-w{options.worker_id}":
        raise RuntimeError("Worker identity does not match this host/worker ID")
    info = rpc(options.server_url, "/experiment/info")
    if info["run_id"] != run_id or info["bundle_id"] != bundle_id:
        raise RuntimeError("Target run ID or bundle differs from this worker")
    if info["method"] != options.method:
        raise RuntimeError("Target method differs from this worker")
    if bundle_id != os.environ["FASTSD_SUITE_BUNDLE_ID"]:
        raise RuntimeError("Launcher bundle differs from this worker")
    if info["trace_sha256"] != os.environ["FASTSD_SUITE_TRACE_SHA256"]:
        raise RuntimeError("Launcher trace differs from Target")
    if trace_hash(info["trace_rows"]) != info["trace_sha256"]:
        raise RuntimeError("Target trace hash does not match the supplied rows")
    node_config = info["nodes"][node]
    if options.gamma != int(node_config["gamma"]) or int(node_config["threads"]) != 8:
        raise RuntimeError("Worker gamma/thread settings differ from requested configuration")
    if not 0 <= options.worker_id < int(node_config["workers"]):
        raise RuntimeError("Unexpected worker number")
    if int(node_config.get("interop_threads", 1)) != 1 or set(info["eos_token_ids"]) != {151645, 151643}:
        raise RuntimeError("Config1 requires one interop thread and the common Qwen3 EOS set")
    affinity = [int(x) for x in options.affinity.split(",")]
    if len(affinity) != 8 or len(set(affinity)) != 8:
        raise RuntimeError("Expected exactly eight distinct logical CPUs")
    if not set(affinity).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Assigned CPUs are outside this process's allowed affinity")
    os.sched_setaffinity(0, affinity)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU worker must be launched with CUDA_VISIBLE_DEVICES empty")
    output = Path(options.exp_name).resolve()
    if not output.is_dir():
        raise RuntimeError("Launcher must create the dedicated worker output directory")
    if (output / "worker-status.json").exists() or list(output.glob("edge_metrics_proc*.jsonl")):
        raise FileExistsError("Worker results already exist; use a new run ID")
    inputs = [json.loads(x) for x in Path(options.inputs).read_text(encoding="utf-8").splitlines() if x.strip()]
    expected = [r for r in info["trace_rows"] if r["node"] == node and int(r["worker"]) == options.worker_id]
    if not inputs or inputs != expected:
        raise RuntimeError("Worker inputs differ from the full frozen assignment")
    for sample in inputs:
        validate_canonical_sample(sample)
    status = {"worker": identity, "run_id": run_id, "state": "loading", "started_epoch_s": time.time()}
    write_json(output / "worker-status.json", status)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("worker signal " + str(signum))
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"
        import torch
        from src.runtime import configure_torch_threads
        configure_torch_threads(int(node_config["threads"]))
        from edge.edge import EdgeRunner, parse_edge_arguments
        flags = build_runtime_flags(options, info, node_config)
        saved_argv = sys.argv
        try:
            sys.argv = [str(ROOT / "worker.py")] + flags
            args = parse_edge_arguments()
        finally:
            sys.argv = saved_argv
        args.canonical_tokenizer_model = node_config["tokenizer_model"]
        args.eos_token_ids = [int(token) for token in info["eos_token_ids"]]
        args.warmup_max_new_tokens = 1
        args.canonical_inputs = True
        write_json(output / "runtime-arguments.json", {"argv": flags, "resolved_parameters": vars(args)})
        holder = OriginHolder()
        runner_type = observed_runner_class(EdgeRunner, torch, holder, output / "actual-model.json")
        runner = runner_type(args)
        runner.run_draft_process_http(runner.tokenizer, 0, NoopBarrier(), holder)
        records = [json.loads(x) for x in (output / "edge_metrics_proc0.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        expected_ids = [r["sample_id"] for r in expected]
        actual_ids = [r["task_id"] for r in records]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
            raise RuntimeError("Incomplete or duplicate worker outputs")
        sample_by_id = {sample["sample_id"]: sample for sample in expected}
        for record in records:
            validate_output_record(record, sample_by_id[record["task_id"]], args.max_tokens, args.eos_token_ids)
        finish_worker(args)
        status.update(state="complete", requests=len(records), local_t0_monotonic_s=holder.value)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["finished_epoch_s"] = time.time()
        write_json(output / "worker-status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--method", required=True, choices=("sd", "fastsd"))
    parser.add_argument("--gamma", required=True, type=int)
    parser.add_argument("--affinity", required=True)
    parser.add_argument("--worker-id", required=True, type=int)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
