"""Experiment-only Qwen3-32B Target on physical GPU 3 with CPU offload.

Run this file with the usual cloud/cloud_service.py arguments.  The patch is
installed at module import, so multiprocessing's spawn worker installs it too.
The underlying service, scheduler, verification and KV-cache code is unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if not SOURCE.is_dir():
    SOURCE = ROOT.parents[1]
sys.path.insert(0, str(SOURCE))

# Set the visibility before importing anything that can initialize CUDA.  Never
# silently widen this experiment to GPUs occupied by unrelated jobs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
if os.environ["CUDA_VISIBLE_DEVICES"] != "3":
    raise RuntimeError("This experiment is restricted to physical GPU 3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM

from src.engine import Decoding


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _load_offloaded_target(self, model_path: str, device: str):
    if str(device) not in {"cuda", "cuda:0"}:
        raise ValueError("Use logical --target_device cuda:0 (physical GPU 3)")
    if str(getattr(self.args, "target_dtype", "")) not in {"float16", "fp16"}:
        raise ValueError("This experiment requires --target_dtype float16")
    if getattr(self.args, "kv_batch_mode", None) != "padded":
        raise ValueError("This experiment uses --kv_batch_mode padded for both methods")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device: physical GPU 3")

    output = Path(self.args.exp_name)
    output.mkdir(parents=True, exist_ok=True)
    ready_path = output / "target_model_ready.json"
    # A reused run directory must not advertise readiness from an earlier PID.
    _write_json(ready_path, {"ready": False, "pid": os.getpid(), "state": "loading"})
    started = time.monotonic()
    start_epoch = time.time()
    memory_budget = {0: "34GiB", "cpu": "96GiB"}
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    configuration = {
        "pid": os.getpid(),
        "model_path": str(Path(model_path).absolute()),
        "physical_gpu": 3,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "execution_device": "cuda:0",
        "dtype": "float16",
        "kv_batch_mode": "padded",
        "attention_implementation": "sdpa",
        "max_memory": {str(k): v for k, v in memory_budget.items()},
        "load_started_epoch": start_epoch,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "source_directory": str(SOURCE),
    }
    _write_json(output / "target_load_configuration.json", configuration)
    print("TARGET_MODEL_LOADING " + json.dumps(configuration), flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=memory_budget,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            offload_state_dict=True,
        ).eval()
        mapping = {str(k): str(v) for k, v in model.hf_device_map.items()}
        if "disk" in mapping.values():
            raise RuntimeError("Unexpected disk offload; only GPU 3 and CPU are permitted")
        if not any(value in {"0", "cuda:0"} for value in mapping.values()):
            raise RuntimeError("The Target device map did not place any weights on GPU 3")
        if str(model.device) != "cuda:0":
            raise RuntimeError(f"Target input device must be cuda:0, got {model.device}")
        torch.cuda.synchronize(0)
        configuration.update(
            hf_device_map=mapping,
            load_duration_seconds=time.monotonic() - started,
            load_completed_epoch=time.time(),
            gpu_memory_allocated_bytes=torch.cuda.memory_allocated(0),
            gpu_memory_reserved_bytes=torch.cuda.memory_reserved(0),
            gpu_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(0),
        )
        _write_json(output / "target_device_map.json", configuration)
        _write_json(
            ready_path,
            {
                "ready": True,
                "state": "model_loaded",
                "pid": os.getpid(),
                "model_path": configuration["model_path"],
                "completed_epoch": configuration["load_completed_epoch"],
                "configuration_file": str(output / "target_device_map.json"),
                "note": "Model load completed; validate the service with a real prefill/verify smoke request.",
            },
        )
        print("TARGET_MODEL_READY " + json.dumps(configuration), flush=True)
        return model
    except BaseException as exc:
        _write_json(
            ready_path,
            {"ready": False, "state": "failed", "pid": os.getpid(), "error": repr(exc)},
        )
        raise


# This assignment must run for both __main__ and spawn's __mp_main__.  Installing
# it solely under the main guard would leave the child using the single-GPU
# repository loader and would exhaust GPU memory.
Decoding._load_target_model_for_service = _load_offloaded_target


def main() -> None:
    defaults = {
        "--target_model": str(ROOT / "model-view" / "Qwen3-32B"),
        "--target_device": "cuda:0",
        "--target_dtype": "float16",
        "--kv_batch_mode": "padded",
    }
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
            sys.argv.extend([flag, value])
    from cloud.cloud_service import main as cloud_main

    cloud_main()


if __name__ == "__main__":
    main()
