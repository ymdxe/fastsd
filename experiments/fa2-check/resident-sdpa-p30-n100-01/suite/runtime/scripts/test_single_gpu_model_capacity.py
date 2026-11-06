import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:  # pragma: no cover - remote env decides availability
    AutoGPTQForCausalLM = None


def parse_args():
    parser = argparse.ArgumentParser(description="Probe how many model replicas fit on a single GPU.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-models", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    return parser.parse_args()


def snapshot_memory(gpu_index: int):
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
    return {
        "free_gib": round(free_bytes / 1024**3, 2),
        "total_gib": round(total_bytes / 1024**3, 2),
        "allocated_gib": round(torch.cuda.memory_allocated(gpu_index) / 1024**3, 2),
        "reserved_gib": round(torch.cuda.memory_reserved(gpu_index) / 1024**3, 2),
    }


def load_model(model_path: Path, gpu_index: int, dtype_name: str):
    dtype = getattr(torch, dtype_name)
    if (model_path / "quantize_config.json").exists():
        if AutoGPTQForCausalLM is None:
            raise RuntimeError("auto_gptq is required for quantized models")
        return AutoGPTQForCausalLM.from_quantized(
            str(model_path),
            device=f"cuda:{gpu_index}",
            use_safetensors=True,
            trust_remote_code=True,
        )

    return AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": f"cuda:{gpu_index}"},
    ).eval()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    gc.collect()

    loaded = []
    records = []
    failure = None

    for idx in range(1, args.max_models + 1):
        try:
            model = load_model(model_path, args.gpu, args.dtype)
            loaded.append(model)
            torch.cuda.synchronize(args.gpu)
            memory = snapshot_memory(args.gpu)
            record = {"count": idx, **memory}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        except Exception as exc:  # pragma: no cover - probing script
            failure = {"count": idx, "error_type": type(exc).__name__, "error": str(exc)}
            print(json.dumps(failure, ensure_ascii=False), flush=True)
            break

    summary = {
        "model_path": str(model_path),
        "gpu": args.gpu,
        "loaded_models": len(loaded),
        "failure": failure,
        "final_memory": snapshot_memory(args.gpu),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
