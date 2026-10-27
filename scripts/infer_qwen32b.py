#!/usr/bin/env python3
"""Stream a local Qwen3 checkpoint using BF16/FP16 across selected GPUs."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数值")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="本地模型目录，不自动下载")
    parser.add_argument("--gpus", required=True, help="物理 GPU 编号，例如 0,3")
    parser.add_argument("--prompt", default="请用中文简要介绍一下你自己。")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--gpu-memory-gib", type=positive_float, default=40.0,
                        help="每卡用于模型权重的显存上限，默认 40 GiB")
    parser.add_argument("--reserve-gib", type=positive_float, default=4.0,
                        help="每卡至少留出的运行时显存，默认 4 GiB")
    parser.add_argument("--max-input-tokens", type=positive_int, default=4096)
    parser.add_argument("--max-new-tokens", type=positive_int, default=512)
    parser.add_argument("--thinking", action="store_true", help="启用 Qwen3 思考模式")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="仅检查配置和显存分配，不加载权重")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",")]
    if not gpu_ids or any(not item.isdecimal() for item in gpu_ids):
        parser.error("--gpus 必须是逗号分隔的非负整数，例如 0,3")
    gpu_ids = [str(int(item)) for item in gpu_ids]
    if len(gpu_ids) != len(set(gpu_ids)):
        parser.error("--gpus 不能包含重复编号")
    args.gpus = ",".join(gpu_ids)
    args.model = args.model.expanduser().resolve()
    if not (args.model / "config.json").is_file():
        parser.error(f"找不到 {args.model / 'config.json'}；请先准备完整本地模型")
    if not args.prompt.strip():
        parser.error("--prompt 不能为空")
    return args


def plan_device_map(config, dtype, gpu_memory_gib, reserve_gib):
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请在能访问 NVIDIA GPU 的终端运行，并检查 nvidia-smi")
    max_memory = {"cpu": 0}
    for device_index in range(torch.cuda.device_count()):
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability(device_index)[0] < 8:
            raise RuntimeError(f"cuda:{device_index} 不支持本脚本的 BF16 路径，请使用 --dtype float16")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        budget = min(int(gpu_memory_gib * 1024**3), free_bytes - int(reserve_gib * 1024**3))
        if budget <= 0:
            raise RuntimeError(f"cuda:{device_index} 空闲显存不足，请选择其他 GPU")
        max_memory[device_index] = budget
        print(
            f"[GPU] cuda:{device_index} {torch.cuda.get_device_name(device_index)} "
            f"free={free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB "
            f"weight_budget={budget / 1024**3:.2f} GiB",
            file=sys.stderr,
        )

    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    empty_model.tie_weights()
    parameter_count = sum(parameter.numel() for parameter in empty_model.parameters())
    print(f"[MODEL] parameters={parameter_count:,}; "
          f"16-bit weights≈{parameter_count * 2 / 1024**3:.2f} GiB", file=sys.stderr)
    no_split = empty_model._no_split_modules
    balanced_memory = get_balanced_memory(
        empty_model, max_memory=max_memory, no_split_module_classes=no_split, dtype=dtype,
    )
    device_map = infer_auto_device_map(
        empty_model, max_memory=balanced_memory, no_split_module_classes=no_split, dtype=dtype,
    )
    if any(not isinstance(device, int) for device in device_map.values()):
        raise RuntimeError(
            "所选 GPU 的权重预算不足，无法让模型完全驻留 GPU。"
            "Qwen3-32B BF16 请使用两张有足够空闲显存的 48GB 卡；"
            "确认空闲显存后再调整 --gpu-memory-gib。"
        )
    print("[DEVICE_MAP] " + json.dumps(device_map), file=sys.stderr)
    return device_map


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed

    print(f"[CONFIG] model={args.model}; physical_gpus={args.gpus}; dtype={args.dtype}", file=sys.stderr)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3" or getattr(config, "quantization_config", None):
        raise ValueError("本脚本需要未量化的 Qwen3 权重，例如 Qwen/Qwen3-32B")
    dtype = getattr(torch, args.dtype)
    device_map = plan_device_map(config, dtype, args.gpu_memory_gib, args.reserve_gib)
    if args.check_only:
        print("[CHECK_OK] 配置和权重显存规划通过；尚未加载权重或分配 KV cache。", file=sys.stderr)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking,
        return_tensors="pt", return_dict=True,
    )
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens > args.max_input_tokens:
        raise ValueError(f"输入有 {input_tokens} tokens，超过 --max-input-tokens={args.max_input_tokens}")
    if input_tokens + args.max_new_tokens > config.max_position_embeddings:
        raise ValueError("输入长度加 --max-new-tokens 超过模型配置的上下文上限")

    set_seed(args.seed)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, use_safetensors=True,
        torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()

    print("加载后的设备映射：", model.hf_device_map)

    inputs = inputs.to(model.get_input_embeddings().weight.device)
    used_devices = sorted(set(device_map.values()))
    for device_index in used_devices:
        torch.cuda.synchronize(device_index)
    print(f"[READY] load_seconds={time.perf_counter() - load_start:.2f}; input_tokens={input_tokens}",
          file=sys.stderr, flush=True)

    generation_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=0.6 if args.thinking else 0.7,
            top_p=0.95 if args.thinking else 0.8, top_k=20, repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
        )
    for device_index in used_devices:
        torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - generation_start
    output_tokens = output.shape[1] - input_tokens
    print(f"[STATS] generated_tokens={output_tokens}; seconds={elapsed:.2f}; "
          f"tokens_per_second_including_prefill={output_tokens / elapsed:.2f}", file=sys.stderr)
    if output_tokens == args.max_new_tokens:
        print("[LIMIT] 已达到输出 token 上限，回答可能未结束；可增加 --max-new-tokens。", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
