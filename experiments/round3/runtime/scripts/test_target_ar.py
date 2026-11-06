#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(sys.path[0], "../"))

from src.util import model_zoo, norm_logits, sample  # noqa: E402


def resolve_model_path(name_or_path: str, use_model_zoo: bool) -> str:
    if not use_model_zoo:
        return name_or_path

    class _Args:
        def __init__(self, model_name: str):
            self.draft_model = model_name
            self.target_model = model_name
            self.vocab_size = None

    args = _Args(name_or_path)
    model_zoo(args)
    return args.target_model


def load_target_model(model_path: str, device: str):
    # Prefer GPTQ path first to align with your cloud target setup.
    try:
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device=device,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
        )
        return model, "gptq"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        return model, "hf"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Target-only autoregressive debug script")
    parser.add_argument("--target_model", type=str, required=True, help="model name/path")
    parser.add_argument("--prompt", type=str, required=True, help="input prompt")
    parser.add_argument("--use_model_zoo", action="store_true", help="resolve model alias via src.util.model_zoo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--print_steps", type=int, default=32, help="print first N generated steps")
    args = parser.parse_args()

    model_path = resolve_model_path(args.target_model, args.use_model_zoo)
    print(f"[INFO] target_model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model, backend = load_target_model(model_path, args.device)
    model_device = next(model.parameters()).device
    print(f"[INFO] backend={backend}, model_device={model_device}")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(model_device)
    x = input_ids
    past_key_values: Optional[object] = None

    print(f"[PROMPT] {repr(args.prompt)}")
    print(f"[PROMPT_TOKEN_LEN] {input_ids.shape[1]}")

    for step in range(args.max_new_tokens):
        if past_key_values is None:
            outputs = model(x, use_cache=True)
        else:
            outputs = model(x[:, -1:], past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        probs = norm_logits(
            outputs.logits[:, -1, :],
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        next_token = sample(probs)
        x = torch.cat((x, next_token), dim=1)

        if step < args.print_steps:
            tid = int(next_token.item())
            ttxt = tokenizer.decode([tid], skip_special_tokens=False)
            print(f"[STEP {step:03d}] id={tid} text={repr(ttxt)}")

    generated = tokenizer.decode(x[0, input_ids.shape[1]:], skip_special_tokens=True)
    print("\n[GENERATED_TEXT]")
    print(generated)


if __name__ == "__main__":
    main()

