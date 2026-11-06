#!/usr/bin/env python3
"""Single CPU draft worker with HTTP interface.

This worker:
- Loads one copy of the draft model on CPU
- Listens on a unique port
- Generates draft tokens for incoming requests
- Returns draft sequences to the coordinator
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI(title="FastSD CPU Draft Worker")

# Global state
draft_model = None
tokenizer = None
worker_id = None
generation_count = 0


class DraftRequest(BaseModel):
    prompt: str | None = None
    input_ids: list[int] | None = None
    max_new_tokens: int = 10
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    request_id: str = ""


class DraftResponse(BaseModel):
    draft_tokens: list[int]
    worker_id: int
    generation_time_ms: float
    request_id: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "worker_id": worker_id,
        "generation_count": generation_count,
        "device": "cpu",
    }


@app.post("/generate", response_model=DraftResponse)
def generate(request: DraftRequest):
    global generation_count

    if draft_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Prepare input
    if request.input_ids is not None:
        input_ids = torch.tensor([request.input_ids], dtype=torch.long)
    elif request.prompt is not None:
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
    else:
        raise HTTPException(status_code=400, detail="Must provide prompt or input_ids")

    # Generate
    with torch.no_grad():
        output = draft_model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_k=request.top_k if request.top_k > 0 else None,
            top_p=request.top_p if request.top_p < 1.0 else None,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Extract only the new tokens
    draft_tokens = output[0, input_ids.shape[1]:].tolist()

    elapsed_ms = (time.time() - start) * 1000
    generation_count += 1

    return DraftResponse(
        draft_tokens=draft_tokens,
        worker_id=worker_id,
        generation_time_ms=elapsed_ms,
        request_id=request.request_id,
    )


def load_model_and_tokenizer(model_path: str, device: str = "cpu"):
    """Load draft model and tokenizer."""
    global draft_model, tokenizer

    print(f"[WORKER-{worker_id}] Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    draft_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    ).eval()

    print(f"[WORKER-{worker_id}] Model loaded on {device}")


def main():
    global worker_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft_model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dataset", default="humaneval")

    args = parser.parse_args()
    worker_id = args.worker_id

    # Verify CPU-only
    if args.device != "cpu":
        print(f"WARNING: Device set to {args.device}, forcing CPU")
        args.device = "cpu"

    if torch.cuda.is_available():
        print(f"WARNING: CUDA is available but worker is CPU-only")

    # Load model
    try:
        load_model_and_tokenizer(args.draft_model, args.device)
    except Exception as e:
        print(f"[WORKER-{worker_id}] FATAL: Failed to load model: {e}")
        return 1

    # Start server
    print(f"[WORKER-{worker_id}] Starting server on port {args.port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
