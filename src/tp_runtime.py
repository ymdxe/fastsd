"""Explicit target tensor-parallel runtime helpers.

The single-GPU path deliberately stays untouched.  TP=2 is opt-in and uses
PyTorch's distributed process group plus Transformers' native ``tp_plan``
when the installed model implementation supports it.  This module only
initializes/records the runtime; it never downloads a checkpoint or silently
falls back to a duplicated full model.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def configured_tp(args: Any) -> tuple[int, list[int]]:
    size = int(getattr(args, "target_tp_size", os.environ.get("TARGET_TP_SIZE", 1)))
    raw = str(getattr(args, "target_tp_gpus", os.environ.get("TARGET_TP_GPUS", "")))
    gpus = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if size == 1 and not gpus:
        gpus = [0]
    if size < 1 or len(gpus) != size:
        raise RuntimeError(
            f"explicit TP configuration invalid: TARGET_TP_SIZE={size}, "
            f"TARGET_TP_GPUS={raw!r}"
        )
    if size > 1 and not torch.cuda.is_available():
        raise RuntimeError("TARGET_TP_SIZE>1 requires CUDA/NCCL; CPU TP is unsupported")
    if size > 1 and any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in gpus):
        raise RuntimeError(
            f"TARGET_TP_GPUS={gpus} is outside visible CUDA devices "
            f"(count={torch.cuda.device_count()})"
        )
    return size, gpus


def init_process_group_if_needed(size: int, gpus: list[int]) -> int:
    """Initialize NCCL for an externally-launched rank and return local rank."""

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if size <= 1:
        return 0
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable; cannot run explicit TP")
    if not torch.distributed.is_initialized():
        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            raise RuntimeError(
                "TP=2 requires torchrun-style MASTER_ADDR/MASTER_PORT/RANK/LOCAL_RANK "
                "environment; refusing to run two full replicas"
            )
        torch.cuda.set_device(gpus[local_rank])
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=size
        )
    if torch.distributed.get_world_size() != size:
        raise RuntimeError(
            f"NCCL world size {torch.distributed.get_world_size()} != TARGET_TP_SIZE {size}"
        )
    return local_rank


def memory_snapshot(gpus: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not torch.cuda.is_available():
        return result
    for gpu in gpus:
        stats = torch.cuda.memory_stats(gpu)
        result[str(gpu)] = {
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(gpu)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(gpu)),
            "cuda_allocated_bytes_peak": int(stats.get("allocated_bytes.all.peak", 0)),
            "cuda_reserved_bytes_peak": int(stats.get("reserved_bytes.all.peak", 0)),
        }
    return result

