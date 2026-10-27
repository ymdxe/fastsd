"""Small runtime adapters shared by CPU and CUDA evaluation paths.

The project has several entry points that load the same draft model.  Keeping
device, dtype, synchronization, and thread policy here prevents a CPU path
from accidentally inheriting CUDA-only assumptions from the original
benchmark scripts.
"""

from __future__ import annotations

import os
from typing import Any

import torch


_DTYPE_NAMES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def is_cpu_device(device: Any) -> bool:
    """Return whether *device* names a CPU device."""

    return str(device).split(":", 1)[0].lower() == "cpu"


def resolve_dtype(dtype: str | torch.dtype | None, device: Any) -> torch.dtype:
    """Resolve a model dtype, using fp32 for CPU ``auto`` and bf16 for CUDA.

    ``auto`` is deliberately device-aware.  Explicit dtype requests are never
    silently changed; callers can therefore make a reproducibility decision
    visible in a manifest.
    """

    if isinstance(dtype, torch.dtype):
        return dtype
    name = "auto" if dtype is None else str(dtype).lower()
    if name == "auto":
        return torch.float32 if is_cpu_device(device) else torch.bfloat16
    try:
        return _DTYPE_NAMES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_DTYPE_NAMES))
        raise ValueError(f"Unsupported dtype {dtype!r}; use one of {supported}, auto") from exc


def configure_torch_threads(num_threads: int | None) -> int | None:
    """Configure the in-process PyTorch thread pool and return the value used.

    Launchers set OMP/MKL variables before Python starts, and this function
    repeats the values for child/runtime paths that configure threads after
    import.  ``torch.set_num_threads`` remains the authoritative in-process
    setting used by the latency and 4x8 tracks.
    """

    if num_threads is None:
        return None
    value = int(num_threads)
    if value <= 0:
        raise ValueError("thread count must be positive")
    thread_value = str(value)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = thread_value
    os.environ["TORCH_NUM_THREADS"] = thread_value
    torch.set_num_threads(value)
    try:
        torch.set_num_interop_threads(value)
    except RuntimeError:
        # PyTorch raises when another component initialized the inter-op pool.
        # The intra-op pool above is still deterministic for this process.
        pass
    return value


def synchronize(device: Any) -> None:
    """Synchronize CUDA work when applicable; CPU execution is a no-op."""

    if is_cpu_device(device):
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def thread_environment(num_threads: int) -> dict[str, str]:
    """Return conservative BLAS environment values for a worker launcher."""

    value = str(int(num_threads))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "TORCH_NUM_THREADS": value,
    }


def describe_runtime(device: Any, dtype: str | torch.dtype | None, threads: int | None) -> dict[str, Any]:
    """Create a serializable runtime description for manifests and JSONL."""

    resolved = resolve_dtype(dtype, device)
    return {
        "device": str(device),
        "dtype": str(resolved).removeprefix("torch."),
        "requested_dtype": "auto" if dtype is None else str(dtype),
        "threads": int(threads) if threads is not None else None,
        "cuda_synchronize": not is_cpu_device(device),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
