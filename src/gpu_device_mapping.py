"""Map target compute-device ordinals to physical NVML GPU indices."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence


def _device_type(device: Any) -> str:
    value = str(device or "")
    return value.split(":", 1)[0].lower()


def _logical_index(device: Any) -> int:
    value = str(device or "cuda:0")
    if ":" not in value:
        return 0
    return int(value.rsplit(":", 1)[1])


def _visible_tokens(env: Mapping[str, str]) -> list[str]:
    raw = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    return [token.strip() for token in raw.split(",") if token.strip()]


def _uuid_to_nvml_index(token: str) -> int:
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES contains a GPU UUID, but pynvml is unavailable"
        ) from exc
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByUUID(token.encode())
        return int(pynvml.nvmlDeviceGetIndex(handle))
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def _initialized_cuda_uuid(index: int) -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_initialized():
        return None
    value = getattr(torch.cuda.get_device_properties(index), "uuid", None)
    if value:
        uuid = str(value)
        return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    return None


def resolve_nvml_gpu_indices(
    compute_device: Any,
    *,
    logical_gpu_indices: Sequence[int] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Resolve visible CUDA devices to physical NVML indices.

    ``logical_gpu_indices`` are ordinals in the process-visible CUDA namespace,
    which is also the namespace used by ``target_tp_gpus``.  Numeric
    An initialized CUDA target resolves by runtime UUID. Before CUDA starts,
    explicit numeric visibility is interpreted in PCI bus order and UUID entries
    are resolved through NVML. CPU targets intentionally return no GPUs.
    """

    if _device_type(compute_device) != "cuda":
        return ()
    environ = os.environ if env is None else env
    logical = (
        [int(index) for index in logical_gpu_indices]
        if logical_gpu_indices is not None
        else [_logical_index(compute_device)]
    )
    if any(index < 0 for index in logical):
        raise ValueError(f"logical CUDA GPU indices must be non-negative: {logical}")
    visible = _visible_tokens(environ)
    if env is None:
        uuids = [_initialized_cuda_uuid(index) for index in logical]
        if all(uuids):
            return tuple(dict.fromkeys(_uuid_to_nvml_index(uuid) for uuid in uuids))
    if not visible:
        return tuple(dict.fromkeys(logical))
    physical: list[int] = []
    for index in logical:
        if index >= len(visible):
            raise RuntimeError(
                f"logical CUDA GPU {index} is outside CUDA_VISIBLE_DEVICES={visible!r}"
            )
        token = visible[index]
        try:
            physical_index = int(token)
        except ValueError:
            physical_index = _uuid_to_nvml_index(token)
        if physical_index < 0:
            raise RuntimeError(f"invalid physical GPU index in CUDA_VISIBLE_DEVICES: {token!r}")
        physical.append(physical_index)
    return tuple(dict.fromkeys(physical))


def resolve_target_nvml_gpu_indices(
    *,
    target_device: Any,
    target_tp_size: int = 1,
    target_tp_gpus: str | Sequence[int] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Resolve one target device or all visible devices used by target TP."""

    if _device_type(target_device) != "cuda":
        return ()
    if target_tp_size > 1:
        if target_tp_gpus is None:
            raise RuntimeError("target TP requires explicit target_tp_gpus for power sampling")
        if isinstance(target_tp_gpus, str):
            raw = target_tp_gpus.strip()
            if raw.startswith("[") and raw.endswith("]"):
                raw = raw[1:-1]
            logical = [int(part.strip()) for part in raw.split(",") if part.strip()]
        else:
            logical = [int(index) for index in target_tp_gpus]
        if len(logical) != target_tp_size:
            raise RuntimeError(
                f"target TP GPU count {len(logical)} does not match target_tp_size={target_tp_size}"
            )
        return resolve_nvml_gpu_indices(target_device, logical_gpu_indices=logical, env=env)
    if target_tp_gpus is not None and str(target_tp_gpus).strip():
        raw = str(target_tp_gpus).strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        logical = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if len(logical) != 1:
            raise RuntimeError("single-GPU target configuration must name exactly one GPU")
        return resolve_nvml_gpu_indices(target_device, logical_gpu_indices=logical, env=env)
    return resolve_nvml_gpu_indices(target_device, env=env)
