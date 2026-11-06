"""Explicit CPU adapter for the pinned official SpecEdge client core.

The official ``GraphEngine`` is intentionally CUDA-Graph-only.  This module
implements the same engine protocol (forward, prefill, gather, reset) with the
official model and ``KVCache`` objects, while leaving the official Tree,
SpecExec, and proactive-draft algorithms untouched.  It is therefore reported
as ``specedge_cpu_adapted`` rather than as an untouched official run.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch

import log
import util
from model.cache import KVCache
from specedge.client.specexec import SpecExecClient


class CPUCompatibleSpecEdgeEngine:
    """Non-CUDA engine with the official GraphEngine call contract."""

    supports_cuda_graphs = False
    adapter_name = "specedge_cpu_adapted"

    def __init__(self, model: Any, max_len: int, max_n_beams: int) -> None:
        if str(model.device).split(":", 1)[0] != "cpu":
            raise ValueError("CPUCompatibleSpecEdgeEngine requires a CPU model")
        self._logger = log.get_logger()
        self.max_len = int(max_len)
        self._max_n_beams = int(max_n_beams)
        self._model = model
        self._device = model.device
        self._dtype = model.dtype
        self._config = model.config
        self._past_key_values = KVCache(
            config=self._config,
            batch_size=1,
            max_n_beams=self._max_n_beams,
            max_len=self.max_len,
            device=self._device,
            dtype=self._dtype,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ``KVCache.update`` uses ``seq_indices`` to select the source
        # positions from the model's per-call K/V tensors.  The official
        # CUDA GraphEngine sets this while capturing each beam-width graph;
        # the CPU path has no capture step, so it must set the same mapping
        # for every forward.  ``cache_seq_indices`` are tree destinations and
        # may be sparse (for example [2, 4, 5]), so they must not be reused as
        # source indices here.
        self._past_key_values.seq_indices = torch.arange(
            input_ids.size(1), device=self._device, dtype=torch.long
        )
        return self._model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=util.invert_mask(attention_mask),
            past_key_values=self._past_key_values,
        )[0]

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        batch_idx: int,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        cache_batch_indices = torch.zeros(
            (input_ids.size(-1),), dtype=torch.long, device=self._device
        )
        with self._past_key_values.prefill_context(input_ids.size(-1), batch_idx):
            self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=util.invert_mask(attention_mask),
                past_key_values=self._past_key_values,
            )

    def gather(self, src_indices: torch.Tensor, dest_indices: torch.Tensor) -> None:
        """Keep the KV cache aligned with official Tree sequence reordering."""

        self._past_key_values.gather(0, src_indices, dest_indices)

    def reset(self) -> None:
        self._past_key_values.clear()


class CPUCompatibleTiming:
    """Official Timing contract with a CPU-safe synchronization policy."""

    def __init__(self, device: Any = None, mode: str = "no-sync", enabled: bool = True):
        if mode not in {"no-sync", "sync", "event"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.device = device
        self.mode = mode
        self.enabled = enabled
        self.elapsed = 0.0
        self._delegate = None
        if str(device).split(":", 1)[0] != "cpu":
            self._delegate = util.Timing(device=device, mode=mode, enabled=enabled)

    def __enter__(self):
        if self._delegate is not None:
            self._delegate.__enter__()
        elif self.enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._delegate is not None:
            self._delegate.__exit__(exc_type, exc_val, exc_tb)
            self.elapsed = self._delegate.elapsed
        elif self.enabled:
            self.elapsed = (time.perf_counter() - self._start) * 1000.0


@contextmanager
def cpu_timing_adapter() -> Iterator[None]:
    """Scope the CPU Timing adapter around an official SpecExec cycle.

    The official module is not edited.  The adapter is scoped to one client
    process/cycle and restored immediately, which makes the compatibility
    boundary explicit and testable instead of relying on a shell-level patch.
    """

    original_timing = util.Timing
    util.Timing = CPUCompatibleTiming
    try:
        yield
    finally:
        util.Timing = original_timing


class CPUCompatibleSpecExecClient(SpecExecClient):
    """Official tree/proactive client with CPU-safe timing only."""

    adapter_name = "specedge_cpu_adapted"

    async def _cycle(self, req_idx: int, step_idx: int, prefill: bool = False):
        with cpu_timing_adapter():
            return await super()._cycle(req_idx, step_idx, prefill=prefill)
