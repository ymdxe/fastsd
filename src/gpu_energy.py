"""NVML GPU readings and energy accounting for explicit measurement windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class GpuEnergySample:
    power_w: float | None = None
    counter_mj: int | None = None
    counter_unsupported: bool = False


class NvidiaPowerReader:
    def __init__(self, gpu_index: int, *, pynvml_module: Any = None):
        self.available = False
        self.error: str | None = None
        self.counter_unsupported = False
        self._initialized = False
        self._pynvml = pynvml_module
        try:
            if self._pynvml is None:
                import pynvml
                self._pynvml = pynvml
            self._pynvml.nvmlInit()
            self._initialized = True
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.available = True
        except Exception as exc:
            self.error = f"NVML unavailable: {exc}"
            self.close()

    def sample(self) -> GpuEnergySample:
        if not self.available:
            return GpuEnergySample()
        counter_mj = None
        self.error = None
        if not self.counter_unsupported:
            if not hasattr(self._pynvml, "nvmlDeviceGetTotalEnergyConsumption"):
                # Older pynvml builds lack the total-energy counter entirely.
                # Treat that exactly like NVML_ERROR_NOT_SUPPORTED so the
                # accumulator falls back to power integration.
                self.counter_unsupported = True
            else:
                try:
                    counter_mj = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                    if not isinstance(counter_mj, int) or counter_mj < 0:
                        raise ValueError(f"invalid energy counter: {counter_mj!r}")
                except AttributeError:
                    counter_mj = None
                    self.counter_unsupported = True
                except Exception as exc:
                    counter_mj = None
                    if (isinstance(exc, getattr(self._pynvml, "NVMLError", ()))
                            and exc.value == self._pynvml.NVML_ERROR_NOT_SUPPORTED):
                        self.counter_unsupported = True
                    else:
                        self.error = f"NVML total energy read failed: {exc}"
        power_w = None
        try:
            value = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            if math.isfinite(value) and value >= 0:
                power_w = value
        except Exception as exc:
            self.error = self.error or f"NVML power read failed: {exc}"
        return GpuEnergySample(power_w, counter_mj, self.counter_unsupported)

    def close(self) -> None:
        if self._initialized:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
        self.available = False


def sum_power_w(samples: Sequence[GpuEnergySample]) -> float | None:
    if samples and all(sample.power_w is not None for sample in samples):
        return sum(sample.power_w for sample in samples)
    return None


class _GpuWindow:
    def __init__(self, first: GpuEnergySample):
        self.first = self.last = first
        self.last_counter = first.counter_mj
        self.counter_reset = False
        self.integrated_j = self.integrated_s = 0.0

    def update(self, sample: GpuEnergySample, elapsed: float) -> None:
        if sample.counter_mj is not None:
            if self.last_counter is not None and sample.counter_mj < self.last_counter:
                self.counter_reset = True
            self.last_counter = sample.counter_mj
        if sample.power_w is not None and elapsed > 0:
            self.integrated_j += sample.power_w * elapsed
            self.integrated_s += elapsed
        self.last = sample

    def measurement(self, duration_s: float) -> tuple[float | None, float, str]:
        if self.last.counter_unsupported:
            return (self.integrated_j if self.integrated_s else None,
                    self.integrated_s, "power_integration")
        if self.counter_reset or self.first.counter_mj is None or self.last.counter_mj is None:
            return None, 0.0, "nvml_total_energy"
        return (self.last.counter_mj - self.first.counter_mj) / 1000.0, duration_s, "nvml_total_energy"


class EnergyAccumulator:
    """Combine disjoint windows without charging energy between them."""

    def __init__(self, gpu_count: int):
        self._windows: list[_GpuWindow] = []
        self._totals: list[float | None] = [0.0] * gpu_count
        self._coverage = [0.0] * gpu_count
        self._sources: set[str] = set()
        self._duration_s = 0.0
        self._power_samples: list[float] = []

    @property
    def is_running(self) -> bool:
        return bool(self._windows)

    def start(self, timestamp_s: float, samples: Sequence[GpuEnergySample]) -> None:
        if self.is_running:
            raise RuntimeError("accumulator is already running")
        self._started_at = self._last_at = timestamp_s
        self._windows = [_GpuWindow(sample) for sample in samples]

    def update(self, timestamp_s: float, samples: Sequence[GpuEnergySample]) -> None:
        elapsed = max(0.0, timestamp_s - self._last_at)
        for window, sample in zip(self._windows, samples, strict=True):
            window.update(sample, elapsed)
        self._last_at = timestamp_s
        power_w = sum_power_w(samples)
        if power_w is not None and elapsed > 0:
            self._power_samples.append(power_w)

    def stop(self, timestamp_s: float, samples: Sequence[GpuEnergySample]) -> None:
        if not self.is_running:
            raise RuntimeError("accumulator is not running")
        self.update(timestamp_s, samples)
        duration = self._last_at - self._started_at
        self._totals, self._coverage, self._sources = self._combined(duration)
        self._duration_s += duration
        self._windows.clear()

    def _combined(self, duration: float) -> tuple[list[float | None], list[float], set[str]]:
        totals, coverage, sources = self._totals.copy(), self._coverage.copy(), self._sources.copy()
        if duration > 0:
            for index, window in enumerate(self._windows):
                energy, seconds, source = window.measurement(duration)
                prior = totals[index]
                totals[index] = prior + energy if prior is not None and energy is not None else None
                coverage[index] += seconds
                sources.add(source)
        return totals, coverage, sources

    def measurement(self) -> dict[str, Any]:
        duration = self._last_at - self._started_at if self.is_running else 0.0
        totals, coverage, sources = self._combined(duration)
        energy = None if None in totals else sum(totals)
        average = None if energy is None else sum(
            total / seconds for total, seconds in zip(totals, coverage) if seconds > 0
        )
        return {
            "energy_j": energy,
            "duration_s": self._duration_s + duration,
            "sampled_duration_s": min(coverage, default=0.0),
            "avg_power_w": average,
            "energy_source": "mixed" if len(sources) > 1 else next(iter(sources), None),
        }

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self._power_samples)

        def quantile(q: float) -> float:
            if not ordered:
                return 0.0
            position = (len(ordered) - 1) * q
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        return {
            **self.measurement(),
            "peak_power_w": max(ordered, default=0.0),
            "power_p95_w": quantile(0.95),
            "power_p99_w": quantile(0.99),
            "power_sample_count": len(ordered),
        }
