"""Low-overhead power monitoring for comparative FastSD experiments.

The monitor deliberately lives beside the inference pipeline.  It reads raw
hardware counters, samples the CPU time of an explicitly supplied process
tree, and writes raw and estimated values to separate files.  It never changes
speculative decoding or scheduling behaviour.

Typical use from a shell wrapper::

    python -m src.power_monitor --role edge --experiment-id run-1 \
      --output-dir exp/run-1/edge --root-pid "$PPID"

The process tree is opt-in.  This is important because a machine's package
counter includes unrelated work; process estimates therefore use the host CPU
time denominator and cannot silently assign all package energy to FastSD.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_RAPL_ROOT = "/sys/class/powercap"


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class RaplDomain:
    path: Path
    name: str
    max_energy_uj: Optional[float]


@dataclass(frozen=True)
class RaplReading:
    timestamp_epoch_s: float
    timestamp_monotonic_s: float
    energy_uj: Optional[float]
    package_power_w: Optional[float]
    sampling_interval_s: Optional[float]
    available: bool
    error: Optional[str] = None
    wrapped: bool = False


class RaplReader:
    """Read package energy counters under ``/sys/class/powercap``.

    Only top-level ``intel-rapl:N`` domains are included.  Child DRAM/core
    domains would double count package energy and are intentionally ignored.
    """

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_RAPL_ROOT,
                 *, clock=time.time, monotonic=time.monotonic):
        self.root = Path(root)
        self.clock = clock
        self.monotonic = monotonic
        self.domains = self._discover_domains()
        self.available = bool(self.domains)
        self.error = None if self.available else "intel RAPL package counters unavailable"
        self._previous_counters: dict[str, float] = {}
        self._previous_mono: Optional[float] = None

    def _discover_domains(self) -> list[RaplDomain]:
        domains: list[RaplDomain] = []
        try:
            candidates = sorted(
                path for path in self.root.glob("intel-rapl:*")
                if path.is_dir() and re.fullmatch(r"intel-rapl:[0-9]+", path.name)
            )
        except OSError:
            return domains
        for path in candidates:
            energy_path = path / "energy_uj"
            if not energy_path.is_file():
                continue
            name = path.name
            try:
                name_file = path / "name"
                if name_file.is_file():
                    name = name_file.read_text(encoding="utf-8", errors="replace").strip() or name
            except OSError:
                pass
            maximum: Optional[float] = None
            try:
                maximum = _finite((path / "max_energy_range_uj").read_text().strip())
                if maximum is not None and maximum <= 0:
                    maximum = None
            except (OSError, ValueError):
                pass
            domains.append(RaplDomain(path=path, name=name, max_energy_uj=maximum))
        return domains

    @staticmethod
    def _read_counter(path: Path) -> Optional[float]:
        try:
            value = _finite(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        return value if value is not None and value >= 0 else None

    def _read_counters(self) -> tuple[dict[str, float], Optional[str]]:
        if not self.domains:
            return {}, self.error
        counters: dict[str, float] = {}
        missing: list[str] = []
        for domain in self.domains:
            current = self._read_counter(domain.path / "energy_uj")
            if current is None:
                missing.append(domain.path.name)
                continue
            counters[domain.path.name] = current
        if missing:
            return {}, "RAPL counter read failed: " + ",".join(missing)
        return counters, None

    def sample(self) -> RaplReading:
        epoch = float(self.clock())
        mono = float(self.monotonic())
        counters, error = self._read_counters()
        current = sum(counters.values()) if counters else None
        wrapped = False
        interval = mono - self._previous_mono if self._previous_mono is not None else None
        power: Optional[float] = None
        if counters and self._previous_counters and interval and interval > 0:
            delta_total = 0.0
            for domain in self.domains:
                key = domain.path.name
                if key not in counters or key not in self._previous_counters:
                    continue
                delta = counters[key] - self._previous_counters[key]
                if delta < 0:
                    if domain.max_energy_uj is not None:
                        delta += domain.max_energy_uj
                        wrapped = True
                    else:
                        error = "RAPL counter decreased without max_energy_range_uj"
                        delta = 0.0
                delta_total += max(0.0, delta)
            power = (delta_total * 1e-6) / interval
        if counters:
            self._previous_counters = counters
        self._previous_mono = mono
        return RaplReading(
            timestamp_epoch_s=epoch,
            timestamp_monotonic_s=mono,
            energy_uj=current,
            package_power_w=power,
            sampling_interval_s=interval,
            available=current is not None,
            error=error,
            wrapped=wrapped,
        )

    read = sample


@dataclass(frozen=True)
class ProcessStat:
    pid: int
    process_name: str
    cpu_time_ticks: int
    cpu_time_s: float
    state: str


def parse_proc_stat(text: str, pid: int, clk_tck: int = 100) -> ProcessStat:
    """Parse Linux ``/proc/<pid>/stat`` without breaking names containing ``)``."""
    closing = text.rfind(")")
    opening = text.find("(")
    if opening < 0 or closing <= opening:
        raise ValueError("malformed /proc stat: missing process name")
    process_name = text[opening + 1:closing]
    fields = text[closing + 1:].strip().split()
    # fields[0] is field 3 (state), fields[11] is field 14 (utime), and
    # fields[12] is field 15 (stime).
    if len(fields) < 13:
        raise ValueError("malformed /proc stat: missing CPU fields")
    state = fields[0]
    ticks = int(fields[11]) + int(fields[12])
    return ProcessStat(
        pid=int(pid),
        process_name=process_name,
        cpu_time_ticks=ticks,
        cpu_time_s=ticks / float(clk_tck),
        state=state,
    )


class ProcReader:
    """Read process CPU time and total host CPU time from procfs."""

    def __init__(self, proc_root: str | os.PathLike[str] = "/proc", *, clk_tck: Optional[int] = None):
        self.proc_root = Path(proc_root)
        self.clk_tck = int(clk_tck or os.sysconf(os.sysconf_names["SC_CLK_TCK"]))

    def read_stat(self, pid: int) -> Optional[ProcessStat]:
        try:
            text = (self.proc_root / str(int(pid)) / "stat").read_text(encoding="utf-8")
            return parse_proc_stat(text, int(pid), self.clk_tck)
        except (OSError, ValueError, TypeError):
            return None

    def total_cpu_time_s(self) -> Optional[float]:
        try:
            with (self.proc_root / "stat").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("cpu "):
                        values = [int(value) for value in line.split()[1:9]]
                        return sum(values) / float(self.clk_tck)
        except (OSError, ValueError):
            return None
        return None

    def children(self, pid: int) -> list[int]:
        try:
            raw = (self.proc_root / str(int(pid)) / "task" / str(int(pid)) / "children").read_text()
        except OSError:
            return []
        result: list[int] = []
        for token in raw.split():
            try:
                result.append(int(token))
            except ValueError:
                continue
        return result

    def process_tree(self, roots: Iterable[int], *, include_descendants: bool = True,
                     excluded: Iterable[int] = ()) -> list[int]:
        excluded_set = {int(pid) for pid in excluded}
        pending = [int(pid) for pid in roots]
        seen: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in seen or pid in excluded_set or pid <= 0:
                continue
            seen.add(pid)
            if include_descendants:
                pending.extend(self.children(pid))
        return sorted(seen)


class NvidiaPowerReader:
    """Best-effort NVML power reader; CPU-only hosts remain usable."""

    def __init__(self, gpu_index: int = 0, *, pynvml_module: Any = None):
        self.available = False
        self.error: Optional[str] = None
        self._pynvml = pynvml_module
        self._handle = None
        try:
            if self._pynvml is None:
                import pynvml  # type: ignore
                self._pynvml = pynvml
            self._pynvml.nvmlInit()
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            self.available = True
        except Exception as exc:  # NVML is optional on edge/CPU nodes.
            self.error = f"NVML unavailable: {exc}"

    def read_power_w(self) -> Optional[float]:
        if not self.available or self._handle is None:
            return None
        try:
            value = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            return value if math.isfinite(value) and value >= 0 else None
        except Exception as exc:
            self.error = f"NVML read failed: {exc}"
            return None

    read = read_power_w

    def close(self) -> None:
        if self.available and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self.available = False


def attribute_package_power(
    package_power_w: Optional[float],
    process_cpu_deltas_s: Mapping[int, float],
    host_cpu_delta_s: Optional[float] = None,
) -> dict[int, Optional[float]]:
    """Allocate measured package power to selected processes.

    With a host denominator, unrelated CPU work receives the remainder and is
    never attributed to FastSD.  The denominator-free form is retained for
    deterministic offline analysis and tests, where it distributes power
    proportionally among the supplied PIDs.
    """
    if package_power_w is None or package_power_w < 0:
        return {int(pid): None for pid in process_cpu_deltas_s}
    positive = {int(pid): max(0.0, float(delta)) for pid, delta in process_cpu_deltas_s.items()}
    total_selected = sum(positive.values())
    denominator = float(host_cpu_delta_s) if host_cpu_delta_s is not None and host_cpu_delta_s > 0 else total_selected
    if denominator <= 0:
        return {pid: 0.0 for pid in positive}
    return {pid: float(package_power_w) * delta / denominator for pid, delta in positive.items()}


@dataclass
class MonitorConfig:
    experiment_id: str
    role: str
    output_dir: Path
    sample_interval_s: float = 0.25
    pids: tuple[int, ...] = ()
    root_pids: tuple[int, ...] = ()
    excluded_pids: tuple[int, ...] = ()
    include_descendants: bool = True
    gpu_index: int = 0
    rapl_root: str = DEFAULT_RAPL_ROOT
    enable_gpu: bool = True
    host: str = field(default_factory=socket.gethostname)


class PowerMonitor:
    """Threaded sampler with explicit experiment start/stop boundaries."""

    def __init__(self, config: MonitorConfig, *, rapl: RaplReader | None = None,
                 gpu: NvidiaPowerReader | None = None, proc: ProcReader | None = None,
                 clock=time.time, monotonic=time.monotonic):
        self.config = config
        self.rapl = rapl or RaplReader(config.rapl_root, clock=clock, monotonic=monotonic)
        self.gpu = gpu if gpu is not None else (NvidiaPowerReader(config.gpu_index) if config.enable_gpu else None)
        self.proc = proc or ProcReader()
        self.clock = clock
        self.monotonic = monotonic
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started = False
        self._start_epoch: Optional[float] = None
        self._start_mono: Optional[float] = None
        self._end_epoch: Optional[float] = None
        self._end_mono: Optional[float] = None
        self._last_process: dict[int, ProcessStat] = {}
        self._last_total_cpu_s: Optional[float] = None
        self._cumulative_package_energy_j = 0.0
        self._cumulative_gpu_energy_j = 0.0
        self._cumulative_process_energy_j: dict[int, float] = {}
        self.hardware_samples: list[dict[str, Any]] = []
        self.process_samples: list[dict[str, Any]] = []

    @staticmethod
    def _optional_sum(*values: Optional[float]) -> Optional[float]:
        available = [value for value in values if value is not None]
        return sum(available) if available else None

    def _pids(self) -> list[int]:
        roots = tuple(self.config.root_pids) + tuple(self.config.pids)
        return self.proc.process_tree(roots, include_descendants=self.config.include_descendants,
                                      excluded=(os.getpid(),) + tuple(self.config.excluded_pids))

    def sample_once(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rapl = self.rapl.sample()
        now_epoch = float(self.clock())
        now_mono = float(self.monotonic())
        gpu_power = self.gpu.read_power_w() if self.gpu is not None else None
        dt = rapl.sampling_interval_s
        if dt is None or dt <= 0:
            dt = (now_mono - self._start_mono) if self._start_mono is not None else None
        if dt is not None and dt > 0:
            if rapl.package_power_w is not None:
                self._cumulative_package_energy_j += rapl.package_power_w * dt
            if gpu_power is not None:
                self._cumulative_gpu_energy_j += gpu_power * dt

        pids = self._pids()
        current: dict[int, ProcessStat] = {}
        for pid in pids:
            stat = self.proc.read_stat(pid)
            if stat is not None:
                current[pid] = stat
        total_cpu = self.proc.total_cpu_time_s()
        host_delta = total_cpu - self._last_total_cpu_s if total_cpu is not None and self._last_total_cpu_s is not None else None
        deltas: dict[int, float] = {}
        for pid, stat in current.items():
            previous = self._last_process.get(pid)
            if previous is not None:
                delta = max(0.0, stat.cpu_time_s - previous.cpu_time_s)
                deltas[pid] = delta
        estimates = attribute_package_power(rapl.package_power_w, deltas, host_delta)
        process_rows: list[dict[str, Any]] = []
        for pid, stat in current.items():
            delta = deltas.get(pid, 0.0)
            estimate = estimates.get(pid)
            energy = (estimate * dt) if estimate is not None and dt is not None and dt > 0 else None
            if energy is not None:
                self._cumulative_process_energy_j[pid] = (
                    self._cumulative_process_energy_j.get(pid, 0.0) + energy
                )
            cumulative_process_energy = (
                self._cumulative_process_energy_j.get(pid) if rapl.available else None
            )
            process_rows.append({
                "schema_version": 1,
                "experiment_id": self.config.experiment_id,
                "role": self.config.role,
                "host": self.config.host,
                "timestamp_epoch_s": now_epoch,
                "timestamp_monotonic_s": now_mono,
                "sampling_interval_s": dt,
                "pid": pid,
                "process_name": stat.process_name,
                "cpu_time_s": stat.cpu_time_s,
                "cpu_time_delta_s": delta,
                "cpu_utilization_pct": (delta / dt * 100.0) if dt and dt > 0 else None,
                "cpu_utilization": (delta / dt * 100.0) if dt and dt > 0 else None,
                "host_cpu_time_delta_s": host_delta,
                "measured_package_power_w": rapl.package_power_w,
                "gpu_power_w": gpu_power,
                "estimated_process_power_w": estimate,
                "estimated_process_energy_j": energy,
                "cumulative_estimated_process_energy_j": cumulative_process_energy,
                "cumulative_energy_j": cumulative_process_energy,
                "cumulative_package_energy_j": (
                    self._cumulative_package_energy_j if rapl.available else None
                ),
                "cumulative_gpu_energy_j": (
                    self._cumulative_gpu_energy_j
                    if self.gpu is not None and self.gpu.available else None
                ),
                "attribution": "host_cpu_time_proportional" if host_delta and host_delta > 0 else "selected_pid_proportional",
            })
        cumulative_package_energy = (
            self._cumulative_package_energy_j if rapl.available else None
        )
        cumulative_gpu_energy = (
            self._cumulative_gpu_energy_j
            if self.gpu is not None and self.gpu.available else None
        )
        hardware_row = {
            "schema_version": 1,
            "experiment_id": self.config.experiment_id,
            "role": self.config.role,
            "host": self.config.host,
            "timestamp_epoch_s": now_epoch,
            "timestamp_monotonic_s": now_mono,
            "sampling_interval_s": dt,
            "rapl_available": rapl.available,
            "rapl_error": rapl.error,
            "rapl_energy_uj": rapl.energy_uj,
            "rapl_wrapped": rapl.wrapped,
            "measured_package_power_w": rapl.package_power_w,
            "nvml_available": bool(self.gpu is not None and self.gpu.available),
            "nvml_error": self.gpu.error if self.gpu is not None else "GPU monitoring disabled",
            "gpu_power_w": gpu_power,
            "cumulative_package_energy_j": cumulative_package_energy,
            "cumulative_gpu_energy_j": cumulative_gpu_energy,
            "cumulative_energy_j": self._optional_sum(cumulative_package_energy, cumulative_gpu_energy),
            "selected_pid_count": len(current),
        }
        with self._lock:
            self.hardware_samples.append(hardware_row)
            self.process_samples.extend(process_rows)
        self._last_process = current
        self._last_total_cpu_s = total_cpu
        return hardware_row, process_rows

    def start(self) -> None:
        if self._started:
            raise RuntimeError("power monitor is already running")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._start_epoch = float(self.clock())
        self._start_mono = float(self.monotonic())
        # Establish baselines without charging a zero-length interval.
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, name="fastsd-power-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = max(0.01, float(self.config.sample_interval_s))
        while not self._stop_event.wait(interval):
            try:
                self.sample_once()
            except Exception as exc:  # keep the benchmark alive if a counter vanishes
                with self._lock:
                    self.hardware_samples.append({
                        "schema_version": 1,
                        "experiment_id": self.config.experiment_id,
                        "role": self.config.role,
                        "host": self.config.host,
                        "timestamp_epoch_s": float(self.clock()),
                        "timestamp_monotonic_s": float(self.monotonic()),
                        "sampling_interval_s": interval,
                        "monitor_error": repr(exc),
                    })

    def stop(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("power monitor is not running")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.config.sample_interval_s))
        try:
            self.sample_once()
        except Exception:
            pass
        self._end_epoch = float(self.clock())
        self._end_mono = float(self.monotonic())
        self._started = False
        self._write_outputs()
        result = self.summary()
        if self.gpu is not None:
            self.gpu.close()
        return result

    @staticmethod
    def _weighted_average(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        energy = 0.0
        duration = 0.0
        for row in rows:
            value = _finite(row.get(key))
            dt = _finite(row.get("sampling_interval_s"))
            if value is not None and dt is not None and dt > 0:
                energy += value * dt
                duration += dt
        return energy / duration if duration > 0 else None

    def summary(self) -> dict[str, Any]:
        hardware = list(self.hardware_samples)
        process = list(self.process_samples)
        duration = ((self._end_mono - self._start_mono)
                    if self._start_mono is not None and self._end_mono is not None else None)
        rapl_available = any(
            bool(row.get("rapl_available")) and row.get("rapl_energy_uj") is not None
            for row in hardware
        )
        nvml_available = any(bool(row.get("nvml_available")) for row in hardware)
        package_energy = None
        if rapl_available:
            package_energy = sum(
                (_finite(row.get("measured_package_power_w")) or 0.0)
                * (_finite(row.get("sampling_interval_s")) or 0.0)
                for row in hardware
            )
        gpu_energy = None
        if nvml_available:
            gpu_energy = sum(
                (_finite(row.get("gpu_power_w")) or 0.0)
                * (_finite(row.get("sampling_interval_s")) or 0.0)
                for row in hardware
            )
        process_energy = None
        if rapl_available:
            process_energy = sum(
                _finite(row.get("estimated_process_energy_j")) or 0.0 for row in process
            )
        per_pid: dict[str, dict[str, Any]] = {}
        for row in process:
            pid = str(row["pid"])
            entry = per_pid.setdefault(pid, {"pid": row["pid"], "process_name": row.get("process_name"),
                                             "estimated_energy_j": 0.0 if rapl_available else None,
                                             "sample_count": 0})
            if rapl_available:
                entry["estimated_energy_j"] += _finite(row.get("estimated_process_energy_j")) or 0.0
            entry["sample_count"] += 1
        average_package = self._weighted_average(hardware, "measured_package_power_w")
        average_gpu = self._weighted_average(hardware, "gpu_power_w")
        return {
            "schema_version": 1,
            "experiment_id": self.config.experiment_id,
            "role": self.config.role,
            "host": self.config.host,
            "start_timestamp_epoch_s": self._start_epoch,
            "end_timestamp_epoch_s": self._end_epoch,
            "duration_s": duration,
            "hardware": {
                "rapl_available": rapl_available,
                "rapl_error": self.rapl.error,
                "nvml_available": nvml_available,
                "nvml_error": self.gpu.error if self.gpu is not None else "GPU monitoring disabled",
                "sample_count": len(hardware),
            },
            "cpu_package": {
                "measured": rapl_available,
                "average_power_w": average_package,
                "total_energy_j": package_energy,
            },
            "gpu": {
                "measured": nvml_available,
                "average_power_w": average_gpu,
                "total_energy_j": gpu_energy,
            },
            "process_attribution": {
                "estimated": rapl_available,
                "method": "package_power_times_process_cpu_time_over_host_cpu_time",
                "aggregate_estimated_cpu_power_w": (process_energy / duration
                                                     if process_energy is not None and duration and duration > 0 else None),
                "aggregate_estimated_cpu_energy_j": process_energy,
                "process_count": len(per_pid),
                "processes": list(per_pid.values()),
            },
            "total_measured_energy_j": self._optional_sum(package_energy, gpu_energy),
            "average_cpu_package_power_w": average_package,
            "average_gpu_power_w": average_gpu,
            "total_energy_j": self._optional_sum(package_energy, gpu_energy),
            "combined_computational_energy_j": process_energy if self.config.role == "edge" else gpu_energy,
            "combined_computational_power_w": ((process_energy if self.config.role == "edge" else gpu_energy) /
                                                duration
                                                if (process_energy if self.config.role == "edge" else gpu_energy) is not None
                                                and duration and duration > 0 else None),
        }

    def _write_jsonl(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")

    def _write_csv(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys or ["timestamp_epoch_s"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_outputs(self) -> None:
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        # Hardware files intentionally remain separate from estimates.
        self._write_jsonl(out / "power_samples.jsonl", self.hardware_samples)
        self._write_jsonl(out / "power_process_samples.jsonl", self.process_samples)
        self._write_csv(out / "power_samples.csv", self.hardware_samples)
        self._write_csv(out / "power_process_samples.csv", self.process_samples)
        (out / "power_summary.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )


# Public aliases keep the names obvious to callers that prefer initialisms.
RAPLReader = RaplReader
PowerMonitorConfig = MonitorConfig


def load_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _integrate_aligned_samples(
    rows: Sequence[Mapping[str, Any]],
    power_key: str,
    start: Optional[float],
    end: Optional[float],
) -> Optional[float]:
    if start is None or end is None or end <= start:
        return None
    total = 0.0
    seen = False
    for row in rows:
        row_end = _finite(row.get("timestamp_epoch_s"))
        dt = _finite(row.get("sampling_interval_s"))
        power = _finite(row.get(power_key))
        if row_end is None or dt is None or dt <= 0 or power is None:
            continue
        row_start = row_end - dt
        overlap = max(0.0, min(row_end, end) - max(row_start, start))
        if overlap > 0:
            total += power * overlap
            seen = True
    return total if seen else None


def merge_experiment_summaries(
    target: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
    *,
    target_hardware_samples: Sequence[Mapping[str, Any]] = (),
    edge_process_samples: Sequence[Sequence[Mapping[str, Any]]] = (),
) -> dict[str, Any]:
    """Combine Edge estimated CPU energy with Target measured GPU energy.

    The interval is the timestamp intersection.  Energy counters were sampled
    over each monitor's own boundary, so the summary preserves both source
    windows and reports the aligned window used for average combined power.
    """
    target_start = _finite(target.get("start_timestamp_epoch_s"))
    target_end = _finite(target.get("end_timestamp_epoch_s"))
    starts = [value for value in [target_start] + [_finite(edge.get("start_timestamp_epoch_s")) for edge in edges] if value is not None]
    ends = [value for value in [target_end] + [_finite(edge.get("end_timestamp_epoch_s")) for edge in edges] if value is not None]
    aligned_start = max(starts) if starts else None
    aligned_end = min(ends) if ends else None
    duration = aligned_end - aligned_start if aligned_start is not None and aligned_end is not None and aligned_end >= aligned_start else None
    edge_values = [
        value for value in (
            _finite((edge.get("process_attribution") or {}).get("aggregate_estimated_cpu_energy_j"))
            for edge in edges
        )
        if value is not None
    ]
    edge_values = [
        _finite((edge.get("process_attribution") or {}).get("aggregate_estimated_cpu_energy_j"))
        for edge in edges
    ]
    edge_cpu_known = bool(edge_values) and all(value is not None for value in edge_values)
    edge_cpu_energy = sum(edge_values) if edge_cpu_known else None
    target_gpu_energy = _finite((target.get("gpu") or {}).get("total_energy_j"))
    raw_edge_cpu_energy = edge_cpu_energy
    raw_target_gpu_energy = target_gpu_energy
    if target_hardware_samples and aligned_start is not None and aligned_end is not None:
        aligned_target = _integrate_aligned_samples(
            target_hardware_samples, "gpu_power_w", aligned_start, aligned_end
        )
        if aligned_target is not None:
            target_gpu_energy = aligned_target
    if edge_process_samples and aligned_start is not None and aligned_end is not None:
        aligned_edge = 0.0
        found_edge = False
        for rows in edge_process_samples:
            value = _integrate_aligned_samples(
                rows, "estimated_process_power_w", aligned_start, aligned_end
            )
            if value is not None:
                aligned_edge += value
                found_edge = True
        if found_edge:
            edge_cpu_energy = aligned_edge
    combined_energy = None if target_gpu_energy is None or edge_cpu_energy is None else edge_cpu_energy + target_gpu_energy

    return {
        "schema_version": 1,
        "experiment_id": target.get("experiment_id"),
        "target_role": target.get("role", "target"),
        "edge_roles": [edge.get("role", "edge") for edge in edges],
        "source_windows": {
            "target_start_timestamp_epoch_s": target_start,
            "target_end_timestamp_epoch_s": target_end,
            "edge_start_timestamp_epoch_s": [_finite(edge.get("start_timestamp_epoch_s")) for edge in edges],
            "edge_end_timestamp_epoch_s": [_finite(edge.get("end_timestamp_epoch_s")) for edge in edges],
        },
        "aligned_start_timestamp_epoch_s": aligned_start,
        "aligned_end_timestamp_epoch_s": aligned_end,
        "aligned_duration_s": duration,
        "edge_cpu_estimated_energy_j": edge_cpu_energy,
        "target_gpu_measured_energy_j": target_gpu_energy,
        "combined_computational_energy_j": combined_energy,
        "average_gpu_power_w": target_gpu_energy / duration
        if target_gpu_energy is not None and duration and duration > 0 else None,
        "average_edge_cpu_power_w": edge_cpu_energy / duration
        if edge_cpu_energy is not None and duration and duration > 0 else None,
        "total_energy_j": combined_energy,
        "raw_edge_cpu_estimated_energy_j": raw_edge_cpu_energy,
        "raw_target_gpu_measured_energy_j": raw_target_gpu_energy,
        "edge_cpu_estimated_power_w": edge_cpu_energy / duration
        if edge_cpu_energy is not None and duration and duration > 0 else None,
        "target_gpu_measured_power_w": target_gpu_energy / duration
        if target_gpu_energy is not None and duration and duration > 0 else None,
        "combined_computational_power_w": combined_energy / duration
        if combined_energy is not None and duration and duration > 0 else None,
        "double_counting_policy": "edge estimated CPU plus target measured GPU only",
    }


def _enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _run_cli(args: argparse.Namespace) -> int:
    config = MonitorConfig(
        experiment_id=args.experiment_id,
        role=args.role,
        output_dir=args.output_dir,
        sample_interval_s=max(0.01, args.sample_interval_ms / 1000.0),
        pids=tuple(args.pid),
        root_pids=tuple(args.root_pid),
        excluded_pids=tuple(args.exclude_pid),
        include_descendants=args.include_descendants,
        gpu_index=args.gpu_index,
        rapl_root=args.rapl_root,
        enable_gpu=args.enable_gpu and args.role != "edge_cpu",
    )
    monitor = PowerMonitor(config)
    stop = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    monitor.start()
    deadline = time.monotonic() + args.duration_s if args.duration_s > 0 else None
    while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
        stop.wait(0.25)
    summary = monitor.stop()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["target", "edge", "edge_cpu"], default="edge")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-interval-ms", type=float, default=250.0)
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--root-pid", type=int, action="append", default=[])
    parser.add_argument("--exclude-pid", type=int, action="append", default=[])
    parser.add_argument("--include-descendants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--enable-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rapl-root", default=DEFAULT_RAPL_ROOT)
    parser.add_argument("--duration-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    return _run_cli(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
