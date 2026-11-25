"""Sample target GPU power with NVML outside the inference process."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .gpu_device_mapping import resolve_target_nvml_gpu_indices
from .gpu_energy import EnergyAccumulator, NvidiaPowerReader, sum_power_w


@dataclass
class MonitorConfig:
    experiment_id: str
    output_dir: Path
    sample_interval_s: float = 0.25
    gpu_indices: tuple[int, ...] = ()
    host: str = field(default_factory=socket.gethostname)


class PowerMonitor:
    """Measure energy and sample power over the monitor's own time window."""

    def __init__(self, config: MonitorConfig, *, gpu: NvidiaPowerReader | None = None,
                 clock=time.time, monotonic=time.monotonic):
        self.config = config
        if not config.gpu_indices:
            raise ValueError("target GPU power monitoring requires at least one physical GPU")
        if gpu is not None and len(config.gpu_indices) != 1:
            raise ValueError("a single injected GPU reader requires one physical GPU index")
        self.gpu_readers = (
            [gpu] if gpu is not None else [NvidiaPowerReader(index) for index in config.gpu_indices]
        )
        self.clock = clock
        self.monotonic = monotonic
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_epoch: float | None = None
        self._start_mono: float | None = None
        self._end_epoch: float | None = None
        self._end_mono: float | None = None
        self._previous_sample_mono: float | None = None
        self._energy = EnergyAccumulator(len(self.gpu_readers))
        self.hardware_samples: list[dict[str, Any]] = []

    def sample_once(self, *, final: bool = False) -> dict[str, Any]:
        samples = [reader.sample() for reader in self.gpu_readers]
        now_epoch = float(self.clock())
        now_mono = float(self.monotonic())
        previous = self._previous_sample_mono
        interval = now_mono - previous if previous is not None else None
        self._previous_sample_mono = now_mono
        if final:
            self._energy.stop(now_mono, samples)
        elif previous is None:
            self._energy.start(now_mono, samples)
        else:
            self._energy.update(now_mono, samples)
        energy = self._energy.measurement()
        row = {
            "schema_version": 2,
            "experiment_id": self.config.experiment_id,
            "role": "target",
            "host": self.config.host,
            "gpu_indices": list(self.config.gpu_indices),
            "timestamp_epoch_s": now_epoch,
            "timestamp_monotonic_s": now_mono,
            "sampling_interval_s": interval,
            "nvml_available": all(reader.available for reader in self.gpu_readers),
            "nvml_error": next((reader.error for reader in self.gpu_readers if reader.error), None),
            "gpu_power_w": sum_power_w(samples),
            "energy_source": energy["energy_source"],
            "cumulative_gpu_energy_j": energy["energy_j"] if energy["duration_s"] else None,
        }
        self.hardware_samples.append(row)
        return row

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("power monitor has already started")
        first = self.sample_once()
        self._start_epoch = first["timestamp_epoch_s"]
        self._start_mono = first["timestamp_monotonic_s"]
        self._thread = threading.Thread(target=self._loop, name="fastsd-power-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = max(0.01, self.config.sample_interval_s)
        while not self._stop_event.wait(interval):
            self.sample_once()

    def stop(self) -> dict[str, Any]:
        if self._thread is None or self._stop_event.is_set():
            raise RuntimeError("power monitor is not running")
        self._stop_event.set()
        self._thread.join()
        try:
            last = self.sample_once(final=True)
            self._end_epoch = last["timestamp_epoch_s"]
            self._end_mono = last["timestamp_monotonic_s"]
            result = self.summary()
            self._write_outputs(result)
            return result
        finally:
            for reader in self.gpu_readers:
                reader.close()

    def summary(self) -> dict[str, Any]:
        duration = (self._end_mono - self._start_mono
                    if self._start_mono is not None and self._end_mono is not None else None)
        measurement = self._energy.measurement()
        energy = measurement["energy_j"] if measurement["duration_s"] else None
        measured = energy is not None
        average = measurement["avg_power_w"] if measured else None
        fallback_indices = [index for index, reader in zip(self.config.gpu_indices, self.gpu_readers)
                            if reader.counter_unsupported]
        return {
            "schema_version": 2,
            "experiment_id": self.config.experiment_id,
            "role": "target",
            "host": self.config.host,
            "gpu_indices": list(self.config.gpu_indices),
            "start_timestamp_epoch_s": self._start_epoch,
            "end_timestamp_epoch_s": self._end_epoch,
            "duration_s": duration,
            "energy_source": measurement["energy_source"],
            "power_integration_gpu_indices": fallback_indices,
            "energy_counter_fallback_reason": "NVML_ERROR_NOT_SUPPORTED" if fallback_indices else None,
            "hardware": {
                "nvml_available": any(row["nvml_available"] for row in self.hardware_samples),
                "nvml_error": next((reader.error for reader in self.gpu_readers if reader.error), None),
                "sample_count": len(self.hardware_samples),
            },
            "gpu": {
                "measured": measured,
                "average_power_w": average,
                "total_energy_j": energy,
                "sampled_duration_s": measurement["sampled_duration_s"],
            },
            "average_gpu_power_w": average,
            "total_energy_j": energy,
        }

    def _write_outputs(self, summary: dict[str, Any]) -> None:
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        with (out / "power_samples.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.hardware_samples:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        with (out / "power_samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.hardware_samples[0]))
            writer.writeheader()
            writer.writerows(self.hardware_samples)
        (out / "power_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["target"], default="target")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-interval-ms", type=float, default=250.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    gpu_indices = resolve_target_nvml_gpu_indices(
        target_device=os.environ.get("FASTSD_TARGET_DEVICE", "cpu"),
        target_tp_size=int(os.environ.get("TARGET_TP_SIZE", "1")),
        target_tp_gpus=os.environ.get("TARGET_TP_GPUS", ""),
    )
    if not gpu_indices:
        raise SystemExit("target GPU power monitoring skipped: target device is CPU")
    monitor = PowerMonitor(MonitorConfig(
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        sample_interval_s=max(0.01, args.sample_interval_ms / 1000.0),
        gpu_indices=gpu_indices,
    ))
    stop = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    monitor.start()
    stop.wait(args.duration_s if args.duration_s > 0 else None)
    print(json.dumps(monitor.stop(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
