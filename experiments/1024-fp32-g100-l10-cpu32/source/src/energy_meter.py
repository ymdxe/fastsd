import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    BaseModel = object

    def Field(*args, **kwargs):  # type: ignore
        del args, kwargs
        return None

    uvicorn = None


class EnergyAccumulator:
    """Accumulate energy by integrating sampled power over time."""

    def __init__(self):
        self._total_energy_j = 0.0
        self._total_duration_s = 0.0
        self._running = False

    def start(self, timestamp_s: float) -> None:
        del timestamp_s
        if self._running:
            raise RuntimeError("accumulator is already running")
        self._running = True

    def update(self, power_w: float, dt_s: float) -> None:
        if not self._running:
            return
        if dt_s <= 0:
            return
        self._total_energy_j += power_w * dt_s
        self._total_duration_s += dt_s

    def pause(self, timestamp_s: float) -> None:
        del timestamp_s
        if self._running:
            self._running = False

    def stop(self, timestamp_s: float) -> dict[str, float]:
        if not self._running:
            raise RuntimeError("accumulator is not running")
        self.pause(timestamp_s)
        return self.summary()

    def summary(self) -> dict[str, float]:
        avg_power = (
            self._total_energy_j / self._total_duration_s
            if self._total_duration_s > 0
            else 0.0
        )
        return {
            "energy_j": self._total_energy_j,
            "duration_s": self._total_duration_s,
            "avg_power_w": avg_power,
        }

    @property
    def is_running(self) -> bool:
        return self._running


class MeasureStartRequest(BaseModel):
    exp_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeasureStopRequest(BaseModel):
    exp_id: str = "default"
    total_tasks: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class EnergyServiceConfig:
    gpu_index: int
    host: str
    port: int
    sample_interval_ms: int
    output_path: str


class _CloudEnergyManager:
    def __init__(self, cfg: EnergyServiceConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sample_interval_s = max(0.01, float(cfg.sample_interval_ms) / 1000.0)

        self._campaign = EnergyAccumulator()
        self._active = EnergyAccumulator()

        self._active_depth = 0
        self._campaign_started = False
        self._exp_id = "default"
        self._result: Optional[dict[str, Any]] = None

        try:
            import pynvml  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pynvml is required for energy measurement. Install with `pip install nvidia-ml-py3`."
            ) from exc

        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(cfg.gpu_index)

        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)

    def start(self) -> None:
        self._sampler_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._sampler_thread.join(timeout=3.0)
        self._pynvml.nvmlShutdown()

    def _read_power_w(self) -> float:
        return float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0

    def _sampler_loop(self) -> None:
        prev_t = time.monotonic()
        while not self._stop_event.is_set():
            power_w = self._read_power_w()
            now_t = time.monotonic()
            dt = now_t - prev_t
            prev_t = now_t

            with self._lock:
                if self._campaign_started and self._campaign.is_running:
                    self._campaign.update(power_w, dt)
                if self._active_depth > 0 and self._active.is_running:
                    self._active.update(power_w, dt)

            time.sleep(self._sample_interval_s)

    def start_campaign(self, exp_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        del metadata
        now_t = time.monotonic()
        with self._lock:
            if self._campaign_started:
                return {
                    "status": "already_started",
                    "exp_id": self._exp_id,
                }
            self._campaign_started = True
            self._exp_id = exp_id
            self._campaign.start(now_t)
        return {
            "status": "started",
            "exp_id": exp_id,
        }

    def stop_campaign(
        self,
        exp_id: str,
        total_tasks: Optional[int],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        now_t = time.monotonic()
        with self._lock:
            if not self._campaign_started:
                raise RuntimeError("campaign has not started")
            if self._exp_id != exp_id:
                raise RuntimeError(
                    f"exp_id mismatch: running={self._exp_id}, got={exp_id}"
                )

            self._campaign.pause(now_t)
            if self._active.is_running:
                self._active.pause(now_t)

            active_summary = self._active.summary()
            campaign_summary = self._campaign.summary()
            overhead_j = campaign_summary["energy_j"] - active_summary["energy_j"]
            overhead_ratio = (
                campaign_summary["energy_j"] / active_summary["energy_j"]
                if active_summary["energy_j"] > 0
                else 0.0
            )

            self._campaign_started = False
            self._active_depth = 0

        self._result = {
            "exp_id": exp_id,
            "result_a_active_only": active_summary,
            "result_b_campaign_window": campaign_summary,
            "delta_idle_overhead": {
                "energy_j": overhead_j,
                "ratio": overhead_ratio,
            },
            "total_tasks": total_tasks,
            "metadata": metadata,
        }

        out_path = Path(self._cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self._result, f, ensure_ascii=False, indent=2)

        return self._result

    def enter_active(self) -> None:
        now_t = time.monotonic()
        with self._lock:
            self._active_depth += 1
            if self._active_depth == 1 and not self._active.is_running:
                self._active.start(now_t)

    def exit_active(self) -> None:
        now_t = time.monotonic()
        with self._lock:
            if self._active_depth <= 0:
                self._active_depth = 0
                return
            self._active_depth -= 1
            if self._active_depth == 0 and self._active.is_running:
                self._active.pause(now_t)


class EnergyControlService:
    def __init__(self, cfg: EnergyServiceConfig):
        if FastAPI is None or uvicorn is None:
            raise RuntimeError(
                "FastAPI energy control requires fastapi/uvicorn. Install with `pip install fastapi uvicorn`."
            )
        self._cfg = cfg
        self._manager = _CloudEnergyManager(cfg)

        app = FastAPI()

        @app.post("/measure/start")
        def start_measure(req: MeasureStartRequest):
            try:
                return self._manager.start_campaign(req.exp_id, req.metadata)
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @app.post("/measure/stop")
        def stop_measure(req: MeasureStopRequest):
            try:
                return self._manager.stop_campaign(
                    req.exp_id,
                    req.total_tasks,
                    req.metadata,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        self._app = app
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=cfg.host,
                port=cfg.port,
                log_level="warning",
            )
        )
        self._server_thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._manager.start()
        self._server_thread.start()

        timeout_s = 10.0
        start_t = time.time()
        while not self._server.started:
            if time.time() - start_t > timeout_s:
                raise RuntimeError("energy FastAPI service failed to start in time")
            time.sleep(0.05)

    def shutdown(self) -> None:
        self._server.should_exit = True
        self._server_thread.join(timeout=3.0)
        self._manager.shutdown()

    def enter_active(self) -> None:
        self._manager.enter_active()

    def exit_active(self) -> None:
        self._manager.exit_active()
