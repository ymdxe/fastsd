import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .gpu_energy import EnergyAccumulator, NvidiaPowerReader

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


class MeasureStartRequest(BaseModel):
    exp_id: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeasureStopRequest(BaseModel):
    exp_id: str = "default"
    total_tasks: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class EnergyServiceConfig:
    gpu_indices: tuple[int, ...]
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

        self._active_depth = 0
        self._campaign_started = False
        self._exp_id = "default"
        self._result: Optional[dict[str, Any]] = None

        if not cfg.gpu_indices:
            raise ValueError("energy measurement requires physical GPU indices")
        self._readers = [NvidiaPowerReader(index) for index in cfg.gpu_indices]
        error = next((reader.error for reader in self._readers if not reader.available), None)
        if error is not None:
            for reader in self._readers:
                reader.close()
            raise RuntimeError(error)

        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)

    def start(self) -> None:
        self._sampler_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._sampler_thread.join()
        for reader in self._readers:
            reader.close()

    def _sample(self):
        samples = [reader.sample() for reader in self._readers]
        return time.monotonic(), samples

    def _sampler_loop(self) -> None:
        while not self._stop_event.wait(self._sample_interval_s):
            with self._lock:
                if self._campaign_started:
                    snapshot = self._sample()
                    self._campaign.update(*snapshot)
                    if self._active.is_running:
                        self._active.update(*snapshot)

    def start_campaign(self, exp_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        del metadata
        with self._lock:
            if self._campaign_started:
                return {
                    "status": "already_started",
                    "exp_id": self._exp_id,
                }
            self._campaign = EnergyAccumulator(len(self._readers))
            self._active = EnergyAccumulator(len(self._readers))
            self._campaign.start(*self._sample())
            self._campaign_started = True
            self._exp_id = exp_id
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
        with self._lock:
            if not self._campaign_started:
                raise RuntimeError("campaign has not started")
            if self._exp_id != exp_id:
                raise RuntimeError(
                    f"exp_id mismatch: running={self._exp_id}, got={exp_id}"
                )

            snapshot = self._sample()
            self._campaign.stop(*snapshot)
            if self._active.is_running:
                self._active.stop(*snapshot)

            active_summary = self._active.summary()
            campaign_summary = self._campaign.summary()
            campaign_j, active_j = campaign_summary["energy_j"], active_summary["energy_j"]
            overhead_j = campaign_j - active_j if campaign_j is not None and active_j is not None else None
            overhead_ratio = (campaign_j / active_j if active_j > 0 else 0.0) if overhead_j is not None else None
            fallback_indices = [index for index, reader in zip(self._cfg.gpu_indices, self._readers)
                                if reader.counter_unsupported]
            nvml_error = next((reader.error for reader in self._readers if reader.error), None)

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
            # Explicit system-prefixed aliases make the report consumable by
            # evaluators that distinguish system energy from request energy.
            "system_energy_j": campaign_summary["energy_j"],
            "system_duration_s": campaign_summary["duration_s"],
            "system_avg_power_w": campaign_summary["avg_power_w"],
            "system_peak_power_w": campaign_summary["peak_power_w"],
            "system_power_p95_w": campaign_summary["power_p95_w"],
            "system_power_p99_w": campaign_summary["power_p99_w"],
            "system_power_sample_count": campaign_summary["power_sample_count"],
            "sampled_gpu_indices": list(self._cfg.gpu_indices),
            "energy_source": campaign_summary["energy_source"],
            "power_integration_gpu_indices": fallback_indices,
            "energy_counter_fallback_reason": "NVML_ERROR_NOT_SUPPORTED" if fallback_indices else None,
            "nvml_error": nvml_error,
            "total_tasks": total_tasks,
            "metadata": metadata,
        }

        out_path = Path(self._cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self._result, f, ensure_ascii=False, indent=2)

        return self._result

    def enter_active(self) -> None:
        with self._lock:
            if not self._campaign_started:
                return
            self._active_depth += 1
            if self._active_depth == 1:
                self._active.start(*self._sample())

    def exit_active(self) -> None:
        with self._lock:
            if self._active_depth <= 0:
                self._active_depth = 0
                return
            self._active_depth -= 1
            if self._active_depth == 0 and self._active.is_running:
                self._active.stop(*self._sample())


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
