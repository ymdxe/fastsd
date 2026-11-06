"""Durable per-host sequence: SD, FastSD, then SpecEdge on one frozen trace.

Start one GPU manager and one CPU manager on each configured CPU host.
Each invocation claims a new campaign directory; use another suffix to retry.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
METHODS = ("sd", "fastsd", "specedge")


def write_state(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def request(url, route, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + route, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def identities(trace, suffix):
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", suffix):
        raise ValueError("Use a short alphanumeric campaign suffix, optionally with dot, underscore or hyphen")
    return [f"{method}_{trace}_{suffix}" for method in METHODS]


def configured_root(config, role, node=None):
    if role == "gpu":
        return Path(config.get("gpu_result_root", config.get("result_root", "")))
    node_config = config["nodes"][node]
    value = node_config.get("result_root", config.get("cpu_result_root", config.get("result_root")))
    if not value:
        raise ValueError("A CPU result root must be configured")
    return Path(value)


def child_environment(config, role):
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NO_PROXY": "*", "no_proxy": "*",
    })
    if role == "gpu":
        uuid = config.get("gpu_uuid", config.get("target_gpu_uuid", config.get("gpu_visible_devices")))
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or "," in uuid:
            raise ValueError("GPU campaign requires a single explicitly configured GPU UUID")
        env["CUDA_VISIBLE_DEVICES"] = uuid
    else:
        # clients.py sets each Draft's eight-thread environment before spawning it.
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def stop_child(process, grace=45):
    """Signal only a child launched here; its own launcher cleans up workers."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is None:
        # start_new_session=True made this child's PID its private process group.
        try:
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError("Owned child's process-group identity changed")
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def query_gpu_memory(uuid, timeout=10):
    """Read only the selected physical GPU; this function never creates CUDA."""
    if not isinstance(uuid, str) or not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("Expected one configured physical GPU UUID")
    result = subprocess.run(
        ["nvidia-smi", "--id=" + uuid,
         "--query-gpu=uuid,memory.total,memory.free,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip()[:500])
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("Expected one physical GPU memory record")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 4 or fields[0] != uuid:
        raise RuntimeError("nvidia-smi returned a different GPU identity")
    total, free, used = map(float, fields[1:])
    if not all(math.isfinite(x) and x >= 0 for x in (total, free, used)) or total <= 0 or free > total:
        raise RuntimeError("Invalid physical GPU memory record")
    return {"gpu_uuid": uuid, "total_mib": total, "free_mib": free, "used_mib": used}


def wait_gpu_memory(config, output, method):
    """Bounded pre-CUDA admission gate; coordinator remains live while waiting."""
    if method not in METHODS:
        raise ValueError("Unknown experiment method")
    defaults = {"sd": 20480, "fastsd": 20480, "specedge": 26624}
    required = float(config.get("gpu_start_free_mib", defaults)[method])
    timeout = float(config.get("gpu_start_wait_timeout_s", 86400))
    if not math.isfinite(required) or required <= 0:
        raise ValueError("GPU startup free-memory threshold must be positive")
    if not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("GPU resource waiting must be bounded by 24 hours")
    uuid = config["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("GPU resource gate does not match CUDA_VISIBLE_DEVICES")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started, epoch = time.monotonic(), time.time()
    deadline = started + timeout
    status = {
        "phase": "model_start_wait", "method": method, "pid": os.getpid(),
        "gpu_uuid": uuid, "required_free_mib": required,
        "wait_started_epoch_s": epoch, "wait_deadline_epoch_s": epoch + timeout,
        "wait_timeout_s": timeout, "heartbeat_interval_s": 15,
        "gpu_memory_fraction": config["gpu_memory_fraction"],
        "model_ready": False, "query_failures": 0,
    }

    def publish():
        status["updated_epoch_s"] = time.time()
        status["wait_elapsed_s"] = time.monotonic() - started
        status["wait_remaining_s"] = max(0.0, deadline - time.monotonic())
        write_state(output / "model-start.json", status)

    publish()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("GPU free-memory admission exceeded " + str(timeout) + " seconds")
            try:
                memory = query_gpu_memory(uuid, timeout=min(10, remaining))
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                status["query_failures"] += 1
                status["last_query_error"] = repr(exc)
                publish()
                if status["query_failures"] >= 3:
                    raise RuntimeError("Three consecutive GPU memory-query failures") from exc
            else:
                status.update(memory)
                status["query_failures"] = 0
                status.pop("last_query_error", None)
                if required > memory["total_mib"]:
                    raise ValueError("GPU startup threshold exceeds physical device capacity")
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU resource-wait deadline elapsed during memory query")
                if memory["free_mib"] >= required:
                    status.update(phase="loading", admitted_epoch_s=time.time())
                    publish()
                    return dict(status)
                publish()
            time.sleep(min(15, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        status.update(phase="failed", error=type(exc).__name__ + ": " + str(exc))
        publish()
        raise


def model_start_health(output, *, alive, ready):
    """Publish a resource wait separately from ordinary model loading."""
    health = {"status": "ok" if ready else ("loading" if alive else "failed"),
              "model_ready": bool(ready), "worker_alive": bool(alive)}
    path = Path(output) / "model-start.json"
    if not path.exists():
        return health
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        health["model_start_state_error"] = repr(exc)
        return health
    health["model_start"] = state
    age = max(0.0, time.time() - float(state.get("updated_epoch_s", 0)))
    health["resource_heartbeat_age_s"] = age
    # Remaining time is computed on the GPU host, so CPU clock skew cannot
    # extend the resource deadline. The Target also enforces it monotonically.
    health["resource_wait_remaining_s"] = max(
        0.0, float(state.get("wait_remaining_s", 0)) - age)
    if alive and not ready:
        if state.get("phase") == "model_start_wait":
            health["status"] = "model_start_wait"
        elif state.get("phase") == "failed":
            health["status"] = "failed"
    return health


class Campaign:
    def __init__(self, config, options):
        self.config = config
        self.options = options
        self.node = platform.node().split(".")[0]
        self.run_ids = identities(options.trace, options.suffix)
        timeout = float(config["campaign_timeout_s"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("campaign_timeout_s must be finite and positive")
        if options.role == "cpu" and self.node not in config["nodes"]:
            raise ValueError("This host is not a configured CPU node")
        self.env = child_environment(config, options.role)
        result_root = configured_root(config, options.role, self.node)
        if not str(result_root) or str(result_root) == ".":
            raise ValueError("An explicit campaign result root is required")
        self.result_root = result_root
        directory = result_root / f"_campaign_{options.trace}_{options.suffix}"
        if options.role == "cpu":
            directory = directory / self.node
        directory.mkdir(parents=True, exist_ok=False)
        self.output = directory
        self.started = time.monotonic()
        self.deadline = self.started + timeout
        self.server_url = config.get("server_url", "http://127.0.0.1:29659")
        self.child = None
        self.active_info = None
        self.status = {
            "role": options.role, "host": self.node, "trace": options.trace,
            "suffix": options.suffix, "run_ids": self.run_ids, "pid": os.getpid(),
            "started_epoch_s": time.time(), "campaign_timeout_s": timeout,
            "state": "starting", "stages": [],
        }
        manifest = ROOT / "manifest.json"
        self.bundle_id = json.loads(manifest.read_text(encoding="utf-8"))["bundle_id"]
        self.status["bundle_id"] = self.bundle_id
        self.save()

    def save(self):
        self.status["updated_epoch_s"] = time.time()
        self.status["elapsed_s"] = time.monotonic() - self.started
        write_state(self.output / "status.json", self.status)

    def remaining(self):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Campaign exceeded campaign_timeout_s")
        return value

    def pause(self, duration):
        until = time.monotonic() + duration
        while time.monotonic() < until:
            self.remaining()
            time.sleep(min(0.5, max(0, until - time.monotonic())))

    def read_remote(self, route):
        remaining = self.remaining()
        startup_deadline = getattr(self, "startup_deadline", None)
        if startup_deadline is not None:
            remaining = min(remaining, startup_deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Target startup deadline elapsed")
        return request(self.server_url, route, timeout=min(5, max(0.1, remaining)))

    def check_upstream(self, expected_run, stage_index):
        info = self.read_remote("/experiment/info")
        observed_run = info.get("run_id")
        if observed_run not in self.run_ids[:stage_index + 1]:
            return info, None
        state = self.read_remote("/experiment/state")
        if state.get("run_id") == observed_run and state.get("aborted"):
            raise RuntimeError("Upstream aborted " + observed_run + ": " + str(state["aborted"]))
        return info, state

    def wait_ready(self, run_id, method, stage_index, stage):
        startup_timeout = min(1800.0, self.remaining())
        self.startup_deadline = time.monotonic() + startup_timeout
        stage["startup_timeout_s"] = startup_timeout
        last_save = 0
        while True:
            self.remaining()
            if time.monotonic() >= self.startup_deadline:
                detail = stage.get("last_connection_error", stage.get("last_health", "no matching ready target"))
                raise TimeoutError(f"Target {run_id} was not ready within {startup_timeout:.0f}s: {detail}")
            try:
                info, remote_state = self.check_upstream(run_id, stage_index)
                observed = info.get("run_id")
                stage["observed_run_id"] = observed
                stage.pop("last_connection_error", None)
                if observed in self.run_ids[stage_index + 1:]:
                    raise RuntimeError("GPU has advanced past this CPU stage: " + str(observed))
                if observed in self.run_ids and info.get("bundle_id") != self.bundle_id:
                    raise RuntimeError("Upstream frozen bundle differs from this CPU host")
                if observed in self.run_ids[:stage_index]:
                    expected_workers = set(info.get("expected_workers", []))
                    reported = set((remote_state or {}).get("reported", []))
                    globally_complete = bool(expected_workers) and reported == expected_workers
                    if not globally_complete:
                        health = self.read_remote("/experiment/health")
                        if health.get("status") in ("failed", "error", "dead"):
                            raise RuntimeError("Previous global stage failed: " + str(health))
                        if health.get("status") == "ok" and health.get("model_ready") is True:
                            # This host may finish hours before slower hosts. Their
                            # healthy remaining work is not next-model startup time.
                            self.startup_deadline = time.monotonic() + min(1800.0, self.remaining())
                            stage["waiting_reason"] = "previous global stage still draining"
                            stage["previous_stage_reported_workers"] = len(reported)
                    else:
                        stage["waiting_reason"] = "previous global stage completed; waiting for next model"
                if observed == run_id:
                    if info.get("method") != method or info.get("trace") != self.options.trace:
                        raise RuntimeError("Upstream method/trace does not match the campaign")
                    if info.get("bundle_id") != self.bundle_id:
                        raise RuntimeError("Upstream frozen bundle differs from this CPU host")
                    health = self.read_remote("/experiment/health")
                    stage["last_health"] = health
                    if health.get("status") in ("failed", "error", "dead"):
                        raise RuntimeError("Upstream target failed: " + str(health))
                    if health.get("status") == "model_start_wait":
                        resource = health.get("model_start", {})
                        wait_limit = float(self.config.get("gpu_start_wait_timeout_s", 86400))
                        remote_limit = float(resource.get("wait_timeout_s", 0))
                        remaining_wait = float(health.get("resource_wait_remaining_s", 0))
                        heartbeat_age = float(health.get("resource_heartbeat_age_s", float("inf")))
                        if (health.get("worker_alive") is not True or health.get("model_ready") is not False
                                or resource.get("method") != method
                                or resource.get("gpu_uuid") != self.config["gpu_uuid"]
                                or not 0 < remote_limit <= min(wait_limit, 86400)):
                            raise RuntimeError("Invalid upstream GPU resource-wait state: " + str(health))
                        if not math.isfinite(remaining_wait) or remaining_wait <= 0 or remaining_wait > remote_limit:
                            raise RuntimeError("Upstream GPU resource-wait deadline elapsed or is invalid")
                        if not math.isfinite(heartbeat_age) or heartbeat_age > 60:
                            raise RuntimeError("Upstream GPU resource-wait heartbeat is stale")
                        if heartbeat_age <= 60:
                            self.startup_deadline = time.monotonic() + min(1800.0, self.remaining())
                            stage["waiting_reason"] = "GPU free memory below model startup threshold"
                            stage["resource_wait"] = resource
                            stage["resource_wait_was_active"] = True
                    elif stage.get("resource_wait_was_active") and not stage.get("resource_wait_completed_epoch_s"):
                        # Start a fresh bounded loading interval once admission ends.
                        self.startup_deadline = time.monotonic() + min(1800.0, self.remaining())
                        stage["resource_wait_completed_epoch_s"] = time.time()
                        stage["waiting_reason"] = "GPU memory admitted; waiting for real model readiness"
                    if health.get("status") == "ok" and health.get("model_ready") is True:
                        self.startup_deadline = None
                        self.active_info = info
                        stage["target_ready_epoch_s"] = time.time()
                        self.save()
                        return info
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                # Loading/restarting the target may temporarily remove the HTTP endpoint.
                stage["last_connection_error"] = repr(exc)
            if time.monotonic() - last_save >= 15:
                self.save()
                last_save = time.monotonic()
            self.pause(1)

    def monitor_child(self, run_id, stage_index, stage):
        last_remote = 0
        outage_started = None
        last_save = 0
        while self.child.poll() is None:
            self.remaining()
            if self.options.role == "cpu" and time.monotonic() - last_remote >= 5:
                try:
                    info, remote_state = self.check_upstream(run_id, stage_index)
                    if info.get("run_id") == run_id:
                        health = self.read_remote("/experiment/health")
                        if health.get("status") in ("failed", "error", "dead"):
                            raise RuntimeError("Upstream target failed: " + str(health))
                    outage_started = None
                    stage.pop("last_connection_error", None)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                    stage["last_connection_error"] = repr(exc)
                    outage_started = time.monotonic() if outage_started is None else outage_started
                    if time.monotonic() - outage_started >= 1800:
                        raise TimeoutError("Upstream HTTP endpoint continuously unreachable for 1800s") from exc
                last_remote = time.monotonic()
            if time.monotonic() - last_save >= 15:
                self.save()
                last_save = time.monotonic()
            self.pause(0.5)
        code = self.child.wait()
        stage["exit_code"] = code
        stage["finished_epoch_s"] = time.time()
        self.child = None
        self.save()
        if code != 0:
            raise RuntimeError(run_id + " exited with status " + str(code))

    def run_stage(self, stage_index, method, run_id):
        stage = {
            "method": method, "run_id": run_id, "state": "waiting",
            "started_epoch_s": time.time(),
        }
        self.status["stages"].append(stage)
        self.status.update(state="running", active_run_id=run_id)
        self.save()
        if self.options.role == "gpu":
            command = [
                self.config["gpu_python"], "-B", "-u", str(ROOT / "suite.py"),
                "server", method, self.options.trace, run_id,
            ]
        else:
            info = self.wait_ready(run_id, method, stage_index, stage)
            command = [
                info["nodes"][self.node]["python"], "-B", "-u", str(ROOT / "clients.py"),
                run_id, "--server-url", self.server_url, "--timeout", str(self.remaining()),
            ]
        stage["command"] = command
        stage["log_file"] = str(self.output / (run_id + ".log"))
        with Path(stage["log_file"]).open("x", encoding="utf-8") as log:
            self.child = subprocess.Popen(
                command, cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stage.update(state="running", pid=self.child.pid, pgid=self.child.pid,
                         child_started_epoch_s=time.time())
            self.save()
            print(json.dumps({"campaign": str(self.output), "run_id": run_id, "pid": self.child.pid}), flush=True)
            self.monitor_child(run_id, stage_index, stage)
        if self.options.role == "gpu":
            summary_path = self.result_root / run_id / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("run_id") != run_id or summary.get("method") != method:
                raise RuntimeError("Summary does not match the completed stage")
            if summary.get("measurement_valid") is not True:
                raise RuntimeError("Measurement is incomplete or invalid: " + str(summary_path))
            stage["summary_file"] = str(summary_path)
            stage["measurement_valid"] = True
            stage["completed_requests"] = summary.get("completed_requests")
            stage["total_output_tokens"] = summary.get("total_output_tokens")
        stage["state"] = "complete"
        self.save()
        print(json.dumps({"run_id": run_id, "state": "complete"}), flush=True)
        if self.options.role == "gpu" and stage_index < len(METHODS) - 1:
            # CPU launchers may still be flushing logs after their final report.
            self.pause(10)

    def abort_own_run(self, reason):
        run_id = self.status.get("active_run_id")
        if self.options.role != "cpu" or not run_id:
            return
        try:
            info = request(self.server_url, "/experiment/info", timeout=3)
            if info.get("run_id") == run_id and info.get("bundle_id") == self.bundle_id:
                request(self.server_url, "/experiment/abort", {
                    "run_id": run_id, "reason": self.node + " campaign: " + reason,
                }, timeout=3)
        except Exception:
            pass

    def run(self):
        def interrupted(signum, frame):
            raise KeyboardInterrupt("campaign signal " + str(signum))

        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupted)
        try:
            for index, (method, run_id) in enumerate(zip(METHODS, self.run_ids)):
                self.run_stage(index, method, run_id)
            self.status["state"] = "complete"
        except BaseException as exc:
            self.status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
            if self.status["stages"] and self.status["stages"][-1]["state"] != "complete":
                self.status["stages"][-1]["state"] = "failed"
            self.save()
            self.abort_own_run(repr(exc))
            try:
                stop_child(self.child)
            except Exception as cleanup_error:
                self.status["cleanup_error"] = repr(cleanup_error)
            raise
        finally:
            self.status["finished_epoch_s"] = time.time()
            self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--trace", choices=("smoke", "main"), required=True)
    parser.add_argument("--suffix", default="a")
    options = parser.parse_args(argv)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    Campaign(config, options).run()


if __name__ == "__main__":
    main()
