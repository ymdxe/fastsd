"""Run the aligned single-chain SpecEdge comparison after both native methods."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request

from native_supervisor import collect, wait_until

ROOT = Path(__file__).resolve().parent
GPU_PYTHON = "/home/hdd/zhangh/envs/new_fastsd/bin/python"
REMOTE_ROOT = "/home/zhangh/workspace/fastsd/experiments/1024-hetero-16draft"
PACKAGE_SHA256 = "584031572543e2657de1b7bb700fc04413ae0f931e3808ca1dbf759f3875546b"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def record(stage, **data):
    value = {"stage": stage, "epoch": time.time(), **data}
    (ROOT / "specedge_status.json").write_text(json.dumps(value, indent=2))
    with (ROOT / "specedge_stages.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def management(path, body=None):
    request = urllib.request.Request("http://127.0.0.1:18626/" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    return json.loads(OPENER.open(request, timeout=90).read())


def send_cpu(node, smoke=False):
    pane = "6:0.0" if node == "node1" else "6:0.1"
    python = "/home/zhangh/venvs/fastsd_cpu/bin/python" if node == "node1" else "python"
    command = python + " -u " + REMOTE_ROOT + "/specedge/launch_clients.py " + node
    if smoke:
        command += " --run smoke-specedge --gamma 2 --max-new-tokens 4"
    subprocess.run(["tmux", "send-keys", "-t", pane, "-l", command], check=True)
    subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], check=True)


def main():
    record("waiting_for_native")
    def native_finished():
        path = ROOT / "native_status.json"
        if not path.exists():
            return False
        status = json.loads(path.read_text())
        if status["stage"] == "failed":
            raise RuntimeError("Native experiment failed; SpecEdge has not been started: " + repr(status))
        return status["stage"] == "native_complete"
    wait_until(native_finished, timeout=28800)
    record("distributing_specedge")
    package = ROOT / "specedge-code.tar.gz"
    assert hashlib.sha256(package.read_bytes()).hexdigest() == PACKAGE_SHA256
    shutil.copyfile(package, ROOT / "bundle.tar.gz")
    for pane in ["6:0.0", "6:0.1"]:
        subprocess.run([sys.executable, str(ROOT / "tmux_io.py"), "bootstrap", pane], check=True)
    record("specedge_target_loading")
    with (ROOT / "specedge-launch.log").open("w") as log:
        subprocess.Popen([GPU_PYTHON, "-u", str(ROOT / "specedge/launch_server.py"), "--smoke-first"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    ready_path = ROOT / "specedge/target/target_model_ready.json"
    def ready():
        if not ready_path.exists():
            return False
        value = json.loads(ready_path.read_text())
        if value.get("state") == "failed":
            raise RuntimeError(value)
        if not value.get("ready", False):
            return False
        health = management("health")
        return health.get("ready") and health.get("worker_alive")
    wait_until(ready, timeout=1800)
    record("specedge_smoke_running")
    send_cpu("node1", smoke=True)
    collect("smoke-specedge", ["node1"])
    phase = management("phase", {"phase": "formal"})
    (ROOT / "specedge/phase_transition.json").write_text(json.dumps(phase, indent=2))
    record("specedge_running")
    send_cpu("node1")
    send_cpu("node3")
    collect("specedge", ["node1", "node3"])
    record("specedge_complete")
    stopped = management("shutdown", {})
    (ROOT / "specedge/target/shutdown.json").write_text(json.dumps(stopped))
    wait_until(lambda: (ROOT / "specedge/target/exit.json").exists(), timeout=180)
    assert json.loads((ROOT / "specedge/target/exit.json").read_text())["exit_status"] == 0
    subprocess.run([sys.executable, str(ROOT / "summarize.py"), "--require-complete"], check=True)
    record("all_methods_complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("failed", error=repr(exc))
        raise
