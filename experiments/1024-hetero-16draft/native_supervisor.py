"""Finish the native paired run, checking each stage before advancing."""
from pathlib import Path, PurePosixPath
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
GPU_PYTHON = "/home/hdd/zhangh/envs/fastsd/bin/python"
REMOTE_ROOT = "/home/zhangh/workspace/fastsd/experiments/1024-hetero-16draft"


def record(stage, **data):
    entry = {"stage": stage, "epoch": time.time(), **data}
    (ROOT / "native_status.json").write_text(json.dumps(entry, indent=2))
    with (ROOT / "native_stages.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


def wait_until(check, timeout=14400):
    start = time.monotonic()
    while not check():
        if time.monotonic() - start > timeout:
            raise TimeoutError("Stage exceeded its allowed wall time")
        time.sleep(5)


def collect(method, nodes):
    wait_until(lambda: all((ROOT / method / (node + ".tar.gz")).exists() for node in nodes))
    for node in nodes:
        with tarfile.open(ROOT / method / (node + ".tar.gz")) as archive:
            for item in archive.getmembers():
                path = PurePosixPath(item.name)
                assert not path.is_absolute() and ".." not in path.parts and path.parts[0] == node
                assert item.isfile() or item.isdir()
            archive.extractall(ROOT / method)
        node_root = ROOT / method / node
        status = json.loads((node_root / "exit.json").read_text())
        assert status["exit_status"] == 0, (method, node, status)
        rows = [json.loads(line) for path in (node_root / "run").glob("edge_metrics_proc*.jsonl") for line in path.read_text().splitlines()]
        assert len(rows) == (1 if method.startswith("smoke-") else 8), (method, node, len(rows))
        rpc = [json.loads(line) for path in (node_root / "run").glob("rpc-*.jsonl") for line in path.read_text().splitlines()]
        assert all(row["ok"] for row in rpc)
        if not method.startswith("smoke-"):
            expected = 100 if node == "node1" else 20
            assert all(row["gamma"] == expected for row in rpc if row["endpoint"] == "/verify")


def stop_target(method):
    output = ROOT / method / "target"
    pid = int((output / "target.pid").read_text())
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    assert str(ROOT / "target_adapter.py").encode() in command
    request = urllib.request.Request("http://127.0.0.1:18625/exit", data=b"{}", headers={"Content-Type": "application/json"})
    response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30).read().decode()
    (output / "shutdown.json").write_text(response)
    worker = json.loads((output / "target_model_ready.json").read_text())["pid"]
    # A worker may be a zombie until its server parent handles shutdown.
    def worker_stopped():
        path = Path(f"/proc/{worker}/status")
        return not path.exists() or "State:\tZ" in path.read_text()
    wait_until(worker_stopped, timeout=180)
    os.kill(pid, signal.SIGINT)
    wait_until(lambda: (output / "exit.json").exists(), timeout=60)


def send_cpu(method, node):
    pane = "6:0.0" if node == "node1" else "6:0.1"
    python = "/home/zhangh/venvs/fastsd_cpu/bin/python" if node == "node1" else "python"
    command = python + " -u " + REMOTE_ROOT + "/launch_cpu.py " + method + " " + node
    subprocess.run(["tmux", "send-keys", "-t", pane, "-l", command], check=True)
    subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], check=True)


def main():
    record("standard_running")
    collect("standard", ["node1", "node3"])
    record("standard_complete")
    stop_target("standard")
    record("fastsd_target_loading")
    with (ROOT / "fastsd-launch.log").open("w") as log:
        subprocess.Popen([GPU_PYTHON, "-u", str(ROOT / "launch_target.py"), "fastsd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    ready = ROOT / "fastsd/target/target_model_ready.json"
    def target_ready():
        if not ready.exists():
            return False
        value = json.loads(ready.read_text())
        if value.get("state") == "failed":
            raise RuntimeError(value)
        return value.get("ready", False)
    wait_until(target_ready, timeout=1800)
    record("fastsd_smoke_running")
    send_cpu("smoke-fastsd", "node1")
    collect("smoke-fastsd", ["node1"])
    record("fastsd_running")
    send_cpu("fastsd", "node1")
    send_cpu("fastsd", "node3")
    collect("fastsd", ["node1", "node3"])
    record("fastsd_complete")
    stop_target("fastsd")
    subprocess.run([sys.executable, str(ROOT / "summarize.py"), "--methods", "standard", "fastsd", "--require-complete"], check=True)
    record("native_complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("failed", error=repr(exc))
        raise
