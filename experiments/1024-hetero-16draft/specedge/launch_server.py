"""Launch the isolated SpecEdge Target after Standard SD/FastSD have stopped."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["specedge", "smoke-specedge"], default="specedge")
    parser.add_argument("--smoke-first", action="store_true", help="Load once; smoke gamma2/sync1, then management POST /phase enables the formal run")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18625)
    parser.add_argument("--management-port", type=int, default=18626)
    parser.add_argument("--sync-clients", type=int)
    parser.add_argument("--node1-gamma", type=int)
    parser.add_argument("--node3-gamma", type=int)
    args = parser.parse_args()
    output = ROOT.parent / args.run / "target"
    if (output / "launch.json").exists():
        raise FileExistsError(f"Refusing to overwrite a Target run: {output}")
    with socket.socket() as probe:
        # Match the real server's address reuse: native HTTP connections may
        # still be in TIME_WAIT even after its listener has shut down.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.host, args.port))
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.management_port))
    smoke = args.run == "smoke-specedge" or args.smoke_first
    command = [args.python, "-u", str(ROOT / "server_adapter.py"),
               "--host", args.host, "--port", str(args.port), "--output", str(output),
               "--management-port", str(args.management_port),
               "--model", str(ROOT.parent / "model-view" / "Qwen3-32B"),
               "--batch-size", "8", "--max-len", "768",
               "--phase", "smoke" if smoke else "formal",
               "--sync-clients", str(args.sync_clients or (1 if smoke else 16)),
               "--node1-gamma", str(args.node1_gamma or (2 if smoke else 100)),
               "--node3-gamma", str(args.node3_gamma or (2 if smoke else 20))]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="3", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                       TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.json").write_text(json.dumps({
        "command": command,
        "environment": {k: environment[k] for k in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "HF_HUB_OFFLINE"]},
        "method": "specedge_cpu_adapted_single_chain", "started_epoch": time.time(),
    }, indent=2) + "\n")
    with (output / "target.log").open("w") as stream:
        child = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        (output / "target.pid").write_text(str(child.pid) + "\n")
        status = child.wait()
    (output / "exit.json").write_text(json.dumps({"exit_status": status, "ended_epoch": time.time()}) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
