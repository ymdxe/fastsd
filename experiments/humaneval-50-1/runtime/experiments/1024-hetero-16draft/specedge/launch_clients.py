"""Launch this CPU node's independent SpecEdge clients; no SSH fan-out."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("node1", "node3"))
    parser.add_argument("--run", default="specedge", choices=("specedge", "smoke-specedge"))
    parser.add_argument("--server", default="127.0.0.1:28625")
    parser.add_argument("--control-url", default="http://127.0.0.1:28624")
    parser.add_argument("--model", default="/mnt/nfs0/zhangh/models/Qwen3-1.7B")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--workers", type=int, choices=(1, 8))
    parser.add_argument("--expected", type=int)
    parser.add_argument("--client-offset", type=int)
    parser.add_argument("--gamma", type=int, help="Smoke only: override the node's fixed gamma")
    parser.add_argument("--max-new-tokens", type=int, help="Smoke only: override output cap")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--token-file", type=Path, default=EXPERIMENT / ".coordinator-token")
    return parser.parse_args()


def make_environment(args, worker, gamma, cap, output):
    env = os.environ.copy()
    values = {
        "OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1,
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": 1, "TRANSFORMERS_OFFLINE": 1,
        "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": 1,
        "SPECEDGE_OPTIMIZATION": 2, "SPECEDGE_RESULT_PATH": str(output.parent),
        "SPECEDGE_EXP_NAME": "run", "SPECEDGE_PROCESS_NAME": f"client_{worker}",
        "SPECEDGE_SEED": 1234 + worker,
        "SPECEDGE_MAX_LEN": args.max_len, "SPECEDGE_DRAFT_MODEL": args.model,
        "SPECEDGE_DEVICE": "cpu", "SPECEDGE_DTYPE": "fp16", "SPECEDGE_THREADS": 1,
        "SPECEDGE_DATASET": "fastsd_external", "SPECEDGE_REASONING": "False",
        "SPECEDGE_MAX_N_BEAMS": 1, "SPECEDGE_MAX_BEAM_LEN": gamma,
        "SPECEDGE_MAX_BRANCH_WIDTH": 1, "SPECEDGE_MAX_BUDGET": gamma,
        "SPECEDGE_PROACTIVE_TYPE": "included", "SPECEDGE_PROACTIVE_MAX_N_BEAMS": 1,
        "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": gamma, "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": 1,
        "SPECEDGE_PROACTIVE_MAX_BUDGET": gamma + 1, "SPECEDGE_MAX_NEW_TOKENS": cap,
        "SPECEDGE_MAX_REQUEST_NUM": 1, "SPECEDGE_REQ_OFFSET": 0, "SPECEDGE_SAMPLE_REQ_CNT": 1,
        "SPECEDGE_HOST": args.server, "SPECEDGE_CLIENT_IDX": args.client_offset + worker,
        "EXP_NODE": args.node, "EXP_GAMMA": gamma, "EXP_REQUEST_TIMEOUT": 7200,
        "EXP_RPC_LOG": str(output / f"rpc-{worker:02d}.jsonl"),
        "EXP_CYCLE_LOG": str(output / f"cycle-{worker:02d}.jsonl"),
    }
    env.update({key: str(value) for key, value in values.items()})
    if args.node == "node1":
        # Match this experiment's established node1 CPU environment: the venv
        # provides torch/Transformers, and Conda supplies remaining libraries.
        env["PYTHONPATH"] = (
            "/home/zhangh/venvs/fastsd_cpu/lib/python3.11/site-packages:"
            "/home/zhangh/miniconda3/envs/fastsd/lib/python3.11/site-packages"
        )
    return env, values


def main():
    args = parse_args()
    smoke = args.run == "smoke-specedge"
    if not smoke and any(value is not None for value in (args.gamma, args.max_new_tokens)):
        raise ValueError("Formal run gamma and output cap are frozen to this experiment")
    args.workers = args.workers if args.workers is not None else (1 if smoke else 8)
    args.expected = args.expected if args.expected is not None else (1 if smoke else 16)
    args.client_offset = args.client_offset if args.client_offset is not None else (0 if smoke or args.node == "node1" else 8)
    gamma = args.gamma if args.gamma is not None else (100 if args.node == "node1" else 20)
    cap = args.max_new_tokens if args.max_new_tokens is not None else (4 if smoke else 256)
    if not 1 <= gamma <= 100 or not 1 <= cap <= 256 or not 1 <= args.expected <= 16:
        raise ValueError("Invalid gamma, cap, or expected client count")
    if not smoke and (args.workers != 8 or args.expected != 16 or args.client_offset != (0 if args.node == "node1" else 8)):
        raise ValueError("Formal run requires 8 clients/node with unique global client IDs")
    placement = EXPERIMENT / f"{args.node}-worker-map.json"
    if not placement.is_file():
        raise FileNotFoundError("Reuse the completed Standard/FastSD worker mapping: " + str(placement))
    mapping = json.loads(placement.read_text())
    if len(mapping) < args.workers:
        raise ValueError("Worker mapping has too few CPU placements")
    node_output = EXPERIMENT / args.run / args.node
    output = node_output / "run"
    if output.exists():
        raise FileExistsError("Refusing to overwrite previous SpecEdge results: " + str(output))
    output.mkdir(parents=True)
    (node_output / "worker_map.json").write_bytes(placement.read_bytes())
    (node_output / "preflight.json").write_text(json.dumps({"node": args.node, "python": args.python, "epoch": time.time(), "loadavg": os.getloadavg(),
                                                            "meminfo": Path("/proc/meminfo").read_text()}, indent=2))
    processes = []
    logs = []
    launches = []
    try:
        for worker in range(args.workers):
            env, settings = make_environment(args, worker, gamma, cap, output)
            command = [args.python, "-u", str(ROOT / "client_adapter.py"), "--node", args.node,
                       "--worker", str(worker), "--run", args.run, "--expected", str(args.expected),
                       "--output", str(output), "--control-url", args.control_url,
                       "--token-file", str(args.token_file)]
            logfile = (node_output / f"worker-{worker:02d}.log").open("w")
            logs.append(logfile)
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
            processes.append(process)
            launches.append({"worker": worker, "pid": process.pid, "command": command, "settings": settings})
        (node_output / "launch.json").write_text(json.dumps({"gamma": gamma, "cap": cap, "workers": args.workers, "expected": args.expected,
                                                           "method": "specedge_cpu_adapted_single_chain", "launches": launches}, indent=2))
        while any(process.poll() is None for process in processes):
            progress = {"epoch": time.time(), "node": args.node, "ready": len(list(output.glob("worker-*.json"))),
                        "completed": len(list(output.glob("edge_metrics_proc*.jsonl"))), "loadavg": os.getloadavg(),
                        "exit_codes": [p.poll() for p in processes]}
            with (node_output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\n")
            print(args.run, args.node, json.dumps(progress), flush=True)
            # A failed worker prevents startup barriers from completing. Stop
            # only siblings created by this launcher, preserving other jobs.
            if any(p.poll() is not None and p.returncode != 0 for p in processes):
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            running = [p for p in processes if p.poll() is None]
            if running:
                try:
                    running[0].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for logfile in logs:
            logfile.close()
    exit_codes = [p.returncode for p in processes]
    code = 0 if len(processes) == args.workers and all(value == 0 for value in exit_codes) else 1
    (node_output / "exit.json").write_text(json.dumps({"exit_status": code, "worker_exit_codes": exit_codes, "epoch": time.time()}, indent=2))
    if not args.skip_upload:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(node_output, arcname=args.node)
        query = urllib.parse.urlencode({"run": args.run, "node": args.node})
        request = urllib.request.Request(args.control_url.rstrip("/") + "/upload?" + query,
                                         data=buffer.getvalue(), method="POST",
                                         headers={"Authorization": "Bearer " + args.token_file.read_text().strip()})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=120) as response:
            print("EXP_UPLOAD", response.read().decode(), "EXIT", code, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
