"""Run one node's independent drafts and upload the complete experiment archive."""
from pathlib import Path
import collections
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
method, node = sys.argv[1:3]
assert method in {"standard", "fastsd", "smoke-standard", "smoke-fastsd"}
assert node in {"node1", "node3"}
smoke = method.startswith("smoke-")
fast = method.endswith("fastsd")
count = 1 if smoke else 8
gamma = 2 if smoke else (100 if node == "node1" else 20)
cap = 4 if smoke else 256
out = ROOT / method / node
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run").exists(), "Refusing to overwrite an existing run"
(out / "run").mkdir()
mapping_path = ROOT / (node + "-worker-map.json")
if not mapping_path.exists():
    def cpustats():
        return {int(p[0][3:]): [int(x) for x in p[1:]] for line in Path("/proc/stat").read_text().splitlines() if (p := line.split()) and p[0].startswith("cpu") and p[0][3:].isdigit()}
    first = cpustats()
    time.sleep(1)
    second = cpustats()
    busy = {}
    for cpu in first:
        delta = [b-a for a,b in zip(first[cpu], second[cpu])]
        busy[cpu] = 1 - (delta[3] + delta[4]) / max(1, sum(delta[:8]))
    topology = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], capture_output=True, text=True, check=True).stdout
    groups = {}
    allowed = os.sched_getaffinity(0)
    for line in topology.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, numa = map(int, line.split(","))
        if cpu in allowed:
            groups.setdefault((socket, core), []).append({"cpu": cpu, "core": core, "socket": socket, "numa": numa})
    by_node = collections.defaultdict(list)
    for siblings in groups.values():
        p = min(siblings, key=lambda x: x["cpu"])
        p["observed_core_busy_fraction"] = max(busy[x["cpu"]] for x in siblings)
        by_node[p["numa"]].append(p)
    selected = []
    for numa, candidates in sorted(by_node.items()):
        selected.extend(sorted(candidates, key=lambda x: (x["observed_core_busy_fraction"], x["cpu"]))[:8 // len(by_node)])
    assert len(selected) == 8
    mapping_path.write_text(json.dumps(selected, indent=2))
(out / "worker_map.json").write_bytes(mapping_path.read_bytes())
preflight = {"node": node, "python": sys.executable, "epoch": time.time(), "loadavg": os.getloadavg(), "meminfo": Path("/proc/meminfo").read_text(), "topology": subprocess.run(["lscpu"], capture_output=True, text=True).stdout}
(out / "preflight.json").write_text(json.dumps(preflight, indent=2))
model = "/mnt/nfs0/zhangh/models/Qwen3-1.7B"
args = [sys.executable, "-u", str(ROOT / "edge_adapter.py"), "--draft_model", model, "--target_model", model,
        "--draft_dtype", "float16", "--target_dtype", "float16", "--edge_use_cpu", "--edge_threads", "1",
        "--num_drafts", str(count), "--server_url", "http://127.0.0.1:28625", "--request_timeout", "7200",
        "--dataset", "gsm8k", "--dataset_file", str(ROOT / (node + "-workload.jsonl")),
        "--profile", "custom" if fast else "vanilla", "--server_sched_mode", "fastsd" if fast else "vanilla",
        "--no-pipeline_gamma_adapt", "--pipeline_gamma_min", str(gamma), "--pipeline_gamma_max", str(gamma),
        "--gamma", str(gamma), "--max_tokens", str(cap), "--verify_method", "greedy", "--temp", "0",
        "--top_k", "0", "--top_p", "1", "--stop_policy", "eos", "--warmup_requests", "0", "--exp_name", str(out / "run")]
for feature in ["enable_pipeline", "enable_proactive_draft", "enable_latency_priority"]:
    args.append("--" + ("" if fast else "no-") + feature)
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1",
           EXP_RUN=method, EXP_NODE=node, EXP_EXPECTED="1" if smoke else "16")
if node == "node1":
    env["PYTHONPATH"] = "/home/zhangh/venvs/fastsd_cpu/lib/python3.11/site-packages:/home/zhangh/miniconda3/envs/fastsd/lib/python3.11/site-packages"
(out / "launch.json").write_text(json.dumps({"command": args, "gamma": gamma, "cap": cap, "workers": count}, indent=2))
with (out / "run.log").open("w") as log:
    child = subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (out / "draft.pid").write_text(str(child.pid))
    while child.poll() is None:
        completed = sum(len(p.read_text().splitlines()) for p in (out / "run").glob("edge_metrics_proc*.jsonl"))
        row = {"epoch": time.time(), "ready": len(list((out / "run").glob("worker-*.json"))), "completed": completed, "loadavg": os.getloadavg()}
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(method, node, row, flush=True)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
    code = child.returncode
(out / "exit.json").write_text(json.dumps({"exit_status": code, "epoch": time.time()}))
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
    archive.add(out, arcname=node)
request = urllib.request.Request("http://127.0.0.1:28624/upload?run="+method+"&node="+node, data=buffer.getvalue(), headers={"Authorization": "Bearer "+(ROOT/".coordinator-token").read_text().strip()}, method="POST")
response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120)
print("EXP_UPLOAD", response.read().decode(), "EXIT", code, flush=True)
sys.exit(code)
