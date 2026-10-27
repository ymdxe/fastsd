from pathlib import Path
import collections
import json
import re
import tarfile
import sys
import time
import urllib.request
import urllib.error

root = Path(__file__).resolve().parent
method = sys.argv[1] if len(sys.argv) > 1 else "standard"
token = (root / ".coordinator-token").read_text().strip()
request = urllib.request.Request("http://10.66.0.5:18624/status?run=" + method, headers={"Authorization": "Bearer " + token})
try:
    state = json.loads(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10).read())
except urllib.error.URLError:
    # Final timing records remain useful after the temporary coordinator exits.
    state = json.loads((root / method / "timing.json").read_text())
result = {"method": method, "epoch": time.time(), "ready": len(state["ready"]), "clock_done": len(state["clock_done"]), "done": len(state["done"]), "worker_success": sum(v["success"] for v in state["done"].values())}
result["done_by_node"] = {node: sum(key.startswith(node + "-") for key in state["done"]) for node in ["node1", "node3"]}
if "model_barrier_epoch" in state:
    result["seconds_since_model_barrier"] = state.get("completed_epoch", time.time()) - state["model_barrier_epoch"]
log = root / method / "target/target.log"
if log.exists():
    lines = log.read_text(errors="replace").splitlines()
    result["http_responses"] = dict(collections.Counter(re.findall(r'"(?:GET|POST) ([^ ]+) HTTP/1.1" ([0-9]+)', "\n".join(lines))))
    result["http_responses"] = {" ".join(k): v for k, v in result["http_responses"].items()}
    errors = [line for line in lines if any(x in line for x in ["Traceback", "Error:", "error:", "OutOfMemory"])]
    result["errors"] = errors[-8:]
result["uploaded_nodes"] = [p.stem.split(".")[0] for p in (root / method).glob("node*.tar.gz")]
result["node_exit_status"] = {}
for archive_path in (root / method).glob("node*.tar.gz"):
    node = archive_path.name.split(".")[0]
    try:
        with tarfile.open(archive_path) as archive:
            result["node_exit_status"][node] = json.load(archive.extractfile(node + "/exit.json"))["exit_status"]
    except (tarfile.TarError, EOFError, KeyError, json.JSONDecodeError):
        pass
batch_log = root / method / "target/batches-formal.jsonl"
if batch_log.exists():
    batches = []
    for line in batch_log.read_text().splitlines():
        try:
            batches.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    requests = [row for batch in batches for row in batch["requests"]]
    result["grpc_requests_completed"] = len(requests)
    result["grpc_gamma_counts"] = dict(collections.Counter(row["gamma"] for row in requests))
    result["grpc_clients_seen"] = len({row["client_idx"] for row in requests})
    result["grpc_batches"] = len(batches)
print(json.dumps(result, ensure_ascii=False, indent=2))
with (root / "monitor.jsonl").open("a") as stream:
    stream.write(json.dumps(result) + "\n")
