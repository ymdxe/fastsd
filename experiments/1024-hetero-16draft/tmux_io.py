"""Local control helper: CPU commands in session 6; GPU commands in session 8."""
import argparse
import base64
from pathlib import Path
import shlex
import subprocess
import zlib
import hashlib
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["send", "capture", "gpu", "netprobe", "forward", "unforward", "bootstrap", "status"])
parser.add_argument("target")
parser.add_argument("text", nargs="?", default="")
args = parser.parse_args()
if args.action in {"send", "bootstrap", "forward", "unforward", "netprobe"}:
    assert args.target in {"6:0.0", "6:0.1"}
    mode = subprocess.run(["tmux", "display-message", "-p", "-t", args.target, "#{pane_in_mode}"], capture_output=True, text=True, check=True).stdout.strip()
    if mode == "1":
        subprocess.run(["tmux", "send-keys", "-t", args.target, "-X", "cancel"], check=True)
if args.action == "bootstrap":
    assert args.target in {"6:0.0", "6:0.1"}
    root = Path(__file__).parent
    token = (root / ".coordinator-token").read_text().strip()
    digest = hashlib.sha256((root / "bundle.tar.gz").read_bytes()).hexdigest()
    remote = '''import urllib.request, pathlib, tarfile, io, hashlib, os
root=pathlib.Path('/home/zhangh/workspace/fastsd/experiments/1024-hetero-16draft')
root.mkdir(parents=True,exist_ok=True)
token=TOKEN
request=urllib.request.Request('http://127.0.0.1:28624/bundle.tar.gz',headers={'Authorization':'Bearer '+token})
data=urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=120).read()
assert hashlib.sha256(data).hexdigest()==DIGEST
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive.getmembers():
  p=pathlib.PurePosixPath(member.name)
  assert not p.is_absolute() and '..' not in p.parts and (member.isfile() or member.isdir())
 archive.extractall(root)
fd=os.open(root/'.coordinator-token',os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
with os.fdopen(fd,'w') as stream:stream.write(token)
print('EXP1024_BOOTSTRAP_OK',len(data),flush=True)
'''.replace('TOKEN', repr(token)).replace('DIGEST', repr(digest))
    packed = base64.b64encode(zlib.compress(remote.encode())).decode()
    python = "/home/zhangh/venvs/fastsd_cpu/bin/python" if args.target == "6:0.0" else "python"
    command = python + " -c " + shlex.quote("import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(packed) + ")))")
    subprocess.run(["tmux", "send-keys", "-t", args.target, "-l", command], check=True)
    subprocess.run(["tmux", "send-keys", "-t", args.target, "Enter"], check=True)
elif args.action in {"forward", "unforward"}:
    assert args.target in {"6:0.0", "6:0.1"}
    rules = ["-R 127.0.0.1:28624:10.66.0.5:18624", "-R 127.0.0.1:28625:127.0.0.1:18625"] if args.action == "forward" else ["-KR 127.0.0.1:28624", "-KR 127.0.0.1:28625"]
    for rule in rules:
        subprocess.run(["tmux", "send-keys", "-t", args.target, "Enter"], check=True)
        subprocess.run(["tmux", "send-keys", "-t", args.target, "-l", "~C"], check=True)
        subprocess.run(["tmux", "send-keys", "-t", args.target, "-l", " " + rule], check=True)
        subprocess.run(["tmux", "send-keys", "-t", args.target, "Enter"], check=True)
elif args.action == "netprobe":
    assert args.target in {"6:0.0", "6:0.1"}
    token = (Path(__file__).parent / ".coordinator-token").read_text().strip()
    remote = "import urllib.request,os; print('EXP1024_NET_BEGIN',os.getloadavg(),flush=True); request=urllib.request.Request('http://127.0.0.1:28624/health',headers={'Authorization': 'Bearer ' + " + repr(token) + "}); print(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=8).read().decode()); print('EXP1024_NET_END')"
    python = "/home/zhangh/venvs/fastsd_cpu/bin/python" if args.target == "6:0.0" else "python"
    command = python + " -c " + shlex.quote(remote)
    subprocess.run(["tmux", "send-keys", "-t", args.target, "-l", command], check=True)
    subprocess.run(["tmux", "send-keys", "-t", args.target, "Enter"], check=True)
elif args.action == "send":
    assert args.target in {"6:0.0", "6:0.1"}
    subprocess.run(["tmux", "send-keys", "-t", args.target, "-l", args.text], check=True)
    subprocess.run(["tmux", "send-keys", "-t", args.target, "Enter"], check=True)
elif args.action == "capture":
    assert args.target in {"6:0.0", "6:0.1", "8:0.0", "8:0.1"}
    output = subprocess.run(["tmux", "capture-pane", "-p", "-J", "-t", args.target, "-S", "-8"], check=True, capture_output=True, text=True).stdout
    token = (Path(__file__).parent / ".coordinator-token").read_text().strip()
    print("\n".join(line[:500] for line in output.replace(token, "[REDACTED]").splitlines()[-14:]))
elif args.action == "status":
    assert args.target in {"standard", "fastsd", "specedge", "smoke-standard", "smoke-fastsd", "smoke-specedge"}
    root = Path(__file__).resolve().parent
    output = Path("/tmp/fastsd-1024-live-monitor.json")
    started = time.time()
    command = "python3 " + shlex.quote(str(root / "status.py")) + " " + shlex.quote(args.target) + " > " + shlex.quote(str(output)) + " 2>&1"
    subprocess.run(["tmux", "run-shell", "-t", "8:0.1", command], check=True)
    while True:
        try:
            if output.stat().st_mtime >= started:
                data = json.loads(output.read_text())
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.time() - started > 15:
            raise RuntimeError(output.read_text() if output.exists() else "No status output")
        time.sleep(.1)
    brief = {"method": args.target, "done": data["done"], "nodes": data["done_by_node"],
             "elapsed_minutes": round(data.get("seconds_since_model_barrier", 0) / 60, 1), "errors": data.get("errors", [])}
    brief["worker_success"] = data["worker_success"]
    brief["node_exit_status"] = data.get("node_exit_status", {})
    if args.target != "specedge":
        brief["verify_http_success"] = data.get("http_responses", {}).get("/verify 200", 0)
    else:
        for key in ["grpc_requests_completed", "grpc_gamma_counts", "grpc_clients_seen"]:
            brief[key] = data.get(key, 0)
    for name in ["native", "specedge"]:
        path = root / (name + "_status.json")
        if path.exists():
            brief[name + "_stage"] = json.loads(path.read_text())["stage"]
    print(json.dumps(brief, ensure_ascii=False))
else:
    assert args.target in {"8:0.0", "8:0.1"}
    subprocess.run(["tmux", "run-shell", "-t", args.target, args.text], check=True)
