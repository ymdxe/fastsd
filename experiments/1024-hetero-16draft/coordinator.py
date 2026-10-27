"""Experiment-only rendezvous, transfer, and one-clock completion recording."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import hmac
import secrets
import threading
import time

ROOT = Path(__file__).resolve().parent
LOCK = threading.RLock()
STATES = {}
RUNS = {"probe", "standard", "fastsd", "smoke-standard", "smoke-fastsd", "specedge", "smoke-specedge"}
TOKEN_PATH = ROOT / ".coordinator-token"
if not TOKEN_PATH.exists():
    import os
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(secrets.token_hex(32))
TOKEN = TOKEN_PATH.read_text().strip()


def state(run):
    return STATES.setdefault(run, {"ready": {}, "clock_done": {}, "done": {}, "clock_owner": None})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if self.client_address[0] not in {"10.66.0.2", "10.66.0.3", "10.66.0.5"}:
            self.reply({"error": "client not allowed"}, 403)
            return False
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN):
            self.reply({"error": "authentication required"}, 401)
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        p = urlparse(self.path)
        if p.path == "/bundle.tar.gz":
            data = (ROOT / "bundle.tar.gz").read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        args = parse_qs(p.query)
        run = args.get("run", ["probe"])[0]
        if run not in RUNS:
            return self.reply({"error": "invalid run"}, 400)
        worker = args.get("worker", [""])[0]
        expected = int(args.get("expected", ["16"])[0])
        if not 1 <= expected <= 16 or len(worker) > 32:
            return self.reply({"error": "invalid worker count"}, 400)
        with LOCK:
            s = state(run)
            now = time.time()
            if p.path == "/health":
                return self.reply({"status": "ok", "time": now})
            if p.path == "/model_ready":
                s["ready"].setdefault(worker, now)
                if len(s["ready"]) >= expected:
                    s.setdefault("model_barrier_epoch", now + 2)
                release = s.get("model_barrier_epoch")
                return self.reply({"release": release is not None and now >= release, "count": len(s["ready"])})
            if p.path == "/clock_acquire":
                if s["clock_owner"] in (None, worker):
                    s["clock_owner"] = worker
                    return self.reply({"acquired": True})
                return self.reply({"acquired": False})
            if p.path == "/clock_release":
                if s["clock_owner"] == worker:
                    s["clock_owner"] = None
                return self.reply({"ok": True})
            if p.path == "/clock_done":
                s["clock_done"].setdefault(worker, now)
                if len(s["clock_done"]) >= expected:
                    s.setdefault("request_barrier_epoch", now + 2)
                release = s.get("request_barrier_epoch")
                return self.reply({"release": release is not None and now >= release})
            if p.path == "/done":
                s["done"].setdefault(worker, {"epoch": now, "success": args.get("success", ["0"])[0] == "1"})
                if len(s["done"]) >= expected:
                    s["completed_epoch"] = max(x["epoch"] for x in s["done"].values())
                    s["wall_seconds"] = s["completed_epoch"] - s["model_barrier_epoch"]
                dest = ROOT / run
                dest.mkdir(exist_ok=True)
                (dest / "timing.json").write_text(json.dumps(s, indent=2))
                return self.reply({"ok": True})
            if p.path == "/status":
                return self.reply(s)
            return self.reply({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        if not self.authorized():
            return
        p = urlparse(self.path)
        args = parse_qs(p.query)
        run = args.get("run", [""])[0]
        node = args.get("node", [""])[0]
        if p.path != "/upload" or run not in RUNS - {"probe"} or node not in {"node1", "node3"}:
            return self.reply({"error": "invalid upload"}, 400)
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size < 50_000_000:
            return self.reply({"error": "invalid size"}, 400)
        dest = ROOT / run
        dest.mkdir(exist_ok=True)
        (dest / (node + ".tar.gz")).write_bytes(self.rfile.read(size))
        self.reply({"ok": True, "bytes": size})


if __name__ == "__main__":
    import os
    (ROOT / "coordinator.pid").write_text(str(os.getpid()))
    ThreadingHTTPServer(("10.66.0.5", 18624), Handler).serve_forever()
