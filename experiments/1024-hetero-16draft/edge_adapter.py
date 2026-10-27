"""CPU placement, global rendezvous and tracing around native inference."""
from pathlib import Path
import ctypes
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import torch
import edge.edge as edge_module

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOKEN = (ROOT / ".coordinator-token").read_text().strip()


def control(endpoint, **extra):
    query = {"run": os.environ["EXP_RUN"], "worker": os.environ.get("EXP_WORKER", ""), "expected": os.environ["EXP_EXPECTED"]}
    query.update(extra)
    request = urllib.request.Request("http://127.0.0.1:28624/" + endpoint + "?" + urllib.parse.urlencode(query), headers={"Authorization": "Bearer " + TOKEN})
    return json.loads(OPENER.open(request, timeout=30).read())


def barrier(endpoint, field="release"):
    start = time.monotonic()
    while not control(endpoint).get(field):
        if time.monotonic() - start > 1200:
            raise TimeoutError(endpoint + " barrier did not complete")
        time.sleep(0.15)


def bind(cpu, node):
    os.sched_setaffinity(0, {cpu})
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    lib.numa_allocate_nodemask.restype = ctypes.c_void_p
    lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.numa_bitmask_setbit.restype = ctypes.c_void_p
    lib.numa_set_membind.argtypes = [ctypes.c_void_p]
    lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    mask = lib.numa_allocate_nodemask()
    if not mask:
        raise RuntimeError("NUMA mask allocation failed")
    try:
        lib.numa_bitmask_setbit(mask, node)
        lib.numa_set_membind(mask)
    finally:
        lib.numa_bitmask_free(mask)


class TracedClient(edge_module.EdgeClient):
    def synchronize_clock(self, samples=5):
        barrier("clock_acquire", "acquired")
        try:
            info = super().synchronize_clock(samples)
            Path(os.environ["EXP_RPC_LOG"].replace("rpc-", "clock-").replace(".jsonl", ".json")).write_text(json.dumps(info))
        finally:
            control("clock_release")
        barrier("clock_done")
        return info

    def _post(self, path, payload):
        start = time.monotonic()
        row = {"endpoint": path, "started_monotonic": start, "started_epoch": time.time(),
               "pid": os.getpid(), "node": os.environ["EXP_NODE"], "task_id": payload.get("task_id"),
               "gamma": payload.get("gamma"), "tail_only": payload.get("tail_only")}
        try:
            result = super()._post(path, payload)
            row.update(ok=True, cloud_total_ms=result.get("cloud_total_ms"), verify_ms=result.get("verify_ms"), suggested_gamma=result.get("suggested_gamma"))
            return result
        except Exception as exc:
            row.update(ok=False, error=repr(exc), response_body=getattr(getattr(exc, "response", None), "text", None))
            raise
        finally:
            row.update(finished_monotonic=time.monotonic(), http_ms=(time.monotonic() - start) * 1000)
            with open(os.environ["EXP_RPC_LOG"], "a") as stream:
                stream.write(json.dumps(row) + "\n")

    def _attach_pull(self, response, edge_receive_time_ns):
        if self.timing_priority:
            return super()._attach_pull(response, edge_receive_time_ns)
        response.setdefault("pull_s", 0.0)
        return response


edge_module.EdgeClient = TracedClient


class BoundRunner(edge_module.EdgeRunner):
    def _encode_input_ids(self, tokenizer, input_text):
        value = tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        return value if torch.is_tensor(value) else torch.tensor([value], dtype=torch.long)

    def run_draft_process_http(self, tokenizer, proc_id, arrival_barrier=None, arrival_start_time=None):
        self.worker_id = proc_id
        placement = json.loads((ROOT / (os.environ["EXP_NODE"] + "-worker-map.json")).read_text())[proc_id]
        bind(placement["cpu"], placement["numa"])
        os.environ["EXP_WORKER"] = os.environ["EXP_NODE"] + "-" + str(proc_id)
        os.environ["EXP_RPC_LOG"] = str(Path(self.args.exp_name) / f"rpc-{proc_id:02d}.jsonl")
        success = False
        try:
            result = super().run_draft_process_http(tokenizer, proc_id, arrival_barrier, arrival_start_time)
            success = True
            return result
        finally:
            row = {"worker_id": proc_id, "node": os.environ["EXP_NODE"], "success": success, "completed_monotonic": time.monotonic(), "completed_epoch": time.time()}
            (Path(self.args.exp_name) / f"completion-{proc_id:02d}.json").write_text(json.dumps(row))
            try:
                control("done", success=int(success))
            except Exception as exc:
                print("COMPLETION_REPORT_ERROR", repr(exc), flush=True)

    def _load_draft_model(self, model_path, device):
        start = time.monotonic()
        model = super()._load_draft_model(model_path, device)
        assert {p.dtype for p in model.parameters()} == {torch.float16}
        row = {"worker_id": self.worker_id, "pid": os.getpid(), "node": os.environ["EXP_NODE"],
               "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads(),
               "dtype": "float16", "device": "cpu", "load_seconds": time.monotonic() - start,
               "numa_policies": sorted({line.split()[1] for line in Path("/proc/self/numa_maps").read_text().splitlines() if len(line.split()) > 1})}
        dest = Path(self.args.exp_name) / f"worker-{self.worker_id:02d}.json"
        dest.write_text(json.dumps(row))
        print("MODEL_READY", os.environ["EXP_WORKER"], flush=True)
        barrier("model_ready")
        row.update(barrier_release_monotonic=time.monotonic(), barrier_release_epoch=time.time())
        dest.write_text(json.dumps(row))
        return model


if __name__ == "__main__":
    args = edge_module.parse_edge_arguments()
    Path(args.exp_name).mkdir(parents=True, exist_ok=True)
    args.max_tasks_per_draft = 1
    BoundRunner(args).eval()
