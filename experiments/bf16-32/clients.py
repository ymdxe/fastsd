"""Launch all CPU Drafts on one host with frozen inputs and disjoint affinity."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
sys.path[:0] = [str(ROOT), str(RUNTIME)]
from common import trace_hash, valid_id, write_json
from experiment_hooks import rpc


def expand_cpu_list(text):
    result = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def read_numa_groups(allowed):
    allowed = set(allowed)
    groups = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")):
        selected = sorted(set(expand_cpu_list(path.read_text())) & allowed)
        if selected:
            groups[path.parent.name] = selected
    covered = set(cpu for cpus in groups.values() for cpu in cpus)
    if allowed - covered:
        groups["unmapped"] = sorted(allowed - covered)
    return groups


def allocate_affinity(allowed, numa_groups, workers, threads):
    """Round-robin NUMA-local blocks; only combine leftovers if necessary."""
    allowed = set(allowed)
    if workers <= 0 or threads <= 0 or len(allowed) < workers * threads:
        raise RuntimeError("Not enough allowed logical CPUs for disjoint worker affinities")
    pools = []
    seen = set()
    for name, cpus in sorted(numa_groups.items()):
        selected = sorted((set(cpus) & allowed) - seen)
        seen.update(selected)
        if selected:
            pools.append([name, selected])
    if allowed - seen:
        pools.append(["unmapped", sorted(allowed - seen)])
    assignments = []
    while len(assignments) < workers:
        made_progress = False
        for name, pool in pools:
            if len(pool) >= threads and len(assignments) < workers:
                assignments.append({"cpus": pool[:threads], "numa_nodes": [name]})
                del pool[:threads]
                made_progress = True
        if not made_progress:
            leftovers = [(cpu, name) for name, pool in pools for cpu in pool]
            while len(assignments) < workers:
                chunk, leftovers = leftovers[:threads], leftovers[threads:]
                if len(chunk) != threads:
                    raise RuntimeError("Affinity allocation exhausted available CPUs")
                assignments.append({"cpus": [cpu for cpu, name in chunk],
                    "numa_nodes": sorted({name for cpu, name in chunk})})
            break
    used = [cpu for item in assignments for cpu in item["cpus"]]
    if len(set(used)) != workers * threads or not set(used).issubset(allowed):
        raise RuntimeError("Overlapping or forbidden CPU allocation")
    return assignments


def build_worker_command(info, node, worker, inputs, worker_dir, affinity, server_url):
    cfg = info["nodes"][node]
    identity = f"{node}-w{worker}"
    argv = [cfg["python"], "-B", "-u", str(ROOT / "worker.py"),
        "--inputs", str(inputs), "--exp-name", str(worker_dir),
        "--server-url", server_url, "--method", info["method"],
        "--gamma", str(cfg["gamma"]), "--affinity", ",".join(map(str, affinity)),
        "--worker-id", str(worker)]
    env_patch = {
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_BIN": cfg["python"],
        "PYTHONPATH": str(ROOT) + os.pathsep + str(RUNTIME),
        "FASTSD_SUITE_ROOT": str(ROOT), "FASTSD_SUITE_WORKER": identity,
        "FASTSD_SUITE_RUN_ID": info["run_id"], "FASTSD_SUITE_BUNDLE_ID": info["bundle_id"],
        "FASTSD_SUITE_TRACE_SHA256": info["trace_sha256"], "NO_PROXY": "*", "no_proxy": "*",
        "ACCELERATE_USE_CPU": "true",
    }
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TORCH_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env_patch[key] = str(cfg["threads"])
    return argv, env_patch


def stop_owned_children(children):
    for identity, process, log in children:
        if process.poll() is None:
            try:
                if os.getpgid(process.pid) != process.pid:
                    raise RuntimeError("Child process is not leader of its dedicated group")
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for identity, process, log in children:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError("Refusing to kill an unexpected process group")
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


def run(options):
    from suite import verify_bundle
    bundle_id = verify_bundle()
    node = platform.node().split(".")[0]
    run_id = valid_id(options.run_id)
    info = rpc(options.server_url, "/experiment/info")
    if info["run_id"] != run_id or info["bundle_id"] != bundle_id:
        raise RuntimeError("Wrong Target run ID or bundle")
    if info["method"] not in ("sd", "fastsd"):
        raise RuntimeError("Unexpected experiment method")
    if node not in info["nodes"]:
        raise RuntimeError("This host has no configured workers")
    if trace_hash(info["trace_rows"]) != info["trace_sha256"]:
        raise RuntimeError("Target trace hash mismatch")
    cfg = info["nodes"][node]
    if int(cfg["threads"]) != 8 or info["dtype"] != "bfloat16":
        raise RuntimeError("This launcher requires eight-thread BF16 Drafts")
    expected_workers = [f"{node}-w{w}" for w in range(int(cfg["workers"]))]
    supplied_workers = [w for w in info["expected_workers"] if w.startswith(node + "-")]
    if sorted(expected_workers) != sorted(supplied_workers):
        raise RuntimeError("Configured and expected worker sets differ")
    health = rpc(options.server_url, "/experiment/health")
    if health.get("status") != "ok" or not health.get("model_ready"):
        raise RuntimeError("Target model is not ready: " + str(health))
    allowed = sorted(os.sched_getaffinity(0))
    numa_groups = read_numa_groups(allowed)
    assignments = allocate_affinity(allowed, numa_groups, int(cfg["workers"]), int(cfg["threads"]))
    output = Path(info["cpu_result_root"]) / run_id / node
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", info)
    write_json(output / "affinity.json", {"allowed_cpus": allowed, "numa_groups": numa_groups,
        "workers": {identity: assignments[w] for w, identity in enumerate(expected_workers)},
        "loadavg_before": list(os.getloadavg())})
    children = []
    commands = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt("launcher signal " + str(signum))
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    def save_processes():
        write_json(output / "processes.json", {"launcher_pid": os.getpid(),
            "workers": [{"worker": name, "pid": p.pid, "pgid": p.pid} for name, p, log in children]})
    started = time.monotonic()
    status = {"state": "starting", "run_id": run_id, "node": node, "started_epoch_s": time.time()}
    write_json(output / "launcher-status.json", status)
    try:
        for worker, identity in enumerate(expected_workers):
            sample_rows = [r for r in info["trace_rows"] if r["node"] == node and int(r["worker"]) == worker]
            sample_ids = [r["sample_id"] for r in sample_rows]
            if len(sample_rows) != int(info["tasks_per_worker"]) or len(set(sample_ids)) != len(sample_rows):
                raise RuntimeError("Missing or duplicate assigned requests for " + identity)
            worker_dir = output / f"worker{worker}"
            worker_dir.mkdir(exist_ok=False)
            inputs = worker_dir / "inputs.jsonl"
            with inputs.open("x", encoding="utf-8") as handle:
                for row in sample_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            command, patch = build_worker_command(info, node, worker, inputs, worker_dir,
                assignments[worker]["cpus"], options.server_url)
            env = os.environ.copy()
            env.update(patch)
            command_record = {"argv": command, "environment": patch, "cwd": str(RUNTIME),
                "affinity": assignments[worker], "input_requests": len(sample_rows)}
            write_json(worker_dir / "command.json", command_record)
            commands.append({"worker": identity, **command_record})
            write_json(output / "commands.json", commands)
            log = (worker_dir / "worker.log").open("x", encoding="utf-8")
            try:
                process = subprocess.Popen(command, cwd=RUNTIME, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            children.append((identity, process, log))
            save_processes()
        status["state"] = "running"
        write_json(output / "launcher-status.json", status)
        print(f"{node}: started {len(children)} workers; model load -> warmup -> common T0", flush=True)
        last_print = 0.0
        while any(process.poll() is None for identity, process, log in children):
            failed = [(name, p.returncode) for name, p, log in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError("Workers failed: " + str(failed))
            if time.monotonic() - started > options.timeout:
                raise TimeoutError("CPU launcher exceeded " + str(options.timeout) + " seconds")
            if time.monotonic() - last_print >= 15:
                print(json.dumps({"node": node, "running": [name for name, p, log in children if p.poll() is None],
                    "elapsed_s": round(time.monotonic() - started)}), flush=True)
                last_print = time.monotonic()
            time.sleep(1)
        failed = [(name, p.returncode) for name, p, log in children if p.returncode != 0]
        if failed:
            raise RuntimeError("Workers failed: " + str(failed))
        status.update(state="complete", workers=len(children))
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        try:
            rpc(options.server_url, "/experiment/abort",
                {"run_id": run_id, "reason": node + ": " + str(exc)}, timeout=5)
        except Exception:
            pass
        stop_owned_children(children)
        raise
    finally:
        for identity, process, log in children:
            log.close()
        status.update(finished_epoch_s=time.time(), elapsed_s=time.monotonic() - started)
        write_json(output / "launcher-status.json", status)
        write_json(output / "exit-status.json", {name: p.poll() for name, p, log in children})


def main(argv=None):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--server-url", default=config["server_url"])
    parser.add_argument("--timeout", type=float, default=1800)
    options = parser.parse_args(argv)
    if options.timeout <= 0:
        parser.error("--timeout must be positive")
    run(options)


if __name__ == "__main__":
    main()
