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
from common import metric_record_error, trace_hash, valid_id, write_json
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
    entrypoint = {"specedge":"specedge_worker.py", "target_only":"ar_worker.py"}.get(info["method"], "worker.py")
    argv = [cfg["python"], "-B", "-u", str(ROOT / entrypoint),
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
        "TORCH_NUM_INTEROP_THREADS": "1",
    }
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TORCH_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env_patch[key] = str(cfg["threads"])
    return argv, env_patch


def worker_progress(worker_dir, expected_ids, process=None):
    """Count complete, valid, unique JSONL request records; never infer success from exit alone."""
    completed = set()
    progress = {"completed_requests": 0, "metric_rows": 0, "invalid_rows": 0,
        "duplicate_rows": 0, "partial_lines": 0, "read_errors": []}
    for path in sorted(Path(worker_dir).glob("edge_metrics_proc*.jsonl")):
        try:
            with path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        progress["partial_lines"] += 1
                        continue
                    if not line.strip():
                        continue
                    progress["metric_rows"] += 1
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        progress["invalid_rows"] += 1
                        continue
                    if metric_record_error(record, expected_ids):
                        progress["invalid_rows"] += 1
                        continue
                    if record["task_id"] in completed:
                        progress["duplicate_rows"] += 1
                        continue
                    completed.add(record["task_id"])
        except OSError as exc:
            if len(progress["read_errors"]) < 4:
                progress["read_errors"].append(path.name + ": " + str(exc))
    code = process.poll() if process is not None else None
    progress.update(completed_requests=len(completed), process_returncode=code,
        state="not_started" if process is None else ("running" if code is None else ("exited" if code == 0 else "failed")))
    return progress


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
    if info["method"] not in ("sd", "fastsd", "specedge", "target_only"):
        raise RuntimeError("Unexpected experiment method")
    if node not in info["nodes"]:
        raise RuntimeError("This host has no configured workers")
    if trace_hash(info["trace_rows"]) != info["trace_sha256"]:
        raise RuntimeError("Target trace hash mismatch")
    cfg = info["nodes"][node]
    if int(cfg["threads"]) != 8 or info["dtype"] != "bfloat16":
        raise RuntimeError("This launcher requires eight-thread BF16 Drafts")
    if int(cfg["workers"]) != {"node1": 4, "node2": 4, "node3": 8}.get(node):
        raise RuntimeError("This launcher requires the 4/4/8 Draft topology")
    expected_workers = [f"{node}-w{w}" for w in range(int(cfg["workers"]))]
    supplied_workers = [w for w in info["expected_workers"] if w.startswith(node + "-")]
    if sorted(expected_workers) != sorted(supplied_workers):
        raise RuntimeError("Configured and expected worker sets differ")
    health = rpc(options.server_url, "/experiment/health")
    if health.get("status") != "ok" or not health.get("model_ready"):
        raise RuntimeError("Target model is not ready: " + str(health))
    original_allowed = set(os.sched_getaffinity(0))
    excluded = {22, 70} if node == "node2" else set()
    allowed = sorted(original_allowed - excluded)
    numa_groups = read_numa_groups(allowed)
    frozen = json.loads((ROOT / "affinity.json").read_text(encoding="utf-8"))[node]["workers"]
    assignments = []
    used = set()
    for w, entry in enumerate(frozen):
        cpus = entry['logical_cpus']
        numa = 'node' + str(entry['numa_node'])
        if (entry['worker'] != w or len(cpus) != 8 or len(entry['physical_cores']) != 4
                or not set(cpus).issubset(allowed) or used.intersection(cpus)
                or not set(cpus).issubset(numa_groups.get(numa, []))):
            raise RuntimeError('Frozen CPU affinity is not valid on this host: '+str(entry))
        used.update(cpus)
        assignments.append({'cpus':cpus, 'numa_nodes':[numa], 'physical_cores':entry['physical_cores']})
    if len(assignments) != int(cfg['workers']):
        raise RuntimeError('Frozen worker count mismatch')
    output = Path(cfg.get("result_root", info["cpu_result_root"])) / run_id / node
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", info)
    write_json(output / "affinity.json", {"allowed_cpus": allowed, "original_allowed_cpus": sorted(original_allowed),
        "excluded_cpus": sorted(excluded), "numa_groups": numa_groups,
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
    expected_ids = {identity: {row["sample_id"] for row in info["trace_rows"]
        if row["node"] == node and int(row["worker"]) == worker}
        for worker, identity in enumerate(expected_workers)}
    heartbeat_interval = 30.0
    def send_progress():
        processes = {name: process for name, process, log in children}
        samples = {identity: worker_progress(output / f"worker{worker}", expected_ids[identity], processes.get(identity))
            for worker, identity in enumerate(expected_workers)}
        status["progress"] = {"sampled_epoch_s": time.time(), "workers": samples,
            "completed_requests": sum(sample["completed_requests"] for sample in samples.values()),
            "expected_requests": sum(len(ids) for ids in expected_ids.values())}
        response = None
        try:
            response = rpc(options.server_url, "/experiment/heartbeat", {
                "run_id": run_id, "node": node, "bundle_id": bundle_id,
                "trace_sha256": info["trace_sha256"], "workers": samples}, timeout=5)
            status["heartbeat_last_success_epoch_s"] = time.time()
            status.pop("heartbeat_error", None)
        except Exception as exc:
            status["heartbeat_error"] = repr(exc)
            status["heartbeat_errors"] = status.get("heartbeat_errors", 0) + 1
        write_json(output / "launcher-status.json", status)
        print(json.dumps({"node": node, "state": status["state"],
            "elapsed_s": round(time.monotonic()-started), "progress": status["progress"],
            "heartbeat_error": status.get("heartbeat_error")}), flush=True)
        if response and response.get("aborted"):
            raise RuntimeError("Experiment aborted: " + str(response["aborted"]))
        return samples
    try:
        for worker, identity in enumerate(expected_workers):
            sample_rows = [r for r in info["trace_rows"] if r["node"] == node and int(r["worker"]) == worker]
            sample_ids = [r["sample_id"] for r in sample_rows]
            if len(sample_rows) != int(info["worker_request_counts"][identity]) or len(set(sample_ids)) != len(sample_rows):
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
            if time.monotonic() - last_print >= heartbeat_interval:
                send_progress()
                last_print = time.monotonic()
            time.sleep(1)
        failed = [(name, p.returncode) for name, p, log in children if p.returncode != 0]
        if failed:
            raise RuntimeError("Workers failed: " + str(failed))
        samples = send_progress()
        incomplete = [name for name, sample in samples.items()
            if sample["completed_requests"] != len(expected_ids[name]) or sample["invalid_rows"]
            or sample["duplicate_rows"] or sample["partial_lines"] or sample["read_errors"]]
        if incomplete:
            raise RuntimeError("Incomplete or invalid local request evidence: " + str(incomplete))
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
    parser.add_argument("--timeout", type=float, default=float(config.get("launcher_timeout", 86400)))
    options = parser.parse_args(argv)
    if options.timeout <= 0:
        parser.error("--timeout must be positive")
    run(options)


if __name__ == "__main__":
    main()
