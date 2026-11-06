"""Model-free reproduction of the archived prefetch planning/submit path.

This executes the exact archived nested functions with synthetic work queues.
It does not load a model, allocate CUDA memory, or test tensor transfers.
"""
from pathlib import Path
import ast
import collections
import hashlib
import importlib.util
import json
import queue
import statistics
import sys
import threading
import types

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "source/src/engine.py"
SCHEDULER = ROOT / "source/src/fastsd_scheduler.py"
spec = importlib.util.spec_from_file_location("archived_prefetch_scheduler", SCHEDULER)
scheduler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scheduler
spec.loader.exec_module(scheduler)
tree = ast.parse(ENGINE.read_text())


def exact_function(name, namespace):
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(candidates) == 1, name
    unit = ast.Module(body=[candidates[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(ENGINE), "exec"), namespace)
    return namespace[name]


def item(name, gamma, category="short"):
    return scheduler.WorkItem(
        work_id=name, proc_id=name, task_type="verify", category=category,
        request={"proc_id": name, "task_type": "verify", "gamma": gamma,
                 "prefix_len": 77, "current_time": 0.0},
        total_tokens=gamma, bridge_pending=1)


def context(items):
    queues = {kind: {cat: queue.Queue() for cat in ["short", "mid", "long"]} for kind in ["verify", "prefill"]}
    for work in items:
        queues[work.task_type][work.category].put(work)
    counters = {key: 0 for key in ["prefetch_plans", "prefetch_evictions", "prefetch_hits", "prefetch_misses",
                                  "prefetch_async_submitted", "prefetch_async_dropped"]}
    args = types.SimpleNamespace(token_budget=512, batch_size=8, min_prefill_chunk_tokens=16, prefill_chunk_quantum=128)
    namespace = {"self": types.SimpleNamespace(args=args), "os": types.SimpleNamespace(environ={}),
                 "plan_iteration": scheduler.plan_iteration, "scheduler_state": {"current_cycle": 0, "wrr_cursor": 0},
                 "accept_stats": {}, "time": types.SimpleNamespace(monotonic=lambda: 100.0),
                 "cache_lock": threading.RLock(), "preloaded_gpu_pids": set(),
                 "scheduler_metrics": counters, "queue": queue, "prefetch_tasks": queue.Queue(maxsize=1),
                 "task_queues": queues, "WorkItem": scheduler.WorkItem,
                 "_length_category": scheduler.length_category, "_compute_priority_score": scheduler.compute_priority_score,
                 "len_r1": 128, "len_r2": 512}
    namespace["_plan_queues"] = lambda: {kind: {cat: list(q.queue) for cat, q in group.items()} for kind, group in queues.items()}
    exact_function("prefetch_next_plan", namespace)
    exact_function("sort_task_queues", namespace)
    return namespace


def plan(c):
    return scheduler.plan_iteration(c["_plan_queues"](), token_budget=512, max_num_seqs=8,
                                    min_prefill_chunk_tokens=16, prefill_chunk_quantum=128, now=100.0)


def inspect_case(name, gammas):
    c = context([item(f"worker-{n}", gamma) for n, gamma in enumerate(gammas)])
    current = plan(c)
    c["prefetch_next_plan"](reserved_work_ids=current.selected_work_ids)
    predicted = sorted(c["prefetch_tasks"].get_nowait()) if not c["prefetch_tasks"].empty() else []
    return {"name": name, "queued_gammas": gammas, "current_selected": current.selected_work_ids,
            "current_forward_tokens": current.used_tokens, "current_slices": [vars(s) for s in current.verify_slices],
            "prefetch_targets": predicted, "counters": dict(c["scheduler_metrics"])}


cases = [inspect_case("one_gamma100", [100]), inspect_case("eight_gamma20", [20] * 8),
         inspect_case("four_gamma100_four_gamma20", [100] * 4 + [20] * 4),
         inspect_case("eight_gamma100_backlog", [100] * 8), inspect_case("all_sixteen_backlog", [100] * 8 + [20] * 8)]
assert all(not case["prefetch_targets"] for case in cases[:3])
assert all(case["prefetch_targets"] and case["counters"]["prefetch_async_submitted"] == 1 for case in cases[3:])

# An arrival in the multiprocessing ingress queue is invisible to the exact
# prefetch function, which only reads the scheduler's task_queues snapshot.
c = context([item("active", 100)])
active = plan(c)
ingress = queue.Queue()
incoming = item("arrived_during_compute", 20)
ingress.put(incoming)
c["prefetch_next_plan"](reserved_work_ids=active.selected_work_ids)
assert c["scheduler_metrics"]["prefetch_async_submitted"] == 0
invisible = {"ingress_size": ingress.qsize(), "submitted_before_ingress_drain": 0}
c["task_queues"]["verify"]["short"].put(ingress.get_nowait())
c["prefetch_next_plan"](reserved_work_ids=active.selected_work_ids)
invisible["targets_if_admitted_while_current_plan_reserved"] = sorted(c["prefetch_tasks"].get_nowait())
assert invisible["targets_if_admitted_while_current_plan_reserved"] == ["arrived_during_compute"]

# Reproduce the real sequence: arrivals wait until current execution returns,
# then become the NEXT current plan and are excluded from its lookahead again.
c = context([item("previous", 100)])
previous = plan(c)
scheduler.commit_admission_plan(previous, completed_work_ids=previous.selected_work_ids)
c["task_queues"]["verify"]["short"].put(item("late_arrival", 20))
c["sort_task_queues"]()
next_current = plan(c)
c["prefetch_next_plan"](reserved_work_ids=next_current.selected_work_ids)
invisible["targets_after_next_current_plan_reservation"] = sorted(c["prefetch_tasks"].get_nowait()) if not c["prefetch_tasks"].empty() else []
assert next_current.verify_proc_ids == ["late_arrival"] and not invisible["targets_after_next_current_plan_reservation"]

# Finished entries remain in the real queue; the outer any(not q.empty())
# condition remains true even though the planner has no executable work.
finished = item("finished", 20)
finished.finished = True
finished.state = "finished"
finished.cursor = 20
c = context([finished])
for _ in range(3):
    c["sort_task_queues"]()
    assert any(not q.empty() for group in c["task_queues"].values() for q in group.values())
    current = plan(c)
    assert not current.selected_work_ids
    c["prefetch_next_plan"](reserved_work_ids=current.selected_work_ids)
stale = {"queue_entries_after_three_empty_iterations": sum(q.qsize() for group in c["task_queues"].values() for q in group.values()),
         "prefetch_plans": c["scheduler_metrics"]["prefetch_plans"],
         "submitted": c["scheduler_metrics"]["prefetch_async_submitted"]}
assert stale == {"queue_entries_after_three_empty_iterations": 1, "prefetch_plans": 3, "submitted": 0}

# Group observed responses by identical server verify_ms and close corrected
# epoch receive times. This estimates actual compute batches; it is not a
# recording of internal queue contents or exact compute start/end timestamps.
responses = []
for node in ["node1", "node3"]:
    folder = ROOT / "fastsd" / node / "run"
    for path in folder.glob("rpc-*.jsonl"):
        clock = json.loads(path.with_name(path.name.replace("rpc-", "clock-").replace(".jsonl", ".json")).read_text())
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["endpoint"] != "/verify":
                continue
            row = dict(row, node=node, worker=path.stem,
                       approx_start_cloud_epoch=row["started_epoch"] + clock["clock_offset_ns"] / 1e9)
            row["approx_end_cloud_epoch"] = row["approx_start_cloud_epoch"] + row["http_ms"] / 1000
            responses.append(row)
batches = []
by_duration = collections.defaultdict(list)
for response in responses:
    by_duration[response["verify_ms"]].append(response)
for duration, rows in by_duration.items():
    group = []
    for row in sorted(rows, key=lambda r: r["approx_end_cloud_epoch"]):
        if group and row["approx_end_cloud_epoch"] - group[0]["approx_end_cloud_epoch"] > .2:
            batches.append(group)
            group = []
        group.append(row)
    if group:
        batches.append(group)
events = [(r["approx_start_cloud_epoch"], 1) for r in responses] + [(r["approx_end_cloud_epoch"], -1) for r in responses]
active = peak = 0
for _, delta in sorted(events):
    active += delta
    peak = max(peak, active)
observed = {"formal_verify_rpcs": len(responses), "estimated_compute_batches": len(batches),
            "estimated_batch_size_counts": dict(collections.Counter(len(batch) for batch in batches)),
            "estimated_largest_batch": max(map(len, batches)), "peak_outstanding_verify_rpcs": peak,
            "mean_compute_s": statistics.mean(r["verify_ms"] for r in responses) / 1000,
            "mean_http_s": statistics.mean(r["http_ms"] for r in responses) / 1000,
            "note": "Batch grouping uses equal reported verify_ms plus corrected epoch finish times within 200ms. In-flight HTTP counts include execution, ingress waiting and transport; they are not scheduler-queue lengths."}

counters = json.loads((ROOT / "fastsd/target/scheduler_metrics.json").read_text())["fastsd_scheduler"]
report = {"archived_counters": counters,
          "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ENGINE, SCHEDULER]},
          "current_source_matches_archive": all((ROOT.parents[1] / "src" / name).read_bytes() == (ROOT / "source/src" / name).read_bytes() for name in ["engine.py", "fastsd_scheduler.py"]),
          "submit_path_cases": cases, "ingress_visibility": invisible, "finished_queue_reproduction": stale,
          "observed_rpc_evidence": observed,
          "limitations": ["No model or CUDA allocation; this checks scheduling/submission, not GPU cache-transfer correctness or overlap speedup.",
                          "The original experiment did not log per-iteration live queue snapshots, so its exact queue trajectory cannot be replayed.",
                          "Synthetic backlogged workloads are positive controls, not measurements from the completed experiment."]}
(ROOT / "prefetch_diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"current_source_matches_archive": report["current_source_matches_archive"],
                  "cases": [{"name": c["name"], "current_count": len(c["current_selected"]), "prefetch_targets": c["prefetch_targets"]} for c in cases],
                  "ingress_visibility": invisible, "finished_queue_reproduction": stale, "observed_rpc_evidence": observed}, ensure_ascii=False, indent=2))
