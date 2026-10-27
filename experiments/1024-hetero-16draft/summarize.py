#!/usr/bin/env python3
"""Summarize archived heterogeneous CPU drafts without importing the ML stack.

Layout: ROOT/{standard,fastsd,specedge}/{node1,node3}/run/*.json[l].
Optional ROOT/METHOD/timing.json must contain an explicit coordinator-measured
``wall_seconds``; cross-host monotonic timestamps are never subtracted here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


METHODS = ("standard", "fastsd", "specedge")
GAMMAS = {"node1": 100, "node3": 20}


def stats(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}

    def quantile(q):
        position = (len(values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"count": len(values), "mean": statistics.mean(values), "p50": quantile(.5),
            "p95": quantile(.95), "min": values[0], "max": values[-1]}


def read_records(folder, pattern, issues):
    records = []
    for path in sorted(folder.glob(pattern)):
        try:
            text = path.read_text()
            data = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
            for record in data:
                record = dict(record)
                record["_source"] = str(path)
                records.append(record)
        except (ValueError, OSError, TypeError) as exc:
            issues.append(f"Cannot read {path}: {exc}")
    return records


def numeric(records, key, scale=1):
    return stats(r.get(key) / scale if isinstance(r.get(key), (int, float)) else None for r in records)


def sum_field(records, key):
    return sum(r.get(key, 0) or 0 for r in records)


def host_load_summary(progress):
    """Each value is a sampled whole-host load average, never a background estimate."""
    values = []
    epochs = []
    for record in progress:
        loadavg = record.get("loadavg")
        if isinstance(loadavg, (list, tuple)) and loadavg and isinstance(loadavg[0], (int, float)):
            if math.isfinite(loadavg[0]):
                values.append(loadavg[0])
        epoch = record.get("epoch")
        if isinstance(epoch, (int, float)) and math.isfinite(epoch):
            epochs.append(epoch)
    result = stats(values)
    result.update(snapshot_count=len(progress), invalid_load_snapshot_count=len(progress) - len(values),
                  first_snapshot_epoch=min(epochs) if epochs else None,
                  last_snapshot_epoch=max(epochs) if epochs else None)
    return result


def summarize_records(records, rpcs, completions, workers, expected):
    verify = [r for r in rpcs if r.get("endpoint") == "/verify"]
    prefill = [r for r in rpcs if r.get("endpoint") == "/prefill"]
    accepted = sum_field(records, "accepted_total")
    drafted = sum_field(records, "drafted_total")
    tokens = sum_field(records, "generated_tokens")
    rpc_errors = [r for r in rpcs if r.get("ok") is False]
    return {
        "expected_requests": expected,
        "completed_requests": len(records),
        "missing_requests": max(0, expected - len(records)),
        "worker_records": len(workers),
        "completion_records": len(completions),
        "failed_workers": sum(r.get("success") is False for r in completions),
        "successful_workers": sum(r.get("success") is True for r in completions),
        "generated_tokens": tokens,
        "generated_tokens_per_request": numeric(records, "generated_tokens"),
        "ttft_seconds": numeric(records, "ttft_ms", 1000),
        "tpot_ms": numeric([r for r in records if r.get("generated_tokens", 0) > 1], "tpot_ms"),
        "request_e2e_seconds": numeric(records, "request_e2e_ms", 1000),
        "first_draft_seconds": numeric(records, "first_draft_ms", 1000),
        "decode_ttft_seconds": numeric(records, "decode_ttft_ms", 1000),
        "accepted_draft_tokens": accepted,
        "drafted_tokens": drafted,
        "acceptance_rate_weighted": accepted / drafted if drafted else None,
        "verify_rounds_per_request": numeric(records, "rounds"),
        "reuse_hit_rounds": sum_field(records, "reuse_hit_rounds"),
        "reuse_miss_rounds": sum_field(records, "reuse_miss_rounds"),
        "reuse_skip_not_full_accept_rounds": sum_field(records, "reuse_skip_not_full_accept_rounds"),
        "rpc_count": len(rpcs),
        "rpc_counts": dict(Counter(r.get("endpoint", "unknown") for r in rpcs)),
        "rpc_protocols": dict(Counter(r.get("protocol", "http" if "http_ms" in r else "unknown") for r in rpcs)),
        "rpc_error_count": len(rpc_errors),
        "rpc_errors": [{k: r.get(k) for k in ("_source", "endpoint", "error", "response_body")} for r in rpc_errors],
        "observed_verify_gammas": dict(Counter(str(r.get("gamma")) for r in verify)),
        "verify_http_ms": numeric(verify, "http_ms"),
        "verify_rpc_ms": stats(r.get("rpc_ms", r.get("http_ms")) for r in verify),
        "verify_cloud_total_ms": numeric(verify, "cloud_total_ms"),
        "verify_compute_ms": numeric(verify, "verify_ms"),
        "prefill_http_ms": numeric(prefill, "http_ms"),
        "application_request_bytes": sum_field(records, "request_bytes"),
        "application_response_bytes": sum_field(records, "response_bytes"),
    }


def check_node(method, node, records, rpcs, completions, expected, max_tokens):
    issues = []
    if len(records) != expected:
        issues.append(f"Expected {expected} requests, found {len(records)}")
    identifiers = [str(r.get("task_id")) for r in records]
    if len(set(identifiers)) != len(identifiers):
        issues.append("Duplicate task_id values within this node")
    if len(completions) != expected or not all(r.get("success") is True for r in completions):
        issues.append("Worker completion records are missing or include failures")
    for r in records:
        label = str(r.get("task_id"))
        n = r.get("generated_tokens")
        if not isinstance(n, int) or not 0 < n <= max_tokens or len(r.get("output_token_ids", [])) != n:
            issues.append(f"{label}: invalid output token count")
        e2e, ttft, tpot = (r.get(k) for k in ("request_e2e_ms", "ttft_ms", "tpot_ms"))
        if not isinstance(e2e, (int, float)) or not isinstance(ttft, (int, float)) or not 0 <= ttft <= e2e:
            issues.append(f"{label}: invalid request E2E/TTFT")
        elif isinstance(n, int) and n > 1 and (not isinstance(tpot, (int, float)) or not math.isclose(tpot, (e2e - ttft) / (n - 1), rel_tol=1e-7, abs_tol=1e-4)):
            issues.append(f"{label}: native TPOT does not match (E2E-TTFT)/(tokens-1)")
        if method in {"standard", "fastsd"}:
            fast = method == "fastsd"
            if r.get("server_sched_mode") != ("fastsd" if fast else "vanilla"):
                issues.append(f"{label}: unexpected server_sched_mode")
            for flag in ("enable_pipeline", "enable_proactive_draft", "enable_latency_priority"):
                if r.get(flag) is not fast:
                    issues.append(f"{label}: unexpected {flag}")
        elif method == "specedge":
            if r.get("method") != "specedge_cpu_adapted_single_chain":
                issues.append(f"{label}: missing or unexpected SpecEdge implementation label")
            if r.get("gamma") != GAMMAS[node]:
                issues.append(f"{label}: unexpected SpecEdge gamma")
    verify = [r for r in rpcs if r.get("endpoint") == "/verify"]
    if len(verify) != sum_field(records, "rounds"):
        issues.append("Verify RPC count differs from completed-request round total")
    if any(r.get("gamma") != GAMMAS[node] for r in verify):
        issues.append(f"Verify RPC gamma differs from requested {GAMMAS[node]} (or was not recorded)")
    if any(r.get("suggested_gamma") not in (None, GAMMAS[node]) for r in verify):
        issues.append("Server suggested a gamma that differs from this node's fixed gamma")
    if method == "specedge":
        expected_chain = {"protocol": "grpc", "candidate_tokens": GAMMAS[node],
                          "input_tokens_including_root": GAMMAS[node] + 1,
                          "chain_depth": GAMMAS[node], "max_children_per_parent": 1}
        for field, expected_value in expected_chain.items():
            if any(r.get(field) != expected_value for r in verify):
                issues.append(f"SpecEdge verification trace has unexpected {field}; expected {expected_value!r}")
    if any(r.get("ok") is False for r in rpcs):
        issues.append("RPC errors were recorded")
    return issues


def compare_outputs(standard, candidate, method):
    a = {(r["node"], str(r["task_id"])): r for r in standard}
    b = {(r["node"], str(r["task_id"])): r for r in candidate}
    pairs = sorted(a.keys() & b.keys())
    mismatches = []
    for key in pairs:
        left, right = a[key]["output_token_ids"], b[key]["output_token_ids"]
        if left != right:
            common = min(len(left), len(right))
            first = next((i for i in range(common) if left[i] != right[i]), common)
            mismatches.append({"node": key[0], "task_id": key[1], "first_different_token_index": first,
                               "standard_tokens": len(left), "candidate_tokens": len(right),
                               "standard_source": a[key]["_source"], "candidate_source": b[key]["_source"]})
    return {
        "baseline_method": "standard", "candidate_method": method,
        "paired_requests": len(pairs), "identical_output_requests": len(pairs) - len(mismatches),
        "mismatched_outputs": mismatches,
        "only_standard": [list(k) for k in sorted(a.keys() - b.keys())],
        "only_candidate": [list(k) for k in sorted(b.keys() - a.keys())],
        "paired_deltas_candidate_minus_standard": {
            metric: stats(b[k].get(metric) - a[k].get(metric) for k in pairs
                          if isinstance(a[k].get(metric), (int, float)) and isinstance(b[k].get(metric), (int, float))
                          and (metric != "tpot_ms" or min(a[k].get("generated_tokens", 0), b[k].get("generated_tokens", 0)) > 1))
            for metric in ("ttft_ms", "tpot_ms", "request_e2e_ms")
        },
    }


def format_number(value):
    return "NA" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-per-node", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="Methods to summarize; default includes all three")
    parser.add_argument("--require-complete", action="store_true", help="Write reports, then exit 2 if a run is incomplete or invalid")
    args = parser.parse_args()
    root = args.root.resolve()
    report = {"schema_version": 1, "root": str(root), "gamma_by_node": GAMMAS,
              "max_new_tokens": args.max_new_tokens, "methods": {}, "issues": [],
              "mechanism_observations": []}
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report["configuration"] = {key: manifest.get(key) for key in (
        "draft_model", "target_model", "dtype", "physical_gpu", "target_weight_gpu_budget_GiB",
        "target_cpu_offload", "kv_batch_mode", "batch_size", "token_budget", "dataset",
        "threads_per_draft", "pipeline_gamma_adapt", "transport", "source_sha")}
    all_requests = {}
    flat_rows = []
    methods = list(dict.fromkeys(args.methods))
    for method in methods:
        all_requests[method] = []
        all_rpcs, all_completions, all_workers = [], [], []
        nodes = {}
        for node, gamma in GAMMAS.items():
            folder = root / method / node / "run"
            issues = []
            records = read_records(folder, "edge_metrics_proc*.jsonl", issues)
            rpcs = read_records(folder, "rpc-*.jsonl", issues)
            completions = read_records(folder, "completion-*.json", issues)
            workers = read_records(folder, "worker-*.json", issues)
            progress = read_records(folder.parent, "progress.jsonl", issues)
            issues += check_node(method, node, records, rpcs, completions, args.expected_per_node, args.max_new_tokens)
            if len(workers) != args.expected_per_node:
                issues.append(f"Expected {args.expected_per_node} worker records, found {len(workers)}")
            summary = summarize_records(records, rpcs, completions, workers, args.expected_per_node)
            summary["host_load1"] = host_load_summary(progress)
            summary.update(gamma=gamma, issues=issues, status="completed" if not issues else "incomplete_or_invalid")
            nodes[node] = summary
            report["issues"].extend(f"{method}/{node}: {issue}" for issue in issues)
            for r in records:
                r["node"] = node
            all_requests[method].extend(records)
            all_rpcs.extend(rpcs)
            all_completions.extend(completions)
            all_workers.extend(workers)
        overall = summarize_records(all_requests[method], all_rpcs, all_completions, all_workers, args.expected_per_node * len(GAMMAS))
        overall.update(wall_seconds=None, aggregate_tokens_per_second=None)
        timing_path = root / method / "timing.json"
        if timing_path.exists():
            try:
                timing = json.loads(timing_path.read_text())
                wall = timing.get("wall_seconds")
                if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
                    raise ValueError("wall_seconds must be a finite positive coordinator duration")
                overall.update(wall_seconds=wall, aggregate_tokens_per_second=overall["generated_tokens"] / wall,
                               timing_definition=timing.get("definition", "Explicit coordinator-measured window; see timing.json"))
            except (ValueError, OSError) as exc:
                report["issues"].append(f"{method}/timing.json: {exc}")
        scheduler_path = root / method / "target" / "scheduler_metrics.json"
        native = method in {"standard", "fastsd"}
        scheduler = {
            "available": False, "source": str(scheduler_path), "metrics": None,
            "scope": "target_process_lifetime_including_smoke_and_idle_polling" if native else "formal_phase_after_statistics_and_KV_reset",
            "scope_note": (
                "Native Target counters span its process lifetime, including a separate 4-token smoke and idle polling. Formal TTFT/TPOT/E2E and RPC counts use only the archived formal workload. Prefill slices can exceed request counts because of chunking. No inferred smoke plan count is subtracted."
                if native else
                "SpecEdge resets statistics and KV at the smoke-to-formal phase transition; any displayed server counters belong to the formal phase. FastSD-specific scheduler counters are not synthesized for SpecEdge."
            ),
            "formal_workload_rpc_counts": dict(overall["rpc_counts"]),
            "async_prefetch_status": "unknown",
        }
        if scheduler_path.exists():
            try:
                saved = json.loads(scheduler_path.read_text())
                metrics = saved.get("fastsd_scheduler", saved)
                if not isinstance(metrics, dict):
                    raise ValueError("scheduler metrics must be an object")
                scheduler.update(available=True, metrics=metrics)
                submitted = metrics.get("prefetch_async_submitted")
                failed = metrics.get("prefetch_async_failed", 0) or 0
                if method == "fastsd" and submitted == 0:
                    scheduler["async_prefetch_status"] = "not_exercised_no_tasks_submitted"
                    report["mechanism_observations"].append({
                        "method": method, "kind": "prefetch_not_exercised",
                        "message": "No asynchronous prefetch task was submitted during the Target process lifetime (prefetch_async_submitted=0). This mechanism was not exercised; zero completed/failed counters are not evidence of successful prefetch execution.",
                    })
                elif isinstance(submitted, (int, float)) and submitted > 0:
                    scheduler["async_prefetch_status"] = "submitted_with_recorded_failures" if failed else "submitted_without_recorded_failures"
                if failed:
                    report["mechanism_observations"].append({
                        "method": method, "kind": "prefetch_failures",
                        "message": f"Asynchronous prefetch recorded {failed} failure(s), even if all requests completed. Successful requests do not establish that every enabled mechanism succeeded.",
                        "prefetch_async_failed": failed,
                        "prefetch_async_last_error": metrics.get("prefetch_async_last_error"),
                    })
                if method == "fastsd" and not metrics.get("plans", 0):
                    report["mechanism_observations"].append({"method": method, "kind": "no_scheduling_plans",
                        "message": "No FastSD scheduling plans were recorded; configuration flags alone do not verify execution."})
            except (OSError, ValueError, TypeError) as exc:
                report["issues"].append(f"{method}/target/scheduler_metrics.json: {exc}")
        elif method == "fastsd":
            report["mechanism_observations"].append({"method": method, "kind": "scheduler_metrics_unavailable",
                "message": "Target scheduler metrics are unavailable; scheduling and asynchronous prefetch execution cannot be confirmed from enable flags alone."})
        report["methods"][method] = {"overall": overall, "nodes": nodes, "target_scheduler": scheduler}
        for group, summary in [("overall", overall), *nodes.items()]:
            row = {"method": method, "group": group, "gamma": summary.get("gamma", "mixed"),
                   "completed_requests": summary["completed_requests"], "generated_tokens": summary["generated_tokens"],
                   "acceptance_rate_weighted": summary["acceptance_rate_weighted"],
                   "rpc_errors": summary["rpc_error_count"], "failed_workers": summary["failed_workers"],
                   "wall_seconds": summary.get("wall_seconds"), "aggregate_tokens_per_second": summary.get("aggregate_tokens_per_second")}
            load = summary.get("host_load1", {})
            row.update({"host_load1_" + key: load.get(key) for key in ("count", "snapshot_count", "min", "mean", "max")})
            for metric in ("ttft_seconds", "tpot_ms", "request_e2e_seconds", "first_draft_seconds", "verify_rpc_ms"):
                row.update({metric + "_" + stat: summary[metric][stat] for stat in ("count", "mean", "p50", "p95")})
            flat_rows.append(row)
    report["output_comparisons"] = {
        method: compare_outputs(all_requests["standard"], all_requests[method], method)
        for method in methods if method != "standard" and "standard" in all_requests
    }
    report["status"] = "completed" if not report["issues"] else "incomplete_or_invalid"
    report["definitions"] = {
        "ttft_seconds": "Native request_start (before session/init) to first verified output block; excludes model loading, startup clock calibration and warmup.",
        "tpot_ms": "(request E2E - TTFT)/(generated_tokens - 1), including remaining executor work at request completion. Requests with fewer than two output tokens are excluded. This is amortized block-output latency, not a measured inter-token arrival distribution.",
        "percentiles": "Linear interpolation at (n-1)*q; overall percentiles pool individual requests from both nodes, not node percentiles.",
        "acceptance_rate_weighted": "sum(accepted_total)/sum(drafted_total); includes verified candidate tokens discarded at the final output cap.",
        "throughput": "Total retained output tokens divided only by explicit coordinator wall_seconds. No subtraction of monotonic timestamps from different hosts.",
        "startup_window": "The current coordinator starts its common window at the model-ready barrier, including subsequent clock calibration; native request TTFT/E2E start after calibration. Consult each method's timing.json for actual boundaries.",
        "verify_rpc_ms": "Observed verification round-trip duration: HTTP for Standard SD/FastSD and gRPC for SpecEdge; protocol latency is not pure network RTT.",
        "output_comparison": "Pair by (node, task_id), because task identifiers may overlap across nodes.",
        "host_load1": "Whole-host Linux one-minute load average sampled from progress.jsonl, including this experiment and unrelated activity. Mean is the arithmetic mean of snapshots, not a time-weighted average. Snapshots cover launcher monitoring, including startup. Host load is not CPU utilization or a measurement of background load alone; values from different nodes are not pooled.",
        "scheduler_counters": "Native Target counters cover process lifetime including a separate 4-token smoke and idle polling; formal request metrics exclude smoke. No inferred smoke plans are subtracted. SpecEdge server counters, if shown, start after its formal-phase statistics/KV reset. Request success and asynchronous prefetch execution are reported separately; zero submitted tasks means the mechanism was not exercised. No missing counter is assumed to be zero.",
    }
    (root / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = ["# Heterogeneous draft experiment results", "", f"Status: **{report['status']}**.", "",
             f"node1: {args.expected_per_node} Qwen3-1.7B drafts, fixed γ=100. node3: {args.expected_per_node} Qwen3-1.7B drafts, fixed γ=20.",
             f"Requested output cap: {args.max_new_tokens} tokens.",
             f"Target: {manifest.get('target_model', 'see manifest')}; dtype={manifest.get('dtype', 'NA')}; physical GPU={manifest.get('physical_gpu', 'NA')}; CPU weight offload={manifest.get('target_cpu_offload', 'NA')}; GPU weight budget={manifest.get('target_weight_gpu_budget_GiB', 'NA')} GiB.", "",
             "| Method | Group | Requests | Output tokens | TTFT mean / p50 / p95 (s) | TPOT mean / p50 / p95 (ms) | E2E mean / p50 / p95 (s) |",
             "| --- | --- | ---: | ---: | --- | --- | --- |"]
    for row in flat_rows:
        triplet = lambda key: " / ".join(format_number(row[key + "_" + stat]) for stat in ("mean", "p50", "p95"))
        lines.append(f"| {row['method']} | {row['group']} | {row['completed_requests']} | {row['generated_tokens']} | {triplet('ttft_seconds')} | {triplet('tpot_ms')} | {triplet('request_e2e_seconds')} |")
    lines += ["", "| Method | Output tokens | Draft acceptance | RPC errors | Coordinator window (s) | Aggregate throughput (tokens/s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        summary = report["methods"][method]["overall"]
        acceptance = summary["acceptance_rate_weighted"]
        acceptance_text = "NA" if acceptance is None else f"{acceptance:.2%}"
        lines.append(f"| {method} | {summary['generated_tokens']} | {acceptance_text} | {summary['rpc_error_count']} | {format_number(summary['wall_seconds'])} | {format_number(summary['aggregate_tokens_per_second'])} |")
    for method, comparison in report["output_comparisons"].items():
        lines += ["", f"{method} vs Standard SD: output token sequences match for **{comparison['identical_output_requests']}/{comparison['paired_requests']}** paired requests; mismatch details are in results.json."]
    lines += ["", "## Observed host load", "",
              "| Method | Node | Valid / total snapshots | Load1 min | Load1 mean | Load1 max |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method in methods:
        for node in GAMMAS:
            load = report["methods"][method]["nodes"][node]["host_load1"]
            lines.append(f"| {method} | {node} | {load['count']} / {load['snapshot_count']} | {format_number(load['min'])} | {format_number(load['mean'])} | {format_number(load['max'])} |")
    lines += ["", "These shared-host snapshots include experiment work and unrelated activity. They are not background-only load or CPU utilization. Methods were run sequentially under different host loads, so their latency and throughput differences cannot be attributed to the algorithm alone. The complete recorded load traces are plotted in `host_load.png` when the plotting script is run."]
    lines += ["", "## Target scheduling and asynchronous prefetch", "",
              "| Method | Counter scope | Plans | Verify slices | Prefill slices | Prefetch submitted | Prefetch completed | Prefetch failed |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    counters = ("plans", "verify_slices", "prefill_slices", "prefetch_async_submitted", "prefetch_async_completed", "prefetch_async_failed")
    for method in methods:
        metrics = report["methods"][method]["target_scheduler"]["metrics"] or {}
        values = [str(metrics[key]) if metrics.get(key) is not None else "NA" for key in counters]
        scope = "Lifetime incl. smoke / polling" if method in {"standard", "fastsd"} else "Formal phase; native counters NA"
        lines.append("| " + method + " | " + scope + " | " + " | ".join(values) + " |")
    lines += ["", "Native counters come from archived `target/scheduler_metrics.json` and cover the Target process lifetime, including a separate 4-token smoke and idle polling. Plans are shown exactly as archived; no inferred smoke plan count is subtracted. Formal latency metrics exclude smoke. Missing values are NA."]
    for method in methods:
        if method in {"standard", "fastsd"}:
            rpc_counts = report["methods"][method]["target_scheduler"]["formal_workload_rpc_counts"]
            lines.append(f"- **{method} formal workload only**: {rpc_counts.get('/verify', 0)} verify RPCs and {rpc_counts.get('/prefill', 0)} prefill RPCs. Lifetime slices in the table also include smoke; chunked prefill may produce multiple slices per RPC.")
    lines += ["", "SpecEdge resets its statistics and KV when entering the formal phase; any SpecEdge server counters shown use that phase's scope. FastSD-specific counters are not synthesized for it. Request completion alone does not demonstrate that all enabled scheduling/prefetch mechanisms worked."]
    for observation in report["mechanism_observations"]:
        lines.append(f"- **{observation['method']}**: {observation['message']}")
        if observation.get("prefetch_async_last_error"):
            lines.append("  Last recorded prefetch error: " + json.dumps(observation["prefetch_async_last_error"], ensure_ascii=False))
    lines += ["", "## Metric definitions", ""]
    lines += [f"- **{key}**: {value}" for key, value in report["definitions"].items()]
    lines += ["", "## Interpretation", "",
              "- TTFT includes generating and verifying the first entire candidate block; γ=100 therefore directly affects time to first visible output.",
              "- This is one run per method on shared hosts. Node hardware, runtime versions, prompt assignments and fixed γ differ; node differences cannot be attributed to γ alone.",
              "- Host load changed between methods, particularly on node3. These observations are not a controlled algorithm speed comparison; all request measurements and host snapshots are retained.",
              "- GPU/CPU weight offload, when enabled, is part of the measured Target configuration and should be reported with every performance comparison.",
              "- Short capped outputs do not constitute a GSM8K accuracy evaluation; complete recorded output equality is separate from reference-answer correctness."]
    if "specedge" in methods:
        lines += ["- SpecEdge is reported as `specedge_cpu_adapted_single_chain`: single-chain SpecExec with proactive drafting and gRPC, using experiment adapters for CPU execution, heterogeneous candidate lengths and Target weight offload. See the archived SpecEdge adapters and `specedge_plan.md` for implementation details.",
                  "- SpecEdge candidate acceptance excludes each cycle's Target bonus token; retained output tokens are counted after EOS and the output cap. Its gRPC round-trip timings are reported separately from the other methods' HTTP timings."]
    if report["issues"]:
        lines += ["", "## Incomplete data or validation issues", ""]
        lines += ["- " + issue for issue in report["issues"]]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "files": [str(root / p) for p in ("results.json", "results.csv", "RESULTS.md")],
                      "output_comparisons": {m: {k: v[k] for k in ("paired_requests", "identical_output_requests")} for m, v in report["output_comparisons"].items()},
                      "issue_count": len(report["issues"])}, ensure_ascii=False, indent=2))
    return 2 if args.require_complete and report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
