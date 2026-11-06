#!/usr/bin/env python3
"""Plot observed request latencies from complete, archived raw metrics.

Example (the existing new_fastsd environment includes Matplotlib)::

    /home/hdd/zhangh/envs/new_fastsd/bin/python plot_results.py

Reads METHOD/NODE/run/edge_metrics_proc*.jsonl and writes
latency_comparison.png, latency_comparison.pdf and host_load.png. Summary percentiles cannot
recover a distribution, so this script intentionally uses the raw requests.
It refuses missing or incomplete methods instead of drawing placeholder data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METHODS = ("standard", "fastsd", "specedge")
LABELS = {"standard": "Standard SD", "fastsd": "FastSD", "specedge": "SpecEdge\n(CPU adapted)"}
COLORS = {"standard": "#377EB8", "fastsd": "#E68632", "specedge": "#3B9369"}
NODES = ("node1", "node3")


def load_records(root, methods, expected):
    records = {}
    for method in methods:
        records[method] = []
        for node in NODES:
            folder = root / method / node / "run"
            rows = []
            for path in sorted(folder.glob("edge_metrics_proc*.jsonl")):
                for line in path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        row["node"] = node
                        rows.append(row)
            if len(rows) != expected:
                raise ValueError(f"{method}/{node}: expected {expected} requests, found {len(rows)}; plot not generated")
            if len({str(r["task_id"]) for r in rows}) != expected:
                raise ValueError(f"{method}/{node}: duplicate task identifiers")
            for row in rows:
                n = row.get("generated_tokens")
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(f"{method}/{node}/{row['task_id']}: invalid generated_tokens")
                required = ["ttft_ms", "request_e2e_ms"] + (["tpot_ms"] if n > 1 else [])
                for key in required:
                    value = row.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{method}/{node}/{row['task_id']}: invalid {key}")
            records[method].extend(sorted(rows, key=lambda row: (row["node"], str(row["task_id"]))))
    return records


def points(records, group, metric):
    selected = [r for r in records if group == "overall" or r["node"] == group]
    if metric == "tpot_ms":
        selected = [r for r in selected if r["generated_tokens"] > 1]
    scale = 1000 if metric == "ttft_ms" else 1
    return [(r["node"] + "/" + str(r["task_id"]), r[metric] / scale) for r in selected]


def jitter(identifier):
    # Identical requests retain the same small horizontal offset across methods.
    value = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big") / (2**64 - 1)
    return (value - .5) * .26


def load_host_progress(root, methods):
    traces = {}
    for method in methods:
        traces[method] = {}
        for node in NODES:
            path = root / method / node / "progress.jsonl"
            if not path.exists():
                raise ValueError(f"Missing {path}; host load comparison cannot be generated")
            rows = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                epoch, loadavg = record.get("epoch"), record.get("loadavg")
                if (not isinstance(epoch, (int, float)) or not math.isfinite(epoch)
                        or not isinstance(loadavg, (list, tuple)) or not loadavg
                        or not isinstance(loadavg[0], (int, float)) or not math.isfinite(loadavg[0])):
                    raise ValueError(f"Invalid epoch/loadavg snapshot in {path}; refusing to silently omit it")
                rows.append((epoch, loadavg[0]))
            if not rows:
                raise ValueError(f"Empty host load trace: {path}")
            rows.sort()
            origin = rows[0][0]
            traces[method][node] = [((epoch - origin) / 60, load) for epoch, load in rows]
    return traces


def draw_host_load(plt, traces, methods, root, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    maximum = max(value for method in methods for node in NODES for _, value in traces[method][node])
    for ax, node in zip(axes, NODES):
        for method in methods:
            trace = traces[method][node]
            ax.plot([t for t, _ in trace], [value for _, value in trace],
                    color=COLORS[method], linewidth=1.2, marker=".", markersize=3,
                    label=LABELS[method].replace("\n", " ") + f" (n={len(trace)})")
        ax.set_title("cpunode1 · γ = 100" if node == "node1" else "cpunode3 · γ = 20")
        ax.set_xlabel("Time since first host snapshot (min)")
        ax.set_xlim(left=0)
        ax.set_ylim(0, max(1, maximum * 1.08))
        ax.grid(color="#E1E4E7", linewidth=.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("Host load average (1 min)")
    fig.suptitle("Shared-host load during each method", fontsize=14, y=.98)
    fig.text(.065, .025,
             "All recorded snapshots are shown; t=0 is each method/node's first launcher snapshot, including startup.\n"
             "Load includes experiment work and unrelated activity; it is not background-only load or CPU utilization.\n"
             "Different host loads confound the method comparison: latency/throughput differences cannot be attributed to the algorithm alone.",
             fontsize=8.4, color="#444444", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.86, wspace=.12)
    destination = root / "host_load.png"
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--expected-per-node", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve()
    methods = list(dict.fromkeys(args.methods))
    records = load_records(root, methods, args.expected_per_node)
    traces = load_host_progress(root, methods)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    # Load the plotting stack only after data completeness has been established.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharey="row")
    groups = (("overall", "All requests · mixed γ"), ("node1", "cpunode1 · γ = 100"),
              ("node3", "cpunode3 · γ = 20"))
    for column, (group, title) in enumerate(groups):
        for row_index, (metric, ylabel) in enumerate((("ttft_ms", "TTFT (s)"), ("tpot_ms", "TPOT (ms)"))):
            ax = axes[row_index, column]
            labels = []
            for position, method in enumerate(methods, 1):
                observed = points(records[method], group, metric)
                values = [value for _, value in observed]
                labels.append(LABELS[method] + f"\nn={len(values)}")
                if values:
                    artists = ax.boxplot([values], positions=[position], widths=.48,
                                         patch_artist=True, showfliers=False, manage_ticks=False,
                                         medianprops={"color": "#222222", "linewidth": 2},
                                         boxprops={"edgecolor": COLORS[method], "linewidth": 1.2},
                                         whiskerprops={"color": COLORS[method], "linewidth": 1},
                                         capprops={"color": COLORS[method], "linewidth": 1})
                    artists["boxes"][0].set_facecolor(COLORS[method])
                    artists["boxes"][0].set_alpha(.18)
                    ax.scatter([position + jitter(key) for key, _ in observed], values,
                               s=25, color=COLORS[method], alpha=.85, linewidths=.5,
                               edgecolors="white", zorder=3)
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_xlim(.5, len(methods) + .5)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="#DDE1E5", linewidth=.7, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(title, pad=12)

    fig.suptitle("Qwen3-1.7B drafts → Qwen3-32B target: observed request latencies", fontsize=15, y=.985)
    if manifest.get("target_cpu_offload"):
        hardware = (f"Target: {manifest.get('dtype', 'FP16')}, physical GPU {manifest.get('physical_gpu', '3')} + CPU weight offload"
                    f"; GPU weight budget {manifest.get('target_weight_gpu_budget_GiB', 'NA')} GiB.")
    else:
        hardware = "Target device and precision: see manifest.json."
    caption = (f"One run per method; {args.expected_per_node * len(NODES)} requests per method ({args.expected_per_node} per node). "
               "Dots: individual requests; boxes: IQR; black line: median; whiskers: 1.5×IQR.\n"
               f"{hardware} No confidence intervals are implied.\n"
               "TTFT ends at the first verified output block. TPOT = (E2E − TTFT)/(output tokens − 1); single-token outputs excluded. "
               "Node, γ and prompt assignment vary together.\n"
               "Host load differs between methods (see host_load.png); these observations do not isolate algorithm effects.")
    fig.text(.055, .025, caption, ha="left", va="bottom", fontsize=8.3, color="#444444", linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.90, wspace=.13, hspace=.40)
    destinations = [root / "latency_comparison.png", root / "latency_comparison.pdf"]
    fig.savefig(destinations[0], dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(destinations[1], bbox_inches="tight", facecolor="white",
                metadata={"Title": "Heterogeneous draft request latencies",
                          "Subject": "Observed TTFT and TPOT distributions from one run per method"})
    plt.close(fig)
    destinations.append(draw_host_load(plt, traces, methods, root, args.dpi))
    print(json.dumps({"files": [str(p) for p in destinations],
                      "requests_per_method": {method: len(rows) for method, rows in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
