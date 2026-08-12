#!/usr/bin/env python3
"""Read-only integrity validation for a finalized cloud-edge run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"JSON object required: {path}")
        return None
    return payload


def nonempty_jsonl(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append(f"missing JSONL artifact: {path}")
        return 0
    count = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL at {path}:{lineno}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"JSON object required at {path}:{lineno}")
        else:
            count += 1
    if count == 0:
        errors.append(f"no JSON records in {path}")
    return count


def nonempty_csv(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append(f"missing CSV artifact: {path}")
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                errors.append(f"missing CSV header: {path}")
                return 0
            count = sum(1 for _ in reader)
    except (OSError, csv.Error) as exc:
        errors.append(f"invalid CSV {path}: {exc}")
        return 0
    if count == 0:
        errors.append(f"no GPU samples in {path}")
    return count


def validate(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    report: dict[str, Any] = {"run_dir": str(run_dir), "checks": {}}
    if str(run_dir) != "/home/hdd" and not str(run_dir).startswith("/home/hdd/"):
        return [f"run directory must be under /home/hdd: {run_dir}"], report

    root = load_json(run_dir / "manifest.json", errors)
    edge = load_json(run_dir / "edge" / "manifest.json", errors)
    cloud = load_json(run_dir / "cloud" / "manifest.json", errors)
    if root and edge and cloud:
        report["run_id"] = root.get("run_id")
        report["method"] = root.get("method")
        for label, manifest in (("root", root), ("edge", edge), ("cloud", cloud)):
            if manifest.get("status") != "complete":
                errors.append(f"{label} manifest status is not complete: {manifest.get('status')!r}")
        if len({root.get("run_id"), edge.get("run_id"), cloud.get("run_id")}) != 1:
            errors.append("root, edge, and cloud manifests have different run IDs")
        if len({root.get("method"), edge.get("method"), cloud.get("method")}) != 1:
            errors.append("root, edge, and cloud manifests have different methods")

    for relative in (
        "config/resolved.yaml",
        "workload/arrival_trace.jsonl",
        "workload/manifest.json",
        "workload/prompts.jsonl",
        "edge/metrics/gpu_samples.csv",
        "cloud/metrics/gpu_samples.csv",
        "edge/metrics/requests.jsonl",
        "edge/metrics/summary.json",
        "edge/outputs/completions.jsonl",
        "metrics/requests.jsonl",
        "metrics/summary.json",
        "metrics/gpu_samples.csv",
        "outputs/completions.jsonl",
    ):
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required artifact: {path}")

    trace_path = run_dir / "workload" / "arrival_trace.jsonl"
    edge_trace_path = run_dir / "edge" / "workload" / "arrival_trace.jsonl"
    if trace_path.is_file() and edge_trace_path.is_file():
        root_trace_sha = sha256(trace_path)
        edge_trace_sha = sha256(edge_trace_path)
        report["checks"]["trace_sha256"] = root_trace_sha
        if root_trace_sha != edge_trace_sha:
            errors.append("root workload trace differs from edge workload trace")
        if edge:
            recorded = (
                edge.get("artifacts", {})
                .get("workload", {})
                .get("arrival_trace", {})
                .get("sha256")
            )
            if recorded and recorded != edge_trace_sha:
                errors.append("edge manifest trace SHA does not match copied trace")

    trace_count = nonempty_jsonl(trace_path, errors) if trace_path.is_file() else 0
    request_count = nonempty_jsonl(run_dir / "edge" / "metrics" / "requests.jsonl", errors)
    completion_count = nonempty_jsonl(run_dir / "edge" / "outputs" / "completions.jsonl", errors)
    report["checks"].update(
        {
            "trace_request_count": trace_count,
            "metric_request_count": request_count,
            "completion_count": completion_count,
        }
    )
    if trace_count and request_count != trace_count:
        errors.append(f"trace has {trace_count} requests but metrics has {request_count}")
    if trace_count and completion_count != trace_count:
        errors.append(f"trace has {trace_count} requests but completions has {completion_count}")

    request_artifact = run_dir / "edge" / "metrics" / "requests.jsonl"
    if request_artifact.is_file():
        for lineno, raw in enumerate(request_artifact.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "completed":
                errors.append(
                    f"edge request is not completed at {request_artifact}:{lineno}: "
                    f"{record.get('status')!r}"
                )

    summary = load_json(run_dir / "edge" / "metrics" / "summary.json", errors)
    if summary is not None:
        report["checks"]["summary_keys"] = sorted(summary.keys())
    report["checks"]["edge_gpu_samples"] = nonempty_csv(
        run_dir / "edge" / "metrics" / "gpu_samples.csv", errors
    )
    report["checks"]["cloud_gpu_samples"] = nonempty_csv(
        run_dir / "cloud" / "metrics" / "gpu_samples.csv", errors
    )
    return errors, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    errors, report = validate(Path(args.run_dir))
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
