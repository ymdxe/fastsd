#!/usr/bin/env python3
"""Reproducibility helpers for the pinned official SpecEdge baseline.

The actual model runtime remains untouched under ../official.  This module adds
environment checks, paper-aligned config validation, draft-depth calibration,
and machine-readable metric normalization around the upstream implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by doctor on incomplete envs
    yaml = None


SCHEMA_VERSION = 1
UPSTREAM_URL = "https://github.com/kaist-ina/specedge.git"
UPSTREAM_COMMIT = "1edcaf02ffc41a7b57726450c5357ed216a3b9bc"
OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "official"

# Paper prices: A100 40GB $4.05/h, A100 80GB $5.05/h, RTX 4090 $0.35/h.
GPU_COST_PER_SECOND = {
    "A100-40": 4.05 / 3600,
    "A100-80": 5.05 / 3600,
}
RTX4090_COST_PER_SECOND = 0.35 / 3600

SPEC_BENCH_OFFSETS = {
    "multi_turn": (0, 80),
    "translation": (80, 160),
    "summarization": (160, 240),
    "question_answering": (240, 320),
    "mathematical_reasoning": (320, 400),
    "retrieval": (400, math.inf),
}


class ReproError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ReproError("PyYAML is required for config commands: python -m pip install pyyaml")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ReproError(f"Config must contain a YAML mapping: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReproError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ReproError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ReproError(f"No records found in {path}")
    return records


def _nested(record: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = record
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return None, None
    mean = statistics.fmean(materialized)
    std = statistics.stdev(materialized) if len(materialized) > 1 else 0.0
    return mean, std


def _safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise ReproError(f"Cannot compute {label}: denominator is {denominator}")
    return numerator / denominator


def _filter_subset(
    records: list[dict[str, Any]], subset: str
) -> list[dict[str, Any]]:
    if subset == "overall":
        return records
    lower, upper = SPEC_BENCH_OFFSETS[subset]
    return [
        record
        for record in records
        if lower <= int(record.get("req_idx", -1)) < upper
    ]


def _timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ReproError(f"Invalid ISO timestamp: {value!r}") from exc


def _config_metadata(path: Path, method: str) -> dict[str, Any]:
    config = _load_yaml(path)
    base = config.get("base", {})
    server = config.get("server", {})
    client = config.get("client", {})
    if method == "specedge":
        max_batch_size = server.get("max_batch_size")
        num_clients = server.get("num_clients")
    else:
        max_batch_size = client.get("max_batch_size")
        num_clients = server.get("num_clients")

    node_devices = sum(
        len(entries or []) for entries in (config.get("node", {}) or {}).values()
    )
    return {
        "config_path": str(path.resolve()),
        "config_sha256": _sha256(path),
        "exp_name": base.get("exp_name"),
        "seed": base.get("seed"),
        "dtype": base.get("dtype"),
        "max_len": base.get("max_len"),
        "target_model": server.get("target_model"),
        "draft_model": client.get("draft_model"),
        "dataset": client.get("dataset"),
        "temperature": server.get("temperature"),
        "max_batch_size": max_batch_size,
        "num_clients": num_clients,
        "node_devices": node_devices,
        "sample_req_cnt": client.get("sample_req_cnt"),
        "req_offset": client.get("req_offset", 0),
        "max_n_beams": client.get("max_n_beams"),
        "max_beam_len": client.get("max_beam_len"),
        "max_branch_width": client.get("max_branch_width"),
        "max_budget": client.get("max_budget"),
        "max_new_tokens": client.get("max_new_tokens"),
        "proactive_type": (client.get("proactive") or {}).get("type", "disabled"),
        "host": client.get("host"),
    }


def validate_paper_config(path: Path, method: str) -> dict[str, list[str]]:
    metadata = _config_metadata(path, method)
    errors: list[str] = []
    warnings: list[str] = []

    for key in ("target_model", "draft_model", "dataset", "max_batch_size"):
        if metadata.get(key) in (None, ""):
            errors.append(f"missing required config field: {key}")
    if metadata.get("max_budget") != 32:
        errors.append("paper main experiments use a draft tree budget of 32")
    if metadata.get("max_new_tokens") != 256:
        errors.append("paper main experiments generate up to 256 output tokens per request")
    if metadata.get("temperature") != 0.7:
        warnings.append("paper default temperature is 0.7")
    if metadata.get("dataset") != "specbench":
        warnings.append("paper main results use SpecBench; other datasets are appendix coverage")

    if method == "specedge":
        expected_clients = metadata.get("num_clients")
        if expected_clients != metadata.get("node_devices"):
            errors.append("server.num_clients must equal the number of configured edge processes")
        if expected_clients != 2 * int(metadata.get("max_batch_size") or 0):
            warnings.append("paper main setup uses 2 edge clients per server batch slot")
        host = str(metadata.get("host") or "")
        if host.startswith("127.0.0.1") or host.startswith("localhost"):
            warnings.append("localhost does not reproduce the paper's measured 14.07 ms WAN RTT")
        if "example" in host or "CHANGE_ME" in host:
            errors.append("replace the placeholder client.host with the real server endpoint")
    return {"errors": errors, "warnings": warnings}


def _git_commit(repo: Path) -> str | None:
    if not shutil.which("git"):
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def doctor(config_path: Path, method: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    add("linux", platform.system() == "Linux", platform.platform())
    add(
        "python_3_14",
        sys.version_info >= (3, 14),
        sys.version.split()[0] + " (upstream pyproject requires ~=3.14.0)",
    )
    for command in ("git", "uv", "bash", "ssh", "nvidia-smi"):
        location = shutil.which(command)
        add(command, location is not None, location or "not found")
    add("tc_optional", shutil.which("tc") is not None, shutil.which("tc") or "not found", False)

    commit = _git_commit(OFFICIAL_ROOT)
    add("upstream_commit", commit == UPSTREAM_COMMIT, commit or "unavailable")
    validation = validate_paper_config(config_path, method)
    add("paper_config", not validation["errors"], json.dumps(validation, ensure_ascii=False))

    gpu_lines: list[str] = []
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            gpu_lines = [line for line in result.stdout.splitlines() if line.strip()]

    return {
        "schema_version": SCHEMA_VERSION,
        "ready": all(item["ok"] for item in checks if item["required"]),
        "checks": checks,
        "config": _config_metadata(config_path, method),
        "config_validation": validation,
        "gpus": gpu_lines,
        "upstream": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT},
    }


def _specedge_metrics(
    data_dir: Path, subset: str, gpu: str
) -> dict[str, Any]:
    client_paths = sorted(data_dir.glob("client_*.jsonl"))
    server_path = data_dir / "server.jsonl"
    if not client_paths:
        raise ReproError(f"No client_*.jsonl files found in {data_dir}")
    if not server_path.exists():
        raise ReproError(f"Missing {server_path}")

    client_records: list[dict[str, Any]] = []
    for path in client_paths:
        client_records.extend(_read_jsonl(path))
    client_records = _filter_subset(client_records, subset)
    if not client_records:
        raise ReproError(f"No client records remain for subset {subset}")
    server_records = _read_jsonl(server_path)

    server_start = _timestamp(str(server_records[0].get("timestamp")))
    server_end = _timestamp(str(server_records[-1].get("timestamp")))
    server_seconds = (server_end - server_start).total_seconds()
    if server_seconds <= 0:
        raise ReproError("SpecEdge server log must span more than zero seconds")

    non_prefill = [
        record
        for record in client_records
        if int(record.get("step_idx", 0)) != 0
        and int(_nested(record, "target.prefill", 0)) == 0
    ]
    non_prefill_ms = [
        float(_nested(record, "draft.end_to_end", 0.0))
        + float(_nested(record, "target.end_to_end", 0.0))
        for record in non_prefill
    ]
    non_prefill_tokens = sum(int(record.get("num_accepted_tokens", 0)) for record in non_prefill)
    generated_tokens = sum(int(record.get("num_accepted_tokens", 0)) for record in client_records)
    edge_ms = sum(
        float(_nested(record, "draft.end_to_end", 0.0))
        + float(_nested(record, "target.end_to_end", 0.0))
        for record in client_records
    )
    accepted_mean, accepted_std = _mean_std(
        int(record.get("num_accepted_tokens", 0)) for record in client_records
    )
    server_verify_mean, server_verify_std = _mean_std(
        float(_nested(record, "target.server_end_to_end_t"))
        for record in server_records
        if int(_nested(record, "target.prefill", 0)) == 0
        and _nested(record, "target.server_end_to_end_t") is not None
    )
    forward_times = [
        float(value)
        for record in non_prefill
        for value in (_nested(record, "draft.forward", []) or [])
    ]
    draft_forward_mean, draft_forward_std = _mean_std(forward_times)
    proactive_ratio = sum(
        bool(_nested(record, "target.prev_proactive", False)) for record in client_records
    ) / len(client_records)

    server_cost = GPU_COST_PER_SECOND[gpu] * server_seconds
    edge_cost = RTX4090_COST_PER_SECOND * edge_ms / 1000
    total_cost = server_cost + edge_cost
    return {
        "generated_tokens": generated_tokens,
        "server_throughput_tok_s": generated_tokens / server_seconds,
        "inter_token_latency_ms": _safe_ratio(
            sum(non_prefill_ms), non_prefill_tokens, "inter-token latency"
        ),
        "accepted_tokens_per_verify": {"mean": accepted_mean, "std": accepted_std},
        "proactive_alignment_ratio": proactive_ratio,
        "server_verify_ms": {"mean": server_verify_mean, "std": server_verify_std},
        "draft_forward_ms": {"mean": draft_forward_mean, "std": draft_forward_std},
        "server_running_time_s": server_seconds,
        "edge_aggregate_running_time_s": edge_ms / 1000,
        "server_cost_usd": server_cost,
        "edge_cost_usd": edge_cost,
        "dollars_per_1m_tokens": _safe_ratio(total_cost, generated_tokens, "token cost") * 1_000_000,
        "cost_efficiency_1k_tokens_per_dollar": _safe_ratio(
            generated_tokens, total_cost, "cost efficiency"
        )
        / 1000,
        "record_count": len(client_records),
        "client_file_count": len(client_paths),
    }


def _server_only_metrics(data_dir: Path, gpu: str) -> dict[str, Any]:
    path = data_dir / "server_only.jsonl"
    if not path.exists():
        raise ReproError(f"Missing {path}")
    records = _read_jsonl(path)
    non_prefill = [record for record in records if int(record.get("prefill", 0)) == 0]
    generated_tokens = sum(int(record.get("num_accepted_tokens", 0)) for record in non_prefill)

    grouped: dict[int, dict[str, Any]] = {}
    for record in non_prefill:
        grouped.setdefault(int(record["server_iter_idx"]), record)
    server_ms = sum(
        float(_nested(record, "draft.end_to_end", 0.0))
        + float(_nested(record, "target.end_to_end", 0.0))
        for record in grouped.values()
    )
    total_non_prefill_ms = sum(
        float(_nested(record, "draft.end_to_end", 0.0))
        + float(_nested(record, "target.end_to_end", 0.0))
        for record in non_prefill
    )
    accepted_mean, accepted_std = _mean_std(
        int(record.get("num_accepted_tokens", 0)) for record in records
    )
    server_cost = GPU_COST_PER_SECOND[gpu] * server_ms / 1000
    return {
        "generated_tokens": generated_tokens,
        "server_throughput_tok_s": _safe_ratio(
            generated_tokens, server_ms / 1000, "server-only throughput"
        ),
        "inter_token_latency_ms": _safe_ratio(
            total_non_prefill_ms, generated_tokens, "server-only inter-token latency"
        ),
        "accepted_tokens_per_verify": {"mean": accepted_mean, "std": accepted_std},
        "proactive_alignment_ratio": None,
        "server_verify_ms": {"mean": None, "std": None},
        "draft_forward_ms": {"mean": None, "std": None},
        "server_running_time_s": server_ms / 1000,
        "edge_aggregate_running_time_s": 0.0,
        "server_cost_usd": server_cost,
        "edge_cost_usd": 0.0,
        "dollars_per_1m_tokens": _safe_ratio(server_cost, generated_tokens, "token cost") * 1_000_000,
        "cost_efficiency_1k_tokens_per_dollar": _safe_ratio(
            generated_tokens, server_cost, "cost efficiency"
        )
        / 1000,
        "record_count": len(records),
        "client_file_count": 0,
    }


def normalize(
    method: str,
    data_dir: Path,
    config_path: Path,
    gpu: str,
    subset: str = "overall",
    rtt_ms: float | None = None,
) -> dict[str, Any]:
    metrics = (
        _specedge_metrics(data_dir, subset, gpu)
        if method == "specedge"
        else _server_only_metrics(data_dir, gpu)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "metric_policy": "official_script_compatible",
        "source": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT},
        "config": _config_metadata(config_path, method),
        "environment": {"server_gpu": gpu, "measured_rtt_ms": rtt_ms},
        "subset": subset,
        "metrics": metrics,
        "caveats": [
            "SpecEdge upstream throughput uses the full server timestamp span.",
            "RTT is recorded separately because upstream client timing combines network and server wait.",
        ],
    }


def recommend_depth(data_dir: Path, rtt_ms: float) -> dict[str, Any]:
    server_records = _read_jsonl(data_dir / "server.jsonl")
    client_records: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("client_*.jsonl")):
        client_records.extend(_read_jsonl(path))
    non_prefill = [
        record
        for record in client_records
        if int(record.get("step_idx", 0)) != 0
        and int(_nested(record, "target.prefill", 0)) == 0
    ]
    forward_times = [
        float(value)
        for record in non_prefill
        for value in (_nested(record, "draft.forward", []) or [])
    ]
    verify_times = [
        float(_nested(record, "target.server_end_to_end_t"))
        for record in server_records
        if int(_nested(record, "target.prefill", 0)) == 0
        and _nested(record, "target.server_end_to_end_t") is not None
    ]
    draft_ms, _ = _mean_std(forward_times)
    verify_ms, _ = _mean_std(verify_times)
    if draft_ms is None or verify_ms is None:
        raise ReproError("Logs do not contain draft.forward and server verification timings")
    raw_depth = (verify_ms - rtt_ms) / draft_ms
    depth = max(1, round(raw_depth))
    return {
        "equation": "verify_ms ~= draft_depth * draft_forward_ms + rtt_ms",
        "server_verify_ms": verify_ms,
        "draft_forward_ms": draft_ms,
        "rtt_ms": rtt_ms,
        "raw_depth": raw_depth,
        "recommended_max_beam_len": depth,
    }


def prepare_sweep(
    config_path: Path, output_dir: Path, depths: list[int]
) -> list[Path]:
    config = _load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for depth in depths:
        candidate = copy.deepcopy(config)
        candidate["client"]["max_beam_len"] = depth
        candidate["base"]["exp_name"] = f"{candidate['base']['exp_name']}-depth-{depth}"
        output = output_dir / f"{config_path.stem}-depth-{depth}.yaml"
        if output.exists():
            raise ReproError(f"Refusing to overwrite existing sweep config: {output}")
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(candidate, handle, sort_keys=False)
        created.append(output)
    return created


def compare(specedge: dict[str, Any], server_only: dict[str, Any]) -> dict[str, Any]:
    if specedge.get("method") != "specedge" or server_only.get("method") != "server_only":
        raise ReproError("compare expects one specedge and one server_only normalized result")
    comparable_fields = (
        "target_model",
        "draft_model",
        "dataset",
        "temperature",
        "max_batch_size",
        "max_budget",
        "max_new_tokens",
        "seed",
        "dtype",
    )
    mismatches = {
        field: {
            "specedge": specedge["config"].get(field),
            "server_only": server_only["config"].get(field),
        }
        for field in comparable_fields
        if specedge["config"].get(field) != server_only["config"].get(field)
    }
    if mismatches:
        raise ReproError("Unfair comparison; config mismatches: " + json.dumps(mismatches))

    s_metrics = specedge["metrics"]
    b_metrics = server_only["metrics"]
    return {
        "schema_version": SCHEMA_VERSION,
        "fairness_check": "passed",
        "config": {field: specedge["config"].get(field) for field in comparable_fields},
        "gains": {
            "server_throughput_x": _safe_ratio(
                s_metrics["server_throughput_tok_s"],
                b_metrics["server_throughput_tok_s"],
                "throughput gain",
            ),
            "cost_efficiency_x": _safe_ratio(
                s_metrics["cost_efficiency_1k_tokens_per_dollar"],
                b_metrics["cost_efficiency_1k_tokens_per_dollar"],
                "cost-efficiency gain",
            ),
            "itl_reduction_percent": (
                1
                - _safe_ratio(
                    s_metrics["inter_token_latency_ms"],
                    b_metrics["inter_token_latency_ms"],
                    "ITL ratio",
                )
            )
            * 100,
            "accepted_tokens_per_verify_x": _safe_ratio(
                s_metrics["accepted_tokens_per_verify"]["mean"],
                b_metrics["accepted_tokens_per_verify"]["mean"],
                "accepted-token gain",
            ),
        },
        "paper_reference": {
            "server_throughput_x": 2.22,
            "cost_efficiency_x": 1.91,
            "itl_reduction_percent": 11.24,
        },
    }


def comparison_markdown(result: dict[str, Any]) -> str:
    gains = result["gains"]
    refs = result["paper_reference"]
    rows = [
        ("Server throughput", f"{gains['server_throughput_x']:.3f}x", f"{refs['server_throughput_x']:.2f}x"),
        ("Cost efficiency", f"{gains['cost_efficiency_x']:.3f}x", f"{refs['cost_efficiency_x']:.2f}x"),
        ("ITL reduction", f"{gains['itl_reduction_percent']:.3f}%", f"{refs['itl_reduction_percent']:.2f}%"),
        ("Accepted tokens/verify", f"{gains['accepted_tokens_per_verify_x']:.3f}x", "model/task dependent"),
    ]
    lines = [
        "# SpecEdge reproduction comparison",
        "",
        "| Metric | Reproduced | Paper reference |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {value} | {reference} |" for name, value, reference in rows)
    lines.extend(["", "Fairness check: passed.", ""])
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _parse_depths(value: str) -> list[int]:
    depths = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not depths or any(depth < 1 for depth in depths):
        raise argparse.ArgumentTypeError("depths must be positive comma-separated integers")
    return depths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--config", type=Path, required=True)
    doctor_parser.add_argument("--method", choices=("specedge", "server_only"), required=True)
    doctor_parser.add_argument("--output", type=Path)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--method", choices=("specedge", "server_only"), required=True)
    normalize_parser.add_argument("--data", type=Path, required=True)
    normalize_parser.add_argument("--config", type=Path, required=True)
    normalize_parser.add_argument("--gpu", choices=tuple(GPU_COST_PER_SECOND), required=True)
    normalize_parser.add_argument("--subset", choices=("overall", *SPEC_BENCH_OFFSETS), default="overall")
    normalize_parser.add_argument("--rtt-ms", type=float)
    normalize_parser.add_argument("--output", type=Path, required=True)

    depth_parser = subparsers.add_parser("recommend-depth")
    depth_parser.add_argument("--data", type=Path, required=True)
    depth_parser.add_argument("--rtt-ms", type=float, required=True)
    depth_parser.add_argument("--output", type=Path)

    sweep_parser = subparsers.add_parser("prepare-sweep")
    sweep_parser.add_argument("--config", type=Path, required=True)
    sweep_parser.add_argument("--depths", type=_parse_depths, default=_parse_depths("2,3,4,5,6,7,8"))
    sweep_parser.add_argument("--output-dir", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--specedge", type=Path, required=True)
    compare_parser.add_argument("--server-only", type=Path, required=True)
    compare_parser.add_argument("--output-json", type=Path, required=True)
    compare_parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.config, args.method)
            if args.output:
                _write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 2
        if args.command == "normalize":
            result = normalize(
                args.method, args.data, args.config, args.gpu, args.subset, args.rtt_ms
            )
            _write_json(args.output, result)
            print(args.output)
            return 0
        if args.command == "recommend-depth":
            result = recommend_depth(args.data, args.rtt_ms)
            if args.output:
                _write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "prepare-sweep":
            for path in prepare_sweep(args.config, args.output_dir, args.depths):
                print(path)
            return 0
        if args.command == "compare":
            with args.specedge.open("r", encoding="utf-8") as handle:
                specedge_result = json.load(handle)
            with args.server_only.open("r", encoding="utf-8") as handle:
                server_only_result = json.load(handle)
            result = compare(specedge_result, server_only_result)
            _write_json(args.output_json, result)
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(comparison_markdown(result), encoding="utf-8")
            print(args.output_json)
            print(args.output_md)
            return 0
    except (OSError, ReproError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
