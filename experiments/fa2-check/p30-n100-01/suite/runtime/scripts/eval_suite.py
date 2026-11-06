"""Prepare, validate, normalize, and compare the four-method evaluation suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import random
import shlex
import statistics
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FORMAL_CPUSET = "56-71,80-95"
FORMAL_CPU_PREFIX = "nice -n 5 numactl --physcpubind=56-71,80-95 --interleave=2,3"

ANALYSIS_DATASETS = ("humaneval", "gsm8k", "mgsm", "mt_bench")
ANALYSIS_METHODS = ("fastsd", "specedge_cpu_adapted", "standard_sd", "draft_only")
ANALYSIS_BOOTSTRAP_SEED = 42
ANALYSIS_BOOTSTRAP_ITERATIONS = 10_000
ANALYSIS_SPEEDUP_METRICS = {
    "e2e_ms": "lower_is_better",
    "ttft_ms": "lower_is_better",
    "tpot_ms": "lower_is_better",
}

from src.common_metrics import paired_method_analysis, percentile, summarize_requests
from src.evaluation import (
    DATASET_FILES,
    load_canonical_jsonl,
    load_evaluation_records,
    workload_fingerprint,
    write_canonical_jsonl,
)
from src.parity import (
    FOUR_METHODS,
    ORACLE_METHOD,
    build_token_parity_rows,
    exact_gate_summary,
    summarize_token_parity,
    write_token_parity_report,
)
from src.run_artifacts import (
    append_command,
    append_status,
    copy_file_once,
    git_sha,
    require_unique_sample_ids,
    write_json_once,
    write_text_once,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_execution(
    config: dict[str, Any], overrides: dict[str, Any] | None = None
) -> dict[str, str]:
    """Resolve the exact host/path/interpreter values used by the plan.

    The explicit keys are intentionally host-qualified.  Legacy ``python``,
    ``target_python``, ``specedge_python`` and ``repo_path_linux`` remain valid
    fallbacks so historical configs can still be inspected without silently
    changing their meaning.
    """

    execution = config.get("execution", {})
    overrides = overrides or {}

    def pick(name: str, *legacy: str, default: Any = None) -> str:
        candidates = [overrides.get(name), execution.get(name), config.get(name)]
        candidates.extend(execution.get(key) for key in legacy)
        candidates.extend(config.get(key) for key in legacy)
        candidates.append(default)
        for value in candidates:
            if value is not None and str(value).strip():
                return str(value)
        raise ValueError(f"missing execution value: {name}")

    node3_repo = pick(
        "node3_repo",
        "repo_path_linux",
        default=config.get("repo_path_linux", ""),
    )
    node2_repo = pick(
        "node2_repo",
        "repo_path_linux",
        default=node3_repo,
    )
    node3_edge_python = pick(
        "node3_edge_python",
        "python",
        default="python",
    )
    node3_specedge_python = pick(
        "node3_specedge_python",
        "specedge_python",
        "node3_edge_python",
        default=node3_edge_python,
    )
    node2_target_python = pick(
        "node2_target_python",
        "target_python",
        "node3_edge_python",
        default=node3_edge_python,
    )
    node2_specedge_python = pick(
        "node2_specedge_python",
        "specedge_python",
        "node2_target_python",
        default=node2_target_python,
    )
    return {
        "node3_repo": node3_repo,
        "node2_repo": node2_repo,
        "node3_edge_python": node3_edge_python,
        "node3_specedge_python": node3_specedge_python,
        "node2_target_python": node2_target_python,
        "node2_specedge_python": node2_specedge_python,
    }


def resolve_models(config: dict[str, Any]) -> dict[str, str]:
    """Resolve host-local draft copies while keeping the legacy draft fallback."""

    models = config.get("models", {})
    legacy_draft = models.get("draft") or config.get("draft_model")
    if not legacy_draft:
        raise ValueError("missing models.draft fallback")
    return {
        "node3_draft": str(
            models.get("node3_draft")
            or config.get("node3_draft")
            or legacy_draft
        ),
        "node2_draft": str(
            models.get("node2_draft")
            or config.get("node2_draft")
            or legacy_draft
        ),
        "target": str(models["target"]),
    }


def shell_prefix(value: Any) -> tuple[str, str]:
    """Return a safe shell rendering and the raw prefix for audit records."""

    if value is None:
        return "", ""
    if isinstance(value, list):
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError("topology.cpu_prefix list entries must be non-empty strings")
        raw = " ".join(value)
        return shlex.join(value) + " ", raw
    if not isinstance(value, str):
        raise ValueError("topology.cpu_prefix must be a string or argv list")
    raw = value
    if not raw.strip():
        return "", raw
    try:
        argv = shlex.split(raw, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid topology.cpu_prefix: {exc}") from exc
    if not argv:
        return "", raw
    return shlex.join(argv) + " ", raw


def prefixed_command(
    prefix: str,
    command: str,
    *,
    environment: dict[str, Any] | None = None,
) -> str:
    """Place env assignments after taskset/numactl and before the command."""

    env_text = ""
    if environment:
        assignments = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in environment.items()
        )
        env_text = f"env {assignments} "
    return f"{prefix}{env_text}{command}"


def validate_experiment_config(config: dict[str, Any]) -> list[str]:
    """Validate the frozen CPU experiment tracks.

    Older comparison configurations predate the explicit ``track`` field and
    remain renderable for backwards compatibility.  Once a configuration opts
    into a named track, however, silently mixing latency and throughput
    settings would invalidate the comparison, so fail early with a useful
    error instead of producing a misleading run directory.
    """

    generation = config.get("generation", {})
    topology = config.get("topology", {})
    dataset = config.get("dataset", {})
    max_new_tokens = int(generation.get("max_new_tokens", 0))
    if max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if max_new_tokens <= 16:
        scope = config.get("evaluation_scope", topology.get("evaluation_scope"))
        if scope not in {None, "communication_smoke"}:
            raise ValueError(
                "max_new_tokens <= 16 is communication_smoke only; "
                f"got evaluation_scope={scope!r}"
            )

    track = topology.get("track")
    if track is None:
        return []
    if track not in {"latency", "throughput"}:
        raise ValueError("topology.track must be 'latency' or 'throughput'")

    arrival = dataset.get("arrival_distribution", "immediate")
    concurrency = int(topology.get("concurrency", 1))
    workers = int(topology.get("draft_workers", 1))
    threads = int(topology.get("draft_threads", 1))
    warmups = int(topology.get("warmup_requests", 0))
    if warmups < 0:
        raise ValueError("topology.warmup_requests must be non-negative")

    if track == "latency":
        expected = {
            "dataset.arrival_distribution": (arrival, "immediate"),
            "topology.concurrency": (concurrency, 1),
            "topology.draft_workers": (workers, 1),
            "topology.draft_threads": (threads, 32),
            "topology.warmup_requests": (warmups, 10),
        }
    else:
        expected = {
            "dataset.arrival_distribution": (arrival, "poisson"),
            "topology.concurrency": (concurrency, 4),
            "topology.draft_workers": (workers, 4),
            "topology.draft_threads": (threads, 8),
        }
    mismatches = [
        f"{name}={actual!r}, expected {expected_value!r} for {track} track"
        for name, (actual, expected_value) in expected.items()
        if actual != expected_value
    ]
    if mismatches:
        raise ValueError("invalid frozen track configuration: " + "; ".join(mismatches))

    draft_devices = list(topology.get("specedge_draft_devices", []))
    if draft_devices and not all(
        str(device).split(":", 1)[0].lower() == "cpu" for device in draft_devices
    ):
        raise ValueError("the CPU comparison tracks require CPU SpecEdge draft devices")
    draft_only_devices = list(topology.get("draft_only_devices", draft_devices))
    if draft_only_devices and not all(
        str(device).split(":", 1)[0].lower() == "cpu" for device in draft_only_devices
    ):
        raise ValueError("the CPU comparison tracks require CPU draft-only devices")
    _rendered_prefix, raw_prefix = shell_prefix(topology.get("cpu_prefix", ""))
    if topology.get("shared_load", True) is not True:
        raise ValueError("formal CPU comparison requires topology.shared_load=true")
    physical_cpu_count = int(topology.get("physical_cpu_count", 32))
    numa_cores_per_node = int(topology.get("numa_cores_per_node", 16))
    if physical_cpu_count != 32:
        raise ValueError("formal CPU comparison fixes physical_cpu_count=32")
    if numa_cores_per_node != 16:
        raise ValueError("formal CPU comparison fixes numa_cores_per_node=16")
    numa_node_ids = list(topology.get("numa_node_ids", [2, 3]))
    if numa_node_ids != [2, 3]:
        raise ValueError("formal CPU comparison fixes NUMA nodes [2, 3]")
    fixed_cpuset = topology.get("fixed_cpuset", "")
    if not isinstance(fixed_cpuset, str):
        raise ValueError("topology.fixed_cpuset must be a string")
    if fixed_cpuset != FORMAL_CPUSET:
        raise ValueError(
            f"formal CPU comparison requires fixed_cpuset={FORMAL_CPUSET}; "
            "older 8-15/32-39 candidates are forbidden"
        )
    if shlex.split(raw_prefix) != shlex.split(FORMAL_CPU_PREFIX):
        raise ValueError(
            "formal CPU comparison requires cpu_prefix='nice -n 5 numactl "
            "--physcpubind=56-71,80-95 --interleave=2,3'"
        )
    method_order = list(topology.get("method_order", FOUR_METHODS))
    if method_order != list(FOUR_METHODS) and sorted(method_order) != sorted(FOUR_METHODS):
        raise ValueError(
            "topology.method_order must contain each four-method label exactly once"
        )
    repeat_index = int(topology.get("repeat_index", 0))
    if repeat_index < 0:
        raise ValueError("topology.repeat_index must be non-negative")

    notes = [
        "max_new_tokens <= 16 is communication_smoke only"
        if max_new_tokens <= 16
        else "max_new_tokens > 16 is eligible for quality/performance reporting"
    ]
    notes.append(
        "shared-load fair comparison: fixed cpuset 56-71,80-95; "
        "NUMA nodes 2/3 with 16 physical cores each"
    )
    return notes


def output_layout(
    config: dict[str, Any], execution: dict[str, str] | None = None
) -> dict[str, Any]:
    local_root = REPO_ROOT / "exp" / "comparison" / config["run_id"]
    execution = execution or resolve_execution(config)
    node3_root = (
        PurePosixPath(execution["node3_repo"])
        / "exp"
        / "comparison"
        / config["run_id"]
    )
    node2_root = (
        PurePosixPath(execution["node2_repo"])
        / "exp"
        / "comparison"
        / config["run_id"]
    )
    return {
        "local_root": local_root,
        "execution": execution,
        # ``linux_root`` remains the node3 alias for existing callers.
        "linux_root": node3_root,
        "node3_root": node3_root,
        "node2_root": node2_root,
        "canonical": local_root / "inputs" / "canonical.jsonl",
        "raw_input": local_root / "inputs" / "raw_input.jsonl",
        "linux_canonical": node3_root / "inputs" / "canonical.jsonl",
        "node3_canonical": node3_root / "inputs" / "canonical.jsonl",
        "node2_canonical": node2_root / "inputs" / "canonical.jsonl",
        "manifest": local_root / "run_manifest.json",
        "commands": local_root / "commands.txt",
        "status": local_root / "run_status.jsonl",
        "cpu_preflight": local_root / "cpu_preflight.json",
        "cpu_postflight": local_root / "cpu_postflight.json",
        "specedge_config": local_root / "specedge" / "specedge.yaml",
        "node3_specedge_config": local_root / "specedge" / "specedge.yaml",
        "node2_specedge_config": local_root / "specedge" / "node2.yaml",
        "linux_specedge_config": node3_root / "specedge" / "specedge.yaml",
        "node3_specedge_config_linux": node3_root / "specedge" / "specedge.yaml",
        "node2_specedge_config_linux": node2_root / "specedge" / "node2.yaml",
    }


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def render_specedge_config(
    config: dict[str, Any],
    workload_hash: str,
    layout: dict[str, Any],
    *,
    host_role: str = "node3",
) -> str:
    if host_role not in {"node2", "node3"}:
        raise ValueError("host_role must be node2 or node3")
    generation = config["generation"]
    topology = config["topology"]
    models = resolve_models(config)
    draft_devices = topology["specedge_draft_devices"]
    draft_is_cpu = bool(draft_devices) and all(
        str(device).split(":", 1)[0].lower() == "cpu" for device in draft_devices
    )
    client_dtype = "fp32" if draft_is_cpu else "bf16"
    client_engine = "cpu_adapter" if draft_is_cpu else "official_graph_engine"
    client_threads = int(topology.get("draft_threads", 1))
    execution = layout.get("execution") or resolve_execution(config)
    if host_role == "node2":
        linux_root = layout.get("node2_root", layout["linux_root"])
        canonical_path = layout.get("node2_canonical", layout["linux_canonical"])
        integration_python = execution["node2_specedge_python"]
        repo_root = execution["node2_repo"]
    else:
        linux_root = layout.get("node3_root", layout["linux_root"])
        canonical_path = layout.get("node3_canonical", layout["linux_canonical"])
        integration_python = execution["node3_specedge_python"]
        repo_root = execution["node3_repo"]
    specedge_host = topology.get("specedge_host", "127.0.0.1:18000")
    specedge_host_name = str(specedge_host).rsplit(":", 1)[0]
    lines = [
        "version: 1",
        "opt: 2",
        "base:",
        f"  result_path: {_yaml_scalar(linux_root / 'specedge' / 'raw')}",
        f"  exp_name: {_yaml_scalar(config['run_id'])}",
        "  dtype: bf16",
        f"  seed: {int(generation['seed'])}",
        "  ssh_key: ~/.ssh/id_ed25519",
        "  max_len: 4096",
        "server:",
        "  process_name: server",
        f"  target_model: {_yaml_scalar(models['target'])}",
        f"  device: {_yaml_scalar(topology['specedge_target_device'])}",
        f"  temperature: {float(generation['temperature'])}",
        f"  max_batch_size: {len(draft_devices)}",
        f"  num_clients: {len(draft_devices)}",
        "  batch_type: dynamic",
        "  cache_prefill: false",
        "client:",
        f"  host: {_yaml_scalar(specedge_host)}",
        "  process_name: client",
        f"  draft_model: {_yaml_scalar(models['node2_draft'] if host_role == 'node2' else models['node3_draft'])}",
        "  dataset: fastsd_external",
        f"  dtype: {client_dtype}",
        f"  engine: {client_engine}",
        f"  threads: {client_threads}",
        "  reasoning: false",
        "  sample_req_cnt: 1",
        "  req_offset: 0",
        "  max_n_beams: 32",
        f"  max_beam_len: {int(generation['gamma'])}",
        "  max_branch_width: 16",
        "  max_budget: 32",
        "  proactive:",
        "    type: included",
        "    max_n_beams: 32",
        "    max_beam_len: 3",
        "    max_branch_width: 16",
        "    max_budget: 32",
        f"  max_new_tokens: {int(generation['max_new_tokens'])}",
        "  max_request_num: -1",
        "node:",
        "  local:",
    ]
    lines.extend(f"    - device: {_yaml_scalar(device)}" for device in draft_devices)
    lines.extend(
        [
            "integration:",
            f"  host_role: {_yaml_scalar(host_role)}",
            f"  repo_root: {_yaml_scalar(repo_root)}",
            f"  python: {_yaml_scalar(integration_python)}",
            f"  server_host: {_yaml_scalar(specedge_host_name)}",
            f"  server_port: {int(topology.get('specedge_port', 18000))}",
            f"  dataset_file: {_yaml_scalar(canonical_path)}",
            f"  completion_dir: {_yaml_scalar(linux_root / 'specedge' / 'requests')}",
            f"  workload_hash: {_yaml_scalar(workload_hash)}",
            f"  method: {_yaml_scalar('specedge_cpu_adapted' if draft_is_cpu else 'specedge')}",
            f"  commands_path: {_yaml_scalar(linux_root / 'commands.txt')}",
            f"  status_path: {_yaml_scalar(linux_root / 'run_status.jsonl')}",
            f"  arrival_distribution: {_yaml_scalar(config['dataset'].get('arrival_distribution', 'immediate'))}",
            f"  warmup_requests: {int(topology.get('warmup_requests', 0))}",
            "  startup_delay_s: 15",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_command_record(path: Path, config_path: str | Path, body: str) -> None:
    """Append a timestamped, copyable command block to the run directory."""
    append_command(path, body, config=config_path)


def resolve_prepare_context(
    execution: dict[str, str], current_repo: str | Path | None = None
) -> dict[str, str]:
    """Choose the interpreter that actually owns this prepare invocation."""

    actual_repo = os.path.realpath(os.path.abspath(str(current_repo or REPO_ROOT)))
    actual_repo_key = os.path.normcase(actual_repo)

    def same_path(candidate: str) -> bool:
        return actual_repo_key == os.path.normcase(
            os.path.realpath(os.path.abspath(str(candidate)))
        )

    for role, python_key in (
        ("node3", "node3_edge_python"),
        ("node2", "node2_target_python"),
    ):
        if same_path(execution[f"{role}_repo"]):
            return {
                "host_role": role,
                "repo": actual_repo,
                "python": execution[python_key],
            }
    return {
        "host_role": "local/unknown",
        "repo": actual_repo,
        "python": sys.executable,
    }


def prepare(config_path: str) -> int:
    config = load_config(config_path)
    validation_notes = validate_experiment_config(config)
    execution = resolve_execution(config)
    prepare_context = resolve_prepare_context(execution)
    layout = output_layout(config)
    dataset = config["dataset"]
    generation = config["generation"]
    model_paths = resolve_models(config)
    data_path = Path(dataset["data_path"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    raw_source = data_path / DATASET_FILES[dataset["name"]] if data_path.is_dir() else data_path
    records = load_evaluation_records(
        dataset["name"],
        data_path,
        max_requests=int(dataset.get("max_requests", -1)),
        arrival_distribution=dataset.get("arrival_distribution", "immediate"),
        arrival_rate=float(dataset.get("arrival_rate_rps", 1.0)),
        arrival_seed=int(dataset.get("arrival_seed", 1234)),
    )
    fingerprint = workload_fingerprint(records, generation)
    require_unique_sample_ids(
        {
            "sample_id": record.sample_id
        }
        for record in records
    )
    canonical_payload = "".join(
        json.dumps(
            {
                **record.__dict__,
                "global_index": index,
            },
            ensure_ascii=False,
        )
        + "\n"
        for index, record in enumerate(records)
    )
    write_text_once(layout["canonical"], canonical_payload)
    copy_file_once(raw_source, layout["raw_input"])
    layout["specedge_config"].parent.mkdir(parents=True, exist_ok=True)
    write_text_once(
        layout["specedge_config"], render_specedge_config(config, fingerprint, layout)
    )
    write_text_once(
        layout["node2_specedge_config"],
        render_specedge_config(config, fingerprint, layout, host_role="node2"),
    )
    max_new_tokens = int(generation.get("max_new_tokens", 0))
    evaluation_scope = "communication_smoke" if max_new_tokens <= 16 else "quality"
    draft_devices = list(config["topology"].get("specedge_draft_devices", []))
    methods = list(FOUR_METHODS)
    git_info = {"fastsd": git_sha(REPO_ROOT)}
    official_root = REPO_ROOT / "baselines" / "specedge" / "official"
    git_info["specedge_official"] = git_sha(official_root)
    manifest = {
        "schema_version": 2,
        "run_id": config["run_id"],
        "workload_hash": fingerprint,
        "num_requests": len(records),
        "dataset": dataset,
        "models": config["models"],
        "models_by_host": model_paths,
        "generation": generation,
        "evaluation_scope": evaluation_scope,
        "mt_bench_turn_policy": (
            "first_turn_only" if dataset["name"] == "mt_bench" else "not_applicable"
        ),
        "canonical_dataset": str(layout["node3_canonical"]),
        "canonical_dataset_by_host": {
            "node3": str(layout["node3_canonical"]),
            "node2": str(layout["node2_canonical"]),
        },
        "raw_input": str(layout["raw_input"]),
        "git_sha": git_info,
        "methods": methods,
        "oracle_method": ORACLE_METHOD,
        "execution": execution,
        "prepare_context": prepare_context,
        "specedge_configs": {
            "node3": str(layout["node3_specedge_config_linux"]),
            "node2": str(layout["node2_specedge_config_linux"]),
        },
        "topology": {
            "draft_devices": draft_devices,
            "draft_threads": config["topology"].get("draft_threads"),
            "thread_environment": {
                "OMP_NUM_THREADS": config["topology"].get("draft_threads"),
                "MKL_NUM_THREADS": config["topology"].get("draft_threads"),
                "OPENBLAS_NUM_THREADS": config["topology"].get("draft_threads"),
                "TORCH_NUM_THREADS": config["topology"].get("draft_threads"),
            },
            "track": config["topology"].get("track", "latency"),
            "concurrency": config["topology"].get("concurrency"),
            "warmup_requests": config["topology"].get("warmup_requests", 0),
            "cpu_prefix": config["topology"].get("cpu_prefix", ""),
            "shared_load": bool(config["topology"].get("shared_load", True)),
            "fixed_cpuset": config["topology"].get("fixed_cpuset", ""),
            "physical_cpu_count": config["topology"].get("physical_cpu_count"),
            "numa_cores_per_node": config["topology"].get("numa_cores_per_node"),
            "method_order": config["topology"].get("method_order", list(FOUR_METHODS)),
            "repeat_index": config["topology"].get("repeat_index", 0),
            "cpu_preflight": str(layout["linux_root"] / "cpu_preflight.json"),
            "cpu_postflight": str(layout["linux_root"] / "cpu_postflight.json"),
            "target_host": config["topology"].get("target_host", "127.0.0.1"),
            "target_port": config["topology"].get("target_port", 18001),
            "target_bind_host": config["topology"].get("target_bind_host", "127.0.0.1"),
            "specedge_host": config["topology"].get("specedge_host", "127.0.0.1:18000"),
            "specedge_bind_host": config["topology"].get("specedge_bind_host", "127.0.0.1"),
            "specedge_port": config["topology"].get("specedge_port", 18000),
        },
        "validation_notes": validation_notes,
    }
    write_json_once(layout["manifest"], manifest)
    prepare_command = (
        f"cd {shlex.quote(prepare_context['repo'])} && "
        f"{shlex.quote(prepare_context['python'])} scripts/eval_suite.py prepare "
        f"--config {shlex.quote(str(config_path))}"
    )
    _append_command_record(
        layout["commands"],
        config_path,
        "\n".join(
            [
                f"# prepare_host_role={prepare_context['host_role']}",
                f"# prepare_repo={prepare_context['repo']}",
                f"# prepare_python={prepare_context['python']}",
                f"# resolved_execution={json.dumps(execution, ensure_ascii=False, sort_keys=True)}",
                f"# node3_specedge_config={layout['node3_specedge_config_linux']}",
                f"# node2_specedge_config={layout['node2_specedge_config_linux']}",
                prepare_command,
            ]
        ),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"SpecEdge config: {layout['specedge_config']}")
    print(f"Command record: {layout['commands']}")
    append_status(
        layout["status"],
        method="suite",
        phase="prepare",
        exit_code=0,
        command=prepare_command,
    )
    return 0


def _read_manifest(config: dict[str, Any], layout: dict[str, Path]) -> dict[str, Any]:
    if not layout["manifest"].is_file():
        raise FileNotFoundError("run manifest is missing; run prepare first")
    return json.loads(layout["manifest"].read_text(encoding="utf-8"))


def print_plan(
    config_path: str,
    *,
    python_bin: str | None = None,
    target_host: str | None = None,
    target_port: int | None = None,
    cpu_prefix: str | None = None,
    node3_edge_python: str | None = None,
    node3_specedge_python: str | None = None,
    node2_target_python: str | None = None,
    node2_specedge_python: str | None = None,
    node3_repo: str | None = None,
    node2_repo: str | None = None,
) -> int:
    config = load_config(config_path)
    validate_experiment_config(config)
    execution_overrides = {
        key: value
        for key, value in {
            "node3_edge_python": node3_edge_python or python_bin,
            "node3_specedge_python": node3_specedge_python,
            "node2_target_python": node2_target_python,
            "node2_specedge_python": node2_specedge_python,
            "node3_repo": node3_repo,
            "node2_repo": node2_repo,
        }.items()
        if value is not None
    }
    resolved_execution = resolve_execution(config, execution_overrides)
    manifest_layout = output_layout(config)
    manifest = _read_manifest(config, manifest_layout)
    if execution_overrides:
        manifest_execution = manifest.get("execution")
        if not isinstance(manifest_execution, dict):
            raise ValueError(
                "plan interpreter/repo overrides require a prepare manifest with "
                "resolved execution values; rerun prepare with the resolved config"
            )
        mismatches = {
            key: {"manifest": manifest_execution.get(key), "plan": resolved_execution[key]}
            for key in resolved_execution
            if str(manifest_execution.get(key)) != str(resolved_execution[key])
        }
        if mismatches:
            raise ValueError(
                "plan execution overrides differ from manifest; rerun prepare with the "
                f"resolved config: {json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
            )
    execution = resolved_execution
    layout = output_layout(config, execution=execution)
    run_id = config["run_id"]
    dataset = config["dataset"]["name"]
    generation = config["generation"]
    models = resolve_models(config)
    topology = config["topology"]
    node3_repo = execution["node3_repo"]
    node2_repo = execution["node2_repo"]
    node3_edge_python = execution["node3_edge_python"]
    node3_specedge_python = execution["node3_specedge_python"]
    node2_target_python = execution["node2_target_python"]
    node2_specedge_python = execution["node2_specedge_python"]
    draft_devices = list(topology.get("specedge_draft_devices", ["cpu"]))
    worker_count = int(topology.get("draft_workers", len(draft_devices)))
    batch_size = int(topology.get("batch_size", 4))
    token_budget = int(topology.get("token_budget", 512))
    draft_threads = int(topology.get("draft_threads", 32 if worker_count == 1 else 8))
    physical_cpu_count = int(topology.get("physical_cpu_count", 32))
    target_host = target_host or topology.get("target_host", "127.0.0.1")
    target_port = int(
        target_port if target_port is not None else topology.get("target_port", 18001)
    )
    warmup_requests = int(topology.get("warmup_requests", 0))
    raw_cpu_prefix = cpu_prefix if cpu_prefix is not None else topology.get("cpu_prefix", "")
    command_prefix, raw_cpu_prefix = shell_prefix(raw_cpu_prefix)
    cpu_environment = {
        "OMP_NUM_THREADS": draft_threads,
        "MKL_NUM_THREADS": draft_threads,
        "OPENBLAS_NUM_THREADS": draft_threads,
        "TORCH_NUM_THREADS": draft_threads,
    }
    target_url = f"http://{target_host}:{target_port}"
    target_bind_host = str(topology.get("target_bind_host", "127.0.0.1"))
    specedge_host = topology.get(
        "specedge_host",
        f"{target_host}:{int(topology.get('specedge_port', 18000))}",
    )
    specedge_bind_host = str(topology.get("specedge_bind_host", "127.0.0.1"))
    specedge_port = int(topology.get("specedge_port", 18000))
    draft_device = str(draft_devices[0] if draft_devices else "cpu")
    cpu_draft = draft_device.split(":", 1)[0].lower() == "cpu"
    common = (
        f"--dataset {dataset} --dataset_file {layout['node3_canonical']} "
        f"--workload_hash {manifest['workload_hash']} --draft_model {models['node3_draft']} "
        f"--target_model {models['target']} --max_tokens {generation['max_new_tokens']} "
        f"--temp {generation['temperature']} --top_k {generation['top_k']} "
        f"--top_p {generation['top_p']} --gamma {generation['gamma']} --seed {generation['seed']} "
        f"--verify_method {generation.get('verify_method', 'rejection')} "
        f"--draft_top_k {generation.get('draft_top_k', 64)} "
        f"--stop_policy {generation.get('stop_policy', 'eos')} "
        f"--warmup_requests {warmup_requests} --max_tasks_per_draft 0"
    )
    node2_integration = f"{node2_repo}/baselines/specedge/integration"
    node2_official = f"{node2_repo}/baselines/specedge/official"
    node3_integration = f"{node3_repo}/baselines/specedge/integration"
    node3_official = f"{node3_repo}/baselines/specedge/official"
    node3_pythonpath = ":".join(
        (node3_repo, node3_integration, f"{node3_official}/src")
    )
    specedge_raw = f"{layout['node3_root']}/specedge/raw/{run_id}"
    prepare_command = (
        f"cd {shlex.quote(node3_repo)} && {shlex.quote(node3_edge_python)} "
        f"scripts/eval_suite.py prepare --config {shlex.quote(str(config_path))}"
    )
    node2_prepare_command = (
        f"cd {shlex.quote(node2_repo)} && {shlex.quote(node2_target_python)} "
        f"scripts/eval_suite.py prepare --config {shlex.quote(str(config_path))}"
    )
    node2_hash_command = (
        f"cd {shlex.quote(node2_repo)} && {shlex.quote(node2_target_python)} "
        "scripts/eval_suite.py workload-hash "
        f"--manifest {layout['node2_root']}/run_manifest.json "
        f"--expected {shlex.quote(str(manifest['workload_hash']))}"
    )
    plan = "\n".join(
        [
            "[两台服务器：host-specific roots/interpreters；先生成完全相同的 workload 文件；不下载数据/模型]",
            f"# resolved_execution={json.dumps(execution, ensure_ascii=False, sort_keys=True)}",
            f"# resolved_models={json.dumps(models, ensure_ascii=False, sort_keys=True)}",
            f"# node3_specedge_config={layout['node3_specedge_config_linux']}",
            f"# node2_specedge_config={layout['node2_specedge_config_linux']}",
            f"# node3_specedge_pythonpath={json.dumps(node3_pythonpath, ensure_ascii=False)}",
            f"# cpu_prefix_raw={json.dumps(raw_cpu_prefix, ensure_ascii=False)}",
            f"# shared_load={str(bool(topology.get('shared_load', True))).lower()} "
            f"fixed_cpuset={json.dumps(str(topology.get('fixed_cpuset', '')), ensure_ascii=False)} "
            f"method_order={json.dumps(topology.get('method_order', list(FOUR_METHODS)), ensure_ascii=False)} "
            f"repeat_index={int(topology.get('repeat_index', 0))}",
            "\n[node3：prepare（node3 canonical + node3 SpecEdge YAML）]",
            prepare_command,
            "\n[node2：prepare（node2 canonical + node2 SpecEdge YAML；不得跳过）]",
            node2_prepare_command,
            "\n[两端 workload_hash 门禁：不一致则停止，不运行正式方法]",
            node2_hash_command,
            "\n[node3：CPU 绑定/背景负载预检（不假设 32 核独占；必须先于正式方法）]",
            f"cd {shlex.quote(node3_repo)} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(node3_edge_python)} scripts/preflight_cpu.py "
                f"--threads {draft_threads} --expected-cpu-count {physical_cpu_count} "
                f"--expected-cpuset {shlex.quote(str(topology.get('fixed_cpuset', '')))} --phase pre "
                f"--output {layout['node3_root']}/cpu_preflight.json "
                f"--commands-path {layout['node3_root']}/commands.txt",
                environment=cpu_environment,
            ),
            "\n[node2：FastSD target（target_host/port 与 bind host 可配置；物理 GPU 由外部 CUDA_VISIBLE_DEVICES 选择）]",
            f"cd {shlex.quote(node2_repo)} && CLOUD_SERVICE_HOST={shlex.quote(target_bind_host)} CLOUD_SERVICE_PORT={target_port} "
            f"FASTSD_TARGET_DEVICE={topology['standard_sd_target_device']} "
            f"{shlex.quote(node2_target_python)} cloud/cloud_service.py "
            f"--target_model {models['target']} --draft_model {models['node2_draft']} --dataset {dataset} "
            f"--server_sched_mode fastsd --kv_batch_mode varlen --batch_size {batch_size} "
            f"--token_budget {token_budget}",
            "\n[node3：FastSD stateful EdgeClient（/session/init + /prefill + /verify + rollback）]",
            f"cd {shlex.quote(node3_repo)} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(node3_edge_python)} edge/edge.py "
                f"--server_url {shlex.quote(target_url)} --profile custom "
                f"--server_sched_mode fastsd --kv_batch_mode varlen "
                f"--batch_size {batch_size} --token_budget {token_budget} "
                "--enable_latency_priority --overlap_prefill_first_draft "
                "--enable_proactive_draft --no-enable_pipeline "
                f"--exp_name comparison/{run_id}/fastsd {common} --num_drafts {worker_count} "
                f"--edge_gpus 1 {'--edge_use_cpu' if cpu_draft else '--no-edge_use_cpu'} --edge_threads {draft_threads} "
                f"--arrival_distribution {config['dataset']['arrival_distribution']} "
                f"--arrival_rate {config['dataset']['arrival_rate_rps']} "
                f"--arrival_seed {config['dataset']['arrival_seed']}",
                environment=cpu_environment,
            ),
            "\n[node2：SpecEdge target server（official target/tree verification core；bind host 可配置）]",
            f"cd {shlex.quote(node2_repo)} && FASTSD_EVAL_ROLE=server "
            f"FASTSD_EVAL_DATASET_FILE={layout['node2_canonical']} "
            f"PYTHONPATH={shlex.quote(node2_integration + ':' + node2_official + '/src')} "
            f"{shlex.quote(node2_specedge_python)} -O {node2_integration}/server.py "
            f"--config {layout['node2_specedge_config_linux']} --host {shlex.quote(specedge_bind_host)} --port {specedge_port}",
            "\n[node3：SpecEdge CPU adapter（official Tree/SpecExec/proactive semantics preserved）]",
            f"cd {shlex.quote(node3_repo)} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(node3_specedge_python)} {node3_repo}/baselines/specedge/integration/client_host.py "
                f"--config {layout['node3_specedge_config_linux']}",
                environment={**cpu_environment, "PYTHONPATH": node3_pythonpath},
            ),
            "\n[node2：停止 FastSD target 后，以 vanilla 调度重启 target（同一端口/bind host）]",
            f"cd {shlex.quote(node2_repo)} && CLOUD_SERVICE_HOST={shlex.quote(target_bind_host)} CLOUD_SERVICE_PORT={target_port} "
            f"FASTSD_TARGET_DEVICE={topology['standard_sd_target_device']} "
            f"{shlex.quote(node2_target_python)} cloud/cloud_service.py "
            f"--target_model {models['target']} --draft_model {models['node2_draft']} --dataset {dataset} "
            f"--server_sched_mode vanilla --kv_batch_mode varlen --batch_size {batch_size} "
            f"--token_budget {token_budget}",
            "\n[node3：标准投机解码（同样 CPU draft + 网络 + A6000 target）]",
            f"cd {shlex.quote(node3_repo)} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(node3_edge_python)} edge/edge.py "
                f"--server_url {shlex.quote(target_url)} --profile vanilla "
                f"--server_sched_mode vanilla --kv_batch_mode varlen "
                f"--batch_size {batch_size} --token_budget {token_budget} "
                f"--exp_name comparison/{run_id}/standard_sd {common} "
                f"--num_drafts {worker_count} --edge_gpus 1 "
                f"{'--edge_use_cpu' if cpu_draft else '--no-edge_use_cpu'} --edge_threads {draft_threads} "
                f"--arrival_distribution {config['dataset']['arrival_distribution']} "
                f"--arrival_rate {config['dataset']['arrival_rate_rps']} "
                f"--arrival_seed {config['dataset']['arrival_seed']}",
                environment=cpu_environment,
            ),
            "\n[node3：Draft-only（同一 CPU runtime；不访问 target）]",
            f"cd {shlex.quote(node3_repo)} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(node3_edge_python)} benchmark/eval_draft_pool.py --config {shlex.quote(str(config_path))}",
                environment={**cpu_environment, "PYTHON_BIN": node3_edge_python},
            ),
            "\n[node2：target-only greedy oracle（只用于 parity/quality reference，不纳入四方法）]",
            f"cd {shlex.quote(node2_repo)} && "
            f"PYTHONPATH={shlex.quote(node2_repo)}${{PYTHONPATH:+:$PYTHONPATH}} "
            f"{shlex.quote(node2_target_python)} benchmark/eval_target_only.py "
            f"--config {shlex.quote(str(config_path))}",
            "\n[网络拓扑说明：默认使用可覆盖的 target_host/port；受限网络可在 node3 建立双段 SSH 本地转发，"
            "让 127.0.0.1:18001 转发到 node2 127.0.0.1:18001。不要把转发命令或凭据写死到仓库。"
            "FastSD/SpecEdge 的请求耗时包含该网络路径，normalize 时保留 TTFT/E2E。]",
            "\n[node3：正式方法完成后记录 CPU 绑定/背景负载 postflight（保留 mpstat/load 快照）]",
            f"cd {shlex.quote(node3_repo)} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(node3_edge_python)} scripts/preflight_cpu.py "
                f"--threads {draft_threads} --expected-cpu-count {physical_cpu_count} "
                f"--expected-cpuset {shlex.quote(str(topology.get('fixed_cpuset', '')))} --phase post "
                f"--output {layout['node3_root']}/cpu_postflight.json "
                f"--commands-path {layout['node3_root']}/commands.txt",
                environment=cpu_environment,
            ),
            f"\n[method label: {'specedge_cpu_adapted' if cpu_draft else 'specedge'}; raw result expected at {specedge_raw}]",
        ]
    )
    print(plan)
    _append_command_record(layout["commands"], config_path, plan)
    append_status(
        layout["status"],
        method="suite",
        phase="plan",
        exit_code=0,
        command=(
            f"{shlex.quote(node3_edge_python)} scripts/eval_suite.py plan "
            f"--config {shlex.quote(str(config_path))}"
        ),
    )
    print(f"\nCommand record: {layout['commands']}")
    return 0


def _read_jsonl_files(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def normalize_fastsd(input_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict], float | None]:
    raw = _read_jsonl_files(sorted(input_dir.glob("edge_metrics_proc*.jsonl")))
    records = [
        {
            "sample_id": item.get("task_id"),
            "generated_tokens": item.get("generated_tokens", 0),
            "ttft_ms": item.get("ttft_ms"),
            "tpot_ms": item.get("tpot_ms"),
            "e2e_ms": item.get("request_e2e_ms"),
            "scheduled_arrival_s": item.get("scheduled_arrival_s"),
            "actual_arrival_s": item.get("actual_arrival_s"),
            "arrival_lag_ms": item.get("arrival_lag_ms"),
            "completion_s": item.get("completion_s"),
            "accepted_tokens": item.get("accepted_total", 0),
            "drafted_tokens": item.get("drafted_total", 0),
            "mean_accepted_tokens_per_verify": item.get(
                "mean_accepted_tokens_per_verify"
            ),
            "output_text": item.get("output_text"),
            "output_token_ids": item.get("output_token_ids", []),
            "workload_hash": item.get("workload_hash", manifest.get("workload_hash")),
            "network_rtt_ms": item.get("avg_transport_rtt_ms"),
            "network_bytes": item.get("network_bytes"),
            "request_bytes": item.get("request_bytes"),
            "response_bytes": item.get("response_bytes"),
            "rpc_count": item.get("rpc_count"),
            "kv_copy_ms": item.get("kv_copy_ms"),
            "kv_copy_bytes": item.get("kv_copy_bytes"),
            "reference": item.get("reference"),
        }
        for item in raw
    ]
    edge_summary = input_dir / "edge_metrics_summary.json"
    wallclock = None
    if edge_summary.is_file():
        summary_payload = json.loads(edge_summary.read_text(encoding="utf-8"))
        if summary_payload.get("wallclock_s") is not None:
            wallclock = float(summary_payload["wallclock_s"])
    return records, wallclock


def normalize_specedge(input_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict], float | None]:
    cycle_files = sorted(input_dir.glob("client_*.jsonl"))
    cycles = _read_jsonl_files(cycle_files)
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in cycles:
        # SpecEdge warmup cycles use negative request IDs and are deliberately
        # excluded from the formal workload while remaining in raw evidence.
        if int(item.get("req_idx", 0)) < 0:
            continue
        grouped[(int(item["client_idx"]), int(item["req_idx"]))].append(item)

    completion_root = input_dir.parents[1] / "requests"
    completions = _read_jsonl_files(sorted(completion_root.glob("client_*_requests.jsonl")))
    completion_by_global = {int(item["global_index"]): item for item in completions}
    records = []
    for (_client_idx, global_index), items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["step_idx"]))
        tokens = [int(item.get("num_accepted_tokens", 0)) for item in ordered]
        completion = completion_by_global.get(global_index, {})
        generated = int(completion.get("generated_tokens", sum(tokens)))
        records.append(
            {
                "sample_id": completion.get("sample_id", str(global_index)),
                "generated_tokens": generated,
                "ttft_ms": completion.get("ttft_ms"),
                "tpot_ms": completion.get("tpot_ms"),
                "e2e_ms": completion.get("request_e2e_ms"),
                "scheduled_arrival_s": completion.get("scheduled_arrival_s"),
                "actual_arrival_s": completion.get("actual_arrival_s"),
                "arrival_lag_ms": completion.get("arrival_lag_ms"),
                "completion_s": completion.get("completion_s"),
                "mean_accepted_tokens_per_verify": statistics.mean(tokens[1:]) if len(tokens) > 1 else None,
                "output_text": completion.get("output_text"),
                "output_token_ids": completion.get("output_token_ids", []),
                "workload_hash": completion.get("workload_hash", manifest.get("workload_hash")),
                "request_bytes": completion.get("request_bytes"),
                "response_bytes": completion.get("response_bytes"),
                "rpc_count": completion.get("rpc_count"),
                "reference": completion.get("reference"),
            }
        )
    return records, None


def normalize(config_path: str, method: str, input_path: str) -> int:
    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    source = Path(input_path)
    normalized_method = "specedge_cpu_adapted" if method == "specedge" else method
    if method == "fastsd":
        records, wallclock = normalize_fastsd(source, manifest)
    elif method in {"specedge", "specedge_cpu_adapted"}:
        records, wallclock = normalize_specedge(source, manifest)
    elif method == "standard_sd":
        records, wallclock = normalize_fastsd(source, manifest)
    elif method == "draft_only":
        records = _read_jsonl_files([source / "requests.jsonl"])
        wallclock = None
    else:
        raise ValueError(f"Unsupported method: {method}")
    output_dir = layout["local_root"] / "normalized" / normalized_method
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    write_text_once(
        requests_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    if config["dataset"]["name"] == "humaneval":
        humaneval_path = output_dir / "humaneval_samples.jsonl"
        write_text_once(
            humaneval_path,
            "".join(
                json.dumps(
                    {
                        "task_id": record["sample_id"],
                        "completion": record.get("output_text", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for record in records
            ),
        )
    target_records = []
    target_path = layout["local_root"] / "target_only" / "requests.jsonl"
    if target_path.is_file():
        target_records = _read_jsonl_files([target_path])
    summary = summarize_requests(
        records,
        method=normalized_method,
        dataset=config["dataset"]["name"],
        workload_hash=manifest["workload_hash"],
        run_id=config["run_id"],
        wallclock_s=wallclock,
        evaluation_scope=manifest.get("evaluation_scope", "quality"),
        paired_records=target_records or None,
        extra={
            "draft_model": config["models"]["draft"],
            "target_model": config["models"]["target"],
            "source_method": method,
        },
    )
    write_json_once(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    append_status(
        layout["status"],
        method=normalized_method,
        phase="normalize",
        exit_code=0,
    )
    return 0


def compare(summary_paths: list[str], output: str | None) -> int:
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in summary_paths]
    hashes = {summary["workload_hash"] for summary in summaries}
    if len(hashes) != 1:
        raise ValueError(f"workload hashes differ: {sorted(hashes)}")
    columns = [
        "method", "evaluation_scope", "num_requests", "throughput_tok_s", "goodput_tok_s",
        "ttft_avg_ms", "ttft_p95_ms",
        "scheduled_ttft_p95_ms",
        "tpot_avg_ms", "tpot_p95_ms", "e2e_avg_ms", "accept_rate",
        "mean_accepted_tokens_per_verify", "quality_exact_match",
        "request_bytes_total", "response_bytes_total", "rpc_count_total",
    ]
    rows = []
    for item in summaries:
        resources = item.get("resource_metrics", {})
        rows.append({
            "method": item["method"],
            "evaluation_scope": item.get("evaluation_scope", "unknown"),
            "num_requests": item["num_requests"],
            "throughput_tok_s": item["throughput_tok_s"],
            "goodput_tok_s": item.get("goodput_tok_s"),
            "ttft_avg_ms": item["ttft_ms"]["avg"],
            "ttft_p95_ms": item["ttft_ms"]["p95"],
            "scheduled_ttft_p95_ms": item["scheduled_ttft_ms"]["p95"],
            "tpot_avg_ms": item["tpot_ms"]["avg"],
            "tpot_p95_ms": item["tpot_ms"]["p95"],
            "e2e_avg_ms": item["e2e_ms"]["avg"],
            "accept_rate": item["accept_rate"],
            "mean_accepted_tokens_per_verify": item["mean_accepted_tokens_per_verify"],
            "quality_exact_match": item["quality_exact_match"],
            "request_bytes_total": resources.get("request_bytes", {}).get("total"),
            "response_bytes_total": resources.get("response_bytes", {}).get("total"),
            "rpc_count_total": resources.get("rpc_count", {}).get("total"),
        })
    print("\t".join(columns))
    for row in rows:
        print("\t".join("" if row[col] is None else str(row[col]) for col in columns))
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        write_text_once(output_path, buffer.getvalue())
    return 0


def _resource_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "[N/A]"}:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", ""))
    return float(match.group(0)) if match else None


def _resource_stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "avg": statistics.mean(values),
        "p95": percentile(values, 0.95),
        "sample_count": len(values),
    }


NVIDIA_SMI_CSV_SCHEMA_COLUMNS: dict[str, tuple[str, ...]] = {
    "full8": (
        "timestamp",
        "index",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.free",
        "power.draw",
    ),
    "compact-name6": (
        "timestamp",
        "index",
        "name",
        "utilization.gpu",
        "memory.used",
        "power.draw",
    ),
    "compact-no-name6": (
        "timestamp",
        "index",
        "utilization.gpu",
        "memory.used",
        "power.draw",
        "temperature.gpu",
    ),
}
NVIDIA_SMI_CSV_SCHEMA_CHOICES = tuple(NVIDIA_SMI_CSV_SCHEMA_COLUMNS)
NVIDIA_SMI_CSV_SCHEMA_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "full8": (
        ("utilization_gpu_pct", "utilization.gpu"),
        ("utilization_memory_pct", "utilization.memory"),
        ("memory_used_mib", "memory.used"),
        ("memory_free_mib", "memory.free"),
        ("power_draw_w", "power.draw"),
    ),
    "compact-name6": (
        ("utilization_gpu_pct", "utilization.gpu"),
        ("memory_used_mib", "memory.used"),
        ("power_draw_w", "power.draw"),
    ),
    "compact-no-name6": (
        ("utilization_gpu_pct", "utilization.gpu"),
        ("memory_used_mib", "memory.used"),
        ("power_draw_w", "power.draw"),
        ("temperature_gpu_c", "temperature.gpu"),
    ),
}
NVIDIA_SMI_RESOURCE_METRIC_NAMES = (
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
    "memory_free_mib",
    "power_draw_w",
    "temperature_gpu_c",
)


def _resolve_nvidia_smi_csv_schema(
    rows: list[tuple[int, list[str]]], requested_schema: str | None
) -> tuple[str, tuple[str, ...]]:
    if requested_schema is not None:
        schema_name = requested_schema.strip()
        if schema_name not in NVIDIA_SMI_CSV_SCHEMA_COLUMNS:
            choices = ", ".join(NVIDIA_SMI_CSV_SCHEMA_CHOICES)
            raise ValueError(
                f"unknown nvidia-smi CSV schema {requested_schema!r}; "
                f"choose one of: {choices}"
            )
        columns = NVIDIA_SMI_CSV_SCHEMA_COLUMNS[schema_name]
        expected_count = len(columns)
        for line_number, row in rows:
            if len(row) != expected_count:
                raise ValueError(
                    f"nvidia-smi CSV schema {schema_name!r} expects "
                    f"{expected_count} columns, but line {line_number} has "
                    f"{len(row)}"
                )
        return schema_name, columns

    if not rows:
        raise ValueError(
            "nvidia-smi CSV is empty; pass --gpu-csv-schema when the "
            "sample layout is not full8"
        )
    observed_counts = {len(row) for _, row in rows}
    if observed_counts == {8}:
        return "full8", NVIDIA_SMI_CSV_SCHEMA_COLUMNS["full8"]
    if 6 in observed_counts:
        if observed_counts == {6}:
            raise ValueError(
                "ambiguous six-column nvidia-smi CSV; pass "
                "--gpu-csv-schema compact-name6 or "
                "--gpu-csv-schema compact-no-name6"
            )
        raise ValueError(
            "inconsistent nvidia-smi CSV column counts including six-column "
            "rows; pass an explicit --gpu-csv-schema"
        )
    counts = ", ".join(str(count) for count in sorted(observed_counts))
    raise ValueError(
        "cannot infer nvidia-smi CSV schema from column counts "
        f"({counts}); only an unambiguous full8 sample may omit "
        "--gpu-csv-schema"
    )


def _parse_resource_core_filter(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            result.update(range(min(start, end), max(start, end) + 1))
        else:
            result.add(int(token))
    return result


def parse_mpstat_cpu_resource(
    text: str, *, cores: str | None = None
) -> dict[str, Any]:
    """Parse ``mpstat -P <cores> 1`` output without trusting its Average row."""

    expected_cores = _parse_resource_core_filter(cores)
    header_cpu_pos: int | None = None
    idle_offset: int | None = None
    numeric_rows: list[tuple[int, float]] = []
    all_rows: list[float] = []
    for raw_line in text.splitlines():
        tokens = raw_line.split()
        if "CPU" in tokens and "%idle" in tokens:
            header_cpu_pos = tokens.index("CPU")
            idle_offset = tokens.index("%idle") - header_cpu_pos
            continue
        if header_cpu_pos is None or idle_offset is None:
            continue
        if not tokens or tokens[0].lower().startswith("average"):
            continue
        core_pos = header_cpu_pos if header_cpu_pos < len(tokens) else None
        if core_pos is None or not (
            tokens[core_pos] == "all" or tokens[core_pos].isdigit()
        ):
            core_pos = next(
                (
                    index
                    for index, token in enumerate(tokens)
                    if token == "all" or token.isdigit()
                ),
                None,
            )
        if core_pos is None:
            continue
        idle_pos = core_pos + idle_offset
        if idle_pos < 0 or idle_pos >= len(tokens):
            continue
        idle = _resource_float(tokens[idle_pos])
        if idle is None or not 0.0 <= idle <= 100.0:
            continue
        utilization = 100.0 - idle
        core_token = tokens[core_pos]
        if core_token == "all":
            if expected_cores is None:
                all_rows.append(utilization)
            continue
        core = int(core_token)
        numeric_rows.append((core, utilization))

    actual_cores = {core for core, _ in numeric_rows}
    if expected_cores is not None:
        missing_cores = sorted(expected_cores - actual_cores)
        extra_cores = sorted(actual_cores - expected_cores)
        if missing_cores or extra_cores:
            raise ValueError(
                "mpstat CPU core set mismatch: "
                f"missing cores={missing_cores}; extra cores={extra_cores}"
            )
    selected_values = [
        value
        for core, value in numeric_rows
        if expected_cores is None or core in expected_cores
    ] or all_rows
    if not selected_values:
        raise ValueError("no parseable mpstat CPU samples")
    selected_cores = sorted({core for core, _ in numeric_rows})
    return {
        "source": "mpstat",
        "definition": "CPU utilization = 100 - %idle",
        "avg": statistics.mean(selected_values),
        "p95": percentile(selected_values, 0.95),
        "sample_count": len(selected_values),
        "core_count": len(selected_cores) if selected_cores else 1,
        "cores": selected_cores,
    }


def parse_nvidia_smi_csv_resource(
    text: str,
    *,
    gpu_index: int | str,
    gpu_uuid: str | None = None,
    gpu_csv_schema: str | None = None,
) -> dict[str, Any]:
    """Parse headerless ``nvidia-smi --format=csv,noheader`` samples.

    The full eight-column query is unambiguous and may omit ``gpu_csv_schema``.
    Either six-column layout must name its schema explicitly because the
    third column is either ``name`` or ``utilization.gpu``.  The CLI parses
    ``--gpu-index`` as an integer, but normalize direct callers too so index
    filtering cannot fail on a string/int mismatch.
    """

    requested_gpu_index = int(gpu_index)
    rows = [
        (line_number, row)
        for line_number, row in enumerate(csv.reader(text.splitlines()), start=1)
        if any(cell.strip() for cell in row)
    ]
    schema_name, columns = _resolve_nvidia_smi_csv_schema(rows, gpu_csv_schema)
    column_positions = {name: position for position, name in enumerate(columns)}
    metric_columns = NVIDIA_SMI_CSV_SCHEMA_METRICS[schema_name]
    values: dict[str, list[float]] = {
        name: [] for name in NVIDIA_SMI_RESOURCE_METRIC_NAMES
    }
    matched_rows = 0
    selected_uuid: str | None = None
    if gpu_uuid and "name" not in column_positions:
        raise ValueError(
            f"nvidia-smi CSV schema {schema_name!r} has no name column; "
            "--gpu-uuid cannot be used with this schema"
        )
    for _, row in rows:
        index = _resource_float(row[column_positions["index"]])
        if index is None or int(index) != requested_gpu_index:
            continue
        row_uuid = (
            row[column_positions["name"]].strip()
            if "name" in column_positions
            else None
        )
        if gpu_uuid and row_uuid != gpu_uuid:
            continue
        selected_uuid = selected_uuid or row_uuid
        matched_rows += 1
        for name, column_name in metric_columns:
            number = _resource_float(row[column_positions[column_name]])
            if number is not None:
                values[name].append(number)
    if matched_rows == 0:
        raise ValueError(
            f"no nvidia-smi rows matched gpu index {requested_gpu_index}"
        )
    return {
        "source": "nvidia-smi",
        "gpu_index": requested_gpu_index,
        "gpu_uuid": selected_uuid,
        "schema": schema_name,
        "columns": list(columns),
        "column_count": len(columns),
        "sample_count": matched_rows,
        **{name: _resource_stats(items) for name, items in values.items()},
    }


def resources(
    method_dir: str,
    *,
    cpu_mpstat: str | None = None,
    cpu_cores: str | None = None,
    gpu_nvidia_csv: str | None = None,
    gpu_index: int | None = None,
    gpu_uuid: str | None = None,
    gpu_csv_schema: str | None = None,
    output: str | None = None,
) -> int:
    """Create an auditable resource sidecar without modifying ``summary.json``."""

    if not cpu_mpstat and not gpu_nvidia_csv:
        raise ValueError("resources requires --cpu-mpstat and/or --gpu-nvidia-csv")
    if gpu_nvidia_csv is not None and gpu_index is None:
        raise ValueError("--gpu-index is required with --gpu-nvidia-csv")
    method_path = Path(method_dir)
    method_path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "method_dir": str(method_path)}
    if cpu_mpstat:
        payload["cpu"] = parse_mpstat_cpu_resource(
            Path(cpu_mpstat).read_text(encoding="utf-8", errors="replace"),
            cores=cpu_cores,
        )
    if gpu_nvidia_csv:
        payload["gpu"] = parse_nvidia_smi_csv_resource(
            Path(gpu_nvidia_csv).read_text(encoding="utf-8", errors="replace"),
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            gpu_csv_schema=gpu_csv_schema,
        )
    output_path = Path(output) if output else method_path / "resource_metrics.json"
    write_json_once(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _analysis_summary_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_analysis_input(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec:
        raise ValueError("analysis input must be DATASET/METHOD=SUMMARY_OR_DIR")
    label, raw_path = spec.split("=", 1)
    if "/" not in label:
        raise ValueError("analysis input must be DATASET/METHOD=SUMMARY_OR_DIR")
    dataset, method = label.split("/", 1)
    if dataset not in ANALYSIS_DATASETS:
        raise ValueError(f"unsupported analysis dataset: {dataset}")
    if method not in ANALYSIS_METHODS:
        raise ValueError(f"unsupported analysis method: {method}")
    return dataset, method, _analysis_summary_path(raw_path)


def _load_analysis_inputs(input_specs: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in input_specs:
        dataset, method, summary_path = _parse_analysis_input(spec)
        key = (dataset, method)
        if key in entries:
            raise ValueError(f"duplicate analysis input: {dataset}/{method}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reported_dataset = summary.get("dataset")
        if reported_dataset and reported_dataset != dataset:
            raise ValueError(
                f"summary dataset mismatch for {dataset}/{method}: {reported_dataset}"
            )
        reported_method = summary.get("method")
        if reported_method and not (
            reported_method == method
            or {reported_method, method} == {"specedge", "specedge_cpu_adapted"}
        ):
            raise ValueError(
                f"summary method mismatch for {dataset}/{method}: {reported_method}"
            )
        request_value = summary.get("requests_path") or summary.get("request_file")
        if request_value:
            request_path = Path(str(request_value))
            if not request_path.is_absolute():
                request_path = summary_path.parent / request_path
        else:
            request_path = summary_path.parent / "requests.jsonl"
        records = _read_jsonl_files([request_path]) if request_path.is_file() else []
        resource_value = summary.get("resource_metrics_path") or summary.get(
            "resource_path"
        )
        if resource_value:
            resource_path = Path(str(resource_value))
            if not resource_path.is_absolute():
                resource_path = summary_path.parent / resource_path
        else:
            resource_path = summary_path.parent / "resource_metrics.json"
        resources_payload = None
        if resource_path.is_file():
            resources_payload = json.loads(resource_path.read_text(encoding="utf-8"))
        entries[key] = {
            "dataset": dataset,
            "method": method,
            "summary_path": summary_path,
            "request_path": request_path,
            "resource_path": resource_path,
            "summary": summary,
            "records": records,
            "resources": resources_payload,
        }
    return entries


def _analysis_scalar(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("avg", "mean", "value"):
        if value.get(key) is not None:
            return value[key]
    return None


def _analysis_lookup(
    summary: dict[str, Any], names: tuple[str, ...], *, stat: str = "avg"
) -> Any:
    containers = [summary]
    for name in ("resource_metrics", "resources", "utilization", "resource_utilization"):
        value = summary.get(name)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                value = container[name]
                if isinstance(value, dict):
                    if value.get(stat) is not None:
                        return value[stat]
                    if stat == "avg":
                        return _analysis_scalar(value)
                elif stat == "avg" or name.endswith("_p95_pct"):
                    return value
    return None


def _analysis_resource(summary: dict[str, Any], key: str, stat: str) -> Any:
    for container_name in ("resource_metrics", "resources"):
        container = summary.get(container_name, {})
        value = container.get(key) if isinstance(container, dict) else None
        if isinstance(value, dict):
            if value.get(stat) is not None:
                return value[stat]
        elif value is not None:
            return value
    return None


def _analysis_sidecar_resource(
    resources_payload: dict[str, Any] | None, key: str, *, stat: str = "avg"
) -> Any:
    if not isinstance(resources_payload, dict):
        return None
    if key == "cpu_util_pct":
        cpu = resources_payload.get("cpu")
        return cpu.get(stat) if isinstance(cpu, dict) else None
    if key == "gpu_util_pct":
        gpu = resources_payload.get("gpu")
        metric = gpu.get("utilization_gpu_pct") if isinstance(gpu, dict) else None
        if isinstance(metric, dict):
            return metric.get(stat)
    return None


def _analysis_aggregate_resources(
    entries: dict[tuple[str, str], dict[str, Any]], method: str
) -> dict[str, Any] | None:
    cpu_weighted_sum = 0.0
    cpu_samples = 0
    gpu_weighted_sum = 0.0
    gpu_samples = 0
    for dataset in ANALYSIS_DATASETS:
        payload = entries.get((dataset, method), {}).get("resources")
        if not isinstance(payload, dict):
            continue
        cpu = payload.get("cpu", {})
        if isinstance(cpu, dict) and cpu.get("avg") is not None:
            sample_count = int(cpu.get("sample_count", 0) or 0)
            if sample_count > 0:
                cpu_weighted_sum += float(cpu["avg"]) * sample_count
                cpu_samples += sample_count
        gpu = payload.get("gpu", {})
        gpu_util = gpu.get("utilization_gpu_pct") if isinstance(gpu, dict) else None
        if isinstance(gpu_util, dict) and gpu_util.get("avg") is not None:
            sample_count = int(gpu_util.get("sample_count", 0) or 0)
            if sample_count > 0:
                gpu_weighted_sum += float(gpu_util["avg"]) * sample_count
                gpu_samples += sample_count
    if not cpu_samples and not gpu_samples:
        return None
    payload: dict[str, Any] = {}
    if cpu_samples:
        payload["cpu"] = {
            "avg": cpu_weighted_sum / cpu_samples,
            "sample_count": cpu_samples,
        }
    if gpu_samples:
        payload["gpu"] = {
            "utilization_gpu_pct": {
                "avg": gpu_weighted_sum / gpu_samples,
                "sample_count": gpu_samples,
            }
        }
    return payload


def _analysis_row_from_summary(
    summary: dict[str, Any] | None,
    *,
    scope: str,
    dataset: str,
    method: str,
    num_requests: int | float | None = None,
    resources_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or {}

    def latency(name: str, stat: str) -> Any:
        value = summary.get(name, {})
        return value.get(stat) if isinstance(value, dict) else None

    request_count = summary.get("num_requests") if num_requests is None else num_requests
    request_bytes_total = _analysis_resource(summary, "request_bytes", "total")
    response_bytes_total = _analysis_resource(summary, "response_bytes", "total")
    rpc_count_total = _analysis_resource(summary, "rpc_count", "total")

    def per_request(value: Any) -> float | None:
        if value is None or request_count in (None, 0):
            return None
        return float(value) / float(request_count)

    cpu_util = _analysis_sidecar_resource(resources_payload, "cpu_util_pct")
    if cpu_util is None:
        cpu_util = _analysis_lookup(
            summary,
            ("cpu_util_pct", "cpu_utilization_pct", "cpu_utilization", "cpu_percent"),
        )
    cpu_util_p95 = _analysis_sidecar_resource(
        resources_payload, "cpu_util_pct", stat="p95"
    )
    if cpu_util_p95 is None:
        cpu_util_p95 = _analysis_lookup(
            summary,
            (
                "cpu_util_p95_pct",
                "cpu_utilization_p95_pct",
                "cpu_p95_pct",
                "cpu_util_pct",
                "cpu_utilization_pct",
                "cpu_utilization",
                "cpu_percent",
            ),
            stat="p95",
        )
    gpu_util = _analysis_sidecar_resource(resources_payload, "gpu_util_pct")
    if gpu_util is None:
        gpu_util = _analysis_lookup(
            summary,
            ("gpu_util_pct", "gpu_utilization_pct", "gpu_utilization", "gpu_percent"),
        )
    gpu_util_p95 = _analysis_sidecar_resource(
        resources_payload, "gpu_util_pct", stat="p95"
    )
    if gpu_util_p95 is None:
        gpu_util_p95 = _analysis_lookup(
            summary,
            (
                "gpu_util_p95_pct",
                "gpu_utilization_p95_pct",
                "gpu_p95_pct",
                "gpu_util_pct",
                "gpu_utilization_pct",
                "gpu_utilization",
                "gpu_percent",
            ),
            stat="p95",
        )

    return {
        "scope": scope,
        "dataset": dataset,
        "method": method,
        "num_requests": request_count,
        "throughput_tok_s": summary.get("throughput_tok_s"),
        "goodput_tok_s": summary.get("goodput_tok_s"),
        "goodput_req_s": summary.get("goodput_req_s"),
        "ttft_avg_ms": latency("ttft_ms", "avg"),
        "ttft_p95_ms": latency("ttft_ms", "p95"),
        "tpot_avg_ms": latency("tpot_ms", "avg"),
        "tpot_p95_ms": latency("tpot_ms", "p95"),
        "e2e_avg_ms": latency("e2e_ms", "avg"),
        "e2e_p95_ms": latency("e2e_ms", "p95"),
        "accept_rate": summary.get("accept_rate"),
        "mean_accepted_tokens_per_verify": summary.get(
            "mean_accepted_tokens_per_verify"
        ),
        "network_rtt_ms_avg": _analysis_resource(summary, "network_rtt_ms", "avg"),
        "request_bytes_total": request_bytes_total,
        "request_bytes_per_request": per_request(request_bytes_total),
        "response_bytes_total": response_bytes_total,
        "response_bytes_per_request": per_request(response_bytes_total),
        "rpc_count_total": rpc_count_total,
        "rpc_count_per_request": per_request(rpc_count_total),
        "cpu_util_pct": cpu_util,
        "cpu_util_p95_pct": cpu_util_p95,
        "gpu_util_pct": gpu_util,
        "gpu_util_p95_pct": gpu_util_p95,
    }


ANALYSIS_PERFORMANCE_FIELDS = (
    "throughput_tok_s",
    "goodput_tok_s",
    "goodput_req_s",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "tpot_avg_ms",
    "tpot_p95_ms",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "accept_rate",
    "mean_accepted_tokens_per_verify",
    "network_rtt_ms_avg",
    "request_bytes_total",
    "request_bytes_per_request",
    "response_bytes_total",
    "response_bytes_per_request",
    "rpc_count_total",
    "rpc_count_per_request",
    "cpu_util_pct",
    "cpu_util_p95_pct",
    "gpu_util_pct",
    "gpu_util_p95_pct",
)


def _analysis_per_dataset_row(
    entries: dict[tuple[str, str], dict[str, Any]], dataset: str, method: str
) -> dict[str, Any]:
    entry = entries.get((dataset, method))
    if entry is None:
        return _analysis_row_from_summary(
            None, scope="per-dataset", dataset=dataset, method=method
        )
    return _analysis_row_from_summary(
        entry["summary"],
        scope="per-dataset",
        dataset=dataset,
        method=method,
        num_requests=entry["summary"].get("num_requests", len(entry["records"])),
        resources_payload=entry.get("resources"),
    )


def _analysis_macro_row(
    rows: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    result = {
        "scope": "macro",
        "dataset": "all",
        "method": method,
        "num_requests": None,
    }
    for field in ANALYSIS_PERFORMANCE_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = statistics.mean(values) if values else None
    for field in (
        "request_bytes_total",
        "response_bytes_total",
        "rpc_count_total",
    ):
        result[field] = None
    counts = [float(row["num_requests"]) for row in rows if row.get("num_requests") is not None]
    result["num_requests"] = statistics.mean(counts) if counts else None
    return result


def _analysis_micro_row(
    entries: dict[tuple[str, str], dict[str, Any]], method: str
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    wallclocks: list[float] = []
    present = 0
    for dataset in ANALYSIS_DATASETS:
        entry = entries.get((dataset, method))
        if entry is None:
            continue
        present += 1
        records.extend(dict(record) for record in entry["records"])
        value = entry["summary"].get("wallclock_s")
        if value is not None:
            wallclocks.append(float(value))
    if not records:
        return _analysis_row_from_summary(
            None, scope="micro", dataset="all", method=method
        )
    wallclock = sum(wallclocks) if present == len(wallclocks) else None
    summary = summarize_requests(
        records,
        method=method,
        dataset="micro",
        workload_hash="mixed-datasets",
        run_id=f"micro-{method}",
        wallclock_s=wallclock,
    )
    return _analysis_row_from_summary(
        summary,
        scope="micro",
        dataset="all",
        method=method,
        num_requests=len(records),
        resources_payload=_analysis_aggregate_resources(entries, method),
    )


def _analysis_index(
    records: list[dict[str, Any]], dataset: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        if record.get("sample_id") is None:
            continue
        key = f"{dataset}::{record['sample_id']}"
        if key in indexed:
            duplicates.append(key)
        else:
            indexed[key] = record
    return indexed, duplicates


def _analysis_empty_pair(reason: str) -> dict[str, Any]:
    return {
        "pair_count": 0,
        "method_count": 0,
        "baseline_count": 0,
        "missing_method_count": 0,
        "missing_baseline_count": 0,
        "duplicate_method_count": 0,
        "duplicate_baseline_count": 0,
        "reason": reason,
        "metrics": {
            key: {
                "n": 0,
                "speedup_x": None,
                "speedup_pct": None,
                "ci95_x": None,
                "iterations": ANALYSIS_BOOTSTRAP_ITERATIONS,
                "seed": ANALYSIS_BOOTSTRAP_SEED,
            }
            for key in ANALYSIS_SPEEDUP_METRICS
        },
        "_pairs": {},
    }


def _analysis_speedup_ratio(
    pairs: list[tuple[float, float]], direction: str
) -> float | None:
    if not pairs:
        return None
    method_mean = statistics.mean(left for left, _ in pairs)
    baseline_mean = statistics.mean(right for _, right in pairs)
    if direction == "lower_is_better":
        return baseline_mean / method_mean if method_mean > 0 else None
    return method_mean / baseline_mean if baseline_mean > 0 else None


def _analysis_speedup_ci(
    pairs: list[tuple[float, float]], direction: str
) -> dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "speedup_x": None,
            "speedup_pct": None,
            "ci95_x": None,
            "iterations": ANALYSIS_BOOTSTRAP_ITERATIONS,
            "seed": ANALYSIS_BOOTSTRAP_SEED,
        }

    point = _analysis_speedup_ratio(pairs, direction)
    if point is None:
        return {
            "n": len(pairs),
            "speedup_x": None,
            "speedup_pct": None,
            "ci95_x": None,
            "iterations": ANALYSIS_BOOTSTRAP_ITERATIONS,
            "seed": ANALYSIS_BOOTSTRAP_SEED,
        }
    rng = random.Random(ANALYSIS_BOOTSTRAP_SEED)
    bootstrapped: list[float] = []
    for _ in range(ANALYSIS_BOOTSTRAP_ITERATIONS):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = _analysis_speedup_ratio(sample, direction)
        if value is not None:
            bootstrapped.append(value)
    interval = (
        [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)]
        if bootstrapped
        else None
    )
    return {
        "n": len(pairs),
        "speedup_x": point,
        "speedup_pct": (point - 1.0) * 100.0,
        "ci95_x": interval,
        "iterations": ANALYSIS_BOOTSTRAP_ITERATIONS,
        "seed": ANALYSIS_BOOTSTRAP_SEED,
    }


def _analysis_pair_records(
    method_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    dataset: str,
) -> dict[str, Any]:
    method_index, method_duplicates = _analysis_index(method_records, dataset)
    baseline_index, baseline_duplicates = _analysis_index(baseline_records, dataset)
    if method_duplicates or baseline_duplicates:
        result = _analysis_empty_pair("duplicate sample_id; pairing refused")
        result.update(
            {
                "method_count": len(method_index),
                "baseline_count": len(baseline_index),
                "duplicate_method_count": len(method_duplicates),
                "duplicate_baseline_count": len(baseline_duplicates),
            }
        )
        return result
    method_ids = set(method_index)
    baseline_ids = set(baseline_index)
    common_ids = sorted(method_ids & baseline_ids)
    result = {
        "pair_count": len(common_ids),
        "method_count": len(method_ids),
        "baseline_count": len(baseline_ids),
        "missing_method_count": len(baseline_ids - method_ids),
        "missing_baseline_count": len(method_ids - baseline_ids),
        "duplicate_method_count": 0,
        "duplicate_baseline_count": 0,
        "reason": None,
        "metrics": {},
        "_pairs": {},
    }
    for key, direction in ANALYSIS_SPEEDUP_METRICS.items():
        pairs = []
        for sample_id in common_ids:
            left = method_index[sample_id].get(key)
            right = baseline_index[sample_id].get(key)
            if left is None or right is None:
                continue
            pairs.append((float(left), float(right)))
        result["metrics"][key] = _analysis_speedup_ci(pairs, direction)
        result["_pairs"][key] = pairs
    return result


def _analysis_combined_records(
    entries: dict[tuple[str, str], dict[str, Any]],
    method: str,
    datasets: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        entry = entries.get((dataset, method))
        if entry is None:
            continue
        for record in entry["records"]:
            copied = dict(record)
            copied["sample_id"] = f"{dataset}::{record.get('sample_id')}"
            records.append(copied)
    return records


def _analysis_macro_pair(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return _analysis_empty_pair("no valid dataset pairs")
    result = {
        "pair_count": sum(part["pair_count"] for part in parts),
        "method_count": sum(part["method_count"] for part in parts),
        "baseline_count": sum(part["baseline_count"] for part in parts),
        "missing_method_count": sum(part["missing_method_count"] for part in parts),
        "missing_baseline_count": sum(part["missing_baseline_count"] for part in parts),
        "duplicate_method_count": sum(part["duplicate_method_count"] for part in parts),
        "duplicate_baseline_count": sum(part["duplicate_baseline_count"] for part in parts),
        "reason": next((part["reason"] for part in parts if part["reason"]), None),
        "metrics": {},
    }
    for key in ANALYSIS_SPEEDUP_METRICS:
        usable_parts = [
            part
            for part in parts
            if part["metrics"][key]["speedup_x"] is not None
            and part.get("_pairs", {}).get(key)
        ]
        if not usable_parts:
            result["metrics"][key] = _analysis_speedup_ci([], ANALYSIS_SPEEDUP_METRICS[key])
            continue
        direction = ANALYSIS_SPEEDUP_METRICS[key]
        macro_units = [part["metrics"][key]["speedup_x"] for part in usable_parts]
        speedup = statistics.mean(macro_units)
        rng = random.Random(ANALYSIS_BOOTSTRAP_SEED)
        macro_bootstrap: list[float] = []
        for _ in range(ANALYSIS_BOOTSTRAP_ITERATIONS):
            dataset_speedups: list[float] = []
            for part in usable_parts:
                pairs = part["_pairs"][key]
                resampled_pairs = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                value = _analysis_speedup_ratio(resampled_pairs, direction)
                if value is not None:
                    dataset_speedups.append(value)
            if dataset_speedups:
                macro_bootstrap.append(statistics.mean(dataset_speedups))
        ci = [
            percentile(macro_bootstrap, 0.025),
            percentile(macro_bootstrap, 0.975),
        ]
        result["metrics"][key] = {
            "n": sum(part["metrics"][key]["n"] for part in usable_parts),
            "speedup_x": speedup,
            "speedup_pct": (speedup - 1.0) * 100.0,
            "ci95_x": ci,
            "iterations": ANALYSIS_BOOTSTRAP_ITERATIONS,
            "seed": ANALYSIS_BOOTSTRAP_SEED,
        }
    return result


def _analysis_pair_for_scope(
    entries: dict[tuple[str, str], dict[str, Any]],
    *,
    scope: str,
    dataset: str,
    method: str,
    baseline: str,
    valid_datasets: set[str],
) -> dict[str, Any]:
    if scope == "per-dataset":
        if dataset not in valid_datasets:
            return _analysis_empty_pair("workload_hash mismatch or missing")
        left = entries.get((dataset, method), {}).get("records", [])
        right = entries.get((dataset, baseline), {}).get("records", [])
        return _analysis_pair_records(left, right, dataset=dataset)
    datasets = tuple(sorted(valid_datasets))
    if scope == "macro":
        parts = []
        for item in datasets:
            left = entries.get((item, method), {}).get("records", [])
            right = entries.get((item, baseline), {}).get("records", [])
            parts.append(_analysis_pair_records(left, right, dataset=item))
        return _analysis_macro_pair(parts)
    left = _analysis_combined_records(entries, method, datasets)
    right = _analysis_combined_records(entries, baseline, datasets)
    return _analysis_pair_records(left, right, dataset="all")


def _analysis_quality_row(
    entries: dict[tuple[str, str], dict[str, Any]], dataset: str, method: str
) -> dict[str, Any]:
    entry = entries.get((dataset, method))
    summary = entry["summary"] if entry else {}
    if dataset in {"gsm8k", "mgsm"}:
        metric = f"{dataset.upper()} exact match"
        value = summary.get("quality_exact_match")
        evidence = (
            "normalized output_text/reference numeric answer extractor"
            if value is not None
            else "N/A: missing normalized output_text/reference"
        )
    elif dataset == "humaneval":
        metric = "HumanEval pass@1"
        value = _analysis_lookup(
            summary, ("humaneval_pass_at_1", "quality_pass_at_1", "pass_at_1")
        )
        evidence_payload = summary.get("quality_evidence", {})
        isolated = summary.get("humaneval_isolated_execution") is True
        if isinstance(evidence_payload, dict):
            humaneval_evidence = evidence_payload.get("humaneval", evidence_payload.get("pass_at_1", {}))
            isolated = isolated or (
                isinstance(humaneval_evidence, dict)
                and humaneval_evidence.get("isolated_execution") is True
            )
        if value is None or not isolated:
            value = None
            evidence = "N/A: no explicit isolated HumanEval executor evidence"
        else:
            evidence = "isolated HumanEval executor evidence declared in summary"
    else:
        metric = "MT-Bench judge score"
        value = None
        policy = summary.get("mt_bench_turn_policy", "first_turn_only")
        evidence = f"N/A: system benchmark, policy={policy}; no judge score"
    return {
        "dataset": dataset,
        "method": method,
        "metric": metric,
        "value": value,
        "mt_bench_policy": (
            summary.get("mt_bench_turn_policy", "first_turn_only")
            if dataset == "mt_bench"
            else "N/A"
        ),
        "evidence": evidence,
    }


def _analysis_format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _analysis_speedup_format(metric: dict[str, Any]) -> str:
    point = metric.get("speedup_x")
    interval = metric.get("ci95_x")
    if point is None:
        return "N/A"
    if interval is None or interval[0] is None or interval[1] is None:
        return f"{point:.4g}x"
    return f"{point:.4g}x [{interval[0]:.4g}, {interval[1]:.4g}]"


def _analysis_markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_analysis_format(value) for value in row) + " |")
    return lines


def _analysis_nonempty_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _analysis_resolve_comparisons(
    *,
    focal_method: str,
    baseline: str | None,
    baselines: list[str] | None,
) -> list[str]:
    if focal_method not in ANALYSIS_METHODS:
        raise ValueError(f"unsupported analysis focal method: {focal_method}")
    if baselines is not None and baseline is not None:
        raise ValueError("use baseline or baselines, not both")
    selected = (
        list(baselines)
        if baselines is not None
        else ([baseline] if baseline is not None else [
            method for method in ANALYSIS_METHODS if method != focal_method
        ])
    )
    if not selected:
        raise ValueError("analysis comparison requires at least one baseline")
    if len(set(selected)) != len(selected):
        raise ValueError("analysis baselines must be unique")
    for item in selected:
        if item not in ANALYSIS_METHODS:
            raise ValueError(f"unsupported analysis baseline: {item}")
        if item == focal_method:
            raise ValueError("analysis focal method cannot be its own baseline")
    return selected


def build_analysis_report(
    input_specs: list[str],
    *,
    baseline: str | None = None,
    focal_method: str = "fastsd",
    baselines: list[str] | None = None,
) -> dict[str, Any]:
    comparison_baselines = _analysis_resolve_comparisons(
        focal_method=focal_method,
        baseline=baseline,
        baselines=baselines,
    )
    entries = _load_analysis_inputs(input_specs)
    issues: list[str] = []
    valid_datasets: set[str] = set()
    for dataset in ANALYSIS_DATASETS:
        missing = [method for method in ANALYSIS_METHODS if (dataset, method) not in entries]
        if missing:
            issues.append(f"{dataset}: missing inputs={','.join(missing)}")
            continue
        summary_hashes = {
            method: _analysis_nonempty_hash(
                entries[(dataset, method)]["summary"].get("workload_hash")
            )
            for method in ANALYSIS_METHODS
        }
        missing_hashes = [method for method, value in summary_hashes.items() if value is None]
        if missing_hashes:
            issues.append(
                f"{dataset}: workload_hash missing for methods={','.join(missing_hashes)}; "
                "pairwise speedup blocked"
            )
            continue
        unique_hashes = set(summary_hashes.values())
        if len(unique_hashes) != 1:
            issues.append(f"{dataset}: workload_hash mismatch; pairwise speedup blocked")
            continue
        summary_hash = next(iter(unique_hashes))
        request_hashes = {
            record_hash
            for method in ANALYSIS_METHODS
            for record in entries[(dataset, method)]["records"]
            if (record_hash := _analysis_nonempty_hash(record.get("workload_hash")))
        }
        conflicting_request_hashes = request_hashes - {summary_hash}
        if conflicting_request_hashes:
            issues.append(
                f"{dataset}: request workload_hash conflict with summary hash "
                f"({', '.join(sorted(conflicting_request_hashes))}); pairwise speedup blocked"
            )
            continue
        valid_datasets.add(dataset)

    performance_rows: list[dict[str, Any]] = []
    for dataset in ANALYSIS_DATASETS:
        for method in ANALYSIS_METHODS:
            performance_rows.append(_analysis_per_dataset_row(entries, dataset, method))
    for method in ANALYSIS_METHODS:
        dataset_rows = [
            row for row in performance_rows if row["method"] == method
        ]
        performance_rows.append(_analysis_macro_row(dataset_rows, method))
        performance_rows.append(_analysis_micro_row(entries, method))

    speedup_rows: list[dict[str, Any]] = []
    for dataset in ANALYSIS_DATASETS:
        for comparison_baseline in comparison_baselines:
            pair = _analysis_pair_for_scope(
                entries,
                scope="per-dataset",
                dataset=dataset,
                method=focal_method,
                baseline=comparison_baseline,
                valid_datasets=valid_datasets,
            )
            speedup_rows.append(
                {
                    "scope": "per-dataset",
                    "dataset": dataset,
                    "method": focal_method,
                    "baseline": comparison_baseline,
                    **pair,
                }
            )
    for comparison_baseline in comparison_baselines:
        for scope in ("macro", "micro"):
            pair = _analysis_pair_for_scope(
                entries,
                scope=scope,
                dataset="all",
                method=focal_method,
                baseline=comparison_baseline,
                valid_datasets=valid_datasets,
            )
            speedup_rows.append(
                {
                    "scope": scope,
                    "dataset": "all",
                    "method": focal_method,
                    "baseline": comparison_baseline,
                    **pair,
                }
            )

    quality_rows = [
        _analysis_quality_row(entries, dataset, method)
        for dataset in ANALYSIS_DATASETS
        for method in ANALYSIS_METHODS
    ]
    missing_cells = []
    for row in performance_rows:
        for field in ANALYSIS_PERFORMANCE_FIELDS:
            if row.get(field) is None:
                missing_cells.append((row["scope"], row["dataset"], row["method"], field))
    markdown_lines = [
        "# FastSD 四数据集四方法分析报告",
        "",
        "## 审计元数据",
        "",
        f"- 输入：{len(entries)}/{len(ANALYSIS_DATASETS) * len(ANALYSIS_METHODS)} 个 dataset/method 组合",
        f"- 配对键：`dataset + sample_id`；直接比较：`{focal_method}` vs "
        f"{', '.join(comparison_baselines)}",
        f"- paired bootstrap：seed=`{ANALYSIS_BOOTSTRAP_SEED}`，iterations=`{ANALYSIS_BOOTSTRAP_ITERATIONS:,}`，95% percentile CI",
        "- macro CI：每个 dataset 内对配对样本重采样，再对 dataset speedup 等权平均；不是对四个 point speedup 重采样",
        "- 通信量：per-dataset 保留 totals 与 per-request；macro totals 为 `N/A`，micro totals 为所有请求之和",
        "- 资源：per-dataset CPU/GPU avg 与 p95 来自 sidecar；macro p95 为各 dataset p95 等权均值；micro 无 raw sidecar 合并时 p95 为 `N/A`",
        "- speedup 定义：延迟指标为 `baseline_mean / method_mean`；缺失、重复或 workload hash 不一致时为 `N/A`",
        "- 缺失数据、未采集资源和未声明隔离质量证据均显示为 `N/A`，不进行推断",
        "",
        "## 性能、吞吐、接受率、网络与资源",
        "",
    ]
    performance_headers = [
        "scope", "dataset", "method", "requests", "throughput tok/s", "goodput tok/s",
        "goodput req/s", "TTFT avg ms", "TTFT p95 ms", "TPOT avg ms", "TPOT p95 ms",
        "E2E avg ms", "E2E p95 ms", "accept rate", "accepted/verify", "RTT avg ms",
        "request bytes total", "request bytes/req", "response bytes total", "response bytes/req",
        "RPC total", "RPC/req", "CPU util avg %", "CPU util p95 %",
        "GPU util avg %", "GPU util p95 %",
    ]
    performance_keys = [
        "scope", "dataset", "method", "num_requests", *ANALYSIS_PERFORMANCE_FIELDS
    ]
    markdown_lines.extend(
        _analysis_markdown_table(
            performance_headers,
            [[row.get(key) for key in performance_keys] for row in performance_rows],
        )
    )
    markdown_lines.extend(
        [
            "",
            "## 按 sample_id 成对 speedup（含 10,000 次 bootstrap 95% CI）",
            "",
            "CI 单位为 speedup 倍数；`pair_count` 是共有 sample_id 数，括号内为各指标实际可计算的 pair 数。",
            "",
        ]
    )
    speedup_headers = [
        "scope", "dataset", "method", "baseline", "pair count", "E2E speedup x [95% CI]",
        "TTFT speedup x [95% CI]", "TPOT speedup x [95% CI]", "metric pair counts", "coverage",
    ]
    speedup_table = []
    for row in speedup_rows:
        metric_counts = "; ".join(
            f"{key}={row['metrics'][key]['n']}" for key in ANALYSIS_SPEEDUP_METRICS
        )
        coverage = (
            f"method={row['method_count']}, baseline={row['baseline_count']}, "
            f"missing_method={row['missing_method_count']}, missing_baseline={row['missing_baseline_count']}"
        )
        if row.get("reason"):
            coverage += f"; {row['reason']}"
        speedup_table.append(
            [
                row["scope"], row["dataset"], row["method"], row["baseline"], row["pair_count"],
                _analysis_speedup_format(row["metrics"]["e2e_ms"]),
                _analysis_speedup_format(row["metrics"]["ttft_ms"]),
                _analysis_speedup_format(row["metrics"]["tpot_ms"]),
                metric_counts,
                coverage,
            ]
        )
    markdown_lines.extend(_analysis_markdown_table(speedup_headers, speedup_table))
    markdown_lines.extend(["", "## 质量指标与证据边界", ""])
    quality_headers = ["dataset", "method", "metric", "value", "MT-Bench policy", "evidence boundary"]
    markdown_lines.extend(
        _analysis_markdown_table(
            quality_headers,
            [
                [
                    row["dataset"], row["method"], row["metric"], row["value"],
                    row["mt_bench_policy"], row["evidence"],
                ]
                for row in quality_rows
            ],
        )
    )
    markdown_lines.extend(["", "## 缺失与阻断项", ""])
    if issues:
        markdown_lines.extend(f"- {issue}" for issue in issues)
    else:
        markdown_lines.append("- 无输入结构或 workload hash 阻断项")
    markdown_lines.append(f"- 性能资源表中的 N/A 单元格：{len(missing_cells)}")
    if missing_cells:
        markdown_lines.append("- 示例缺失项（最多 40 条）：")
        for scope, dataset, method, field in missing_cells[:40]:
            markdown_lines.append(f"  - `{scope}/{dataset}/{method}/{field}` = N/A")
    markdown_lines.extend(
        [
            "",
            "## 质量解释",
            "",
            "- GSM8K/MGSM 的 exact match 仅表示当前 normalized `output_text/reference` 数值抽取结果。",
            "- HumanEval pass@1 只有在 summary 明确声明 isolated executor evidence 时才接受；否则保持 N/A。",
            "- MT-Bench 当前 canonical 只使用 first turn，报告不将其伪称为完整 judge 质量分数。",
            "",
        ]
    )
    return {
        "entries": entries,
        "issues": issues,
        "valid_datasets": sorted(valid_datasets),
        "performance_rows": performance_rows,
        "speedup_rows": speedup_rows,
        "focal_method": focal_method,
        "baselines": comparison_baselines,
        "quality_rows": quality_rows,
        "missing_cells": missing_cells,
        "markdown": "\n".join(markdown_lines),
    }


def analyze(
    input_specs: list[str],
    output: str,
    *,
    baseline: str | None = None,
    focal_method: str = "fastsd",
    baselines: list[str] | None = None,
) -> int:
    report = build_analysis_report(
        input_specs,
        baseline=baseline,
        focal_method=focal_method,
        baselines=baselines,
    )
    output_path = Path(output)
    write_text_once(output_path, report["markdown"] + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "input_count": len(report["entries"]),
                "valid_datasets": report["valid_datasets"],
                "missing_cells": len(report["missing_cells"]),
                "issues": report["issues"],
                "bootstrap_iterations": ANALYSIS_BOOTSTRAP_ITERATIONS,
                "focal_method": report["focal_method"],
                "baselines": report["baselines"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _request_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "requests.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_method_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"method input must be METHOD=PATH: {spec}")
    method, raw_path = spec.split("=", 1)
    if not method:
        raise ValueError(f"method input has an empty method: {spec}")
    return method, _request_path(raw_path)


def paired_analysis(
    config_path: str,
    left_spec: str,
    right_spec: str,
    *,
    output: str | None = None,
) -> int:
    """Write direct paired FastSD-vs-SpecEdge bootstrap evidence."""

    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    left_method, left_path = _parse_method_path(left_spec)
    right_method, right_path = _parse_method_path(right_spec)
    left_records = _read_jsonl_files([left_path])
    right_records = _read_jsonl_files([right_path])
    workload_hashes = {
        str(record.get("workload_hash"))
        for records in (left_records, right_records)
        for record in records
        if record.get("workload_hash") is not None
    }
    if workload_hashes and workload_hashes != {manifest["workload_hash"]}:
        raise ValueError(
            f"paired workload hashes differ from manifest: {sorted(workload_hashes)}"
        )
    payload = {
        "run_id": config["run_id"],
        "workload_hash": manifest["workload_hash"],
        "left": {"method": left_method, "path": str(left_path), "samples": len(left_records)},
        "right": {"method": right_method, "path": str(right_path), "samples": len(right_records)},
        "analysis": paired_method_analysis(
            left_records,
            right_records,
            method=left_method,
            baseline=right_method,
            seed=42,
            iterations=10_000,
        ),
    }
    output_path = (
        Path(output)
        if output
        else layout["local_root"] / "analysis" / "fastsd_vs_specedge_cpu_adapted.json"
    )
    write_json_once(output_path, payload)
    append_status(
        layout["status"],
        method=f"{left_method}_vs_{right_method}",
        phase="paired_analysis",
        exit_code=0,
        command=f"paired --config {config_path} --left {left_spec} --right {right_spec}",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def parity(
    config_path: str,
    input_specs: list[str],
    *,
    reference: str = ORACLE_METHOD,
    output: str | None = None,
    max_samples: int | None = None,
) -> int:
    """Create per-token parity JSONL from normalized/raw request files."""

    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    paths: dict[str, Path] = {}
    for spec in input_specs:
        if "=" not in spec:
            raise ValueError(f"parity input must be METHOD=PATH: {spec}")
        method, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / "requests.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[method] = path
    if reference not in paths:
        oracle_path = layout["local_root"] / ORACLE_METHOD / "requests.jsonl"
        if oracle_path.is_file():
            paths[reference] = oracle_path
    if reference not in paths:
        raise FileNotFoundError(
            f"reference {reference!r} is missing; pass --input {reference}=..."
        )
    records_by_method = {
        method: _read_jsonl_files([path]) for method, path in paths.items()
    }
    workload_hashes = {
        str(record.get("workload_hash"))
        for records in records_by_method.values()
        for record in records
        if record.get("workload_hash") is not None
    }
    if workload_hashes and workload_hashes != {manifest["workload_hash"]}:
        raise ValueError(
            f"parity workload hashes differ from manifest: {sorted(workload_hashes)}"
        )
    if max_samples is None and manifest.get("evaluation_scope") == "communication_smoke":
        max_samples = 20
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("parity max_samples must be positive")
    reference_records = records_by_method[reference]
    selected_ids = [
        str(record["sample_id"])
        for record in reference_records[: int(max_samples) if max_samples else None]
    ]
    selected_id_set = set(selected_ids)
    selected_records = {
        method: [
            record for record in records if str(record.get("sample_id")) in selected_id_set
        ]
        for method, records in records_by_method.items()
    }
    rows = build_token_parity_rows(selected_records, reference_method=reference)
    parity_dir = layout["local_root"] / "parity"
    output_path = Path(output) if output else parity_dir / "token_parity.jsonl"
    summary_path = output_path.with_suffix(".summary.json")
    write_token_parity_report(
        rows,
        output_path,
        methods=paths.keys(),
        reference_method=reference,
    )
    summary = summarize_token_parity(rows, methods=paths.keys(), reference_method=reference)
    gate_records = selected_records if max_samples is not None else records_by_method
    sample_ids_by_method = {
        method: [str(record.get("sample_id")) for record in records]
        for method, records in gate_records.items()
    }
    gate = exact_gate_summary(
        rows,
        reference_method=reference,
        expected_sample_ids=selected_ids,
        sample_ids_by_method=sample_ids_by_method,
    )
    summary.update(
        {
            "max_samples": int(max_samples) if max_samples is not None else None,
            "selected_sample_count": len(selected_ids),
            "selected_sample_ids": selected_ids,
            "comparison_count": len(rows),
            "exact_gate": gate,
        }
    )
    write_json_once(summary_path, summary)
    exit_code = 0 if gate["gate_pass"] else 1
    append_status(
        layout["status"],
        method="suite",
        phase="parity",
        exit_code=exit_code,
        error=None if exit_code == 0 else "exact target-only token parity gate failed",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return exit_code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_models(config_path: str) -> int:
    config = load_config(config_path)
    details = {}
    failed = False
    for role in ("draft", "target"):
        path = Path(config["models"][role])
        if not path.is_dir():
            print(f"[FAIL] {role} model path missing: {path}")
            failed = True
            continue
        model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.is_file():
            print(f"[FAIL] {role} tokenizer.json missing: {tokenizer_file}")
            failed = True
        details[role] = {
            "path": str(path),
            "model_type": model_config.get("model_type"),
            "vocab_size": model_config.get("vocab_size"),
            "tokenizer_sha256": _sha256(tokenizer_file) if tokenizer_file.is_file() else None,
        }
        if details[role]["model_type"] not in {"qwen3", "llama"}:
            print(f"[FAIL] official SpecEdge does not support model_type={details[role]['model_type']}")
            failed = True
    if len(details) == 2:
        for key in ("vocab_size", "tokenizer_sha256"):
            if details["draft"][key] != details["target"][key]:
                print(f"[FAIL] draft/target {key} differs")
                failed = True
        if not failed:
            print("[PASS] model architecture and tokenizer fingerprints are compatible")
    print(json.dumps(details, indent=2, ensure_ascii=False))
    return int(failed)


def workload_hash_gate_values(actual_hash: Any, expected_hash: Any) -> int:
    """Return nonzero when a host-local hash differs from the expected value."""

    payload = {
        "actual_workload_hash": actual_hash,
        "expected_workload_hash": expected_hash,
    }
    if actual_hash != expected_hash:
        print(json.dumps({**payload, "match": False}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps({**payload, "match": True}, indent=2, ensure_ascii=False))
    return 0


def workload_hash_gate(
    manifest: str | None = None,
    expected: str | None = None,
    *,
    node3_manifest: str | None = None,
    node2_manifest: str | None = None,
) -> int:
    """Check a node2 manifest against a copied hash, or use legacy dual files."""

    if manifest is not None:
        if expected is None:
            raise ValueError("--expected is required with --manifest")
        actual = json.loads(Path(manifest).read_text(encoding="utf-8"))
        return workload_hash_gate_values(actual.get("workload_hash"), expected)
    if node3_manifest is None or node2_manifest is None:
        raise ValueError(
            "provide --manifest/--expected for cross-host use, or both legacy manifest paths"
        )
    left = json.loads(Path(node3_manifest).read_text(encoding="utf-8"))
    right = json.loads(Path(node2_manifest).read_text(encoding="utf-8"))
    return workload_hash_gate_values(
        right.get("workload_hash"), left.get("workload_hash")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate-models"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    hash_parser = subparsers.add_parser(
        "workload-hash",
        help="fail unless a host-local prepare manifest matches an expected workload hash",
    )
    hash_parser.add_argument("--manifest", help="host-local manifest, normally on node2")
    hash_parser.add_argument("--expected", help="workload_hash copied from node3 prepare")
    hash_parser.add_argument("--node3-manifest", help="legacy shared-filesystem comparison")
    hash_parser.add_argument("--node2-manifest", help="legacy shared-filesystem comparison")
    planner = subparsers.add_parser("plan")
    planner.add_argument("--config", required=True)
    planner.add_argument(
        "--python",
        dest="python_bin",
        help="legacy alias for node3 FastSD/edge interpreter; recorded verbatim in commands.txt",
    )
    planner.add_argument("--node3-edge-python")
    planner.add_argument("--node3-specedge-python")
    planner.add_argument("--node2-target-python")
    planner.add_argument("--node2-specedge-python")
    planner.add_argument("--node3-repo")
    planner.add_argument("--node2-repo")
    planner.add_argument(
        "--target-host",
        help="target host/IP or local-forward endpoint; topology is user-configurable",
    )
    planner.add_argument("--target-port", type=int, help="configurable target service port")
    planner.add_argument(
        "--cpu-prefix",
        help="optional raw taskset/cpuset/numactl prefix; recorded in commands.txt",
    )
    subparsers.add_parser(
        "validate-config",
        help="validate frozen latency/throughput topology and generation scope",
    ).add_argument("--config", required=True)
    normalizer = subparsers.add_parser("normalize")
    normalizer.add_argument("--config", required=True)
    normalizer.add_argument(
        "--method",
        required=True,
        choices=["fastsd", "specedge", "specedge_cpu_adapted", "standard_sd", "draft_only"],
    )
    normalizer.add_argument("--input", required=True)
    resources_parser = subparsers.add_parser(
        "resources",
        help="parse CPU/GPU sidecars into an immutable method resource_metrics.json",
    )
    resources_parser.add_argument("--method-dir", required=True)
    resources_parser.add_argument("--cpu-mpstat")
    resources_parser.add_argument(
        "--cpu-cores",
        help="comma-separated mpstat CPU ids or ranges, e.g. 56-71,80-95",
    )
    resources_parser.add_argument("--gpu-nvidia-csv")
    resources_parser.add_argument("--gpu-index", type=int)
    resources_parser.add_argument("--gpu-uuid")
    resources_parser.add_argument(
        "--gpu-csv-schema",
        choices=NVIDIA_SMI_CSV_SCHEMA_CHOICES,
        help=(
            "explicit nvidia-smi CSV layout; omit only for unambiguous full8. "
            "Required for six-column input."
        ),
    )
    resources_parser.add_argument("--output")
    comparator = subparsers.add_parser("compare")
    comparator.add_argument("summaries", nargs="+")
    comparator.add_argument("--output")
    paired_parser = subparsers.add_parser(
        "paired",
        help="direct paired FastSD vs SpecEdge bootstrap analysis",
    )
    paired_parser.add_argument("--config", required=True)
    paired_parser.add_argument("--left", required=True, help="METHOD=PATH; normally fastsd=...")
    paired_parser.add_argument(
        "--right",
        required=True,
        help="METHOD=PATH; normally specedge_cpu_adapted=...",
    )
    paired_parser.add_argument("--output")
    analysis_parser = subparsers.add_parser(
        "analyze",
        help="generate a four-dataset/four-method paired analysis Markdown report",
    )
    analysis_parser.add_argument(
        "--input",
        action="append",
        required=True,
        dest="input_specs",
        help="DATASET/METHOD=summary.json or normalized method directory; repeat 16 times",
    )
    analysis_parser.add_argument("--output", required=True, help="output Markdown path")
    analysis_parser.add_argument(
        "--focal-method",
        choices=ANALYSIS_METHODS,
        default="fastsd",
        help="method on the left side of each direct comparison; default is FastSD",
    )
    analysis_parser.add_argument(
        "--baseline",
        choices=ANALYSIS_METHODS,
        action="append",
        dest="baselines",
        help="baseline on the right; repeat for a subset, default is every method except focal",
    )
    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--config", required=True)
    parity_parser.add_argument("--input", action="append", required=True, dest="input_specs")
    parity_parser.add_argument("--reference", default=ORACLE_METHOD)
    parity_parser.add_argument("--output")
    parity_parser.add_argument(
        "--max-samples",
        type=int,
        help="compare the first N reference samples; communication_smoke defaults to 20",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args.config)
    if args.command == "plan":
        return print_plan(
            args.config,
            python_bin=args.python_bin,
            target_host=args.target_host,
            target_port=args.target_port,
            cpu_prefix=args.cpu_prefix,
            node3_edge_python=args.node3_edge_python,
            node3_specedge_python=args.node3_specedge_python,
            node2_target_python=args.node2_target_python,
            node2_specedge_python=args.node2_specedge_python,
            node3_repo=args.node3_repo,
            node2_repo=args.node2_repo,
        )
    if args.command == "normalize":
        return normalize(args.config, args.method, args.input)
    if args.command == "resources":
        return resources(
            args.method_dir,
            cpu_mpstat=args.cpu_mpstat,
            cpu_cores=args.cpu_cores,
            gpu_nvidia_csv=args.gpu_nvidia_csv,
            gpu_index=args.gpu_index,
            gpu_uuid=args.gpu_uuid,
            gpu_csv_schema=args.gpu_csv_schema,
            output=args.output,
        )
    if args.command == "compare":
        return compare(args.summaries, args.output)
    if args.command == "paired":
        return paired_analysis(args.config, args.left, args.right, output=args.output)
    if args.command == "analyze":
        return analyze(
            args.input_specs,
            args.output,
            focal_method=args.focal_method,
            baselines=args.baselines,
        )
    if args.command == "parity":
        return parity(
            args.config,
            args.input_specs,
            reference=args.reference,
            output=args.output,
            max_samples=args.max_samples,
        )
    if args.command == "validate-models":
        return validate_models(args.config)
    if args.command == "workload-hash":
        return workload_hash_gate(
            args.manifest,
            args.expected,
            node3_manifest=args.node3_manifest,
            node2_manifest=args.node2_manifest,
        )
    if args.command == "validate-config":
        notes = validate_experiment_config(load_config(args.config))
        print(json.dumps({"valid": True, "notes": notes}, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
