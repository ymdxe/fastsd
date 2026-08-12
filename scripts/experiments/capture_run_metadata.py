#!/usr/bin/env python3
"""Create and update non-secret, append-only FastSD experiment metadata.

The tool deliberately refuses to overwrite manifests, configs, trace copies, or
command records.  It uses only the Python standard library so it can run in the
FastSD and SpecEdge environments before their Python dependencies are loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "FASTSD_TARGET_DEVICE",
    "FASTSD_DRAFT_MODEL",
    "FASTSD_TARGET_MODEL",
    "FASTSD_RESULTS_DIR",
    "CLOUD_SERVICE_PORT",
    "CLOUD_BIND_HOST",
    "SERVER_URL",
    "SPECEDGE_GRPC_ADDRESS",
    "SPECEDGE_CLIENT_FACTORY",
    "SPECEDGE_OFFICIAL_ROOT",
    "SPECEDGE_EFFECTIVE_CONFIG",
    "SPECEDGE_PROMPTS_PATH",
    "SPECEDGE_CLIENT_ID",
    "SPECEDGE_DEVICE",
    "SPECEDGE_CLIENT_RESULT_PATH",
    "SPECEDGE_CLIENT_EXP_NAME",
    "SPECEDGE_SYNC_TIMEOUT_S",
    "SPECEDGE_BIND_HOST",
    "SPECEDGE_PORT",
    "TOKENIZERS_PARALLELISM",
    "PYTHONNOUSERSITE",
)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_hdd_path(path: Path) -> None:
    raw = str(path)
    if raw != "/home/hdd" and not raw.startswith("/home/hdd/"):
        fail(f"experiment path must be under /home/hdd: {path}")
    if ".." in path.parts:
        fail(f"path traversal is not allowed: {path}")


def run_text(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_snapshot(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"available": False, "reason": "not_a_git_worktree"}
    return {
        "available": True,
        "root": str(repo),
        "head": run_text(["git", "rev-parse", "HEAD"], repo),
        "branch": run_text(["git", "branch", "--show-current"], repo),
        "status": run_text(["git", "status", "--porcelain", "--ignore-submodules=none"], repo),
        "submodules": run_text(["git", "submodule", "status", "--recursive"], repo),
    }


def gpu_snapshot() -> dict[str, Any]:
    result = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    return result


def disk_snapshot(path: Path) -> dict[str, Any]:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {
        "path_checked": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 3),
    }


def network_snapshot(peer_ip: str | None) -> dict[str, Any]:
    if not peer_ip:
        return {}
    return {
        "peer_ip": peer_ip,
        "route": run_text(["ip", "route", "get", peer_ip]),
        "ping": run_text(["ping", "-c", "3", "-W", "2", peer_ip]),
    }


def top_level_inventory(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        return {"path": str(model_path), "exists": False}
    files: list[dict[str, Any]] = []
    for item in sorted(model_path.iterdir(), key=lambda value: value.name):
        if not item.is_file():
            continue
        stat = item.stat()
        entry: dict[str, Any] = {
            "name": item.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if item.name in {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}:
            entry["sha256"] = sha256_file(item)
        files.append(entry)
    return {"path": str(model_path), "exists": True, "files": files}


def safe_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite existing metadata: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing required JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON object expected in {path}")
    return payload


def atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        fail(f"temporary metadata path unexpectedly exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def common_manifest(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    model_paths = [Path(item).expanduser() for item in args.model_path]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "component_role": args.role,
        "method": args.method,
        "status": "running",
        "created_at_utc": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "git": git_snapshot(repo),
        "selected_gpus": {
            "physical": args.physical_gpus.split(",") if args.physical_gpus else [],
            "logical_visible_devices": list(range(len(args.physical_gpus.split(",")))) if args.physical_gpus else [],
        },
        "network": {
            "bind_host": args.bind_host,
            "port": args.port,
            **network_snapshot(args.peer_ip),
        },
        "models": [top_level_inventory(path) for path in model_paths],
        "environment": safe_environment(),
        "disk": disk_snapshot(Path(args.run_dir)),
        "gpu_snapshot": gpu_snapshot(),
        "artifacts": {},
    }


def action_init(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    ensure_hdd_path(run_dir)
    if run_dir.exists() or run_dir.is_symlink():
        fail(f"run directory already exists; refusing to reuse it: {run_dir}")
    repo = Path(args.repo_root).resolve()
    if not repo.is_dir():
        fail(f"repository root does not exist: {repo}")
    config = Path(args.config)
    if not config.is_file():
        fail(f"configuration file does not exist: {config}")

    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("config", "provenance", "logs", "metrics", "outputs", "workload"):
        (run_dir / relative).mkdir(exist_ok=False)

    copied_config = run_dir / "config" / "resolved.yaml"
    shutil.copy2(config, copied_config)

    manifest = common_manifest(args, repo)
    manifest["artifacts"]["resolved_config"] = {
        "path": str(copied_config.relative_to(run_dir)),
        "sha256": sha256_file(copied_config),
    }
    write_new_json(run_dir / "manifest.json", manifest)

    provenance_name = args.provenance_name
    if not provenance_name or not provenance_name.replace("_", "").replace("-", "").isalnum():
        fail(f"invalid provenance name: {provenance_name!r}")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "role": args.role,
        "host": manifest["host"],
        "git": manifest["git"],
        "selected_gpus": manifest["selected_gpus"],
        "gpu_snapshot": manifest["gpu_snapshot"],
        "disk": manifest["disk"],
        "network": manifest["network"],
        "environment": manifest["environment"],
        "models": manifest["models"],
    }
    write_new_json(run_dir / "provenance" / f"{provenance_name}.json", provenance)
    print(run_dir)


def update_manifest(run_dir: Path, mutate: "Callable[[dict[str, Any]], None]") -> None:
    ensure_hdd_path(run_dir)
    manifest_path = run_dir / "manifest.json"
    payload = load_json(manifest_path)
    mutate(payload)
    atomic_replace_json(manifest_path, payload)


def action_attach_command(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    command_file = Path(args.command_file)
    if not command_file.is_file():
        fail(f"command file does not exist: {command_file}")
    if command_file.parent != run_dir / "logs":
        fail("command file must be under the run directory logs/ directory")

    def mutate(payload: dict[str, Any]) -> None:
        payload.setdefault("artifacts", {})["launch_command"] = {
            "path": str(command_file.relative_to(run_dir)),
            "sha256": sha256_file(command_file),
        }

    update_manifest(run_dir, mutate)


def action_attach_trace(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    trace = Path(args.trace)
    if not trace.is_file():
        fail(f"trace does not exist: {trace}")
    target = run_dir / "workload" / "arrival_trace.jsonl"
    if target.exists() or target.is_symlink():
        fail(f"refusing to overwrite trace copy: {target}")
    shutil.copy2(trace, target)
    copied: dict[str, dict[str, str]] = {
        "arrival_trace": {"path": str(target.relative_to(run_dir)), "sha256": sha256_file(target)}
    }
    for name in ("manifest.json", "prompts.jsonl"):
        companion = trace.parent / name
        destination = run_dir / "workload" / name
        if companion.is_file():
            if destination.exists() or destination.is_symlink():
                fail(f"refusing to overwrite workload companion: {destination}")
            shutil.copy2(companion, destination)
            copied[name] = {"path": str(destination.relative_to(run_dir)), "sha256": sha256_file(destination)}

    def mutate(payload: dict[str, Any]) -> None:
        payload.setdefault("artifacts", {})["workload"] = copied

    update_manifest(run_dir, mutate)


def action_attach_closed_loop_plan(args: argparse.Namespace) -> None:
    """Record an immutable closed-loop calibration plan already in workload/."""
    run_dir = Path(args.run_dir)
    plan = run_dir / "workload" / "closed_loop_plan.jsonl"
    workload_manifest = run_dir / "workload" / "manifest.json"
    if not plan.is_file() or not workload_manifest.is_file():
        fail("closed-loop plan and workload manifest must exist before attachment")

    def mutate(payload: dict[str, Any]) -> None:
        artifacts = payload.setdefault("artifacts", {})
        if "workload" in artifacts:
            fail("run manifest already has a workload artifact")
        artifacts["workload"] = {
            "closed_loop_plan": {
                "path": str(plan.relative_to(run_dir)),
                "sha256": sha256_file(plan),
            },
            "manifest": {
                "path": str(workload_manifest.relative_to(run_dir)),
                "sha256": sha256_file(workload_manifest),
            },
            "matrix_eligible": False,
        }

    update_manifest(run_dir, mutate)


def action_finish(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if args.status not in {"complete", "failed", "cancelled"}:
        fail(f"unsupported final status: {args.status}")

    def mutate(payload: dict[str, Any]) -> None:
        if payload.get("status") != "running":
            fail(f"manifest is already finalized with status {payload.get('status')!r}")
        payload["status"] = args.status
        payload["finished_at_utc"] = utc_now()
        payload["exit_code"] = args.exit_code
        if args.error:
            payload["error"] = args.error[:2000]

    update_manifest(run_dir, mutate)


def action_finalize_root(args: argparse.Namespace) -> None:
    root = Path(args.run_dir)
    ensure_hdd_path(root)
    if not root.is_dir():
        fail(f"run root does not exist: {root}")
    destination = root / "manifest.json"
    if destination.exists() or destination.is_symlink():
        fail(f"root manifest already exists: {destination}")
    components: dict[str, Any] = {}
    for role in ("edge", "cloud"):
        path = root / role / "manifest.json"
        payload = load_json(path)
        components[role] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "status": payload.get("status"),
            "method": payload.get("method"),
            "run_id": payload.get("run_id"),
        }
    run_ids = {item["run_id"] for item in components.values()}
    methods = {item["method"] for item in components.values()}
    if len(run_ids) != 1 or len(methods) != 1:
        fail("edge and cloud manifests do not identify the same method and run id")
    write_new_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": next(iter(run_ids)),
            "method": next(iter(methods)),
            "status": "complete" if all(item["status"] == "complete" for item in components.values()) else "incomplete",
            "finalized_at_utc": utc_now(),
            "components": components,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    init = subparsers.add_parser("init", help="create a fresh component run directory")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--role", choices=("edge", "cloud", "specedge-edge", "specedge-cloud"), required=True)
    init.add_argument("--method", required=True)
    init.add_argument("--config", required=True)
    init.add_argument("--repo-root", required=True)
    init.add_argument("--physical-gpus", default="")
    init.add_argument("--bind-host", default="")
    init.add_argument("--port", default="")
    init.add_argument("--peer-ip")
    init.add_argument("--model-path", action="append", default=[])
    init.add_argument("--provenance-name", required=True)

    command = subparsers.add_parser("attach-command", help="record an already-created command file")
    command.add_argument("--run-dir", required=True)
    command.add_argument("--command-file", required=True)

    trace = subparsers.add_parser("attach-trace", help="copy a trace into a fresh run component")
    trace.add_argument("--run-dir", required=True)
    trace.add_argument("--trace", required=True)

    closed_loop = subparsers.add_parser(
        "attach-closed-loop-plan", help="record a fresh deterministic closed-loop calibration plan"
    )
    closed_loop.add_argument("--run-dir", required=True)

    finish = subparsers.add_parser("finish", help="write terminal component status")
    finish.add_argument("--run-dir", required=True)
    finish.add_argument("--status", required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--error", default="")

    root = subparsers.add_parser("finalize-root", help="create immutable root manifest after collection")
    root.add_argument("--run-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    actions = {
        "init": action_init,
        "attach-command": action_attach_command,
        "attach-trace": action_attach_trace,
        "attach-closed-loop-plan": action_attach_closed_loop_plan,
        "finish": action_finish,
        "finalize-root": action_finalize_root,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
