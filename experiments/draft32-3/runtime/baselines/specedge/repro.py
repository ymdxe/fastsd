"""Local integration checks for the pinned official SpecEdge baseline.

This module deliberately uses only the Python standard library so that source and
configuration checks can run before the heavyweight CUDA environment is installed.
"""

from __future__ import annotations

import argparse
import math
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BASELINE_ROOT = Path(__file__).resolve().parent
OFFICIAL_ROOT = BASELINE_ROOT / "official"
CONFIG_ROOT = BASELINE_ROOT / "configs"
EXPECTED_OFFICIAL_SHA = "1edcaf02ffc41a7b57726450c5357ed216a3b9bc"

MODEL_PAIRS = {
    "qwen3_14b_1.7b": ("Qwen/Qwen3-14B", "Qwen/Qwen3-1.7B", "A100-40"),
    "qwen3_14b_0.6b": ("Qwen/Qwen3-14B", "Qwen/Qwen3-0.6B", "A100-40"),
    "qwen3_32b_1.7b": ("Qwen/Qwen3-32B", "Qwen/Qwen3-1.7B", "A100-80"),
}

REQUIRED_OFFICIAL_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "script/batch_server.sh",
    "script/client_host.sh",
    "script/server_only.sh",
    "src/specedge/client/proactive.py",
    "src/strategy/server_verify/specexec/grpc.py",
)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_yaml_scalars(path: Path) -> dict[str, Any]:
    """Read scalar values from the mapping-only subset used by our YAML configs."""

    result: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    key_pattern = re.compile(r"^(?:-\s+)?([^:#]+):(?:\s*(.*))?$")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        match = key_pattern.match(raw_line.strip())
        if not match:
            continue
        key = match.group(1).strip()
        value = (match.group(2) or "").strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not value:
            stack.append((indent, key))
            continue
        dotted = ".".join([entry[1] for entry in stack] + [key])
        result[dotted] = _parse_scalar(value)
    return result


def config_paths() -> list[Path]:
    return sorted(CONFIG_ROOT.glob("*.yaml"))


def validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    values = load_yaml_scalars(path)
    name = path.stem
    mode = "server_only" if name.startswith("server_only_") else "specedge"
    pair_key = name.removeprefix(f"{mode}_")

    if pair_key not in MODEL_PAIRS:
        return [f"unknown model pair in filename: {name}"]

    target, draft, _gpu = MODEL_PAIRS[pair_key]
    expected = {
        "base.dtype": "fp16",
        "base.seed": 42,
        "server.target_model": target,
        "server.temperature": 0.7,
        "client.draft_model": draft,
        "client.dataset": "specbench",
        "client.max_n_beams": 32,
        "client.max_branch_width": 16,
        "client.max_budget": 32,
        "client.max_new_tokens": 256,
    }
    for key, wanted in expected.items():
        actual = values.get(key)
        if actual != wanted:
            errors.append(f"{key}: expected {wanted!r}, got {actual!r}")

    depth = values.get("client.max_beam_len")
    if not isinstance(depth, int) or depth < 1:
        errors.append(f"client.max_beam_len must be a positive integer, got {depth!r}")

    if mode == "specedge":
        mode_expected = {
            "server.max_batch_size": 1,
            "server.num_clients": 2,
            "client.proactive.type": "included",
            "client.proactive.max_budget": 32,
        }
    else:
        mode_expected = {"client.max_batch_size": 1}

    for key, wanted in mode_expected.items():
        actual = values.get(key)
        if actual != wanted:
            errors.append(f"{key}: expected {wanted!r}, got {actual!r}")
    return errors


def recommend_depth(verify_ms: float, draft_ms: float, rtt_ms: float) -> int:
    if verify_ms <= 0 or draft_ms <= 0 or rtt_ms < 0:
        raise ValueError("verify-ms and draft-ms must be positive; rtt-ms cannot be negative")
    available_ms = verify_ms - rtt_ms
    if available_ms <= 0:
        return 1
    return max(1, math.floor(available_ms / draft_ms + 0.5))


def _git_revision() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()


def _gpu_summary() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        proc = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError:
        return None
    return "; ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())


def doctor(strict_runtime: bool = False) -> int:
    failures: list[str] = []
    warnings: list[str] = []

    if not OFFICIAL_ROOT.is_dir():
        failures.append("official submodule is missing; run git submodule update --init --recursive")
    else:
        revision = _git_revision()
        if revision != EXPECTED_OFFICIAL_SHA:
            failures.append(
                f"official revision is {revision or 'unknown'}, expected {EXPECTED_OFFICIAL_SHA}"
            )
        for relative in REQUIRED_OFFICIAL_FILES:
            if not (OFFICIAL_ROOT / relative).is_file():
                failures.append(f"official source file missing: {relative}")

    paths = config_paths()
    if len(paths) != 6:
        failures.append(f"expected 6 paper configs, found {len(paths)}")
    for path in paths:
        for error in validate_config(path):
            failures.append(f"{path.name}: {error}")

    if platform.system() != "Linux":
        warnings.append(
            f"host is {platform.system()}; official end-to-end launch is Linux/CUDA-only"
        )
    if sys.version_info[:2] != (3, 14):
        warnings.append(
            f"Python is {platform.python_version()}; official pyproject requires Python ~=3.14.0"
        )
    for command in ("uv", "bash", "ssh"):
        if shutil.which(command) is None:
            warnings.append(f"runtime command not found: {command}")
    gpu = _gpu_summary()
    if gpu:
        warnings.append(f"detected GPU(s): {gpu}; capacity is not a full runtime validation")
    else:
        warnings.append("nvidia-smi GPU inventory is unavailable")
    warnings.append(
        "official LICENSE restricts commercial deployment despite pyproject.toml saying MIT"
    )

    print(f"SpecEdge official revision: {_git_revision() or 'unavailable'}")
    print(f"Paper config count: {len(paths)}")
    for item in failures:
        print(f"[FAIL] {item}")
    for item in warnings:
        print(f"[WARN] {item}")
    if failures:
        print("RESULT: integration check failed")
        return 1
    if strict_runtime and warnings:
        print("RESULT: strict runtime check failed")
        return 1
    print("RESULT: source and paper-config integration checks passed")
    return 0


def print_matrix() -> int:
    print("mode\ttarget\tdraft\trequired_server_gpu\tconfig")
    for pair_key, (target, draft, gpu) in MODEL_PAIRS.items():
        for mode in ("specedge", "server_only"):
            path = CONFIG_ROOT / f"{mode}_{pair_key}.yaml"
            print(f"{mode}\t{target}\t{draft}\t{gpu}\t{path}")
    return 0


def validate_paths(paths: list[str]) -> int:
    selected = [Path(path) for path in paths] if paths else config_paths()
    failed = False
    for path in selected:
        errors = validate_config(path)
        if errors:
            failed = True
            for error in errors:
                print(f"[FAIL] {path}: {error}")
        else:
            print(f"[PASS] {path}")
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="check source, configs, and runtime")
    doctor_parser.add_argument("--strict-runtime", action="store_true")

    subparsers.add_parser("paper-matrix", help="print the paper-aligned experiment matrix")

    validate_parser = subparsers.add_parser("validate-config", help="validate YAML configs")
    validate_parser.add_argument("paths", nargs="*")

    depth_parser = subparsers.add_parser(
        "recommend-depth", help="estimate draft depth from measured pipeline timings"
    )
    depth_parser.add_argument("--verify-ms", type=float, required=True)
    depth_parser.add_argument("--draft-ms", type=float, required=True)
    depth_parser.add_argument("--rtt-ms", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor(strict_runtime=args.strict_runtime)
    if args.command == "paper-matrix":
        return print_matrix()
    if args.command == "validate-config":
        return validate_paths(args.paths)
    if args.command == "recommend-depth":
        depth = recommend_depth(args.verify_ms, args.draft_ms, args.rtt_ms)
        print(depth)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
