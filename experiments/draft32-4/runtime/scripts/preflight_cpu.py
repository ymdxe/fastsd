"""Record CPU affinity and host load before a CPU draft experiment.

This preflight is intentionally conservative: it verifies that the requested
thread count fits the process-visible affinity, records the actual binding and
background load, and never claims that the host is exclusively owned by the
run.  A cpuset/taskset/numactl prefix is applied by the caller, so this script
observes the resulting binding rather than guessing it from a config file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.run_artifacts import append_command, write_json_once


def _affinity() -> list[int]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return sorted(int(cpu) for cpu in getter(0))
    return list(range(os.cpu_count() or 1))


def parse_cpu_set(spec: str) -> list[int]:
    """Parse Linux cpuset notation such as ``56-71,80-95``."""

    result: set[int] = set()
    if not spec.strip():
        return []
    for part in spec.split(","):
        piece = part.strip()
        if "-" in piece:
            left, right = piece.split("-", 1)
            start, end = int(left), int(right)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range: {piece}")
            result.update(range(start, end + 1))
        else:
            value = int(piece)
            if value < 0:
                raise ValueError(f"invalid CPU id: {piece}")
            result.add(value)
    return sorted(result)


def _allowed_list_from_proc() -> str | None:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return None


def collect_preflight(
    *,
    requested_threads: int,
    expected_cpu_count: int | None = None,
    max_load_ratio: float | None = None,
    phase: str = "pre",
    expected_cpuset: str | None = None,
) -> tuple[dict, int]:
    if requested_threads <= 0:
        raise ValueError("requested_threads must be positive")
    affinity = _affinity()
    cpu_count = len(affinity)
    expected_cpus = parse_cpu_set(expected_cpuset or "")
    loadavg = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
    load_ratio = (float(loadavg[0]) / cpu_count) if loadavg and cpu_count else None
    binding_fits = cpu_count >= requested_threads
    expected_matches = expected_cpu_count is None or cpu_count == expected_cpu_count
    cpuset_matches = not expected_cpus or affinity == expected_cpus
    warnings: list[str] = []
    if not binding_fits:
        warnings.append("requested thread count exceeds process-visible CPU affinity")
    if expected_cpu_count is not None and not expected_matches:
        warnings.append("observed affinity count differs from expected_cpu_count")
    if expected_cpus and not cpuset_matches:
        warnings.append("observed affinity differs from expected_cpuset")
    if load_ratio is not None and max_load_ratio is not None and load_ratio > max_load_ratio:
        warnings.append("background load ratio exceeds max_load_ratio")
    mpstat_text = None
    try:
        mpstat = subprocess.run(
            ["mpstat", "-P", "ALL", "1", "1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        mpstat_text = f"unavailable: {exc}"
    else:
        mpstat_text = (mpstat.stdout or mpstat.stderr).strip()
    payload = {
        "schema_version": 1,
        "phase": phase,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "requested_threads": int(requested_threads),
        "visible_cpu_count": cpu_count,
        "cpu_affinity": affinity,
        "proc_cpus_allowed_list": _allowed_list_from_proc(),
        "binding_verified": bool(binding_fits and expected_matches and cpuset_matches),
        "expected_cpu_count": expected_cpu_count,
        "expected_cpuset": expected_cpus,
        "loadavg": loadavg,
        "background_load_ratio": load_ratio,
        "mpstat": mpstat_text,
        "cpu_exclusive": False,
        "warnings": warnings,
    }
    exit_code = 1 if warnings and (
        not binding_fits
        or not expected_matches
        or not cpuset_matches
        or max_load_ratio is not None and load_ratio is not None and load_ratio > max_load_ratio
    ) else 0
    return payload, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True, dest="requested_threads")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-cpu-count", type=int)
    parser.add_argument("--expected-cpuset")
    parser.add_argument("--max-load-ratio", type=float)
    parser.add_argument("--phase", choices=["pre", "post"], default="pre")
    parser.add_argument("--commands-path")
    args = parser.parse_args(argv)
    if args.expected_cpu_count is not None and args.expected_cpu_count <= 0:
        parser.error("--expected-cpu-count must be positive")
    if args.max_load_ratio is not None and args.max_load_ratio < 0:
        parser.error("--max-load-ratio must be non-negative")
    payload, exit_code = collect_preflight(
        requested_threads=args.requested_threads,
        expected_cpu_count=args.expected_cpu_count,
        max_load_ratio=args.max_load_ratio,
        phase=args.phase,
        expected_cpuset=args.expected_cpuset,
    )
    write_json_once(args.output, payload)
    if args.commands_path:
        append_command(
            args.commands_path,
            shlex.join(sys.argv),
            status=exit_code,
            note=f"cpu_{args.phase}flight; prefix binding observed by this process",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
