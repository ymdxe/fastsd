"""Run the repository-wide CPU/static integration test suite.

The runner intentionally does not install or execute heavyweight GPU model stacks.
It verifies FastSD's unit tests and the pinned SpecEdge source/config integration.
Under Python 3.14 it also compiles the complete official SpecEdge source.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> bool:
    print(f"\n== {label} ==", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        print(f"[FAIL] {label} exited with {completed.returncode}", flush=True)
        return False
    print(f"[PASS] {label}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-official",
        action="store_true",
        help="require Python 3.14 and compile the complete official SpecEdge source",
    )
    args = parser.parse_args(argv)

    official_python = sys.version_info[:2] == (3, 14)
    if args.strict_official and not official_python:
        print(
            "[FAIL] --strict-official requires Python 3.14; "
            f"current interpreter is {sys.version.split()[0]}"
        )
        return 2

    checks = [
        (
            "FastSD and baseline unit tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        ),
        (
            "SpecEdge pinned-source and paper-config doctor",
            [sys.executable, "baselines/specedge/repro.py", "doctor"],
        ),
        (
            "FastSD SpecEdge integration helper compilation",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "baselines/specedge/repro.py",
                "baselines/specedge/integration",
                "benchmark/eval_draft_pool.py",
                "benchmark/eval_target_only.py",
                "benchmark/eval_unified.py",
                "scripts/eval_suite.py",
                "scripts/preflight_cpu.py",
                "scripts/make_evaluation_matrix.py",
                "src/evaluation.py",
                "src/common_metrics.py",
                "src/parity.py",
                "src/run_artifacts.py",
                "src/runtime.py",
                "baselines/specedge/integration/cpu_adapter.py",
                "baselines/specedge/integration/wire_codec.py",
            ],
        ),
    ]
    if official_python:
        checks.append(
            (
                "Official SpecEdge Python 3.14 source compilation",
                [
                    sys.executable,
                    "-m",
                    "compileall",
                    "-q",
                    "baselines/specedge/official/src",
                ],
            )
        )
    else:
        print(
            "[SKIP] Official SpecEdge source compilation requires Python 3.14; "
            f"current interpreter is {sys.version.split()[0]}",
            flush=True,
        )

    passed = True
    for label, command in checks:
        passed = run(label, command) and passed

    print("\nRESULT: " + ("all integration checks passed" if passed else "checks failed"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
