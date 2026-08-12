#!/usr/bin/env python3
"""Sample only explicitly selected NVIDIA GPUs into a new CSV file.

This sidecar never discovers or controls other processes.  Its parent wrapper
may send SIGTERM only to this exact child PID during normal shutdown.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import signal
import subprocess
import sys
import time
from pathlib import Path


STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def parse_gpu_list(value: str) -> list[str]:
    values = value.split(",")
    if not values or any(not item.isdigit() for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("--physical-gpus must be unique comma-separated GPU indexes")
    return values


def sample(gpu_ids: list[str]) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={','.join(gpu_ids)}",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    rows: list[list[str]] = []
    for raw in completed.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi output: {raw!r}")
        rows.append(fields)
    if len(rows) != len(gpu_ids):
        raise RuntimeError(f"expected {len(gpu_ids)} GPU rows, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-gpus", required=True, type=parse_gpu_list)
    parser.add_argument("--interval-ms", type=int, default=50)
    args = parser.parse_args()
    if args.interval_ms < 20:
        raise SystemExit("ERROR: --interval-ms must be at least 20")

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to overwrite GPU samples: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"ERROR: output parent does not exist: {output.parent}")
    if str(output) != "/home/hdd" and not str(output).startswith("/home/hdd/"):
        raise SystemExit(f"ERROR: output must be under /home/hdd: {output}")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp_utc",
                "monotonic_s",
                "physical_gpu",
                "gpu_uuid",
                "utilization_gpu_pct",
                "memory_used_mib",
                "memory_total_mib",
                "power_draw_w",
            ]
        )
        handle.flush()
        while not STOP:
            now_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
            now_monotonic = time.monotonic()
            try:
                rows = sample(args.physical_gpus)
            except Exception as exc:  # preserve the reason in a valid CSV row
                print(f"gpu sampler stopping: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            for row in rows:
                writer.writerow([now_utc, f"{now_monotonic:.9f}", *row])
            handle.flush()
            deadline = time.monotonic() + args.interval_ms / 1000.0
            while not STOP and time.monotonic() < deadline:
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
