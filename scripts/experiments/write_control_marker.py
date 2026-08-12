#!/usr/bin/env python3
"""Exclusively create one non-secret experiment control marker.

It is used only for a run-owned graceful shutdown file.  Existing markers are
never replaced or removed, so a stale artifact cannot affect a new experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path, run_id: str) -> None:
    if not run_id or not all(char.isalnum() or char in "._-" for char in run_id):
        raise SystemExit("ERROR: invalid --run-id")
    if not path.is_absolute() or not str(path).startswith("/home/hdd/"):
        raise SystemExit("ERROR: marker path must be absolute under /home/hdd")
    if path.name != "graceful-shutdown.json" or path.parent.name != "control":
        raise SystemExit("ERROR: marker path must be this run's control/graceful-shutdown.json")
    if not path.parent.is_dir():
        raise SystemExit(f"ERROR: marker parent does not exist: {path.parent}")
    if path.exists() or path.is_symlink():
        raise SystemExit(f"ERROR: refusing to reuse control marker: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    path = Path(args.path)
    validate(path, args.run_id)
    payload = {
        "schema_version": 1,
        "event": "graceful_shutdown",
        "run_id": args.run_id,
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
