#!/usr/bin/env python3
"""Merge already-complete edge/cloud sampler CSV files without replacement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path, role: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        return reader.fieldnames, [{"role": role, **row} for row in reader]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", required=True)
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    edge = Path(args.edge)
    cloud = Path(args.cloud)
    output = Path(args.output)
    for path in (edge, cloud):
        if not path.is_file():
            raise SystemExit(f"ERROR: missing sampler CSV: {path}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to overwrite merged sampler CSV: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"ERROR: output parent does not exist: {output.parent}")
    edge_fields, edge_rows = rows(edge, "edge")
    cloud_fields, cloud_rows = rows(cloud, "cloud")
    if edge_fields != cloud_fields:
        raise SystemExit("ERROR: edge/cloud sampler schemas differ")
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", *edge_fields])
        writer.writeheader()
        writer.writerows(edge_rows)
        writer.writerows(cloud_rows)


if __name__ == "__main__":
    main()
