#!/usr/bin/env python3
"""Create a canonical completion artifact from SpecEdge adapter request records.

The adapter intentionally treats Official SpecEdge response construction as an
explicit factory concern.  This utility preserves that factory result verbatim
under ``client_result`` and exposes common identifiers without inventing text
or token counts that the factory did not return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(value)
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def completion_text(client_result: Any) -> str | None:
    if not isinstance(client_result, dict):
        return None
    for key in ("text", "output", "completion", "generated_text"):
        value = client_result.get(key)
        if isinstance(value, str):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace = Path(args.trace)
    requests = Path(args.requests)
    output = Path(args.output)
    for path in (trace, requests):
        if not path.is_file():
            raise SystemExit(f"ERROR: required input does not exist: {path}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to overwrite completion artifact: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"ERROR: output parent does not exist: {output.parent}")

    trace_rows = read_jsonl(trace)
    request_rows = read_jsonl(requests)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for request in request_rows:
            trace_index = int(request["trace_index"])
            if trace_index < 0 or trace_index >= len(trace_rows):
                raise SystemExit(f"ERROR: request trace_index is absent from trace: {trace_index}")
            trace_row = trace_rows[trace_index]
            result = request.get("client_result")
            record = {
                "schema_version": 1,
                "status": request.get("status"),
                "request_id": request.get("request_id"),
                "trace_index": trace_index,
                "task_id": trace_row.get("task_id"),
                "dataset_index": trace_row.get("dataset_index"),
                "text": completion_text(result),
                "client_result": result,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
