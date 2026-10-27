"""Write-once experiment artifacts and append-only run status records."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text_once(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Create a file without replacing an existing artifact.

    Re-running a command with byte-identical content is harmless and returns
    normally.  A different payload raises instead of destroying the previous
    run, which keeps failed/partial evidence auditable.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = output.read_text(encoding=encoding)
        if existing != content:
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        return output
    output.write_text(content, encoding=encoding)
    return output


def write_json_once(path: str | Path, payload: Any) -> Path:
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return write_text_once(path, content)


def copy_file_once(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if sha256(source_path) != sha256(destination_path):
            raise FileExistsError(
                f"refusing to overwrite existing copied input: {destination_path}"
            )
        return destination_path
    shutil.copyfile(source_path, destination_path)
    return destination_path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output


def append_command(
    path: str | Path,
    command: str,
    *,
    config: str | Path | None = None,
    status: int | None = None,
    note: str | None = None,
) -> Path:
    """Append a copyable command and optional exit status to ``commands.txt``."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# command_record_utc: {utc_now()}\n")
        if config is not None:
            handle.write(f"# config: {config}\n")
        if note:
            handle.write(f"# note: {note}\n")
        handle.write(f"command={command.rstrip()}\n")
        if status is not None:
            handle.write(f"exit_status={int(status)}\n")
    return output


def append_status(
    path: str | Path,
    *,
    method: str,
    phase: str,
    exit_code: int,
    command: str | None = None,
    error: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "recorded_utc": utc_now(),
        "method": method,
        "phase": phase,
        "exit_code": int(exit_code),
    }
    if command is not None:
        payload["command"] = command
    if error is not None:
        payload["error"] = error
    return append_jsonl(path, payload)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(repo_root: str | Path) -> str | None:
    """Return a Git SHA when available, without making it a run prerequisite."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def require_unique_sample_ids(records: Iterable[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in canonical workload: {sample_id}")
        seen.add(sample_id)
