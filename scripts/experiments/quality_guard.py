#!/usr/bin/env python3
"""Create a non-destructive, fixed-size quality artifact for one completed run.

The performance workload and quality workload intentionally remain separate.
This tool reads only the canonical ``outputs/completions.jsonl`` from a
dedicated quality replay, matches it against the first *exactly* 32 records of
one raw benchmark JSONL, and creates a new ``quality/`` directory below the
given run directory.  It never evaluates generated programs or overwrites an
existing quality result.

For GSM8K, scoring is deterministic final-number exact match.  The parser
rules are embedded in both the output manifest and summary.  For HumanEval,
the tool materializes the official ``{task_id, completion}`` sample JSONL and
writes a command hint, but deliberately does not execute untrusted generated
code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn


QUALITY_SAMPLE_COUNT = 32
SCHEMA_VERSION = 1


class QualityGuardError(RuntimeError):
    """Raised for an invalid, incomplete, or non-reproducible quality input."""


# A token is deliberately limited to an optional sign, an integer (with valid
# thousands grouping when commas are used), and an optional decimal fraction.
# The lookarounds prevent a partial match inside a word, a malformed number,
# or a longer decimal/grouped literal.  Sentence punctuation remains allowed.
FINAL_NUMBER_RE = re.compile(
    r"(?<![0-9A-Za-z_.,])"
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?![0-9A-Za-z_]|,\d|\.\d)"
)

GSM8K_PARSER_RULES = [
    "Scan the full UTF-8 completion/reference text from left to right for numeric literals: optional + or -, an integer (plain digits or valid comma-thousands groups), and optional decimal digits.",
    "Ignore numeric-looking substrings embedded in letters, longer decimals, or malformed comma groups; choose the rightmost valid numeric literal as the final number.",
    "Normalize the selected token by removing commas and a leading +, stripping leading integer zeros and trailing fractional zeros, and canonicalizing signed zero to 0.",
    "Mark a completion with no valid numeric literal incorrect; compare the two normalized final-number strings exactly, without executing model output or using floating-point arithmetic.",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> NoReturn:
    raise QualityGuardError(message)


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    _fail(f"blank JSONL record in {label} at {path}:{line_number}")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    _fail(f"invalid JSON in {label} at {path}:{line_number}: {exc.msg}")
                if not isinstance(value, dict):
                    _fail(f"JSON object required in {label} at {path}:{line_number}")
                rows.append(value)
    except OSError as exc:
        _fail(f"unable to read {label} {path}: {exc}")
    if not rows:
        _fail(f"{label} is empty: {path}")
    return rows


def _require_nonnegative_int(value: Any, *, field: str, record_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer in completion record {record_number}")
    return value


def _require_text(value: Any, *, field: str, record_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string in completion record {record_number}")
    return value


def _output_includes_prompt(record: Mapping[str, Any], record_number: int) -> bool:
    """Read the normalized output-origin flag from FastSD or SpecEdge records."""

    candidates: list[Any] = []
    if "output_includes_prompt" in record:
        candidates.append(record["output_includes_prompt"])
    client_result = record.get("client_result")
    if isinstance(client_result, Mapping) and "output_includes_prompt" in client_result:
        candidates.append(client_result["output_includes_prompt"])
    for value in candidates:
        if not isinstance(value, bool):
            _fail(
                "output_includes_prompt must be boolean when present "
                f"in completion record {record_number}"
            )
        if value:
            return True
    return False


def normalize_final_number(token: str) -> str:
    """Canonicalize one literal accepted by :data:`FINAL_NUMBER_RE` exactly."""

    raw = token.replace(",", "")
    sign = ""
    if raw.startswith(("+", "-")):
        sign, raw = raw[0], raw[1:]
    integer, dot, fraction = raw.partition(".")
    integer = integer.lstrip("0") or "0"
    if dot:
        fraction = fraction.rstrip("0")
    normalized = integer if not fraction else f"{integer}.{fraction}"
    if normalized == "0":
        return "0"
    return f"-{normalized}" if sign == "-" else normalized


def extract_final_number(text: str) -> str | None:
    """Return the canonical rightmost numeric literal, or ``None`` if absent."""

    matches = list(FINAL_NUMBER_RE.finditer(text))
    if not matches:
        return None
    return normalize_final_number(matches[-1].group("number"))


def _selected_benchmark_rows(dataset_path: Path, benchmark: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(dataset_path, label=f"raw {benchmark} dataset")
    if len(rows) < QUALITY_SAMPLE_COUNT:
        _fail(
            f"raw {benchmark} dataset has {len(rows)} records; "
            f"the quality gate requires at least {QUALITY_SAMPLE_COUNT}"
        )
    selected = rows[:QUALITY_SAMPLE_COUNT]
    if benchmark == "humaneval":
        task_ids: set[str] = set()
        for index, row in enumerate(selected):
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                _fail(f"HumanEval record {index} lacks a non-empty string task_id")
            if task_id in task_ids:
                _fail(f"duplicate HumanEval task_id in first {QUALITY_SAMPLE_COUNT}: {task_id}")
            task_ids.add(task_id)
    else:
        for index, row in enumerate(selected):
            answer = row.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                _fail(f"GSM8K record {index} lacks a non-empty string answer")
            if extract_final_number(answer) is None:
                _fail(
                    f"GSM8K reference record {index} has no valid final numeric literal"
                )
    return selected


def _match_completions(
    completions_path: Path,
    selected_rows: list[dict[str, Any]],
    *,
    benchmark: str,
) -> list[dict[str, Any]]:
    records = _read_jsonl(completions_path, label="canonical completions")
    if len(records) != QUALITY_SAMPLE_COUNT:
        _fail(
            "canonical completions must contain exactly "
            f"{QUALITY_SAMPLE_COUNT} records for the fixed quality replay; got {len(records)}"
        )

    expected_indices = set(range(QUALITY_SAMPLE_COUNT))
    matched: dict[int, dict[str, Any]] = {}
    for record_number, record in enumerate(records, 1):
        if record.get("status") != "completed":
            _fail(
                f"completion record {record_number} is not completed: "
                f"{record.get('status')!r}"
            )
        dataset_index = _require_nonnegative_int(
            record.get("dataset_index"),
            field="dataset_index",
            record_number=record_number,
        )
        if dataset_index not in expected_indices:
            _fail(
                f"completion record {record_number} uses dataset_index {dataset_index}, "
                f"outside the first {QUALITY_SAMPLE_COUNT} raw records"
            )
        if dataset_index in matched:
            _fail(f"duplicate completion for dataset_index {dataset_index}")
        if _output_includes_prompt(record, record_number):
            _fail(
                f"completion record {record_number} includes its prompt; "
                "quality scoring requires continuation-only text"
            )
        text = _require_text(record.get("text"), field="text", record_number=record_number)

        selected = selected_rows[dataset_index]
        if benchmark == "humaneval":
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                _fail(f"HumanEval completion record {record_number} lacks task_id")
            if task_id != selected["task_id"]:
                _fail(
                    f"HumanEval completion task_id mismatch at dataset_index {dataset_index}: "
                    f"expected {selected['task_id']!r}, got {task_id!r}"
                )
        matched[dataset_index] = {
            "record": record,
            "text": text,
        }

    missing = sorted(expected_indices - set(matched))
    if missing:
        _fail(f"canonical completions are missing dataset indices: {missing}")
    return [matched[index] for index in range(QUALITY_SAMPLE_COUNT)]


def _json_bytes(payload: Any, *, indent: int | None = None) -> bytes:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=indent)
    return (rendered + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _command_hints(
    *,
    benchmark: str,
    sample_path: Path | None,
    dataset_path: Path,
) -> str:
    if benchmark == "gsm8k":
        return (
            "GSM8K is scored by this guard using the deterministic final-number "
            "exact-match rules recorded in summary.json and manifest.json.\n"
            "No generated code was executed.\n"
        )
    assert sample_path is not None
    sample = shlex.quote(str(sample_path))
    problem_file = shlex.quote(str(dataset_path))
    return (
        "This guard only materialized HumanEval samples. It deliberately did not "
        "execute generated code. Run the following only inside an isolated, "
        "disposable environment with an explicit timeout and worker limit:\n\n"
        f"evaluate_functional_correctness {sample} --problem_file {problem_file} "
        "--n_workers 1 --timeout 3.0\n\n"
        "The sample JSONL has exactly the official human-eval fields task_id and "
        "completion, so it is suitable as the sample_file argument to "
        "human_eval.evaluation.evaluate_functional_correctness.\n"
    )


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite quality artifact: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as exc:
        _fail(f"unable to write new quality artifact {path}: {exc}")


def run_quality_guard(
    *,
    run_dir: Path,
    benchmark: str,
    dataset_path: Path,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Validate one fixed quality replay and write an immutable quality bundle.

    Returns the written summary mapping.  Validation is completed before the
    output directory is reserved, so invalid input cannot create a partial
    quality result.  Once a valid output directory is created, the tool never
    removes or reuses it.
    """

    if benchmark not in {"humaneval", "gsm8k"}:
        _fail(f"unsupported benchmark: {benchmark}")
    if not run_dir.is_dir() or run_dir.is_symlink():
        _fail(f"--run-dir must be an existing non-symlink directory: {run_dir}")
    if not dataset_path.is_file() or dataset_path.is_symlink():
        _fail(f"--dataset must be a regular file: {dataset_path}")

    completions_path = run_dir / "outputs" / "completions.jsonl"
    selected_rows = _selected_benchmark_rows(dataset_path, benchmark)
    matched = _match_completions(completions_path, selected_rows, benchmark=benchmark)

    scored_rows: list[dict[str, Any]] = []
    human_eval_rows: list[dict[str, str]] = []
    correct_count = 0
    unparsable_count = 0
    for dataset_index, (dataset_row, completion) in enumerate(zip(selected_rows, matched)):
        record = completion["record"]
        text = completion["text"]
        common: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": benchmark,
            "dataset_index": dataset_index,
            "request_id": record.get("request_id"),
            "status": "completed",
            "text": text,
        }
        if benchmark == "gsm8k":
            reference_number = extract_final_number(dataset_row["answer"])
            assert reference_number is not None  # validated before output reservation
            predicted_number = extract_final_number(text)
            correct = predicted_number == reference_number
            if predicted_number is None:
                unparsable_count += 1
            correct_count += int(correct)
            common.update(
                {
                    "reference_final_number": reference_number,
                    "predicted_final_number": predicted_number,
                    "correct": correct,
                    "score_status": "scored" if predicted_number is not None else "no_final_number",
                }
            )
        else:
            task_id = dataset_row["task_id"]
            human_eval_rows.append({"task_id": task_id, "completion": text})
            common.update(
                {
                    "task_id": task_id,
                    "correct": None,
                    "score_status": "not_executed",
                    "evaluation_note": "Official HumanEval functional execution was intentionally not run by this tool.",
                }
            )
        scored_rows.append(common)

    quality_dir = run_dir / "quality"
    if quality_dir.exists() or quality_dir.is_symlink():
        _fail(f"refusing to reuse existing quality directory: {quality_dir}")

    scored_path = quality_dir / "scored_completions.jsonl"
    summary_path = quality_dir / "summary.json"
    manifest_path = quality_dir / "manifest.json"
    hints_path = quality_dir / "command_hints.txt"
    samples_path = quality_dir / "humaneval_samples.jsonl" if benchmark == "humaneval" else None
    scored_payload = _jsonl_bytes(scored_rows)
    samples_payload = _jsonl_bytes(human_eval_rows) if human_eval_rows else None
    hints_payload = _command_hints(
        benchmark=benchmark,
        sample_path=samples_path,
        dataset_path=dataset_path,
    ).encode("utf-8")

    outputs: dict[str, dict[str, str]] = {
        "scored_completions": {
            "path": scored_path.name,
            "sha256": sha256_bytes(scored_payload),
        },
        "command_hints": {
            "path": hints_path.name,
            "sha256": sha256_bytes(hints_payload),
        },
    }
    if samples_path is not None and samples_payload is not None:
        outputs["humaneval_samples"] = {
            "path": samples_path.name,
            "sha256": sha256_bytes(samples_payload),
        }

    result: dict[str, Any]
    if benchmark == "gsm8k":
        result = {
            "evaluation_status": "scored",
            "total": QUALITY_SAMPLE_COUNT,
            "correct": correct_count,
            "incorrect": QUALITY_SAMPLE_COUNT - correct_count,
            "unparsable_completion_count": unparsable_count,
            "final_number_exact_match": correct_count / QUALITY_SAMPLE_COUNT,
        }
    else:
        result = {
            "evaluation_status": "not_executed",
            "total": QUALITY_SAMPLE_COUNT,
            "official_sample_file": samples_path.name if samples_path is not None else None,
            "reason": "The guard never executes generated HumanEval programs.",
        }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fastsd_quality_gate",
        "benchmark": benchmark,
        "fixed_sample_count": QUALITY_SAMPLE_COUNT,
        "selection": "first 32 non-blank JSONL records in source order",
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "selected_dataset_indices": list(range(QUALITY_SAMPLE_COUNT)),
        },
        "completions": {
            "path": str(completions_path.resolve()),
            "sha256": sha256_file(completions_path),
            "record_count": QUALITY_SAMPLE_COUNT,
        },
        "result": result,
        "outputs": outputs,
        "parser_rules": GSM8K_PARSER_RULES if benchmark == "gsm8k" else None,
        "untrusted_code_executed": False,
    }
    summary_payload = _json_bytes(summary, indent=2)
    manifest_outputs = {
        **outputs,
        "summary": {"path": summary_path.name, "sha256": sha256_bytes(summary_payload)},
    }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fastsd_quality_gate_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "fixed_sample_count": QUALITY_SAMPLE_COUNT,
        "selection": "first 32 non-blank JSONL records in source order",
        "run_dir": str(run_dir.resolve()),
        "inputs": {
            "canonical_completions": {
                "path": str(completions_path.resolve()),
                "sha256": sha256_file(completions_path),
            },
            "raw_dataset": {
                "path": str(dataset_path.resolve()),
                "sha256": sha256_file(dataset_path),
            },
        },
        "outputs": manifest_outputs,
        "parser_rules": GSM8K_PARSER_RULES if benchmark == "gsm8k" else None,
        "human_eval_evaluation_executed": False,
        "untrusted_code_executed": False,
        "command_argv": list(argv or []),
    }
    manifest_payload = _json_bytes(manifest, indent=2)

    # All validation and in-memory serialization happen before reserving the
    # result directory.  mkdir without exist_ok makes concurrent/reused runs
    # fail rather than silently join or replace one another.
    try:
        quality_dir.mkdir()
    except OSError as exc:
        _fail(f"unable to create fresh quality directory {quality_dir}: {exc}")
    _write_new(scored_path, scored_payload)
    if samples_path is not None and samples_payload is not None:
        _write_new(samples_path, samples_payload)
    _write_new(hints_path, hints_payload)
    _write_new(summary_path, summary_payload)
    _write_new(manifest_path, manifest_payload)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=("humaneval", "gsm8k"))
    parser.add_argument("--dataset", required=True, help="Raw HumanEval or GSM8K JSONL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run_quality_guard(
            run_dir=Path(args.run_dir),
            benchmark=args.benchmark,
            dataset_path=Path(args.dataset),
            argv=sys.argv,
        )
    except QualityGuardError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
