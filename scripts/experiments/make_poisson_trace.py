#!/usr/bin/env python3
"""Create one immutable shared Poisson workload for cloud-edge experiments.

The script never cycles a dataset and never overwrites an existing output
directory.  A generated trace is therefore a stable experimental input that
FastSD, Vanilla, and Official SpecEdge can replay byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arrival import build_trace_manifest, generate_poisson_trace, sha256_file, write_trace_jsonl


def _hdd_path(path: Path) -> None:
    resolved = path.resolve()
    hdd = Path("/home/hdd")
    if resolved != hdd and hdd not in resolved.parents:
        raise ValueError(f"experiment output must be below /home/hdd: {resolved}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"dataset does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"dataset record at {path}:{line_number} must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def _raw_prompt(record: dict[str, Any], dataset_format: str) -> str:
    if dataset_format == "mt_bench_first_turn_qwen3":
        turns = record.get("turns")
        if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
            raise ValueError("MT-Bench record must contain a non-empty string turns[0]")
        return turns[0].strip()
    if dataset_format == "gsm8k_question":
        question = record.get("question")
        if not isinstance(question, str):
            raise ValueError("GSM8K record must contain string question")
        return question.strip()
    if dataset_format == "humaneval_prompt":
        prompt = record.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("HumanEval record must contain string prompt")
        return prompt.rstrip()
    raise ValueError(f"unsupported dataset format: {dataset_format}")


def _render_qwen_prompts(
    records: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    dataset_format: str,
    tokenizer_model: str | None,
) -> list[dict[str, Any]]:
    tokenizer = None
    if tokenizer_model:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on experiment env
            raise RuntimeError("transformers is required to render prompts") from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

    prompts: list[dict[str, Any]] = []
    for row in trace:
        dataset_index = int(row["dataset_index"])
        record = records[dataset_index]
        raw = _raw_prompt(record, dataset_format)
        prompt = raw
        if tokenizer is not None and dataset_format == "mt_bench_first_turn_qwen3":
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Older Transformers templates may not accept Qwen's
                # enable_thinking keyword.  Do not silently use a different
                # role format; preserve the common generation prompt behavior.
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        prompts.append(
            {
                "arrival_index": int(row["arrival_index"]),
                "dataset_index": dataset_index,
                "task_id": row["task_id"],
                "prompt": prompt,
                "raw_prompt": raw,
                "dataset_format": dataset_format,
            }
        )
    return prompts


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("mt_bench_first_turn_qwen3", "gsm8k_question", "humaneval_prompt"),
        default="mt_bench_first_turn_qwen3",
    )
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--num-clients", "--clients", type=int, default=2)
    parser.add_argument("--rate-rps", "--request-rate-rps", type=float, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--seed", "--arrival-seed", type=int, required=True)
    parser.add_argument("--output", "--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    _hdd_path(output)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"ERROR: refusing to reuse workload output: {output}")
    records = _read_jsonl(Path(args.dataset))
    trace = generate_poisson_trace(
        records,
        rate_rps=args.rate_rps,
        seed=args.seed,
        num_clients=args.num_clients,
        max_requests=args.max_requests,
    )
    prompts = _render_qwen_prompts(
        records,
        trace,
        dataset_format=args.dataset_format,
        tokenizer_model=args.tokenizer_model,
    )
    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "arrival_trace.jsonl"
    prompt_path = output / "prompts.jsonl"
    write_trace_jsonl(trace_path, trace)
    _write_jsonl_new(prompt_path, prompts)
    manifest = build_trace_manifest(trace_path)
    manifest.update(
        {
            "dataset": str(Path(args.dataset).resolve()),
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_format": args.dataset_format,
            "tokenizer_model": args.tokenizer_model,
            "prompt_sha256": sha256_file(prompt_path),
            "rate_rps": float(args.rate_rps),
            "arrival_seed": int(args.seed),
            "num_clients": int(args.num_clients),
        }
    )
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
