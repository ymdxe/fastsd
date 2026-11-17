#!/usr/bin/env python3
"""Convert FastSD JSONL datasets to a conservative SpecEdge input JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prompt_for(sample: dict, dataset: str, turn: int) -> tuple[str, str]:
    if dataset == "humaneval":
        return str(sample["prompt"]), str(sample.get("task_id", sample.get("sample_id", "")))
    if dataset in {"gsm8k", "mgsm"}:
        key = "question"
        return str(sample[key]), str(sample.get("task_id", sample.get("question_id", sample.get("sample_id", ""))))
    turns = sample.get("turns") or []
    if not turns:
        raise ValueError("MT-Bench sample has no turns")
    index = min(max(0, turn), len(turns) - 1)
    task = str(sample.get("task_id", sample.get("sample_id", "")))
    return str(turns[index]), f"{task}-turn{index + 1}"


def convert(input_path: Path, output_path: Path, dataset: str, turns: int) -> int:
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        for turn in range(turns if dataset == "mt_bench" else 1):
            prompt, sample_id = prompt_for(sample, dataset, turn)
            rows.append({
                "sample_id": sample_id,
                "prompt": prompt,
                "reference": sample.get("reference", sample.get("answer")),
                "dataset": dataset,
                "turn": turn + 1 if dataset == "mt_bench" else 1,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "gsm8k", "mgsm", "mt_bench"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mt-bench-turns", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    count = convert(args.input, args.output, args.dataset, args.mt_bench_turns)
    print(json.dumps({"dataset": args.dataset, "requests": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
