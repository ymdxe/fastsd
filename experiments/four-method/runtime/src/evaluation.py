"""Shared datasets and workload manifests for cross-method evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .arrival import poisson_arrival_offsets


DATASET_FILES = {
    "humaneval": "humaneval.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "mgsm": "mgsm.jsonl",
    "mt_bench": "mt_bench.jsonl",
}


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    dataset: str
    prompt: str
    reference: Any
    scheduled_arrival_s: float = 0.0
    turn_index: int = 0
    turn_count: int = 1


def _dataset_file(dataset: str, data_path: str | Path) -> Path:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    path = Path(data_path)
    return path / DATASET_FILES[dataset] if path.is_dir() else path


def _record_from_raw(dataset: str, raw: dict[str, Any], index: int) -> EvaluationRecord:
    if dataset == "humaneval":
        sample_id = str(raw.get("task_id", f"humaneval-{index}"))
        prompt = str(raw["prompt"]).strip()
        reference = {
            "entry_point": raw.get("entry_point"),
            "canonical_solution": raw.get("canonical_solution"),
            "test": raw.get("test"),
        }
    elif dataset == "gsm8k":
        sample_id = str(raw.get("task_id", f"gsm8k-{index}"))
        prompt = str(raw["question"]).strip()
        reference = raw.get("answer")
    elif dataset == "mgsm":
        sample_id = str(raw.get("question_id", f"mgsm-{index}"))
        prompt = str(raw["question"]).strip()
        reference = raw.get("answer")
    elif dataset == "mt_bench":
        sample_id = str(raw.get("question_id", f"mtbench-{index}"))
        turns = raw.get("turns") or []
        if not turns:
            raise ValueError(f"MT-Bench sample {sample_id} has no turns")
        prompt = str(turns[0]).strip()
        reference = {"category": raw.get("category"), "turns": turns}
    else:  # pragma: no cover - guarded by _dataset_file
        raise ValueError(f"Unsupported dataset: {dataset}")
    turn_count = len(raw.get("turns") or []) if dataset == "mt_bench" else 1
    return EvaluationRecord(
        sample_id,
        dataset,
        prompt,
        reference,
        turn_index=0,
        turn_count=max(1, turn_count),
    )


def load_evaluation_records(
    dataset: str,
    data_path: str | Path,
    max_requests: int = -1,
    arrival_distribution: str = "immediate",
    arrival_rate: float = 1.0,
    arrival_seed: int = 1234,
) -> list[EvaluationRecord]:
    path = _dataset_file(dataset, data_path)
    records: list[EvaluationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                records.append(_record_from_raw(dataset, json.loads(line), index))
    if max_requests > 0:
        records = records[:max_requests]

    if arrival_distribution == "poisson":
        offsets = poisson_arrival_offsets(len(records), arrival_rate, arrival_seed)
    elif arrival_distribution == "immediate":
        offsets = [0.0] * len(records)
    else:
        raise ValueError(f"Unsupported arrival distribution: {arrival_distribution}")

    return [
        EvaluationRecord(
            sample_id=record.sample_id,
            dataset=record.dataset,
            prompt=record.prompt,
            reference=record.reference,
            scheduled_arrival_s=float(offsets[index]),
            turn_index=record.turn_index,
            turn_count=record.turn_count,
        )
        for index, record in enumerate(records)
    ]


def write_canonical_jsonl(records: Iterable[EvaluationRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for global_index, record in enumerate(records):
            payload = asdict(record)
            payload["global_index"] = global_index
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output


def load_canonical_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def workload_fingerprint(records: Iterable[EvaluationRecord], generation: dict[str, Any]) -> str:
    payload = {
        "records": [
            {
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "prompt": record.prompt,
                "scheduled_arrival_s": round(record.scheduled_arrival_s, 9),
            }
            for record in records
        ],
        "generation": generation,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
