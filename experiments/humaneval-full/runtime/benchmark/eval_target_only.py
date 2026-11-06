"""Run the target-only greedy oracle for token parity and quality reference.

This is not one of the four reported methods.  It is a single target-side
process that reads the same canonical JSONL and writes token IDs that can be
used by ``scripts/eval_suite.py parity``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from benchmark.eval_draft_pool import _encode_prompt, _generate, _load_model
from src.common_metrics import summarize_requests, write_json
from src.evaluation import load_canonical_jsonl
from src.run_artifacts import append_command, append_status, write_text_once
from src.runtime import configure_torch_threads


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_root = REPO_ROOT / "exp" / "comparison" / config["run_id"]
    canonical_path = run_root / "inputs" / "canonical.jsonl"
    manifest_path = run_root / "run_manifest.json"
    if not canonical_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run eval_suite.py prepare before target-only evaluation")

    records = load_canonical_jsonl(canonical_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    topology = config["topology"]
    device = str(
        topology.get(
            "target_only_device",
            topology.get("standard_sd_target_device", "cuda:0"),
        )
    )
    threads = topology.get("target_threads")
    configure_torch_threads(int(threads) if threads is not None and device == "cpu" else None)
    model = _load_model(
        config["models"]["target"],
        device,
        config["generation"].get("target_dtype", config.get("execution", {}).get("target_dtype", "bfloat16")),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["models"]["target"], trust_remote_code=True)
    warmup_requests = int(topology.get("warmup_requests", 0))
    if warmup_requests < 0:
        raise ValueError("topology.warmup_requests must be non-negative")
    if warmup_requests and records:
        warmup_generation = {**config["generation"], "temperature": 0.0, "max_new_tokens": 1}
        warmup_input = _encode_prompt(tokenizer, records[0]["prompt"], records[0]["dataset"])
        for warmup_idx in range(warmup_requests):
            _generate(model, warmup_input, tokenizer, warmup_generation)
    output_dir = run_root / "target_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "requests.jsonl"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite target-only output: {output_path}")

    generated_records = []
    run_start = time.perf_counter()
    for record in records:
        request_start = time.perf_counter()
        input_ids = _encode_prompt(tokenizer, record["prompt"], record["dataset"])
        generated, metrics = _generate(
            model,
            input_ids,
            tokenizer,
            {
                **config["generation"],
                "temperature": 0.0,
            },
        )
        generated_ids = generated[0, input_ids.shape[1] :]
        request_e2e_ms = (time.perf_counter() - request_start) * 1000.0
        generated_records.append(
            {
                "schema_version": 1,
                "sample_id": record["sample_id"],
                "global_index": int(record["global_index"]),
                "dataset": record["dataset"],
                "method": "target_only",
                "workload_hash": manifest["workload_hash"],
                "generated_tokens": int(generated_ids.numel()),
                "ttft_ms": float(metrics["ttft_ms"]),
                "tpot_ms": float(metrics["tpot_ms"]),
                "e2e_ms": request_e2e_ms,
                "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "output_token_ids": [int(token_id) for token_id in generated_ids.tolist()],
                "reference": record.get("reference"),
            }
        )

    write_text_once(
        output_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in generated_records),
    )
    summary = summarize_requests(
        generated_records,
        method="target_only",
        dataset=config["dataset"]["name"],
        workload_hash=manifest["workload_hash"],
        run_id=config["run_id"],
        wallclock_s=time.perf_counter() - run_start,
        evaluation_scope=(
            "communication_smoke"
            if int(config["generation"].get("max_new_tokens", 0)) <= 16
            else "quality"
        ),
        extra={
            "oracle": True,
            "device": device,
            "dtype": config["generation"].get("target_dtype", "bfloat16"),
            "warmup_requests": warmup_requests,
            "warmup_included_in_metrics": False,
        },
    )
    write_json(summary, output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        exit_code = run(args.config)
    except Exception as exc:
        failed_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        append_status(
            REPO_ROOT / "exp" / "comparison" / failed_config["run_id"] / "run_status.jsonl",
            method="target_only",
            phase="run",
            exit_code=1,
            error=repr(exc),
        )
        append_command(
            REPO_ROOT / "exp" / "comparison" / failed_config["run_id"] / "commands.txt",
            shlex.join(sys.argv),
            status=1,
            note="target_only",
        )
        raise
    append_status(
        REPO_ROOT / "exp" / "comparison" / json.loads(Path(args.config).read_text(encoding="utf-8"))["run_id"] / "run_status.jsonl",
        method="target_only",
        phase="run",
        exit_code=exit_code,
    )
    success_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    success_root = REPO_ROOT / "exp" / "comparison" / success_config["run_id"]
    append_command(
        success_root / "commands.txt",
        shlex.join(sys.argv),
        status=exit_code,
        note="target_only",
    )
    raise SystemExit(exit_code)
