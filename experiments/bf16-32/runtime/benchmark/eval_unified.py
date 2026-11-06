"""Unified local benchmark for standard speculative decoding and draft-only AR."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(sys.path[0], "../"))

import torch

from src.common_metrics import summarize_requests, write_json
from src.engine import Decoding
from src.evaluation import load_canonical_jsonl
from src.util import parse_arguments, seed_everything


class UnifiedEvaluation(Decoding):
    def __init__(self, args):
        super().__init__(args)
        if args.eval_mode not in {"small", "sd"}:
            raise ValueError("eval_unified.py supports only eval_mode=small or eval_mode=sd")
        if not args.dataset_file:
            raise ValueError("--dataset_file is required; run scripts/eval_suite.py prepare")
        self.load_tokenizer()
        self.load_data()
        self.load_model()

    def load_data(self):
        self.data = load_canonical_jsonl(self.args.dataset_file)

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    @torch.no_grad()
    def eval(self):
        decoding = self.autoregressive_sampling if self.args.eval_mode == "small" else self.speculative_decoding
        method = "draft_only" if self.args.eval_mode == "small" else "standard_sd"
        requests_path = Path(self.args.exp_name) / "requests.jsonl"
        records = []
        run_start = time.monotonic()

        with requests_path.open("w", encoding="utf-8") as output:
            for record in self.data:
                scheduled = float(record.get("scheduled_arrival_s", 0.0))
                remaining = scheduled - (time.monotonic() - run_start)
                if remaining > 0:
                    time.sleep(remaining)
                actual_arrival = max(0.0, time.monotonic() - run_start)

                input_ids = self.tokenizer.encode(
                    record["prompt"], return_tensors="pt", add_special_tokens=True
                )
                seed_everything(self.args.seed + int(record.get("global_index", 0)))
                generated = decoding(input_ids)
                metrics = dict(self.last_generation_metrics)
                new_tokens = generated[0, input_ids.shape[1] :]
                completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                completion_s = max(0.0, time.monotonic() - run_start)
                request_record = {
                    "schema_version": 1,
                    "sample_id": record["sample_id"],
                    "dataset": record["dataset"],
                    "method": method,
                    "workload_hash": self.args.workload_hash,
                    "scheduled_arrival_s": scheduled,
                    "actual_arrival_s": actual_arrival,
                    "completion_s": completion_s,
                    "arrival_lag_ms": max(0.0, (actual_arrival - scheduled) * 1000.0),
                    "generated_tokens": int(metrics["generated_tokens"]),
                    "ttft_ms": float(metrics["ttft_ms"]),
                    "tpot_ms": float(metrics["tpot_ms"]),
                    "e2e_ms": float(metrics["e2e_ms"]),
                    "output_text": completion,
                    "reference": record.get("reference"),
                }
                output.write(json.dumps(request_record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(request_record)

        wallclock_s = max(0.0, time.monotonic() - run_start)
        summary = summarize_requests(
            records,
            method=method,
            dataset=self.args.dataset,
            workload_hash=self.args.workload_hash or "unknown",
            run_id=Path(self.args.exp_name).name,
            wallclock_s=wallclock_s,
            extra={
                "draft_model": self.args.draft_model,
                "target_model": self.args.target_model if method == "standard_sd" else None,
                "gamma": self.args.gamma if method == "standard_sd" else None,
            },
        )
        write_json(summary, Path(self.args.exp_name) / "common_summary.json")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    arguments = parse_arguments()
    UnifiedEvaluation(arguments).eval()
