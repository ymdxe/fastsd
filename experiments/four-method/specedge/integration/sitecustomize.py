"""Runtime-only dataset hook for the untouched official SpecEdge server."""

from __future__ import annotations

import json
import os
from pathlib import Path


dataset_file = os.environ.get("FASTSD_EVAL_DATASET_FILE")
if dataset_file and os.environ.get("FASTSD_EVAL_ROLE") == "server":
    import util as _official_util

    _original_load_dataset = _official_util.load_dataset

    def _load_dataset(name: str, model_name=None, reasoning=False):
        if name != "fastsd_external":
            return _original_load_dataset(name, model_name=model_name, reasoning=reasoning)
        path = Path(dataset_file)
        with path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records or records[0].get("dataset") != "mt_bench":
            return [record["prompt"] for record in records]
        tokenizer = _official_util.load_tokenizer(model_name)
        prompts = []
        for record in records:
            messages = [{"role": "user", "content": record["prompt"]}]
            template_args = {"tokenize": False, "add_generation_prompt": True}
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_args,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, **template_args)
            prompts.append(prompt)
        return prompts

    _official_util.load_dataset = _load_dataset
