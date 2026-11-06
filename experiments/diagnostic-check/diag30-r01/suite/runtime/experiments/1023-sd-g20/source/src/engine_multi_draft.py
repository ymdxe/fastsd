import random
import time

import torch
import torch.multiprocessing as mp
from transformers.cache_utils import DynamicCache

from .cache_offload_policy import should_offload_target_cache
from .kvcache import KVCacheModel
from .util import max_fn, sample, seed_everything


def run_draft_process(args, draft_model, tokenizer, request_queue, response_queue, proc_id):
    """
    Each draft model process reads tasks independently and requests verification from the target worker.
    """
    seed_everything(42 + proc_id)
    model = KVCacheModel(draft_model, args.temp, args.top_k, args.top_p)
    model.vocab_size = tokenizer.vocab_size

    with open(args.data_path, "r") as f:
        samples = [eval(line) for line in f.readlines()]

    for sample in samples:
        input_text = sample["prompt"].strip()
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.draft_device[proc_id])
        prefix = input_ids.clone()

        while prefix.shape[1] < args.max_tokens:
            prefix_len = prefix.shape[1]
            x = model.generate(prefix, args.gamma)

            request = {
                "task_id": sample["task_id"],
                "draft_output": x.cpu(),
                "draft_cache": model._prob_history[:, prefix_len:, :].cpu(),
                "prefix_len": prefix_len,
                "proc_id": proc_id,
            }
            request_queue.put(request)
            response = response_queue.get()

            accepted = response["accepted"]
            final_token = response["final_token"]
            prefix = torch.cat((x[:, :accepted], final_token.to(x.device)), dim=1)
            model.rollback(prefix.shape[1])

        print(f"[Draft {proc_id}] Finished {sample['task_id']} ({prefix.shape[1]} tokens)\n")


def run_target_process(args, target_model, tokenizer, request_queue, response_queues):
    """
    Target model worker for the older multi-draft baseline path.
    """
    model = KVCacheModel(target_model, args.temp, args.top_k, args.top_p)
    model.vocab_size = tokenizer.vocab_size

    def move_dynamic_cache_to(cache, device):
        legacy_cache = cache.to_legacy_cache()
        current_device = legacy_cache[0][0].device
        if str(current_device) == device:
            return cache

        new_cache = []
        for key, value in legacy_cache:
            new_cache.append((key.to(device), value.to(device)))
        return DynamicCache.from_legacy_cache(new_cache)

    while True:
        request = request_queue.get()
        if request is None:
            break

        x = request["draft_output"].to(args.target_device)
        draft_cache = request["draft_cache"].to(args.target_device)
        prefix_len = request["prefix_len"]
        proc_id = request["proc_id"]

        if model._past_key_values is not None:
            model._past_key_values = move_dynamic_cache_to(model._past_key_values, args.target_device)

        _ = model.generate(x, 1)

        n = prefix_len + args.gamma - 1
        for i in range(args.gamma):
            r = torch.rand(1, device=x.device)
            j = x[:, prefix_len + i]
            p_t = model._prob_history[:, prefix_len + i - 1, j]
            p_d = draft_cache[:, i - 1, j]
            if r > p_t / p_d:
                n = prefix_len + i - 1
                break

        accepted_len = n + 1
        if accepted_len < x.shape[1]:
            new_token = sample(max_fn(model._prob_history[:, n, :] - draft_cache[:, n - prefix_len, :]))
        else:
            new_token = sample(model._prob_history[:, -1, :])

        model.rollback(accepted_len + 1 if accepted_len == x.shape[1] else accepted_len)
        if should_offload_target_cache("verify", pid=proc_id):
            model._past_key_values = move_dynamic_cache_to(model._past_key_values, "cpu")

        response_queues[proc_id].put(
            {
                "accepted": accepted_len,
                "final_token": new_token,
            }
        )
