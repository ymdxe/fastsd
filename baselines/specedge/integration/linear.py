"""Grow exactly one path using the official candidate processing and Tree.add."""

from time import perf_counter

import torch


def grow_single_chain(tree, process_candidates, gamma, prefill, *, propose=None):
    forward_times = []
    for step in range(gamma):
        logits, parents, positions, scores, elapsed = process_candidates(prefill and step == 0)
        if parents.numel() != 1 or int(parents.item()) != tree.end - 1:
            raise RuntimeError("single_chain requires exactly one contiguous candidate")
        logprobs, tokens = (torch.log_softmax(logits, dim=-1).max(dim=-1)
                            if propose is None else propose(logits))
        tree.add(token_ids=tokens, token_positions=positions + 1, parent_indices=parents, logprobs=scores + logprobs)
        forward_times.append(elapsed)
    if gamma and propose is not None:
        *_, elapsed = process_candidates(False)
        forward_times.append(elapsed)
    return {"forward_t": forward_times}


class TargetOnlyClient:
    def __init__(self, tokenizer, prompt, eos_ids, validator, *, client_idx, max_len, max_new_tokens):
        self._prefix_tokens = tokenizer.encode(prompt, return_tensors="pt")
        self._num_original_tokens = self._prefix_tokens.size(-1)
        if not 2 <= self._num_original_tokens <= max_len - max_new_tokens:
            raise ValueError("prompt plus output budget exceeds max_len, or prompt has fewer than two tokens")
        self._validator = validator
        self.eos_ids = eos_ids
        self.prompt = prompt
        self.client_idx = client_idx
        self.max_len = max_len
        self.max_new_tokens = max_new_tokens
        self.first_token_time = None
        self.drafted_tokens = self.accepted_tokens = self.verify_steps = 0

    async def generate_exact(self, req_idx):
        try:
            for step in range(self.max_new_tokens):
                position = self._prefix_tokens.size(-1) - 1
                mask = (torch.arange(self.max_len) <= position).reshape(1, 1, 1, -1)
                token, _ = await self._validator.request(
                    client_idx=self.client_idx, req_idx=req_idx, input_ids=self._prefix_tokens[:, -1:],
                    position_ids=torch.tensor([[position]]), cache_seq_indices=torch.tensor([position]),
                    parent_indices=torch.empty(0, dtype=torch.long), attention_mask=mask,
                    prefill=step == 0, prefix=self.prompt if step == 0 else None,
                )
                if self.first_token_time is None:
                    self.first_token_time = perf_counter()
                self._prefix_tokens = torch.cat((self._prefix_tokens, token.reshape(1, 1)), dim=-1)
                self.verify_steps += 1
                if token.item() in self.eos_ids:
                    break
        finally:
            await self._validator.close()
