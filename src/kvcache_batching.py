import torch
from transformers.cache_utils import DynamicCache

from .util import norm_logits


class KVCacheModel_batching:
    """Pack valid tokens into one native model forward, without padding.

    Cached tokens precede all new tokens in the packed cache. A block-causal
    mask isolates devices; position_ids retain each device's own positions.
    Model layers and weights remain owned by the Transformers model.
    """

    def __init__(self, model, temperature=1, top_k=0, top_p=0):
        self._model = model
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._past_key_values = {}
        self._prob_history = {}
        self.vocab_size = model.config.vocab_size

    def reset(self, proc_id):
        self._past_key_values.pop(proc_id, None)
        self._prob_history.pop(proc_id, None)

    @torch.no_grad()
    def forward(self, tokens, proc_ids):
        device = self._model.device
        lengths = [ids.numel() for ids in tokens]
        caches = [self._past_key_values.get(pid) for pid in proc_ids]
        cached = [0 if cache is None else cache.get_seq_length() for cache in caches]
        past_total = sum(cached)
        packed = DynamicCache()
        if past_total:
            histories = [cache.to_legacy_cache() for cache in caches if cache is not None]
            for layer_index, layers in enumerate(zip(*histories)):
                packed.update(torch.cat([layer[0] for layer in layers], dim=-2),
                              torch.cat([layer[1] for layer in layers], dim=-2), layer_index)

        ids = torch.cat([ids.reshape(-1).to(device) for ids in tokens]).unsqueeze(0)
        query_positions = torch.cat([torch.arange(old, old + length, device=device)
                                     for old, length in zip(cached, lengths)])
        query_devices = torch.repeat_interleave(torch.arange(len(proc_ids), device=device),
                                               torch.tensor(lengths, device=device))
        key_devices = torch.cat((torch.repeat_interleave(
            torch.arange(len(proc_ids), device=device), torch.tensor(cached, device=device)), query_devices))
        key_positions = torch.cat([torch.arange(old, device=device) for old in cached] + [query_positions])
        allowed = ((query_devices[:, None] == key_devices[None, :]) &
                   (query_positions[:, None] >= key_positions[None, :]))
        mask = torch.zeros((1, 1, ids.numel(), past_total + ids.numel()),
                           dtype=self._model.dtype, device=device)
        mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        result = self._model(ids, past_key_values=packed, attention_mask=mask,
                             position_ids=query_positions.unsqueeze(0), use_cache=True)
        layers = result.past_key_values.to_legacy_cache()
        probabilities = norm_logits(result.logits[0, :, :self.vocab_size].float(),
                                    self._temperature, self._top_k, self._top_p)
        old_start = new_start = 0
        for pid, old, length in zip(proc_ids, cached, lengths):
            positions = torch.cat((torch.arange(old_start, old_start + old, device=device),
                                   torch.arange(past_total + new_start,
                                                past_total + new_start + length, device=device)))
            cache = DynamicCache()
            for layer_index, (key, value) in enumerate(layers):
                # Independent storage permits per-device offload and rollback.
                cache.update(key.index_select(-2, positions), value.index_select(-2, positions), layer_index)
            rows = probabilities[new_start:new_start + length].unsqueeze(0)
            previous = self._prob_history.get(pid)
            self._prob_history[pid] = rows.clone() if previous is None else torch.cat((previous, rows), dim=1)
            self._past_key_values[pid] = cache
            old_start += old
            new_start += length

    def rollback(self, proc_id, end_pos):
        self._past_key_values[proc_id].crop(end_pos)
        self._prob_history[proc_id] = self._prob_history[proc_id][:, :end_pos, :]
