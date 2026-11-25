"""Eager, in-process Qwen3 tensor parallelism using the official attention and KVCache."""

from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy

import torch
from torch import nn


SHARDED_PROJECTIONS = {
    "self_attn": {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 1},
    "mlp": {"gate_proj": 0, "up_proj": 0, "down_proj": 1},
}


def check_tp(config, size):
    if size == 1:
        return
    if size != 2 or config.model_type != "qwen3" or getattr(config, "use_sliding_window", False):
        raise ValueError("adapted TP supports two CUDA devices and dense Qwen3 only")
    if any(getattr(config, name) % size for name in (
        "num_attention_heads", "num_key_value_heads", "intermediate_size",
    )):
        raise ValueError("attention heads, KV heads and MLP width must divide target TP size")


def parameter_bytes(model, dtype_bytes, size):
    sizes = [0] * size
    for name, param in model.named_parameters():
        parts = name.split(".")
        block = parts[-3] if len(parts) >= 3 else ""
        projections = SHARDED_PROJECTIONS.get(block, {})
        sharded = size > 1 and parts[-2] in projections
        replicated = size > 1 and block == "self_attn" and parts[-2] in {"q_norm", "k_norm"}
        if sharded and not (parts[-1] == "bias" and projections[parts[-2]] == 1):
            for rank in range(size):
                sizes[rank] += param.numel() // size * dtype_bytes
        elif replicated:
            for rank in range(size):
                sizes[rank] += param.numel() * dtype_bytes
        else:
            sizes[0] += param.numel() * dtype_bytes
    return sizes


def shard_linear(linear, dim, rank, devices):
    device = devices[rank]
    weight = linear.weight.detach().chunk(len(devices), dim=dim)[rank].to(device, copy=True)
    bias = linear.bias
    if bias is not None:
        bias = bias.detach().chunk(len(devices))[rank] if dim == 0 else (bias.detach() if rank == 0 else None)
    model = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    model.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        model.bias = nn.Parameter(bias.to(device, copy=True), requires_grad=False)
    return model


def split_block(block, projections, devices):
    shards = []
    for rank, device in enumerate(devices):
        shard = copy(block)
        shard._modules = {
            name: shard_linear(layer, projections[name], rank, devices)
            if name in projections else deepcopy(layer).to(device)
            for name, layer in block.named_children()
        }
        shards.append(shard)
    return nn.ModuleList(shards)


class TPBlock(nn.Module):
    def __init__(self, block, name, devices):
        super().__init__()
        self.shards = split_block(block, SHARDED_PROJECTIONS[name], devices)
        self.devices = devices

    def forward(self, hidden_states, **kwargs):
        outputs = []
        cache = kwargs.get("past_key_values")
        for rank, (device, shard) in enumerate(zip(self.devices, self.shards)):
            args = cache.attention_args(rank, kwargs) if cache is not None else {}
            outputs.append(shard(hidden_states.to(device, non_blocking=True), **args))
        return outputs[0] + outputs[1].to(self.devices[0], non_blocking=True)


def shard_model(model, devices):
    check_tp(model.config, len(devices))
    tied_embeddings = model.lm_head.weight is model.model.embed_tokens.weight
    for layer in model.model.layers:
        for name in SHARDED_PROJECTIONS:
            setattr(layer, name, TPBlock(getattr(layer, name), name, devices))
        layer.input_layernorm.to(devices[0])
        layer.post_attention_layernorm.to(devices[0])
    model.model.embed_tokens.to(devices[0])
    model.model.norm.to(devices[0])
    model.model.rotary_emb.to(devices[0])
    if tied_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        model.lm_head.to(devices[0])
    return model.eval()


class ShardedSlots:
    """Only per-slot indexing/copy is exposed to the official verifier."""

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def __getitem__(self, key):
        layers, slot, *rest = key
        if isinstance(slot, torch.Tensor):
            slot = int(slot.item())
        return ShardedSlots(tensor[layers, slot, *rest] for tensor in self.tensors)

    def copy_(self, source):
        for dest, origin in zip(self.tensors, source.tensors, strict=True):
            dest.copy_(origin)
        return self


class ShardedCache:
    def __init__(self, config, max_len, width, batch_size, devices, dtype):
        from model.cache import KVCache

        shard_config = copy(config)
        shard_config.num_key_value_heads //= len(devices)
        self.shards = [KVCache(shard_config, width, max_len, device, dtype, batch_size) for device in devices]
        self.cached_args = {}

    @property
    def k_cache(self):
        return ShardedSlots(cache.k_cache for cache in self.shards)

    @property
    def v_cache(self):
        return ShardedSlots(cache.v_cache for cache in self.shards)

    def attention_args(self, rank, kwargs):
        if rank not in self.cached_args:
            cache = self.shards[rank]
            self.cached_args[rank] = dict(
                past_key_values=cache,
                position_embeddings=tuple(tensor.to(cache.device, non_blocking=True) for tensor in kwargs["position_embeddings"]),
                **{name: kwargs[name].to(cache.device, non_blocking=True) for name in (
                    "cache_batch_indices", "cache_seq_indices", "attention_mask",
                )},
            )
        return self.cached_args[rank]

    def clear(self):
        for cache in self.shards:
            cache.clear()

    def gather(self, batch_idx, src_indices, dest_indices):
        for cache in self.shards:
            cache.gather(batch_idx, src_indices, dest_indices)

    @contextmanager
    def prefill_context(self, length, batch_idx):
        with ExitStack() as stack:
            for cache in self.shards:
                stack.enter_context(cache.prefill_context(length, batch_idx))
            yield


class TPBatchEngine:
    def __init__(self, model, max_len, max_batch_size, max_n_beams, devices, *, invert_mask):
        self.model = model
        self.devices = devices
        self.invert_mask = invert_mask
        self._past_key_values = ShardedCache(model.config, max_len, max_n_beams, max_batch_size, devices, model.dtype)

    @torch.inference_mode()
    def forward(self, **kwargs):
        kwargs["attention_mask"] = self.invert_mask(kwargs["attention_mask"])
        try:
            return self.model(**kwargs, past_key_values=self._past_key_values)[0]
        finally:
            self._past_key_values.cached_args.clear()

    def prefill(self, batch_idx, **kwargs):
        length = kwargs["input_ids"].size(-1)
        if length:
            kwargs["cache_batch_indices"] = torch.zeros(length, dtype=torch.long, device=self.devices[0])
            with self._past_key_values.prefill_context(length, batch_idx):
                self.forward(**kwargs)

    def gather(self, batch_idx, src_indices, dest_indices):
        self._past_key_values.gather(batch_idx, src_indices, dest_indices)
