"""Padding-free batched forward for the FastSD cloud target.

Replaces the pad-to-max dense path of ``KVCacheModel_batching.generate`` with a
variable-length forward that never feeds padding tokens into the model:

- linear layers (layernorm, projections, MLP) run once over the concatenation
  of all real tokens;
- attention runs per sequence with its own q/k lengths against its own KV
  cache, so no KV zero-padding and no per-step full-KV copy alignment;
- per-proc KV caches and prob history keep exactly the semantics of
  ``KVCacheModel_batching``: ``_past_key_values`` and ``_prob_history`` are
  retain the same semantics and the returned list matches ``generate``.

Callers (``engine.handle_request_batch``) are responsible for slicing each
sequence to its *residual* (the tokens not yet covered by that proc's cache):

- prefill fresh        -> full prompt (cache empty / reset)
- prefill continuation -> the new chunk (``x[:, cached_len:]``)
- verify tail-only     -> the received tail segment as-is
- verify full-prefix   -> tokens beyond the cached length (``x[:, cached_len:]``)

The padded implementation remains available in ``kvcache_batching.py`` and is
selected with ``--kv_batch_mode padded``.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .util import norm_logits, sample
from .cache_views import dynamic_cache_from_views



def sliding_attention_layers(model):
    """Inspect effective layer types, not a dormant sliding_window setting."""
    config = getattr(model, "config", None)
    types = getattr(config, "layer_types", None)
    enabled = getattr(config, "use_sliding_window", None)
    found = []
    for index, layer in enumerate(getattr(getattr(model, "model", None), "layers", ())):
        attention = getattr(layer, "self_attn", None)
        layer_type = getattr(attention, "attention_type", getattr(layer, "attention_type", None))
        declared = types[index] if types is not None and index < len(types) else None
        actual_window = getattr(attention, "sliding_window", None)
        # Effective per-layer settings take precedence over a disabled global
        # default: mixed checkpoints may explicitly instantiate sliding layers.
        active = (layer_type == "sliding_attention" or declared == "sliding_attention"
                  or (actual_window is not None and actual_window > 0))
        if not active and declared != "full_attention" and layer_type != "full_attention" and enabled:
            window = getattr(config, "sliding_window", None)
            max_layers = getattr(config, "max_window_layers", None)
            active = bool(window and (max_layers is None or index < max_layers))
        if active:
            found.append(index)
    return found


def require_full_attention(model):
    if sliding_attention_layers(model):
        raise ValueError("FastSD KV backends currently support full causal attention only; "
                         "this checkpoint enables sliding_attention layers. Sliding mask, "
                         "cache logical lengths and rollback are not supported.")


def supports_varlen_model(model) -> bool:
    """Recognize the decoder layout used by Llama and the existing Qwen3 path.

    Inspect the loaded modules as well as model_type: quantization wrappers
    and custom remote-code models need not expose the same decoder layout.
    Dynamic RoPE changes cached-key semantics and is not supported here.
    """
    if sliding_attention_layers(model):
        return False
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", ())
    rope = getattr(config, "rope_scaling", None) or {}
    if model_type not in {"llama", "qwen3"} or not layers:
        return False
    if rope.get("rope_type", rope.get("type", "default")) not in {"default", "linear"}:
        return False
    if not all(hasattr(decoder, name) for name in ("embed_tokens", "norm", "rotary_emb")):
        return False
    if not hasattr(model, "lm_head"):
        return False
    for layer in layers:
        if not all(hasattr(layer, name) for name in (
            "self_attn", "input_layernorm", "post_attention_layernorm", "mlp"
        )):
            return False
        names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if model_type == "qwen3":
            names += ["q_norm", "k_norm"]
        if not all(hasattr(layer.self_attn, name) for name in names):
            return False
    return True


@torch.no_grad()
def varlen_generate(
    kv_manager,
    model,
    residuals,      # list[Tensor]: each (1, r_i) on the model device
    proc_ids,       # list[Any] aligned with residuals
    pad_token_id,   # signature parity with KVCacheModel_batching.generate; unused
    is_prefill: bool = False,
    input_lens=None,  # signature parity; unused (no padding in this path)
    sample_next: bool = True,
):
    """Padding-free forward for one mixed batch.

    Mirrors ``KVCacheModel_batching.generate``: appends per-proc logits to
    ``_prob_history`` (with per-position ``norm_logits``), appends the new K/V
    to each proc's ``DynamicCache``, and returns the per-proc extended
    sequences (input + sampled token), identical in structure to ``generate``.
    """
    del pad_token_id, input_lens  # no padding / no flag split in this path
    if len(residuals) != len(proc_ids):
        raise ValueError("residuals and proc_ids must be aligned")
    if not residuals:
        return []
    if not supports_varlen_model(model):
        raise ValueError("varlen requires a supported Llama/Qwen3 decoder with default or linear RoPE")
    if len(set(proc_ids)) != len(proc_ids):
        raise ValueError("proc_ids must be unique within a varlen batch")
    if any(res.numel() == 0 for res in residuals):
        raise ValueError("each varlen residual must contain at least one token")

    vocab_size = kv_manager.vocab_size
    temperature = kv_manager._temperature
    top_k = kv_manager._top_k
    top_p = kv_manager._top_p
    device = model.device

    # ---- 1. per-seq cached lengths & flattened input ----------------------
    cached_lens = []
    flat_parts = []
    old_caches = []
    old_buffers = []
    for pid, res in zip(proc_ids, residuals):
        cache = kv_manager._past_key_values.get(pid)
        # A missing cache means fresh prefill: the engine resets (pops) the
        # entry before admission, exactly like the padded path.
        cached_lens.append(0 if cache is None else int(cache.get_seq_length()))
        old_caches.append(cache.to_legacy_cache() if cache is not None else None)
        old_buffers.append(getattr(cache, "_fastsd_buffers", None))
        flat_parts.append(res.to(device).reshape(-1))
    flat_ids = torch.cat(flat_parts, dim=0)  # (total,)

    seq_ranges = []
    start = 0
    for res, cached_len in zip(residuals, cached_lens):
        length = int(res.numel())
        seq_ranges.append((start, start + length, cached_len))
        start += length

    # ---- 2. layer-by-layer forward -----------------------------------------
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.model.layers[0].self_attn, "head_dim", model.config.hidden_size // n_heads)

    hidden_states = model.model.embed_tokens(flat_ids)

    # Positions and continuation masks are invariant across decoder layers.
    positions = torch.cat([
        torch.arange(cached_len, cached_len + e - s, device=device)
        for s, e, cached_len in seq_ranges
    ]).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_states.unsqueeze(0), positions)
    masks = []
    for s, e, cached_len in seq_ranges:
        length = e - s
        masks.append(
            (torch.arange(length, device=device)[:, None] + cached_len
             >= torch.arange(cached_len + length, device=device)[None, :])
            if cached_len and length > 1 else None
        )

    # Retain the very same full K/V used by attention for cache writeback.
    # Do not concatenate the entire history a second time after the forward.
    new_kv_per_layer = []
    buffers_per_layer = []
    block_size = max(1, int(os.environ.get("FASTSD_KV_BLOCK_TOKENS", "128")))
    native_gqa = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")
    native_gqa = native_gqa and os.environ.get("FASTSD_NATIVE_GQA", "1") != "0"

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden_states)

        q = attn.q_proj(normed).view(-1, n_heads, head_dim)
        k = attn.k_proj(normed).view(-1, n_kv_heads, head_dim)
        v = attn.v_proj(normed).view(-1, n_kv_heads, head_dim)
        if model.config.model_type == "qwen3":
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2), cos, sin,
        )

        attn_out = normed.new_empty((flat_ids.numel(), n_heads * head_dim))
        new_kv_layer = []
        buffer_layer = []
        for i, (s, e, cached_len) in enumerate(seq_ranges):
            seq_len = e - s
            qi = q[:, :, s:e, :]
            ki = k[:, :, s:e, :]
            vi = v[s:e].unsqueeze(0).transpose(1, 2)
            required = cached_len + seq_len
            existing = old_buffers[i][layer_idx] if old_buffers[i] is not None else None
            if existing is not None and existing[0].shape[-2] >= required:
                k_buffer, v_buffer = existing
            else:
                capacity = ((required + block_size - 1) // block_size) * block_size
                shape = (1, n_kv_heads, capacity, head_dim)
                k_buffer, v_buffer = ki.new_empty(shape), vi.new_empty(shape)
                if cached_len:
                    k_old, v_old = old_caches[i][layer_idx]
                    k_buffer[:, :, :cached_len].copy_(k_old)
                    v_buffer[:, :, :cached_len].copy_(v_old)
            k_buffer[:, :, cached_len:required].copy_(ki)
            v_buffer[:, :, cached_len:required].copy_(vi)
            k_full, v_full = k_buffer[:, :, :required], v_buffer[:, :, :required]
            buffer_layer.append((k_buffer, v_buffer))

            # Use the native GQA path where supported; older PyTorch keeps
            # the explicit expansion fallback. Stored KV always uses KV heads.
            n_groups = n_heads // n_kv_heads
            k_exp = k_full.repeat_interleave(n_groups, dim=1) if n_groups > 1 and not native_gqa else k_full
            v_exp = v_full.repeat_interleave(n_groups, dim=1) if n_groups > 1 and not native_gqa else v_full

            # Causal masking by *global* position: query global position
            # (cached_len + i) may attend key global positions <= itself.
            # ``is_causal`` uses local indices and is only correct for a
            # fresh prefill; multi-query cached continuations need a mask.
            # A single last-position query may attend the entire KV history.
            o = F.scaled_dot_product_attention(
                qi, k_exp, v_exp, attn_mask=masks[i],
                is_causal=cached_len == 0,
                scale=getattr(attn, "scaling", head_dim ** -0.5),
                **({"enable_gqa": True} if native_gqa and n_groups > 1 else {}),
            )
            o = o.transpose(1, 2).reshape(seq_len, -1)
            attn_out[s:e] = o
            new_kv_layer.append((k_full, v_full))

        hidden_states = hidden_states + attn.o_proj(attn_out)
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        new_kv_per_layer.append(new_kv_layer)
        buffers_per_layer.append(buffer_layer)

    hidden_states = model.model.norm(hidden_states)
    # Normalize and retain probabilities in FP32 even when the model runs in
    # BF16/FP16; this is the representation used by rejection residual math.
    # Prefill needs only the final boundary distribution for each sequence.
    # Avoid materializing total_prompt_tokens x vocabulary FP32 logits.
    boundary_only = is_prefill and kv_manager.bounded_probabilities
    head_input = hidden_states[[e - 1 for _, e, _ in seq_ranges]] if boundary_only else hidden_states
    logits = model.lm_head(head_input)[:, :vocab_size].float()

    # ---- 3. write back per-proc prob history / KV cache --------------------
    new_x = []
    for i, pid in enumerate(proc_ids):
        s, e, cached_len = seq_ranges[i]
        seq_len = e - s

        logits_i = (logits[i:i + 1] if boundary_only else logits[s:e]).unsqueeze(0)  # (1, Lq, V)
        # norm_logits normalizes every row independently, including greedy.
        logits_i = norm_logits(logits_i[0], temperature, top_k, top_p).unsqueeze(0)

        row_start = cached_len + seq_len - 1 if boundary_only else cached_len
        kv_manager.store_probabilities(pid, logits_i, row_start, prefill=is_prefill)

        new_legacy = [layer[i] for layer in new_kv_per_layer]
        kv_manager._past_key_values[pid] = dynamic_cache_from_views(new_legacy)
        kv_manager._past_key_values[pid]._fastsd_buffers = [layer[i] for layer in buffers_per_layer]

        sampled = (
            sample(logits_i[0, -1:, :])
            if sample_next
            else torch.argmax(logits_i[0, -1:, :], dim=-1, keepdim=True)
        )
        next_tokens_1d = sampled.reshape(-1)
        new_x.append(torch.cat([flat_parts[i], next_tokens_1d], dim=0))

    return new_x
