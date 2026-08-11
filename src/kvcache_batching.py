import torch
from .util import norm_logits, sample
from transformers.cache_utils import DynamicCache

class KVCacheModel_batching():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        # KV cache and prob history indexed by proc_id
        self._past_key_values = {}
        self._prob_history = {}
        self.vocab_size = None

    def reset(self, proc_id: int):
        self._past_key_values.pop(proc_id, None)
        self._prob_history.pop(proc_id, None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        gamma: int,
        proc_ids: list[int],
        pad_token_id,
        is_prefill,
        input_lens=None,
    ) -> torch.Tensor:
        """
        input_ids: (B, T)
        proc_ids: list of process ids for each input
        """
        x = input_ids

        # for _ in range(gamma):
        #     q = self._forward_with_kvcache(x, proc_ids)
        #     next_tok = sample(q)
        #     x = torch.cat((x, next_tok), dim=1)

        next_tokens, valid_lens = self._forward_with_kvcache(
            x, proc_ids, pad_token_id, is_prefill, input_lens=input_lens
        )
        new_x = []
        for i in range(len(proc_ids)):
            if is_prefill:
                x_i = x[i, :valid_lens[i]]  # 去掉 padding
                new_x_i = torch.cat([x_i, next_tokens[i].squeeze(0)], dim=0)  # 拼接新 token
            else:
                x_i = x[i]
                new_x_i = torch.cat([x_i.squeeze(0), next_tokens[i].squeeze(0)], dim=0)  # 拼接新 token

            #     x_i = x[i].unsqueeze(0)  # (1 ,Tᵢ)
            #     x_i = x_i[:, :valid_lens[i]]  # 去掉 padding
            #     new_x_i = torch.cat([x_i, next_tokens[i].unsqueeze(0)], dim=1)  # (1 ,Tᵢ+1)
            #     new_x.append(new_x_i)

            new_x.append(new_x_i)

        return new_x

    @torch.no_grad()
    def rollback(self, proc_id: int, end_pos: int):
        # old_pkv = self._past_key_values[proc_id]
        # cropped_pkv = []
        # for key, value in old_pkv:
        #     key_cropped = key[:, :, :end_pos, :].clone()  # (1, H, end_pos, D)
        #     value_cropped = value[:, :, :end_pos, :].clone()
        #     cropped_pkv.append((key_cropped, value_cropped))
        # self._past_key_values[proc_id] = cropped_pkv
        self._past_key_values[proc_id].crop(end_pos)
        self._prob_history[proc_id] = self._prob_history[proc_id][:, :end_pos, :]

    @staticmethod
    def _legacy_cache(cache):
        if hasattr(cache, "to_legacy_cache"):
            return list(cache.to_legacy_cache())
        return list(cache)

    def _normalize_history(self, logits: torch.Tensor) -> torch.Tensor:
        history = logits.clone()
        for position in range(history.shape[-2]):
            history[:, position, :] = norm_logits(
                history[:, position, :],
                self._temperature,
                self._top_k,
                self._top_p,
            )
        return history

    @torch.no_grad()
    def prefill_chunks(
        self,
        input_ids: torch.Tensor,
        proc_ids: list[int],
        pad_token_id,
        input_lens: list[int],
        cursors: list[int],
    ) -> list[dict]:
        """Append Prefill chunks without sampling, acceptance, or rollback."""

        batch_size = len(proc_ids)
        if input_ids.dim() != 2 or input_ids.shape[0] != batch_size:
            raise ValueError("input_ids must have shape [len(proc_ids), T]")
        if len(input_lens) != batch_size or len(cursors) != batch_size:
            raise ValueError("input_lens and cursors must match proc_ids")
        if any(int(length) <= 0 for length in input_lens):
            raise ValueError("every Prefill chunk must contain at least one token")

        metadata = [None] * batch_size
        fresh_indices = [index for index, cursor in enumerate(cursors) if int(cursor) == 0]
        continuation_indices = [index for index, cursor in enumerate(cursors) if int(cursor) > 0]

        if fresh_indices:
            index_tensor = torch.tensor(fresh_indices, device=input_ids.device, dtype=torch.long)
            fresh_inputs = input_ids.index_select(0, index_tensor)
            fresh_mask = torch.zeros_like(fresh_inputs, dtype=torch.long)
            for row, original_index in enumerate(fresh_indices):
                fresh_mask[row, :int(input_lens[original_index])] = 1
            outputs = self._model(
                fresh_inputs,
                attention_mask=fresh_mask,
                use_cache=True,
            )
            logits = outputs.logits[:, :, :self.vocab_size]
            legacy_cache = self._legacy_cache(outputs.past_key_values)
            for row, original_index in enumerate(fresh_indices):
                pid = proc_ids[original_index]
                valid_len = int(input_lens[original_index])
                history = self._normalize_history(logits[row:row + 1, :valid_len, :])
                per_session_cache = []
                for key, value in legacy_cache:
                    per_session_cache.append(
                        (
                            key[row:row + 1, :, :valid_len, :].clone(),
                            value[row:row + 1, :, :valid_len, :].clone(),
                        )
                    )
                self._prob_history[pid] = history
                self._past_key_values[pid] = DynamicCache.from_legacy_cache(per_session_cache)
                metadata[original_index] = {
                    "proc_id": pid,
                    "cursor_start": 0,
                    "cursor_end": valid_len,
                    "cache_len": valid_len,
                }

        if continuation_indices:
            continuation_caches = []
            cached_lens = []
            for original_index in continuation_indices:
                pid = proc_ids[original_index]
                if pid not in self._past_key_values or pid not in self._prob_history:
                    raise ValueError(f"missing Prefill continuation state for proc_id={pid!r}")
                cached_len = int(self._past_key_values[pid].get_seq_length())
                cursor = int(cursors[original_index])
                if cached_len != cursor:
                    raise ValueError(
                        f"Prefill cache length {cached_len} does not match cursor {cursor} "
                        f"for proc_id={pid!r}"
                    )
                cached_lens.append(cached_len)
                continuation_caches.append(self._legacy_cache(self._past_key_values[pid]))

            max_cached_len = max(cached_lens)
            index_tensor = torch.tensor(continuation_indices, device=input_ids.device, dtype=torch.long)
            continuation_inputs = input_ids.index_select(0, index_tensor)
            current_mask = torch.zeros_like(continuation_inputs, dtype=torch.long)
            past_mask = torch.zeros(
                (len(continuation_indices), max_cached_len),
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = torch.zeros_like(continuation_inputs, dtype=torch.long)
            for row, (original_index, cached_len) in enumerate(
                zip(continuation_indices, cached_lens)
            ):
                valid_len = int(input_lens[original_index])
                current_mask[row, :valid_len] = 1
                past_mask[row, :cached_len] = 1
                position_ids[row, :valid_len] = torch.arange(
                    cached_len,
                    cached_len + valid_len,
                    device=input_ids.device,
                    dtype=torch.long,
                )

            padded_by_layer = []
            for layers in zip(*continuation_caches):
                padded_keys = []
                padded_values = []
                for key, value in layers:
                    pad_len = max_cached_len - key.shape[-2]
                    if pad_len:
                        pad_shape = list(key.shape)
                        pad_shape[-2] = pad_len
                        key = torch.cat((key, key.new_zeros(pad_shape)), dim=-2)
                        value = torch.cat((value, value.new_zeros(pad_shape)), dim=-2)
                    padded_keys.append(key)
                    padded_values.append(value)
                padded_by_layer.append(
                    (torch.cat(padded_keys, dim=0), torch.cat(padded_values, dim=0))
                )

            outputs = self._model(
                continuation_inputs,
                attention_mask=torch.cat((past_mask, current_mask), dim=1),
                position_ids=position_ids,
                past_key_values=DynamicCache.from_legacy_cache(padded_by_layer),
                use_cache=True,
            )
            logits = outputs.logits[:, :, :self.vocab_size]
            output_cache = self._legacy_cache(outputs.past_key_values)

            for row, (original_index, cached_len) in enumerate(
                zip(continuation_indices, cached_lens)
            ):
                pid = proc_ids[original_index]
                valid_len = int(input_lens[original_index])
                new_history = self._normalize_history(logits[row:row + 1, :valid_len, :])
                self._prob_history[pid] = torch.cat(
                    (self._prob_history[pid], new_history), dim=1
                )
                per_session_cache = []
                for key, value in output_cache:
                    old_key = key[row:row + 1, :, :cached_len, :]
                    old_value = value[row:row + 1, :, :cached_len, :]
                    new_key = key[
                        row:row + 1,
                        :,
                        max_cached_len:max_cached_len + valid_len,
                        :,
                    ]
                    new_value = value[
                        row:row + 1,
                        :,
                        max_cached_len:max_cached_len + valid_len,
                        :,
                    ]
                    per_session_cache.append(
                        (
                            torch.cat((old_key, new_key), dim=-2).clone(),
                            torch.cat((old_value, new_value), dim=-2).clone(),
                        )
                    )
                new_cache_len = cached_len + valid_len
                self._past_key_values[pid] = DynamicCache.from_legacy_cache(per_session_cache)
                metadata[original_index] = {
                    "proc_id": pid,
                    "cursor_start": cached_len,
                    "cursor_end": new_cache_len,
                    "cache_len": new_cache_len,
                }

        return metadata

    def _forward_with_kvcache(
        self,
        input_ids: torch.Tensor,
        proc_ids: list[int],
        pad_token_id,
        is_prefill: bool = False,
        input_lens=None,
    ) -> torch.Tensor:
        """
        Batch forward with per-proc_id KV cache
        input_ids: (B, T), proc_ids: list of length B
        Returns: logits for next token (B, vocab_size)
        """
        outputs = []
        logits_out = []
        last_q = []
        next_tokens = []
        valid_lens = []

        # Handle new (prefill) samples
        if is_prefill:
            input_batch = input_ids
            if input_lens is None:
                attention_mask = (input_batch != pad_token_id).long()
            else:
                attention_mask = torch.zeros_like(input_batch, dtype=torch.long, device=input_batch.device)
                for i, valid_len in enumerate(input_lens):
                    attention_mask[i, :valid_len] = 1
            out = self._model(input_batch, attention_mask=attention_mask)
            logits = out.logits[:, :, :self.vocab_size]  # shape: (B, T, V)
            past_key_values = out.past_key_values  # List of (key, value), each: (B, H, T, D)

            logits_list = []
            past_key_values_list = []

            batch_size = input_batch.size(0)
            for i in range(batch_size):
                # ---- 1. 截取每个样本有效 token 长度 ----
                valid_len = input_lens[i] if input_lens is not None else (input_batch[i] != pad_token_id).sum().item()
                valid_lens.append(valid_len)

                # ---- 2. 提取该样本对应的 logits ----
                logits_i = logits[i, :valid_len, :].unsqueeze(0)  # shape: (valid_len, vocab_size)
                logits_list.append(logits_i)

                # ---- 3. 提取该样本对应的 past_key_values ----
                pkv_i = []
                for layer in past_key_values:
                    key, value = layer  # shape: (B, H, T, D)
                    key_i = key[i:i + 1, :, :valid_len, :].clone()  # (1, H, valid_len, D)
                    value_i = value[i:i + 1, :, :valid_len, :].clone()  # (1, H, valid_len, D)
                    pkv_i.append((key_i, value_i))
                past_key_values_list.append(pkv_i)

            for idx, pid in enumerate(proc_ids):
                self._prob_history[pid] = logits_list[idx]
                for i in range(self._prob_history[pid].shape[-2]):
                    self._prob_history[pid][:, i, :] = norm_logits(self._prob_history[pid][:, i, :], self._temperature,
                                                              self._top_k,
                                                              self._top_p)
                self._past_key_values[pid] = DynamicCache.from_legacy_cache(past_key_values_list[idx])
                last_q_idx = self._prob_history[pid][:, -1, :]
                sampled_token = sample(last_q_idx)  # (1,)
                next_tokens.append(sampled_token)
        ###############################################################################################################
        else:
            input_tensors = []
            padded_past_list = []
            max_cached_len = 0
            cached_lens = []

            # Step 1: 获取每个样本 cached_len 和 residual
            residuals = []
            residual_lens = []
            for i, pid in enumerate(proc_ids):
                input_ids_i = input_ids[i]  # shape: (T,)
                past_kv = self._past_key_values[pid]

                cached_len = past_kv.get_seq_length()
                cached_lens.append(cached_len)
                max_cached_len = max(max_cached_len, cached_len)

                last_input_ids = input_ids_i[:, cached_len:]  # only feed new tokens
                if last_input_ids.numel() == 0:
                    last_input_ids = input_ids_i[:, -1:]
                residuals.append(last_input_ids.squeeze(0))
                residual_lens.append(last_input_ids.shape[1])

            # Step 2: padding input_ids 到 max_len
            padded_inputs = torch.nn.utils.rnn.pad_sequence(
                residuals, batch_first=True, padding_value=pad_token_id
            )  # shape: [B, T_pad]

            # Step 3: padding past_kv 到 max_cached_len
            for pid in proc_ids:
                legacy = self._past_key_values[pid].to_legacy_cache()  # List[(key, value)]
                padded_layers = []
                for key, value in legacy:
                    # key shape 可能是 4-D or 5-D；seq_len 在倒数第二维
                    seq_dim = -2
                    cached_T = key.shape[seq_dim]
                    pad_len = max_cached_len - cached_T
                    if pad_len > 0:
                        # ── 构造与 key / value **完全同形，除了 seq_len 改为 pad_len** 的零张量 ──
                        pad_shape = list(key.shape)
                        pad_shape[seq_dim] = pad_len
                        pad_k = key.new_zeros(pad_shape)  # dtype / device 对齐
                        pad_v = value.new_zeros(pad_shape)

                        key = torch.cat([key, pad_k], dim=seq_dim)
                        value = torch.cat([value, pad_v], dim=seq_dim)
                    padded_layers.append((key, value))
                padded_past_list.append(padded_layers)

            # Step 4: 前向计算
            # transpose batch-level结构 [B][L](key, value) → [L][B](key, value)
            layer_wise = list(zip(*padded_past_list))  # list of length L

            batched_kv = []
            for layer in layer_wise:  # each is a list of B tensors
                key_list = [kv[0] for kv in layer]
                value_list = [kv[1] for kv in layer]
                key_batched = torch.cat(key_list, dim=0)  # (B, H, T, D)
                value_batched = torch.cat(value_list, dim=0)
                batched_kv.append((key_batched, value_batched))

            batched_cache = DynamicCache.from_legacy_cache(batched_kv)
            # print("batched_cache.key_cache[0].shape", batched_cache.key_cache[0].shape)
            # print("padded_inputs", padded_inputs.shape)

            # Build full attention mask for [past_kv, padded current inputs].
            # Without this, padded tokens in verify stage may be treated as valid tokens.
            current_mask = torch.zeros_like(padded_inputs, dtype=torch.long, device=padded_inputs.device)
            for i, res_len in enumerate(residual_lens):
                current_mask[i, :res_len] = 1
            past_mask = torch.zeros(
                (len(proc_ids), max_cached_len),
                dtype=current_mask.dtype,
                device=current_mask.device,
            )
            for i, cached_len in enumerate(cached_lens):
                past_mask[i, :cached_len] = 1
            attention_mask = torch.cat([past_mask, current_mask], dim=1)
            position_ids = torch.zeros_like(padded_inputs, dtype=torch.long)
            for i, (cached_len, residual_len) in enumerate(
                zip(cached_lens, residual_lens)
            ):
                position_ids[i, :residual_len] = torch.arange(
                    cached_len,
                    cached_len + residual_len,
                    device=padded_inputs.device,
                    dtype=torch.long,
                )

            outputs = self._model(
                padded_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=batched_cache,
                use_cache=True,
            )
            not_cached_q = outputs.logits[:, :, :self.vocab_size]  # (B, T, V)
            past_key_values = outputs.past_key_values  # List of (key, value), each: (B, H, T, D)

            if not_cached_q.dim() == 2:
                not_cached_q = not_cached_q.unsqueeze(0)

            # IMPORTANT:
            # Do not normalize flattened 2D logits in one shot when temperature=0.
            # util.norm_logits' temp==0 branch is row-unsafe for batched inputs,
            # which can mix argmax columns across rows and corrupt verify logits.
            # Keep per-position normalization to match non-batching semantics.
            for b in range(not_cached_q.shape[0]):
                for t in range(not_cached_q.shape[1]):
                    not_cached_q[b:b + 1, t, :] = norm_logits(
                        not_cached_q[b:b + 1, t, :],
                        self._temperature,
                        self._top_k,
                        self._top_p,
                    )

            # Step 5: 更新每个样本的 prob_history 和 KVCache
            logits_list = []
            past_key_values_list = []
            for i, pid in enumerate(proc_ids):
                # ---- 1. 截取每个样本有效 token 长度 ----
                valid_len = residual_lens[i]
                valid_lens.append(valid_len)

                # ---- 2. 提取该样本对应的 logits ----
                logits_i = not_cached_q[i, :valid_len, :].unsqueeze(0)  # shape: (valid_len, vocab_size)
                logits_list.append(logits_i)

                # ---- 3. 提取该样本对应的 past_key_values ----
                pkv_i = []
                for layer in past_key_values:  # each layer is a (key, value)
                    key, value = layer  # shape: (B, H, T, D)
                    old_key = key[i:i + 1, :, :cached_lens[i], :]
                    old_value = value[i:i + 1, :, :cached_lens[i], :]
                    new_key = key[
                        i:i + 1,
                        :,
                        max_cached_len:max_cached_len + valid_len,
                        :,
                    ]
                    new_value = value[
                        i:i + 1,
                        :,
                        max_cached_len:max_cached_len + valid_len,
                        :,
                    ]
                    pkv_i.append(
                        (
                            torch.cat((old_key, new_key), dim=-2).clone(),
                            torch.cat((old_value, new_value), dim=-2).clone(),
                        )
                    )
                past_key_values_list.append(pkv_i)

            for i, pid in enumerate(proc_ids):
                self._prob_history[pid] = torch.cat([self._prob_history[pid], logits_list[i]], dim=1)
                self._past_key_values[pid] = DynamicCache.from_legacy_cache(past_key_values_list[i])
                last_q_i = not_cached_q[i, valid_lens[i] - 1, :].unsqueeze(0)
                sampled_token = sample(last_q_i)
                next_tokens.append(sampled_token)
            ###############################################################################################################

            # for i, pid in enumerate(proc_ids):
            #     input_ids_i = input_ids[i]  # shape: (T,)
            #     past_kv = self._past_key_values[pid]  # List[(1, H, L, D)]
            #
            #     # cached_len = past_kv[0][0].shape[2]  # get valid length from key: shape (1, H, L, D)
            #     cached_len = past_kv.get_seq_length()
            #     last_input_ids = input_ids_i[:, cached_len:]  # only feed new tokens
            #
            #     outputs = self._model(last_input_ids, past_key_values=past_kv, use_cache=True)
            #
            #     not_cached_q = outputs.logits[:, :, :self.vocab_size]  # shape: (1, Tᵢ, V)
            #
            #     if not_cached_q.dim() == 2:
            #         not_cached_q = torch.unsqueeze(not_cached_q, 0)
            #
            #     for t in range(not_cached_q.shape[-2]):
            #         not_cached_q[:, t, :] = norm_logits(not_cached_q[:, t, :], self._temperature, self._top_k, self._top_p)
            #
            #     # 拼接到历史上
            #     self._prob_history[pid] = torch.cat([self._prob_history[pid], not_cached_q], dim=1)
            #
            #     # 更新 past_key_values
            #     self._past_key_values[pid] = DynamicCache.from_legacy_cache(outputs.past_key_values)
            #
            #     last_q_i = not_cached_q[:, -1, :]  # shape: (1, V)
            #     sampled_token = sample(last_q_i)  # (1,)
            #     next_tokens.append(sampled_token)
        ###############################################################################################################

        return next_tokens, valid_lens
