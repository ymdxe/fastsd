"""Single-chain variable-width transport over the pinned fixed-width CUDA graph.

The model, graph engine and batching loop are upstream. Each real row contains
gamma + 1 inputs; extra graph lanes write only into distinct scratch KV slots.
Only the contiguous accepted real prefix survives into the client's next KV.
"""

import json
import os

import torch
import util
from config import SpecEdgeBatchServerConfig as config
from strategy.server_verify.specexec.grpc import InferenceController
from single_chain_contract import (
    SINGLE_CHAIN_ID, accepted_chain_length, scratch_slots, validate_chain_layout,
)
from wire_codec import SpecEdgeWireCodec


class SingleChainInferenceController(InferenceController):
    single_chain_id = SINGLE_CHAIN_ID

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._max_budget != 8 or self._max_n_beams != 9 or config.cache_prefill:
            raise ValueError("Single-chain target requires budget 8 and runtime prefill")
        self._client_gammas = json.loads(os.environ["FASTSD_SPECEDGE_CLIENT_GAMMAS"])
        if len(self._client_gammas) != self._num_clients:
            raise ValueError("Missing per-client gamma mapping")
        self._cache_lengths = [None] * self._num_clients
        self._active_requests = [None] * self._num_clients
        self._batch_number = 0

    @torch.inference_mode()
    def _inference(self, batch):
        if not 0 < len(batch) <= self._batch_size:
            raise ValueError("Invalid single-chain batch size")
        clients = [int(req.client_idx) for req in batch]
        if len(set(clients)) != len(clients):
            raise ValueError("A batch cannot contain duplicate client identities")
        self._engine._past_key_values.clear()
        self._client_indices.fill_(-1)
        self._input_ids.zero_()
        self._position_ids.zero_()
        self._parent_indices.zero_()
        self._attention_mask.zero_()
        # Even inactive lanes use distinct write slots and a nonempty mask.
        inactive_slots = torch.arange(self._max_len - self._max_n_beams,
                                      self._max_len, device=self._device)
        self._cache_seq_indices.copy_(inactive_slots.expand_as(self._cache_seq_indices))
        self._attention_mask[..., 0] = 1
        prefill_indices = []
        layouts = []
        for batch_idx, req in enumerate(batch):
            client_idx = int(req.client_idx)
            if not 0 <= client_idx < self._num_clients:
                raise ValueError("Client outside the single-chain gamma mapping")
            gamma = int(self._client_gammas[client_idx])
            width = gamma + 1
            decode = lambda payload: SpecEdgeWireCodec.decode(
                payload, device="cpu", dtype=torch.long, shape=(-1,)).tolist()
            inputs = decode(req.input_ids)
            positions = decode(req.position_ids)
            slots = decode(req.cache_seq_indices)
            parents = decode(req.parent_indices)
            root = validate_chain_layout(inputs, positions, slots, parents,
                                         gamma=gamma, max_len=self._max_len)
            padding = scratch_slots(root=root, logical_width=width,
                                    physical_width=self._max_n_beams, max_len=self._max_len)
            amask = SpecEdgeWireCodec.decode(req.attention_mask, device="cpu",
                                            dtype=self._dtype, shape=(1, width, self._max_len))
            expected = (torch.arange(self._max_len)[None, :] <=
                        torch.tensor(slots)[:, None]).unsqueeze(0)
            if not torch.equal(amask, expected.to(dtype=self._dtype)):
                raise ValueError("Wire attention mask is not the exact contiguous causal chain")
            self._client_indices[batch_idx] = client_idx
            self._input_ids[batch_idx, :width].copy_(torch.tensor(inputs, device=self._device))
            self._position_ids[batch_idx, :width].copy_(torch.tensor(positions, device=self._device))
            self._parent_indices[batch_idx, :gamma].copy_(torch.tensor(parents, device=self._device))
            self._cache_seq_indices[batch_idx, :width].copy_(torch.tensor(slots, device=self._device))
            self._attention_mask[batch_idx, :, :width].copy_(amask.to(self._device))
            if padding:
                self._cache_seq_indices[batch_idx, width:].copy_(torch.tensor(padding, device=self._device))
                self._position_ids[batch_idx, width:] = root
                self._input_ids[batch_idx, width:] = inputs[0]
                self._attention_mask[batch_idx, :, width:, :].zero_()
                self._attention_mask[batch_idx, :, width:, :root + 1] = 1
            if req.prefill:
                if not req.prefix:
                    raise ValueError("Canonical rendered prefix is required for runtime prefill")
                prompt_ids = self._tokenizer.encode(req.prefix, return_tensors="pt")
                if prompt_ids.size(-1) != root + 1 or int(prompt_ids[0, -1]) != inputs[0]:
                    raise ValueError("Target tokenizer differs from canonical Draft prefix")
                self._iter_idx[client_idx] = 0
                self._active_requests[client_idx] = int(req.req_idx)
                prefill_indices.append((batch_idx, int(req.req_idx)))
                if root:
                    self._engine.prefill(
                        input_ids=prompt_ids[..., :-1].to(self._device),
                        position_ids=self._predefined_position_ids[:, :root],
                        batch_idx=batch_idx,
                        cache_seq_indices=self._predefined_position_ids[:, :root],
                        attention_mask=self._predefined_attention_mask[:, :, :root, :self._max_len],
                    )
            else:
                if (self._active_requests[client_idx] != int(req.req_idx)
                        or self._cache_lengths[client_idx] != root):
                    raise ValueError("Request/prefix does not match the retained client KV")
                self._iter_idx[client_idx] += 1
                self._engine._past_key_values.k_cache[:, batch_idx].copy_(self.k_cache[:, client_idx])
                self._engine._past_key_values.v_cache[:, batch_idx].copy_(self.v_cache[:, client_idx])
            layouts.append((client_idx, gamma, root, inputs, padding, int(req.req_idx)))

        with util.Timing(device=self._device, mode="event") as forward_t:
            logits = self._engine.forward(
                input_ids=self._input_ids, position_ids=self._position_ids,
                cache_batch_indices=self._cache_batch_indices.flatten(),
                cache_seq_indices=self._cache_seq_indices.flatten(),
                attention_mask=self._attention_mask,
            )
        selection = util.sampler_from_logits(logits, temperature=self._temperature)
        audit = []
        for batch_idx, (client_idx, gamma, root, inputs, padding, req_idx) in enumerate(layouts):
            real_selection = selection[batch_idx, :gamma + 1]
            accepted = accepted_chain_length(inputs[1:], real_selection.tolist())
            keep = root + 1 + accepted
            working = self._engine._past_key_values
            # All real KV already occupies the correct contiguous positions.
            # Drop rejected candidates and every scratch lane before persisting.
            working.k_cache[:, batch_idx, :, keep:, :].zero_()
            working.v_cache[:, batch_idx, :, keep:, :].zero_()
            self.k_cache[:, client_idx].copy_(working.k_cache[:, batch_idx])
            self.v_cache[:, client_idx].copy_(working.v_cache[:, batch_idx])
            self._cache_lengths[client_idx] = keep
            self._resp_queue.put(((SpecEdgeWireCodec.encode(real_selection), len(prefill_indices)), client_idx))
            audit.append({"client_idx": client_idx, "req_idx": req_idx,
                          "gamma": gamma, "actual_candidates": gamma,
                          "wire_inputs": gamma + 1, "parent_chain_contiguous": True,
                          "graph_width": self._max_n_beams, "padding_slots": padding,
                          "accepted_candidates": accepted, "retained_kv_length": keep})
        self._logger.info("Single-chain batch %s", json.dumps({
            "single_chain": SINGLE_CHAIN_ID, "batch_number": self._batch_number, "rows": audit,
        }, sort_keys=True))
        self._batch_number += 1
        return forward_t.elapsed, prefill_indices
