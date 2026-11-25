"""Explicit SpecEdge protobuf tensor wire adapter.

The official wire format is raw tensor bytes.  NumPy cannot represent bf16 on
all supported versions, so bf16 is transported as its exact uint16 payload.
The client keeps its CPU attention mask in fp32 but casts that field to the
target's bf16 wire dtype before serialization.  No ``sitecustomize`` or
``util.encode`` monkeypatch is required.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import grpc.aio
import torch

from specedge_grpc import specedge_pb2, specedge_pb2_grpc


class SpecEdgeWireCodec:
    """Encode/decode tensors using the official raw-byte protobuf contract."""

    name = "explicit_bf16_uint16_wire_v1"

    @staticmethod
    def encode(tensor: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> bytes:
        value = tensor if dtype is None else tensor.to(dtype=dtype)
        value = value.detach().contiguous().cpu()
        if value.dtype == torch.bfloat16:
            return value.view(torch.uint16).numpy().tobytes()
        return value.numpy().tobytes()

    @staticmethod
    def decode(
        payload: bytes,
        *,
        device: Any,
        dtype: torch.dtype,
        shape: tuple[int, ...] | int,
    ) -> torch.Tensor:
        value = torch.frombuffer(payload, dtype=dtype)
        return value.reshape(shape).to(device=device)


class ExplicitSpecEdgeGrpcClient:
    """Client-side Validate adapter that owns all wire casting decisions."""

    def __init__(
        self,
        host: str,
        device: torch.device,
        *,
        wire_mask_dtype: torch.dtype = torch.bfloat16,
        codec: type[SpecEdgeWireCodec] = SpecEdgeWireCodec,
        channel=None,
        verifier_budget: int | None = None,
    ) -> None:
        self.client_idx = 0
        self._host = host
        self._device = device
        self._wire_mask_dtype = wire_mask_dtype
        self._codec = codec
        self._channel = channel if channel is not None else grpc.aio.insecure_channel(self._host)
        self._verifier_budget = verifier_budget
        self._stub = specedge_pb2_grpc.SpecEdgeServiceStub(self._channel)
        self._request_bytes = 0
        self._response_bytes = 0
        self._rpc_count = 0
        self.draft_probabilities = None
        self.io_stall = {}

    def snapshot_stats(self) -> dict[str, int]:
        """Return measured application-layer protobuf bytes and RPC count."""

        return {
            "request_bytes": int(self._request_bytes),
            "response_bytes": int(self._response_bytes),
            "rpc_count": int(self._rpc_count),
        }

    async def close(self):
        await self._channel.close()

    async def request(
        self,
        client_idx: int,
        req_idx: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        parent_indices: torch.Tensor,
        prefill: bool = False,
        prefix: Optional[str] = None,
    ):
        if prefill and prefix is None:
            raise ValueError("Prefix must be provided for prefill requests.")
        actual_width = input_ids.size(-1)
        padding = 0 if self._verifier_budget is None else self._verifier_budget - parent_indices.numel()
        if padding < 0 or (self._verifier_budget is not None and actual_width != parent_indices.numel() + 1):
            raise ValueError("SpecEdge verifier requires a rooted tree within its configured budget")
        if padding:
            next_slot = int(cache_seq_indices[-1].item()) + 1
            if next_slot + padding > attention_mask.size(-1):
                raise ValueError("insufficient KV slots for SpecEdge verifier padding")
            input_ids = torch.cat((input_ids, input_ids[..., :1].expand(-1, padding)), dim=-1)
            position_ids = torch.cat((position_ids, cache_seq_indices[:1].expand(1, padding)), dim=-1)
            parent_indices = torch.cat((parent_indices, cache_seq_indices[:1].expand(padding)))
            cache_seq_indices = torch.cat((cache_seq_indices, torch.arange(next_slot, next_slot + padding, device=self._device)))
            attention_mask = torch.cat((attention_mask, attention_mask[..., :1, :].expand(-1, -1, padding, -1)), dim=-2)
        parent_payload = self._codec.encode(parent_indices)
        if self.draft_probabilities is not None:
            if padding or actual_width != self.draft_probabilities.size(0) + 1:
                raise ValueError("rejection requires one probability row per chain proposal")
            parent_payload += self._codec.encode(self.draft_probabilities, dtype=torch.float32)
        request = specedge_pb2.ValidateRequest(
            client_idx=client_idx,
            req_idx=req_idx,
            input_ids=self._codec.encode(input_ids),
            position_ids=self._codec.encode(position_ids),
            cache_seq_indices=self._codec.encode(cache_seq_indices),
            parent_indices=parent_payload,
            attention_mask=self._codec.encode(
                attention_mask, dtype=self._wire_mask_dtype
            ),
            prefill=prefill,
            prefix=prefix,
        )
        call = self._stub.Validate(request)
        response = await call
        if hasattr(call, "trailing_metadata"):
            raw_stats = dict(await call.trailing_metadata()).get("specedge-io-bin")
            if raw_stats:
                for key, value in json.loads(raw_stats).items():
                    self.io_stall[key] = value if isinstance(value, bool) else self.io_stall.get(key, 0) + value
        self._request_bytes += request.ByteSize()
        self._response_bytes += response.ByteSize()
        self._rpc_count += 1
        return self._codec.decode(
            response.selection,
            device=self._device,
            dtype=torch.long,
            shape=input_ids.size(-1),
        )[:actual_width], response.prefill
