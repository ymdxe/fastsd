"""Explicit SpecEdge protobuf tensor wire adapter.

The official wire format is raw tensor bytes.  NumPy cannot represent bf16 on
all supported versions, so bf16 is transported as its exact uint16 payload.
The client keeps its CPU attention mask in fp32 but casts that field to the
target's bf16 wire dtype before serialization.  No ``sitecustomize`` or
``util.encode`` monkeypatch is required.
"""

from __future__ import annotations

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
    ) -> None:
        self.client_idx = 0
        self._host = host
        self._device = device
        self._wire_mask_dtype = wire_mask_dtype
        self._codec = codec
        self._channel = grpc.aio.insecure_channel(self._host)
        self._stub = specedge_pb2_grpc.SpecEdgeServiceStub(self._channel)
        self._request_bytes = 0
        self._response_bytes = 0
        self._rpc_count = 0

    def snapshot_stats(self) -> dict[str, int]:
        """Return measured application-layer protobuf bytes and RPC count."""

        return {
            "request_bytes": int(self._request_bytes),
            "response_bytes": int(self._response_bytes),
            "rpc_count": int(self._rpc_count),
        }

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
        request = specedge_pb2.ValidateRequest(
            client_idx=client_idx,
            req_idx=req_idx,
            input_ids=self._codec.encode(input_ids),
            position_ids=self._codec.encode(position_ids),
            cache_seq_indices=self._codec.encode(cache_seq_indices),
            parent_indices=self._codec.encode(parent_indices),
            attention_mask=self._codec.encode(
                attention_mask, dtype=self._wire_mask_dtype
            ),
            prefill=prefill,
            prefix=prefix,
        )
        response = await self._stub.Validate(request)
        self._request_bytes += request.ByteSize()
        self._response_bytes += response.ByteSize()
        self._rpc_count += 1
        return self._codec.decode(
            response.selection,
            device=self._device,
            dtype=torch.long,
            shape=input_ids.size(-1),
        ), response.prefill
