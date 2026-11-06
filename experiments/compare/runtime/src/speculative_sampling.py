"""Shared speculative-sampling primitives.

The draft side communicates a sparse distribution rather than only the sampled
token.  The sparse support is the *actual* distribution used to draw the draft
token, so the rejection correction remains exact for the target distribution.
"""

from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import Iterable

import torch


_MAGIC = b"FSDQ"
_HEADER = struct.Struct("<4sII")  # magic, number of rows, support width
# Only allow floating-point roundoff from the FP32 wire representation; a
# visibly incomplete probability row must be rejected at the protocol edge.
_NORMALIZATION_TOL = 1e-4


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a deterministic generator owned by one decode session."""

    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        # Some accelerator wrappers expose a non-standard device object.  The
        # caller can still use this generator for CPU tensors and receive a
        # clear device error if it attempts to mix it with CUDA tensors.
        generator = torch.Generator()
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    return generator


def draft_distribution(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Return the dense distribution used by the sparse draft sampler.

    Rejection mode deliberately has one truncation rule: temperature followed
    by top-k and renormalization.  ``top_k`` must be positive so that the
    serialized support is bounded and complete.
    """

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError("draft logits must have shape [batch, vocab]")
    if int(top_k) <= 0:
        raise ValueError("draft_top_k must be positive in rejection mode")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if temperature == 0.0:
        result = torch.zeros_like(logits, dtype=torch.float32)
        indices = torch.argmax(logits, dim=-1, keepdim=True)
        return result.scatter_(1, indices, 1.0)

    scaled = logits.float() / temperature
    k = min(int(top_k), int(scaled.shape[-1]))
    values, indices = torch.topk(scaled, k=k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    result = torch.zeros_like(scaled, dtype=torch.float32)
    result.scatter_(1, indices, probs)
    return result


def sparse_block_from_dense(prob_rows: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complete sparse rows into ``(ids, probabilities)`` tensors."""

    if prob_rows.dim() == 3 and prob_rows.shape[0] == 1:
        prob_rows = prob_rows[0]
    if prob_rows.dim() != 2:
        raise ValueError("probability rows must have shape [rows, vocab]")
    rows, vocab = map(int, prob_rows.shape)
    if rows == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=prob_rows.device),
            torch.empty((0, 0), dtype=torch.float32, device=prob_rows.device),
        )
    k = min(int(top_k), vocab)
    if k <= 0:
        raise ValueError("top_k must be positive")
    values, indices = torch.topk(prob_rows.float(), k=k, dim=-1)
    row_sums = values.sum(dim=-1)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("sparse draft probabilities must be finite and non-negative")
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=_NORMALIZATION_TOL, rtol=0.0):
        raise ValueError(
            "sparse support does not represent the complete draft distribution; "
            "sample the draft from the same top-k support"
        )
    return indices.long(), (values / row_sums.unsqueeze(-1)).float()


def _validate_rows(ids: torch.Tensor, probs: torch.Tensor, vocab_size: int | None = None) -> None:
    if ids.dim() != 2 or probs.dim() != 2 or ids.shape != probs.shape:
        raise ValueError("sparse probability block must contain matching [rows, k] arrays")
    if ids.shape[1] <= 0:
        raise ValueError("sparse probability block support must be non-empty")
    if not torch.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("sparse probabilities must be finite and non-negative")
    if (ids < 0).any() or (vocab_size is not None and (ids >= int(vocab_size)).any()):
        raise ValueError("sparse token id is outside the model vocabulary")
    for row_ids in ids.detach().cpu().tolist():
        if len(set(int(value) for value in row_ids)) != len(row_ids):
            raise ValueError("sparse probability block contains duplicate token ids")
    sums = probs.float().sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=_NORMALIZATION_TOL, rtol=0.0):
        raise ValueError("each sparse probability row must be normalized")


def encode_sparse_block(ids: torch.Tensor, probs: torch.Tensor) -> str:
    """Encode sparse rows as little-endian int32/float32 Base64."""

    ids = ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
    probs = probs.detach().to(device="cpu", dtype=torch.float32).contiguous()
    _validate_rows(ids, probs)
    rows, width = map(int, ids.shape)
    id_bytes = struct.pack("<" + "i" * ids.numel(), *ids.reshape(-1).tolist())
    prob_bytes = struct.pack("<" + "f" * probs.numel(), *probs.reshape(-1).tolist())
    payload = _HEADER.pack(_MAGIC, rows, width) + id_bytes + prob_bytes
    return base64.b64encode(payload).decode("ascii")


def decode_sparse_block(encoded: str, *, device: torch.device | str = "cpu", vocab_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode and validate a sparse probability block."""

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("draft_prob_block is required and must be a Base64 string")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("draft_prob_block is not valid Base64") from exc
    if len(raw) < _HEADER.size:
        raise ValueError("draft_prob_block is truncated")
    magic, rows, width = _HEADER.unpack_from(raw, 0)
    if magic != _MAGIC or rows <= 0 or width <= 0:
        raise ValueError("draft_prob_block has an invalid header")
    count = int(rows) * int(width)
    expected = _HEADER.size + count * 4 + count * 4
    if len(raw) != expected:
        raise ValueError("draft_prob_block has an invalid payload length")
    offset = _HEADER.size
    ids = torch.tensor(struct.unpack_from("<" + "i" * count, raw, offset), dtype=torch.long)
    offset += count * 4
    probs = torch.tensor(struct.unpack_from("<" + "f" * count, raw, offset), dtype=torch.float32)
    ids = ids.reshape(rows, width).to(device=device)
    probs = probs.reshape(rows, width).to(device=device)
    _validate_rows(ids, probs, vocab_size=vocab_size)
    return ids, probs


def slice_sparse_block(encoded: str, start: int, count: int) -> str:
    """Slice rows while preserving the wire representation."""

    ids, probs = decode_sparse_block(encoded)
    start = int(start)
    count = int(count)
    if start < 0 or count <= 0 or start + count > ids.shape[0]:
        raise ValueError("sparse probability slice is outside the encoded rows")
    return encode_sparse_block(ids[start:start + count], probs[start:start + count])


def _candidate_probability(ids: torch.Tensor, probs: torch.Tensor, row: int, token: int) -> torch.Tensor:
    matches = torch.nonzero(ids[row] == int(token), as_tuple=False).flatten()
    if matches.numel() == 0:
        return probs.new_zeros(())
    return probs[row, int(matches[0].item())]


def validate_candidate_support(
    draft_tokens: torch.Tensor | Iterable[int],
    sparse_ids: torch.Tensor,
    sparse_probs: torch.Tensor,
) -> None:
    """Require every transmitted draft token to have positive sparse mass."""

    if isinstance(draft_tokens, torch.Tensor):
        tokens = draft_tokens.reshape(-1).to(device=sparse_ids.device, dtype=torch.long)
    else:
        tokens = torch.tensor(list(draft_tokens), device=sparse_ids.device, dtype=torch.long)
    sparse_ids = sparse_ids.to(device=tokens.device, dtype=torch.long)
    sparse_probs = sparse_probs.to(device=tokens.device, dtype=torch.float32)
    if sparse_ids.shape[0] != tokens.numel():
        raise ValueError("sparse probability rows must match draft token count")
    _validate_rows(sparse_ids, sparse_probs)
    for index, token in enumerate(tokens.tolist()):
        q = _candidate_probability(sparse_ids, sparse_probs, index, int(token))
        if not torch.isfinite(q) or float(q.item()) <= 0.0:
            raise ValueError(
                f"draft token at row {index} is absent from the positive sparse support"
            )


def _random_uniform(device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand((), device=device, generator=generator)


def rejection_sample(
    draft_tokens: torch.Tensor | Iterable[int],
    target_prob_rows: torch.Tensor,
    sparse_ids: torch.Tensor,
    sparse_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    uniforms: Iterable[float] | None = None,
    sample_correction: bool = True,
) -> tuple[int, torch.Tensor | None]:
    """Accept a draft prefix and sample the exact target correction token."""

    if target_prob_rows.dim() == 3 and target_prob_rows.shape[0] == 1:
        target_prob_rows = target_prob_rows[0]
    target_prob_rows = target_prob_rows.float()
    if target_prob_rows.dim() != 2:
        raise ValueError("target probability rows must have shape [gamma+1, vocab]")
    if isinstance(draft_tokens, torch.Tensor):
        tokens = draft_tokens.reshape(-1).to(device=target_prob_rows.device, dtype=torch.long)
    else:
        tokens = torch.tensor(list(draft_tokens), device=target_prob_rows.device, dtype=torch.long)
    gamma = int(tokens.numel())
    if target_prob_rows.shape[0] != gamma + 1:
        raise ValueError("target probability rows must include one correction row")
    sparse_ids = sparse_ids.to(device=target_prob_rows.device, dtype=torch.long)
    sparse_probs = sparse_probs.to(device=target_prob_rows.device, dtype=torch.float32)
    if sparse_ids.shape[0] != gamma:
        raise ValueError("sparse probability rows must match draft token count")
    _validate_rows(sparse_ids, sparse_probs, vocab_size=target_prob_rows.shape[-1])
    validate_candidate_support(tokens, sparse_ids, sparse_probs)
    if not torch.isfinite(target_prob_rows).all() or (target_prob_rows < 0).any():
        raise ValueError("target probabilities must be finite and non-negative")
    target_sums = target_prob_rows.sum(dim=-1)
    if not torch.allclose(target_sums, torch.ones_like(target_sums), atol=_NORMALIZATION_TOL, rtol=0.0):
        raise ValueError("target probability rows must be normalized")

    uniform_values = iter(uniforms) if uniforms is not None else None
    for index in range(gamma):
        token = int(tokens[index].item())
        if token < 0 or token >= target_prob_rows.shape[-1]:
            raise ValueError("draft token is outside the target model vocabulary")
        p = target_prob_rows[index, token]
        q = _candidate_probability(sparse_ids, sparse_probs, index, token)
        if not torch.isfinite(p) or float(p.item()) <= 0.0:
            accepted = False
        else:
            ratio = min(1.0, max(0.0, float((p / q).item())))
            if uniform_values is None:
                u = float(_random_uniform(target_prob_rows.device, generator).item())
            else:
                try:
                    u = float(next(uniform_values))
                except StopIteration as exc:
                    raise ValueError("uniforms must contain one value per draft token") from exc
            if not math.isfinite(u) or u < 0.0 or u > 1.0:
                raise ValueError("uniform values must lie in [0, 1]")
            # torch.rand samples from [0, 1), so this strict comparison is
            # equivalent to the specified ``log u < log(p/q)`` test while
            # retaining the exact boundary semantics for deterministic tests.
            accepted = u < ratio
        if accepted:
            continue

        q_dense = torch.zeros_like(target_prob_rows[index])
        q_dense.scatter_(0, sparse_ids[index], sparse_probs[index])
        residual = torch.clamp(target_prob_rows[index] - q_dense, min=0.0)
        residual_mass = residual.sum()
        if not torch.isfinite(residual_mass) or float(residual_mass.item()) <= 0.0:
            raise ValueError("rejection residual distribution has zero probability mass")
        residual = residual / residual_mass
        token_out = torch.multinomial(residual, 1, generator=generator)
        return index, token_out.reshape(1)

    if not sample_correction:
        # An internal scheduler slice can finish with every draft token
        # accepted while the logical round still has more draft rows.  The
        # caller only needs the accepted count in that case; sampling a
        # provisional correction would consume a random number and insert a
        # token that is immediately discarded.
        return gamma, None
    token_out = torch.multinomial(target_prob_rows[gamma], 1, generator=generator)
    return gamma, token_out.reshape(1)


def greedy_accept_count(draft_tokens: torch.Tensor | Iterable[int], target_prob_rows: torch.Tensor) -> int:
    """Return the accepted prefix length for the legacy strict-match mode."""

    tokens = draft_tokens.reshape(-1).tolist() if isinstance(draft_tokens, torch.Tensor) else list(draft_tokens)
    target = target_prob_rows[0] if target_prob_rows.dim() == 3 else target_prob_rows
    for index, token in enumerate(tokens):
        if int(token) != int(torch.argmax(target[index]).item()):
            return index
    return len(tokens)
