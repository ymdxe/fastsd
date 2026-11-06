"""Canonicalize and validate requests before scheduler/model admission.

The HTTP models validate types, but they do not validate relationships between
``prefix_len``, ``gamma`` and the token payload.  The same validator is used at
the Cloud boundary and again at the worker ingress so malformed internal queue
messages cannot terminate the target worker.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


MAX_REQUEST_PREFIX_LEN = 131_072
DEFAULT_MAX_TOKENS = 400


_TIMING_SECONDS_FIELDS = (
    "local_prefill_s",
    "local_decode_per_token_s",
    "push_s",
    "pull_s",
    "last_pull_s",
    "local_prefill_ms",
    "local_decode_per_token_ms",
    "push_ms",
    "pull_ms",
)


def _validate_timing_fields(out: dict, *, require_prefill: bool = False) -> None:
    """Validate latency telemetry before it can influence scheduling.

    Timing-aware requests carry durations in seconds (``*_s``); millisecond
    aliases are accepted only for migration of old experiment records.  No
    malformed value is converted into an RTT/2 estimate.
    """

    if require_prefill and out.get("timing_ready", True) and "local_prefill_s" not in out and "prefill_first_token_s" not in out:
        raise ValueError("timing-required prefill must report local_prefill_s")
    for field in _TIMING_SECONDS_FIELDS + ("prefill_first_token_s",):
        if field not in out or out[field] is None:
            continue
        try:
            value = float(out[field])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a finite non-negative number") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be a finite non-negative number")
    if "prefill_gamma" in out or require_prefill:
        gamma_value = out.get("prefill_gamma", out.get("gamma"))
        gamma = _integer(gamma_value, "prefill_gamma")
        if gamma < 1 or gamma > int(out.get("max_tokens", DEFAULT_MAX_TOKENS)):
            # ``max_tokens`` is normally supplied by the caller; the explicit
            # bound is repeated here so a malformed timing packet cannot alter
            # m_i in the scheduler.
            raise ValueError("prefill_gamma must be positive and within max_tokens")
        out["prefill_gamma"] = gamma
    if "edge_send_time_ns" in out:
        send_ns = _integer(out["edge_send_time_ns"], "edge_send_time_ns")
        if send_ns <= 0:
            raise ValueError("edge_send_time_ns must be positive")
        out["edge_send_time_ns"] = send_ns
    if "clock_offset_ns" in out:
        # Clock offset may be negative, but it must be integral and finite.
        out["clock_offset_ns"] = _integer(out["clock_offset_ns"], "clock_offset_ns")


def _integer(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be a finite integer")
    return int(number)


def _token_count(draft_output) -> int:
    shape = getattr(draft_output, "shape", None)
    if shape is not None:
        if len(shape) != 2 or int(shape[0]) != 1:
            raise ValueError("draft_output must have shape [1, tokens]")
        return int(shape[1])
    try:
        return len(draft_output)
    except TypeError as exc:
        raise ValueError("draft_output must be a token sequence") from exc


def canonicalize_request(
    request: Mapping,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_prefix_len: int = MAX_REQUEST_PREFIX_LEN,
) -> dict:
    """Return a validated request copy with canonical integer fields.

    ``gamma`` is required and positive for Verify.  Prefill requests do not
    need gamma and are canonicalized to zero.  The token-length checks cover
    both external full-prefix and tail-only payloads, including a physical
    bridge token.
    """
    out = dict(request)
    task_type = str(out.get("task_type", ""))
    if task_type not in {"prefill", "verify"}:
        raise ValueError(f"unknown task_type: {task_type!r}")

    prefix_len = _integer(out.get("prefix_len"), "prefix_len")
    max_tokens = _integer(max_tokens, "max_tokens")
    max_prefix_len = _integer(max_prefix_len, "max_prefix_len")
    if max_tokens <= 0 or max_prefix_len <= 0:
        raise ValueError("request validation limits must be positive")
    if prefix_len <= 0 or prefix_len > max_prefix_len:
        raise ValueError(f"prefix_len must be in [1, {max_prefix_len}]")

    token_count = _token_count(out.get("draft_output"))
    if token_count < 0 or token_count > max_prefix_len + max_tokens + 1:
        raise ValueError("draft_output is too large")

    if task_type == "prefill":
        if token_count < prefix_len:
            raise ValueError("prefill prefix_len exceeds draft_output length")
        requested_gamma = out.get("prefill_gamma", out.get("gamma"))
        if out.get("timing_required", False):
            if requested_gamma is None:
                raise ValueError("timing-required prefill must include gamma")
            out["max_tokens"] = max_tokens
            _validate_timing_fields(out, require_prefill=True)
        elif requested_gamma is not None and requested_gamma not in (0, "0"):
            # Preserve the first-round gamma separately from the historical
            # canonical ``gamma=0`` Prefill field.
            out["max_tokens"] = max_tokens
            _validate_timing_fields(out, require_prefill=False)
        gamma = 0
    else:
        gamma = _integer(out.get("gamma"), "gamma")
        if gamma < 1 or gamma > max_tokens:
            raise ValueError(f"gamma must be in [1, {max_tokens}]")
        bridge = 1 if bool(out.get("has_bridge_token", False)) else 0
        if bool(out.get("tail_only", False)):
            required_tokens = gamma + bridge
        else:
            required_tokens = prefix_len + gamma
        if token_count < required_tokens:
            raise ValueError(
                f"draft_output has {token_count} tokens, requires at least {required_tokens}"
            )
        if any(field in out for field in _TIMING_SECONDS_FIELDS + ("prefill_first_token_s",)):
            _validate_timing_fields(out)

        # Requests created before sparse rejection sampling did not carry a
        # method field.  Preserve their strict-match behavior at this utility
        # boundary; the current Edge/Cloud launchers always send the explicit
        # ``rejection`` default and therefore receive the stricter wire check.
        method = str(out.get("verify_method", "greedy")).lower()
        if method not in {"greedy", "rejection"}:
            raise ValueError("verify_method must be 'greedy' or 'rejection'")
        out["verify_method"] = method
        if "sampling_seed" in out and out["sampling_seed"] is not None:
            out["sampling_seed"] = _integer(out["sampling_seed"], "sampling_seed")
        if method == "rejection":
            block = out.get("draft_prob_block")
            if not isinstance(block, str) or not block:
                raise ValueError("rejection verify requires draft_prob_block")

    out["task_type"] = task_type
    out["prefix_len"] = prefix_len
    out["gamma"] = gamma
    return out
