"""Small, dependency-free helpers for request latency metrics."""


def elapsed_ms(start_s: float, end_s: float) -> float:
    """Return a non-negative elapsed duration in milliseconds."""
    return max(0.0, (float(end_s) - float(start_s)) * 1000.0)


def tpot_ms(first_token_s: float, completion_s: float, output_tokens: int) -> float:
    """Compute standard TPOT after the first output token.

    TPOT = (completion - first-token time) / (output_tokens - 1).  A request
    with fewer than two output tokens has no post-first-token interval and is
    reported as zero rather than dividing by zero.
    """
    if output_tokens <= 1:
        return 0.0
    return elapsed_ms(first_token_s, completion_s) / float(output_tokens - 1)


def request_latency(e2e_ms, first_token_ms, output_tokens):
    if e2e_ms is None:
        return first_token_ms, None, False
    fallback = first_token_ms is None
    ttft = e2e_ms if fallback else first_token_ms
    tpot = (e2e_ms - ttft) / (output_tokens - 1) if output_tokens > 1 else 0.0
    return ttft, tpot, fallback
