"""Arrival-process helpers used by the edge benchmark client."""

import random
from typing import Any, List, Sequence, Tuple


def poisson_arrival_offsets(
    num_arrivals: int,
    rate_rps: float,
    seed: int,
) -> List[float]:
    """Return cumulative arrival offsets for a Poisson process.

    A Poisson process has exponentially distributed inter-arrival times.  The
    first request is delayed by one sampled inter-arrival interval as well,
    which keeps the finite benchmark faithful to the configured process.
    """
    if num_arrivals < 0:
        raise ValueError("num_arrivals must be non-negative")
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")

    rng = random.Random(seed)
    offsets: List[float] = []
    elapsed = 0.0
    for _ in range(num_arrivals):
        elapsed += rng.expovariate(rate_rps)
        offsets.append(elapsed)
    return offsets


def shard_samples(
    samples: Sequence[Any],
    num_shards: int,
    shard_id: int,
    max_items: int,
) -> List[Tuple[int, Any]]:
    """Round-robin samples while preserving each item's global index.

    ``max_items <= 0`` means no per-shard limit, which is useful for a full
    dataset run.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")

    selected = [
        (index, sample)
        for index, sample in enumerate(samples)
        if index % num_shards == shard_id
    ]
    if max_items > 0:
        return selected[:max_items]
    return selected
