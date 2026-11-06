"""Placement policy for target KV caches.

Residency is the default: a cache is only moved to CPU when the caller has
already established memory pressure (explicit budget exceeded) and the cache
is not needed by an in-flight chunk.  There is no "always offload" stage.
"""

import os

_DISABLED = {"1", "true", "yes", "on"}


def target_cache_offload_disabled() -> bool:
    """Operator kill switch; forces residency-only operation."""

    return os.environ.get("FASTSD_DISABLE_TARGET_CACHE_OFFLOAD", "").lower() in _DISABLED


def should_offload_target_cache(task_type, pid=None, pinned_gpu_pids=None, *,
                                continuing=False, memory_pressure=False) -> bool:
    """Return True only when this cache should leave the GPU right now.

    ``pinned_gpu_pids`` is accepted for backward compatibility with callers
    that still pass the legacy resident set; placement is decided by the
    residency manager, not by membership in that set.
    """

    del pid, pinned_gpu_pids
    if target_cache_offload_disabled():
        return False
    if not memory_pressure:
        return False
    if continuing:
        # An unfinished chunk will be forwarded again immediately.
        return False
    return task_type in {"prefill", "verify"}
