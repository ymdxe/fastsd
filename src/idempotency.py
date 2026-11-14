"""Process-local response replay, independent of target session ownership."""

import asyncio
from collections import OrderedDict
from copy import deepcopy
import time


class IdempotencyConflict(ValueError):
    pass


class IdempotencyCache:
    """Bounded LRU with an idle TTL; active operations survive HTTP cancellation.

    Only successful responses are retained. Concurrent callers share the same
    task, including its failure; a later call may retry a failed operation.
    All methods run on the server event loop.
    """

    def __init__(self, max_entries=4096, ttl_s=1800, clock=time.monotonic):
        if int(max_entries) <= 0 or float(ttl_s) <= 0:
            raise ValueError("idempotency cache size and TTL must be positive")
        self.max_entries = int(max_entries)
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self.completed = OrderedDict()
        self.inflight = {}
        self.metrics = dict(idempotent_cache_hit=0, idempotent_inflight_wait=0,
                            cache_evictions=0)

    def _expire(self):
        now = self.clock()
        while self.completed:
            key, (_, _, expires) = next(iter(self.completed.items()))
            if expires > now:
                break
            self.completed.pop(key)
            self.metrics["cache_evictions"] += 1

    async def execute(self, key, fingerprint, operation):
        self._expire()
        if key in self.completed:
            previous, response, _ = self.completed[key]
            if previous != fingerprint:
                raise IdempotencyConflict("request key reused with different payload")
            self.completed[key] = (previous, response, self.clock() + self.ttl_s)
            self.completed.move_to_end(key)
            self.metrics["idempotent_cache_hit"] += 1
            return deepcopy(response), True
        if key in self.inflight:
            previous, task = self.inflight[key]
            if previous != fingerprint:
                raise IdempotencyConflict("request key reused with different payload")
            self.metrics["idempotent_inflight_wait"] += 1
            return deepcopy(await asyncio.shield(task)), True

        async def run():
            try:
                response = await operation()
                if isinstance(response, dict) and "error" not in response:
                    self.completed[key] = (fingerprint, deepcopy(response),
                                           self.clock() + self.ttl_s)
                    self._expire()
                    while len(self.completed) > self.max_entries:
                        self.completed.popitem(last=False)
                        self.metrics["cache_evictions"] += 1
                return response
            finally:
                self.inflight.pop(key, None)

        task = asyncio.create_task(run())
        self.inflight[key] = (fingerprint, task)
        # An abandoned HTTP caller must not leave an unobserved exception.
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        return deepcopy(await asyncio.shield(task)), False

    def stats(self):
        # Health and shutdown handlers may read this from a worker thread.
        # Expiration/mutation stays on the event loop in execute().
        return {**self.metrics, "cache_entries": len(self.completed),
                "inflight_requests": len(self.inflight),
                "max_entries": self.max_entries, "ttl_s": self.ttl_s}
