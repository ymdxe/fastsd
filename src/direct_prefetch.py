"""Small direct cache prefetcher.

The direct implementation deliberately has no coordinator, intent queue,
supersede protocol, arena, chunk scheduler, or event pool.  A prediction calls
``prepare`` once, and a verify demand calls ``take`` once.  The driver is
injected so the class remains usable in CPU-only unit tests and the production
path can reuse the existing complete-cache CUDA copy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .cache_residency import CPU_RESIDENT, EMERGENCY_FREE_BYTES
from .minimal_prefetch import matching_snapshot


@dataclass(eq=False)
class Entry:
    pid: object
    snapshot: object
    cache: object
    copy_done: object
    nbytes: int
    prepared_at: float
    driver_ticket: object = None
    copy_start: object = None
    state: str = "copying"
    submitted: bool = False
    released: bool = False
    reported: bool = False

    @property
    def destination_cache(self):
        """Compatibility alias for the existing CUDA driver."""
        return self.cache

    def matches(self, snapshot) -> bool:
        return (not self.released and matching_snapshot(self.snapshot, snapshot))

    def ready(self) -> bool:
        if self.released:
            return False
        if self.state == "ready":
            return True
        ticket = self.driver_ticket
        if ticket is not None:
            ready = getattr(ticket, "ready", None)
            if callable(ready):
                try:
                    return bool(ready())
                except Exception:
                    return False
        event = self.copy_done
        query = getattr(event, "query", None)
        return bool(query()) if callable(query) else False

    def release(self):
        self.released = True
        self.state = "stale"
        self.snapshot = None
        self.cache = None
        self.copy_done = None
        self.copy_start = None
        self.driver_ticket = None


class DirectPrefetcher:
    """One-entry-per-pid, non-blocking direct prefetch table.

    ``driver`` must expose ``free_bytes``, ``describe``, ``allocate``,
    ``submit``, ``ready``, ``copy_ms``, ``release`` and ``consume``.  This is
    the same small surface implemented by :class:`CudaH1Copy`.
    """

    def __init__(self, *, residency, manager, get_cache, driver, metrics,
                 max_bytes, reclaim=lambda needed: None,
                 tick=lambda: 0, clock=time.perf_counter):
        self.residency = residency
        self.manager = manager
        self.get_cache = get_cache
        self.driver = driver
        self.metrics = metrics
        self.max_bytes = max(0, int(max_bytes))
        self.reclaim = reclaim
        self.tick = tick
        self.clock = clock
        self.lock = residency.lock
        self.prefetched: dict[object, Entry] = {}
        self.claimed: dict[object, Entry] = {}
        self.draining: list[Entry] = []
        self.reserved_bytes = 0
        self.stopped = False
        self.last_take_state = None
        self._init_metrics()

    def _init_metrics(self):
        for name in (
            "prefetch_ready_hit", "prefetch_pending_hit", "prefetch_demand_miss",
            "prefetch_skipped_stale", "prefetch_skipped_no_memory",
            "prefetch_candidates_created", "prefetch_bytes", "prefetch_copy_ms",
            "prefetch_copy_count", "prefetch_lead_ms", "prefetch_lead_count",
            "prefetch_wasted", "prefetch_fallback_to_demand", "prefetch_candidates",
            "prefetch_candidates_peak", "prefetch_allocation_oom",
            "prefetch_async_failed",
        ):
            self.metrics.setdefault(name, 0)

    # The engine uses the worker lifecycle for all implementations.  Direct
    # copies are submitted synchronously by prepare, so these are intentional
    # no-ops rather than a hidden coordinator thread.
    def start(self):
        return None

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        return None

    @property
    def candidates(self):
        return self.prefetched

    def _eligible_locked(self, pid):
        record = self.manager.get(pid)
        return (record is not None and record.state == CPU_RESIDENT
                and not record.in_transfer
                and not record.protected(self.tick())
                and pid not in self.residency.active
                and pid not in self.residency.reserved)

    def _snapshot_locked(self, pid):
        cache = self.get_cache(pid)
        return None if cache is None else self.residency.snapshot_locked(pid, cache)

    def _gauge_locked(self):
        count = len(self.prefetched)
        self.metrics["prefetch_candidates"] = count
        self.metrics["prefetch_candidates_peak"] = max(
            self.metrics.get("prefetch_candidates_peak", 0), count)

    def _entry_from_ticket(self, pid, snapshot, ticket, nbytes, prepared_at):
        cache = getattr(ticket, "destination_cache", None)
        if cache is None:
            cache = getattr(ticket, "cache", None)
        return Entry(
            pid=pid,
            snapshot=snapshot,
            cache=cache,
            copy_done=getattr(ticket, "copy_done", None),
            copy_start=getattr(ticket, "copy_start", None),
            nbytes=int(nbytes),
            prepared_at=prepared_at,
            driver_ticket=ticket,
            submitted=bool(getattr(ticket, "submitted", False)),
        )

    def prepare(self, pid, snapshot=None) -> bool:
        """Submit one complete-cache H2D without waiting for completion."""
        with self.lock:
            if (pid in self.prefetched
                    or pid in self.claimed
                    or any(entry.pid == pid and not entry.released for entry in self.draining)):
                return False
            if not self._eligible_locked(pid):
                return False
            snapshot = snapshot or self._snapshot_locked(pid)
            source_ready = getattr(self.driver, "source_ready", lambda cache: True)
            if snapshot is None or not source_ready(snapshot.cache):
                return False
        try:
            nbytes = int(self.driver.describe(snapshot.cache))
        except Exception as exc:
            self.metrics["prefetch_async_failed"] += 1
            self.metrics["prefetch_async_last_error"] = repr(exc)
            return False
        if nbytes <= 0:
            return False
        with self.lock:
            if self.reserved_bytes + nbytes > self.max_bytes:
                self.metrics["prefetch_skipped_no_memory"] += 1
                return False
        required = nbytes + EMERGENCY_FREE_BYTES
        try:
            free = int(self.driver.free_bytes())
            if free < required:
                self.reclaim(required)
                free = int(self.driver.free_bytes())
            if free < required:
                self.metrics["prefetch_skipped_no_memory"] += 1
                return False
        except Exception as exc:
            self.metrics["prefetch_async_failed"] += 1
            self.metrics["prefetch_async_last_error"] = repr(exc)
            return False
        with self.lock:
            current = self._snapshot_locked(pid)
            if (current is None or not matching_snapshot(snapshot, current)
                    or not self._eligible_locked(pid)):
                self.metrics["prefetch_skipped_stale"] += 1
                return False
            self.reserved_bytes += nbytes
        prepared_at = self.clock()
        ticket = None
        try:
            # allocate() creates the complete destination and event before the
            # driver starts DMA; no allocation occurs during copy submission.
            ticket = self.driver.allocate(pid, snapshot, nbytes, prepared_at)
            with self.lock:
                current = self._snapshot_locked(pid)
                if current is None or not matching_snapshot(snapshot, current) or not self._eligible_locked(pid):
                    self.metrics["prefetch_skipped_stale"] += 1
                    self._release_ticket(ticket)
                    self.reserved_bytes -= nbytes
                    return False
            self.driver.submit(ticket)
            entry = self._entry_from_ticket(pid, snapshot, ticket, nbytes, prepared_at)
            with self.lock:
                self.prefetched[pid] = entry
                self.residency.candidates[pid] = entry
                self._gauge_locked()
            self.metrics["prefetch_candidates_created"] += 1
            self.metrics["prefetch_bytes"] += nbytes
            return True
        except Exception as exc:
            if ticket is not None and bool(getattr(ticket, "submitted", False)):
                entry = self._entry_from_ticket(pid, snapshot, ticket, nbytes, prepared_at)
                with self.lock:
                    self.draining.append(entry)
            else:
                if ticket is not None:
                    self._release_ticket(ticket)
                with self.lock:
                    self.reserved_bytes = max(0, self.reserved_bytes - nbytes)
            is_oom = getattr(self.driver, "is_oom", lambda error: False)(exc)
            self.metrics["prefetch_skipped_no_memory" if is_oom else "prefetch_async_failed"] += 1
            if not is_oom:
                self.metrics["prefetch_async_last_error"] = repr(exc)
            return False

    def _release_ticket(self, ticket):
        try:
            self.driver.release(ticket)
        except Exception:
            pass

    def _retire_locked(self, entry, *, wasted=False):
        if entry is None or entry.released:
            return
        if not entry.ready():
            entry.state = "stale"
            if entry not in self.draining:
                self.draining.append(entry)
            if wasted:
                self.metrics["prefetch_wasted"] += 1
            return
        if entry.driver_ticket is not None:
            try:
                self.metrics["prefetch_copy_ms"] += float(self.driver.copy_ms(entry.driver_ticket))
                self.metrics["prefetch_copy_count"] += 1
            except Exception:
                pass
            self._release_ticket(entry.driver_ticket)
        self.reserved_bytes = max(0, self.reserved_bytes - entry.nbytes)
        entry.release()
        if wasted:
            self.metrics["prefetch_wasted"] += 1

    def take(self, pid, snapshot):
        """Claim an entry without waiting; caller falls back to demand H2D."""
        with self.lock:
            entry = self.prefetched.pop(pid, None)
            self.residency.candidates.pop(pid, None)
            self._gauge_locked()
            if entry is None:
                self.last_take_state = "miss"
                return None
            lead = max(0.0, (self.clock() - entry.prepared_at) * 1000.0)
            self.metrics["prefetch_lead_ms"] += lead
            self.metrics["prefetch_lead_count"] += 1
            if not entry.matches(snapshot):
                self.last_take_state = "stale"
                self.metrics["prefetch_skipped_stale"] += 1
                self._retire_locked(entry, wasted=True)
                return None
            if not entry.ready():
                self.last_take_state = "pending"
                self.metrics["prefetch_pending_hit"] += 1
                self._retire_locked(entry, wasted=True)
                return None
            entry.state = "ready"
            self.last_take_state = "ready"
            self.claimed[pid] = entry
            self.metrics["prefetch_ready_hit"] += 1
            return entry

    def record_demand_fallback(self):
        self.metrics["prefetch_demand_miss"] += 1
        self.metrics["prefetch_fallback_to_demand"] += 1

    def release(self, pid):
        """Drop an unclaimed candidate; an in-flight copy drains safely."""
        with self.lock:
            entry = self.prefetched.pop(pid, None)
            self.residency.candidates.pop(pid, None)
            if entry is None:
                entry = self.claimed.pop(pid, None)
            if entry is not None:
                self._retire_locked(entry, wasted=True)
            self._gauge_locked()

    def retire(self, entry):
        with self.lock:
            if entry is not None:
                self.claimed.pop(entry.pid, None)
            self._retire_locked(entry)

    def collect(self):
        with self.lock:
            entries = list(self.draining)
            self.draining.clear()
            for pid, entry in list(self.prefetched.items()):
                if entry.ready():
                    current = self._snapshot_locked(pid)
                    if current is None or not entry.matches(current):
                        self.prefetched.pop(pid, None)
                        self.residency.candidates.pop(pid, None)
                        entry.state = "stale"
                        self.metrics["prefetch_skipped_stale"] += 1
                        entries.append(entry)
            for entry in entries:
                self._retire_locked(entry, wasted=entry.state == "stale")
            self._gauge_locked()

    def close(self):
        self.stop()
        with self.lock:
            entries = list(self.prefetched.values()) + list(self.claimed.values()) + list(self.draining)
            self.prefetched.clear()
            self.claimed.clear()
            self.draining.clear()
            self.residency.candidates.clear()
            for entry in entries:
                if entry.driver_ticket is not None and not entry.ready():
                    finish = getattr(self.driver, "finish", None)
                    if callable(finish):
                        finish(entry.driver_ticket)
                self._retire_locked(entry)
            self._gauge_locked()

    def stats(self):
        with self.lock:
            result = {key: self.metrics.get(key) for key in (
                "prefetch_ready_hit", "prefetch_pending_hit", "prefetch_demand_miss",
                "prefetch_skipped_stale", "prefetch_skipped_no_memory",
                "prefetch_candidates_created", "prefetch_bytes", "prefetch_copy_ms",
                "prefetch_lead_ms", "prefetch_wasted", "prefetch_fallback_to_demand",
            )}
            result["prefetch_candidates"] = len(self.prefetched)
            result["reserved_bytes"] = self.reserved_bytes
            return result

