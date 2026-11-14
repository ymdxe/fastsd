"""H1-only speculative KV copies, with no arena, chunking or supersession.

The worker owns copy submission. Formal KV is installed only by the executor,
after a nonblocking readiness and snapshot check. Unclaimed copies always run
to completion, including copies invalidated by release or a demand restore.
"""

from collections import deque
from dataclasses import dataclass
import threading
import time

from .cache_protocol import prefetch_horizon, prefetch_pid
from .cache_residency import CPU_RESIDENT, EMERGENCY_FREE_BYTES


def matching_snapshot(previous, current):
    return (previous is not None and current is not None
            and previous.cache is current.cache
            and previous.session_epoch == current.session_epoch
            and previous.content_version == current.content_version
            and previous.valid_len == current.valid_len)


@dataclass(eq=False)
class H1Ticket:
    pid: object
    snapshot: object
    destination_cache: object
    copy_done: object
    copy_start: object
    nbytes: int
    prepared_at: float
    state: str = "prepared"
    reported: bool = False
    submitted: bool = False
    fenced: bool = False
    submitted_bytes: int = 0

    def matches(self, snapshot):
        return matching_snapshot(self.snapshot, snapshot)


class CudaH1Copy:
    """Allocate the complete destination before submitting any H2D work."""

    def __init__(self, device):
        import torch
        from .cache_views import dynamic_cache_from_views
        self.torch = torch
        self.make_cache = dynamic_cache_from_views
        self.device = torch.device(device)
        self.stream = None

    def _stream(self):
        if self.stream is None:
            self.stream = self.torch.cuda.Stream(device=self.device)
        return self.stream

    def free_bytes(self):
        return self.torch.cuda.mem_get_info(self.device)[0]

    @staticmethod
    def source_ready(cache):
        event = getattr(cache, "_fastsd_ready_event", None)
        return event is None or event.query()

    @staticmethod
    def describe(cache):
        tensors = [tensor for pair in cache.to_legacy_cache() for tensor in pair]
        if not tensors or any(t.device.type != "cpu" or not t.is_pinned() for t in tensors):
            return 0
        return sum(t.numel() * t.element_size() for t in tensors)

    def allocate(self, pid, snapshot, nbytes, prepared_at):
        torch = self.torch
        # No buffers/events are allocated after submit() starts the copy.
        with torch.cuda.stream(self._stream()):
            pairs = [tuple(torch.empty_like(t, device=self.device) for t in pair)
                     for pair in snapshot.cache.to_legacy_cache()]
            destination = self.make_cache(pairs)
            start = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            # PyTorch creates CUDA events lazily: materialize both before DMA
            # so recording the completion fence needs no new event allocation.
            start.record(self.stream)
            done.record(self.stream)
        return H1Ticket(pid, snapshot, destination, done, start, nbytes, prepared_at)

    def submit(self, ticket):
        with self.torch.cuda.stream(self._stream()):
            ticket.copy_start.record(self.stream)
            ticket.submitted = True
            try:
                for source_pair, target_pair in zip(ticket.snapshot.cache.to_legacy_cache(),
                                                    ticket.destination_cache.to_legacy_cache()):
                    for source, target in zip(source_pair, target_pair):
                        target.copy_(source, non_blocking=True)
                        ticket.submitted_bytes += source.numel() * source.element_size()
            finally:
                # Also fence a partially submitted copy if submission fails.
                ticket.copy_done.record(self.stream)
                ticket.fenced = True

    @staticmethod
    def ready(ticket):
        return ticket.fenced and ticket.copy_done.query()

    @staticmethod
    def copy_ms(ticket):
        return ticket.copy_start.elapsed_time(ticket.copy_done)

    @staticmethod
    def consume(ticket, compute_stream):
        # query() already proved completion; no wait_event or synchronize here.
        for pair in ticket.destination_cache.to_legacy_cache():
            for tensor in pair:
                tensor.record_stream(compute_stream)
        return ticket.destination_cache

    @staticmethod
    def release(ticket):
        ticket.snapshot = ticket.destination_cache = None
        ticket.copy_start = ticket.copy_done = None
        ticket.state = "retired"

    def finish(self, ticket):
        if not ticket.fenced:
            # A CUDA error prevented recording the final fence. Keep all
            # storage alive until shutdown can drain the whole copy stream.
            self.stream.synchronize()
            ticket.copy_done.record(self.stream)
            ticket.fenced = True
        ticket.copy_done.synchronize()  # shutdown only

    def is_oom(self, error):
        return isinstance(error, self.torch.cuda.OutOfMemoryError)


class H1Prefetcher:
    """One candidate table and at most one outstanding copy per pid.

    max_bytes bounds live destinations, including pending copies discarded by
    demand. A reclaim callback may request eviction from the executor; this
    worker never waits for eviction or mutates formal KV/session placement.
    """

    def __init__(self, *, residency, manager, get_cache, driver, metrics,
                 max_bytes, reclaim=lambda needed: None, thread_setup=lambda: None,
                 tick=lambda: 0, clock=time.perf_counter):
        self.residency, self.manager = residency, manager
        self.get_cache, self.driver = get_cache, driver
        self.metrics, self.max_bytes = metrics, max(0, int(max_bytes))
        self.reclaim, self.thread_setup, self.clock = reclaim, thread_setup, clock
        self.tick = tick
        self.lock = residency.lock
        self.tickets, self.preparing = {}, set()
        self.reserved_bytes = 0
        self.queue, self.queued = deque(), set()
        self.condition = threading.Condition()
        self.stopped = False
        self.thread = None
        for name in ("prefetch_ready_hit", "prefetch_pending_hit", "prefetch_demand_miss",
                     "prefetch_skipped_no_memory", "prefetch_skipped_stale",
                     "prefetch_copy_ms", "prefetch_copy_count", "prefetch_lead_ms",
                     "prefetch_lead_count", "prefetch_bytes", "prefetch_candidates",
                     "prefetch_candidates_created", "prefetch_candidates_peak",
                     "prefetch_allocation_oom", "prefetch_async_failed",
                     "prefetch_reserved_bytes_peak"):
            metrics.setdefault(name, 0)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="fastsd-h1-prefetch", daemon=True)
            self.thread.start()

    def submit(self, targets, **_kwargs):
        added = False
        with self.condition:
            for target in targets:
                pid = prefetch_pid(target)
                if prefetch_horizon(target) == 1 and pid not in self.queued:
                    self.queue.append(pid)
                    self.queued.add(pid)
                    added = True
            self.condition.notify()
        return added

    refresh = submit

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify()

    def join(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)

    def _run(self):
        try:
            self.thread_setup()
        except Exception as exc:
            self._error(exc)
            return
        while True:
            with self.condition:
                if not self.queue and not self.stopped:
                    self.condition.wait(.01)
                if self.stopped:
                    return
                pid = self.queue.popleft() if self.queue else None
                self.queued.discard(pid)
            try:
                self.collect()
                if pid is not None:
                    self.prepare(pid)
            except Exception as exc:
                self._error(exc)

    def _error(self, exc):
        self.metrics["prefetch_async_failed"] += 1
        self.metrics["prefetch_async_last_error"] = repr(exc)

    def _eligible_locked(self, pid):
        record = self.manager.get(pid)
        return (record is not None and record.state == CPU_RESIDENT
                and not record.in_transfer and not record.protected(self.tick())
                and pid not in self.residency.active and pid not in self.residency.reserved)

    def _snapshot_locked(self, pid):
        cache = self.get_cache(pid)
        return None if cache is None else self.residency.snapshot_locked(pid, cache)

    def prepare(self, pid):
        with self.lock:
            if pid in self.tickets or pid in self.preparing or not self._eligible_locked(pid):
                return False
            snapshot = self._snapshot_locked(pid)
            if snapshot is None or not self.driver.source_ready(snapshot.cache):
                return False
            self.preparing.add(pid)
        ticket = None
        accounted = False
        copied_accounted = False
        try:
            size = self.driver.describe(snapshot.cache)
            if size <= 0:
                return False
            with self.lock:
                within_budget = self.reserved_bytes + size <= self.max_bytes
            if not within_budget:
                self.metrics["prefetch_skipped_no_memory"] += 1
                return False
            required = size + EMERGENCY_FREE_BYTES
            free = self.driver.free_bytes()  # one admission check, never per chunk
            if free < required:
                self.reclaim(required)
                if self.driver.free_bytes() < required:
                    self.metrics["prefetch_skipped_no_memory"] += 1
                    return False
            with self.lock:
                current = self._snapshot_locked(pid)
                if (not matching_snapshot(snapshot, current)
                        or not self._eligible_locked(pid)):
                    self.metrics["prefetch_skipped_stale"] += 1
                    return False
                self.reserved_bytes += size
                self.metrics["prefetch_reserved_bytes_peak"] = max(
                    self.metrics["prefetch_reserved_bytes_peak"], self.reserved_bytes)
                accounted = True
            ticket = self.driver.allocate(pid, snapshot, size, self.clock())
            with self.lock:
                current = self._snapshot_locked(pid)
                if current is None or not ticket.matches(current) or not self._eligible_locked(pid):
                    self.metrics["prefetch_skipped_stale"] += 1
                    self.driver.release(ticket)
                    self.reserved_bytes -= size
                    accounted = False
                    return False
            self.driver.submit(ticket)
            ticket.state = "copying"
            self.metrics["prefetch_bytes"] += size
            copied_accounted = True
            self.metrics["prefetch_candidates_created"] += 1
            self.metrics["prefetch_async_submitted"] = self.metrics.get("prefetch_async_submitted", 0) + 1
            with self.lock:
                self.tickets[pid] = ticket
                current = self._snapshot_locked(pid)
                if current is not None and ticket.matches(current) and self._eligible_locked(pid):
                    self.residency.candidates[pid] = ticket
                else:
                    ticket.state = "draining"
                    self.metrics["prefetch_skipped_stale"] += 1
                self._gauge_locked()
            return True
        except Exception as exc:
            if self.driver.is_oom(exc):
                self.metrics["prefetch_allocation_oom"] += 1
                self.metrics["prefetch_skipped_no_memory"] += 1
            else:
                self.metrics["prefetch_async_failed"] += 1
                self.metrics["prefetch_async_last_error"] = repr(exc)
            if ticket is not None and ticket.submitted:
                if not copied_accounted:
                    self.metrics["prefetch_bytes"] += ticket.submitted_bytes
                ticket.state = "draining"
                with self.lock:
                    self.tickets[pid] = ticket
            else:
                if ticket is not None:
                    self.driver.release(ticket)
                if accounted:
                    with self.lock:
                        self.reserved_bytes -= size
            return False
        finally:
            with self.lock:
                self.preparing.discard(pid)

    def _gauge_locked(self):
        count = len(self.residency.candidates)
        self.metrics["prefetch_candidates"] = count
        self.metrics["prefetch_candidates_peak"] = max(self.metrics["prefetch_candidates_peak"], count)

    def _report_locked(self, ticket):
        if not ticket.reported:
            self.metrics["prefetch_copy_ms"] += self.driver.copy_ms(ticket)
            self.metrics["prefetch_copy_count"] += 1
            self.metrics["prefetch_async_completed"] = self.metrics.get("prefetch_async_completed", 0) + 1
            ticket.reported = True

    def _release_locked(self, ticket):
        if self.tickets.get(ticket.pid) is not ticket:
            return
        if self.residency.candidates.get(ticket.pid) is ticket:
            self.residency.candidates.pop(ticket.pid)
        self.tickets.pop(ticket.pid, None)
        self.reserved_bytes -= ticket.nbytes
        self.driver.release(ticket)
        self._gauge_locked()

    def collect(self):
        with self.lock:
            for ticket in list(self.tickets.values()):
                if ticket.state == "claimed" or not self.driver.ready(ticket):
                    continue
                self._report_locked(ticket)
                current = self._snapshot_locked(ticket.pid)
                stale = current is None or not ticket.matches(current)
                if ticket.state == "draining" or stale:
                    if stale and ticket.state != "draining":
                        self.metrics["prefetch_skipped_stale"] += 1
                    self._release_locked(ticket)

    def claim(self, pid, snapshot):
        with self.lock:
            ticket = self.residency.candidates.pop(pid, None)
            self._gauge_locked()
            if ticket is None:
                return None
            self.metrics["prefetch_claims"] = self.metrics.get("prefetch_claims", 0) + 1
            lead = max(0., (self.clock() - ticket.prepared_at) * 1000.)
            self.metrics["prefetch_lead_ms"] += lead
            self.metrics["prefetch_lead_count"] += 1
            self.metrics["prefetch_claim_lead_ms_sum"] = self.metrics.get("prefetch_claim_lead_ms_sum", 0) + lead
            self.metrics["prefetch_claim_lead_ms_max"] = max(
                self.metrics.get("prefetch_claim_lead_ms_max", 0), lead)
            if not ticket.matches(snapshot):
                self.metrics["prefetch_skipped_stale"] += 1
                self.retire(ticket)
                return None
            if not self.driver.ready(ticket):
                self.metrics["prefetch_pending_hit"] += 1
                self.retire(ticket)
                return None
            self._report_locked(ticket)
            ticket.state = "claimed"
            self.metrics["prefetch_ready_hit"] += 1
            return ticket

    def consume(self, ticket, compute_stream):
        return self.driver.consume(ticket, compute_stream)

    def retire(self, ticket):
        with self.lock:
            if ticket.state == "retired":
                return
            if self.driver.ready(ticket):
                self._report_locked(ticket)
                self._release_locked(ticket)
            else:
                # Retain destination, event and pinned source until DMA ends.
                ticket.state = "draining"

    def close(self):
        self.stop()
        self.join()
        with self.lock:
            tickets = list(self.tickets.values())
        for ticket in tickets:
            self.driver.finish(ticket)
            self.retire(ticket)
