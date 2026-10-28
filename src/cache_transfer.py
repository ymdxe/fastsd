"""Event-fenced KV transport. Submission never touches the compute stream."""
from collections import defaultdict
from contextlib import contextmanager
import threading
import time
import weakref
import torch
from .cache_views import dynamic_cache_from_views
from .cache_protocol import CacheSnapshot, CacheLease, KVTransferTicket


class _Lease:
    def __init__(self, tensors):
        self.tensors, self.events = tensors, []

    def ready(self):
        return all(e.query() for e in self.events)


class PinnedBufferPool:
    def __init__(self, max_bytes=128 * 1024 ** 2):
        self.max_bytes = int(max_bytes)
        self.free, self.pending = defaultdict(list), []
        self.bytes, self.closed = 0, False
        self.lock = threading.RLock()

    @staticmethod
    def _key(tensor):
        return tuple(tensor.shape), tensor.dtype

    def collect(self):
        with self.lock:
            pending, self.pending = self.pending, []
            for lease in pending:
                if not lease.ready():
                    self.pending.append(lease)
                    continue
                for tensor in lease.tensors:
                    size = tensor.numel() * tensor.element_size()
                    if not self.closed and self.bytes + size <= self.max_bytes:
                        self.free[self._key(tensor)].append(tensor)
                        self.bytes += size

    def acquire(self, like, *, allocate=True):
        self.collect()
        with self.lock:
            bucket = self.free.get(self._key(like))
            if bucket:
                tensor = bucket.pop()
                if not bucket:
                    self.free.pop(self._key(like), None)
                self.bytes -= tensor.numel() * tensor.element_size()
                return tensor
        return torch.empty_like(like, device="cpu", pin_memory=True) if allocate else None

    def release(self, lease):
        with self.lock:
            self.pending.append(lease)
        self.collect()

    def close(self):
        with self.lock:
            self.closed = True
            pending, self.pending = self.pending, []
            self.free.clear()
            self.bytes = 0
        for lease in pending:
            for event in lease.events:
                event.synchronize()


class CacheTransferManager:
    def __init__(self, device, *, enabled=True, pinned_pool_bytes=128 * 1024 ** 2,
                 mode="protect_forward", prefetch_bytes=64 * 1024 ** 2,
                 chunk_bytes=4 * 1024 ** 2, headroom_bytes=512 * 1024 ** 2):
        if mode not in {"protect_forward", "overlap", "cpu_prepare"}:
            raise ValueError("unknown KV prefetch mode")
        if min(prefetch_bytes, headroom_bytes) < 0 or chunk_bytes <= 0:
            raise ValueError("invalid KV transfer budget")
        self.device = torch.device(device)
        self.enabled = bool(enabled and self.device.type == "cuda")
        self.mode = mode
        self.prefetch_limit, self.chunk_bytes = int(prefetch_bytes), int(chunk_bytes)
        self.headroom_bytes = int(headroom_bytes)
        self.pool = PinnedBufferPool(pinned_pool_bytes)
        self.lock = threading.RLock()  # metadata only, never allocation/copy/wait
        self.gate = threading.Condition(threading.RLock())
        self.admitted = self.executing = self.submitting = 0
        self.last_compute = None
        self.pending = []
        self.speculative_bytes = 0
        self.metrics = defaultdict(float)
        self.h2d_stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        self.prefetch_stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        self.d2h_stream = torch.cuda.Stream(device=self.device) if self.enabled else None

    @staticmethod
    def mark_computed(cache, *, producer_stream):
        legacy = cache.to_legacy_cache()
        if legacy and legacy[0][0].is_cuda:
            if producer_stream is None:
                raise ValueError("CUDA KV requires its actual producer stream")
            event = torch.cuda.Event()
            event.record(producer_stream)
            cache._fastsd_ready_event = event
            return event
        return None

    @staticmethod
    def wait_ready(cache, device, *, compute_stream=None):
        event = getattr(cache, "_fastsd_ready_event", None)
        if event is None:
            return
        if torch.device(device).type == "cuda":
            if compute_stream is None:
                raise ValueError("consumer must specify compute_stream")
            compute_stream.wait_event(event)
            for layer in cache.to_legacy_cache():
                for tensor in layer:
                    tensor.record_stream(compute_stream)
        else:
            event.synchronize()

    def admit(self):
        with self.gate:
            self.admitted += 1

    def release_admission(self):
        with self.gate:
            self.admitted = max(0, self.admitted - 1)
            self.gate.notify_all()

    def _gpu_idle(self):
        return self.last_compute is None or self.last_compute.query()

    @contextmanager
    def _speculation_slot(self, *, preparing=False):
        with self.gate:
            allowed = (self.enabled and self.mode != "cpu_prepare"
                       and not self.submitting and not (self.admitted and not self.executing)
                       and ((self.mode == "overlap" and not preparing)
                            or (not self.admitted and not self.executing and self._gpu_idle())))
            if allowed:
                self.submitting += 1
        try:
            yield allowed
        finally:
            if allowed:
                with self.gate:
                    self.submitting -= 1
                    self.gate.notify_all()

    def begin_execution(self):
        start = time.perf_counter()
        with self.gate:
            self.executing += 1
            while self.submitting:
                self.gate.wait()
        if self.mode == "protect_forward":
            with self.lock:
                events = [t.last_copy for t in self.pending
                          if t.purpose == "prefetch" and t.last_copy is not None]
            for event in events:
                event.synchronize()
        delay = (time.perf_counter() - start) * 1000.0
        self.metrics["prefetch_drain_ms"] += delay
        return delay

    def end_execution(self, *, compute_stream):
        event = None
        if compute_stream is not None:
            event = torch.cuda.Event()
            event.record(compute_stream)
        with self.gate:
            self.last_compute = event
            self.executing -= 1
            self.gate.notify_all()

    def _reserve_speculation(self, size):
        # No reliable headroom => skip; allocations admitted only while idle.
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
        except (RuntimeError, ValueError):
            return False
        with self.lock:
            if (size + self.speculative_bytes > self.prefetch_limit
                    or size + self.headroom_bytes > free):
                self.metrics["prefetch_budget_skips"] += 1
                return False
            self.speculative_bytes += size
            return True

    def submit_h2d(self, snapshot, *, purpose="demand"):
        if not isinstance(snapshot, CacheSnapshot):
            snapshot = CacheSnapshot.capture(snapshot)
        if purpose == "prefetch":
            with self._speculation_slot(preparing=True) as allowed:
                return self._prepare(snapshot, self.device, purpose) if allowed else None
        ticket = self._prepare(snapshot, self.device, purpose)
        self._advance(ticket, all_chunks=True)
        return ticket

    def submit_d2h(self, snapshot, *, purpose="offload"):
        if not isinstance(snapshot, CacheSnapshot):
            snapshot = CacheSnapshot.capture(snapshot)
        ticket = self._prepare(snapshot, torch.device("cpu"), purpose)
        self._advance(ticket, all_chunks=True)
        return ticket

    def _prepare(self, snapshot, device, purpose):
        cache, legacy = snapshot.cache, snapshot.cache.to_legacy_cache()
        size = sum(t.numel() * t.element_size() for layer in legacy for t in layer)
        ticket = KVTransferTicket(snapshot, purpose, source_lease=cache,
                                  bytes_reserved=size,
                                  producer_ready=getattr(cache, "_fastsd_ready_event", None),
                                  direction="h2d" if device.type == "cuda" else "d2h")
        if not legacy or legacy[0][0].device == device:
            ticket.destination_cache = cache
            ticket.copy_done = ticket.producer_ready
            ticket.copy_state = "copying" if ticket.copy_done is not None else "ready"
            ticket.bytes_reserved = 0
            return ticket
        speculative = purpose == "prefetch"
        if speculative and not self._reserve_speculation(size):
            return None
        pinned = []
        try:
            if not self.enabled:
                if ticket.producer_ready is not None:
                    ticket.producer_ready.synchronize()
                copied = [(k.to(device), v.to(device)) for k, v in legacy]
                ticket.destination_cache = dynamic_cache_from_views(copied)
                ticket.copy_state = "ready"
            else:
                if legacy[0][0].is_cuda and ticket.producer_ready is None:
                    raise ValueError("GPU KV must be marked by its producer before transfer")
                copied = []
                for layer in legacy:
                    pair = []
                    for source in layer:
                        if device.type == "cpu":
                            target = self.pool.acquire(source)
                            pinned.append(target)
                        else:
                            if not source.is_pinned():
                                if ticket.producer_ready is not None:
                                    ticket.producer_ready.synchronize()
                                staged = self.pool.acquire(source)
                                staged.copy_(source)
                                pinned.append(staged)
                                source = staged
                            target = torch.empty_like(source, device=device)
                        pair.append(target)
                        # Valid views only, never unused preallocated capacity.
                        per_token = max(1, source.numel() // max(1, source.shape[-2]) * source.element_size())
                        step = max(1, self.chunk_bytes // per_token)
                        for offset in range(0, source.shape[-2], step):
                            ticket.chunks.append((source[..., offset:offset + step, :],
                                                  target[..., offset:offset + step, :]))
                    copied.append(tuple(pair))
                ticket.destination_cache = dynamic_cache_from_views(copied)
            ticket.destination_cache._fastsd_valid_length = snapshot.valid_len
            if pinned:
                lease = _Lease(pinned)
                ticket._pinned_lease = lease
                if device.type == "cpu":
                    ticket.destination_cache._fastsd_pinned_lease = lease
                    weakref.finalize(ticket.destination_cache, self.pool.release, lease)
            with self.lock:
                self.pending.append(ticket)
                self.metrics["copies"] += 1
            return ticket
        except BaseException:
            if speculative:
                with self.lock:
                    self.speculative_bytes -= size
            if pinned:
                self.pool.release(_Lease(pinned))
            raise

    def _advance(self, ticket, *, all_chunks=False):
        if ticket is None:
            return
        # Only the ticket's consumer contends here, never unrelated forward A.
        with ticket.lock:
            if ticket.retired or ticket.copy_state in {"ready", "invalid-draining"}:
                return
            if ticket.next_chunk == len(ticket.chunks):
                return
            stream = (self.d2h_stream if ticket.direction == "d2h" else
                      self.h2d_stream if ticket.claimed_by is not None or ticket.purpose != "prefetch"
                      else self.prefetch_stream)
            with torch.cuda.stream(stream):
                dependency = ticket.last_copy or ticket.producer_ready
                if dependency is not None:
                    stream.wait_event(dependency)
                limit = len(ticket.chunks) if all_chunks else ticket.next_chunk + 1
                for index in range(ticket.next_chunk, limit):
                    source, target = ticket.chunks[index]
                    target.copy_(source, non_blocking=True)
                    if source.is_cuda:
                        source.record_stream(stream)
                    if target.is_cuda:
                        target.record_stream(stream)
                    size = source.numel() * source.element_size()
                    ticket.submitted_bytes += size
                    self.metrics[ticket.direction + "_bytes"] += size
                ticket.next_chunk = limit
                event = torch.cuda.Event()
                event.record(stream)
                ticket.last_copy = event
                ticket.copy_state = "copying"
                if limit == len(ticket.chunks):
                    ticket.copy_done = event  # recorded; covers ALL writes
                    ticket.destination_cache._fastsd_ready_event = event
                for lease in (getattr(ticket, "_pinned_lease", None),
                              getattr(ticket.source_lease, "_fastsd_pinned_lease", None)):
                    if lease is not None:
                        lease.events = [e for e in lease.events if not e.query()]
                        lease.events.append(event)

    def advance_prefetch(self, ticket):
        with self._speculation_slot() as allowed:
            if allowed and ticket.claimed_by is None:
                # At most one uncompleted speculative chunk across all tickets.
                with self.lock:
                    in_flight = any(t.purpose == "prefetch" and t.last_copy is not None
                                    and not t.last_copy.query() for t in self.pending)
                if not in_flight:
                    self._advance(ticket)

    def bind_to_consumer(self, ticket, *, compute_stream):
        self._advance(ticket, all_chunks=True)
        if ticket.retired:
            raise RuntimeError("cannot consume a retired transfer")
        lease = CacheLease(ticket, ticket.destination_cache)
        if self.device.type == "cuda" and compute_stream is not None and ticket.direction == "h2d":
            if ticket.copy_done is not None:
                lease.wait_start = torch.cuda.Event(enable_timing=True)
                lease.wait_end = torch.cuda.Event(enable_timing=True)
                lease.wait_start.record(compute_stream)
                compute_stream.wait_event(ticket.copy_done)
                lease.wait_end.record(compute_stream)
            for layer in lease.cache.to_legacy_cache():
                for tensor in layer:
                    tensor.record_stream(compute_stream)
        elif ticket.copy_done is not None:
            ticket.copy_done.synchronize()
        return lease

    def release_consumer(self, lease, *, compute_stream):
        if compute_stream is not None:
            event = torch.cuda.Event()
            event.record(compute_stream)
            lease.ticket.consumer_done = event
        self.retire(lease.ticket)

    def protect_source_write(self, cache, *, producer_stream):
        # Must be called before overwriting a leased valid prefix.
        with self.lock:
            readers = [t for t in self.pending if t.source_identity == id(cache)]
        for ticket in readers:
            if ticket.last_copy is not None:
                if producer_stream is not None:
                    producer_stream.wait_event(ticket.last_copy)
                else:
                    ticket.last_copy.synchronize()

    def retire(self, ticket):
        with ticket.lock:
            if ticket.retired:
                return
            ticket.retired = True
            ticket.copy_state = "invalid-draining"
            if ticket.purpose == "prefetch" and ticket.claimed_by is None:
                self.metrics["prefetch_wasted_bytes"] += ticket.submitted_bytes

    def poll_completed(self):
        with self.lock:
            tickets = list(self.pending)
        completed, remove = [], []
        for ticket in tickets:
            with ticket.lock:
                if ticket.ready() and not ticket.retired:
                    ticket.copy_state = "ready"
                    if not ticket.reported:
                        completed.append(ticket)
                        ticket.reported = True
                drained = ticket.last_copy is None or ticket.last_copy.query()
                used = ticket.consumer_done is None or ticket.consumer_done.query()
                if drained and used and (ticket.retired or ticket.purpose != "prefetch"):
                    remove.append(ticket)
                    if ticket.direction == "h2d" and getattr(ticket, "_pinned_lease", None) is not None:
                        self.pool.release(ticket._pinned_lease)
                        ticket._pinned_lease = None
                    ticket.chunks.clear()
                    ticket.source_lease = None
        with self.lock:
            for ticket in remove:
                if ticket in self.pending:
                    self.pending.remove(ticket)
                    if ticket.purpose == "prefetch" and not ticket.budget_released:
                        self.speculative_bytes -= ticket.bytes_reserved
                        ticket.budget_released = True
            self.metrics["prefetch_inflight_bytes"] = sum(
                t.bytes_reserved for t in self.pending
                if t.purpose == "prefetch" and not t.ready())
            self.metrics["prefetch_reserved_bytes"] = self.speculative_bytes
        self.pool.collect()
        return completed

    collect = poll_completed

    def move(self, cache, device, *, compute_stream=None):
        """Compatibility demand API; background code uses submit_* only."""
        device = torch.device(device)
        if device.type == "cpu":
            return self.submit_d2h(cache).destination_cache
        ticket = self.submit_h2d(cache)
        if compute_stream is None and device.type == "cuda":
            compute_stream = torch.cuda.current_stream(device)
        return self.bind_to_consumer(ticket, compute_stream=compute_stream).cache

    def close(self):
        with self.lock:
            tickets = list(self.pending)
        for ticket in tickets:
            self.retire(ticket)
            for event in (ticket.last_copy, ticket.consumer_done):
                if event is not None:
                    event.synchronize()
        self.poll_completed()
        self.pool.close()
