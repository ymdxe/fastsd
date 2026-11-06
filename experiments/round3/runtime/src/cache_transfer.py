"""Event-fenced KV transport. Submission never touches the compute stream."""
from collections import defaultdict
from contextlib import contextmanager
import threading
import time
import json
import math
from diagnostic_trace import trace as DIAG, request_meta as diag_req, item_meta as diag_item, queue_rows as diag_queues, backlog as diag_backlog
import weakref
import torch
from .cache_views import dynamic_cache_from_views
from .cache_protocol import CacheSnapshot, CacheLease, KVTransferTicket


def _shape_key(shape):
    """A stable key usable by runtime contexts and saved replay evidence."""
    if isinstance(shape, str):
        return shape
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


def _percentile(values, quantile):
    values = sorted(float(value) for value in values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


class OverlapController:
    """Only replay-calibrated contexts receive speculative compute-window DMA.

    Unknown shapes pause. Explicit probes are bounded and recorded separately;
    normal runtime observations can revoke, but never invent, calibration.
    """

    def __init__(self, group_bytes, rate_bytes_per_second, idle_rate_bytes_per_second,
                 *, clock=time.perf_counter):
        self.group_bytes = int(group_bytes)
        self.rate = float(rate_bytes_per_second)
        self.idle_rate = float(idle_rate_bytes_per_second)
        self.clock = clock
        self.profiles = {}
        self.context = None
        self.state, self.reason = "paused", "uncalibrated_context"
        self.transitions = []
        self.changed_at = clock()
        self.tokens, self.refilled_at = float(group_bytes), clock()
        self.lock = threading.RLock()

    def _transition(self, state, reason, group_bytes, rate):
        now = self.clock()
        if (state, reason) != (self.state, self.reason):
            self.transitions.append(dict(state=state, reason=reason,
                previous_state=self.state, duration_s=now - self.changed_at,
                group_bytes=int(group_bytes), rate_bytes_per_second=float(rate),
                context=self.context, at=now))
            self.state, self.reason, self.changed_at = state, reason, now

    def context_for(self, shape):
        with self.lock:
            self.context = _shape_key(shape)

    def install_calibration(self, shape, *, baseline_ms, sample_ms, group_bytes,
                            rate_bytes_per_second, evidence):
        baseline, sample = list(baseline_ms), list(sample_ms)
        if (len(baseline) < 5 or len(sample) != len(baseline)
                or not evidence or not all(math.isfinite(float(v)) and float(v) > 0
                                           for v in baseline + sample)):
            raise ValueError("calibration requires at least five positive paired samples and evidence")
        if not 0 < int(group_bytes) <= self.group_bytes or not 0 < float(rate_bytes_per_second) <= self.rate:
            raise ValueError("calibration exceeds configured speculative limits")
        p50 = _percentile(sample, .5) / _percentile(baseline, .5) - 1
        p95 = _percentile(sample, .95) / _percentile(baseline, .95) - 1
        passed = p50 <= .01 and p95 <= .02
        with self.lock:
            self.profiles[_shape_key(shape)] = dict(
                state="overlap" if passed else "paused",
                reason="paired_replay_pass" if passed else "paired_replay_failed",
                group_bytes=int(group_bytes), rate_bytes_per_second=float(rate_bytes_per_second),
                p50_increase=p50, p95_increase=p95, baseline_ms=_percentile(baseline, .5),
                evidence=evidence, sample_count=len(baseline), violations=0)
        return passed

    def install_probe(self, shape, *, group_bytes=256 * 1024,
                      rate_bytes_per_second=4 * 1024 ** 2, evidence):
        if not evidence or not 0 < group_bytes <= self.group_bytes or not 0 < rate_bytes_per_second <= self.rate:
            raise ValueError("probe requires evidence and limits within the configured budget")
        with self.lock:
            self.profiles[_shape_key(shape)] = dict(state="probe", reason="explicit_calibration_probe",
                group_bytes=int(group_bytes), rate_bytes_per_second=float(rate_bytes_per_second),
                evidence=evidence, violations=0)

    def allowance(self, *, computing, blocked_reason=None):
        with self.lock:
            profile = self.profiles.get(self.context)
            if blocked_reason:
                state, reason, group, rate = "paused", blocked_reason, 0, 0
            elif not computing:
                state, reason, group, rate = "overlap", "gpu_idle", self.group_bytes, self.idle_rate
            elif profile is None:
                state, reason, group, rate = "paused", "uncalibrated_context", 0, 0
            else:
                state, reason = profile["state"], profile["reason"]
                group, rate = profile["group_bytes"], profile["rate_bytes_per_second"]
                if state == "paused":
                    group, rate = 0, 0
            self._transition(state, reason, group, rate)
            now = self.clock()
            self.tokens = min(float(group), self.tokens + max(0, now - self.refilled_at) * rate)
            self.refilled_at = now
            return min(int(self.tokens), int(group))

    def consume(self, size):
        with self.lock:
            self.tokens = max(0, self.tokens - size)

    def observe(self, shape, milliseconds, *, speculative):
        with self.lock:
            profile = self.profiles.get(_shape_key(shape))
            if not speculative or profile is None or profile.get("baseline_ms") is None:
                return
            ratio = milliseconds / profile["baseline_ms"] - 1
            profile["violations"] = profile.get("violations", 0) + 1 if ratio > .02 else 0
            if profile["violations"] >= 6:
                profile.update(state="paused", reason="runtime_disturbance_limit")
            elif profile["violations"] >= 3:
                profile.update(state="throttled", reason="runtime_disturbance_throttle")
                profile["group_bytes"] = max(1, profile["group_bytes"] // 2)
                profile["rate_bytes_per_second"] /= 2

    def status(self):
        with self.lock:
            return dict(state=self.state, reason=self.reason, context=self.context,
                        duration_s=self.clock() - self.changed_at,
                        transitions=list(self.transitions), profiles=dict(self.profiles))


class _StorageLease:
    """A slab interval survives DMA, consumers, and all registered cache owners."""

    def __init__(self, arena, offset, size):
        self.arena, self.offset, self.size = arena, offset, size
        self.events, self.owners = [], set()
        self.retired = False
        self.on_recycle = None

    def reusable(self):
        return self.retired and not self.owners and all(event.query() for event in self.events)


class _SlabArena:
    def __init__(self, tensor):
        self.tensor = tensor
        self.capacity = tensor.numel()
        self.free = [(0, self.capacity)]
        self.leases = []
        self.lock = threading.RLock()

    def collect(self):
        with self.lock:
            for lease in list(self.leases):
                if lease.reusable():
                    self.leases.remove(lease)
                    self.free.append((lease.offset, lease.size))
                    if lease.on_recycle is not None:
                        lease.on_recycle()
                        lease.on_recycle = None
            merged = []
            for offset, size in sorted(self.free):
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    previous, count = merged[-1]
                    merged[-1] = previous, count + size
                else:
                    merged.append((offset, size))
            self.free = merged

    def acquire(self, tensors):
        sizes = [((tensor.numel() * tensor.element_size() + 255) // 256) * 256 for tensor in tensors]
        total = sum(sizes)
        self.collect()
        with self.lock:
            for index, (offset, size) in enumerate(self.free):
                if size < total:
                    continue
                self.free.pop(index)
                if total < size:
                    self.free.insert(index, (offset + total, size - total))
                lease = _StorageLease(self, offset, total)
                self.leases.append(lease)
                views = []
                cursor = offset
                for source, count in zip(tensors, sizes):
                    length = source.numel() * source.element_size()
                    views.append(self.tensor.narrow(0, cursor, length).view(source.dtype).view(source.shape))
                    cursor += count
                return lease, views
        return None, None


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

    def release(self, lease, *, collect=True):
        with self.lock:
            self.pending.append(lease)
        if collect:
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
                 mode="overlap", prefetch_bytes=64 * 1024 ** 2,
                 chunk_bytes=4 * 1024 ** 2, headroom_bytes=512 * 1024 ** 2,
                 disturbance_budget=0.1, adaptive_fallback=False, record_timeline=False,
                 rate_bytes_per_second=64 * 1024 ** 2,
                 idle_rate_bytes_per_second=512 * 1024 ** 2,
                 event_pool_size=256, warmup_pool=True):
        if mode not in {"protect_forward", "overlap", "cpu_prepare"}:
            raise ValueError("unknown KV prefetch mode")
        if (min(prefetch_bytes, headroom_bytes) < 0 or chunk_bytes <= 0
                or min(rate_bytes_per_second, idle_rate_bytes_per_second) <= 0
                or not all(math.isfinite(float(rate)) for rate in (
                    rate_bytes_per_second, idle_rate_bytes_per_second))
                or event_pool_size < 2):
            raise ValueError("invalid KV transfer budget")
        self.device = torch.device(device)
        self.enabled = bool(enabled and self.device.type == "cuda")
        self.mode = self.requested_mode = mode
        self.prefetch_limit, self.chunk_bytes = int(prefetch_bytes), int(chunk_bytes)
        self.headroom_bytes = int(headroom_bytes)
        self.pool = PinnedBufferPool(pinned_pool_bytes)
        self.lock = threading.RLock()  # metadata only, never allocation/copy/wait
        self.collect_lock = threading.RLock()
        self.gate = threading.Condition(threading.RLock())
        self.admitted = self.executing = self.submitting = 0
        self.demand_submitting = 0
        self.offload_submitting = 0
        self.last_compute = None
        self.pending = []
        self.speculative_bytes = 0
        self.metrics = defaultdict(float)
        self.forward_baselines = {}
        self.controller = OverlapController(chunk_bytes, rate_bytes_per_second,
                                            idle_rate_bytes_per_second)
        self.gpu_arena = None
        self.event_pool_size = int(event_pool_size)
        self._events = []
        self._cache_owners = {}
        self.calibration_context = None
        self.timeline = [] if record_timeline or DIAG.enabled else None
        self.h2d_stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        self.prefetch_stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        self.d2h_stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        if DIAG.enabled and self.enabled: DIAG.gpu_init(torch, self.device)
        if self.enabled and warmup_pool:
            self.warmup_prefetch_pool()

    def warmup_prefetch_pool(self):
        """Controlled maintenance only: no allocation or driver query in forward."""
        if not self.enabled or self.gpu_arena is not None:
            return
        with self.gate:
            if self.executing or self.admitted or self.submitting:
                raise RuntimeError("prefetch pool warmup requires an idle executor")
            free, _ = torch.cuda.mem_get_info(self.device)
            capacity = min(self.prefetch_limit, max(0, free - self.headroom_bytes))
            if capacity <= 0:
                self.metrics["prefetch_pool_unavailable"] += 1
                return
            with torch.cuda.stream(self.prefetch_stream):
                gpu = torch.empty(capacity, dtype=torch.uint8, device=self.device)
                events = [torch.cuda.Event(enable_timing=self.timeline is not None)
                          for _ in range(self.event_pool_size)]
                for event in events:
                    event.record(self.prefetch_stream)
            self.prefetch_stream.synchronize()
            self.gpu_arena = _SlabArena(gpu)
            self._events = events
            self.metrics["prefetch_pool_gpu_bytes"] = capacity

    def set_forward_context(self, shape):
        self.controller.context_for(shape)

    def install_calibration(self, shape, **kwargs):
        return self.controller.install_calibration(shape, **kwargs)

    def install_probe(self, shape, **kwargs):
        return self.controller.install_probe(shape, **kwargs)

    def disable_shape(self, shape, *, reason):
        with self.controller.lock:
            key = _shape_key(shape)
            profile = self.controller.profiles.setdefault(key, dict(
                group_bytes=self.chunk_bytes, rate_bytes_per_second=self.controller.rate))
            profile.update(state="paused", reason=str(reason))

    def load_calibration(self, path, *, context=None):
        """Load replay quotas only for the exact hardware and forward context.

        A shape alone does not identify its model or GPU. Require the caller's
        live context, rather than trusting an artifact to declare its own match.
        Validate the entire file before publishing any newly accepted profiles.
        """
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        required = ("gpu_uuid", "model", "dtype", "attention_backend",
                    "forward_primitive", "kv_batch_mode")
        saved_context = records.get("context") if isinstance(records, dict) else None
        for label, values in (("runtime", context), ("saved", saved_context)):
            if not isinstance(values, dict) or any(
                    key not in values or not str(values[key]).strip() or values[key] is None
                    for key in required):
                raise ValueError(f"{label} calibration context requires " + ", ".join(required))
        mismatch = [key for key in required if (
            str(context[key]).strip().casefold() != str(saved_context[key]).strip().casefold()
            if key == "gpu_uuid" else str(context[key]) != str(saved_context[key]))]
        if mismatch:
            raise ValueError("calibration context mismatch: " + ", ".join(mismatch))
        profiles = records.get("profiles", [])
        if not isinstance(profiles, list):
            raise ValueError("calibration profiles must be a list")
        staged = OverlapController(self.chunk_bytes, self.controller.rate,
                                   self.controller.idle_rate)
        for record in profiles:
            record = dict(record)
            shape = record.pop("shape")
            staged.install_calibration(shape, **record)
        with self.controller.lock:
            self.controller.profiles = dict(staged.profiles)
            self.calibration_context = {key: str(context[key]) for key in required}
        return len(staged.profiles)

    def controller_status(self):
        return dict(requested_mode=self.requested_mode,
                    calibration_context=self.calibration_context, **self.controller.status())

    @staticmethod
    def _storage_ranges(cache):
        ranges = []
        for layer in cache.to_legacy_cache():
            for tensor in layer:
                if tensor.numel():
                    # Account for non-contiguous views sharing slab storage.
                    start = tensor.data_ptr()
                    span = (1 + sum((size - 1) * abs(stride)
                                    for size, stride in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
                    ranges.append((start, start + span))
        return ranges

    def retain_cache(self, cache, *, owner):
        """Register ownership by storage ranges, including aliased cache wrappers."""
        ranges = self._storage_ranges(cache)
        with self.lock:
            previous = self._cache_owners.pop(owner, set())
            current = set()
            if self.gpu_arena is not None:
                base = self.gpu_arena.tensor.data_ptr()
                for lease in self.gpu_arena.leases:
                    low, high = base + lease.offset, base + lease.offset + lease.size
                    if any(start < high and end > low for start, end in ranges):
                        current.add(lease)
            for lease in previous - current:
                lease.owners.discard(owner)
            for lease in current:
                lease.owners.add(owner)
            if current:
                self._cache_owners[owner] = current

    def release_cache(self, *, owner):
        with self.lock:
            for lease in self._cache_owners.pop(owner, ()):
                lease.owners.discard(owner)

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
            # The previous phase's profile cannot authorize copies in the gap
            # before this newly admitted phase installs its own context.
            self.controller.context_for(None)

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
                       and not self.submitting
                       and ((self.mode == "overlap")
                            or (not self.admitted and not self.executing and self._gpu_idle())))
            if allowed:
                self.submitting += 1
        try:
            DIAG.emit("prefetch_gate", preparing=preparing, allowed=bool(allowed), mode=self.mode, admitted=self.admitted, executing=self.executing)
            yield allowed
        finally:
            if allowed:
                with self.gate:
                    self.submitting -= 1
                    self.gate.notify_all()

    def begin_execution(self, shape=None):
        start = time.perf_counter()
        if shape is not None:
            self.set_forward_context(shape)
        with self.gate:
            self.executing += 1
        # A forward only waits for its own demand tickets in bind_to_consumer.
        # Already-issued speculative DMA is never drained on an unrelated PID.
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
        self.poll_completed()

    def _reserve_speculation(self, size):
        # Reject an impossible reservation without entering the CUDA driver.
        # During a forward, the worker may retry several predicted devices;
        # a full local budget cannot be relieved by querying device memory.
        with self.lock:
            if size + self.speculative_bytes > self.prefetch_limit:
                self.metrics["prefetch_budget_skips"] += 1
                self.metrics["prefetch_local_budget_skips"] += 1
                return False
            if self.gpu_arena is not None:
                # Slabs were allocated after checking headroom during warmup.
                self.speculative_bytes += size
                return True
            self.metrics["prefetch_pool_unavailable"] += 1
            return False

    def submit_h2d(self, snapshot, *, purpose="demand"):
        started = time.perf_counter()
        if not isinstance(snapshot, CacheSnapshot):
            snapshot = CacheSnapshot.capture(snapshot)
        if purpose == "prefetch":
            with self._speculation_slot(preparing=True) as allowed:
                return self._prepare(snapshot, self.device, purpose) if allowed else None
        with self.gate:
            self.demand_submitting += 1
        try:
            ticket = self._prepare(snapshot, self.device, purpose)
            ticket.demand_started_at = started
            self._advance(ticket, all_chunks=True)
            return ticket
        finally:
            with self.gate:
                self.demand_submitting -= 1

    def submit_d2h(self, snapshot, *, purpose="offload"):
        if not isinstance(snapshot, CacheSnapshot):
            snapshot = CacheSnapshot.capture(snapshot)
        with self.gate:
            self.offload_submitting += 1
        try:
            ticket = self._prepare(snapshot, torch.device("cpu"), purpose)
            self._advance(ticket, all_chunks=True)
            return ticket
        finally:
            with self.gate:
                self.offload_submitting -= 1

    def _prepare(self, snapshot, device, purpose):
        cache, legacy = snapshot.cache, snapshot.cache.to_legacy_cache()
        size = sum(t.numel() * t.element_size() for layer in legacy for t in layer)
        ticket = KVTransferTicket(snapshot, purpose, source_lease=cache,
                                  bytes_reserved=size,
                                  producer_ready=getattr(cache, "_fastsd_ready_event", None),
                                  direction="h2d" if device.type == "cuda" else "d2h")
        ticket._source_ranges = self._storage_ranges(cache)
        ticket._source_storage_leases = []
        with self.lock:
            if self.gpu_arena is not None:
                base = self.gpu_arena.tensor.data_ptr()
                ticket._source_storage_leases = [lease for lease in self.gpu_arena.leases
                    if any(start < base + lease.offset + lease.size and end > base + lease.offset
                           for start, end in ticket._source_ranges)]
        if not legacy or legacy[0][0].device == device:
            ticket.destination_cache = cache
            ticket.copy_done = ticket.producer_ready
            ticket.copy_state = "copying" if ticket.copy_done is not None else "ready"
            ticket.bytes_reserved = 0
            return ticket
        speculative = purpose == "prefetch"
        sources = [source for layer in legacy for source in layer]
        if speculative and any(not source.is_pinned() for source in sources):
            # A check for GPU idleness cannot serialize a subsequent CPU copy
            # with admission. Pageable sources therefore use normal demand;
            # offloaded production KV is already pinned for asynchronous H2D.
            self.metrics["prefetch_deferred_cpu_staging"] += 1
            return None
        if speculative and not self._reserve_speculation(size):
            return None
        pinned = []
        storage_lease = None
        try:
            with self.lock:
                for source_storage in ticket._source_storage_leases:
                    source_storage.owners.add(("dma-source", id(ticket)))
            if not self.enabled:
                if ticket.producer_ready is not None:
                    ticket.producer_ready.synchronize()
                copied = [(k.to(device), v.to(device)) for k, v in legacy]
                ticket.destination_cache = dynamic_cache_from_views(copied)
                ticket.copy_state = "ready"
            else:
                if legacy[0][0].is_cuda and ticket.producer_ready is None:
                    raise ValueError("GPU KV must be marked by its producer before transfer")
                gpu_targets = None
                if speculative:
                    # Pinned sources only: describe existing host data and
                    # destination slab views without staging or allocation.
                    with self.lock:
                        storage_lease, gpu_targets = self.gpu_arena.acquire(sources)
                        if storage_lease is None:
                            self.metrics["prefetch_pool_capacity_skips"] += 1
                            self.speculative_bytes -= size
                            return None
                    ticket._storage_lease = storage_lease
                    ticket._pooled_events = []
                copied = []
                tensor_index = 0
                for layer in legacy:
                    pair = []
                    for source in layer:
                        if device.type == "cpu":
                            target = self.pool.acquire(source)
                            pinned.append(target)
                        else:
                            if speculative:
                                target = gpu_targets[tensor_index]
                            elif not source.is_pinned():
                                if ticket.producer_ready is not None:
                                    ticket.producer_ready.synchronize()
                                staged = self.pool.acquire(source)
                                staged.copy_(source)
                                pinned.append(staged)
                                source = staged
                            if not speculative:
                                with torch.cuda.stream(self.h2d_stream):
                                    target = torch.empty_like(source, device=device)
                        pair.append(target)
                        tensor_index += 1
                        # Valid views only, never unused preallocated capacity.
                        views = [(source, target)]
                        while views:
                            src, dst = views.pop()
                            split_bytes = min(self.chunk_bytes, 256 * 1024) if speculative else self.chunk_bytes
                            if src.numel() * src.element_size() <= split_bytes:
                                ticket.chunks.append((src, dst))
                                continue
                            dim = max(range(src.ndim), key=lambda d: src.shape[d])
                            if src.shape[dim] <= 1:
                                raise ValueError("chunk budget must fit at least one KV element")
                            half = src.shape[dim] // 2
                            views.append((src.narrow(dim, half, src.shape[dim] - half),
                                          dst.narrow(dim, half, dst.shape[dim] - half)))
                            views.append((src.narrow(dim, 0, half), dst.narrow(dim, 0, half)))
                    copied.append(tuple(pair))
                ticket.destination_cache = dynamic_cache_from_views(copied)
            ticket.destination_cache._fastsd_valid_length = snapshot.valid_len
            if pinned:
                lease = _Lease(pinned)
                ticket._pinned_lease = lease
                if device.type == "cpu":
                    ticket.destination_cache._fastsd_pinned_lease = lease
                    # A CPU-cache wrapper may die when a prefetch worker drops
                    # its last reference during forward. Queue reclamation only;
                    # pinned memory collection happens during idle maintenance.
                    weakref.finalize(ticket.destination_cache, self.pool.release, lease, collect=False)
            with self.lock:
                self.pending.append(ticket)
                self.metrics["copies"] += 1
            DIAG.emit("kv_ticket_prepared", ticket_id=id(ticket), purpose=purpose, direction=ticket.direction, source_identity=snapshot.source_identity, session_epoch=snapshot.session_epoch, content_version=snapshot.content_version, valid_len=snapshot.valid_len, bytes=size)
            return ticket
        except BaseException:
            if speculative:
                with self.lock:
                    self.speculative_bytes -= size
            if storage_lease is not None:
                storage_lease.retired = True
            with self.lock:
                for source_storage in ticket._source_storage_leases:
                    source_storage.owners.discard(("dma-source", id(ticket)))
            if pinned:
                self.pool.release(_Lease(pinned))
            raise

    def _advance(self, ticket, *, all_chunks=False, byte_limit=None):
        if ticket is None:
            return
        # Only the ticket's consumer contends here, never unrelated forward A.
        with ticket.lock:
            if ticket.retired or ticket.copy_state in {"ready", "invalid-draining"}:
                return
            if ticket.next_chunk == len(ticket.chunks):
                return
            if not all_chunks and byte_limit is not None:
                first = ticket.chunks[ticket.next_chunk][0]
                if first.numel() * first.element_size() > byte_limit:
                    self.metrics["prefetch_rate_limit_skips"] += 1
                    return
            stream = (self.d2h_stream if ticket.direction == "d2h" else
                      self.h2d_stream if ticket.claimed_by is not None or ticket.purpose != "prefetch"
                      else self.prefetch_stream)
            speculative = ticket.purpose == "prefetch" and ticket.claimed_by is None and not all_chunks
            event = trace_start = None
            if speculative:
                # Completed intermediate groups can reuse their descriptors;
                # final copy events remain ticket-owned through consumption.
                previous_events = getattr(ticket, "_pooled_events", [])
                if (ticket.copy_done is None and ticket.last_copy is not None
                        and ticket.last_copy.query() and (self.timeline is None or DIAG.enabled)):
                    if DIAG.enabled:
                        DIAG.gpu_collect()
                    lease = ticket._storage_lease
                    lease.events[:] = [old for old in lease.events if not old.query()]
                    with self.lock:
                        self._events.extend(previous_events)
                        ticket._pooled_events = []
                needed = 2 if self.timeline is not None else 1
                with self.lock:
                    if len(self._events) < needed:
                        self.metrics["prefetch_event_pool_skips"] += 1
                        return
                    event = self._events.pop()
                    if needed == 2:
                        trace_start = self._events.pop()
                    ticket._pooled_events.extend(e for e in (event, trace_start) if e is not None)
            with torch.cuda.stream(stream):
                dependency = ticket.last_copy or ticket.producer_ready
                if dependency is not None:
                    stream.wait_event(dependency)
                if self.timeline is not None:
                    if trace_start is None:
                        trace_start = torch.cuda.Event(enable_timing=True)
                    trace_start.record(stream)
                limit = len(ticket.chunks)
                if not all_chunks:
                    # A chunk is a BYTE budget, not one tensor per polling
                    # interval. Small per-layer K/V tensors share a submission.
                    limit, total = ticket.next_chunk, 0
                    while limit < len(ticket.chunks):
                        source = ticket.chunks[limit][0]
                        size = source.numel() * source.element_size()
                        if total + size > (self.chunk_bytes if byte_limit is None else byte_limit):
                            break
                        total += size
                        limit += 1
                ticket.last_chunk_bytes = 0
                for index in range(ticket.next_chunk, limit):
                    source, target = ticket.chunks[index]
                    target.copy_(source, non_blocking=True)
                    if source.is_cuda:
                        source.record_stream(stream)
                    if target.is_cuda:
                        target.record_stream(stream)
                    size = source.numel() * source.element_size()
                    ticket.submitted_bytes += size
                    ticket.last_chunk_bytes += size
                    self.metrics[ticket.direction + "_bytes"] += size
                    if ticket.purpose == "prefetch" and ticket.claimed_by is None:
                        self.metrics["speculative_submitted_bytes"] += size
                ticket.next_chunk = limit
                if event is None:
                    event = torch.cuda.Event(enable_timing=self.timeline is not None)
                event.record(stream)
                if self.timeline is not None:
                    if DIAG.enabled:
                        DIAG.gpu_span("kv_copy", trace_start, event, direction=ticket.direction, purpose=ticket.purpose if ticket.claimed_by is None else "demand", ticket_id=id(ticket), bytes=ticket.last_chunk_bytes, source_identity=ticket.source_identity, session_epoch=ticket.session_epoch, valid_len=ticket.valid_len)
                    else:
                        self.timeline.append((trace_start, event, ticket.direction,
                            ticket.purpose if ticket.claimed_by is None else "demand"))
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
                for lease in (getattr(ticket, "_storage_lease", None),
                              *getattr(ticket, "_source_storage_leases", ())):
                    if lease is not None:
                        lease.events[:] = [old for old in lease.events if not old.query()]
                        lease.events.append(event)

    def advance_prefetch(self, ticket):
        with self._speculation_slot() as allowed:
            if allowed and ticket.claimed_by is None:
                # At most one uncompleted speculative chunk across all tickets.
                with self.lock:
                    in_flight = any(t.purpose == "prefetch" and t.last_copy is not None
                                    and not t.last_copy.query() for t in self.pending)
                    demand = self.demand_submitting or any(
                        (t.claimed_by is not None or t.purpose != "prefetch")
                        and t.direction == "h2d" and (
                            (t.last_copy is not None and not t.last_copy.query())
                            or (not t.ready() and not t.retired))
                        for t in self.pending)
                    offload = self.offload_submitting or any(t.direction == "d2h" and (
                        (t.last_copy is not None and not t.last_copy.query())
                        or (not t.ready() and not t.retired))
                                  for t in self.pending)
                reason = "demand_h2d_pending" if demand else "required_d2h_pending" if offload else None
                allowance = self.controller.allowance(computing=bool(self.admitted or self.executing or not self._gpu_idle()),
                                                       blocked_reason=reason)
                status = self.controller.status()
                self.metrics["overlap_controller_state"] = status["state"]
                self.metrics["overlap_controller_reason"] = status["reason"]
                self.metrics["overlap_controller_allowance_bytes"] = allowance
                if reason:
                    self.metrics["prefetch_" + reason + "_skips"] += 1
                if not in_flight and allowance > 0:
                    before = ticket.submitted_bytes
                    self._advance(ticket, byte_limit=allowance)
                    self.controller.consume(ticket.submitted_bytes - before)

    def bind_to_consumer(self, ticket, *, compute_stream):
        started = ticket.demand_started_at or time.perf_counter()
        if ticket.purpose == "prefetch" and ticket.claimed_by is None:
            ticket.claimed_by = "direct-consumer"
            ticket.demand_started_at = started
        self._advance(ticket, all_chunks=True)
        if ticket.retired:
            raise RuntimeError("cannot consume a retired transfer")
        lease = CacheLease(ticket, ticket.destination_cache)
        lease._storage_leases = [storage for storage in (
            getattr(ticket, "_storage_lease", None),
            *getattr(ticket, "_source_storage_leases", ())) if storage is not None]
        with self.lock:
            for storage in lease._storage_leases:
                storage.owners.add(("consumer", id(lease)))
        if ticket.bytes_reserved:
            lease.host_setup_ms = (time.perf_counter() - started) * 1000.0
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
        event = None
        if compute_stream is not None:
            event = torch.cuda.Event()
            event.record(compute_stream)
            lease.ticket.consumer_done = event
        with self.lock:
            for storage in getattr(lease, "_storage_leases", ()):
                if event is not None:
                    storage.events.append(event)
                storage.owners.discard(("consumer", id(lease)))
        self.retire(lease.ticket)

    def speculative_inflight(self):
        with self.lock:
            return any(t.purpose == "prefetch" and t.claimed_by is None
                       and t.last_copy is not None and not t.last_copy.query()
                       for t in self.pending)

    def observe_forward(self, shape, milliseconds, *, speculative):
        if milliseconds <= 0:
            return
        baseline = self.forward_baselines.get(shape)
        if not speculative:
            self.forward_baselines[shape] = milliseconds if baseline is None else 0.8 * baseline + 0.2 * milliseconds
        elif self.mode == "overlap" and baseline is not None:
            ratio = milliseconds / baseline - 1
            self.metrics["forward_disturbance_ratio_max"] = max(self.metrics["forward_disturbance_ratio_max"], ratio)
        self.controller.observe(shape, milliseconds, speculative=speculative)

    def protect_source_write(self, cache, *, producer_stream):
        # Future chunks must not read a prefix after it is overwritten.
        ranges = self._storage_ranges(cache)
        with self.lock:
            readers = [t for t in self.pending if t.source_identity == id(cache) or any(
                low < end and high > start
                for low, high in getattr(t, "_source_ranges", ()) for start, end in ranges)]
        for ticket in readers:
            if ticket.claimed_by is None:
                self.retire(ticket)
            else:
                self._advance(ticket, all_chunks=True)
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
        # Executor maintenance and prefetch progress can collect concurrently.
        # Serialize the whole scan/release, not just the pending-list removal.
        with self.collect_lock:
            return self._poll_completed()

    def _poll_completed(self):
        DIAG.gpu_collect()
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
                if drained and used and (ticket.retired or
                        (ticket.purpose != "prefetch" and ticket.ready())):
                    remove.append(ticket)
                    if ticket.direction == "h2d" and getattr(ticket, "_pinned_lease", None) is not None:
                        self.pool.release(ticket._pinned_lease,
                                          collect=not (self.executing or self.admitted) and self._gpu_idle())
                        ticket._pinned_lease = None
                    ticket.chunks.clear()
                    ticket.source_lease = None
                    # Timeline consumers retain their event values until export.
                    if DIAG.enabled:
                        DIAG.gpu_collect()
                    with self.lock:
                        if self.timeline is None or DIAG.enabled:
                            events = list(getattr(ticket, "_pooled_events", ()))
                            storage = getattr(ticket, "_storage_lease", None)
                            if storage is not None:
                                # Install the callback before exposing retired
                                # state to another thread's arena.acquire().
                                storage.on_recycle = lambda events=events: self._events.extend(events)
                            else:
                                self._events.extend(events)
                        for source_storage in getattr(ticket, "_source_storage_leases", ()):
                            source_storage.owners.discard(("dma-source", id(ticket)))
                        storage = getattr(ticket, "_storage_lease", None)
                        if storage is not None:
                            storage.events[:] = [event for event in storage.events if not event.query()]
                            storage.retired = True
                        ticket._pooled_events = []
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
            self.metrics["prefetch_dma_inflight_bytes"] = sum(
                t.last_chunk_bytes for t in self.pending if t.purpose == "prefetch"
                and t.claimed_by is None and t.last_copy is not None and not t.last_copy.query())
            self.metrics["d2h_dma_inflight_bytes"] = sum(
                t.last_chunk_bytes for t in self.pending if t.direction == "d2h"
                and t.last_copy is not None and not t.last_copy.query())
        if not (self.executing or self.admitted) and self._gpu_idle():
            self.pool.collect()
        with self.lock:
            if self.gpu_arena is not None:
                self.gpu_arena.collect()
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
