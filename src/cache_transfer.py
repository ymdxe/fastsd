"""Event-fenced KV copies with a bounded reusable pinned-memory pool.

The prefetch worker publishes a cache with a ready event. The consumer waits
on that event in its compute stream. Source storage stays owned until DMA has
finished, even if a generation check rejects the copied cache.
"""

from collections import defaultdict
import threading
import weakref

import torch
from transformers.cache_utils import DynamicCache
from .cache_views import dynamic_cache_from_views


class _Lease:
    def __init__(self, tensors):
        self.tensors = tensors
        self.event = None


class PinnedBufferPool:
    def __init__(self, max_bytes=128 * 1024 ** 2):
        self.max_bytes = int(max_bytes)
        self.free = defaultdict(list)
        self.pending = []
        self.bytes = 0
        self.lock = threading.RLock()
        self.closed = False

    def _key(self, tensor):
        return tuple(tensor.shape), tensor.dtype

    def collect(self):
        with self.lock:
            pending, self.pending = self.pending, []
            for lease in pending:
                if lease.event is not None and not lease.event.query():
                    self.pending.append(lease)
                    continue
                for tensor in lease.tensors:
                    size = tensor.numel() * tensor.element_size()
                    if not self.closed and self.bytes + size <= self.max_bytes:
                        self.free[self._key(tensor)].append(tensor)
                        self.bytes += size

    def acquire(self, like):
        with self.lock:
            self.collect()
            key = self._key(like)
            bucket = self.free.get(key)
            if bucket:
                tensor = bucket.pop()
                if not bucket:
                    self.free.pop(key, None)
                self.bytes -= tensor.numel() * tensor.element_size()
                return tensor
        return torch.empty_like(like, device="cpu", pin_memory=True)

    def release(self, lease):
        with self.lock:
            self.pending.append(lease)
            self.collect()

    def close(self):
        with self.lock:
            self.closed = True
            for lease in self.pending:
                if lease.event is not None:
                    lease.event.synchronize()
            self.pending.clear()
            self.free.clear()
            self.bytes = 0


class CacheTransferManager:
    def __init__(self, device, *, enabled=True, pinned_pool_bytes=128 * 1024 ** 2):
        self.device = torch.device(device)
        self.enabled = bool(enabled and self.device.type == "cuda")
        self.pool = PinnedBufferPool(pinned_pool_bytes)
        self.lock = threading.RLock()
        self.stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        self.pending = []
        self.metrics = {"h2d_bytes": 0, "d2h_bytes": 0, "copies": 0}

    @staticmethod
    def mark_computed(cache):
        legacy = cache.to_legacy_cache()
        if legacy and legacy[0][0].is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(legacy[0][0].device))
            cache._fastsd_ready_event = event

    def collect(self):
        with self.lock:
            self.pending = [(event, sources) for event, sources in self.pending if not event.query()]
            self.pool.collect()

    @staticmethod
    def wait_ready(cache, device):
        event = getattr(cache, "_fastsd_ready_event", None)
        if event is None:
            return
        if torch.device(device).type == "cuda":
            stream = torch.cuda.current_stream(device)
            stream.wait_event(event)
            for layer in cache.to_legacy_cache():
                for tensor in layer:
                    tensor.record_stream(stream)
        else:
            event.synchronize()

    def move(self, cache, device):
        device = torch.device(device)
        legacy = cache.to_legacy_cache()
        if not legacy:
            return cache
        source_device = legacy[0][0].device
        if source_device == device:
            self.wait_ready(cache, device)
            return cache
        with self.lock:
            self.collect()
            size = sum(t.numel() * t.element_size() for layer in legacy for t in layer)
            key = "h2d_bytes" if device.type == "cuda" else "d2h_bytes"
            self.metrics[key] += size
            self.metrics["copies"] += 1
            if not self.enabled:
                self.wait_ready(cache, source_device)
                return DynamicCache.from_legacy_cache([(k.to(device), v.to(device)) for k, v in legacy])

            # A host cache may itself still be receiving a preceding D2H copy.
            # Stream ordering covers both directions without a host-side wait.
            source_event = getattr(cache, "_fastsd_ready_event", None)
            if source_event is None and source_device.type == "cuda":
                source_event = torch.cuda.Event()
                source_event.record(torch.cuda.current_stream(source_device))
            if source_event is not None:
                self.stream.wait_event(source_event)
            copied, pinned = [], []
            with torch.cuda.stream(self.stream):
                for layer in legacy:
                    pair = []
                    for source in layer:
                        if device.type == "cpu":
                            target = self.pool.acquire(source)
                            target.copy_(source, non_blocking=True)
                            pinned.append(target)
                            source.record_stream(self.stream)
                        else:
                            if not source.is_pinned():
                                # A synchronous fallback may have produced an
                                # ordinary CPU cache. Stage it once in pinned RAM.
                                staged = self.pool.acquire(source)
                                staged.copy_(source)
                                pinned.append(staged)
                                source = staged
                            target = source.to(device, non_blocking=True)
                            target.record_stream(self.stream)
                        pair.append(target)
                    copied.append(tuple(pair))
                event = torch.cuda.Event()
                event.record(self.stream)
            result = dynamic_cache_from_views(copied)
            result._fastsd_ready_event = event
            result._fastsd_valid_length = cache.get_seq_length()
            source_lease = getattr(cache, "_fastsd_pinned_lease", None)
            if source_lease is not None:
                source_lease.event = event
            if pinned:
                lease = _Lease(pinned)
                lease.event = event
                if device.type == "cpu":
                    result._fastsd_pinned_lease = lease
                    weakref.finalize(result, self.pool.release, lease)
                else:
                    self.pool.release(lease)
            self.pending.append((event, (cache, legacy)))
            # The current caller may immediately consume the result. The
            # executor repeats this wait when a prefetch result is installed.
            if device.type == "cuda":
                self.wait_ready(result, device)
            return result

    def close(self):
        with self.lock:
            for event, _ in self.pending:
                event.synchronize()
            self.pending.clear()
            self.pool.close()
