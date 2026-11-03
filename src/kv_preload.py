"""Paper Sec. 3.4: inactive-cache preloading on a separate CUDA stream."""

from concurrent.futures import ThreadPoolExecutor

import torch
from transformers.cache_utils import DynamicCache


class KVPreloader:
    def __init__(self, caches, device):
        self.caches = caches
        self.device = torch.device(device)
        self.worker = ThreadPoolExecutor(max_workers=1) if self.device.type == "cuda" else None
        self.stream = torch.cuda.Stream(device=self.device) if self.worker is not None else None
        self.pending = {}
        self.retired = []

    def _load(self, source, length):
        torch.cuda.set_device(self.device)
        host = [(k if k.is_pinned() else k.pin_memory(),
                 v if v.is_pinned() else v.pin_memory()) for k, v in source.to_legacy_cache()]
        with torch.cuda.stream(self.stream), torch.no_grad():
            loaded = DynamicCache.from_legacy_cache(tuple(
                (k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True)) for k, v in host))
            ready = torch.cuda.Event()
            ready.record(self.stream)
        # Retain pinned sources until DMA completes, including abandoned predictions.
        return source, length, loaded, ready, host

    def predict(self, proc_ids, active=()):
        if self.worker is None:
            return
        desired = set(proc_ids) | set(active)
        for pid in list(self.pending):
            if pid not in desired:
                future = self.pending.pop(pid)
                if not future.cancel():
                    self.retired.append(future)
        self.collect()
        for pid in proc_ids:
            source = self.caches.get(pid)
            if source is not None and pid not in self.pending:
                if source.to_legacy_cache()[0][0].device != self.device:
                    self.pending[pid] = self.worker.submit(self._load, source, source.get_seq_length())

    def ensure(self, pid):
        """Only the inference thread installs a copy before consuming the cache."""
        source = self.caches[pid]
        future = self.pending.pop(pid, None)
        if future is not None:
            original, length, loaded, ready, host = future.result()
            self.retired.append(future)
            if original is source and length == source.get_seq_length():
                consumer = torch.cuda.current_stream(self.device)
                consumer.wait_event(ready)
                for key, value in loaded.to_legacy_cache():
                    key.record_stream(consumer)
                    value.record_stream(consumer)
                self.caches[pid] = loaded
                return
        if source.to_legacy_cache()[0][0].device != self.device:
            self.caches[pid] = DynamicCache.from_legacy_cache(tuple(
                (k.to(self.device), v.to(self.device)) for k, v in source.to_legacy_cache()))

    def offload(self, pid):
        source = self.caches[pid]
        if self.device.type != "cuda" or source.to_legacy_cache()[0][0].device.type == "cpu":
            return
        host = []
        for key, value in source.to_legacy_cache():
            k = torch.empty_like(key, device="cpu", pin_memory=True)
            v = torch.empty_like(value, device="cpu", pin_memory=True)
            k.copy_(key)
            v.copy_(value)
            host.append((k, v))
        self.caches[pid] = DynamicCache.from_legacy_cache(tuple(host))

    def collect(self):
        self.retired = [f for f in self.retired if not f.done() or not f.result()[3].query()]

    def close(self):
        if self.worker is not None:
            self.worker.shutdown(wait=True)
            self.stream.synchronize()
            for future in list(self.pending.values()) + self.retired:
                future.result()
            self.pending.clear()
            self.retired.clear()
