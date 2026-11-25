"""Bounded device residency for the official verifier's persistent client KV slots."""

from collections import OrderedDict
from time import perf_counter

import torch


def empty_stats():
    return dict(evictions=0, restores=0, h2d_bytes=0, d2h_bytes=0,
                io_stall_ms_total=0.0, io_stall_count=0,
                exposed_h2d_stall_ms=0.0, io_stall_available=True)


class LRUSlots:
    def __init__(self, sample, capacity, *, synchronize):
        self.device_slots = sample.new_empty((capacity, 2, *sample.shape))
        self.available = list(range(capacity))
        self.resident = OrderedDict()
        self.backing = {}
        self.dirty = set()
        self.synchronize = synchronize
        self.stats = {}
        self.writing = False
        self.slot_bytes = 2 * sample.numel() * sample.element_size()

    def begin(self, requests):
        self.stats = {request.client_idx: empty_stats() for request in requests}
        self.writing = False
        for request in requests:
            if request.prefill:
                self.backing.pop(request.client_idx, None)
                self.dirty.discard(request.client_idx)
                slot = self.resident.pop(request.client_idx, None)
                if slot is not None:
                    self.available.append(slot)

    def slot(self, client):
        if client in self.resident:
            self.resident.move_to_end(client)
            slot = self.resident[client]
        else:
            stats = self.stats[client]
            if self.available:
                slot = self.available.pop()
            else:
                victim, slot = self.resident.popitem(last=False)
                if victim in self.dirty:
                    host = self.backing.get(victim)
                    if host is None:
                        host = torch.empty_like(self.device_slots[slot], device="cpu")
                        self.backing[victim] = host
                    host.copy_(self.device_slots[slot])
                    self.dirty.remove(victim)
                    stats["d2h_bytes"] += self.slot_bytes
                stats["evictions"] += 1
            if not self.writing:
                self.synchronize()
                started = perf_counter()
                self.device_slots[slot].copy_(self.backing[client])
                self.synchronize()
                elapsed = (perf_counter() - started) * 1000
                stats.update(restores=1, h2d_bytes=self.slot_bytes, io_stall_ms_total=elapsed,
                             io_stall_count=1, exposed_h2d_stall_ms=elapsed)
            self.resident[client] = slot
        if self.writing:
            self.dirty.add(client)
        return self.device_slots[slot]


class SlotView:
    def __init__(self, slots, component):
        self.slots = slots
        self.component = component

    def __getitem__(self, key):
        layers, client, *dimensions = key
        return self.slots.slot(int(client))[self.component, layers, *dimensions]


class BatchResponses:
    def __init__(self, queue):
        self.queue = queue
        self.pending = []

    def put(self, value):
        self.pending.append(value)

    def flush(self, stats):
        for payload, client in self.pending:
            self.queue.put(((*payload, stats.get(client, empty_stats())), client))
        self.pending.clear()
