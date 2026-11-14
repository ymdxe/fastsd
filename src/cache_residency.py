"""Explicit target KV cache residency states, budget and victim choice.

``BYPASS`` is the default: every cache stays on the GPU and none of the
transfer machinery (tickets, CUDA events, pinned buffers, prefetch thread,
planner lookahead) may run. ``ACTIVE`` tracks sessions whenever an explicit
byte budget is configured. Offloading starts when actual resident KV reaches
the high watermark, or free GPU memory falls below 256 MiB, using these states::

    GPU_RESIDENT --evict--> EVICTION_PENDING --D2H done--> CPU_RESIDENT
          ^                                                     |
          |                                            prefetch | demand
          +--RESTORE done-- RESTORE_PENDING <--claim-- PREFETCH_PENDING

Only the control owner mutates records.  The executor may read a record and
advance its ticket.  This module performs no CUDA calls and holds no cache
references beyond what callers register, so it is unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time

BYPASS = "bypass"
ACTIVE = "active"

GPU_RESIDENT = "gpu_resident"
EVICTION_PENDING = "eviction_pending"
CPU_RESIDENT = "cpu_resident"
PREFETCH_PENDING = "prefetch_pending"
RESTORE_PENDING = "restore_pending"

EMERGENCY_FREE_BYTES = 256 * 1024 ** 2

STATES = (GPU_RESIDENT, EVICTION_PENDING, CPU_RESIDENT,
          PREFETCH_PENDING, RESTORE_PENDING)

# A cache may only leave a state through one of these edges.  Failure paths
# fall back to the last durable state (GPU_RESIDENT or CPU_RESIDENT) so a
# record is never left in an ambiguous state.
_TRANSITIONS = {
    GPU_RESIDENT: {EVICTION_PENDING},
    EVICTION_PENDING: {CPU_RESIDENT, GPU_RESIDENT},
    CPU_RESIDENT: {PREFETCH_PENDING, RESTORE_PENDING},
    PREFETCH_PENDING: {RESTORE_PENDING, CPU_RESIDENT},
    RESTORE_PENDING: {GPU_RESIDENT, CPU_RESIDENT},
}

_DURABLE = (GPU_RESIDENT, CPU_RESIDENT)


class InvalidTransition(RuntimeError):
    """Raised when a caller tries to move a record along a missing edge."""


@dataclass
class SessionRecord:
    """One live target session's cache ownership and placement."""

    pid: object
    state: str = GPU_RESIDENT
    epoch: int = 0
    version: int = 0
    valid_len: int = 0
    cpu_valid_len: int = 0
    nbytes: int = 0
    bytes_per_token: int = 0
    last_used_tick: int = -1
    last_used_time: float = 0.0
    draft_total_time: float = 0.0
    transport_rtt: float = 0.0
    accept_prob: float = 1.0
    planned_tick: int = -1
    in_transfer: bool = False
    gpu_cache: object = None
    cpu_cache: object = None

    def protected(self, tick: int) -> bool:
        return self.planned_tick >= tick or self.in_transfer

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"SessionRecord(pid={self.pid!r}, state={self.state}, "
                f"v={self.version}, len={self.valid_len}, bytes={self.nbytes})")


class ResidencyManager:
    """Budget-driven placement policy for target KV caches.

    ``budget_bytes <= 0`` disables the manager permanently: ``active`` stays
    False, no records are tracked and callers must take their bypass path.
    """

    def __init__(self, *, budget_bytes=0, high_ratio=0.90, low_ratio=0.75,
                 headroom_bytes=0, clock=time.monotonic):
        if not 0.0 <= float(low_ratio) < float(high_ratio) <= 1.0:
            raise ValueError("residency ratios must satisfy 0 <= low < high <= 1")
        self.budget_bytes = max(0, int(budget_bytes))

        self.high_ratio = float(high_ratio)
        self.low_ratio = float(low_ratio)
        self.headroom_bytes = max(0, int(headroom_bytes))
        self.clock = clock
        self.lock = threading.RLock()
        self.records: dict = {}
        # Management is a property of the budget, not of the current pressure:
        # eviction and prefetch are triggered independently (see engine).
        self.mode = ACTIVE if self.budget_bytes > 0 else BYPASS
        self.high = False
        self.emergency_free_bytes = EMERGENCY_FREE_BYTES
        self.free_bytes = None
        self.emergency = False
        self._started = False
        self.counters = {
            "activations": 0,
            "deactivations": 0,
            "evictions": 0,
            "restores": 0,
            "prefetch_issued": 0,
            "prefetch_claimed": 0,
            "prefetch_cancelled": 0,
            "invalid_transitions": 0,
            "pressure_samples": 0,
            "pressure_guard_trips": 0,
            "emergency_pressure_samples": 0,
            "emergency_eviction_rounds": 0,
            "emergency_evictions": 0,
            "emergency_blocked_rounds": 0,
        }

    # -- mode ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0

    @property
    def active(self) -> bool:
        return self.mode == ACTIVE

    def mark_started(self) -> bool:
        """One-time resource setup hook; returns True the first time."""

        with self.lock:
            if not self.enabled or self._started:
                return False
            self._started = True
            self.counters["activations"] += 1
            return True

    # -- records ---------------------------------------------------------
    def note_created(self, pid, *, nbytes=0, bytes_per_token=0, valid_len=0,
                     epoch=0, version=0, cache=None):
        with self.lock:
            record = self.records.get(pid)
            if record is None:
                record = SessionRecord(pid=pid, last_used_time=self.clock())
                self.records[pid] = record
            record.state = GPU_RESIDENT
            record.valid_len = int(valid_len)
            record.cpu_valid_len = int(valid_len)
            record.gpu_cache = cache
            record.epoch = int(epoch)
            record.version = int(version)
            if bytes_per_token:
                record.bytes_per_token = int(bytes_per_token)
            record.nbytes = int(nbytes) if nbytes else (
                record.valid_len * record.bytes_per_token)
            return record

    def note_used(self, pid, tick):
        with self.lock:
            record = self.records.get(pid)
            if record is not None:
                record.last_used_tick = int(tick)
                record.last_used_time = self.clock()
                record.planned_tick = int(tick)

    def note_request_cost(self, pid, *, draft_total_time=0.0, transport_rtt=0.0,
                          accept_prob=1.0):
        """Remember the latest round's seconds and acceptance estimate."""
        values = (float(draft_total_time), float(transport_rtt), float(accept_prob))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("offload timing and acceptance must be finite and non-negative")
        with self.lock:
            record = self.records.get(pid)
            if record is not None:
                record.draft_total_time, record.transport_rtt, probability = values
                record.accept_prob = max(probability, 1e-6)

    def note_mutated(self, pid, *, version=0, valid_len=0):
        with self.lock:
            record = self.records.get(pid)
            if record is None:
                return None
            record.version = int(version) if version else record.version + 1
            if valid_len:
                record.valid_len = int(valid_len)
            if record.bytes_per_token:
                record.nbytes = record.valid_len * record.bytes_per_token
            return record

    def release(self, pid):
        with self.lock:
            return self.records.pop(pid, None)

    def get(self, pid):
        with self.lock:
            return self.records.get(pid)

    def state_of(self, pid):
        record = self.get(pid)
        return None if record is None else record.state

    # -- transitions -----------------------------------------------------
    def transition(self, pid, new_state):
        with self.lock:
            record = self.records.get(pid)
            if record is None:
                raise InvalidTransition(f"unknown session {pid!r}")
            if new_state not in _TRANSITIONS.get(record.state, ()):
                self.counters["invalid_transitions"] += 1
                raise InvalidTransition(
                    f"{pid!r}: {record.state} -> {new_state}")
            record.state = new_state
            record.in_transfer = new_state in (EVICTION_PENDING, PREFETCH_PENDING,
                                               RESTORE_PENDING)
            if new_state == GPU_RESIDENT:
                record.gpu_cache = record.gpu_cache
            elif new_state == CPU_RESIDENT:
                record.cpu_valid_len = record.valid_len
            return record

    def force_state(self, pid, state):
        """Recovery only: land in a durable state after a failed transfer."""

        if state not in _DURABLE:
            raise InvalidTransition(f"force_state requires a durable state, got {state}")
        with self.lock:
            record = self.records.get(pid)
            if record is None:
                return None
            record.state = state
            record.in_transfer = False
            return record

    def protect(self, pid, until_tick):
        with self.lock:
            record = self.records.get(pid)
            if record is not None:
                record.planned_tick = max(record.planned_tick, int(until_tick))

    def unprotect(self, pid):
        with self.lock:
            record = self.records.get(pid)
            if record is not None:
                record.planned_tick = -1

    # -- pressure --------------------------------------------------------
    def used_bytes(self) -> int:
        with self.lock:
            return sum(r.nbytes for r in self.records.values()
                       if r.state in (GPU_RESIDENT, EVICTION_PENDING, RESTORE_PENDING))

    def high_watermark_bytes(self) -> int:
        return int(self.budget_bytes * self.high_ratio)

    def low_watermark_bytes(self) -> int:
        return int(self.budget_bytes * self.low_ratio)

    def update_pressure(self, used_bytes, *, guard_free_bytes=None) -> bool:
        """Sample budget and free memory; return whether eviction is needed."""

        with self.lock:
            self.counters["pressure_samples"] += 1
            used = int(used_bytes)
            high = self.enabled and used >= self.high_watermark_bytes()
            if guard_free_bytes is not None and guard_free_bytes < self.headroom_bytes:
                # The ordinary headroom setting is diagnostic. Only the
                # separate 256 MiB emergency threshold bypasses high.
                self.counters["pressure_guard_trips"] += 1
            self._record_free_bytes(guard_free_bytes)
            self.counters["emergency_pressure_samples"] += int(self.emergency)
            self.high = high
            return high or self.emergency

    def _record_free_bytes(self, free_bytes):
        self.free_bytes = None if free_bytes is None else max(0, int(free_bytes))
        self.emergency = (self.enabled and self.free_bytes is not None
                          and self.free_bytes < self.emergency_free_bytes)

    def needs_eviction(self) -> bool:
        with self.lock:
            return self.enabled and (self.used_bytes() >= self.high_watermark_bytes()
                                     or self.emergency)

    def deficit_bytes(self, extra_bytes=0, *, guard_free_bytes=None) -> int:
        """Trim at high, with a separate emergency free-memory override.

        Below high, only free memory < 256 MiB triggers eviction. In that case
        reclaim the free-memory shortfall plus the upcoming restores, even
        when KV usage is already below low. Restore estimates alone cannot
        trigger eviction. A fresh executor sample supersedes the planner's.
        """

        with self.lock:
            used = self.used_bytes()
            if guard_free_bytes is not None:
                self._record_free_bytes(guard_free_bytes)
            if not self.enabled:
                return 0
            extra = max(0, int(extra_bytes))
            normal = (max(0, used + extra - self.low_watermark_bytes())
                      if used >= self.high_watermark_bytes() else 0)
            emergency = (self.emergency_free_bytes - self.free_bytes + extra
                         if self.emergency else 0)
            return max(normal, emergency)

    def pending_restore(self, pids):
        """Pids in ``pids`` whose cache is not currently on the GPU."""

        with self.lock:
            out = []
            for pid in pids:
                record = self.records.get(pid)
                if record is not None and record.state != GPU_RESIDENT:
                    out.append(pid)
            return out

    def cpu_resident_count(self) -> int:
        with self.lock:
            return sum(1 for r in self.records.values() if r.state == CPU_RESIDENT)

    # -- victims ---------------------------------------------------------
    def rank_victims(self, tick, *, exclude=()):
        """Rank unprotected GPU sessions by cost - idle, descending.

        cost = (draft_total_time + transport_rtt) / accept_prob
        idle = now - last_used_time
        Both terms use seconds; scheduling ticks only control protection.
        """
        excluded = set(exclude)
        with self.lock:
            now = self.clock()
            candidates = [
                r for r in self.records.values()
                if r.state == GPU_RESIDENT and not r.in_transfer
                and r.nbytes > 0 and r.pid not in excluded and not r.protected(tick)
            ]
            candidates.sort(key=lambda r: (
                (r.draft_total_time + r.transport_rtt) / r.accept_prob
                - (now - r.last_used_time)), reverse=True)
            return [record.pid for record in candidates]

    def select_victims(self, needed_bytes, tick, *, exclude=()):
        """Take the ranked prefix that can reclaim the requested bytes."""
        if needed_bytes <= 0:
            return []
        chosen, total = [], 0
        for pid in self.rank_victims(tick, exclude=exclude):
            if total >= needed_bytes:
                break
            chosen.append(pid)
            total += self.records[pid].nbytes
        return chosen

    # -- reporting -------------------------------------------------------
    def stats(self):
        with self.lock:
            counts = {state: 0 for state in STATES}
            for record in self.records.values():
                counts[record.state] = counts.get(record.state, 0) + 1
            return {
                "mode": self.mode,
                "victim_policy": "cost_minus_idle_descending",
                "budget_bytes": self.budget_bytes,
                "high_watermark_bytes": self.high_watermark_bytes(),
                "low_watermark_bytes": self.low_watermark_bytes(),
                "used_bytes": self.used_bytes(),
                "high": self.high,
                "emergency": self.emergency,
                "emergency_free_bytes": self.emergency_free_bytes,
                "last_free_bytes": self.free_bytes,
                "states": counts,
                "sessions": len(self.records),
                **{f"residency_{k}": v for k, v in self.counters.items()},
            }
