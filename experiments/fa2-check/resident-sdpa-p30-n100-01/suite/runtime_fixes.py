"""Isolated lifecycle repair; the frozen upstream runtime remains unchanged.

Only completed transfer collection is added after each executor cycle.  The
upstream event fences decide what can be released, and all collection callers
share one lock.  No memory quotas, transfer policies, or scheduler rules change.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
import weakref
from pathlib import Path

FIX_ID = "collect-completed-kv-transfers-after-execution-v1"


def build_manager_class(base, register=None):
    """Build the wrapper separately so real collector logic can be mock-tested."""
    class CollectingCacheTransferManager(base):
        _runtime_fix_id = FIX_ID

        def __init__(self, *args, **kwargs):
            self._runtime_fix_poll_lock = threading.RLock()
            self._runtime_fix_stats = {
                "cycles": 0,
                "poll_calls": 0,
                "poll_errors": 0,
                "removed_tickets": 0,
                "peak_pending": 0,
                "remaining_pending": 0,
            }
            super().__init__(*args, **kwargs)
            if register is not None:
                register(self, self._runtime_fix_stats)

        def _runtime_fix_observe_pending(self):
            with self.lock:
                pending = len(self.pending)
                # Upstream increments copies under this same lock on every
                # append, and only poll_completed removes pending tickets.
                copies = int(self.metrics.get("copies", 0))
            stats = self._runtime_fix_stats
            stats["remaining_pending"] = pending
            stats["peak_pending"] = max(stats["peak_pending"], pending)
            stats["removed_tickets"] = max(stats["removed_tickets"], copies - pending)

        def poll_completed(self):
            with self._runtime_fix_poll_lock:
                self._runtime_fix_stats["poll_calls"] += 1
                self._runtime_fix_observe_pending()
                try:
                    return super().poll_completed()
                except BaseException:
                    self._runtime_fix_stats["poll_errors"] += 1
                    raise
                finally:
                    self._runtime_fix_observe_pending()

        # The upstream alias points to its original function, so it must be
        # rebound explicitly to keep every maintenance path serialized.
        collect = poll_completed

        def end_execution(self, *, compute_stream):
            super().end_execution(compute_stream=compute_stream)
            with self._runtime_fix_poll_lock:
                self._runtime_fix_stats["cycles"] += 1
                self.poll_completed()

        def stats_snapshot(self):
            with self._runtime_fix_poll_lock:
                self._runtime_fix_observe_pending()
                return dict(self._runtime_fix_stats)

    CollectingCacheTransferManager.__name__ = "CollectingCacheTransferManager"
    CollectingCacheTransferManager.__qualname__ = "CollectingCacheTransferManager"
    return CollectingCacheTransferManager


class Installation:
    def __init__(self, output_dir, provenance):
        self.output_path = Path(output_dir) / "runtime-fixes.json"
        self.provenance = provenance
        self._lock = threading.RLock()
        self._managers = []

    def register(self, manager, stats):
        # Keep statistics after destruction without extending any GPU cache
        # object's lifetime through the diagnostic handle.
        with self._lock:
            self._managers.append((weakref.ref(manager), stats))

    def snapshot(self):
        with self._lock:
            entries = list(self._managers)
        managers = []
        for ref, retained_stats in entries:
            manager = ref()
            stats = manager.stats_snapshot() if manager is not None else dict(retained_stats)
            managers.append({"manager_alive": manager is not None, **stats})
        totals = {
            key: sum(item[key] for item in managers)
            for key in ("cycles", "poll_calls", "poll_errors", "removed_tickets", "remaining_pending")
        }
        totals["peak_pending"] = max((item["peak_pending"] for item in managers), default=0)
        return {
            **self.provenance,
            "snapshot_epoch_s": time.time(),
            "manager_count": len(managers),
            "statistics": totals,
            "managers": managers,
            "peak_pending_definition": "Maximum pending count observed before/after a serialized collection; not a sampling-free high-water mark.",
        }

    def write_snapshot(self):
        document = self.snapshot()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(self.output_path.name + ".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.output_path)
        return document


def install(output_dir, *, source_commit=None):
    """Install once inside the Target child; call write_snapshot in its finally."""
    from src import engine

    base = engine.CacheTransferManager
    if getattr(base, "_runtime_fix_id", None) is not None:
        raise RuntimeError("CacheTransferManager lifecycle repair is already installed")
    source_path = Path(inspect.getsourcefile(base)).resolve()
    handle = Installation(output_dir, {
        "fix_id": FIX_ID,
        "installed_epoch_s": time.time(),
        "pid": os.getpid(),
        "source_commit": source_commit,
        "base_class": base.__module__ + "." + base.__qualname__,
        "base_source_file": str(source_path),
        "base_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "hook": "src.engine.CacheTransferManager",
        "behavior": "Call upstream nonblocking poll_completed after every end_execution; serialize poll_completed and collect with one RLock.",
        "scope": "Applied identically to standard SD and FastSD; official runtime files unchanged.",
        "quota_changed": False,
        "statistics_write_policy": "At install and explicitly at Target exit; no inference-loop file writes.",
    })
    engine.CacheTransferManager = build_manager_class(base, handle.register)
    handle.write_snapshot()
    return handle
