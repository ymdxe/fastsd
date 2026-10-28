"""Thread coordination for the FastSD target scheduler.

Ownership model
---------------
* Exactly one **control owner** (the thread calling
  :meth:`TargetSchedulerRuntime.run`) mutates scheduler queues, ``WorkItem``
  state, admission/WRR state and the *intent* of the KV prefetcher.  This is
  what makes the planner snapshot stable without a scheduler-wide lock.
* Exactly one **model executor** thread runs target forward batches.  It never
  touches scheduler queues or ``WorkItem`` fields; batches are handed over
  fully sliced and raw responses come back through a result queue.
* Exactly one **KV prefetch** thread performs CPU<->GPU copies.  It only
  touches cache residency and cache objects, always under the cache lock, and
  never holds that lock across a tensor copy.

The control owner keeps draining ingress while the executor is blocked in a
forward.  That is what makes a request B, which arrives during forward A,
visible to the lookahead planner (and therefore a prefetch candidate) before A
finishes.  Requests are still executed by a single forward path in arrival
order chosen by the existing pure planner; strict-FCFS modes bypass this module
entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional

CATEGORIES = ("short", "mid", "long")
TASK_TYPES = ("verify", "prefill")


class ShutdownRequested(Exception):
    """Raised by the ingress callback when the worker receives ``None``."""


class NotifyingQueue(queue.Queue):
    """Publish before notifying, so a consumer can clear/drain/wait safely."""

    def __init__(self, wake_event):
        super().__init__()
        self.wake_event = wake_event

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        self.wake_event.set()


class IngressRelay:
    """Bridge process ingress into the control loop's shared event channel."""

    def __init__(self, source, wake_event):
        self.source = source
        self.queue = NotifyingQueue(wake_event)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="fastsd-ingress")

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self.source.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                item = None
            self.queue.put(item)
            if item is None:
                return

    def close(self):
        self._stop.set()
        self._thread.join()


def invoke_no_grad(callback: Callable[[], Any]) -> Any:
    """Run one model batch with autograd disabled in the *current* thread.

    ``torch.no_grad()`` is thread-local, so the decorator on the target
    process entry point does not cover a dedicated executor thread.  This
    helper is the production path used by the engine's executor job.
    """

    import torch

    with torch.no_grad():
        return callback()


def set_thread_device(device: Any, set_device: Optional[Callable[[Any], None]] = None) -> bool:
    """Bind a newly created worker thread to the target CUDA device.

    CUDA's current device is thread-local.  Without this, a thread created by
    the executor/prefetch pool would default to ``cuda:0`` even when the
    target model lives on another device.  CPU-only deployments must not touch
    the CUDA API at all.
    """

    if isinstance(device, str):
        if not device.startswith("cuda"):
            return False
    elif getattr(device, "type", None) != "cuda":
        return False
    if set_device is None:
        import torch

        set_device = torch.cuda.set_device
    set_device(device)
    return True


def iter_execution_phases(
    job: Any,
    run_batch: Callable[[list], dict],
):
    """Yield the Verify then Prefill phases of one admitted plan.

    Verify is yielded as a non-final stage so the control owner can apply its
    progress and return already-available responses before Prefill starts.
    The final stage commits the plan.  Keeping this protocol here (rather than
    in an engine-local closure) makes the exact production behaviour testable
    without loading a model.
    """

    verify_responses = {}
    if job.verify_batch:
        verify_responses = run_batch(job.verify_batch)
        if not job.prefill_batch:
            yield ExecutionOutcome(
                job=job,
                verify_responses=verify_responses,
                phase="complete",
                final=True,
            )
            return
        yield ExecutionOutcome(
            job=job,
            verify_responses=verify_responses,
            phase="verify",
            final=False,
        )
    prefill_responses = run_batch(job.prefill_batch) if job.prefill_batch else {}
    yield ExecutionOutcome(
        job=job,
        verify_responses=verify_responses,
        prefill_responses=prefill_responses,
        phase="complete",
        final=True,
    )


def make_forward_batch_runner(
    handle_batch: Callable[[list], dict],
    on_batch_start: Optional[Callable[[], None]] = None,
    on_batch_end: Optional[Callable[[], None]] = None,
) -> Callable[[list], dict]:
    """Production executor batch wrapper: bookkeeping plus ``no_grad``."""

    def run_batch(batch: list) -> dict:
        if not batch:
            return {}
        if on_batch_start is not None:
            on_batch_start()
        try:
            return invoke_no_grad(lambda: handle_batch(batch))
        finally:
            if on_batch_end is not None:
                on_batch_end()

    return run_batch


def make_forward_job_runner(run_batch: Callable[[list], dict]):
    """Build the engine's executor job: a two-phase forward-only generator."""

    def run_job(job: Any):
        yield from iter_execution_phases(job, run_batch)

    return run_job


# ---------------------------------------------------------------------------
# Scheduler queues
# ---------------------------------------------------------------------------
class SchedulerQueues:
    """Single-owner classification queues for persistent WorkItems.

    Plain lists are enough because only the control owner reads or writes the
    queues.  The previous ``queue.Queue`` objects were used as plain
    containers; their locks could not protect the planner snapshot anyway.
    """

    def __init__(self, metrics: Optional[dict] = None) -> None:
        self._queues = {
            task_type: {category: [] for category in CATEGORIES}
            for task_type in TASK_TYPES
        }
        self.metrics = metrics if metrics is not None else {}
        self.dirty = False

    def enqueue(self, item: Any) -> None:
        self._queues[item.task_type][item.category].append(item)
        self.dirty = True

    def mark_dirty(self) -> None:
        """Force the next sort to re-bucket (e.g. after a chunk commit)."""

        self.dirty = True

    def snapshot(self) -> dict[str, dict[str, list]]:
        """Return a stable copy for the pure planner."""

        return {
            task_type: {
                category: list(self._queues[task_type][category])
                for category in CATEGORIES
            }
            for task_type in TASK_TYPES
        }

    def iter_items(self):
        for task_type in TASK_TYPES:
            for category in CATEGORIES:
                yield from self._queues[task_type][category]

    def has_items(self) -> bool:
        return any(True for _ in self.iter_items())

    def has_live_items(self) -> bool:
        return any(
            not getattr(item, "finished", False)
            and item.state != "finished"
            and getattr(item, "remaining_tokens", 0) > 0
            for item in self.iter_items()
        )

    def has_live_task(self, task_type: str) -> bool:
        for category in CATEGORIES:
            for item in self._queues[task_type][category]:
                if (
                    not getattr(item, "finished", False)
                    and item.state != "finished"
                    and item.remaining_tokens > 0
                ):
                    return True
        return False

    def remove_work_ids(self, work_ids: Iterable[str]) -> int:
        """Remove exact WorkItems by id, preserving other work on the same pid."""

        targets = set(work_ids)
        if not targets:
            return 0
        removed = 0
        for task_type in TASK_TYPES:
            for category in CATEGORIES:
                bucket = self._queues[task_type][category]
                kept = [item for item in bucket if item.work_id not in targets]
                removed += len(bucket) - len(kept)
                if len(kept) != len(bucket):
                    bucket[:] = kept
        if removed:
            self.dirty = True
        return removed

    def remove_where(self, predicate: Callable[[Any], bool]) -> int:
        """Remove matching items from every bucket (used by Prefill cancel)."""

        removed = 0
        for task_type in TASK_TYPES:
            for category in CATEGORIES:
                bucket = self._queues[task_type][category]
                kept = [item for item in bucket if not predicate(item)]
                removed += len(bucket) - len(kept)
                if len(kept) != len(bucket):
                    bucket[:] = kept
        if removed:
            self.dirty = True
        return removed

    def prune_finished(self) -> int:
        """Drop terminated WorkItems so they cannot cause empty plans."""

        removed = 0
        for task_type in TASK_TYPES:
            for category in CATEGORIES:
                bucket = self._queues[task_type][category]
                kept = [
                    item
                    for item in bucket
                    if not getattr(item, "finished", False)
                    and item.state != "finished"
                    and getattr(item, "remaining_tokens", 1) > 0
                ]
                removed += len(bucket) - len(kept)
                if len(kept) != len(bucket):
                    bucket[:] = kept
        if removed:
            self.dirty = True
            self.metrics["finished_work_items_removed"] = (
                self.metrics.get("finished_work_items_removed", 0) + removed
            )
        return removed

    def sort(
        self,
        accept_stats: Mapping[Any, Any],
        now: float,
        category_for: Callable[[Any], str],
        priority: Callable[[Any, Mapping[Any, Any], float], float],
    ) -> None:
        """Re-bucket and priority-sort every queue when it has changed."""

        if not self.dirty:
            return
        self.prune_finished()
        drained = []
        for task_type in TASK_TYPES:
            for category in CATEGORIES:
                drained.extend(self._queues[task_type][category])
                self._queues[task_type][category] = []
        for item in drained:
            if item.task_type == "prefill":
                # A chunked Prefill keeps its WorkItem, so its remaining
                # length can move it between buckets.
                item.category = category_for(item)
            self._queues[item.task_type][item.category].append(item)
        for task_type in TASK_TYPES:
            for category in CATEGORIES:
                self._queues[task_type][category].sort(
                    key=lambda item: priority(item.request, accept_stats, now),
                    reverse=True,
                )
        self.dirty = False


def cancel_prefill_work_items(
    *,
    work_items: dict,
    queues: SchedulerQueues,
    proc_id: Any,
    task_id: str,
) -> tuple[set, int]:
    """Remove matching Prefill WorkItems, deferring in-flight ones.

    Returns ``(removed_work_ids, deferred_count)``.  A WorkItem already
    admitted into the running plan cannot stop its forward; it is marked
    ``cancel_pending`` so the completion path suppresses its response.  Work
    ids are unique, so unrelated work for the same pid is preserved.
    """

    removed_ids: set = set()
    deferred = 0
    for work_id, item in list(work_items.items()):
        if item.proc_id != proc_id or item.task_type != "prefill":
            continue
        item_task_id = str(item.request.get("task_id", ""))
        if task_id and item_task_id != task_id:
            continue
        removed_ids.add(work_id)
        if item.state == "reserved":
            item.cancel_pending = True
            deferred += 1
        else:
            item.finished = True
            item.state = "finished"
        work_items.pop(work_id, None)
    queues.remove_work_ids(removed_ids)
    return removed_ids, deferred


def commit_admission_progress(
    plan: Any,
    queues: SchedulerQueues,
    commit: Callable[..., None],
    *,
    completed_work_ids: Iterable[str] = (),
    progress: Optional[Mapping[str, int]] = None,
) -> None:
    """Commit plan progress and force re-bucketing of chunked WorkItems.

    A partial Prefill chunk changes ``remaining_tokens`` without enqueueing
    anything.  The category (and therefore the WRR bucket) must therefore be
    refreshed even though the queue contents themselves did not change.
    """

    commit(plan, completed_work_ids=completed_work_ids, progress=progress)
    queues.mark_dirty()


def terminate_pending_work_items(
    *,
    work_items: dict,
    queues: SchedulerQueues,
    response_queues: Mapping[Any, Any],
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Give accepted-but-unrunnable work exactly one terminal error response.

    Used by graceful shutdown after the drain found no runnable plan.  A
    Prefill that never received its timing update gets a specific reason; all
    other residual/chunked work gets the generic shutdown reason.  Items are
    removed by ``work_id`` and both the classification queues and the index
    are left empty.
    """

    removed_ids: set = set()
    for work_id, item in list(work_items.items()):
        if not item.finished:
            if (
                item.task_type == "prefill"
                and item.request.get("timing_required", False)
                and not item.request.get("timing_ready", False)
            ):
                payload = {
                    "error": "target worker shutdown before prefill timing update"
                }
            else:
                payload = {"error": "target worker shutdown before completing request"}
            try:
                response_queues[item.response_key or item.proc_id].put(payload)
            except Exception as exc:  # diagnostics only
                if log is not None:
                    log(f"[SHUTDOWN-RESPONSE-FAILED] {work_id}: {exc}")
        item.finished = True
        item.state = "finished"
        removed_ids.add(work_id)
        work_items.pop(work_id, None)
    queues.remove_work_ids(removed_ids)
    queues.prune_finished()
    return len(removed_ids)


# ---------------------------------------------------------------------------
# Cache residency / generation tracking
# ---------------------------------------------------------------------------
from .cache_protocol import CacheSnapshot, KVTransferTicket


class CacheResidency:
    """Content generations are independent of reserve/activate ownership."""

    def __init__(self, lock=None):
        self.lock = lock or threading.RLock()
        self.resident, self.active, self.reserved = set(), set(), set()
        self.versions, self.epochs, self.owners = {}, {}, {}
        self.candidates = {}
        self._next_epoch = 0

    def snapshot_locked(self, pid, cache):
        if pid not in self.epochs:
            self._next_epoch += 1
            self.epochs[pid] = self._next_epoch
        return CacheSnapshot.capture(cache, session_epoch=self.epochs[pid],
                                     content_version=self.versions.get(pid, 0))

    def claim_locked(self, pid, cache, job_id):
        snapshot = self.snapshot_locked(pid, cache)
        ticket = self.candidates.get(pid)
        if ticket is None or not ticket.matches(snapshot):
            return None
        self.candidates.pop(pid)
        ticket.claimed_by = job_id
        return ticket

    def retire_session_locked(self, pid):
        self.epochs.pop(pid, None)
        self.versions.pop(pid, None)
        self.owners.pop(pid, None)
        self.resident.discard(pid)
        return self.candidates.pop(pid, None)

    def bump(self, pids: Iterable[Any]) -> None:
        with self.lock:
            for pid in pids:
                self.versions[pid] = self.versions.get(pid, 0) + 1

    def activate(self, pids: Iterable[Any]) -> None:
        with self.lock:
            for pid in pids:
                self.active.add(pid)
                self.owners[pid] = self.owners.get(pid, 0) + 1

    def deactivate(self, pids: Iterable[Any]) -> None:
        with self.lock:
            self.active.difference_update(pids)

    def reserve(self, pids: Iterable[Any]) -> None:
        with self.lock:
            for pid in pids:
                self.reserved.add(pid)
                self.owners[pid] = self.owners.get(pid, 0) + 1

    def release(self, pids: Iterable[Any]) -> None:
        with self.lock:
            self.reserved.difference_update(pids)

    def protected_snapshot(self) -> set[Any]:
        with self.lock:
            return set(self.active) | set(self.reserved)

    def resident_snapshot(self) -> set[Any]:
        with self.lock:
            return set(self.resident)

    def version(self, pid: Any) -> int:
        with self.lock:
            return self.versions.get(pid, 0)

    def bump_locked(self, pid: Any) -> None:
        """Bump a version; caller already holds :attr:`lock`."""

        self.versions[pid] = self.versions.get(pid, 0) + 1


# ---------------------------------------------------------------------------
# Prefetch intent
# ---------------------------------------------------------------------------
class PrefetchCoordinator:
    """Turn a lookahead pid set into at most one outstanding prefetch intent.

    Only the newest intent matters.  An empty lookahead (no next verify
    candidate) is a correct zero-submission result: it must not evict caches
    that are still useful.  Residency that is active or reserved by an
    admitted plan is removed from both load and eviction targets.

    Unchanged intent is normally not resubmitted, so a valid transfer is not
    cancelled for no reason.  A retry *is* allowed when the residency still
    does not satisfy the intent and the worker has no pending intent (the
    previous copy was skipped because of a generation/protection race or a
    worker error).  Retries are throttled and only happen on a state-change
    refresh, never in a polling loop.
    """

    def __init__(
        self,
        *,
        submit: Callable[[Iterable[Any]], bool],
        resident_snapshot: Callable[[], set],
        protected_snapshot: Callable[[], set],
        metrics: dict,
        pending_snapshot: Optional[Callable[[], bool]] = None,
        retry_interval: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._submit = submit
        self._resident_snapshot = resident_snapshot
        self._protected_snapshot = protected_snapshot
        self.metrics = metrics
        self._pending_snapshot = pending_snapshot or (lambda: False)
        self._retry_interval = float(retry_interval)
        self._clock = clock
        self._last_desired: Optional[frozenset] = None
        self._last_submit_at = 0.0

    def refresh(self, desired_pids: Iterable[Any]) -> bool:
        requested = set(desired_pids)
        protected = self._protected_snapshot()
        excluded = requested & protected
        if excluded:
            self.metrics["prefetch_reserved_excluded"] += len(excluded)
        desired = frozenset(requested - protected)

        self.metrics["prefetch_plans"] += 1
        if not desired:
            self.metrics["prefetch_empty_desired"] += 1
            if self._last_desired == frozenset():
                self.metrics["prefetch_desired_unchanged"] += 1
            if self._last_desired:
                self._submit(frozenset())
            self._last_desired = frozenset()
            return False

        current = set(self._resident_snapshot())
        stale = set()  # prediction never evicts an official cache
        to_gpu = (desired - current) - protected

        if self._last_desired == desired:
            self.metrics["prefetch_desired_unchanged"] += 1
            if not stale and not to_gpu:
                return False
            if self._pending_snapshot():
                # A transfer for this exact intent is still queued/running.
                self.metrics["prefetch_intent_inflight"] += 1
                return False
            now = self._clock()
            if now - self._last_submit_at < self._retry_interval:
                self.metrics["prefetch_retry_throttled"] += 1
                return False
            if self._submit(desired):
                self._last_submit_at = now
                self.metrics["prefetch_retry"] += 1
                return True
            return False

        self.metrics["prefetch_evictions"] += len(stale)
        self.metrics["prefetch_hits"] += len(current & desired)
        self.metrics["prefetch_misses"] += len(to_gpu)

        self._last_desired = desired
        if not stale and not to_gpu:
            self.metrics["prefetch_noop"] += 1
            return False
        if self._submit(desired):
            self._last_submit_at = self._clock()
            return True
        return False


# ---------------------------------------------------------------------------
# KV prefetch worker
# ---------------------------------------------------------------------------
class KVPrefetchWorker:
    """Background CPU<->GPU cache mover driven by the latest desired pid set.

    The task queue has capacity one and new predictions replace older ones, so
    only the freshest intent is ever executed.  At execution time the worker
    re-derives stale/load sets from current residency instead of trusting the
    snapshot that was taken when the intent was submitted.
    """

    def __init__(
        self,
        *,
        residency: CacheResidency,
        get_cache: Callable[[Any], Any],
        set_cache: Callable[[Any, Any], None],
        move_cache: Callable[[Any, str], Any],
        gpu_device: Any,
        metrics: dict,
        on_error: Optional[Callable[[str], None]] = None,
        queue_maxsize: int = 1,
        thread_setup: Optional[Callable[[], None]] = None,
        transfers=None,
    ) -> None:
        self.transfers = transfers
        self.residency = residency
        self._get_cache = get_cache
        self._set_cache = set_cache
        self._move_cache = move_cache
        self._gpu_device = gpu_device
        self.metrics = metrics
        self._on_error = on_error
        self._thread_setup = thread_setup
        self._tasks: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._pending = 0
        self._latest_intent: Optional[frozenset] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="fastsd-kv-prefetch"
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- submission --------------------------------------------------------
    def has_pending(self) -> bool:
        """Whether an intent is queued or currently being executed."""

        with self._state_lock:
            return self._pending > 0

    def submit(self, desired: Iterable[Any]) -> bool:
        """Queue the newest desired set, replacing one not yet picked up.

        Pending accounting is reserved *before* publishing the task.  The
        worker may consume and finish the task as soon as it is published, so
        incrementing afterwards would leave a stale pending flag (and
        permanently suppress identical retries).
        """

        if self._stop.is_set():
            return False
        intent = frozenset(desired)
        with self._state_lock:
            self._latest_intent = intent
            self._pending += 1
        try:
            self._tasks.put_nowait(intent)
            self.metrics["prefetch_async_submitted"] += 1
            return True
        except queue.Full:
            self._decrement_pending()

        # Replace the older queued (not executing) intent.  The worker may
        # have just consumed it, in which case the queue is empty and the new
        # intent is published normally.
        try:
            self._tasks.get_nowait()
            self._decrement_pending()
        except queue.Empty:
            pass
        with self._state_lock:
            self._pending += 1
        try:
            self._tasks.put_nowait(intent)
            self.metrics["prefetch_async_submitted"] += 1
            return True
        except queue.Full:
            self._decrement_pending()
            self.metrics["prefetch_async_dropped"] += 1
            return False

    def _is_superseded(self, desired: frozenset) -> bool:
        with self._state_lock:
            return self._latest_intent != desired

    def _decrement_pending(self) -> None:
        """Release one pending slot; an underflow is a bookkeeping bug."""

        with self._state_lock:
            if self._pending <= 0:
                self.metrics["prefetch_pending_underflow"] += 1
                return
            self._pending -= 1

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                desired = self._tasks.get(timeout=0.005)
            except queue.Empty:
                with self._state_lock:
                    desired = self._latest_intent
                if desired:
                    try:
                        self._execute(desired)
                    except Exception as exc:
                        self.metrics["prefetch_async_failed"] += 1
                        self.metrics["prefetch_async_last_error"] = str(exc)
                else:
                    self._advance_candidates()
                continue
            try:
                if self._thread_setup is not None:
                    # CUDA current-device is thread-local.
                    self._thread_setup()
                moved = self._execute(desired)
            except Exception as exc:  # never take down the target worker
                with self._state_lock:
                    self.metrics["prefetch_async_failed"] += 1
                    self.metrics["prefetch_async_last_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                if self._on_error is not None:
                    self._on_error(f"{type(exc).__name__}: {exc}")
                continue
            else:
                self.metrics["prefetch_async_completed"] += 1
                self.metrics["prefetch_async_moved"] += moved
            finally:
                self._decrement_pending()

    def _retire(self, ticket):
        if self.transfers is not None:
            self.transfers.retire(ticket)
        else:
            ticket.retired = True
            ticket.copy_state = "invalid-draining"

    def _execute(self, desired: frozenset) -> int:
        # Background publishes candidates only. Formal cache ownership is
        # transferred atomically by admission and installed by the executor.
        residency = self.residency
        stale = []
        with residency.lock:
            for pid, ticket in list(residency.candidates.items()):
                cache = self._get_cache(pid)
                matches = cache is not None and ticket.matches(residency.snapshot_locked(pid, cache))
                if pid not in desired or not matches:
                    stale.append(residency.candidates.pop(pid))
            to_gpu = sorted(desired - residency.resident - residency.active - residency.reserved)
        for ticket in stale:
            self._retire(ticket)
        moved = 0
        for pid in to_gpu:
            if self._is_superseded(desired):
                self.metrics["prefetch_async_superseded"] += 1
                break
            with residency.lock:
                if pid in residency.active or pid in residency.reserved or pid in residency.candidates:
                    continue
                cache = self._get_cache(pid)
                if cache is None:
                    self.metrics["prefetch_async_missing_cache"] += 1
                    continue
                snapshot = residency.snapshot_locked(pid, cache)
            # Buffer preparation and submission never hold the registry lock.
            # Production submit_h2d(prefetch) only prepares; no DMA until the
            # ticket is published, so a concurrent demand cannot double-copy.
            if self.transfers is not None:
                ticket = self.transfers.submit_h2d(snapshot, purpose="prefetch")
                if ticket is None:
                    continue
            else:
                ticket = KVTransferTicket(snapshot, "prefetch",
                    destination_cache=self._move_cache(cache, self._gpu_device),
                    source_lease=cache, copy_state="ready")
            with residency.lock:
                current = self._get_cache(pid)
                valid = (not self._is_superseded(desired)
                         and pid not in residency.active and pid not in residency.reserved
                         and current is cache
                         and ticket.matches(residency.snapshot_locked(pid, current)))
                if valid:
                    residency.candidates[pid] = ticket
                    moved += 1
            if not valid:
                self._retire(ticket)
                self.metrics["prefetch_async_skipped_stale"] += 1
                if self._is_superseded(desired):
                    self.metrics["prefetch_async_superseded"] += 1
        self._advance_candidates()
        return moved

    def _advance_candidates(self):
        if self.transfers is None:
            return
        self.transfers.poll_completed()
        with self.residency.lock:
            candidates = list(self.residency.candidates.values())
        for ticket in candidates:
            self.transfers.advance_prefetch(ticket)


# ---------------------------------------------------------------------------
# Single model executor
# ---------------------------------------------------------------------------
@dataclass
class ExecutionOutcome:
    """Raw result of one admitted plan; never mutates scheduler state.

    A plan may be produced in phases (Verify then Prefill).  An intermediate
    phase is marked ``final=False``; the control owner applies it, delivers
    the already-available Verify responses, and then acknowledges the stage so
    the executor can start Prefill.  This preserves the "Verify response is
    not delayed until Prefill completes" ordering while keeping every
    scheduler mutation on the control owner.
    """

    job: Any
    verify_responses: dict = field(default_factory=dict)
    prefill_responses: dict = field(default_factory=dict)
    phase: str = "complete"
    final: bool = True
    error: Optional[BaseException] = None


@dataclass
class ExecutorResult:
    """One phase handed back to the control owner, plus its release gate."""

    outcome: ExecutionOutcome
    final: bool
    ack: Optional[threading.Event] = None


class SinglePlanExecutor:
    """One background thread; at most one admitted plan in flight.

    Lifecycle contract:

    * ``stop()`` asks the thread to exit.  A pending phase acknowledgement is
      abandoned rather than waited on forever, and no subsequent model phase
      may start after stop is observed.
    * ``abort()`` is the fatal path (control-side exception).  It is the same
      signal but named explicitly at call sites to document that a partially
      applied plan must not continue.
    * Graceful shutdown is the caller's job: drain/ack the current plan first,
      then stop.  This class never fabricates an acknowledgement.
    """

    def __init__(
        self,
        run_job: Callable[[Any], ExecutionOutcome],
        name: str = "fastsd-target-executor",
        thread_setup: Optional[Callable[[], None]] = None,
        wake_event: Optional[threading.Event] = None,
    ) -> None:
        self._run_job = run_job
        self._name = name
        self._thread_setup = thread_setup
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue = NotifyingQueue(wake_event) if wake_event is not None else queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._outstanding = 0
        self._outstanding_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name=self._name
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def abort(self) -> None:
        """Fatal stop: do not begin the next phase of a partial plan."""

        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def busy(self) -> bool:
        with self._outstanding_lock:
            return self._outstanding > 0

    def submit(self, job: Any) -> None:
        with self._outstanding_lock:
            if self._outstanding:
                raise RuntimeError("single target executor is already busy")
            self._outstanding = 1
        self._jobs.put_nowait(job)

    def poll_result(self) -> Optional[ExecutorResult]:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        if result.final:
            with self._outstanding_lock:
                self._outstanding = 0
        return result

    def wait_result(self, timeout: Optional[float] = None) -> Optional[ExecutorResult]:
        """Block for the next phase (used during graceful shutdown)."""

        try:
            result = self._results.get(timeout=timeout)
        except queue.Empty:
            return None
        if result.final:
            with self._outstanding_lock:
                self._outstanding = 0
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._thread_setup is not None:
                    # CUDA's current device is thread-local; bind this
                    # executor thread before any model/tensor operation.
                    self._thread_setup()
                produced = self._run_job(job)
                if isinstance(produced, ExecutionOutcome):
                    self._results.put(ExecutorResult(produced, final=True))
                    continue
                # Generator jobs produce Verify and Prefill phases.  Wait for
                # the control owner to apply each non-final phase before
                # asking the model for the next one.
                for outcome in produced:
                    if outcome.error is not None:
                        self._results.put(ExecutorResult(outcome, final=True))
                        break
                    ack = threading.Event()
                    self._results.put(
                        ExecutorResult(outcome, final=outcome.final, ack=ack)
                    )
                    if not outcome.final:
                        # Also observe stop so a control-owner failure cannot
                        # leave this thread blocked on an acknowledgement that
                        # will never arrive.
                        while not ack.wait(0.1):
                            if self._stop.is_set():
                                return
                        # A stop that raced the acknowledgement still wins:
                        # never begin the next model phase after stop/abort.
                        if self._stop.is_set():
                            return
                    if outcome.final:
                        break
            except BaseException as exc:  # surfaced to the control owner
                self._results.put(
                    ExecutorResult(ExecutionOutcome(job=job, error=exc), final=True)
                )


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
class TargetSchedulerRuntime:
    """Single-owner control loop around the executor and prefetcher.

    Callbacks are supplied by the engine (and by tests) so the loop itself can
    be exercised without loading a model:

    ``poll_ingress(timeout)``
        Drain ingress, applying control messages.  ``timeout=None`` blocks
        until at least one message arrives.  Returns the number of state
        changes; raise :class:`ShutdownRequested` on the ``None`` sentinel.
    ``build_plan()``
        Build an admission plan from the current stable queue snapshot.
    ``submit_plan(plan)``
        Atomically reserve the plan and hand it to the executor.
    ``apply_outcome(outcome)``
        Apply completions/responses and remove terminated WorkItems.
    ``refresh_prefetch()``
        Recompute the lookahead residency intent from current state.
    ``terminate_unrunnable()``
        Optional.  During graceful shutdown, explicitly finish work that can
        never run (for example a Prefill still waiting for its timing update)
        and return how many items were terminated.  Shutdown must never
        silently leave accepted work in the queues or wait forever for a
        control message that will not arrive.
    """

    def __init__(
        self,
        *,
        executor: SinglePlanExecutor,
        queues: SchedulerQueues,
        poll_ingress: Callable[[Optional[float]], int],
        build_plan: Callable[[], Any],
        submit_plan: Callable[[Any], None],
        apply_outcome: Callable[[ExecutionOutcome], None],
        refresh_prefetch: Callable[[], None],
        metrics: dict,
        executor_poll_timeout: float = 0.01,
        shutdown_timeout: Optional[float] = None,
        terminate_unrunnable: Optional[Callable[[], int]] = None,
        wake_event: Optional[threading.Event] = None,
        maintenance: Optional[Callable[[], None]] = None,
    ) -> None:
        self.executor = executor
        self.queues = queues
        self._poll_ingress = poll_ingress
        self._build_plan = build_plan
        self._submit_plan = submit_plan
        self._apply_outcome = apply_outcome
        self._refresh_prefetch = refresh_prefetch
        self._terminate_unrunnable = terminate_unrunnable
        self.metrics = metrics
        self.executor_poll_timeout = executor_poll_timeout
        self.shutdown_timeout = shutdown_timeout
        self._shutdown = False
        self._wake_event = wake_event
        self._maintenance = maintenance

    def run(self) -> None:
        try:
            self._run_loop()
        except ShutdownRequested:
            self._shutdown = True
            self._drain_on_shutdown()
        except BaseException:
            # Fatal control-side failure: a partially applied plan must not
            # continue into its next model phase.  The engine joins the
            # executor in its own finally block.
            self.executor.abort()
            raise

    def _run_loop(self) -> None:
        while True:
            if self._wake_event is not None:
                # Clear before observing either queue. A racing producer then
                # leaves the event set; no completion/ingress wakeup is lost.
                self._wake_event.clear()
            if self._maintenance is not None:
                self._maintenance()
            stage = self.executor.poll_result()
            if stage is not None:
                self._apply_stage(stage)
                continue

            if self.executor.busy():
                # Forward in flight: keep accepting requests so the lookahead
                # can see B while A is still computing, and notice shutdown.
                if self._poll_ingress(0 if self._wake_event is not None else self.executor_poll_timeout):
                    self._refresh_prefetch()
                elif self._wake_event is not None:
                    self._wake_event.wait(1.0)
                continue

            # Executor idle.  Drain a bounded batch first, then *always* try
            # to admit already-ready work.  Scheduling must not depend on the
            # ingress queue becoming empty, otherwise a continuous stream of
            # requests can starve every model submission.
            if self._poll_ingress(0):
                self._refresh_prefetch()
            plan = self._build_plan()
            if plan is not None and plan.selected_work_ids:
                self._submit_plan(plan)
                self._refresh_prefetch()
                continue
            if plan is not None:
                self.metrics["empty_plans"] += 1
                if self.queues.has_live_items():
                    self.metrics["empty_plans_with_live_work"] += 1
            # Nothing runnable: block until ingress/control arrives.  This is
            # an event wait, not a poll, so finished-work leftovers can never
            # spin the planner.
            self.metrics["idle_waits"] += 1
            if self._wake_event is not None:
                self._wake_event.wait(1.0)
            else:
                self._poll_ingress(None)

    def _apply_stage(self, stage: ExecutorResult) -> None:
        """Apply one phase; only acknowledge it if application succeeded."""

        try:
            self._apply_outcome(stage.outcome)
        except BaseException:
            # Do not release the phase gate: the executor must not start the
            # next model phase after a control-side failure.
            self.executor.abort()
            raise
        if stage.ack is not None:
            stage.ack.set()
        self._refresh_prefetch()

    def _drain_on_shutdown(self) -> None:
        """Graceful shutdown: complete or explicitly terminate accepted work.

        Ingress is closed once the ``None`` sentinel is observed, but every
        request already accepted into the scheduler queues is still run (or
        terminated with an explicit error when it can never become runnable).
        An in-flight Verify response is applied exactly once; a pending
        Prefill phase is acknowledged so the current plan can finish.
        """

        while True:
            stage = self.executor.poll_result()
            if stage is None and self.executor.busy():
                # The model call is not interruptible; wait for its phase.
                stage = self.executor.wait_result(timeout=self.shutdown_timeout)
                if stage is None:
                    self.metrics["shutdown_inflight_timeout"] += 1
                    # A configured timeout means we cannot wait for the
                    # forward; make the remaining accepted work terminal
                    # instead of leaving it queued forever.
                    self._terminate_remaining()
                    return
            if stage is not None:
                try:
                    self._apply_outcome(stage.outcome)
                except BaseException:
                    self.executor.abort()
                    raise
                if stage.ack is not None:
                    stage.ack.set()
                continue

            try:
                plan = self._build_plan()
                if plan is not None and plan.selected_work_ids:
                    self._submit_plan(plan)
                    self.metrics["shutdown_drained_plans"] += 1
                    continue
            except BaseException:
                # Make accepted-but-unfinished work terminal before surfacing
                # a scheduling failure during shutdown.
                self._terminate_remaining()
                raise
            self._terminate_remaining()
            return

    def _terminate_remaining(self) -> int:
        if self._terminate_unrunnable is None:
            return 0
        terminated = int(self._terminate_unrunnable() or 0)
        self.metrics["shutdown_terminated_work_items"] += terminated
        return terminated
