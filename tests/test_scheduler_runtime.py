import queue
import threading
import time
import types
import unittest
from collections import defaultdict

from src.fastsd_scheduler import (
    WorkItem,
    commit_admission_plan,
    plan_iteration,
    reserve_admission_plan,
)
from src.scheduler_runtime import (
    CacheResidency,
    ExecutionOutcome,
    KVPrefetchWorker,
    PrefetchCoordinator,
    SchedulerQueues,
    ShutdownRequested,
    SinglePlanExecutor,
    TargetSchedulerRuntime,
    cancel_prefill_work_items,
    commit_admission_progress,
    invoke_no_grad,
    make_forward_job_runner,
    set_thread_device,
    terminate_pending_work_items,
)


def runtime_metrics():
    metrics = defaultdict(int)
    metrics["prefetch_async_last_error"] = None
    return metrics


def make_item(work_id, task_type, total, *, category="short", state="ready", **request):
    request = dict(request)
    request.setdefault("current_time", 0.0)
    request.setdefault("proc_id", work_id)
    request.setdefault("task_type", task_type)
    request.setdefault("gamma", total if task_type == "verify" else 0)
    return WorkItem(
        work_id=work_id,
        proc_id=work_id,
        task_type=task_type,
        category=category,
        request=request,
        total_tokens=total,
        state=state,
    )


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class _AutoResponseQueues:
    def __init__(self, factory):
        self._factory = factory

    def __getitem__(self, key):
        return self._factory(key)


class FakeCache:
    def __init__(self, pid, seq_len, device="cpu"):
        self.pid = pid
        self.seq_len = seq_len
        self.device = device

    def get_seq_length(self):
        return self.seq_len


class SchedulerQueueTests(unittest.TestCase):
    def test_sort_prunes_finished_and_keeps_chunked_and_timing_gated_work(self):
        metrics = runtime_metrics()
        queues = SchedulerQueues(metrics)
        done = make_item("v-done", "verify", 4)
        done.finished = True
        done.state = "finished"
        live_verify = make_item("v-live", "verify", 4)
        chunked = make_item("p-chunk", "prefill", 20)
        chunked.cursor = 4
        timing = make_item("p-timing", "prefill", 20)
        timing.request["timing_required"] = True
        timing.request["timing_ready"] = False
        for item in (done, live_verify, chunked, timing):
            queues.enqueue(item)

        for _ in range(3):
            queues.sort({}, now=time.monotonic(), category_for=lambda item: "short", priority=lambda r, s, n: 0.0)
            self.assertFalse(any(item is done for item in queues.iter_items()))

        remaining = {item.work_id for item in queues.iter_items()}
        self.assertEqual(remaining, {"v-live", "p-chunk", "p-timing"})
        self.assertEqual(metrics["finished_work_items_removed"], 1)
        self.assertTrue(queues.has_live_items())
        self.assertTrue(queues.has_live_task("verify"))

    def test_partial_prefill_commit_marks_dirty_and_rebuckets(self):
        metrics = runtime_metrics()
        queues = SchedulerQueues(metrics)
        item = make_item("P", "prefill", 600, category="long")
        queues.enqueue(item)

        def category_for(work):
            if work.remaining_tokens <= 128:
                return "short"
            if work.remaining_tokens <= 512:
                return "mid"
            return "long"

        queues.sort({}, 0.0, category_for, lambda *args: 0.0)
        expected = [(472, "mid"), (344, "mid"), (216, "mid"), (88, "short")]
        for remaining, expected_category in expected:
            plan = plan_iteration(
                queues.snapshot(),
                token_budget=128,
                max_num_seqs=1,
                min_prefill_chunk_tokens=16,
                prefill_chunk_quantum=128,
            )
            self.assertEqual(plan.selected_work_ids, ["P"])
            reserve_admission_plan(plan)
            commit_admission_progress(
                plan, queues, commit_admission_plan
            )
            # A chunk changes remaining_tokens without enqueueing anything.
            self.assertTrue(queues.dirty)
            self.assertEqual(item.remaining_tokens, remaining)
            queues.sort({}, 0.0, category_for, lambda *args: 0.0)
            self.assertEqual(item.category, expected_category)
            self.assertIn(item, queues.snapshot()["prefill"][expected_category])

    def test_remove_work_ids_preserves_other_work_on_same_pid(self):
        item_a = make_item("a", "verify", 4)
        item_a.proc_id = "session"
        item_b = make_item("b", "verify", 4)
        item_b.proc_id = "session"
        queues = SchedulerQueues(runtime_metrics())
        queues.enqueue(item_a)
        queues.enqueue(item_b)

        self.assertEqual(queues.remove_work_ids(["a"]), 1)
        self.assertEqual([item.work_id for item in queues.iter_items()], ["b"])

    def test_sort_recategorizes_chunked_prefill_by_remaining_tokens(self):
        metrics = runtime_metrics()
        queues = SchedulerQueues(metrics)
        item = make_item("p", "prefill", 600, category="long")
        item.cursor = 550  # remaining 50 -> short
        queues.enqueue(item)

        queues.sort(
            {},
            now=0.0,
            category_for=lambda work: "short" if work.remaining_tokens <= 128 else "long",
            priority=lambda r, s, n: 0.0,
        )
        self.assertEqual(item.category, "short")
        self.assertEqual(queues.snapshot()["prefill"]["short"], [item])

    def test_cancel_defers_reserved_work_and_removes_queued_work(self):
        work_items = {}
        queues = SchedulerQueues(runtime_metrics())
        reserved = make_item("p-reserved", "prefill", 16, state="reserved")
        queued = make_item("p-queued", "prefill", 16)
        other_task = make_item("p-other", "prefill", 16)
        other_pid = make_item("z", "prefill", 16)
        for item in (reserved, queued, other_task):
            item.proc_id = "p-reserved"
        other_pid.proc_id = "other"
        for item in (reserved, queued, other_task, other_pid):
            work_items[item.work_id] = item
            queues.enqueue(item)
        reserved.request["task_id"] = "t1"
        queued.request["task_id"] = "t1"
        other_task.request["task_id"] = "t2"

        removed, deferred = cancel_prefill_work_items(
            work_items=work_items, queues=queues, proc_id="p-reserved", task_id="t1"
        )

        # Matching task ids are removed; the reserved one is deferred.
        self.assertEqual(removed, {"p-reserved", "p-queued"})
        self.assertEqual(deferred, 1)
        self.assertTrue(reserved.cancel_pending)
        self.assertFalse(reserved.finished)
        self.assertTrue(queued.finished)
        self.assertEqual(set(work_items), {"p-other", "z"})
        self.assertEqual({i.work_id for i in queues.iter_items()}, {"p-other", "z"})


class PrefetchCoordinatorTests(unittest.TestCase):
    def _coordinator(self, submitted, metrics, resident=(), protected=()):
        resident_set = set(resident)
        protected_set = set(protected)
        return PrefetchCoordinator(
            submit=lambda desired: submitted.append(set(desired)) or True,
            resident_snapshot=lambda: set(resident_set),
            protected_snapshot=lambda: set(protected_set),
            metrics=metrics,
        )

    def test_empty_lookahead_makes_zero_submissions_and_no_eviction(self):
        metrics = runtime_metrics()
        submitted = []
        coordinator = self._coordinator(submitted, metrics, resident={"a"})

        self.assertFalse(coordinator.refresh([]))
        self.assertEqual(submitted, [])
        self.assertEqual(metrics["prefetch_empty_desired"], 1)
        self.assertEqual(metrics["prefetch_evictions"], 0)

    def test_unchanged_desired_is_not_resubmitted(self):
        metrics = runtime_metrics()
        submitted = []
        coordinator = self._coordinator(submitted, metrics, resident={"a"})

        self.assertTrue(coordinator.refresh(["b"]))
        self.assertFalse(coordinator.refresh(["b"]))
        self.assertEqual(submitted, [{"b"}])
        self.assertEqual(metrics["prefetch_desired_unchanged"], 1)

    def test_reserved_pids_are_excluded_from_targets_and_eviction(self):
        metrics = runtime_metrics()
        submitted = []
        coordinator = self._coordinator(
            submitted, metrics, resident={"a", "b"}, protected={"a"}
        )

        coordinator.refresh(["a", "b", "c"])

        self.assertEqual(submitted, [{"b", "c"}])
        self.assertEqual(metrics["prefetch_reserved_excluded"], 1)
        self.assertEqual(metrics["prefetch_evictions"], 0)  # b stays desired

    def test_satisfied_desired_set_avoids_submission(self):
        metrics = runtime_metrics()
        submitted = []
        coordinator = self._coordinator(submitted, metrics, resident={"a"})

        self.assertFalse(coordinator.refresh(["a"]))
        self.assertEqual(submitted, [])
        self.assertEqual(metrics["prefetch_noop"], 1)


class KVPrefetchWorkerTests(unittest.TestCase):
    def _worker(self, store, metrics, move=None, residency=None):
        residency = residency or CacheResidency()

        def default_move(cache, device):
            if cache.device == device:
                return cache
            return FakeCache(cache.pid, cache.seq_len, device)

        return KVPrefetchWorker(
            residency=residency,
            get_cache=store.get,
            set_cache=store.__setitem__,
            move_cache=move or default_move,
            gpu_device="cuda",
            metrics=metrics,
        ), residency

    def test_loads_desired_and_evicts_stale_caches(self):
        metrics = runtime_metrics()
        a = FakeCache("a", 8, "cuda")
        b = FakeCache("b", 8, "cpu")
        store = {"a": a, "b": b}
        worker, residency = self._worker(store, metrics)
        residency.resident.add("a")
        worker.start()
        try:
            worker.submit({"b"})
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_completed"] == 1)
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertEqual(store["a"].device, "cpu")
        self.assertEqual(store["b"].device, "cuda")
        self.assertEqual(residency.resident, {"b"})
        self.assertEqual(metrics["prefetch_async_moved"], 2)

    def test_copy_racing_a_generation_bump_is_not_installed(self):
        metrics = runtime_metrics()
        cache = FakeCache("p", 5, "cpu")
        store = {"p": cache}
        move_started = threading.Event()
        release = threading.Event()

        def blocking_move(source, device):
            move_started.set()
            release.wait(2.0)
            return FakeCache(source.pid, source.seq_len, device)

        worker, residency = self._worker(store, metrics, move=blocking_move)
        worker.start()
        try:
            worker.submit({"p"})
            self.assertTrue(move_started.wait(2.0))
            # The main/executor side mutates or replaces this cache while the
            # copy is in flight.
            residency.bump(["p"])
            release.set()
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_completed"] == 1)
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertEqual(store["p"].device, "cpu")
        self.assertNotIn("p", residency.resident)
        self.assertEqual(metrics["prefetch_async_moved"], 0)
        self.assertEqual(metrics["prefetch_async_skipped_stale"], 1)

    def test_replaced_cache_object_is_not_overwritten(self):
        metrics = runtime_metrics()
        original = FakeCache("p", 5, "cpu")
        store = {"p": original}
        move_started = threading.Event()
        release = threading.Event()

        def blocking_move(source, device):
            move_started.set()
            release.wait(2.0)
            return FakeCache(source.pid, source.seq_len, device)

        worker, residency = self._worker(store, metrics, move=blocking_move)
        worker.start()
        try:
            worker.submit({"p"})
            self.assertTrue(move_started.wait(2.0))
            replacement = FakeCache("p", 6, "cpu")
            store["p"] = replacement
            release.set()
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_completed"] == 1)
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertIs(store["p"], replacement)
        self.assertEqual(store["p"].device, "cpu")
        self.assertNotIn("p", residency.resident)
        self.assertEqual(metrics["prefetch_async_skipped_stale"], 1)

    def test_same_length_in_place_mutation_is_not_overwritten(self):
        metrics = runtime_metrics()
        cache = FakeCache("p", 5, "cpu")
        store = {"p": cache}
        move_started = threading.Event()
        release = threading.Event()

        def blocking_move(source, device):
            move_started.set()
            release.wait(2.0)
            return FakeCache(source.pid, source.seq_len, device)

        worker, residency = self._worker(store, metrics, move=blocking_move)
        worker.start()
        try:
            worker.submit({"p"})
            self.assertTrue(move_started.wait(2.0))
            cache.seq_len = 7  # in-place append without object replacement
            release.set()
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_completed"] == 1)
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertIs(store["p"], cache)
        self.assertEqual(store["p"].device, "cpu")
        self.assertEqual(metrics["prefetch_async_skipped_stale"], 1)

    def test_active_or_reserved_pid_is_never_moved(self):
        metrics = runtime_metrics()
        cache = FakeCache("p", 5, "cpu")
        store = {"p": cache}
        worker, residency = self._worker(store, metrics)
        residency.reserve(["p"])
        worker.start()
        try:
            worker.submit({"p"})
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_completed"] == 1)
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertEqual(store["p"].device, "cpu")
        self.assertEqual(metrics["prefetch_async_moved"], 0)

    def test_pending_clears_when_consumer_finishes_before_publisher_returns(self):
        class Cache:
            def get_seq_length(self):
                return 2

        caches = {"B": Cache()}
        metrics = runtime_metrics()
        worker = KVPrefetchWorker(
            residency=CacheResidency(),
            get_cache=caches.get,
            set_cache=caches.__setitem__,
            move_cache=lambda cache, device: Cache(),
            gpu_device="fake-cuda",
            metrics=metrics,
        )

        class PauseAfterPublish(queue.Queue):
            def put_nowait(self, value):
                super().put_nowait(value)
                # Simulate the producer being descheduled right after
                # publishing until the consumer has completed the item.
                # With correct accounting the pending slot is reserved
                # before publishing, so this can only end after the worker
                # releases it.  A post-publish increment leaves it stuck.
                deadline = time.monotonic() + 2.0
                while worker.has_pending() and time.monotonic() < deadline:
                    time.sleep(0.001)

        worker._tasks = PauseAfterPublish(maxsize=1)
        worker.start()
        try:
            self.assertTrue(worker.submit({"B"}))
            self.assertEqual(metrics["prefetch_async_completed"], 1)
            self.assertFalse(worker.has_pending())
            self.assertEqual(metrics["prefetch_pending_underflow"], 0)
        finally:
            worker.stop()
            worker.join(2.0)
        self.assertFalse(worker.is_alive())

    def test_newer_intent_supersedes_an_inflight_copy(self):
        metrics = runtime_metrics()
        a = FakeCache("a", 5, "cpu")
        b = FakeCache("b", 5, "cpu")
        store = {"a": a, "b": b}
        move_started = threading.Event()
        release = threading.Event()

        def blocking_move(source, device):
            if source.pid == "a":
                move_started.set()
                release.wait(2.0)
            return FakeCache(source.pid, source.seq_len, device)

        worker, residency = self._worker(store, metrics, move=blocking_move)
        worker.start()
        try:
            worker.submit({"a"})
            self.assertTrue(move_started.wait(2.0))
            worker.submit({"b"})  # supersedes the in-flight intent
            release.set()
            self.assertTrue(
                wait_until(lambda: store["b"].device == "cuda")
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertEqual(store["a"].device, "cpu")
        self.assertNotIn("a", residency.resident)
        self.assertIn("b", residency.resident)
        self.assertGreaterEqual(metrics["prefetch_async_superseded"], 1)

    def test_move_failure_is_recorded_and_worker_survives(self):
        metrics = runtime_metrics()
        store = {"p": FakeCache("p", 5, "cpu")}
        attempts = {"n": 0}

        def flaky_move(source, device):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("copy failed")
            return FakeCache(source.pid, source.seq_len, device)

        worker, residency = self._worker(store, metrics, move=flaky_move)
        worker.start()
        try:
            worker.submit({"p"})
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_failed"] == 1)
            )
            worker.submit({"p"})
            self.assertTrue(
                wait_until(lambda: metrics["prefetch_async_completed"] == 1)
            )
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertIn("copy failed", metrics["prefetch_async_last_error"])
        self.assertEqual(store["p"].device, "cuda")


class ShutdownTerminationTests(unittest.TestCase):
    def test_terminates_each_residual_item_once_with_a_reason(self):
        class Collector:
            def __init__(self):
                self.payloads = []

            def put(self, payload):
                self.payloads.append(payload)

        metrics = runtime_metrics()
        queues = SchedulerQueues(metrics)
        work_items = {}
        verify = make_item("v", "verify", 4)
        timing = make_item("p", "prefill", 8)
        timing.request["timing_required"] = True
        timing.request["timing_ready"] = False
        done = make_item("done", "verify", 4)
        done.finished = True
        done.state = "finished"
        for item in (verify, timing, done):
            work_items[item.work_id] = item
            queues.enqueue(item)
        responses = defaultdict(Collector)

        removed = terminate_pending_work_items(
            work_items=work_items,
            queues=queues,
            response_queues=responses,
        )

        self.assertEqual(removed, 3)
        self.assertIn("shutdown", responses["v"].payloads[0]["error"])
        self.assertIn("timing update", responses["p"].payloads[0]["error"])
        self.assertEqual(responses["done"].payloads, [])
        self.assertEqual(work_items, {})
        self.assertFalse(queues.has_items())

        self.assertEqual(
            terminate_pending_work_items(
                work_items=work_items, queues=queues, response_queues=responses
            ),
            0,
        )
        self.assertEqual(len(responses["v"].payloads), 1)
        self.assertEqual(len(responses["p"].payloads), 1)


class PrefetchRetryTests(unittest.TestCase):
    def test_same_desired_retries_after_a_skipped_copy(self):
        metrics = runtime_metrics()
        submitted = []
        pending = {"value": False}
        coordinator = PrefetchCoordinator(
            submit=lambda desired: submitted.append(set(desired)) or True,
            resident_snapshot=lambda: set(),
            protected_snapshot=lambda: set(),
            metrics=metrics,
            pending_snapshot=lambda: pending["value"],
            retry_interval=0.0,
        )

        self.assertTrue(coordinator.refresh(["p"]))
        pending["value"] = True
        self.assertFalse(coordinator.refresh(["p"]))
        self.assertEqual(metrics["prefetch_intent_inflight"], 1)
        pending["value"] = False
        self.assertTrue(coordinator.refresh(["p"]))
        self.assertEqual(submitted, [{"p"}, {"p"}])
        self.assertEqual(metrics["prefetch_retry"], 1)

    def test_retry_is_throttled_until_the_interval_elapses(self):
        metrics = runtime_metrics()
        submitted = []
        clock = {"now": 100.0}
        coordinator = PrefetchCoordinator(
            submit=lambda desired: submitted.append(set(desired)) or True,
            resident_snapshot=lambda: set(),
            protected_snapshot=lambda: set(),
            metrics=metrics,
            pending_snapshot=lambda: False,
            retry_interval=0.05,
            clock=lambda: clock["now"],
        )

        self.assertTrue(coordinator.refresh(["p"]))
        clock["now"] = 100.01
        self.assertFalse(coordinator.refresh(["p"]))
        self.assertEqual(metrics["prefetch_retry_throttled"], 1)
        clock["now"] = 100.1
        self.assertTrue(coordinator.refresh(["p"]))
        self.assertEqual(len(submitted), 2)


class ExecutorLifecycleTests(unittest.TestCase):
    def test_stop_unblocks_unconsumed_phase_ack_without_advancing(self):
        advanced = threading.Event()

        def run_job(job):
            yield ExecutionOutcome(job=job, phase="verify", final=False)
            advanced.set()
            yield ExecutionOutcome(job=job)

        executor = SinglePlanExecutor(run_job)
        executor.start()
        try:
            executor.submit(object())
            stage = executor.wait_result(2.0)
            self.assertIsNotNone(stage)
            self.assertFalse(stage.final)
            executor.stop()
            executor.join(1.0)
            self.assertFalse(executor._thread.is_alive())
            self.assertFalse(advanced.is_set())
        finally:
            stage.ack.set()
            executor.stop()
            executor.join(1.0)

    def test_abort_also_prevents_the_next_phase(self):
        advanced = threading.Event()

        def run_job(job):
            yield ExecutionOutcome(job=job, phase="verify", final=False)
            advanced.set()
            yield ExecutionOutcome(job=job)

        executor = SinglePlanExecutor(run_job)
        executor.start()
        try:
            executor.submit(object())
            stage = executor.wait_result(2.0)
            executor.abort()
            executor.join(1.0)
            self.assertFalse(executor._thread.is_alive())
            self.assertFalse(advanced.is_set())
        finally:
            stage.ack.set()
            executor.stop()
            executor.join(1.0)

    def test_join_waits_for_non_interruptible_work_past_legacy_timeout(self):
        forward_started = threading.Event()
        forward_release = threading.Event()
        copy_started = threading.Event()
        copy_release = threading.Event()

        def run_job(job):
            forward_started.set()
            forward_release.wait(10.0)
            yield ExecutionOutcome(job=job)

        class Cache:
            def get_seq_length(self):
                return 1

        caches = {"p": Cache()}
        metrics = runtime_metrics()

        def blocking_move(cache, device):
            copy_started.set()
            copy_release.wait(10.0)
            return Cache()

        executor = SinglePlanExecutor(run_job)
        worker = KVPrefetchWorker(
            residency=CacheResidency(),
            get_cache=caches.get,
            set_cache=caches.__setitem__,
            move_cache=blocking_move,
            gpu_device="fake-cuda",
            metrics=metrics,
        )
        executor.start()
        worker.start()
        try:
            executor.submit(object())
            self.assertTrue(forward_started.wait(2.0))
            worker.submit({"p"})
            self.assertTrue(copy_started.wait(2.0))

            executor.stop()
            worker.stop()
            executor_joiner = threading.Thread(target=executor.join)
            worker_joiner = threading.Thread(target=worker.join)
            executor_joiner.start()
            worker_joiner.start()

            # Both calls are non-interruptible; join must wait past the old
            # 2-second cutoff rather than returning with live threads.
            time.sleep(2.2)
            self.assertTrue(executor_joiner.is_alive())
            self.assertTrue(worker_joiner.is_alive())
            self.assertTrue(executor.is_alive())
            self.assertTrue(worker.is_alive())

            forward_release.set()
            copy_release.set()
            executor_joiner.join(2.0)
            worker_joiner.join(2.0)
            self.assertFalse(executor.is_alive())
            self.assertFalse(worker.is_alive())
        finally:
            forward_release.set()
            copy_release.set()
            executor.stop()
            executor.join(2.0)
            worker.stop()
            worker.join(2.0)

    def test_cpu_device_setup_does_not_touch_cuda(self):
        calls = []
        self.assertFalse(set_thread_device("cpu", set_device=calls.append))
        self.assertEqual(calls, [])

    def test_cuda_device_setup_binds_the_target_device(self):
        class FakeCudaDevice:
            type = "cuda"

        calls = []
        device = FakeCudaDevice()
        self.assertTrue(set_thread_device(device, set_device=calls.append))
        self.assertEqual(calls, [device])


class _RuntimeHarness:
    """Minimal engine-shaped wiring for the real runtime control loop."""

    def __init__(self, *, token_budget=16, max_num_seqs=2):
        self.metrics = runtime_metrics()
        self.queues = SchedulerQueues(self.metrics)
        self.work_items = {}
        self.accept_stats = {}
        self.state = {"wrr_cursor": 0, "current_cycle": 0}
        self.in_flight = {"plan": None}
        self.requests = queue.Queue()
        self.responses = {}
        self.residency = CacheResidency()
        self.cache_store = {}
        self.moves = []
        self.token_budget = token_budget
        self.max_num_seqs = max_num_seqs
        self.min_prefill = 8
        self.quantum = 8
        self.ingress_limit = 64
        self.category_for = lambda item: item.category
        self.fail_phase = None
        self.poll_error = None
        self.fail_forward = False
        self.first_submit_qsize = None
        self.runtime_error = None

        self.verify_gate = threading.Event()
        self.verify_gate.set()
        self.verify_started = threading.Event()
        self.prefill_gate = threading.Event()
        self.prefill_gate.set()
        self.prefill_started = threading.Event()
        self.events = []
        # Use the production phase protocol, not a test re-implementation.
        self.executor = SinglePlanExecutor(
            make_forward_job_runner(self._run_batch)
        )
        self.prefetch = KVPrefetchWorker(
            residency=self.residency,
            get_cache=self.cache_store.get,
            set_cache=self.cache_store.__setitem__,
            move_cache=self._move_cache,
            gpu_device="cuda",
            metrics=self.metrics,
        )
        self.coordinator = PrefetchCoordinator(
            submit=self.prefetch.submit,
            resident_snapshot=self.residency.resident_snapshot,
            protected_snapshot=self.residency.protected_snapshot,
            metrics=self.metrics,
            pending_snapshot=self.prefetch.has_pending,
        )
        self.runtime = TargetSchedulerRuntime(
            executor=self.executor,
            queues=self.queues,
            poll_ingress=self._poll_ingress,
            build_plan=self._build_plan,
            submit_plan=self._submit_plan,
            apply_outcome=self._apply_outcome,
            refresh_prefetch=self._refresh_prefetch,
            metrics=self.metrics,
            executor_poll_timeout=0.005,
            terminate_unrunnable=self._terminate_unrunnable,
        )
        self.thread = threading.Thread(target=self._run_runtime, daemon=True)

    def _run_runtime(self):
        try:
            self.runtime.run()
        except BaseException as exc:
            self.runtime_error = exc

    def _move_cache(self, cache, device):
        if cache.device == device:
            return cache
        self.moves.append((cache.pid, device))
        return FakeCache(cache.pid, cache.seq_len, device)

    def _response_queue(self, pid):
        if pid not in self.responses:
            self.responses[pid] = queue.Queue()
        return self.responses[pid]

    def _poll_ingress(self, timeout):
        if self.poll_error is not None:
            raise self.poll_error
        changed = 0
        limit = self.ingress_limit
        next_timeout = timeout
        while changed < limit:
            try:
                if next_timeout is None:
                    req = self.requests.get()
                elif next_timeout <= 0:
                    req = self.requests.get_nowait()
                else:
                    req = self.requests.get(timeout=next_timeout)
            except queue.Empty:
                break
            next_timeout = 0
            if req is None:
                raise ShutdownRequested
            changed += 1
            if self.executor.busy():
                self.metrics["incoming_during_forward"] += 1
            if req.get("control_type") == "prefill_cancel":
                removed, deferred = cancel_prefill_work_items(
                    work_items=self.work_items,
                    queues=self.queues,
                    proc_id=req["proc_id"],
                    task_id=str(req.get("task_id", "")),
                )
                self.metrics["prefill_cancels"] += len(removed)
                self.metrics["prefill_cancels_deferred"] += deferred
                continue
            self._ingest(req)
        return changed

    def _ingest(self, req):
        proc_id = req["proc_id"]
        self._response_queue(proc_id)
        request = dict(req)
        request["current_time"] = time.time()
        request.setdefault("server_enqueue_monotonic", time.monotonic())
        request.setdefault("response_key", proc_id)
        total = int(req.get("total_tokens") or req.get("gamma") or 0)
        work_id = str(req.get("work_id", f"{proc_id}:{time.monotonic_ns()}"))
        request["work_id"] = work_id
        item = WorkItem(
            work_id=work_id,
            proc_id=proc_id,
            task_type=req["task_type"],
            category=req.get("category", "short"),
            request=request,
            total_tokens=total,
            base_prefix_len=int(req.get("prefix_len", 0) or 0),
            original_gamma=int(req.get("gamma", 0) or 0),
        )
        if req["task_type"] == "verify" and proc_id not in self.cache_store:
            self.cache_store[proc_id] = FakeCache(
                proc_id, item.base_prefix_len, "cpu"
            )
        self.work_items[work_id] = item
        self.queues.enqueue(item)

    def _build_plan(self, reserved_work_ids=(), cycle=None, cursor=None):
        self.queues.sort(
            self.accept_stats,
            now=time.monotonic(),
            category_for=self.category_for,
            priority=lambda r, s, n: 0.0,
        )
        return plan_iteration(
            self.queues.snapshot(),
            token_budget=self.token_budget,
            current_cycle=self.state["current_cycle"] if cycle is None else cycle,
            wrr_cursor=self.state["wrr_cursor"] if cursor is None else cursor,
            max_num_seqs=self.max_num_seqs,
            min_prefill_chunk_tokens=self.min_prefill,
            prefill_chunk_quantum=self.quantum,
            accept_stats=self.accept_stats,
            now=time.monotonic(),
            reserved_work_ids=reserved_work_ids,
        )

    def _refresh_prefetch(self):
        plan = self.in_flight["plan"]
        if plan is None and not self.executor.busy():
            # Mirror the engine: do not prefetch the plan that is about to be
            # admitted; wait until a forward is actually in flight.
            return
        if plan is not None:
            lookahead = self._build_plan(
                reserved_work_ids=plan.selected_work_ids,
                cycle=plan.next_cycle,
                cursor=plan.next_wrr_cursor,
            )
        else:
            lookahead = self._build_plan()
        self.coordinator.refresh(lookahead.verify_proc_ids)

    def _submit_plan(self, plan):
        if self.first_submit_qsize is None:
            self.first_submit_qsize = self.requests.qsize()
        reserve_admission_plan(plan)
        item_by_id = {item.work_id: item for item in plan._selected_items}
        pids = {sl.proc_id for sl in plan.verify_slices} | {
            sl.proc_id for sl in plan.prefill_slices
        }
        self.residency.reserve(pids)

        def entries(slices):
            return [
                types.SimpleNamespace(
                    proc_id=sl.proc_id,
                    task_type=sl.task_type,
                    draft_token_count=sl.draft_token_count,
                    prefix_len=item_by_id[sl.work_id].base_prefix_len,
                    offset=sl.offset,
                )
                for sl in slices
            ]

        job = types.SimpleNamespace(
            plan=plan,
            item_by_id=item_by_id,
            verify_batch=entries(plan.verify_slices),
            prefill_batch=entries(plan.prefill_slices),
            reserved_pids=pids,
            completed=set(),
            progress={},
            verify_applied=False,
        )
        self.in_flight["plan"] = plan
        self.executor.submit(job)
        self.last_job = job

    def _run_batch(self, batch):
        """Forward-only batch body used by production make_forward_job_runner."""

        pids = [entry.proc_id for entry in batch]
        self.residency.activate(pids)
        try:
            task_type = batch[0].task_type
            if self.fail_forward and task_type == "verify":
                raise RuntimeError("forward failed")
            self.events.append((task_type + "_start", tuple(pids)))
            if task_type == "verify":
                self.verify_started.set()
                self.verify_gate.wait(5.0)
            else:
                self.prefill_started.set()
                self.prefill_gate.wait(5.0)
            responses = {}
            for entry in batch:
                if entry.task_type == "verify":
                    responses[entry.proc_id] = {
                        "accepted": entry.prefix_len + entry.draft_token_count
                    }
                else:
                    responses[entry.proc_id] = {"status": "prefill_ok"}
            self.events.append((task_type + "_end", tuple(pids)))
            return responses
        finally:
            self.residency.deactivate(pids)

    def _apply_verify(self, job, responses):
        if job.verify_applied:
            return
        plan = job.plan
        for sl in plan.verify_slices:
            item = job.item_by_id[sl.work_id]
            response = responses[item.proc_id]
            accepted_chunk = max(0, int(response["accepted"]) - item.base_prefix_len)
            item.cursor += accepted_chunk
            item.accepted_so_far += accepted_chunk
            job.progress[item.work_id] = accepted_chunk
            if accepted_chunk < sl.draft_token_count or item.cursor >= item.total_tokens:
                item.finished = True
                job.completed.add(item.work_id)
                payload = dict(response)
                payload["accepted"] = item.base_prefix_len + item.accepted_so_far
                self._response_queue(item.proc_id).put(payload)
        job.verify_applied = True

    def _apply_prefill(self, job, responses):
        for sl in job.plan.prefill_slices:
            item = job.item_by_id[sl.work_id]
            item.cursor = sl.offset + sl.draft_token_count
            if getattr(item, "cancel_pending", False):
                item.finished = True
                item.state = "finished"
                job.completed.add(item.work_id)
                continue
            if item.cursor >= item.total_tokens:
                item.finished = True
                job.completed.add(item.work_id)
                self._response_queue(item.proc_id).put(responses[item.proc_id])
        for sl in job.plan.prefill_slices:
            job.progress[sl.work_id] = sl.draft_token_count

    def _terminate_unrunnable(self):
        return terminate_pending_work_items(
            work_items=self.work_items,
            queues=self.queues,
            response_queues=_AutoResponseQueues(self._response_queue),
        )

    def _apply_outcome(self, outcome):
        job = outcome.job
        plan = job.plan
        committed = False
        try:
            if self.fail_phase is not None and outcome.phase == self.fail_phase:
                raise RuntimeError("apply failed")
            if outcome.error is not None:
                from src.fastsd_scheduler import abort_admission_plan

                abort_admission_plan(plan)
                raise outcome.error
            if outcome.phase == "verify":
                self._apply_verify(job, outcome.verify_responses)
                return
            if not job.verify_applied and outcome.verify_responses:
                self._apply_verify(job, outcome.verify_responses)
            self._apply_prefill(job, outcome.prefill_responses)
            commit_admission_progress(
                plan,
                self.queues,
                commit_admission_plan,
                completed_work_ids=job.completed,
                progress=job.progress,
            )
            committed = True
            self.state["wrr_cursor"] = plan.next_wrr_cursor
            self.state["current_cycle"] = plan.next_cycle
            self.metrics["plans"] += 1
            finished = set(job.completed)
            for item in job.item_by_id.values():
                if item.finished:
                    finished.add(item.work_id)
            removed = self.queues.remove_work_ids(finished)
            self.metrics["finished_work_items_removed"] += removed
            for work_id in finished:
                self.work_items.pop(work_id, None)
            self.queues.prune_finished()
        except BaseException:
            if not committed:
                from src.fastsd_scheduler import abort_admission_plan

                abort_admission_plan(plan)
            self.in_flight["plan"] = None
            self.residency.release(job.reserved_pids)
            raise
        finally:
            if outcome.final:
                self.in_flight["plan"] = None
                self.residency.release(job.reserved_pids)

    def start(self):
        self.executor.start()
        self.prefetch.start()
        self.thread.start()

    def close(self):
        self.requests.put(None)
        self.thread.join(3.0)
        self.executor.stop()
        self.executor.join(2.0)
        self.prefetch.stop()
        self.prefetch.join(2.0)
        self.assert_threads_stopped()

    def assert_threads_stopped(self):
        assert not self.thread.is_alive(), "runtime control thread did not stop"
        assert not self.executor._thread.is_alive(), "executor thread did not stop"
        assert not self.prefetch._thread.is_alive(), "prefetch thread did not stop"


class RuntimeIntegrationTests(unittest.TestCase):
    def test_idle_runtime_blocks_without_planning_churn(self):
        harness = _RuntimeHarness()
        harness.start()
        try:
            self.assertTrue(wait_until(lambda: harness.metrics["idle_waits"] >= 1))
            time.sleep(0.15)
            self.assertEqual(harness.metrics["idle_waits"], 1)
            self.assertEqual(harness.metrics["empty_plans"], 1)
            self.assertEqual(harness.metrics["prefetch_plans"], 0)
        finally:
            harness.close()

    def test_finished_leftover_is_pruned_and_does_not_spin_empty_plans(self):
        harness = _RuntimeHarness()
        done = make_item("done", "verify", 4)
        done.finished = True
        done.state = "finished"
        harness.queues.enqueue(done)
        harness.work_items[done.work_id] = done
        harness.start()
        try:
            self.assertTrue(wait_until(lambda: harness.metrics["idle_waits"] >= 1))
            time.sleep(0.15)
            self.assertEqual(harness.metrics["idle_waits"], 1)
            self.assertEqual(harness.metrics["empty_plans"], 1)
            self.assertEqual(harness.metrics["empty_plans_with_live_work"], 0)
            self.assertEqual(harness.metrics["finished_work_items_removed"], 1)
            self.assertFalse(harness.queues.has_items())
        finally:
            harness.close()

    def test_shutdown_runs_work_accepted_before_sentinel_in_same_drain(self):
        harness = _RuntimeHarness()
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "B", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            harness.requests.put(None)
            harness.thread.join(3.0)
            self.assertFalse(harness.thread.is_alive())
            payload = harness.responses["B"].get_nowait()
            self.assertEqual(payload["accepted"], 12)
            self.assertEqual(harness.metrics["plans"], 1)
            self.assertEqual(harness.work_items, {})
            self.assertFalse(harness.queues.has_items())
        finally:
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)
            harness.assert_threads_stopped()

    def test_shutdown_drains_inflight_verify_and_pending_request(self):
        harness = _RuntimeHarness()
        harness.verify_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "A", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            self.assertTrue(harness.verify_started.wait(2.0))
            harness.requests.put(
                {"proc_id": "B", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            harness.requests.put(None)
            self.assertTrue(wait_until(lambda: harness.runtime._shutdown))
            harness.verify_gate.set()
            harness.thread.join(3.0)
            self.assertFalse(harness.thread.is_alive())
            self.assertEqual(harness.responses["A"].qsize(), 1)
            self.assertEqual(harness.responses["B"].qsize(), 1)
            self.assertEqual(harness.metrics["plans"], 2)
        finally:
            harness.verify_gate.set()
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)
            harness.assert_threads_stopped()

    def test_shutdown_finishes_a_chunked_verify_residual_once(self):
        harness = _RuntimeHarness(token_budget=4, max_num_seqs=1)
        harness.min_prefill = 4  # keep the planner's prefill floor valid
        harness.verify_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {
                    "proc_id": "V",
                    "task_type": "verify",
                    "gamma": 4,
                    "total_tokens": 8,
                    "prefix_len": 8,
                }
            )
            self.assertTrue(harness.verify_started.wait(2.0))
            harness.requests.put(None)
            self.assertTrue(wait_until(lambda: harness.runtime._shutdown))
            harness.verify_gate.set()
            harness.thread.join(3.0)
            self.assertFalse(harness.thread.is_alive())
            # The first chunk produced no response; the drain completes the
            # residual chunk and returns exactly one final response.
            self.assertEqual(harness.responses["V"].qsize(), 1)
            payload = harness.responses["V"].get_nowait()
            self.assertEqual(payload["accepted"], 16)
            self.assertEqual(harness.metrics["plans"], 2)
            self.assertEqual(harness.work_items, {})
        finally:
            harness.verify_gate.set()
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)
            harness.assert_threads_stopped()

    def test_shutdown_terminates_timing_gated_prefill_with_explicit_error(self):
        harness = _RuntimeHarness()
        harness.start()
        try:
            harness.requests.put(
                {
                    "proc_id": "P",
                    "task_type": "prefill",
                    "total_tokens": 8,
                    "prefix_len": 0,
                    "timing_required": True,
                    "timing_ready": False,
                }
            )
            self.assertTrue(wait_until(lambda: harness.metrics["idle_waits"] >= 1))
            harness.requests.put(None)
            harness.thread.join(3.0)
            self.assertFalse(harness.thread.is_alive())
            payload = harness.responses["P"].get_nowait()
            self.assertIn("timing update", payload["error"])
            self.assertEqual(harness.work_items, {})
            self.assertFalse(harness.queues.has_items())
            self.assertEqual(harness.metrics["shutdown_terminated_work_items"], 1)
        finally:
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)
            harness.assert_threads_stopped()

    def test_continuous_ingress_does_not_starve_model_submission(self):
        harness = _RuntimeHarness()
        harness.ingress_limit = 1  # bounded drains can never empty the queue
        for index in range(20):
            harness.requests.put(
                {
                    "proc_id": f"p{index}",
                    "task_type": "verify",
                    "gamma": 4,
                    "prefix_len": 8,
                }
            )
        harness.start()
        try:
            self.assertTrue(
                wait_until(lambda: harness.metrics["plans"] >= 1, 2.0),
                "a ready plan was never submitted",
            )
            # The first submission must happen while later ingress messages
            # are still queued; waiting for the queue to empty is the
            # starvation bug this test guards against.
            self.assertIsNotNone(harness.first_submit_qsize)
            self.assertGreater(harness.first_submit_qsize, 0)
        finally:
            harness.close()
        self.assertGreater(harness.metrics["plans"], 1)
        for index in range(20):
            self.assertEqual(harness.responses[f"p{index}"].qsize(), 1)

    def test_apply_error_aborts_without_starting_prefill(self):
        harness = _RuntimeHarness(token_budget=32, max_num_seqs=2)
        harness.fail_phase = "verify"
        harness.requests.put(
            {"proc_id": "V", "task_type": "verify", "gamma": 4, "prefix_len": 8}
        )
        harness.requests.put(
            {"proc_id": "P", "task_type": "prefill", "total_tokens": 8, "prefix_len": 0}
        )
        harness.executor.start()
        harness.prefetch.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "apply failed"):
                harness.runtime.run()
            harness.executor.stop()
            harness.executor.join(2.0)
            self.assertFalse(harness.executor._thread.is_alive())
            self.assertNotIn(("prefill_start", ("P",)), harness.events)
            self.assertFalse(harness.prefill_started.is_set())
        finally:
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)

    def test_ingress_error_aborts_executor(self):
        harness = _RuntimeHarness()
        harness.poll_error = RuntimeError("ingress failed")
        harness.executor.start()
        harness.prefetch.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "ingress failed"):
                harness.runtime.run()
            harness.executor.stop()
            harness.executor.join(2.0)
            self.assertFalse(harness.executor._thread.is_alive())
        finally:
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)

    def test_ingress_error_while_a_phase_is_pending_does_not_advance(self):
        harness = _RuntimeHarness(token_budget=32, max_num_seqs=2)
        harness.verify_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "V", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            harness.requests.put(
                {
                    "proc_id": "P",
                    "task_type": "prefill",
                    "total_tokens": 8,
                    "prefix_len": 0,
                }
            )
            self.assertTrue(harness.verify_started.wait(2.0))
            harness.poll_error = RuntimeError("ingress failed")
            self.assertTrue(wait_until(lambda: harness.runtime_error is not None))
            harness.verify_gate.set()
            harness.thread.join(2.0)
            harness.executor.stop()
            harness.executor.join(2.0)
            self.assertIsInstance(harness.runtime_error, RuntimeError)
            self.assertFalse(harness.thread.is_alive())
            self.assertFalse(harness.executor._thread.is_alive())
            self.assertFalse(harness.prefill_started.is_set())
        finally:
            harness.verify_gate.set()
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)

    def test_forward_error_is_surfaced_and_executor_stops(self):
        harness = _RuntimeHarness()
        harness.fail_forward = True
        harness.requests.put(
            {"proc_id": "V", "task_type": "verify", "gamma": 4, "prefix_len": 8}
        )
        harness.executor.start()
        harness.prefetch.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                harness.runtime.run()
            harness.executor.stop()
            harness.executor.join(2.0)
            self.assertFalse(harness.executor._thread.is_alive())
        finally:
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)

    def test_repeated_partial_prefill_rebuckets_without_new_ingress(self):
        harness = _RuntimeHarness(token_budget=128, max_num_seqs=1)
        harness.min_prefill = 16
        harness.quantum = 128
        harness.category_for = lambda item: (
            "short"
            if item.remaining_tokens <= 128
            else "mid"
            if item.remaining_tokens <= 512
            else "long"
        )
        harness.prefill_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {
                    "work_id": "P",
                    "proc_id": "P",
                    "task_type": "prefill",
                    "total_tokens": 600,
                    "prefix_len": 0,
                    "category": "long",
                }
            )
            self.assertTrue(harness.prefill_started.wait(2.0))
            item = harness.work_items["P"]
            self.assertEqual(item.category, "long")
            harness.prefill_gate.set()
            # No further ingress: the chunk commit alone must re-bucket the
            # WorkItem from long to mid (and eventually short).
            self.assertTrue(
                wait_until(lambda: item.category != "long", 2.0),
                "chunked Prefill was not re-bucketed after a partial commit",
            )
        finally:
            harness.prefill_gate.set()
            harness.close()

    def test_request_arriving_during_forward_is_prefetched_and_never_evicts_current(self):
        harness = _RuntimeHarness()
        harness.verify_gate.clear()
        # A is already GPU-resident and will be the admitted current plan.
        harness.cache_store["A"] = FakeCache("A", 8, "cuda")
        harness.residency.resident.add("A")
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "A", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            self.assertTrue(harness.verify_started.wait(2.0))
            self.assertTrue(
                wait_until(lambda: harness.in_flight["plan"] is not None)
            )

            # B arrives while forward A is blocked in the model call.
            harness.requests.put(
                {"proc_id": "B", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            self.assertTrue(
                wait_until(
                    lambda: "B" in harness.residency.resident
                    and harness.cache_store["B"].device == "cuda"
                ),
                "B was not prefetched while A was in flight",
            )
            self.assertFalse(harness.verify_gate.is_set())

            # A itself was reserved and must never be an eviction target.
            self.assertEqual(harness.cache_store["A"].device, "cuda")
            self.assertNotIn(("A", "cpu"), harness.moves)
            self.assertGreaterEqual(harness.metrics["incoming_during_forward"], 1)

            harness.verify_gate.set()
            self.assertTrue(harness.responses["A"].get(timeout=2.0))
            self.assertTrue(harness.responses["B"].get(timeout=2.0))
            self.assertTrue(
                wait_until(lambda: harness.metrics["plans"] >= 2),
                "B was not executed after A completed",
            )
            self.assertGreater(len(harness.moves), 0)
        finally:
            harness.verify_gate.set()
            harness.close()

    def test_verify_response_is_returned_before_prefill_forward_finishes(self):
        harness = _RuntimeHarness(token_budget=32, max_num_seqs=2)
        harness.prefill_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "V", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            harness.requests.put(
                {
                    "proc_id": "P",
                    "task_type": "prefill",
                    "total_tokens": 8,
                    "prefix_len": 0,
                }
            )
            self.assertTrue(harness.prefill_started.wait(2.0))
            # The Verify response is already available while Prefill is still
            # blocked inside the forward.
            payload = harness.responses["V"].get_nowait()
            self.assertEqual(payload["accepted"], 12)
            self.assertIn(("prefill_start", ("P",)), harness.events)
            self.assertNotIn(("prefill_end", ("P",)), harness.events)
        finally:
            harness.prefill_gate.set()
            harness.close()

    def test_shutdown_waits_for_inflight_forward_and_delivers_response_once(self):
        harness = _RuntimeHarness()
        harness.verify_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "A", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            self.assertTrue(harness.verify_started.wait(2.0))

            # The sentinel is consumed while A is still computing.  Shutdown
            # must wait for the forward instead of dropping its response.
            harness.requests.put(None)
            time.sleep(0.05)
            self.assertEqual(harness.responses["A"].qsize(), 0)
            self.assertTrue(harness.thread.is_alive())

            harness.verify_gate.set()
            harness.thread.join(3.0)
            self.assertFalse(harness.thread.is_alive())
            self.assertEqual(harness.responses["A"].qsize(), 1)
            self.assertEqual(harness.metrics["plans"], 1)
        finally:
            harness.verify_gate.set()
            harness.executor.stop()
            harness.executor.join(2.0)
            harness.prefetch.stop()
            harness.prefetch.join(2.0)
            harness.assert_threads_stopped()

    def test_cancel_of_admitted_prefill_suppresses_its_response(self):
        harness = _RuntimeHarness(token_budget=8, max_num_seqs=1)
        harness.prefill_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {
                    "proc_id": "P",
                    "task_type": "prefill",
                    "total_tokens": 8,
                    "prefix_len": 0,
                    "task_id": "t1",
                }
            )
            self.assertTrue(harness.prefill_started.wait(2.0))
            harness.requests.put(
                {
                    "control_type": "prefill_cancel",
                    "proc_id": "P",
                    "task_id": "t1",
                }
            )
            self.assertTrue(
                wait_until(lambda: harness.metrics["prefill_cancels_deferred"] == 1)
            )
            harness.prefill_gate.set()
            self.assertTrue(
                wait_until(lambda: harness.metrics["plans"] == 1)
            )
            time.sleep(0.05)
            self.assertEqual(harness.responses["P"].qsize(), 0)
            self.assertEqual(harness.work_items, {})
            self.assertFalse(harness.queues.has_items())
        finally:
            harness.prefill_gate.set()
            harness.close()

    def test_no_next_verify_candidate_submits_nothing(self):
        harness = _RuntimeHarness()
        harness.verify_gate.clear()
        harness.start()
        try:
            harness.requests.put(
                {"proc_id": "A", "task_type": "verify", "gamma": 4, "prefix_len": 8}
            )
            self.assertTrue(harness.verify_started.wait(2.0))
            self.assertTrue(wait_until(lambda: harness.metrics["prefetch_plans"] >= 1))
            self.assertEqual(harness.metrics["prefetch_async_submitted"], 0)
            self.assertEqual(harness.metrics["prefetch_empty_desired"], 1)
            self.assertEqual(harness.metrics["prefetch_evictions"], 0)
        finally:
            harness.verify_gate.set()
            harness.close()


if __name__ == "__main__":
    unittest.main()
