import queue
import threading
import time
import unittest
from collections import defaultdict

from src.cache_offload_policy import should_offload_target_cache
from src.common_metrics import exact_match_quality
from src.fastsd_scheduler import WorkItem, plan_iteration, project_after_plan
from src.scheduler_runtime import (IngressRelay, ExecutionOutcome, SinglePlanExecutor,
                                   SchedulerQueues, ShutdownRequested, TargetSchedulerRuntime)
from src.session_lifecycle import SessionLifecycle


class ReviewRuntimeTests(unittest.TestCase):
    def test_quality_counts_missing_and_non_numeric_outputs(self):
        records = [{"reference": "42", "output_text": out} for out in ("42", "I do not know", None)]
        records.append({"reference": "no numeric reference", "output_text": "42"})
        self.assertEqual(exact_match_quality(records, "gsm8k"), 1 / 3)

    def test_single_prefill_uses_budget_even_with_one_sequence_slot(self):
        p = WorkItem("p", "p", "prefill", "long", total_tokens=2048)
        p.request = {"task_type": "prefill", "proc_id": "p"}
        for slots in (1, 4):
            plan = plan_iteration({"prefill": {"long": [p]}}, 512, max_num_seqs=slots)
            self.assertEqual(plan.used_tokens, 512)
            self.assertEqual(len(plan.prefill_slices), 1)
            self.assertEqual(p.cursor, 0)

    def test_prefill_stays_quantized_when_verify_waits(self):
        p = WorkItem("p", "p", "prefill", "short", total_tokens=2048, missed_cycles=2)
        v = WorkItem("v", "v", "verify", "short", total_tokens=8)
        p.request = {"task_type": "prefill", "proc_id": "p"}
        v.request = {"task_type": "verify", "proc_id": "v"}
        plan = plan_iteration({"prefill": {"short": [p]}, "verify": {"short": [v]}},
                              512, max_num_seqs=1)
        self.assertEqual(plan.used_tokens, 128)

    def test_projection_keeps_partial_verify_and_does_not_mutate_it(self):
        a = WorkItem("a", "a", "verify", "short", total_tokens=8)
        b = WorkItem("b", "b", "verify", "short", total_tokens=4)
        a.request = {"task_type": "verify", "proc_id": "a"}
        b.request = {"task_type": "verify", "proc_id": "b"}
        queues = {"verify": {"short": [a, b]}}
        first = plan_iteration(queues, 4, max_num_seqs=1, min_prefill_chunk_tokens=1)
        second = plan_iteration(project_after_plan(queues, first), 4, max_num_seqs=1,
                                min_prefill_chunk_tokens=1)
        self.assertEqual(second.verify_proc_ids, ["a"])
        self.assertEqual(second.verify_slices[0].offset, 4)
        self.assertEqual(a.cursor, 0)

    def test_continuing_cache_is_retained_until_pressure(self):
        for stage in ("prefill", "verify"):
            self.assertFalse(should_offload_target_cache(stage, continuing=True))
            self.assertTrue(should_offload_target_cache(stage, continuing=True, memory_pressure=True))

    def test_release_waits_for_forward_and_expires_idle_sessions(self):
        protected, busy, cleaned, replies = {"active"}, {"queued"}, [], []
        now = [0.0]
        lifecycle = SessionLifecycle(cleanup=cleaned.append, cancel=lambda p: None,
            protected=lambda: protected.copy(), busy=lambda: busy.copy(),
            reply=lambda k, r: replies.append((k, r)), idle_timeout=5, clock=lambda: now[0])
        for pid in ("active", "idle", "queued"):
            lifecycle.touch(pid)
        lifecycle.release("active", "ack")
        now[0] = 10
        lifecycle.sweep()
        self.assertEqual(cleaned, ["idle"])
        self.assertEqual(replies, [])
        protected.clear()
        lifecycle.sweep()
        self.assertEqual(cleaned, ["idle", "active"])
        self.assertEqual(replies, [("ack", {"status": "released"})])
        self.assertEqual(set(lifecycle.last_seen), {"queued"})

    def test_completion_wakes_controller_without_ingress_poll_timeout(self):
        # A deliberately huge legacy poll timeout makes this a deterministic
        # event contract test rather than a microsecond timing benchmark.
        wake = threading.Event()
        source = queue.Queue()
        relay = IngressRelay(source, wake)
        completed = threading.Event()
        executor = SinglePlanExecutor(lambda job: ExecutionOutcome(job), wake_event=wake)
        metrics = defaultdict(int)
        def poll(timeout):
            try:
                item = relay.queue.get_nowait()
            except queue.Empty:
                return 0
            if item is None:
                raise ShutdownRequested
            return 1
        def apply(outcome):
            completed.set()
            source.put(None)
        runtime = TargetSchedulerRuntime(executor=executor, queues=SchedulerQueues(metrics),
            poll_ingress=poll, build_plan=lambda: None, submit_plan=lambda p: None,
            apply_outcome=apply, refresh_prefetch=lambda: None, metrics=metrics,
            executor_poll_timeout=30, wake_event=wake)
        relay.start()
        executor.start()
        executor.submit("job")
        control = threading.Thread(target=runtime.run, daemon=True)
        control.start()
        try:
            self.assertTrue(completed.wait(1.0))
            control.join(1.0)
            self.assertFalse(control.is_alive())
        finally:
            source.put(None)
            executor.stop()
            executor.join()
            relay.close()


if __name__ == "__main__":
    unittest.main()
