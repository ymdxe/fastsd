import math
import queue
import unittest
from collections import deque

from src.fastsd_scheduler import (
    FASTSD_VERIFY_WRR_ORDER,
    OversizeVerifyError,
    ScheduleEntry,
    SchedulePlan,
    UnifiedSchedulerState,
    advance_prefill_cursor,
    build_fixed_wrr_order,
    build_unified_schedule_plan,
    compute_priority_score,
    drain_queue_bounded,
    ordered_plan_microbatches,
    request_budget_tokens,
    update_length_thresholds,
    uses_strict_fcfs,
)


class FastSDSchedulerTests(unittest.TestCase):
    @staticmethod
    def make_request(
        proc_id,
        task_type,
        payload_len,
        *,
        prefix_len=None,
        tail_only=True,
        category="short",
    ):
        if prefix_len is None:
            prefix_len = payload_len if task_type == "prefill" else 0
        return {
            "proc_id": proc_id,
            "task_id": f"task-{proc_id}",
            "task_type": task_type,
            "draft_output": [list(range(payload_len))],
            "prefix_len": prefix_len,
            "tail_only": tail_only,
            "current_time": 0.0,
            "_fastsd_category": category,
        }

    @staticmethod
    def empty_queues():
        return {
            "short": deque(),
            "mid": deque(),
            "long": deque(),
        }

    def test_build_fixed_wrr_order_uses_exact_631_ratio(self):
        order = build_fixed_wrr_order()
        self.assertEqual(tuple(order), FASTSD_VERIFY_WRR_ORDER)
        self.assertEqual(order.count("short"), 6)
        self.assertEqual(order.count("mid"), 3)
        self.assertEqual(order.count("long"), 1)

    def test_update_length_thresholds_uses_runtime_quantiles_after_warmup(self):
        recent_lengths = list(range(1, 101))
        r1, r2 = update_length_thresholds(
            recent_lengths, default_r1=128, default_r2=512, min_samples=100
        )
        self.assertEqual((r1, r2), (70, 90))

    def test_update_length_thresholds_keeps_defaults_before_warmup(self):
        self.assertEqual(
            update_length_thresholds(
                [10, 20, 30], default_r1=128, default_r2=512, min_samples=100
            ),
            (128, 512),
        )

    def test_priority_uses_latency_acceptance_and_wait(self):
        req = {
            "proc_id": "draft-1",
            "task_type": "verify",
            "lag": 0.4,
            "transport_rtt": 0.1,
            "current_time": 95.0,
        }
        score = compute_priority_score(req, {"draft-1": [8, 10]}, now=100.0)
        expected = -(0.5 / (9 / 11)) + math.exp(0.05)
        self.assertAlmostEqual(score, expected, places=6)

        req.update(task_type="prefill", lag=100.0, transport_rtt=100.0)
        self.assertAlmostEqual(
            compute_priority_score(req, {"draft-1": [0, 1]}, now=100.0),
            math.exp(0.05),
            places=6,
        )

    def test_effective_token_cost_covers_prefill_and_both_verify_payloads(self):
        prefill = self.make_request("p", "prefill", 12)
        tail_verify = self.make_request(
            "vt", "verify", 7, prefix_len=100, tail_only=True
        )
        full_verify = self.make_request(
            "vf", "verify", 17, prefix_len=12, tail_only=False
        )

        self.assertEqual(
            request_budget_tokens(prefill, chunk_start=3, chunk_end=8), 5
        )
        self.assertEqual(request_budget_tokens(tail_verify), 7)
        self.assertEqual(request_budget_tokens(full_verify), 5)

    def test_one_global_budget_admits_verify_then_prefill_without_overrun(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        verify["short"].append(self.make_request("v1", "verify", 6))
        prefill["mid"].append(
            self.make_request("p1", "prefill", 8, category="mid")
        )
        state = UnifiedSchedulerState(order=("short", "mid"))

        plan = build_unified_schedule_plan(verify, prefill, state, token_budget=10)

        self.assertEqual([entry.request_id for entry in plan.verify_entries], ["v1"])
        self.assertEqual([entry.request_id for entry in plan.prefill_entries], ["p1"])
        self.assertEqual(
            (plan.prefill_entries[0].chunk_start, plan.prefill_entries[0].chunk_end),
            (0, 4),
        )
        self.assertEqual(plan.used_tokens, 10)
        self.assertLessEqual(plan.used_tokens, plan.token_budget)

    def test_verify_that_only_exceeds_remaining_budget_waits_for_next_tick(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        verify["short"].append(self.make_request("v1", "verify", 6))
        verify["mid"].append(
            self.make_request("v2", "verify", 6, category="mid")
        )
        prefill["mid"].append(
            self.make_request("p1", "prefill", 4, category="mid")
        )
        state = UnifiedSchedulerState(order=("short", "mid"))

        plan = build_unified_schedule_plan(verify, prefill, state, token_budget=10)

        self.assertEqual([entry.request_id for entry in plan.verify_entries], ["v1"])
        self.assertEqual([entry.request_id for entry in plan.prefill_entries], ["p1"])
        self.assertEqual([req["proc_id"] for req in verify["mid"]], ["v2"])

    def test_oversize_atomic_verify_is_rejected_before_queues_mutate(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        request = self.make_request("too-large", "verify", 11)
        verify["short"].append(request)
        state = UnifiedSchedulerState()

        with self.assertRaises(OversizeVerifyError) as captured:
            build_unified_schedule_plan(verify, prefill, state, token_budget=10)

        self.assertIs(captured.exception.request, request)
        self.assertEqual(captured.exception.cost, 11)
        self.assertIs(verify["short"][0], request)
        self.assertEqual(state.cursor, 0)

    def test_partial_budget_flushes_without_waiting_for_exact_fill(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        verify["short"].append(self.make_request("v1", "verify", 3))

        plan = build_unified_schedule_plan(
            verify, prefill, UnifiedSchedulerState(), token_budget=10
        )

        self.assertEqual(plan.used_tokens, 3)
        self.assertEqual([entry.request_id for entry in plan.verify_entries], ["v1"])

    def test_cursor_persists_between_ticks(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        verify["short"].extend(
            [self.make_request("s1", "verify", 1), self.make_request("s2", "verify", 1)]
        )
        state = UnifiedSchedulerState()

        first = build_unified_schedule_plan(verify, prefill, state, token_budget=1)
        second = build_unified_schedule_plan(verify, prefill, state, token_budget=1)

        self.assertEqual((first.start_cursor, first.end_cursor), (0, 1))
        self.assertEqual((second.start_cursor, second.end_cursor), (1, 2))
        self.assertEqual(state.cursor, 2)

    def test_one_complete_cycle_visits_exact_631_slots(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        for index in range(6):
            verify["short"].append(self.make_request(f"s{index}", "verify", 1))
        for index in range(3):
            verify["mid"].append(
                self.make_request(f"m{index}", "verify", 1, category="mid")
            )
        verify["long"].append(
            self.make_request("l0", "verify", 1, category="long")
        )
        state = UnifiedSchedulerState()

        plan = build_unified_schedule_plan(verify, prefill, state, token_budget=10)

        categories = [entry.category for entry in plan.verify_entries]
        self.assertEqual(tuple(categories), FASTSD_VERIFY_WRR_ORDER)
        self.assertEqual(state.cursor, 0)
        self.assertEqual(plan.completed_cycles, 1)

    def test_empty_cycle_advances_and_stops_without_spinning(self):
        state = UnifiedSchedulerState()
        plan = build_unified_schedule_plan(
            self.empty_queues(), self.empty_queues(), state, token_budget=10
        )
        self.assertEqual(plan.entries, [])
        self.assertEqual(plan.completed_cycles, 1)
        self.assertEqual(state.cursor, 0)

    def test_each_category_forces_one_prefill_after_two_missed_cycles(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        for category in ("short", "mid", "long"):
            verify[category].extend(
                self.make_request(f"{category}-v{index}", "verify", 1, category=category)
                for index in range(3)
            )
            prefill[category].append(
                self.make_request(f"{category}-p", "prefill", 2, category=category)
            )
        state = UnifiedSchedulerState(
            order=("short", "mid", "long"), prefill_max_wait_cycles=2
        )

        first = build_unified_schedule_plan(verify, prefill, state, token_budget=3)
        second = build_unified_schedule_plan(verify, prefill, state, token_budget=3)
        third = build_unified_schedule_plan(verify, prefill, state, token_budget=6)

        self.assertEqual(len(first.prefill_entries), 0)
        self.assertEqual(len(second.prefill_entries), 0)
        self.assertEqual(
            [entry.category for entry in third.prefill_entries],
            ["short", "mid", "long"],
        )
        self.assertEqual(state.prefill_wait_cycles, {"short": 0, "mid": 0, "long": 0})

    def test_prefill_wait_counters_are_category_local(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        for category in ("short", "mid", "long"):
            verify[category].append(
                self.make_request(f"{category}-v", "verify", 1, category=category)
            )
        prefill["short"].append(self.make_request("short-p", "prefill", 2))
        state = UnifiedSchedulerState(order=("short", "mid", "long"))

        build_unified_schedule_plan(verify, prefill, state, token_budget=3)

        self.assertEqual(state.prefill_wait_cycles["short"], 1)
        self.assertEqual(state.prefill_wait_cycles["mid"], 0)
        self.assertEqual(state.prefill_wait_cycles["long"], 0)

    def test_due_prefill_survives_budget_exhaustion_before_its_slot(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        verify["short"].append(self.make_request("short-v", "verify", 1))
        prefill["mid"].append(
            self.make_request("mid-p", "prefill", 2, category="mid")
        )
        state = UnifiedSchedulerState(order=("short", "mid"))
        state.prefill_wait_cycles["mid"] = 2

        first = build_unified_schedule_plan(verify, prefill, state, token_budget=1)
        self.assertEqual([entry.request_id for entry in first.verify_entries], ["short-v"])
        self.assertEqual(state.cursor, 1)
        self.assertEqual(state.prefill_wait_cycles["mid"], 2)

        second = build_unified_schedule_plan(verify, prefill, state, token_budget=1)
        self.assertEqual([entry.request_id for entry in second.prefill_entries], ["mid-p"])
        self.assertEqual(state.prefill_wait_cycles["mid"], 0)

    def test_partial_prefill_is_scheduled_once_per_plan_and_resumes(self):
        verify = self.empty_queues()
        prefill = self.empty_queues()
        request = self.make_request("p1", "prefill", 5)
        prefill["short"].append(request)
        state = UnifiedSchedulerState(order=("short", "short"))

        first = build_unified_schedule_plan(verify, prefill, state, token_budget=3)
        self.assertEqual(len(first.prefill_entries), 1)
        self.assertEqual(
            (first.prefill_entries[0].chunk_start, first.prefill_entries[0].chunk_end),
            (0, 3),
        )
        self.assertFalse(advance_prefill_cursor(request, 3))
        prefill["short"].append(request)

        second = build_unified_schedule_plan(verify, prefill, state, token_budget=3)
        self.assertEqual(len(second.prefill_entries), 1)
        self.assertEqual(
            (second.prefill_entries[0].chunk_start, second.prefill_entries[0].chunk_end),
            (3, 5),
        )
        self.assertTrue(advance_prefill_cursor(request, 5))

    def test_prefill_completion_is_reported_exactly_once(self):
        request = self.make_request("p1", "prefill", 4)
        self.assertFalse(advance_prefill_cursor(request, 2))
        self.assertTrue(advance_prefill_cursor(request, 4))
        with self.assertRaises(ValueError):
            advance_prefill_cursor(request, 4)

    def test_execution_batches_are_homogeneous_verify_first_and_size_limited(self):
        make_entry = lambda name, task: ScheduleEntry(
            request={"proc_id": name, "task_type": task},
            request_id=name,
            task_type=task,
            category="short",
            effective_tokens=1,
        )
        plan = SchedulePlan(
            verify_entries=[make_entry(f"v{i}", "verify") for i in range(5)],
            prefill_entries=[make_entry(f"p{i}", "prefill") for i in range(3)],
        )

        batches = ordered_plan_microbatches(plan, max_num_seqs=2)

        self.assertEqual([task for task, _ in batches], ["verify"] * 3 + ["prefill"] * 2)
        self.assertTrue(all(len(batch) <= 2 for _, batch in batches))
        for task_type, batch in batches:
            self.assertTrue(all(entry.task_type == task_type for entry in batch))

    def test_bounded_ingress_never_reads_more_than_limit(self):
        class CountingQueue:
            def __init__(self, items):
                self.items = deque(items)
                self.calls = 0

            def get_nowait(self):
                self.calls += 1
                if not self.items:
                    raise queue.Empty
                return self.items.popleft()

        source = CountingQueue(range(20))
        items, stopped = drain_queue_bounded(source, max_items=6)
        self.assertEqual(items, list(range(6)))
        self.assertEqual(source.calls, 6)
        self.assertFalse(stopped)

    def test_bounded_ingress_recognizes_stop_sentinel(self):
        source = queue.Queue()
        source.put("request")
        source.put(None)
        source.put("unread")
        items, stopped = drain_queue_bounded(source, max_items=6)
        self.assertEqual(items, ["request"])
        self.assertTrue(stopped)
        self.assertEqual(source.get_nowait(), "unread")

    def test_pipeline_and_vanilla_keep_strict_fcfs(self):
        self.assertTrue(uses_strict_fcfs("pipeline"))
        self.assertTrue(uses_strict_fcfs("vanilla"))
        self.assertFalse(uses_strict_fcfs("fastsd"))


if __name__ == "__main__":
    unittest.main()
