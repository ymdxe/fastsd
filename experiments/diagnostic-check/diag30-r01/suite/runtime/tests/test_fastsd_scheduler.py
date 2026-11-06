import math
import unittest

from src.fastsd_scheduler import (
    AdmissionPlan,
    WorkItem,
    full_prefix_bridge_tokens,
    verify_logit_position,
    build_fixed_wrr_order,
    commit_admission_plan,
    compute_priority_score,
    plan_iteration,
    predict_next_verify_proc_ids,
    reserve_admission_plan,
    should_switch_to_prefill,
    update_length_thresholds,
)


class FastSDSchedulerTests(unittest.TestCase):
    def test_full_prefix_bridge_count_tracks_uncached_correction(self):
        self.assertEqual(full_prefix_bridge_tokens(106, 106), 0)
        self.assertEqual(full_prefix_bridge_tokens(107, 106), 1)

        with self.assertRaisesRegex(ValueError, "trail logical prefix"):
            full_prefix_bridge_tokens(108, 106)

    def test_bridge_does_not_shift_logical_verify_logits(self):
        self.assertEqual(verify_logit_position(38, 0), 37)
        self.assertEqual(verify_logit_position(38, 3), 40)

        with self.assertRaises(ValueError):
            verify_logit_position(0, 0)

    def test_build_fixed_wrr_order_uses_631_ratio(self):
        order = build_fixed_wrr_order()
        self.assertEqual(len(order), 10)
        self.assertEqual(order.count("short"), 6)
        self.assertEqual(order.count("mid"), 3)
        self.assertEqual(order.count("long"), 1)

    def test_switch_to_prefill_requires_six_underutilized_verify_opportunities(self):
        self.assertFalse(should_switch_to_prefill([1, 1, 1, 1, 1], has_prefill_tasks=True))
        self.assertFalse(should_switch_to_prefill([1, 1, 1, 1, 1, 1], has_prefill_tasks=False))
        self.assertTrue(should_switch_to_prefill([1, 0, 1, 1, 1, 1, 0, 1, 0, 0], has_prefill_tasks=True))

    def test_update_length_thresholds_uses_runtime_quantiles_after_enough_samples(self):
        recent_lengths = list(range(1, 101))
        r1, r2 = update_length_thresholds(recent_lengths, default_r1=128, default_r2=512, min_samples=100)
        self.assertEqual(r1, 70)
        self.assertEqual(r2, 90)

    def test_update_length_thresholds_keeps_defaults_before_warmup(self):
        r1, r2 = update_length_thresholds([10, 20, 30], default_r1=128, default_r2=512, min_samples=100)
        self.assertEqual((r1, r2), (128, 512))

    def test_verify_priority_uses_lag_transport_wait_and_acceptance(self):
        req = {
            "proc_id": "draft-1",
            "task_type": "verify",
            "lag": 0.4,
            "transport_rtt": 0.1,
            "current_time": 95.0,
        }
        accept_stats = {"draft-1": [8, 10]}
        score = compute_priority_score(req, accept_stats, now=100.0, lamda=0.01)
        expected_acc_prob = (8 + 1) / (10 + 1)
        expected = -((0.4 + 0.1) / expected_acc_prob) + math.exp(0.01 * 5.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_verify_priority_uses_smoothed_acceptance_for_cold_start(self):
        req = {
            "proc_id": "draft-cold",
            "task_type": "verify",
            "lag": 0.4,
            "transport_rtt": 0.1,
            "current_time": 99.0,
        }
        score = compute_priority_score(req, {}, now=100.0, lamda=0.01)
        expected = -0.5 + math.exp(0.01)
        self.assertAlmostEqual(score, expected, places=6)

    def test_none_latency_fields_do_not_opt_legacy_verify_into_new_formula(self):
        req = {
            "proc_id": "draft-legacy",
            "task_type": "verify",
            "lag": 0.4,
            "transport_rtt": 0.1,
            "current_time": 95.0,
            "local_decode_per_token_s": None,
            "last_pull_s": None,
            "push_s": None,
        }
        score = compute_priority_score(req, {"draft-legacy": [8, 10]}, now=100.0, lamda=0.01)
        acceptance = (8 + 1) / (10 + 1)
        expected = -((0.4 + 0.1) / acceptance) + math.exp(0.05)
        self.assertAlmostEqual(score, expected, places=6)

    def test_prefill_pn_field_cannot_override_fixed_denominator(self):
        base = {
            "proc_id": "draft-1",
            "task_type": "prefill",
            "timing_required": True,
            "timing_ready": True,
            "local_prefill_s": 0.2,
            "local_decode_per_token_s": 0.03,
            "prefill_gamma": 4,
            "push_s": 0.01,
            "server_enqueue_monotonic": 95.0,
        }
        with_pn = dict(base, P_n=100.0)
        self.assertAlmostEqual(
            compute_priority_score(base, {}, now=100.0),
            compute_priority_score(with_pn, {}, now=100.0),
            places=6,
        )

    def test_prefill_priority_only_uses_wait_time(self):
        req = {
            "proc_id": "draft-1",
            "task_type": "prefill",
            "lag": 100.0,
            "transport_rtt": 100.0,
            "current_time": 99.0,
        }
        score = compute_priority_score(req, {"draft-1": [0, 1]}, now=100.0, lamda=0.01)
        self.assertAlmostEqual(score, math.exp(0.01), places=6)

    def test_prefill_priority_uses_cloud_arrival_time_before_token_upload(self):
        req = {
            "proc_id": "draft-arrival",
            "task_type": "prefill",
            "arrival_monotonic_s": 95.0,
            "current_time": 0.0,
        }
        score = compute_priority_score(req, {}, now=100.0, lamda=0.01)
        self.assertAlmostEqual(score, math.exp(0.05), places=6)

    def test_latency_aware_prefill_priority_uses_tp_td_and_push(self):
        req = {
            "proc_id": "draft-1",
            "task_type": "prefill",
            "timing_required": True,
            "timing_ready": True,
            "local_prefill_s": 0.20,
            "local_decode_per_token_s": 0.03,
            "prefill_gamma": 4,
            "push_s": 0.01,
            "server_enqueue_monotonic": 95.0,
        }
        score = compute_priority_score(req, {}, now=100.0, lamda=0.01)
        expected = -(0.20 + 0.03 * 3 + 0.01) + math.exp(0.05)
        self.assertAlmostEqual(score, expected, places=6)

    def test_latency_aware_verify_priority_uses_previous_pull(self):
        req = {
            "proc_id": "draft-1",
            "task_type": "verify",
            "gamma": 4,
            "local_decode_per_token_s": 0.03,
            "push_s": 0.01,
            "last_pull_s": 0.02,
            "server_enqueue_monotonic": 95.0,
        }
        score = compute_priority_score(req, {"draft-1": [3, 4]}, now=100.0, lamda=0.01)
        acceptance = (3 + 1) / (4 + 1)
        expected = -((0.03 * 4 + 0.01 + 0.02) / acceptance) + math.exp(0.05)
        self.assertAlmostEqual(score, expected, places=6)

    def test_timing_required_prefill_is_not_admitted_before_update(self):
        item = WorkItem(
            work_id="p1",
            proc_id="p1",
            task_type="prefill",
            category="short",
            request={
                "proc_id": "p1",
                "task_type": "prefill",
                "timing_required": True,
                "timing_ready": False,
                "current_time": 0.0,
                "prefill_gamma": 4,
            },
            total_tokens=20,
        )
        plan = plan_iteration(
            {"prefill": {"short": [item], "mid": [], "long": []}, "verify": {"short": [], "mid": [], "long": []}},
            token_budget=8,
            min_prefill_chunk_tokens=8,
        )
        self.assertEqual(plan.selected_work_ids, [])

    def test_predict_next_verify_proc_ids_looks_across_future_slots(self):
        order = ["short", "mid", "long"]
        verify_queues = {
            "short": [{"proc_id": "s1"}],
            "mid": [{"proc_id": "m1"}, {"proc_id": "m2"}],
            "long": [{"proc_id": "l1"}],
        }

        predicted = predict_next_verify_proc_ids(
            verify_queues,
            order,
            batch_size=3,
            start_idx=0,
        )

        self.assertEqual(predicted, ["s1", "m1", "m2"])

    def test_predict_next_verify_proc_ids_skips_already_preloaded_pids(self):
        order = ["short", "mid", "long"]
        verify_queues = {
            "short": [{"proc_id": "s1"}],
            "mid": [{"proc_id": "m1"}],
            "long": [{"proc_id": "l1"}],
        }

        predicted = predict_next_verify_proc_ids(
            verify_queues,
            order,
            batch_size=3,
            start_idx=0,
            pinned_gpu_pids={"s1"},
        )

        self.assertEqual(predicted, ["m1", "l1"])

    @staticmethod
    def _item(work_id, task_type, category, total, *, bridge=0, cycle=0, missed=0):
        return WorkItem(
            work_id=work_id,
            proc_id=work_id,
            task_type=task_type,
            category=category,
            request={
                "proc_id": work_id,
                "task_type": task_type,
                "current_time": 0.0,
                "gamma": total,
            },
            total_tokens=total,
            bridge_pending=bridge,
            enqueue_cycle=cycle,
            missed_cycles=missed,
        )

    def test_plan_uses_one_budget_for_verify_and_prefill_and_slices(self):
        queues = {
            "verify": {"short": [self._item("v1", "verify", "short", 8)], "mid": [], "long": []},
            "prefill": {"short": [self._item("p1", "prefill", "short", 100)], "mid": [], "long": []},
        }
        plan = plan_iteration(queues, token_budget=16, max_num_seqs=2, min_prefill_chunk_tokens=8, prefill_chunk_quantum=8)
        self.assertLessEqual(plan.used_tokens, 16)
        self.assertEqual(plan.verify_slices[0].forward_token_count, 8)
        self.assertEqual(plan.prefill_slices[0].draft_token_count, 8)
        self.assertEqual(plan.selected_work_ids, ["v1", "p1"])
        self.assertEqual(queues["prefill"]["short"][0].cursor, 0)

    def test_bridge_cost_counts_against_budget(self):
        item = self._item("v1", "verify", "short", 4, bridge=1)
        plan = plan_iteration(
            {"verify": {"short": [item], "mid": [], "long": []}, "prefill": {"short": [], "mid": [], "long": []}},
            token_budget=4,
            max_num_seqs=1,
            min_prefill_chunk_tokens=4,
        )
        self.assertEqual(plan.used_tokens, 4)
        self.assertEqual(plan.verify_slices[0].draft_token_count, 3)
        self.assertEqual(plan.verify_slices[0].forward_token_count, 4)
        self.assertTrue(plan.verify_slices[0].includes_bridge)

    def test_initial_external_bridge_is_admission_cost(self):
        request = {
            "proc_id": "v1",
            "task_type": "verify",
            "gamma": 3,
            "has_bridge_token": True,
            "prefix_len": 8,
            "draft_output": [[101, 11, 12, 13]],
        }
        item = WorkItem.from_request(request, category="short")
        plan = plan_iteration(
            {"verify": {"short": [item], "mid": [], "long": []}, "prefill": {"short": [], "mid": [], "long": []}},
            token_budget=4,
            max_num_seqs=1,
            min_prefill_chunk_tokens=4,
        )
        self.assertEqual(plan.used_tokens, 4)
        self.assertEqual(plan.verify_slices[0].draft_token_count, 3)
        self.assertTrue(plan.verify_slices[0].includes_bridge)

    def test_verify_is_chunked_when_remaining_budget_is_smaller_than_gamma(self):
        item = self._item("v1", "verify", "short", 8)
        plan = plan_iteration(
            {"verify": {"short": [item], "mid": [], "long": []}, "prefill": {"short": [], "mid": [], "long": []}},
            token_budget=3,
            max_num_seqs=1,
            min_prefill_chunk_tokens=3,
        )
        self.assertEqual(plan.used_tokens, 3)
        self.assertEqual(plan.verify_slices[0].draft_token_count, 3)

    def test_dry_run_and_reservation_commit_lifecycle(self):
        item = self._item("p1", "prefill", "short", 20)
        queues = {"prefill": {"short": [item], "mid": [], "long": []}, "verify": {"short": [], "mid": [], "long": []}}
        before = (item.cursor, item.state)
        plan = plan_iteration(queues, token_budget=8, max_num_seqs=1, min_prefill_chunk_tokens=8, prefill_chunk_quantum=8)
        self.assertEqual((item.cursor, item.state), before)
        reserve_admission_plan(plan)
        self.assertEqual(item.state, "reserved")
        commit_admission_plan(plan)
        self.assertEqual(item.cursor, 8)
        self.assertEqual(item.state, "ready")

    def test_plan_after_commit_selects_next_item_for_prefetch(self):
        first = self._item("v1", "verify", "short", 4)
        second = self._item("v2", "verify", "short", 4)
        queues = {
            "verify": {"short": [first, second], "mid": [], "long": []},
            "prefill": {"short": [], "mid": [], "long": []},
        }
        current = plan_iteration(
            queues,
            token_budget=4,
            max_num_seqs=1,
            min_prefill_chunk_tokens=4,
        )
        reserve_admission_plan(current)
        commit_admission_plan(current, completed_work_ids={"v1"})

        lookahead = plan_iteration(
            queues,
            token_budget=4,
            current_cycle=current.next_cycle,
            wrr_cursor=current.next_wrr_cursor,
            max_num_seqs=1,
            min_prefill_chunk_tokens=4,
            reserved_work_ids=current.selected_work_ids,
        )
        self.assertEqual(current.verify_proc_ids, ["v1"])
        self.assertEqual(lookahead.verify_proc_ids, ["v2"])

    def test_overdue_prefill_overrides_verify_within_category(self):
        verify = self._item("v1", "verify", "short", 4)
        prefill = self._item("p1", "prefill", "short", 20, missed=2)
        plan = plan_iteration(
            {"verify": {"short": [verify], "mid": [], "long": []}, "prefill": {"short": [prefill], "mid": [], "long": []}},
            token_budget=8,
            max_num_seqs=1,
            min_prefill_chunk_tokens=8,
            prefill_chunk_quantum=8,
        )
        self.assertEqual(plan.prefill_slices[0].work_id, "p1")


if __name__ == "__main__":
    unittest.main()
