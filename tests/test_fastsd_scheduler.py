import math
import unittest

from src.fastsd_scheduler import (
    build_fixed_wrr_order,
    compute_priority_score,
    predict_next_verify_proc_ids,
    should_switch_to_prefill,
    update_length_thresholds,
)


class FastSDSchedulerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
