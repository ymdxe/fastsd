import unittest
from src.progress_control import AcceptanceProfile, BatchCostModel, AheadController, choose_gamma, draft_seconds_per_token
from src.fastsd_scheduler import WorkItem, plan_iteration


class ProgressControlTests(unittest.TestCase):
    def test_short_all_accept_round_is_censored(self):
        profile = AcceptanceProfile()
        profile.observe(2, 2)
        self.assertEqual(profile.expected_progress(2), 3)
        self.assertGreater(profile.expected_progress(4), 3)
        profile.observe(0, 8)
        self.assertLess(profile.expected_progress(8), 5)

    def test_gamma_uses_logical_round_draft_speed(self):
        req = {"lag": 0.16, "gamma": 4, "logical_gamma": 16}
        self.assertEqual(draft_seconds_per_token(req), 0.01)
        req["local_decode_per_token_s"] = 0.02
        self.assertEqual(draft_seconds_per_token(req), 0.02)
        self.assertEqual(draft_seconds_per_token({"lag": "bad", "gamma": float("inf")}), 0.0)

    def test_low_acceptance_reduces_gamma_and_queue_cannot_increase_it(self):
        profile = AcceptanceProfile()
        for _ in range(10):
            profile.observe(0, 8)
        self.assertLess(choose_gamma(profile, current=8, draft_s=0.01, verify_s=0.04), 8)
        profile = AcceptanceProfile()
        for _ in range(10):
            profile.observe(16, 16)
        self.assertLessEqual(choose_gamma(profile, current=4, draft_s=0.001,
                                        verify_s=0.1, queue_s=2), 4)

    def test_ahead_stops_when_no_compute_window_or_net_benefit(self):
        controller = AheadController()
        self.assertFalse(controller.allow(steps=0, cap=4, elapsed_s=0.01,
                                         expected_wait_s=0.015, draft_s=0.02))
        for _ in range(4):
            controller.observe(4, 0, 0.01, 0.01)
        self.assertFalse(controller.allow(steps=0, cap=4, elapsed_s=0,
                                         expected_wait_s=1, draft_s=0.01))

    def test_batch_profile_changes_with_context_and_actual_observations(self):
        costs = BatchCostModel()
        costs.observe("verify", [(128, 8)], 0.016)
        self.assertAlmostEqual(costs.estimate("verify", [(128, 8)]), 0.016)
        self.assertGreater(costs.estimate("verify", [(8192, 8)]), 0.016)

    def test_fcfs_keeps_same_batch_capacity(self):
        a = WorkItem("a", "a", "verify", "short", total_tokens=4,
                     request={"task_type": "verify", "proc_id": "a", "server_enqueue_monotonic": 2})
        b = WorkItem("b", "b", "verify", "long", total_tokens=4,
                     request={"task_type": "verify", "proc_id": "b", "server_enqueue_monotonic": 1})
        plan = plan_iteration({"verify": {"short": [a], "long": [b]}}, 8,
                              min_prefill_chunk_tokens=1, max_num_seqs=2,
                              selection_policy="fcfs", now=3)
        self.assertEqual(plan.verify_proc_ids, ["b", "a"])
        self.assertEqual(plan.used_tokens, 8)


if __name__ == "__main__":
    unittest.main()
