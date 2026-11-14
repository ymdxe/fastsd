"""Latency ablations keep WRR but use arrival order within each queue."""

from types import SimpleNamespace
import unittest

from src.fastsd_scheduler import (
    WorkItem, compute_priority_score, plan_iteration, predict_next_plan,
)
from tests import test_round3_runtime as engine_checks


class LatencyAblationTests(unittest.TestCase):
    def items(self):
        def make(pid, arrival, lag):
            return WorkItem(pid, pid, "verify", "short", total_tokens=8,
                request={"task_type": "verify", "proc_id": pid, "gamma": 8,
                         "server_enqueue_monotonic": arrival, "lag": lag})
        return make("old_slow", 1, 10), make("new_fast", 2, 0.01)

    def test_disable_latency_scoring_in_wrr_and_rr(self):
        for policy in ("legacy", "rr"):
            for enabled, expected in ((True, "new_fast"), (False, "old_slow")):
                with self.subTest(policy=policy, enabled=enabled):
                    items = self.items()
                    plan = plan_iteration({"verify": {"short": list(items)}},
                        token_budget=8, min_prefill_chunk_tokens=4, now=3,
                        selection_policy=policy, latency_priority=enabled)
                    self.assertEqual(plan.verify_proc_ids, [expected])

    def test_prediction_obeys_the_same_disabled_latency_scoring(self):
        plan = predict_next_plan({"verify": {"short": list(self.items())}},
            token_budget=8, min_prefill_chunk_tokens=4, planning_time=3,
            latency_priority=False)
        self.assertEqual(plan.verify_proc_ids, ["old_slow"])

    def test_disabled_score_ignores_acceptance_and_timing(self):
        old, new = self.items()
        old.request.update(timing_priority=True, local_decode_per_token_s=99,
                           push_s=88, P_n=0.01)
        stats = {old.proc_id: (0, 100)}
        self.assertGreater(
            compute_priority_score(old.request, stats, now=3, latency_priority=False),
            compute_priority_score(new.request, stats, now=3, latency_priority=False))

    def test_engine_disables_latency_for_sort_and_admission(self):
        env, _, _, _ = engine_checks.EnginePredictionIntegrationTests().environment(*self.items())
        env["self"].args.enable_latency_priority = False
        env["self"].args.scheduler_policy = "legacy"
        env["time"] = SimpleNamespace(monotonic=lambda: 3,
            monotonic_ns=lambda: 3000, perf_counter=lambda: 3)
        env.update(_compute_priority_score=compute_priority_score,
                   ensure_cache_management=lambda: None, observe_memory=lambda: None)
        engine_checks.engine_closure("_sort_queues_if_dirty", env)
        engine_checks.engine_closure("build_plan", env)
        plan = env["build_plan"]()
        self.assertEqual(plan.verify_proc_ids, ["old_slow"])
        self.assertEqual(list(env["task_queues"].iter_items())[0].proc_id, "old_slow")


if __name__ == "__main__":
    unittest.main()
