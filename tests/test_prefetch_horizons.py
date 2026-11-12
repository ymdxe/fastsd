"""Bounded prefetch priority, production prediction and admission accounting."""
from collections import defaultdict
import threading
from types import SimpleNamespace
import unittest

from src.cache_protocol import CacheSnapshot, KVTransferTicket, PrefetchTarget, ordered_prefetch_targets
from src.fastsd_scheduler import reserve_admission_plan, abort_admission_plan
from src.scheduler_runtime import CacheResidency, KVPrefetchWorker, PrefetchCoordinator
from tests import test_round3_runtime as engine_checks


class BudgetTransfers:
    def __init__(self, limit=2, reject=()):
        self.limit, self.reject = limit, set(reject)
        self.tickets, self.prepared, self.advanced = [], [], []

    def submit_h2d(self, snapshot, purpose):
        self.prepared.append(snapshot.cache.pid)
        if snapshot.cache.pid in self.reject or len(self.tickets) >= self.limit:
            return None
        ticket = KVTransferTicket(snapshot, purpose, bytes_reserved=64)
        self.tickets.append(ticket)
        return ticket

    def advance_prefetch(self, ticket):
        self.advanced.append(ticket.snapshot.cache.pid)

    def retire(self, ticket):
        ticket.retired = True
        ticket.copy_state = "invalid-draining"

    def poll_completed(self):
        self.tickets[:] = [ticket for ticket in self.tickets if not ticket.retired]


class HorizonWorkerTests(unittest.TestCase):
    def setup_worker(self, transfers=None):
        registry, metrics = CacheResidency(), defaultdict(int)
        store = {pid: SimpleNamespace(pid=pid, get_seq_length=lambda: 8) for pid in "abcd"}
        transfers = transfers or BudgetTransfers()
        worker = KVPrefetchWorker(residency=registry, get_cache=store.get, set_cache=None,
            move_cache=None, gpu_device="cuda", metrics=metrics, transfers=transfers)
        return worker, registry, transfers, metrics

    def execute(self, worker, targets):
        worker._latest_intent = ordered_prefetch_targets(targets)
        worker._execute(worker._latest_intent)

    def test_earliest_horizon_wins_and_order_is_stable(self):
        targets = ordered_prefetch_targets([PrefetchTarget("a", 3), PrefetchTarget("b", 1),
            PrefetchTarget("a", 1), PrefetchTarget("c", 2), PrefetchTarget("b", 2)])
        self.assertEqual([(t.pid, t.horizon) for t in targets], [("a", 1), ("b", 1), ("c", 2)])
        self.assertEqual(ordered_prefetch_targets(["z", "a", "z"]), ("z", "a"))

    def test_h1_gets_budget_before_lower_horizons(self):
        worker, registry, transfers, metrics = self.setup_worker(BudgetTransfers(limit=1))
        self.execute(worker, [PrefetchTarget("a", 3), PrefetchTarget("b", 2), PrefetchTarget("c", 1)])
        self.assertEqual(list(registry.candidates), ["c"])
        self.assertEqual(transfers.prepared[0], "c")
        self.assertEqual(set(transfers.advanced), {"c"})
        self.assertEqual(metrics["prefetch_candidates_created_by_horizon"], {"1": 1})

    def test_failed_h1_reservation_blocks_lower_horizon_allocations(self):
        worker, registry, transfers, _ = self.setup_worker(BudgetTransfers(reject="a"))
        self.execute(worker, [PrefetchTarget("a", 1), PrefetchTarget("b", 1), PrefetchTarget("c", 2)])
        self.assertEqual(transfers.prepared, ["a", "b"])
        self.assertEqual(list(registry.candidates), ["b"])

    def test_lower_horizon_only_refresh_keeps_and_advances_existing_h1(self):
        worker, registry, transfers, metrics = self.setup_worker()
        self.execute(worker, [PrefetchTarget("a", 1, 1)])
        original = registry.candidates["a"]
        transfers.advanced.clear()
        self.execute(worker, [PrefetchTarget("b", 2, 1), PrefetchTarget("c", 3, 1)])
        self.assertIs(registry.candidates["a"], original)
        self.assertFalse(original.retired)
        self.assertEqual(set(transfers.advanced), {"a"})
        self.assertEqual(metrics["prefetch_retired_by_horizon"], {})

    def test_same_tick_refresh_preserves_h1_and_later_tick_releases_obsolete_h1(self):
        worker, registry, _, metrics = self.setup_worker()
        self.execute(worker, [PrefetchTarget("a", 1, 1)])
        original = registry.candidates["a"]
        self.execute(worker, [PrefetchTarget("b", 1, 1)])
        self.assertIs(registry.candidates["a"], original)
        self.execute(worker, [PrefetchTarget("b", 1, 2)])
        self.assertTrue(original.retired)
        self.assertNotIn("a", registry.candidates)
        self.assertEqual(metrics["prefetch_retired_by_horizon"], {"1": 1})

    def test_valid_h1_is_not_demoted_when_later_predicted_at_h2(self):
        worker, registry, _, _ = self.setup_worker()
        self.execute(worker, [PrefetchTarget("a", 1, 1)])
        original = registry.candidates["a"]
        self.execute(worker, [PrefetchTarget("b", 1, 2), PrefetchTarget("a", 2, 2)])
        self.assertIs(registry.candidates["a"], original)
        self.assertEqual(original.horizon, 1)

    def test_h2_is_retired_to_free_budget_for_new_h1(self):
        worker, registry, _, metrics = self.setup_worker(BudgetTransfers(limit=1))
        self.execute(worker, [PrefetchTarget("a", 2, 1)])
        old = registry.candidates["a"]
        self.execute(worker, [PrefetchTarget("b", 1, 1), PrefetchTarget("a", 2, 1)])
        self.assertTrue(old.retired)
        self.assertEqual(list(registry.candidates), ["b"])
        self.assertEqual(metrics["prefetch_retired_by_horizon"], {"2": 1})

    def test_promoted_h2_is_kept_when_other_h1_is_missing(self):
        worker, registry, _, _ = self.setup_worker()
        self.execute(worker, [PrefetchTarget("a", 2, 1)])
        original = registry.candidates["a"]
        self.execute(worker, [PrefetchTarget("a", 1, 2), PrefetchTarget("b", 1, 2)])
        self.assertIs(registry.candidates["a"], original)
        self.assertEqual(original.horizon, 1)
        self.assertEqual(original.transfer_horizon, 2)

    def test_content_change_invalidates_protected_h1(self):
        worker, registry, _, _ = self.setup_worker()
        self.execute(worker, [PrefetchTarget("a", 1, 1)])
        original = registry.candidates["a"]
        registry.bump(["a"])
        self.execute(worker, [PrefetchTarget("b", 2, 1)])
        self.assertTrue(original.retired)
        self.assertNotIn("a", registry.candidates)

    def test_explicit_empty_intent_releases_candidates(self):
        worker, registry, _, _ = self.setup_worker()
        self.execute(worker, [PrefetchTarget("a", 1, 1)])
        self.execute(worker, [])
        self.assertEqual(registry.candidates, {})

    def test_h1_preparation_survives_concurrent_lower_horizon_refresh(self):
        worker, registry, transfers, _ = self.setup_worker()
        entered, release = threading.Event(), threading.Event()
        prepare = transfers.submit_h2d
        def blocked(snapshot, purpose):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test preparation timed out")
            return prepare(snapshot, purpose)
        transfers.submit_h2d = blocked
        intent = (PrefetchTarget("a", 1, 1),)
        worker._latest_intent = intent
        thread = threading.Thread(target=worker._execute, args=(intent,))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            worker.submit([PrefetchTarget("a", 1, 1), PrefetchTarget("b", 2, 1)])
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIn("a", registry.candidates)
        self.assertFalse(registry.candidates["a"].retired)

    def test_coordinator_protection_and_counts_use_pid_not_target_object(self):
        sent, metrics = [], defaultdict(int)
        coordinator = PrefetchCoordinator(submit=lambda intent: sent.append(intent) or True,
            resident_snapshot=lambda: set(), protected_snapshot=lambda: {"a"}, metrics=metrics)
        coordinator.refresh([PrefetchTarget("b", 2), PrefetchTarget("a", 1), PrefetchTarget("c", 1)])
        self.assertEqual([(t.pid, t.horizon) for t in sent[0]], [("c", 1), ("b", 2)])
        self.assertEqual(metrics["prefetch_intent_by_horizon"], {"1": 1, "2": 1})
        self.assertEqual(metrics["prefetch_reserved_excluded"], 1)

    def test_filtering_all_new_targets_does_not_clear_same_tick_h1(self):
        sent = []
        coordinator = PrefetchCoordinator(submit=lambda intent: sent.append(intent) or True,
            resident_snapshot=lambda: set(), protected_snapshot=lambda: set(), metrics=defaultdict(int),
            pending_snapshot=lambda: True)
        h1 = PrefetchTarget("a", 1, 1)
        coordinator.refresh([h1], prediction_tick=1)
        coordinator.refresh([], prediction_tick=1)
        self.assertEqual(sent, [(h1,)])
        coordinator.refresh([], prediction_tick=2)
        self.assertEqual(sent[-1], ())


class EngineHorizonTests(unittest.TestCase):
    def environment(self, count=6):
        helper = engine_checks.EnginePredictionIntegrationTests()
        return helper.environment(*(helper.item(str(i)) for i in range(count)))

    def test_k3_predicts_three_but_only_prefetches_h1(self):
        env, _, submitted, _ = self.environment()
        env["PREFETCH_LOOKAHEAD_ROUNDS"] = 3
        env["refresh_prefetch"]()
        metrics = env["scheduler_metrics"]
        self.assertEqual(len(env["prediction_snapshots"][1]), 3)
        self.assertEqual(metrics["prefetch_refresh_count"], 1)
        self.assertEqual(metrics["lookahead_build_count"], 3)
        self.assertEqual(metrics["prefetch_skipped_horizon"], {"2": 1, "3": 1})
        self.assertEqual(submitted, [["0"]])
        env["refresh_prefetch"]()
        self.assertEqual(metrics["prefetch_refresh_count"], 1)

    def test_horizon_zero_is_prediction_only(self):
        env, _, submitted, _ = self.environment()
        env.update(PREFETCH_LOOKAHEAD_ROUNDS=3, PREFETCH_HORIZON_LIMIT=0)
        env["refresh_prefetch"]()
        self.assertEqual(submitted, [[]])
        self.assertEqual(len(env["prediction_snapshots"][1]), 3)

    def test_h2_cap_and_default_h3_exclusion(self):
        env, _, submitted, _ = self.environment(count=12)
        env["self"].args.batch_size = 3
        env["self"].args.token_budget = 24
        env.update(PREFETCH_LOOKAHEAD_ROUNDS=3, PREFETCH_HORIZON_LIMIT=2, PREFETCH_H2_MAX_PIDS=1)
        env["refresh_prefetch"]()
        self.assertEqual(len(submitted[0]), 4)
        self.assertEqual(env["scheduler_metrics"]["prefetch_skipped_horizon"], {"2": 2, "3": 3})

    def test_no_lookahead_or_intent_when_management_is_bypassed(self):
        env, _, submitted, _ = self.environment()
        env["residency_manager"].enabled = False
        env["refresh_prefetch"]()
        self.assertEqual(submitted, [])
        self.assertEqual(env["scheduler_metrics"]["lookahead_build_count"], 0)

    def submit_environment(self):
        env, _, _, _ = self.environment()
        metrics = env["scheduler_metrics"]
        metrics["prefetch_prediction_by_horizon"] = {}
        events = []
        env.update(_slice_request=lambda item, sl: {"proc_id": item.proc_id},
            _sort_queues_if_dirty=lambda **kw: None,
            ensure_cache_management=lambda: None, observe_memory=lambda: None,
            diag_req=lambda request: request,
            reserve_admission_plan=reserve_admission_plan, abort_admission_plan=abort_admission_plan,
            executor=SimpleNamespace(submit=lambda job: events.append("submit")),
            _ExecutionJob=lambda **kw: SimpleNamespace(verify_applied=False, progress={}, completed=set(), **kw),
            PREFETCH_BEFORE_SUBMIT=False)
        env["residency_manager"].active = False
        engine_checks.engine_closure("submit_plan", env)
        return env, events

    def test_snapshot_counts_are_separate_from_actual_plan_counts(self):
        env, _ = self.submit_environment()
        env["PREFETCH_LOOKAHEAD_ROUNDS"] = 3
        env["refresh_prefetch"]()
        env["refresh_prefetch"]()
        env["submit_plan"](env["build_plan"]())
        for _ in range(2):
            env["in_flight"].update(job=None, plan=None)
            for item in list(env["task_queues"].iter_items()):
                if item.state == "reserved":
                    item.state = "finished"
                    item.finished = True
            env["submit_plan"](env["build_plan"]())
        metrics = env["scheduler_metrics"]
        self.assertEqual(metrics["actual_plans"], 3)
        self.assertEqual(metrics["prediction_snapshots_consumed"], 3)
        self.assertEqual({h: b["plans"] for h, b in metrics["prefetch_prediction_by_horizon"].items()},
                         {"1": 1, "2": 1, "3": 1})
        self.assertEqual({h: b["hits"] for h, b in metrics["prefetch_prediction_by_horizon"].items()},
                         {"1": 1, "2": 1, "3": 1})

    def test_opt_in_prediction_runs_after_reservation_before_executor_submit(self):
        env, events = self.submit_environment()
        def prediction():
            self.assertIsNotNone(env["in_flight"]["plan"])
            self.assertTrue(all(item.state == "reserved" for item in env["in_flight"]["plan"]._selected_items))
            events.append("prediction")
        env.update(PREFETCH_BEFORE_SUBMIT=True, refresh_prefetch=prediction)
        env["submit_plan"](env["build_plan"]())
        self.assertEqual(events, ["prediction", "submit"])


if __name__ == "__main__":
    unittest.main()
