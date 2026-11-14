"""Exercise real FCFS and eviction bodies without loading a target model."""

import ast
from collections import defaultdict
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import unittest

from src.cache_offload_policy import should_offload_target_cache
from src.cache_residency import (
    CPU_RESIDENT, EVICTION_PENDING, GPU_RESIDENT, RESTORE_PENDING, ResidencyManager,
)
from src.scheduler_runtime import CacheResidency
from tests.test_round3_runtime import engine_closure


def run_fcfs_loop(namespace):
    source = Path(__file__).resolve().parents[1] / "src" / "engine.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    branch = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                  and node.test.id == "use_strict_fcfs")
    wrapper = ast.parse("def run():\n    pass\n").body[0]
    wrapper.body = [branch]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 str(source), "exec"), namespace)
    namespace["run"]()


class ResidencyBoundaryTests(unittest.TestCase):
    def environment(self):
        manager = ResidencyManager(budget_bytes=1000)
        caches = {}
        copies = []

        def add(pid, size):
            cache = SimpleNamespace(get_seq_length=lambda: size // 10)
            caches[pid] = cache
            manager.note_created(pid, nbytes=size)

        def offload(cache):
            copies.append(cache)
            return SimpleNamespace(destination_cache=cache)

        env = dict(
            residency_manager=manager, residency=CacheResidency(),
            kv_cache_manager=SimpleNamespace(_past_key_values=caches),
            cache_lock=threading.RLock(), plan_sequence={"tick": 1},
            scheduler_metrics=defaultdict(int), preloaded_gpu_pids=set(),
            compute_stream=None,
            target_model=SimpleNamespace(device=SimpleNamespace(type="cpu")),
            transfers=SimpleNamespace(mark_computed=lambda *a, **k: None,
                submit_d2h=offload, retain_cache=lambda *a, **k: None),
            should_offload_target_cache=should_offload_target_cache,
            GPU_RESIDENT=GPU_RESIDENT, CPU_RESIDENT=CPU_RESIDENT,
            EVICTION_PENDING=EVICTION_PENDING,
            DIAG=SimpleNamespace(emit=lambda *a, **k: None),
        )
        engine_closure("rebalance_cache_memory", env)
        return env, manager, add, copies

    def test_cache_management_only_starts_prefetch_resources_when_enabled(self):
        for fcfs, prefetch in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(fcfs=fcfs, prefetch=prefetch):
                env, manager, _, _ = self.environment()
                started = []
                env.update(use_strict_fcfs=fcfs, prefetch_enabled=True,
                    self=SimpleNamespace(args=SimpleNamespace(kv_prefetch=prefetch)),
                    cache_bytes_per_token=lambda: 10,
                    prefetch_worker=SimpleNamespace(start=lambda: started.append("worker")))
                env["transfers"].warmup_prefetch_pool = lambda: started.append("pool")
                engine_closure("ensure_cache_management", env)
                self.assertTrue(env["ensure_cache_management"]())
                self.assertFalse(env["ensure_cache_management"]())
                self.assertEqual(started, ["pool", "worker"] if prefetch and not fcfs else [])

    def test_below_high_watermark_never_offloads(self):
        env, manager, add, copies = self.environment()
        add("old", 400)
        add("batch", 400)
        self.assertGreater(manager.used_bytes(), manager.low_watermark_bytes())
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 0)
        self.assertEqual(copies, [])

    def test_restore_estimate_cannot_trigger_early_offload(self):
        env, manager, add, copies = self.environment()
        add("old", 800)
        add("restore", 300)
        manager.transition("restore", EVICTION_PENDING)
        manager.transition("restore", CPU_RESIDENT)
        self.assertGreater(manager.used_bytes() + 300, manager.high_watermark_bytes())
        self.assertEqual(env["rebalance_cache_memory"](["restore"]), 0)
        self.assertEqual(copies, [])

    def test_prefetch_admission_can_reclaim_at_executor_boundary_below_high(self):
        env, manager, add, copies = self.environment()
        add("old", 400)
        add("batch", 400)
        free = 300 * 1024 ** 2
        env["target_model"].device.type = "cuda"
        env["torch"] = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (free, free)))
        self.assertEqual(env["rebalance_cache_memory"](["batch"], min_free_bytes=free + 100), 1)
        self.assertEqual(len(copies), 1)
        self.assertEqual(manager.get("batch").state, GPU_RESIDENT)
        self.assertEqual(manager.counters["emergency_evictions"], 0)

    def test_memory_guard_cannot_trigger_early_offload(self):
        env, manager, add, copies = self.environment()
        add("old", 400)
        add("batch", 400)
        manager.headroom_bytes = 512 * 1024 ** 2
        self.assertFalse(manager.update_pressure(800, guard_free_bytes=300 * 1024 ** 2))
        self.assertFalse(manager.needs_eviction())
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 0)
        self.assertEqual(copies, [])

    def gpu_free(self, env, free_bytes):
        env["target_model"].device.type = "cuda"
        env["torch"] = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (free_bytes, 1024 ** 3)))

    def test_emergency_offloads_below_low_but_protects_active_and_reserved_kv(self):
        env, manager, add, copies = self.environment()
        for pid in ("active", "reserved", "batch", "old"):
            add(pid, 100)
        env["residency"].activate(["active"])
        env["residency"].reserve(["reserved"])
        self.assertLess(manager.used_bytes(), manager.low_watermark_bytes())
        self.gpu_free(env, 255 * 1024 ** 2)
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 1)
        self.assertEqual(manager.state_of("old"), CPU_RESIDENT)
        for pid in ("active", "reserved", "batch"):
            self.assertEqual(manager.state_of(pid), GPU_RESIDENT)
        self.assertEqual(len(copies), 1)
        self.assertEqual(manager.counters["emergency_eviction_rounds"], 1)
        self.assertEqual(manager.counters["emergency_evictions"], 1)

    def test_exact_256_mib_free_does_not_override_high_watermark(self):
        env, _, add, copies = self.environment()
        add("old", 400)
        add("batch", 400)
        self.gpu_free(env, 256 * 1024 ** 2)
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 0)
        self.assertEqual(copies, [])

    def test_executor_refresh_clears_a_stale_planner_emergency(self):
        env, manager, add, copies = self.environment()
        add("old", 400)
        add("batch", 400)
        self.assertTrue(manager.update_pressure(800, guard_free_bytes=255 * 1024 ** 2))
        self.gpu_free(env, 300 * 1024 ** 2)
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 0)
        self.assertFalse(manager.emergency)
        self.assertEqual(copies, [])

    def test_emergency_with_no_eligible_victims_is_counted(self):
        env, manager, add, copies = self.environment()
        add("batch", 400)
        self.gpu_free(env, 255 * 1024 ** 2)
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 0)
        self.assertEqual(copies, [])
        self.assertEqual(manager.counters["emergency_blocked_rounds"], 1)

    def test_emergency_sample_does_not_count_as_a_kv_high_watermark_sample(self):
        env, manager, add, _ = self.environment()
        add("old", 400)
        self.gpu_free(env, 255 * 1024 ** 2)
        engine_closure("observe_memory", env)
        self.assertTrue(env["observe_memory"]())
        self.assertFalse(manager.high)
        self.assertEqual(env["scheduler_metrics"]["residency_high_samples"], 0)
        self.assertEqual(manager.counters["emergency_pressure_samples"], 1)

    def test_reaching_high_watermark_trims_to_low(self):
        env, manager, add, copies = self.environment()
        add("old", 200)
        add("batch", 700)
        self.assertEqual(manager.used_bytes(), manager.high_watermark_bytes())
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 1)
        self.assertEqual(manager.used_bytes(), 700)
        self.assertEqual(len(copies), 1)

    def test_current_batch_and_reserved_sessions_remain_protected(self):
        env, manager, add, _ = self.environment()
        add("reserved", 300)
        add("batch", 400)
        add("old", 300)
        env["residency"].reserve(["reserved"])
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 1)
        self.assertEqual(manager.state_of("reserved"), GPU_RESIDENT)
        self.assertEqual(manager.state_of("batch"), GPU_RESIDENT)
        self.assertEqual(manager.state_of("old"), CPU_RESIDENT)

    def test_eviction_does_not_restart_between_low_and_high(self):
        env, manager, add, copies = self.environment()
        add("old", 200)
        add("batch", 700)
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 1)
        add("new", 100)
        self.assertEqual(manager.used_bytes(), 800)
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 0)
        self.assertEqual(len(copies), 1)

    def test_fresh_prefill_records_its_actual_last_use(self):
        env, manager, add, _ = self.environment()
        add("fresh", 400)
        manager.release("fresh")
        env.update(cache_bytes_per_token=lambda: 10,
                   target_model=SimpleNamespace(device=SimpleNamespace(type="cpu")),
                   RESTORE_PENDING=RESTORE_PENDING)
        env["transfers"].metrics = {}
        engine_closure("finish_cache_phase", env)
        env["finish_cache_phase"]([{"proc_id": "fresh"}])
        self.assertEqual(manager.get("fresh").last_used_tick, 1)
        self.assertTrue(manager.get("fresh").protected(1))
        self.assertFalse(manager.get("fresh").protected(2))

    def test_completed_round_records_cost_and_acceptance_for_offload(self):
        env, manager, add, _ = self.environment()
        add("p", 400)
        env.update(cache_bytes_per_token=lambda: 10, accept_stats_lock=threading.Lock(),
                   accept_stats={"p": [3, 7]}, RESTORE_PENDING=RESTORE_PENDING)
        env["transfers"].metrics = {}
        engine_closure("finish_cache_phase", env)
        env["finish_cache_phase"]([dict(proc_id="p", task_type="verify", lag=4., transport_rtt=1.)])
        record = manager.get("p")
        self.assertEqual((record.draft_total_time, record.transport_rtt, record.accept_prob), (4., 1., .5))
        self.assertGreater(record.last_used_time, 0.)

    def test_failed_top_candidate_does_not_prevent_reclaiming_from_next(self):
        env, manager, add, copies = self.environment()
        for pid, size in (("fails", 200), ("works", 200), ("batch", 600)):
            add(pid, size)
        manager.note_request_cost("fails", draft_total_time=100)
        original = env["transfers"].submit_d2h
        def offload(cache):
            if cache is env["kv_cache_manager"]._past_key_values["fails"]:
                raise RuntimeError("copy failed")
            return original(cache)
        env["transfers"].submit_d2h = offload
        self.assertEqual(env["rebalance_cache_memory"](["batch"]), 1)
        self.assertEqual(manager.state_of("works"), CPU_RESIDENT)
        self.assertEqual(manager.state_of("fails"), GPU_RESIDENT)
        self.assertEqual(env["scheduler_metrics"]["kv_residency_eviction_errors"], 1)

    def test_fcfs_advances_recency_and_allows_previous_batch_eviction(self):
        env, manager, add, copies = self.environment()
        add("a", 400)
        add("b", 400)
        ingress = queue.Queue()
        for pid in ("a", "b"):
            ingress.put({"task_type": "verify", "proc_id": pid, "gamma": 8})
        ingress.put(None)
        setup, samples = [], []

        def handle(batch):
            pid = batch[0]["proc_id"]
            env["rebalance_cache_memory"]([pid])
            manager.note_used(pid, env["plan_sequence"]["tick"])
            if pid == "a":
                add("new", 200)
                env["residency"].reserve(["new"])

        env.update(use_strict_fcfs=True, request_queue=ingress,
            queue=queue, time=time, energy_service=None,
            lifecycle=SimpleNamespace(touch=lambda pid: None),
            canonicalize_ingress=lambda req: req,
            handle_request_batch=handle, flush_session_releases=lambda: None,
            ensure_cache_management=lambda: setup.append(True),
            observe_memory=lambda: samples.append(manager.used_bytes()),
            cost_features=lambda req: (0, 8))
        run_fcfs_loop(env)
        self.assertEqual(env["plan_sequence"]["tick"], 3)
        self.assertEqual(len(setup), 2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(manager.state_of("a"), CPU_RESIDENT)
        self.assertEqual(len(copies), 1)


if __name__ == "__main__":
    unittest.main()
