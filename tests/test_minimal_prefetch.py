"""H1 ownership, admission and demand fallback without a CUDA device."""
from collections import defaultdict
from types import SimpleNamespace
import unittest

from src.cache_protocol import PrefetchTarget
from src.cache_residency import (
    CPU_RESIDENT, EVICTION_PENDING, EMERGENCY_FREE_BYTES, GPU_RESIDENT,
    PREFETCH_PENDING, RESTORE_PENDING, ResidencyManager,
)
from src.minimal_prefetch import H1Prefetcher, H1Ticket
from src.scheduler_runtime import CacheResidency
from tests import test_round3_runtime as runtime_checks
from tests.test_round3_runtime import engine_closure


class Cache:
    def __init__(self, length=8):
        self.length = length
        self.source_ready = True

    def get_seq_length(self):
        return self.length

    def __eq__(self, other):
        raise AssertionError("cache equality must use object identity")


class CopyDriver:
    def __init__(self):
        self.free = 2 * EMERGENCY_FREE_BYTES
        self.free_checks = 0
        self.allocated, self.copied, self.released, self.consumed = [], [], [], []
        self.after_allocate = lambda: None
        self.allocate_error = self.submit_error = None

    def source_ready(self, cache):
        return cache.source_ready

    def describe(self, cache):
        return cache.length * 8

    def free_bytes(self):
        self.free_checks += 1
        return self.free

    def allocate(self, pid, snapshot, size, prepared_at):
        if self.allocate_error:
            raise self.allocate_error
        ticket = H1Ticket(pid, snapshot, Cache(snapshot.valid_len),
                          SimpleNamespace(done=False), object(), size, prepared_at)
        self.allocated.append(ticket)
        self.after_allocate()
        return ticket

    def submit(self, ticket):
        ticket.submitted = True
        ticket.fenced = True
        ticket.submitted_bytes = ticket.nbytes
        self.copied.append(ticket)
        if self.submit_error:
            raise self.submit_error

    def ready(self, ticket):
        return ticket.copy_done.done

    def copy_ms(self, ticket):
        assert self.ready(ticket)
        return 3.0

    def consume(self, ticket, stream):
        assert self.ready(ticket), "never consume a pending copy"
        self.consumed.append(ticket)
        return ticket.destination_cache

    def release(self, ticket):
        self.released.append(ticket)
        ticket.snapshot = ticket.destination_cache = ticket.copy_done = ticket.copy_start = None
        ticket.state = "retired"

    def finish(self, ticket):
        ticket.copy_done.done = True

    def is_oom(self, error):
        return isinstance(error, MemoryError)


class MinimalPrefetchTests(unittest.TestCase):
    def setUp(self):
        self.residency = CacheResidency()
        self.manager = ResidencyManager(budget_bytes=10000)
        self.caches, self.metrics = {}, defaultdict(int)
        self.driver = CopyDriver()
        self.tick, self.now = 5, 10.0
        self.worker = H1Prefetcher(
            residency=self.residency, manager=self.manager, get_cache=self.caches.get,
            driver=self.driver, metrics=self.metrics, max_bytes=1024,
            tick=lambda: self.tick, clock=lambda: self.now)
        self.add("p")

    def add(self, pid):
        self.caches[pid] = Cache()
        self.manager.note_created(pid, nbytes=64)
        self.manager.transition(pid, EVICTION_PENDING)
        self.manager.transition(pid, CPU_RESIDENT)

    def snapshot(self, pid="p"):
        return self.residency.snapshot_locked(pid, self.caches[pid])

    def prepare(self, pid="p"):
        self.assertTrue(self.worker.prepare(pid))
        return self.worker.tickets[pid]

    def test_only_h1_enqueued_once_and_new_intent_does_not_cancel_old_ticket(self):
        ticket = self.prepare()
        self.worker.refresh([PrefetchTarget("p"), PrefetchTarget("q"),
                             PrefetchTarget("h2", 2), PrefetchTarget("h3", 3), PrefetchTarget("q")])
        self.assertEqual(list(self.worker.queue), ["p", "q"])
        self.worker.refresh([])
        self.assertIs(self.residency.candidates["p"], ticket)
        self.assertEqual(ticket.state, "copying")

    def test_gpu_protected_active_reserved_and_in_transfer_are_excluded(self):
        for reason in ("gpu", "planned", "active", "reserved", "transfer", "producer"):
            with self.subTest(reason=reason):
                record = self.manager.get("p")
                record.state, record.planned_tick, record.in_transfer = CPU_RESIDENT, -1, False
                self.residency.active.clear()
                self.residency.reserved.clear()
                self.caches["p"].source_ready = True
                if reason == "gpu":
                    record.state = GPU_RESIDENT
                elif reason == "planned":
                    record.planned_tick = self.tick
                elif reason == "active":
                    self.residency.active.add("p")
                elif reason == "reserved":
                    self.residency.reserved.add("p")
                elif reason == "transfer":
                    record.in_transfer = True
                else:
                    self.caches["p"].source_ready = False
                self.assertFalse(self.worker.prepare("p"))
                self.assertEqual(self.driver.free_checks, 0)

    def test_exact_256mib_admission_and_one_free_check(self):
        self.driver.free = EMERGENCY_FREE_BYTES + 64
        self.prepare()
        self.assertEqual(self.driver.free_checks, 1)
        self.assertEqual(self.worker.reserved_bytes, 64)
        self.assertEqual(self.metrics["prefetch_bytes"], 64)

    def test_shortage_requests_eviction_rechecks_then_skips(self):
        requests = []
        self.worker.reclaim = requests.append
        self.driver.free = EMERGENCY_FREE_BYTES + 63
        self.assertFalse(self.worker.prepare("p"))
        self.assertEqual(requests, [EMERGENCY_FREE_BYTES + 64])
        self.assertEqual(self.driver.free_checks, 2)
        self.assertEqual(self.driver.allocated, [])
        self.assertEqual(self.metrics["prefetch_skipped_no_memory"], 1)

    def test_reclaim_can_make_room_before_allocation(self):
        self.driver.free = 0
        self.worker.reclaim = lambda required: setattr(self.driver, "free", required)
        self.prepare()
        self.assertEqual(self.driver.free_checks, 2)

    def test_snapshot_revalidated_after_allocation_before_dma(self):
        self.driver.after_allocate = lambda: self.residency.bump(["p"])
        self.assertFalse(self.worker.prepare("p"))
        self.assertEqual(self.driver.copied, [])
        self.assertEqual(self.worker.reserved_bytes, 0)
        self.assertEqual(self.metrics["prefetch_skipped_stale"], 1)

    def test_ready_claim_consumes_once_without_touching_source_or_length(self):
        ticket = self.prepare()
        ticket.copy_done.done = True
        self.now += .02
        source = self.caches["p"]
        self.assertIs(self.worker.claim("p", self.snapshot()), ticket)
        self.assertIsNone(self.worker.claim("p", self.snapshot()))
        self.assertEqual(self.worker.consume(ticket, None).length, source.length)
        self.assertEqual(source.length, 8)
        self.assertEqual(self.metrics["prefetch_ready_hit"], 1)
        self.assertAlmostEqual(self.metrics["prefetch_lead_ms"], 20.)
        self.assertEqual(self.metrics["prefetch_copy_ms"], 3.)
        self.worker.retire(ticket)
        self.worker.retire(ticket)
        self.assertEqual(len(self.driver.released), 1)
        self.assertEqual(self.worker.reserved_bytes, 0)

    def test_pending_claim_never_waits_and_retains_both_buffers_until_done(self):
        ticket = self.prepare()
        source, destination = ticket.snapshot.cache, ticket.destination_cache
        self.assertIsNone(self.worker.claim("p", self.snapshot()))
        self.assertEqual(ticket.state, "draining")
        self.assertIs(ticket.snapshot.cache, source)
        self.assertIs(ticket.destination_cache, destination)
        self.assertFalse(self.worker.prepare("p"))
        self.worker.collect()
        self.assertEqual(self.driver.released, [])
        ticket.copy_done.done = True
        self.worker.collect()
        self.assertIsNone(ticket.snapshot)
        self.assertEqual(self.metrics["prefetch_pending_hit"], 1)
        self.assertEqual(self.worker.reserved_bytes, 0)

    def test_each_snapshot_component_and_session_release_invalidate(self):
        for change in ("identity", "epoch", "version", "length", "release"):
            with self.subTest(change=change):
                self.setUp()
                ticket = self.prepare()
                if change == "identity":
                    self.caches["p"] = Cache()
                elif change == "epoch":
                    self.residency.epochs["p"] += 1
                elif change == "version":
                    self.residency.bump(["p"])
                elif change == "length":
                    self.caches["p"].length += 1
                else:
                    self.residency.retire_session_locked("p")
                    del self.caches["p"]
                self.worker.collect()
                self.assertEqual(self.driver.released, [])
                ticket.copy_done.done = True
                self.worker.collect()
                self.assertEqual(self.worker.tickets, {})
                self.assertEqual(self.metrics["prefetch_skipped_stale"], 1)

    def test_budget_counts_draining_destinations_and_allocation_oom_recovers(self):
        self.worker.max_bytes = 64
        ticket = self.prepare()
        self.worker.claim("p", self.snapshot())
        self.add("q")
        self.assertFalse(self.worker.prepare("q"))
        ticket.copy_done.done = True
        self.worker.collect()
        self.driver.allocate_error = MemoryError("allocation race")
        self.assertFalse(self.worker.prepare("q"))
        self.assertEqual(self.metrics["prefetch_allocation_oom"], 1)
        self.assertEqual(self.worker.reserved_bytes, 0)
        self.driver.allocate_error = None
        self.prepare("q")

    def test_partial_submission_failure_is_drained(self):
        self.driver.submit_error = RuntimeError("copy submit failed")
        self.assertFalse(self.worker.prepare("p"))
        ticket = self.worker.tickets["p"]
        self.assertEqual(self.residency.candidates, {})
        self.assertEqual(self.worker.reserved_bytes, 64)
        self.worker.close()
        self.assertEqual(self.worker.reserved_bytes, 0)
        self.assertEqual(self.metrics["prefetch_async_failed"], 1)

    def engine_environment(self):
        demand, binds = [], []
        demand_cache = Cache()
        env = dict(residency_manager=self.manager, residency=self.residency,
            kv_cache_manager=SimpleNamespace(_past_key_values=self.caches),
            cache_lock=self.residency.lock, plan_sequence={"tick": self.tick},
            use_h1_prefetch=True, h1_prefetch=self.worker, scheduler_metrics=self.metrics,
            compute_stream=None, phase_leases=[], phase_prefetch_tickets=[],
            GPU_RESIDENT=GPU_RESIDENT, CPU_RESIDENT=CPU_RESIDENT,
            EVICTION_PENDING=EVICTION_PENDING, RESTORE_PENDING=RESTORE_PENDING,
            PREFETCH_PENDING=PREFETCH_PENDING,
            transfers=SimpleNamespace(
                submit_h2d=lambda snapshot, **kw: demand.append(snapshot) or SimpleNamespace(),
                bind_to_consumer=lambda ticket, **kw: binds.append(ticket) or SimpleNamespace(cache=demand_cache),
                protect_source_write=lambda *a, **kw: None, retain_cache=lambda *a, **kw: None))
        self.metrics["prefetch_demand_miss_by_horizon"] = {}
        self.metrics["prefetch_ready_by_horizon"] = {}
        self.metrics["prefetch_pending_by_horizon"] = {}
        engine_closure("acquire_cache", env)
        return env, demand, binds

    def test_engine_ready_path_never_binds_legacy_consumer_or_advances_kv(self):
        ticket = self.prepare()
        ticket.copy_done.done = True
        env, demand, binds = self.engine_environment()
        req = dict(proc_id="p", task_type="verify")
        cache = env["acquire_cache"](req)
        self.assertIs(cache, ticket.destination_cache)
        self.assertIs(env["acquire_cache"](req), cache)
        self.assertEqual(cache.length, 8)
        self.assertEqual(demand + binds, [])
        self.assertEqual(self.manager.get("p").state, GPU_RESIDENT)
        self.worker.retire(ticket)
        self.assertEqual(cache.length, 8)

    def test_engine_pending_copy_falls_back_before_event_done(self):
        ticket = self.prepare()
        env, demand, binds = self.engine_environment()
        cache = env["acquire_cache"](dict(proc_id="p", task_type="verify"))
        self.assertFalse(ticket.copy_done.done)
        self.assertIs(self.caches["p"], cache)
        self.assertEqual(len(demand), 1)
        self.assertEqual(len(binds), 1)
        self.assertEqual(self.metrics["prefetch_demand_miss"], 1)
        ticket.copy_done.done = True
        self.worker.collect()
        self.assertIs(self.caches["p"], cache)
        self.assertEqual(cache.length, 8)

    def test_h1_predictor_builds_only_one_plan_despite_legacy_horizon_env(self):
        fixtures = runtime_checks.EnginePredictionIntegrationTests()
        env, _, submitted, _ = fixtures.environment(fixtures.item("p"))
        env.update(use_h1_prefetch=True, PREFETCH_LOOKAHEAD_ROUNDS=10)
        env["transfers"].enabled, env["transfers"].mode = True, "overlap"
        env["refresh_prefetch"]()
        self.assertEqual(env["scheduler_metrics"]["lookahead_build_count"], 1)
        self.assertEqual(len(submitted), 1)

    def test_h1_resource_start_does_not_warm_legacy_arena(self):
        env = dict(residency_manager=self.manager, residency=self.residency,
            kv_cache_manager=SimpleNamespace(_past_key_values={}),
            use_h1_prefetch=True, use_strict_fcfs=False, prefetch_enabled=True,
            self=SimpleNamespace(args=SimpleNamespace(kv_prefetch=True)),
            cache_bytes_per_token=lambda: 8, scheduler_metrics=self.metrics,
            transfers=SimpleNamespace(enabled=True, mode="overlap"),
            prefetch_worker=SimpleNamespace(start=lambda: self.metrics.update(started=1)))
        engine_closure("ensure_cache_management", env)()
        self.assertEqual(self.metrics["started"], 1)

    def test_cuda_copy_roundtrip_without_arena(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        from src.cache_views import dynamic_cache_from_views
        from src.minimal_prefetch import CudaH1Copy
        source = torch.arange(64, dtype=torch.float32).reshape(1, 2, 8, 4).pin_memory()
        self.caches["p"] = dynamic_cache_from_views([(source, (source + 1).pin_memory())])
        self.worker.driver = driver = CudaH1Copy("cuda:0")
        ticket = self.prepare()
        driver.stream.synchronize()  # test harness only; demand never waits
        self.assertIs(self.worker.claim("p", self.snapshot()), ticket)
        cache = self.worker.consume(ticket, torch.cuda.current_stream())
        for original, restored in zip(self.caches["p"].to_legacy_cache()[0], cache.to_legacy_cache()[0]):
            torch.testing.assert_close(restored.cpu(), original)
        self.worker.retire(ticket)
        self.assertEqual(self.worker.reserved_bytes, 0)

    def test_cuda_64_restored_sessions_match_baseline_forward(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        from transformers import LlamaConfig, LlamaForCausalLM
        from src.cache_views import dynamic_cache_from_views
        from src.kvcache_batching import KVCacheModel_batching
        from src.kvcache_varlen import varlen_generate
        from src.minimal_prefetch import CudaH1Copy
        torch.manual_seed(17)
        config = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to("cuda").eval()
        caches = KVCacheModel_batching(model, temperature=1.0)
        caches.vocab_size, caches.bounded_probabilities = 97, True
        pids = [f"session-{i}" for i in range(64)]
        inputs = torch.randint(1, 97, (64, 12), device="cuda")
        with torch.no_grad():
            varlen_generate(caches, model, [row[None, :8] for row in inputs], pids,
                            0, is_prefill=True, sample_next=False)
            expected = model(inputs).logits[:, 8:].softmax(-1)
        self.caches.clear()
        self.worker.driver = driver = CudaH1Copy("cuda:0")
        self.worker.max_bytes = 16 * 1024 ** 2
        for pid in pids:
            self.add(pid)
            original = caches._past_key_values[pid]
            self.caches[pid] = dynamic_cache_from_views([
                tuple(t.cpu().pin_memory() for t in pair) for pair in original.to_legacy_cache()])
        env, demand, binds = self.engine_environment()
        env["compute_stream"] = torch.cuda.current_stream()
        try:
            for pid in pids:
                self.prepare(pid)
            driver.stream.synchronize()  # force the ready case in this harness
            for pid in pids:
                restored = env["acquire_cache"](dict(proc_id=pid, task_type="verify"))
                self.assertEqual(restored.get_seq_length(), 8)
                caches._past_key_values[pid] = restored
            with torch.no_grad():
                varlen_generate(caches, model, [row[None, 8:] for row in inputs], pids,
                                0, sample_next=False)
            for index, pid in enumerate(pids):
                self.assertEqual(caches._past_key_values[pid].get_seq_length(), 12)
                torch.testing.assert_close(caches.probability_rows(pid, 8, 12),
                                           expected[index:index + 1], atol=2e-5, rtol=2e-4)
            self.assertEqual(demand + binds, [])
            self.assertEqual(self.metrics["prefetch_ready_hit"], 64)
        finally:
            self.worker.close()


if __name__ == "__main__":
    unittest.main()
