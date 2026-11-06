"""Unit tests for the explicit KV residency state machine."""

import unittest

from src.cache_residency import (
    ACTIVE,
    BYPASS,
    CPU_RESIDENT,
    EVICTION_PENDING,
    GPU_RESIDENT,
    PREFETCH_PENDING,
    RESTORE_PENDING,
    InvalidTransition,
    ResidencyManager,
)


def manager(budget=1000, **kwargs):
    return ResidencyManager(budget_bytes=budget, **kwargs)


class BudgetTests(unittest.TestCase):
    def test_zero_budget_disables_management(self):
        m = manager(budget=0)
        self.assertFalse(m.enabled)
        self.assertFalse(m.active)
        self.assertEqual(m.mode, BYPASS)
        self.assertFalse(m.mark_started())
        m.note_created("p", nbytes=10)
        self.assertEqual(m.pending_restore(["p"]), [])

    def test_budget_enables_management_without_pressure(self):
        m = manager(budget=1000)
        self.assertTrue(m.enabled)
        self.assertTrue(m.active)
        self.assertEqual(m.mode, ACTIVE)
        self.assertFalse(m.needs_eviction())

    def test_pressure_samples_do_not_gate_management(self):
        m = manager(budget=1000)
        m.note_created("p", nbytes=950)
        self.assertTrue(m.update_pressure(950))
        self.assertTrue(m.needs_eviction())
        m.note_mutated("p", version=1, valid_len=10)
        self.assertTrue(m.needs_eviction())

    def test_high_watermark_uses_the_budget(self):
        m = manager(budget=1000)
        self.assertEqual(m.high_watermark_bytes(), 900)
        self.assertEqual(m.low_watermark_bytes(), 750)
        m.note_created("a", nbytes=800)
        self.assertFalse(m.needs_eviction())
        m.note_created("b", nbytes=150)
        self.assertTrue(m.needs_eviction())

    def test_deficit_accounts_for_bytes_that_must_be_restored(self):
        m = manager(budget=1000)
        m.note_created("a", nbytes=800)
        self.assertEqual(m.deficit_bytes(), 50)
        # Admitting a 100-byte restore must also make room for it.
        self.assertEqual(m.deficit_bytes(100), 150)

    def test_pending_restore_reports_non_gpu_sessions(self):
        m = manager(budget=1000)
        m.note_created("a", nbytes=100)
        m.note_created("b", nbytes=100)
        m.transition("b", EVICTION_PENDING)
        m.transition("b", CPU_RESIDENT)
        self.assertEqual(m.pending_restore(["a", "b", "missing"]), ["b"])
        self.assertEqual(m.cpu_resident_count(), 1)

    def test_guard_free_bytes_flags_high(self):
        m = manager(budget=1000, headroom_bytes=512)
        self.assertTrue(m.update_pressure(10, guard_free_bytes=100))
        self.assertEqual(m.counters["pressure_guard_trips"], 1)

    def test_mark_started_is_one_shot(self):
        m = manager(budget=1000)
        self.assertTrue(m.mark_started())
        self.assertFalse(m.mark_started())
        self.assertEqual(m.counters["activations"], 1)

    def test_ratio_validation(self):
        with self.assertRaises(ValueError):
            manager(budget=1000, high_ratio=0.5, low_ratio=0.6)


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.m = manager()
        self.m.note_created("p", nbytes=100, valid_len=10, bytes_per_token=10)

    def test_happy_path_round_trip(self):
        m = self.m
        m.transition("p", EVICTION_PENDING)
        self.assertTrue(m.get("p").in_transfer)
        m.transition("p", CPU_RESIDENT)
        self.assertFalse(m.get("p").in_transfer)
        self.assertEqual(m.get("p").cpu_valid_len, 10)
        m.transition("p", PREFETCH_PENDING)
        m.transition("p", RESTORE_PENDING)
        m.transition("p", GPU_RESIDENT)
        self.assertEqual(m.state_of("p"), GPU_RESIDENT)

    def test_prefetch_may_be_cancelled_back_to_cpu(self):
        m = self.m
        m.transition("p", EVICTION_PENDING)
        m.transition("p", CPU_RESIDENT)
        m.transition("p", PREFETCH_PENDING)
        m.transition("p", CPU_RESIDENT)
        self.assertEqual(m.state_of("p"), CPU_RESIDENT)

    def test_illegal_transition_is_rejected_and_counted(self):
        with self.assertRaises(InvalidTransition):
            self.m.transition("p", PREFETCH_PENDING)  # GPU -> prefetch is illegal
        self.assertEqual(self.m.counters["invalid_transitions"], 1)

    def test_unknown_session_transition_is_rejected(self):
        with self.assertRaises(InvalidTransition):
            self.m.transition("missing", CPU_RESIDENT)

    def test_force_state_only_accepts_durable_states(self):
        self.m.transition("p", EVICTION_PENDING)
        with self.assertRaises(InvalidTransition):
            self.m.force_state("p", EVICTION_PENDING)
        self.m.force_state("p", GPU_RESIDENT)
        self.assertEqual(self.m.state_of("p"), GPU_RESIDENT)
        self.assertFalse(self.m.get("p").in_transfer)

    def test_mutation_updates_size_from_bytes_per_token(self):
        self.m.note_mutated("p", version=3, valid_len=20)
        record = self.m.get("p")
        self.assertEqual(record.version, 3)
        self.assertEqual(record.nbytes, 200)


class VictimTests(unittest.TestCase):
    def make(self, pid, nbytes, used):
        self.m.note_created(pid, nbytes=nbytes)
        self.m.note_used(pid, used)

    def setUp(self):
        self.m = manager(budget=1000)
        self.make("a", 100, 1)
        self.make("b", 100, 2)
        self.make("c", 100, 3)

    def test_least_recently_used_first(self):
        self.assertEqual(self.m.select_victims(150, 10), ["a", "b"])

    def test_protected_sessions_are_skipped(self):
        self.m.protect("a", 10)
        self.assertEqual(self.m.select_victims(150, 10), ["b", "c"])

    def test_excluded_pids_are_skipped(self):
        self.assertEqual(self.m.select_victims(150, 10, exclude={"a"}), ["b", "c"])

    def test_only_gpu_resident_sessions_are_candidates(self):
        self.m.transition("a", EVICTION_PENDING)
        self.m.transition("a", CPU_RESIDENT)
        self.assertEqual(self.m.select_victims(100, 10), ["b"])

    def test_single_victim_is_returned_even_below_the_target(self):
        # One very small cache must not loop forever looking for the target.
        self.assertEqual(len(self.m.select_victims(10_000, 10)), 3)

    def test_used_bytes_counts_only_gpu_side_states(self):
        self.m.transition("a", EVICTION_PENDING)
        self.assertEqual(self.m.used_bytes(), 300)
        self.m.transition("a", CPU_RESIDENT)
        self.assertEqual(self.m.used_bytes(), 200)

    def test_release_drops_the_record(self):
        self.m.release("a")
        self.assertIsNone(self.m.state_of("a"))
        self.assertEqual(self.m.used_bytes(), 200)


class StatsTests(unittest.TestCase):
    def test_stats_reports_state_counts_without_mutating(self):
        m = manager(budget=1000)
        m.mark_started()
        m.note_created("a", nbytes=100)
        m.note_created("b", nbytes=50)
        m.transition("b", EVICTION_PENDING)
        stats = m.stats()
        self.assertEqual(stats["mode"], ACTIVE)
        self.assertEqual(stats["states"][GPU_RESIDENT], 1)
        self.assertEqual(stats["states"][EVICTION_PENDING], 1)
        self.assertEqual(stats["used_bytes"], 150)
        self.assertEqual(stats["residency_activations"], 1)

    def test_ratio_validation(self):
        with self.assertRaises(ValueError):
            manager(budget=1000, high_ratio=0.5, low_ratio=0.6)


if __name__ == "__main__":
    unittest.main()
