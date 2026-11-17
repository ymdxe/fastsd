"""Tests for the lazy KV residency activation boundary."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.cache_residency import ACTIVE, BYPASS, SHADOW, ResidencyManager


def engine_closure(name, namespace):
    source = Path(__file__).resolve().parents[1] / "src" / "engine.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class LazyActivationBoundaryTests(unittest.TestCase):
    def test_shadow_and_bypass_reject_record_creation(self):
        for mode in ("lazy", "bypass"):
            manager = ResidencyManager(budget_bytes=1000, activation_mode=mode)
            self.assertIsNone(manager.note_created("pid", nbytes=10))
            self.assertIsNone(manager.get("pid"))

    def test_manager_activation_is_sticky_and_one_shot(self):
        manager = ResidencyManager(budget_bytes=1000, activation_mode="lazy")
        self.assertTrue(manager.activate())
        self.assertFalse(manager.activate())
        self.assertTrue(manager.mark_started())
        self.assertFalse(manager.mark_started())

    def test_eager_starts_active(self):
        manager = ResidencyManager(budget_bytes=1000, activation_mode="eager")
        self.assertEqual(manager.mode, ACTIVE)
        self.assertTrue(manager.mark_started())

    def make(self, *, free, used=0, activation_mode="lazy"):
        manager = ResidencyManager(
            budget_bytes=1000,
            high_ratio=0.90,
            activation_mode=activation_mode,
        )
        events = []
        setup_state = {"started": False}
        def ensure():
            if setup_state["started"]:
                return False
            setup_state["started"] = True
            events.append("ensure")
            return True
        metrics = defaultdict(int)
        namespace = dict(
            residency_manager=manager,
            target_model=SimpleNamespace(device=SimpleNamespace(type="cuda")),
            scheduler_metrics=metrics,
            plan_sequence={"tick": 0},
            plan_boundary_free={"value": None},
            _sample_free_memory=lambda: free,
            shadow_live_kv_bytes=lambda: used,
            ensure_cache_management=ensure,
            observe_memory=lambda: events.append("observe"),
            time=SimpleNamespace(perf_counter=lambda: 1.0),
        )
        namespace["prepare_plan_boundary"] = engine_closure(
            "prepare_plan_boundary", namespace
        )
        return namespace, manager, events

    def test_high_free_stays_shadow_without_setup(self):
        env, manager, events = self.make(free=4 * 1024 ** 3)
        self.assertFalse(env["prepare_plan_boundary"]())
        self.assertEqual(manager.mode, SHADOW)
        self.assertEqual(events, [])
        self.assertEqual(env["scheduler_metrics"]["shadow_pressure_samples"], 1)
        self.assertFalse(env["scheduler_metrics"]["lazy_activation_triggered"])

    def test_300_mib_stays_shadow(self):
        env, manager, events = self.make(free=300 * 1024 ** 2)
        env["prepare_plan_boundary"]()
        self.assertEqual(manager.mode, SHADOW)
        self.assertEqual(events, [])

    def test_255_mib_activates_once_and_reuses_boundary_sample(self):
        env, manager, events = self.make(free=255 * 1024 ** 2)
        self.assertTrue(env["prepare_plan_boundary"]())
        self.assertEqual(manager.mode, ACTIVE)
        self.assertEqual(events, ["ensure"])
        self.assertEqual(env["plan_boundary_free"]["value"], 255 * 1024 ** 2)
        self.assertTrue(env["scheduler_metrics"]["lazy_activation_triggered"])
        self.assertEqual(env["scheduler_metrics"]["lazy_activation_tick"], 1)

        # ACTIVE is sticky: no second activation/setup.
        self.assertFalse(env["prepare_plan_boundary"]())
        self.assertEqual(events, ["ensure", "observe"])
        self.assertEqual(manager.mode, ACTIVE)

    def test_budget_high_watermark_can_activate_before_emergency(self):
        env, manager, events = self.make(free=4 * 1024 ** 3, used=900)
        env["prepare_plan_boundary"]()
        self.assertEqual(manager.mode, ACTIVE)
        self.assertEqual(events, ["ensure"])
        self.assertEqual(env["scheduler_metrics"]["lazy_activation_free_bytes"], 4 * 1024 ** 3)

    def test_bypass_never_activates(self):
        env, manager, events = self.make(free=1, activation_mode="bypass")
        self.assertFalse(env["prepare_plan_boundary"]())
        self.assertEqual(manager.mode, BYPASS)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
