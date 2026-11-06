"""Online conservative overlap (no calibration file) and its self-disabling gate."""

import unittest

try:
    import torch  # noqa: F401
    from src.cache_transfer import CacheTransferManager, OverlapController
    DEPS = True
except ImportError:  # pragma: no cover
    DEPS = False

SHAPE = ("verify", 1, (4,), (100,), "torch.bfloat16", "sdpa_per_sequence")


@unittest.skipUnless(DEPS, "requires torch")
class OnlineAllowanceTests(unittest.TestCase):
    def make(self, fraction=0.25):
        controller = OverlapController(4 << 20, 64 << 20, 512 << 20,
                                       online_fraction=fraction)
        controller.context_for(SHAPE)
        return controller

    def test_strict_gate_without_online_fraction(self):
        controller = self.make(fraction=0.0)
        self.assertEqual(controller.allowance(computing=True), 0)
        self.assertEqual(controller.status()["reason"], "uncalibrated_context")

    def test_online_quota_allows_a_bounded_compute_window_burst(self):
        controller = self.make()
        allowance = controller.allowance(computing=True)
        self.assertGreater(allowance, 0)
        self.assertLessEqual(allowance, controller.group_bytes)
        self.assertEqual(controller.status()["reason"], "online_conservative")

    def test_idle_window_keeps_the_full_group(self):
        controller = self.make()
        controller.allowance(computing=True)
        self.assertEqual(controller.allowance(computing=False), controller.group_bytes)

    def test_blocked_reason_still_pauses_everything(self):
        controller = self.make()
        self.assertEqual(
            controller.allowance(computing=True, blocked_reason="demand_h2d_pending"), 0)

    def test_repeated_disturbance_disables_the_shape(self):
        controller = self.make()
        controller.observe(SHAPE, 100.0, speculative=False)   # baseline
        self.assertGreater(controller.allowance(computing=True), 0)
        for _ in range(6):
            controller.observe(SHAPE, 300.0, speculative=True)  # 200% slower
        self.assertEqual(controller.allowance(computing=True), 0)
        self.assertIn("online_disturbance_limit", controller.status()["reason"])
        self.assertTrue(controller.status()["online_paused_shapes"])

    def test_one_clean_forward_does_not_trigger_the_gate(self):
        controller = self.make()
        controller.observe(SHAPE, 100.0, speculative=False)
        controller.observe(SHAPE, 101.0, speculative=True)     # 1% < 2% budget
        self.assertGreater(controller.allowance(computing=True), 0)
        self.assertEqual(controller.status()["online_paused_shapes"], [])

    def test_three_violations_halve_the_online_rate(self):
        controller = self.make()
        start = controller.online_rate
        controller.observe(SHAPE, 100.0, speculative=False)
        for _ in range(3):
            controller.observe(SHAPE, 300.0, speculative=True)
        self.assertLess(controller.online_rate, start)


@unittest.skipUnless(DEPS, "requires torch")
class ManagerWiringTests(unittest.TestCase):
    def test_manager_forwards_the_online_fraction(self):
        manager = CacheTransferManager("cpu", online_fraction=0.25,
                                       idle_rate_bytes_per_second=512 << 20)
        try:
            self.assertAlmostEqual(manager.controller.online_rate, (512 << 20) * 0.25)
        finally:
            manager.close()

    def test_default_manager_stays_strict(self):
        manager = CacheTransferManager("cpu")
        try:
            self.assertEqual(manager.controller.online_fraction, 0.0)
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
