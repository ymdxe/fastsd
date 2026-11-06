"""BYPASS (residency) contracts: nothing is allocated, nothing is submitted."""

import unittest

try:
    import torch  # noqa: F401
    from src.cache_transfer import CacheTransferManager
    DEPS = True
except ImportError:  # pragma: no cover
    DEPS = False


@unittest.skipUnless(DEPS, "requires torch")
class LazyAllocationTests(unittest.TestCase):
    def test_construction_allocates_no_cuda_state(self):
        # warmup_pool now defaults to False; a CPU manager must stay inert.
        manager = CacheTransferManager("cpu", warmup_pool=True, event_pool_size=8)
        try:
            self.assertFalse(manager.enabled)
            self.assertIsNone(manager.gpu_arena)
            self.assertEqual(manager._events, [])
            self.assertIsNone(manager.h2d_stream)
            self.assertIsNone(manager.prefetch_stream)
            self.assertIsNone(manager.d2h_stream)
        finally:
            manager.close()

    def test_warmup_is_a_noop_without_cuda(self):
        manager = CacheTransferManager("cpu")
        try:
            manager.warmup_prefetch_pool()
            self.assertIsNone(manager.gpu_arena)
            self.assertEqual(manager.metrics.get("prefetch_pool_unavailable", 0), 0)
        finally:
            manager.close()

    def test_protect_forward_mode_is_removed(self):
        with self.assertRaises(ValueError):
            CacheTransferManager("cpu", mode="protect_forward")

    def test_supported_modes_construct(self):
        for mode in ("overlap", "cpu_prepare"):
            manager = CacheTransferManager("cpu", mode=mode)
            try:
                self.assertEqual(manager.mode, mode)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
