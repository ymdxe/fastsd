import unittest
from unittest import mock

from src.cache_offload_policy import (
    should_offload_target_cache,
    target_cache_offload_disabled,
)


class CacheOffloadPolicyTests(unittest.TestCase):
    def test_residency_is_the_default_for_every_stage(self):
        for stage in ("prefill", "verify"):
            self.assertFalse(should_offload_target_cache(stage, pid="req-1"))

    def test_pressure_enables_offload(self):
        for stage in ("prefill", "verify"):
            self.assertTrue(
                should_offload_target_cache(stage, pid="req-1", memory_pressure=True))

    def test_continuing_chunk_never_leaves_the_gpu(self):
        for stage in ("prefill", "verify"):
            self.assertFalse(
                should_offload_target_cache(
                    stage, continuing=True, memory_pressure=True))

    def test_unknown_stage_does_not_offload(self):
        self.assertFalse(
            should_offload_target_cache("other", pid="req-1", memory_pressure=True))

    def test_env_flag_disables_all_target_cache_offload(self):
        with mock.patch.dict("os.environ", {"FASTSD_DISABLE_TARGET_CACHE_OFFLOAD": "1"}):
            self.assertTrue(target_cache_offload_disabled())
            for stage in ("prefill", "verify"):
                self.assertFalse(
                    should_offload_target_cache(stage, pid="req-1", memory_pressure=True))

    def test_legacy_pinned_set_argument_is_ignored(self):
        # Placement is decided by the residency manager, not by membership in
        # the old resident set.
        self.assertFalse(
            should_offload_target_cache("verify", pid="req-1", pinned_gpu_pids=set()))
        self.assertTrue(
            should_offload_target_cache(
                "verify", pid="req-1", pinned_gpu_pids={"req-1"}, memory_pressure=True))


if __name__ == "__main__":
    unittest.main()
