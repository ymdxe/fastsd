import unittest
from unittest import mock

from src.cache_offload_policy import should_offload_target_cache


class CacheOffloadPolicyTests(unittest.TestCase):
    def test_prefill_always_offloads_target_cache(self):
        self.assertTrue(should_offload_target_cache("prefill", pid="req-1"))

    def test_verify_offloads_by_default(self):
        self.assertTrue(should_offload_target_cache("verify", pid="req-1"))

    def test_verify_can_keep_selected_pids_on_gpu(self):
        self.assertFalse(
            should_offload_target_cache(
                "verify",
                pid="req-1",
                pinned_gpu_pids={"req-1"},
            )
        )
        self.assertTrue(
            should_offload_target_cache(
                "verify",
                pid="req-2",
                pinned_gpu_pids={"req-1"},
            )
        )

    def test_unknown_stage_does_not_offload(self):
        self.assertFalse(should_offload_target_cache("other", pid="req-1"))

    def test_env_flag_disables_all_target_cache_offload(self):
        with mock.patch.dict("os.environ", {"FASTSD_DISABLE_TARGET_CACHE_OFFLOAD": "1"}):
            self.assertFalse(should_offload_target_cache("prefill", pid="req-1"))
            self.assertFalse(should_offload_target_cache("verify", pid="req-1"))


if __name__ == "__main__":
    unittest.main()
