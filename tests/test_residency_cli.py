"""Command-line contract for the residency / prefetch switches."""

import sys
import unittest
from unittest import mock

from src.util import parse_arguments


def parse(*extra):
    with mock.patch.object(sys, "argv", ["prog", *extra]):
        return parse_arguments()


class ResidencyCliTests(unittest.TestCase):
    def test_h1_default_and_legacy_rollback(self):
        with mock.patch.dict("os.environ", {"KV_PREFETCH_IMPLEMENTATION": "h1"}):
            self.assertEqual(parse().kv_prefetch_implementation, "h1")
            self.assertEqual(parse("--kv_prefetch_implementation", "legacy").kv_prefetch_implementation, "legacy")
        with mock.patch.dict("os.environ", {"KV_PREFETCH_IMPLEMENTATION": "legacy"}):
            self.assertEqual(parse().kv_prefetch_implementation, "legacy")

    def test_transfer_environment_defaults_and_cli_override(self):
        with mock.patch.dict("os.environ", {"KV_PREFETCH_CHUNK_KB": "512",
                "KV_PREFETCH_RATE_MB_S": "256", "KV_PREFETCH_IDLE_RATE_MB_S": "1024"}):
            args = parse()
            self.assertEqual((args.kv_prefetch_chunk_kb, args.kv_prefetch_rate_mb_s,
                              args.kv_prefetch_idle_rate_mb_s), (512, 256, 1024))
            self.assertEqual(args.kv_prefetch_speculative_chunk_kb, 4096)
            self.assertEqual(parse("--kv_prefetch_rate_mb_s", "128").kv_prefetch_rate_mb_s, 128)

    def test_speculative_chunk_environment_override(self):
        with mock.patch.dict("os.environ", {"KV_PREFETCH_SPECULATIVE_CHUNK_KB": "4096"}):
            self.assertEqual(parse().kv_prefetch_speculative_chunk_kb, 4096)

    def test_residency_is_the_default(self):
        args = parse()
        self.assertEqual(args.kv_residency_budget_mb, 0)
        self.assertEqual(args.kv_residency_high_ratio, 0.90)
        self.assertEqual(args.kv_residency_low_ratio, 0.75)
        self.assertFalse(args.measure_forward_gpu)
        self.assertEqual(args.kv_prefetch_mode, "overlap")
        self.assertEqual(args.kv_prefetch_rate_mb_s, 0.0)
        self.assertEqual(args.kv_prefetch_idle_rate_mb_s, 0.0)

    def test_explicit_budget_is_accepted(self):
        args = parse("--kv_residency_budget_mb", "4096")
        self.assertEqual(args.kv_residency_budget_mb, 4096)

    def test_online_fraction_is_bounded(self):
        self.assertEqual(parse().kv_prefetch_online_fraction, 1.0)
        with self.assertRaises(SystemExit):
            parse("--kv_prefetch_online_fraction", "1.5")

    def test_removed_mode_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse("--kv_prefetch_mode", "protect_forward")

    def test_ratios_must_be_ordered(self):
        with self.assertRaises(SystemExit):
            parse("--kv_residency_low_ratio", "0.95",
                  "--kv_residency_high_ratio", "0.90")

    def test_negative_budgets_are_rejected(self):
        with self.assertRaises(SystemExit):
            parse("--kv_residency_budget_mb", "-1")


if __name__ == "__main__":
    unittest.main()
