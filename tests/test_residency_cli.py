"""Command-line contract for the residency / prefetch switches."""

import sys
import unittest
from unittest import mock

from src.util import parse_arguments


def parse(*extra):
    with mock.patch.object(sys, "argv", ["prog", *extra]):
        return parse_arguments()


class ResidencyCliTests(unittest.TestCase):
    def test_residency_is_the_default(self):
        args = parse()
        self.assertEqual(args.kv_residency_budget_mb, 0)
        self.assertEqual(args.kv_residency_high_ratio, 0.90)
        self.assertEqual(args.kv_residency_low_ratio, 0.75)
        self.assertFalse(args.measure_forward_gpu)
        self.assertEqual(args.kv_prefetch_mode, "overlap")

    def test_explicit_budget_is_accepted(self):
        args = parse("--kv_residency_budget_mb", "4096")
        self.assertEqual(args.kv_residency_budget_mb, 4096)

    def test_online_fraction_is_bounded(self):
        self.assertEqual(parse().kv_prefetch_online_fraction, 0.25)
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
