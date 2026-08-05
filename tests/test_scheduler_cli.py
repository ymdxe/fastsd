import argparse
import unittest

from src.util import positive_int


class SchedulerCliTests(unittest.TestCase):
    def test_positive_scheduler_limits_are_accepted(self):
        self.assertEqual(positive_int("1"), 1)
        self.assertEqual(positive_int("512"), 512)

    def test_zero_and_negative_scheduler_limits_are_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_int(value)


if __name__ == "__main__":
    unittest.main()
