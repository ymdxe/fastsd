import unittest

from src.metrics import elapsed_ms, tpot_ms


class LatencyMetricTests(unittest.TestCase):
    def test_elapsed_ms_is_non_negative(self):
        self.assertAlmostEqual(elapsed_ms(1.25, 1.75), 500.0)
        self.assertEqual(elapsed_ms(2.0, 1.0), 0.0)

    def test_tpot_uses_tokens_after_first(self):
        self.assertAlmostEqual(tpot_ms(10.0, 10.9, 4), 300.0)

    def test_tpot_is_zero_without_post_first_token_interval(self):
        self.assertEqual(tpot_ms(1.0, 2.0, 1), 0.0)
        self.assertEqual(tpot_ms(1.0, 2.0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
