import unittest

from src.energy_meter import EnergyAccumulator


class EnergyAccumulatorTests(unittest.TestCase):
    def test_integrates_power_over_time(self):
        meter = EnergyAccumulator()
        meter.start(100.0)
        meter.update(120.0, 0.2)
        meter.update(80.0, 0.1)
        summary = meter.stop(130.0)

        self.assertAlmostEqual(summary["energy_j"], 120.0 * 0.2 + 80.0 * 0.1, places=6)
        self.assertAlmostEqual(summary["duration_s"], 0.3, places=6)
        self.assertAlmostEqual(summary["avg_power_w"], summary["energy_j"] / summary["duration_s"], places=6)

    def test_stop_without_start_raises(self):
        meter = EnergyAccumulator()
        with self.assertRaises(RuntimeError):
            meter.stop(1.0)


if __name__ == "__main__":
    unittest.main()
