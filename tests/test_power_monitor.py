import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.power_monitor import (
    MonitorConfig,
    NvidiaPowerReader,
    PowerMonitor,
    ProcReader,
    RaplReader,
    attribute_package_power,
    merge_experiment_summaries,
    parse_proc_stat,
)


class _Clock:
    def __init__(self):
        self.value = 100.0

    def time(self):
        return self.value

    def mono(self):
        return self.value


class PowerMonitorTests(unittest.TestCase):
    def test_rapl_wraparound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "intel-rapl:0"
            domain.mkdir()
            (domain / "name").write_text("package-0")
            (domain / "max_energy_range_uj").write_text("1000")
            counter = domain / "energy_uj"
            counter.write_text("950")
            clock = _Clock()
            reader = RaplReader(root, clock=clock.time, monotonic=clock.mono)
            self.assertIsNone(reader.sample().package_power_w)
            clock.value += 2
            counter.write_text("50")
            reading = reader.sample()
            self.assertTrue(reading.wrapped)
            self.assertAlmostEqual(reading.package_power_w, 0.00005, places=8)
            self.assertEqual(reading.sampling_interval_s, 2.0)

    def test_proc_stat_name_with_parenthesis(self):
        # field 3 contains a ')' and must still parse fields 14/15 correctly.
        text = "123 (worker ) fast) S " + " ".join(["0"] * 10) + " 7 5 " + " ".join(["0"] * 20)
        stat = parse_proc_stat(text, 123, clk_tck=100)
        self.assertEqual(stat.process_name, "worker ) fast")
        self.assertEqual(stat.cpu_time_ticks, 12)

    def test_process_power_uses_host_denominator(self):
        result = attribute_package_power(100.0, {10: 1.0, 11: 3.0}, host_cpu_delta_s=8.0)
        self.assertAlmostEqual(result[10], 12.5)
        self.assertAlmostEqual(result[11], 37.5)
        self.assertLess(sum(result.values()), 100.0)

    def test_missing_nvml_is_graceful(self):
        reader = NvidiaPowerReader(pynvml_module=object())
        self.assertFalse(reader.available)
        self.assertIsNone(reader.read_power_w())

    def test_unavailable_rapl_serializes_null_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            monitor = PowerMonitor(MonitorConfig(
                experiment_id="no-rapl", role="edge_cpu", output_dir=output_dir,
                enable_gpu=False, rapl_root=str(output_dir / "missing-rapl"),
            ))
            monitor.start()
            summary = monitor.stop()
            self.assertIsNone(summary["cpu_package"]["total_energy_j"])
            self.assertIsNone(summary["process_attribution"]["aggregate_estimated_cpu_energy_j"])
            self.assertIsNone(summary["total_energy_j"])
            sample = json.loads((output_dir / "power_samples.jsonl").read_text().splitlines()[0])
            self.assertIsNone(sample["cumulative_package_energy_j"])
            self.assertIsNone(sample["cumulative_energy_j"])

    def test_merge_uses_edge_cpu_and_target_gpu_only(self):
        target = {
            "experiment_id": "x", "role": "target",
            "start_timestamp_epoch_s": 1.0, "end_timestamp_epoch_s": 11.0,
            "gpu": {"total_energy_j": 20.0},
        }
        edge = {
            "experiment_id": "x", "role": "edge",
            "start_timestamp_epoch_s": 2.0, "end_timestamp_epoch_s": 10.0,
            "process_attribution": {"aggregate_estimated_cpu_energy_j": 12.0},
        }
        summary = merge_experiment_summaries(target, [edge])
        self.assertEqual(summary["combined_computational_energy_j"], 32.0)
        self.assertAlmostEqual(summary["combined_computational_power_w"], 4.0)

    def test_merge_clips_startup_energy_to_timestamp_intersection(self):
        target = {
            "experiment_id": "x", "role": "target",
            "start_timestamp_epoch_s": 0.0, "end_timestamp_epoch_s": 10.0,
            "gpu": {"total_energy_j": 100.0},
        }
        edge = {
            "experiment_id": "x", "role": "edge",
            "start_timestamp_epoch_s": 5.0, "end_timestamp_epoch_s": 10.0,
            "process_attribution": {"aggregate_estimated_cpu_energy_j": 50.0},
        }
        target_rows = [
            {"timestamp_epoch_s": 5.0, "sampling_interval_s": 5.0, "gpu_power_w": 10.0},
            {"timestamp_epoch_s": 10.0, "sampling_interval_s": 5.0, "gpu_power_w": 10.0},
        ]
        edge_rows = [[
            {"timestamp_epoch_s": 10.0, "sampling_interval_s": 5.0,
             "estimated_process_power_w": 4.0},
        ]]
        summary = merge_experiment_summaries(
            target, [edge], target_hardware_samples=target_rows,
            edge_process_samples=edge_rows,
        )
        self.assertAlmostEqual(summary["target_gpu_measured_energy_j"], 50.0)
        self.assertAlmostEqual(summary["edge_cpu_estimated_energy_j"], 20.0)
        self.assertAlmostEqual(summary["combined_computational_energy_j"], 70.0)

    def test_merge_preserves_unavailable_energy_as_null(self):
        target = {
            "experiment_id": "x", "role": "target",
            "start_timestamp_epoch_s": 1.0, "end_timestamp_epoch_s": 2.0,
            "gpu": {"total_energy_j": None},
        }
        edge = {
            "experiment_id": "x", "role": "edge",
            "start_timestamp_epoch_s": 1.0, "end_timestamp_epoch_s": 2.0,
            "process_attribution": {"aggregate_estimated_cpu_energy_j": None},
        }
        summary = merge_experiment_summaries(target, [edge])
        self.assertIsNone(summary["edge_cpu_estimated_energy_j"])
        self.assertIsNone(summary["target_gpu_measured_energy_j"])
        self.assertIsNone(summary["combined_computational_energy_j"])

    def test_merge_power_cli_reports_missing_edge(self):
        script = Path(__file__).resolve().parents[1] / "scripts/zjh_ab/merge_power.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps({"experiment_id": "x"}))
            result = subprocess.run(
                [sys.executable, str(script), "--target", str(target),
                 "--edge", str(root / "missing-edge.json"),
                 "--output", str(root / "out.json")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("edge power summary missing", result.stderr)

    def test_monitor_writes_raw_and_process_files_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MonitorConfig(
                experiment_id="test", role="edge_cpu", output_dir=Path(directory),
                pids=(1,), root_pids=(), enable_gpu=False, sample_interval_s=0.01,
                rapl_root=str(Path(directory) / "missing-rapl"),
            )
            monitor = PowerMonitor(config)
            monitor.start()
            monitor.stop()
            self.assertTrue((Path(directory) / "power_samples.jsonl").exists())
            self.assertTrue((Path(directory) / "power_process_samples.jsonl").exists())
            payload = json.loads((Path(directory) / "power_summary.json").read_text())
            self.assertFalse(payload["hardware"]["rapl_available"])


if __name__ == "__main__":
    unittest.main()
