import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "crop_gpu_samples.py"
SPEC = importlib.util.spec_from_file_location("crop_gpu_samples", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CROP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CROP)


class CropGpuSamplesTests(unittest.TestCase):
    def test_interpolates_boundaries_per_gpu_and_preserves_only_measurement_window(self):
        fields = ["monotonic_s", "physical_gpu", "power_draw_w", "memory_used_mib"]
        raw_rows = []
        for gpu in ("0", "1"):
            raw_rows.extend(
                [
                    {"monotonic_s": "0", "physical_gpu": gpu, "power_draw_w": "10", "memory_used_mib": "1"},
                    {"monotonic_s": "1", "physical_gpu": gpu, "power_draw_w": "20", "memory_used_mib": "2"},
                    {"monotonic_s": "2", "physical_gpu": gpu, "power_draw_w": "30", "memory_used_mib": "3"},
                ]
            )
        records = [
            {"status": "completed", "start": 0.5, "end": 1.25},
            {"status": "completed", "start": 0.75, "end": 1.5},
        ]
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            raw_path = root / "gpu_samples_raw.csv"
            records_path = root / "requests.jsonl"
            with raw_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(raw_rows)
            records_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            start, end = CROP.resolve_window(CROP.read_jsonl(records_path), "start", "end")
            self.assertEqual((start, end), (0.5, 1.5))
            _, samples = CROP.read_samples(raw_path)
            cropped = CROP.crop_rows(samples, start, end)
            self.assertEqual(len(cropped), 6)
            for gpu in ("0", "1"):
                times = [
                    float(row["monotonic_s"])
                    for row in cropped
                    if row["physical_gpu"] == gpu
                ]
                self.assertEqual(times, [0.5, 1.0, 1.5])
            self.assertEqual(float(cropped[0]["power_draw_w"]), 15.0)

    def test_rejects_energy_window_not_covered_by_raw_samples(self):
        rows = [
            {"monotonic_s": "1", "physical_gpu": "0", "power_draw_w": "10"},
            {"monotonic_s": "2", "physical_gpu": "0", "power_draw_w": "10"},
        ]
        with self.assertRaisesRegex(ValueError, "does not cover"):
            CROP.crop_rows(rows, 0.5, 1.5)

    def test_uses_validate_events_without_counting_sync_prewarm(self):
        events = [
            {"method": "Sync", "event": "enter", "server_monotonic_s": 1.0},
            {"method": "Sync", "event": "leave", "server_monotonic_s": 2.0},
            {"method": "Validate", "event": "enter", "server_monotonic_s": 10.0},
            {"method": "Validate", "event": "leave", "server_monotonic_s": 10.2},
            {"method": "Validate", "event": "enter", "server_monotonic_s": 11.0},
            {"method": "Validate", "event": "leave", "server_monotonic_s": 11.5},
        ]
        self.assertEqual(
            CROP.resolve_event_window(
                events, timestamp_field="server_monotonic_s", method="Validate"
            ),
            (10.0, 11.5),
        )


if __name__ == "__main__":
    unittest.main()
