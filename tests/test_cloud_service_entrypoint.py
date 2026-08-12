"""Static, no-GPU regression checks for the cloud HTTP response contract."""

from __future__ import annotations

import pathlib
import unittest


class CloudServiceEntrypointTests(unittest.TestCase):
    def test_prefill_preserves_cloud_measurement_bounds(self) -> None:
        source = pathlib.Path("cloud/cloud_service.py").read_text(encoding="utf-8")
        start = source.index('@app.post("/prefill")')
        end = source.index('@app.post("/verify")', start)
        prefill = source[start:end]
        for field in (
            "prefill_chunks",
            "prefill_queue_ms",
            "prefill_service_ms",
            "server_enqueue_monotonic_s",
            "server_completed_monotonic_s",
        ):
            self.assertIn(field, prefill)
        self.assertIn('"status": "prefill_ok"', prefill)


if __name__ == "__main__":
    unittest.main()
