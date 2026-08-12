import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "analyze_matrix.py"
SPEC = importlib.util.spec_from_file_location("analyze_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class AnalyzeMatrixTests(unittest.TestCase):
    def test_slo_goodput_handles_fastsd_and_specedge_request_schemas(self):
        with tempfile.TemporaryDirectory() as raw_root:
            run = Path(raw_root)
            metrics = run / "metrics"
            metrics.mkdir()
            rows = [
                {"status": "completed", "task_e2e_ms": 90.0},
                {"status": "completed", "e2e_from_arrival_s": 0.11},
                {"status": "error", "task_e2e_ms": 1.0},
            ]
            (metrics / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = {"measurement_window_s": 2.0}
            self.assertEqual(
                ANALYZE.slo_goodput_req_per_s(run, summary, slo_e2e_ms=100.0), 0.5
            )

    def test_missing_request_artifact_keeps_goodput_unavailable(self):
        with tempfile.TemporaryDirectory() as raw_root:
            self.assertIsNone(
                ANALYZE.slo_goodput_req_per_s(
                    Path(raw_root), {"measurement_window_s": 1.0}, slo_e2e_ms=100.0
                )
            )


if __name__ == "__main__":
    unittest.main()
