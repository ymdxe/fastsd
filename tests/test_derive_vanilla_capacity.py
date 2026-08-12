import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "derive_vanilla_capacity.py"
SPEC = importlib.util.spec_from_file_location("derive_vanilla_capacity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CAPACITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPACITY)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class DeriveVanillaCapacityTests(unittest.TestCase):
    def test_accepts_only_complete_slo_conforming_candidate(self):
        with tempfile.TemporaryDirectory() as raw_root:
            run = Path(raw_root) / "candidate"
            write_json(run / "manifest.json", {"method": "vanilla", "status": "complete", "run_id": "v"})
            write_json(run / "metrics" / "summary.json", {"completion_rate": 1.0, "task_e2e_ms_p95": 199.0})
            write_json(run / "workload" / "manifest.json", {"rate_rps": 3.0})
            result = CAPACITY.candidate_result(run, 100.0)
            self.assertTrue(result["eligible"])
            self.assertEqual(result["rate_rps"], 3.0)

    def test_rejects_partial_or_slo_violating_candidate(self):
        with tempfile.TemporaryDirectory() as raw_root:
            run = Path(raw_root) / "candidate"
            write_json(run / "manifest.json", {"method": "vanilla", "status": "complete", "run_id": "v"})
            write_json(run / "metrics" / "summary.json", {"completion_rate": 0.99, "task_e2e_ms_p95": 201.0})
            write_json(run / "workload" / "manifest.json", {"rate_rps": 3.0})
            self.assertFalse(CAPACITY.candidate_result(run, 100.0)["eligible"])


if __name__ == "__main__":
    unittest.main()
