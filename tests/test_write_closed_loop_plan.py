import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "write_closed_loop_plan.py"
SPEC = importlib.util.spec_from_file_location("write_closed_loop_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


class ClosedLoopPlanTests(unittest.TestCase):
    def test_source_records_static_round_robin_assignment_and_matrix_exclusion(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"arrival_mode": "closed_loop"', source)
        self.assertIn('"matrix_eligible": False', source)
        self.assertIn("dataset_index = client_id + local_index * args.num_clients", source)
        self.assertIn('plan_path.open("x"', source)

    def test_checksum_helper_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "data.jsonl"
            path.write_text('{"x":1}\n', encoding="utf-8")
            self.assertEqual(PLAN.sha256(path), PLAN.sha256(path))


if __name__ == "__main__":
    unittest.main()
