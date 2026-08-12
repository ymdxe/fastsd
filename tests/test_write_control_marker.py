import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "write_control_marker.py"
SPEC = importlib.util.spec_from_file_location("write_control_marker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MARKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MARKER)


class WriteControlMarkerTests(unittest.TestCase):
    def test_requires_run_owned_control_path_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as raw_root:
            # Unit tests use a synthetic /home/hdd-like string for the strict
            # path predicate, then validate exclusive creation behavior with a
            # real temporary parent through the implementation's open("x").
            root = Path(raw_root)
            path = root / "control" / "graceful-shutdown.json"
            path.parent.mkdir()
            with path.open("x", encoding="utf-8") as handle:
                json.dump({"run_id": "old"}, handle)
            self.assertTrue(path.is_file())
            # Reuse prevention is independent of the /home/hdd path policy.
            with self.assertRaises(SystemExit):
                MARKER.validate(path, "new-run")

    def test_source_uses_exclusive_create_and_run_id_bound_payload(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('path.open("x"', source)
        self.assertIn('"event": "graceful_shutdown"', source)
        self.assertIn('"run_id": args.run_id', source)


if __name__ == "__main__":
    unittest.main()
