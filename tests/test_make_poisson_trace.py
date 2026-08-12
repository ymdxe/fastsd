import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MakePoissonTraceTests(unittest.TestCase):
    @property
    def script(self):
        return Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "make_poisson_trace.py"

    def test_rejects_non_hdd_output_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text('{"task_id":"x","turns":["hello"]}\n', encoding="utf-8")
            output = root / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--dataset",
                    str(dataset),
                    "--rate-rps",
                    "1",
                    "--max-requests",
                    "1",
                    "--seed",
                    "1",
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_parser_exposes_shared_trace_contract(self):
        source = self.script.read_text(encoding="utf-8")
        self.assertIn('"--num-clients", "--clients"', source)
        self.assertIn('"arrival_trace.jsonl"', source)
        self.assertIn('"prompts.jsonl"', source)


if __name__ == "__main__":
    unittest.main()
