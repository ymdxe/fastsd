import unittest
from pathlib import Path


class SpeculativeDecodingImplTests(unittest.TestCase):
    def test_speculative_decoding_does_not_move_full_prob_history_each_step(self):
        repo = Path(__file__).resolve().parents[1]
        engine_text = (repo / "src" / "engine.py").read_text(encoding="utf-8")

        self.assertNotIn("_prob_history.to(draft_device)", engine_text)


if __name__ == "__main__":
    unittest.main()
