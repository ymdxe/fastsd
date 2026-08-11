import py_compile
from pathlib import Path
import unittest


class EdgeEntrypointTests(unittest.TestCase):
    @staticmethod
    def edge_path():
        return Path(__file__).resolve().parents[1] / "edge" / "edge.py"

    def test_edge_module_compiles(self):
        py_compile.compile(str(self.edge_path()), doraise=True)

    def test_experiment_task_limit_and_prefill_metrics_are_wired(self):
        edge_text = self.edge_path().read_text(encoding="utf-8")
        self.assertIn("self.args.max_tasks_per_draft", edge_text)
        self.assertIn("self._load_draft_model_for_service", edge_text)
        self.assertNotIn("from auto_gptq import AutoGPTQForCausalLM", edge_text)
        self.assertIn('"prefill_queue_ms"', edge_text)
        self.assertIn('"prefill_service_ms"', edge_text)
        self.assertIn('"prefill_chunks"', edge_text)


if __name__ == "__main__":
    unittest.main()
