import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FinalEntrypointTests(unittest.TestCase):
    def test_runall_declares_three_prefetch_conditions(self):
        source = (ROOT / "scripts/zjh_ab/runall.sh").read_text(encoding="utf-8")
        for name in ("_noprefetch", "_prefetch_h1", "_prefetch_direct"):
            self.assertIn(f'${{RUN_TAG}}{name}', source)
        self.assertIn("--expected-requests", source)

    def test_ar_path_has_no_draft_or_verify_protocol(self):
        client = (ROOT / "scripts/zjh_ab/client.sh").read_text(encoding="utf-8")
        cloud = (ROOT / "cloud/cloud_service.py").read_text(encoding="utf-8")
        edge = (ROOT / "edge/edge.py").read_text(encoding="utf-8")
        self.assertIn("--case ar", client)
        self.assertIn("@app.post(\"/generate\")", cloud)
        self.assertIn("run_target_process_autoregressive", cloud)
        self.assertIn("_eval_target_only_ar", edge)

    def test_model_aliases_and_explicit_tp_guard_exist(self):
        source = (ROOT / "src/util.py").read_text(encoding="utf-8")
        for alias in ("qwen3-32b", "qwen3-1.7b-fp8", "tinyllama-1.1b-python-v0.1", "deepseek-coder-1.3b-instruct"):
            self.assertIn(alias, source)
        self.assertIn("requires explicit TARGET_TP_SIZE=2", source)

    def test_specedge_server_todo_and_runner_exist(self):
        self.assertTrue((ROOT / "docs/specedge_server_todo.md").is_file())
        self.assertTrue((ROOT / "scripts/zjh_ab/run_specedge.sh").is_file())
        self.assertTrue((ROOT / "scripts/zjh_ab/collect_specedge.py").is_file())

    def test_runall_supports_three_cpu_edges_and_local_target_gpu(self):
        source = (ROOT / "scripts/zjh_ab/runall.sh").read_text(encoding="utf-8")
        for name in ("N3_HOST", "N3_REPO", "N3_DATA_PATH", "N3_DRAFTS", "TARGET_NUM_DRAFTS"):
            self.assertIn(name, source)
        self.assertIn("GPU_HOST=local", (ROOT / "scripts/zjh_ab/README.md").read_text(encoding="utf-8"))
        self.assertIn("EDGE_CPU_AFFINITY", (ROOT / "scripts/zjh_ab/client.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
