import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "baselines" / "specedge" / "tools" / "repro.py"
)
SPEC = importlib.util.spec_from_file_location("specedge_repro", MODULE_PATH)
repro = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repro)


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class SpecEdgeMetricTests(unittest.TestCase):
    def test_all_shipped_configs_match_paper_token_and_tree_limits(self):
        config_dir = MODULE_PATH.parents[1] / "configs"
        for path in config_dir.glob("*.yaml"):
            method = "server_only" if path.name.endswith("server-only.yaml") else "specedge"
            validation = repro.validate_paper_config(path, method)
            non_host_errors = [
                error
                for error in validation["errors"]
                if "placeholder client.host" not in error
            ]
            self.assertEqual(non_host_errors, [], path.name)

    def test_specedge_metric_normalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "client_0.jsonl",
                [
                    {
                        "req_idx": 0,
                        "step_idx": 0,
                        "draft": {"end_to_end": 10, "forward": [4]},
                        "target": {"end_to_end": 20, "prefill": 1, "prev_proactive": False},
                        "num_accepted_tokens": 1,
                    },
                    {
                        "req_idx": 0,
                        "step_idx": 1,
                        "draft": {"end_to_end": 10, "forward": [4, 4]},
                        "target": {"end_to_end": 20, "prefill": 0, "prev_proactive": True},
                        "num_accepted_tokens": 3,
                    },
                ],
            )
            write_jsonl(
                root / "client_1.jsonl",
                [
                    {
                        "req_idx": 1,
                        "step_idx": 1,
                        "draft": {"end_to_end": 12, "forward": [4, 4]},
                        "target": {"end_to_end": 18, "prefill": 0, "prev_proactive": False},
                        "num_accepted_tokens": 3,
                    }
                ],
            )
            write_jsonl(
                root / "server.jsonl",
                [
                    {"timestamp": "2026-01-01T00:00:00", "target": {"prefill": 1, "server_end_to_end_t": 90}},
                    {"timestamp": "2026-01-01T00:00:02", "target": {"prefill": 0, "server_end_to_end_t": 80}},
                ],
            )

            metrics = repro._specedge_metrics(root, "overall", "A100-40")

            self.assertEqual(metrics["generated_tokens"], 7)
            self.assertAlmostEqual(metrics["server_throughput_tok_s"], 3.5)
            self.assertAlmostEqual(metrics["inter_token_latency_ms"], 10.0)
            self.assertAlmostEqual(metrics["draft_forward_ms"]["mean"], 4.0)
            self.assertAlmostEqual(metrics["server_verify_ms"]["mean"], 80.0)

    def test_server_only_metric_normalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "server_only.jsonl",
                [
                    {
                        "iter_idx": 0,
                        "server_iter_idx": 0,
                        "prefill": 1,
                        "draft": {"end_to_end": 10},
                        "target": {"end_to_end": 20},
                        "num_accepted_tokens": 1,
                    },
                    {
                        "iter_idx": 1,
                        "server_iter_idx": 1,
                        "prefill": 0,
                        "draft": {"end_to_end": 10},
                        "target": {"end_to_end": 20},
                        "num_accepted_tokens": 3,
                    },
                ],
            )

            metrics = repro._server_only_metrics(root, "A100-40")

            self.assertEqual(metrics["generated_tokens"], 3)
            self.assertAlmostEqual(metrics["server_throughput_tok_s"], 100.0)
            self.assertAlmostEqual(metrics["inter_token_latency_ms"], 10.0)

    def test_depth_recommendation_uses_paper_equation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "client_0.jsonl",
                [
                    {
                        "step_idx": 1,
                        "draft": {"forward": [11, 11]},
                        "target": {"prefill": 0},
                    }
                ],
            )
            write_jsonl(
                root / "server.jsonl",
                [{"target": {"prefill": 0, "server_end_to_end_t": 94.2}}],
            )

            result = repro.recommend_depth(root, 15.0)

            self.assertEqual(result["recommended_max_beam_len"], 7)

    def test_compare_rejects_mismatched_tree_budget(self):
        common = {
            "target_model": "Qwen/Qwen3-14B",
            "draft_model": "Qwen/Qwen3-1.7B",
            "dataset": "specbench",
            "temperature": 0.7,
            "max_batch_size": 1,
            "max_budget": 32,
            "max_new_tokens": 256,
            "seed": 42,
            "dtype": "fp16",
        }
        metrics = {
            "server_throughput_tok_s": 60.0,
            "cost_efficiency_1k_tokens_per_dollar": 50.0,
            "inter_token_latency_ms": 28.0,
            "accepted_tokens_per_verify": {"mean": 4.0},
        }
        specedge = {"method": "specedge", "config": common, "metrics": metrics}
        baseline_config = dict(common, max_budget=64)
        baseline = {
            "method": "server_only",
            "config": baseline_config,
            "metrics": metrics,
        }

        with self.assertRaises(repro.ReproError):
            repro.compare(specedge, baseline)


if __name__ == "__main__":
    unittest.main()
