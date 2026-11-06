import json
import py_compile
import unittest
from pathlib import Path

from scripts.eval_suite import (
    FORMAL_CPUSET,
    FORMAL_CPU_PREFIX,
    prefixed_command,
    render_specedge_config,
    output_layout,
    resolve_prepare_context,
    resolve_execution,
    validate_experiment_config,
    workload_hash_gate_values,
)
from scripts.preflight_cpu import parse_cpu_set
from src.common_metrics import paired_bootstrap_ci, paired_method_analysis, summarize_requests
from src.parity import build_token_parity_rows, exact_gate_summary, summarize_token_parity
from src.run_artifacts import write_text_once


REPO_ROOT = Path(__file__).resolve().parents[1]


class CpuEvaluationStaticTests(unittest.TestCase):
    def test_cpu_runtime_and_adapter_sources_compile(self):
        paths = [
            REPO_ROOT / "src" / "runtime.py",
            REPO_ROOT / "src" / "parity.py",
            REPO_ROOT / "src" / "run_artifacts.py",
            REPO_ROOT / "baselines" / "specedge" / "integration" / "cpu_adapter.py",
            REPO_ROOT / "baselines" / "specedge" / "integration" / "wire_codec.py",
            REPO_ROOT / "benchmark" / "eval_target_only.py",
            REPO_ROOT / "scripts" / "preflight_cpu.py",
            REPO_ROOT / "scripts" / "make_evaluation_matrix.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                py_compile.compile(str(path), doraise=True)

    def test_stateful_and_explicit_adapter_boundaries_are_present(self):
        edge_source = (REPO_ROOT / "edge" / "edge.py").read_text(encoding="utf-8")
        self.assertIn("client.init_session()", edge_source)
        self.assertIn("client.prefill(", edge_source)
        self.assertIn("client.verify(", edge_source)
        self.assertIn("approx_model_cache.rollback(accepted)", edge_source)
        self.assertIn("output_token_ids", edge_source)

        client_source = (
            REPO_ROOT / "baselines" / "specedge" / "integration" / "client.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CPUCompatibleSpecEdgeEngine", client_source)
        self.assertIn("ExplicitSpecEdgeGrpcClient", client_source)
        self.assertNotIn("util.encode =", client_source)

        cloud_source = (REPO_ROOT / "cloud" / "cloud_service.py").read_text(encoding="utf-8")
        self.assertIn('CLOUD_SERVICE_HOST", "127.0.0.1"', cloud_source)
        suite_source = (REPO_ROOT / "scripts" / "eval_suite.py").read_text(encoding="utf-8")
        self.assertIn("target_bind_host", suite_source)
        self.assertIn("specedge_bind_host", suite_source)
        self.assertNotIn("--host 0.0.0.0", suite_source)
        self.assertIn("warmup_stateful_requests", edge_source)

        draft_source = (REPO_ROOT / "benchmark" / "eval_draft_pool.py").read_text(encoding="utf-8")
        target_source = (REPO_ROOT / "benchmark" / "eval_target_only.py").read_text(encoding="utf-8")
        specedge_source = (
            REPO_ROOT / "baselines" / "specedge" / "integration" / "client.py"
        ).read_text(encoding="utf-8")
        self.assertIn("warmup_generation", draft_source)
        self.assertIn("warmup_requests", target_source)
        self.assertIn("warmup_specedge", specedge_source)

    def test_legacy_preparer_is_blocked(self):
        source = (REPO_ROOT / "scripts" / "prepare_32worker_experiment.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("INCOMPATIBLE_WITH_STATEFUL_FASTSD", source)
        self.assertIn("/session/init", source)
        self.assertIn("/generate", source)

    def test_cpu_specedge_yaml_declares_adapter_and_fp32(self):
        config = {
            "run_id": "cpu",
            "repo_path_linux": "/srv/fastsd",
            "models": {
                "draft": "/m/legacy-draft",
                "node3_draft": "/m/node3-draft",
                "node2_draft": "/m/node2-draft",
                "target": "/m/target",
            },
            "generation": {"seed": 42, "temperature": 0, "gamma": 4, "max_new_tokens": 256},
            "topology": {
                "specedge_draft_devices": ["cpu"],
                "specedge_target_device": "cuda:0",
                "specedge_host": "node2:18000",
                "draft_threads": 32,
                "specedge_port": 18000,
            },
            "dataset": {"arrival_distribution": "immediate"},
            "execution": {"specedge_python": "/explicit/python314"},
        }
        layout = {
            "linux_root": Path("/srv/fastsd/exp/comparison/cpu"),
            "linux_canonical": Path("/srv/fastsd/exp/comparison/cpu/inputs/canonical.jsonl"),
        }
        rendered = render_specedge_config(config, "hash", layout)
        self.assertIn("dtype: fp32", rendered)
        self.assertIn("engine: cpu_adapter", rendered)
        self.assertIn("threads: 32", rendered)
        self.assertIn("method: \"specedge_cpu_adapted\"", rendered)
        self.assertIn("python: \"/explicit/python314\"", rendered)

    def test_host_specific_execution_values_reach_both_specedge_yamls(self):
        config = {
            "run_id": "host-specific",
            "repo_path_linux": "/legacy/repo",
            "models": {
                "draft": "/m/legacy-draft",
                "node3_draft": "/m/node3-draft",
                "node2_draft": "/m/node2-draft",
                "target": "/m/target",
            },
            "generation": {"seed": 42, "temperature": 0, "gamma": 4, "max_new_tokens": 256},
            "dataset": {"arrival_distribution": "immediate"},
            "topology": {
                "specedge_draft_devices": ["cpu"],
                "specedge_target_device": "cuda:0",
                "draft_threads": 32,
                "specedge_port": 18000,
            },
            "execution": {
                "node3_repo": "/srv/node3-fast",
                "node2_repo": "/srv/node2-fast",
                "node3_edge_python": "/env/node3-edge/bin/python314_glibc",
                "node3_specedge_python": "/env/node3-spec/bin/python",
                "node2_target_python": "/env/node2-target/bin/python",
                "node2_specedge_python": "/env/node2-spec/bin/python",
            },
        }
        resolved = resolve_execution(config)
        self.assertEqual(resolved["node3_repo"], "/srv/node3-fast")
        self.assertEqual(resolved["node2_repo"], "/srv/node2-fast")
        self.assertEqual(resolved["node3_edge_python"], "/env/node3-edge/bin/python314_glibc")
        self.assertEqual(resolved["node3_specedge_python"], "/env/node3-spec/bin/python")
        self.assertEqual(resolved["node2_target_python"], "/env/node2-target/bin/python")
        self.assertEqual(resolved["node2_specedge_python"], "/env/node2-spec/bin/python")
        layout = {
            "linux_root": Path("/srv/node3-fast/exp/comparison/host-specific"),
            "node3_root": Path("/srv/node3-fast/exp/comparison/host-specific"),
            "node2_root": Path("/srv/node2-fast/exp/comparison/host-specific"),
            "linux_canonical": Path("/srv/node3-fast/inputs/canonical.jsonl"),
            "node3_canonical": Path("/srv/node3-fast/inputs/canonical.jsonl"),
            "node2_canonical": Path("/srv/node2-fast/inputs/canonical.jsonl"),
        }
        node3_yaml = render_specedge_config(config, "hash", layout, host_role="node3")
        node2_yaml = render_specedge_config(config, "hash", layout, host_role="node2")
        self.assertIn('python: "/env/node3-spec/bin/python"', node3_yaml)
        self.assertIn('repo_root: "/srv/node3-fast"', node3_yaml)
        self.assertIn('draft_model: "/m/node3-draft"', node3_yaml)
        self.assertIn('python: "/env/node2-spec/bin/python"', node2_yaml)
        self.assertIn('repo_root: "/srv/node2-fast"', node2_yaml)
        self.assertIn('draft_model: "/m/node2-draft"', node2_yaml)
        generated_layout = output_layout(config)
        self.assertTrue(str(generated_layout["node2_specedge_config"]).endswith("node2.yaml"))
        self.assertTrue(
            str(generated_layout["node2_specedge_config_linux"]).endswith("node2.yaml")
        )
        suite_source = (REPO_ROOT / "scripts" / "eval_suite.py").read_text(encoding="utf-8")
        self.assertIn("node2_prepare_command", suite_source)
        self.assertIn("node2_hash_command", suite_source)
        self.assertIn("--manifest", suite_source)
        self.assertIn("--expected", suite_source)
        self.assertIn("models['node2_draft']", suite_source)
        self.assertIn("plan execution overrides differ from manifest", suite_source)
        draft_pool_source = (REPO_ROOT / "benchmark" / "eval_draft_pool.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('config["models"].get("node3_draft")', draft_pool_source)

    def test_prepare_context_uses_actual_host_or_marks_unknown(self):
        execution = {
            "node3_repo": "/srv/node3-fast",
            "node2_repo": "/srv/node2-fast",
            "node3_edge_python": "/env/node3/python",
            "node3_specedge_python": "/env/node3/specedge",
            "node2_target_python": "/env/node2/python",
            "node2_specedge_python": "/env/node2/specedge",
        }
        node2_actual_repo = str(Path("/srv/node2-fast").resolve())
        self.assertEqual(
            resolve_prepare_context(execution, "/srv/node2-fast"),
            {
                "host_role": "node2",
                "repo": node2_actual_repo,
                "python": "/env/node2/python",
            },
        )
        unknown = resolve_prepare_context(execution, "/srv/other-fast")
        self.assertEqual(unknown["host_role"], "local/unknown")
        self.assertTrue(unknown["python"])

    def test_frozen_cpu_binding_and_track_validation(self):
        config = json.loads(
            (REPO_ROOT / "configs" / "evaluation" / "llama2_7b_tinyllama_four_method_cpu.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["topology"]["fixed_cpuset"], FORMAL_CPUSET)
        self.assertEqual(config["topology"]["cpu_prefix"], FORMAL_CPU_PREFIX)
        self.assertEqual(parse_cpu_set(FORMAL_CPUSET), list(range(56, 72)) + list(range(80, 96)))
        notes = validate_experiment_config(config)
        self.assertTrue(any("shared-load" in note for note in notes))

        invalid = json.loads(json.dumps(config))
        invalid["dataset"]["arrival_distribution"] = "poisson"
        with self.assertRaisesRegex(ValueError, "arrival_distribution"):
            validate_experiment_config(invalid)

    def test_cpu_prefix_places_environment_after_numactl(self):
        command = prefixed_command(
            "nice -n 5 numactl --physcpubind=56-71,80-95 --interleave=2,3 ",
            "python benchmark/eval_draft_pool.py",
            environment={"OMP_NUM_THREADS": 32, "TORCH_NUM_THREADS": 32},
        )
        self.assertIn("numactl --physcpubind=56-71,80-95 --interleave=2,3 env", command)
        self.assertIn("OMP_NUM_THREADS=32", command)

    def test_dataset_matrix_keeps_shared_generation_and_unique_runs(self):
        from scripts.make_evaluation_matrix import generate_matrix
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = {
                "run_id": "matrix",
                "generation": {"seed": 42, "max_new_tokens": 256},
                "topology": {"track": "latency", "draft_threads": 32},
                "dataset": {"name": "humaneval", "data_path": "data"},
            }
            base_path = root / "base.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            manifest = generate_matrix(base_path, root / "out", "data")
            self.assertEqual(
                {item["dataset"] for item in manifest["configs"]},
                {"humaneval", "mgsm", "gsm8k", "mt_bench"},
            )
            self.assertEqual(len({item["run_id"] for item in manifest["configs"]}), 4)
            mt = next(item for item in manifest["configs"] if item["dataset"] == "mt_bench")
            mt_config = json.loads(Path(mt["config"]).read_text(encoding="utf-8"))
            self.assertEqual(mt_config["metadata"]["mt_bench_turn_policy"], "first_turn_only")
            self.assertEqual(mt_config["generation"], base["generation"])
            self.assertEqual(mt_config["topology"], base["topology"])

    def test_matrix_dataset_suffix_is_stable(self):
        from scripts.make_evaluation_matrix import stable_run_id

        self.assertEqual(stable_run_id("run_humaneval", "humaneval"), "run_humaneval")
        self.assertEqual(stable_run_id("run_humaneval", "mgsm"), "run_mgsm")
        self.assertEqual(stable_run_id("run_mt_bench", "mt_bench"), "run_mt_bench")

    def test_client_host_records_actual_orchestration_entrypoint(self):
        source = (
            REPO_ROOT / "baselines" / "specedge" / "integration" / "client_host.py"
        ).read_text(encoding="utf-8")
        self.assertIn("host_script = Path(__file__).resolve()", source)
        self.assertIn("str(host_script), \"--config\"", source)

    def test_transport_audit_schema_is_explicit_for_http_and_grpc(self):
        edge_source = (REPO_ROOT / "edge" / "edge.py").read_text(encoding="utf-8")
        wire_source = (
            REPO_ROOT / "baselines" / "specedge" / "integration" / "wire_codec.py"
        ).read_text(encoding="utf-8")
        suite_source = (REPO_ROOT / "scripts" / "eval_suite.py").read_text(encoding="utf-8")
        metrics_source = (REPO_ROOT / "src" / "common_metrics.py").read_text(encoding="utf-8")
        for field in ("request_bytes", "response_bytes", "rpc_count"):
            self.assertIn(field, edge_source)
            self.assertIn(field, wire_source)
            self.assertIn(field, suite_source)
            self.assertIn(field, metrics_source)
        self.assertIn("request.ByteSize()", wire_source)
        self.assertIn("response.ByteSize()", wire_source)
        self.assertIn("TCP/IP", metrics_source)


class ParityAndMetricTests(unittest.TestCase):
    def test_cross_host_workload_hash_gate_rejects_nonmatching_hash(self):
        self.assertNotEqual(workload_hash_gate_values("node2-hash", "node3-hash"), 0)

    def test_token_parity_is_per_position_and_records_missing_tail(self):
        records = {
            "target_only": [
                {"sample_id": "a", "global_index": 0, "workload_hash": "h", "output_token_ids": [1, 2, 3]}
            ],
            "fastsd": [
                {"sample_id": "a", "global_index": 0, "workload_hash": "h", "output_token_ids": [1, 9]}
            ],
        }
        rows = build_token_parity_rows(records)
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0]["equal_to_reference"]["fastsd"])
        self.assertFalse(rows[1]["equal_to_reference"]["fastsd"])
        self.assertIsNone(rows[2]["token_ids"]["fastsd"])
        summary = summarize_token_parity(rows, methods=records.keys())
        self.assertEqual(summary["methods"]["fastsd"]["matching_positions"], 1)

    def test_exact_gate_fails_target_methods_but_not_for_draft_only(self):
        records = {
            "target_only": [
                {"sample_id": "a", "output_token_ids": [1, 2]},
                {"sample_id": "b", "output_token_ids": [3]},
            ],
            "fastsd": [
                {"sample_id": "a", "output_token_ids": [1, 2]},
                {"sample_id": "b", "output_token_ids": [3]},
            ],
            "specedge_cpu_adapted": [
                {"sample_id": "a", "output_token_ids": [1, 9]},
                {"sample_id": "b", "output_token_ids": [3]},
            ],
            "standard_sd": [
                {"sample_id": "a", "output_token_ids": [1, 2]},
                {"sample_id": "b", "output_token_ids": [3]},
            ],
            "draft_only": [
                {"sample_id": "a", "output_token_ids": [99]},
                {"sample_id": "b", "output_token_ids": [98]},
            ],
        }
        rows = build_token_parity_rows(records)
        gate = exact_gate_summary(
            rows,
            expected_sample_ids=["a", "b"],
            sample_ids_by_method={
                method: [record["sample_id"] for record in values]
                for method, values in records.items()
            },
        )
        self.assertFalse(gate["gate_pass"])
        self.assertEqual(gate["methods"]["fastsd"]["mismatch"], 0)
        self.assertGreater(gate["methods"]["specedge_cpu_adapted"]["mismatch"], 0)
        self.assertNotIn("draft_only", gate["methods"])
        self.assertIn("draft_only", gate["report_only_methods"])

    def test_direct_paired_analysis_uses_fixed_ten_thousand_bootstrap(self):
        left = [
            {"sample_id": "a", "ttft_ms": 8, "e2e_ms": 20, "tpot_ms": 4, "generated_tokens": 4},
            {"sample_id": "b", "ttft_ms": 10, "e2e_ms": 25, "tpot_ms": 5, "generated_tokens": 5},
        ]
        right = [
            {"sample_id": "a", "ttft_ms": 10, "e2e_ms": 25, "tpot_ms": 5, "generated_tokens": 4},
            {"sample_id": "b", "ttft_ms": 12, "e2e_ms": 30, "tpot_ms": 6, "generated_tokens": 5},
        ]
        result = paired_method_analysis(left, right)
        self.assertEqual(result["bootstrap_iterations"], 10_000)
        self.assertEqual(result["bootstrap_seed"], 42)
        self.assertGreater(result["metrics"]["e2e_ms"]["improvement_pct"], 0)
        self.assertIn("improvement_ci95_pct", result["metrics"]["throughput_tok_s"])

    def test_summary_labels_smoke_and_exposes_goodput_and_paired_ci(self):
        records = [
            {
                "sample_id": "a",
                "generated_tokens": 3,
                "ttft_ms": 10,
                "e2e_ms": 30,
                "output_text": "2",
                "reference": "2",
                "success": True,
            }
        ]
        baseline = [{"sample_id": "a", "ttft_ms": 12, "e2e_ms": 40, "generated_tokens": 3}]
        summary = summarize_requests(
            records,
            method="fastsd",
            dataset="gsm8k",
            workload_hash="h",
            run_id="r",
            wallclock_s=1.0,
            evaluation_scope="communication_smoke",
            paired_records=baseline,
        )
        self.assertEqual(summary["evaluation_scope"], "communication_smoke")
        self.assertIn("goodput_tok_s", summary)
        self.assertIn("paired_ci_vs_baseline", summary)
        self.assertEqual(paired_bootstrap_ci(records, baseline, key="e2e_ms")["n"], 1)

    def test_run_artifact_write_once_preserves_previous_payload(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.jsonl"
            write_text_once(path, "first\n")
            write_text_once(path, "first\n")
            with self.assertRaises(FileExistsError):
                write_text_once(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\n")


if __name__ == "__main__":
    unittest.main()
