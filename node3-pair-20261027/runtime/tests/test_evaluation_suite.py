import json
import os
import random
import statistics
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from scripts.eval_suite import (
    ANALYSIS_BOOTSTRAP_ITERATIONS,
    ANALYSIS_BOOTSTRAP_SEED,
    _analysis_aggregate_resources,
    _analysis_macro_pair,
    _analysis_pair_records,
    analyze,
    build_parser,
    build_analysis_report,
    normalize_fastsd,
    normalize_specedge,
    output_layout,
    parse_mpstat_cpu_resource,
    parse_nvidia_smi_csv_resource,
    resources,
    render_specedge_config,
)
from src.common_metrics import percentile, summarize_requests
from src.evaluation import (
    load_canonical_jsonl,
    load_evaluation_records,
    workload_fingerprint,
    write_canonical_jsonl,
)


class EvaluationDatasetTests(unittest.TestCase):
    def test_all_repository_dataset_schemas_are_adapted(self):
        fixtures = {
            "humaneval": {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "assert f()"},
            "gsm8k": {"question": "1+1?", "answer": "#### 2"},
            "mgsm": {"question_id": "m1", "question": "1+1?", "answer": "2"},
            "mt_bench": {"question_id": 7, "category": "math", "turns": ["hello", "again"]},
        }
        names = {
            "humaneval": "humaneval.jsonl",
            "gsm8k": "gsm8k.jsonl",
            "mgsm": "mgsm.jsonl",
            "mt_bench": "mt_bench.jsonl",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for dataset, fixture in fixtures.items():
                (root / names[dataset]).write_text(json.dumps(fixture) + "\n", encoding="utf-8")
                records = load_evaluation_records(dataset, root)
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0].prompt)
                self.assertEqual(records[0].dataset, dataset)

    def test_poisson_manifest_is_deterministic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "gsm8k.jsonl"
            source.write_text(
                "\n".join(json.dumps({"question": str(i), "answer": str(i)}) for i in range(3)),
                encoding="utf-8",
            )
            left = load_evaluation_records("gsm8k", source, arrival_distribution="poisson", arrival_seed=9)
            right = load_evaluation_records("gsm8k", source, arrival_distribution="poisson", arrival_seed=9)
            self.assertEqual(left, right)
            self.assertEqual(workload_fingerprint(left, {"seed": 1}), workload_fingerprint(right, {"seed": 1}))
            canonical = Path(temp_dir) / "canonical.jsonl"
            write_canonical_jsonl(left, canonical)
            loaded = load_canonical_jsonl(canonical)
            self.assertEqual([item["global_index"] for item in loaded], [0, 1, 2])
            self.assertTrue(all(loaded[i]["scheduled_arrival_s"] < loaded[i + 1]["scheduled_arrival_s"] for i in range(2)))


class CommonMetricTests(unittest.TestCase):
    def test_summary_reports_service_and_queue_inclusive_ttft(self):
        summary = summarize_requests(
            [
                {
                    "generated_tokens": 3,
                    "ttft_ms": 10,
                    "arrival_lag_ms": 5,
                    "tpot_ms": 2,
                    "e2e_ms": 14,
                    "actual_arrival_s": 1,
                    "completion_s": 1.014,
                    "output_text": "2",
                    "reference": "#### 2",
                }
            ],
            method="draft_only",
            dataset="gsm8k",
            workload_hash="abc",
            run_id="run",
        )
        self.assertEqual(summary["ttft_ms"]["avg"], 10)
        self.assertEqual(summary["scheduled_ttft_ms"]["avg"], 15)
        self.assertEqual(summary["quality_exact_match"], 1.0)


class DraftOnlyPromptTests(unittest.TestCase):
    def test_mt_bench_uses_chat_template_with_optional_thinking_switch(self):
        source = (Path(__file__).parents[1] / "benchmark" / "eval_draft_pool.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn('arrival_distribution == "poisson"', source)
        self.assertIn("actual_arrival = 0.0", source)
        self.assertIn("configure_spawn_executable()", source)
        self.assertIn("mp.set_executable(executable)", source)


class SpecEdgeAdapterTests(unittest.TestCase):
    def _config(self):
        return {
            "run_id": "run",
            "repo_path_linux": "/srv/new_fastsd",
            "dataset": {"arrival_distribution": "immediate"},
            "models": {"draft": "/models/TinyLlama-1.1B", "target": "/models/Llama-2-7B"},
            "generation": {"seed": 42, "temperature": 0, "gamma": 4, "max_new_tokens": 8},
            "topology": {
                "specedge_host": "127.0.0.1:18000",
                "specedge_target_device": "cuda:0",
                "specedge_draft_devices": ["cuda:0", "cuda:1"],
            },
        }

    def test_linux_paths_remain_posix_on_windows(self):
        layout = output_layout(self._config())
        self.assertIsInstance(layout["linux_root"], PurePosixPath)
        self.assertEqual(str(layout["linux_canonical"]), "/srv/new_fastsd/exp/comparison/run/inputs/canonical.jsonl")

    def test_rendered_yaml_has_two_client_devices(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is unavailable")
        layout = output_layout(self._config())
        rendered = yaml.safe_load(render_specedge_config(self._config(), "hash", layout))
        self.assertEqual(rendered["server"]["max_batch_size"], 2)
        self.assertEqual(rendered["node"]["local"], [{"device": "cuda:0"}, {"device": "cuda:1"}])
        self.assertEqual(rendered["base"]["dtype"], "bf16")
        self.assertEqual(rendered["integration"]["arrival_distribution"], "immediate")
        self.assertEqual(rendered["integration"]["server_port"], 18000)
        self.assertEqual(
            rendered["integration"]["python"],
            "python",
        )

    def test_specedge_mt_bench_adapter_uses_chat_template_and_immediate_arrivals(self):
        source = (
            Path(__file__).parents[1]
            / "baselines"
            / "specedge"
            / "integration"
            / "client.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn("actual_arrival = 0.0", source)
        wire_source = (
            Path(__file__).parents[1]
            / "baselines"
            / "specedge"
            / "integration"
            / "wire_codec.py"
        ).read_text(encoding="utf-8")
        self.assertIn("torch.bfloat16", wire_source)
        self.assertIn("view(torch.uint16)", wire_source)
        self.assertIn("ExplicitSpecEdgeGrpcClient", wire_source)
        server_source = (
            Path(__file__).parents[1]
            / "baselines"
            / "specedge"
            / "integration"
            / "server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("official_server.SpecExecBatchServer", server_source)
        self.assertIn("server.add_insecure_port", server_source)

    def test_specedge_normalizer_uses_precise_client_measurements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir) / "specedge" / "raw" / "run"
            requests = Path(temp_dir) / "specedge" / "requests"
            raw.mkdir(parents=True)
            requests.mkdir(parents=True)
            cycles = [
                {"client_idx": 0, "req_idx": 2, "step_idx": 0, "draft": {"end_to_end": 3}, "target": {"end_to_end": 7}, "num_accepted_tokens": 1},
                {"client_idx": 0, "req_idx": 2, "step_idx": 1, "draft": {"end_to_end": 4}, "target": {"end_to_end": 8}, "num_accepted_tokens": 3},
            ]
            (raw / "client_0.jsonl").write_text("\n".join(json.dumps(item) for item in cycles), encoding="utf-8")
            completion = {
                "global_index": 2,
                "sample_id": "HumanEval/2",
                "generated_tokens": 4,
                "ttft_ms": 13,
                "tpot_ms": 5,
                "request_e2e_ms": 28,
                "arrival_lag_ms": 2,
            }
            (requests / "client_0_requests.jsonl").write_text(json.dumps(completion), encoding="utf-8")
            normalized, wallclock = normalize_specedge(raw, {"workload_hash": "hash"})
            self.assertIsNone(wallclock)
            self.assertEqual(normalized[0]["ttft_ms"], 13)
            self.assertEqual(normalized[0]["tpot_ms"], 5)
            self.assertEqual(normalized[0]["mean_accepted_tokens_per_verify"], 3)

    def test_fastsd_normalizer_uses_run_wallclock_for_closed_loop_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir)
            request = {
                "task_id": "81",
                "generated_tokens": 8,
                "ttft_ms": 10,
                "tpot_ms": 2,
                "request_e2e_ms": 24,
                "completion_s": 0.024,
                "actual_arrival_s": 0.0,
                "accepted_total": 3,
                "drafted_total": 8,
                "mean_accepted_tokens_per_verify": 1.5,
            }
            (raw / "edge_metrics_proc0.jsonl").write_text(
                json.dumps(request) + "\n", encoding="utf-8"
            )
            (raw / "edge_metrics_summary.json").write_text(
                json.dumps({"wallclock_s": 30.0}), encoding="utf-8"
            )

            normalized, wallclock = normalize_fastsd(raw, {"workload_hash": "hash"})
            self.assertEqual(wallclock, 30.0)
            self.assertEqual(normalized[0]["mean_accepted_tokens_per_verify"], 1.5)


class ComparisonAnalysisTests(unittest.TestCase):
    def test_four_dataset_analysis_pairs_by_sample_id_and_reports_boundaries(self):
        methods = ("fastsd", "specedge_cpu_adapted", "standard_sd", "draft_only")
        datasets = ("humaneval", "gsm8k", "mgsm", "mt_bench")
        root = Path(__file__).resolve().parents[1]
        temporary_paths = []
        try:
            specs = []
            for dataset in datasets:
                for method in methods:
                    summary_fd, summary_name = tempfile.mkstemp(
                        prefix=f"analysis_{dataset}_{method}_", suffix=".json", dir=root
                    )
                    request_fd, request_name = tempfile.mkstemp(
                        prefix=f"analysis_{dataset}_{method}_", suffix=".jsonl", dir=root
                    )
                    os.close(summary_fd)
                    os.close(request_fd)
                    summary_path = Path(summary_name)
                    request_path = Path(request_name)
                    temporary_paths.extend((summary_path, request_path))
                    resource_path = None
                    if dataset == "humaneval" and method == "fastsd":
                        resource_fd, resource_name = tempfile.mkstemp(
                            prefix="analysis_resource_", suffix=".json", dir=root
                        )
                        os.close(resource_fd)
                        resource_path = Path(resource_name)
                        temporary_paths.append(resource_path)
                        resource_path.write_text(
                            json.dumps(
                                {
                                    "schema_version": 1,
                                    "cpu": {
                                        "avg": 37.5,
                                        "p95": 45.0,
                                        "sample_count": 4,
                                        "core_count": 2,
                                    },
                                    "gpu": {
                                        "utilization_gpu_pct": {
                                            "avg": 62.5,
                                            "p95": 70.0,
                                            "sample_count": 2,
                                        }
                                    },
                                }
                            ),
                            encoding="utf-8",
                        )
                    baseline = method == "specedge_cpu_adapted"
                    e2e_a = 16.0 if baseline else 8.0
                    e2e_b = 24.0 if baseline else 12.0
                    summary = {
                        "run_id": f"{dataset}-{method}",
                        "method": method,
                        "dataset": dataset,
                        "workload_hash": f"{dataset}-hash",
                        "num_requests": 2,
                        "wallclock_s": 2.0,
                        "throughput_tok_s": 4.0,
                        "goodput_tok_s": 3.0,
                        "goodput_req_s": 1.0,
                        "ttft_ms": {"avg": 10.0 if baseline else 5.0, "p95": 12.0},
                        "tpot_ms": {"avg": 4.0 if baseline else 2.0, "p95": 5.0},
                        "e2e_ms": {"avg": (e2e_a + e2e_b) / 2.0, "p95": e2e_b},
                        "accept_rate": 0.5,
                        "mean_accepted_tokens_per_verify": 2.0,
                        "quality_exact_match": 0.75 if dataset in {"gsm8k", "mgsm"} else None,
                        "mt_bench_turn_policy": "first_turn_only" if dataset == "mt_bench" else "not_applicable",
                        "requests_path": str(request_path),
                        "resource_metrics": {
                            "network_rtt_ms": {"avg": 3.0, "total": 6.0},
                            "request_bytes": {"total": 100.0},
                            "response_bytes": {"total": 200.0},
                            "rpc_count": {"total": 4.0},
                        },
                    }
                    if resource_path is not None:
                        summary["resource_metrics_path"] = str(resource_path)
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    records = [
                        {
                            "sample_id": "b",
                            "e2e_ms": e2e_b,
                            "ttft_ms": 6.0 if baseline else 3.0,
                            "tpot_ms": 5.0 if baseline else 2.5,
                            "generated_tokens": 4,
                            "request_bytes": 50,
                            "response_bytes": 100,
                            "rpc_count": 2,
                        },
                        {
                            "sample_id": "a",
                            "e2e_ms": e2e_a,
                            "ttft_ms": 4.0 if baseline else 2.0,
                            "tpot_ms": 3.0 if baseline else 1.5,
                            "generated_tokens": 4,
                            "request_bytes": 50,
                            "response_bytes": 100,
                            "rpc_count": 2,
                        },
                    ]
                    request_path.write_text(
                        "".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8",
                    )
                    specs.append(f"{dataset}/{method}={summary_path}")

            report = build_analysis_report(specs)
            self.assertEqual(set(report["valid_datasets"]), set(datasets))
            fastsd_row = next(
                row
                for row in report["speedup_rows"]
                if row["scope"] == "per-dataset"
                and row["dataset"] == "humaneval"
                and row["method"] == "fastsd"
            )
            self.assertEqual(fastsd_row["pair_count"], 2)
            self.assertEqual(fastsd_row["metrics"]["e2e_ms"]["n"], 2)
            for scope, dataset in (
                ("per-dataset", "humaneval"),
                ("macro", "all"),
                ("micro", "all"),
            ):
                self.assertEqual(
                    {
                        row["baseline"]
                        for row in report["speedup_rows"]
                        if row["method"] == "fastsd"
                        and row["scope"] == scope
                        and row["dataset"] == dataset
                    },
                    {"specedge_cpu_adapted", "standard_sd", "draft_only"},
                )
            self.assertAlmostEqual(
                fastsd_row["metrics"]["e2e_ms"]["speedup_x"], 2.0
            )
            self.assertEqual(
                fastsd_row["metrics"]["e2e_ms"]["iterations"],
                ANALYSIS_BOOTSTRAP_ITERATIONS,
            )
            self.assertTrue(
                any(row["scope"] == "macro" for row in report["performance_rows"])
            )
            self.assertTrue(
                any(row["scope"] == "micro" for row in report["performance_rows"])
            )
            self.assertAlmostEqual(fastsd_row["metrics"]["e2e_ms"]["ci95_x"][0], 2.0)
            fastsd_performance = next(
                row
                for row in report["performance_rows"]
                if row["scope"] == "per-dataset"
                and row["dataset"] == "humaneval"
                and row["method"] == "fastsd"
            )
            self.assertEqual(fastsd_performance["cpu_util_pct"], 37.5)
            self.assertEqual(fastsd_performance["cpu_util_p95_pct"], 45.0)
            self.assertEqual(fastsd_performance["gpu_util_pct"], 62.5)
            self.assertEqual(fastsd_performance["gpu_util_p95_pct"], 70.0)
            self.assertEqual(fastsd_performance["request_bytes_per_request"], 50.0)
            micro_performance = next(
                row
                for row in report["performance_rows"]
                if row["scope"] == "micro" and row["method"] == "fastsd"
            )
            self.assertEqual(micro_performance["request_bytes_total"], 400.0)
            self.assertEqual(micro_performance["request_bytes_per_request"], 50.0)
            self.assertIsNone(micro_performance["cpu_util_p95_pct"])
            self.assertIsNone(micro_performance["gpu_util_p95_pct"])
            macro_performance = next(
                row
                for row in report["performance_rows"]
                if row["scope"] == "macro" and row["method"] == "fastsd"
            )
            self.assertIsNone(macro_performance["request_bytes_total"])
            self.assertEqual(macro_performance["request_bytes_per_request"], 50.0)
            self.assertEqual(macro_performance["cpu_util_p95_pct"], 45.0)
            self.assertEqual(macro_performance["gpu_util_p95_pct"], 70.0)
            humaneval_quality = next(
                row
                for row in report["quality_rows"]
                if row["dataset"] == "humaneval" and row["method"] == "fastsd"
            )
            self.assertIsNone(humaneval_quality["value"])
            self.assertIn("isolated HumanEval", humaneval_quality["evidence"])
            mt_quality = next(
                row
                for row in report["quality_rows"]
                if row["dataset"] == "mt_bench" and row["method"] == "fastsd"
            )
            self.assertEqual(mt_quality["mt_bench_policy"], "first_turn_only")
            self.assertIn("N/A", report["markdown"])
            self.assertIn("macro", report["markdown"])
            self.assertIn("micro", report["markdown"])

            output_fd, output_name = tempfile.mkstemp(
                prefix="analysis_report_", suffix=".md", dir=root
            )
            os.close(output_fd)
            output = Path(output_name)
            output.unlink()
            temporary_paths.append(output)
            self.assertEqual(
                analyze(specs, str(output), baseline="specedge_cpu_adapted"),
                0,
            )
            self.assertIn(
                "# FastSD 四数据集四方法分析报告",
                output.read_text(encoding="utf-8"),
            )
        finally:
            for path in temporary_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def test_workload_hash_gate_requires_all_summary_hashes_and_consistent_requests(self):
        root = Path(__file__).resolve().parents[1]
        temporary_paths = []

        def make_case(prefix: str, missing_summary_method: str | None = None, bad_request_method: str | None = None):
            specs = []
            for method in ("fastsd", "specedge_cpu_adapted", "standard_sd", "draft_only"):
                summary_fd, summary_name = tempfile.mkstemp(
                    prefix=f"analysis_{prefix}_{method}_", suffix=".json", dir=root
                )
                request_fd, request_name = tempfile.mkstemp(
                    prefix=f"analysis_{prefix}_{method}_", suffix=".jsonl", dir=root
                )
                os.close(summary_fd)
                os.close(request_fd)
                summary_path = Path(summary_name)
                request_path = Path(request_name)
                temporary_paths.extend((summary_path, request_path))
                summary = {
                    "dataset": "humaneval",
                    "method": method,
                    "num_requests": 1,
                    "requests_path": str(request_path),
                    "e2e_ms": {"avg": 1.0, "p95": 1.0},
                    "ttft_ms": {"avg": 1.0, "p95": 1.0},
                    "tpot_ms": {"avg": 1.0, "p95": 1.0},
                }
                if method != missing_summary_method:
                    summary["workload_hash"] = "same-workload"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                request = {"sample_id": "s1", "e2e_ms": 1.0, "ttft_ms": 1.0, "tpot_ms": 1.0}
                if method == bad_request_method:
                    request["workload_hash"] = "different-workload"
                request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
                specs.append(f"humaneval/{method}={summary_path}")
            return specs

        try:
            missing_report = build_analysis_report(
                make_case("missing", missing_summary_method="standard_sd")
            )
            self.assertNotIn("humaneval", missing_report["valid_datasets"])
            self.assertTrue(
                any(
                    "workload_hash missing for methods=standard_sd" in issue
                    for issue in missing_report["issues"]
                )
            )

            conflict_report = build_analysis_report(
                make_case("request_conflict", bad_request_method="draft_only")
            )
            self.assertNotIn("humaneval", conflict_report["valid_datasets"])
            self.assertTrue(
                any("request workload_hash conflict" in issue for issue in conflict_report["issues"])
            )
        finally:
            for path in temporary_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def test_macro_bootstrap_resamples_pairs_inside_each_dataset(self):
        def records(values):
            return [
                {
                    "sample_id": str(index),
                    "e2e_ms": value,
                    "ttft_ms": value,
                    "tpot_ms": value,
                }
                for index, value in enumerate(values)
            ]

        parts = [
            _analysis_pair_records(
                records([1.0, 1.0]), records([2.0, 2.0]), dataset="dataset_a"
            ),
            _analysis_pair_records(
                records([1.0, 3.0]), records([2.0, 2.0]), dataset="dataset_b"
            ),
        ]
        first = _analysis_macro_pair(parts)
        second = _analysis_macro_pair(parts)
        metric = first["metrics"]["e2e_ms"]
        self.assertEqual(metric, second["metrics"]["e2e_ms"])
        self.assertEqual(metric["iterations"], ANALYSIS_BOOTSTRAP_ITERATIONS)
        self.assertEqual(metric["seed"], ANALYSIS_BOOTSTRAP_SEED)
        self.assertAlmostEqual(metric["speedup_x"], 1.5)

        expected_bootstrap = []
        rng = random.Random(ANALYSIS_BOOTSTRAP_SEED)
        pair_sets = [
            [(1.0, 2.0), (1.0, 2.0)],
            [(1.0, 2.0), (3.0, 2.0)],
        ]
        for _ in range(ANALYSIS_BOOTSTRAP_ITERATIONS):
            dataset_speedups = []
            for pairs in pair_sets:
                sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                dataset_speedups.append(
                    statistics.mean(right for _, right in sample)
                    / statistics.mean(left for left, _ in sample)
                )
            expected_bootstrap.append(statistics.mean(dataset_speedups))
        expected_ci = [
            percentile(expected_bootstrap, 0.025),
            percentile(expected_bootstrap, 0.975),
        ]
        self.assertAlmostEqual(metric["ci95_x"][0], expected_ci[0])
        self.assertAlmostEqual(metric["ci95_x"][1], expected_ci[1])

    def test_resource_sidecars_parse_real_mpstat_and_nvidia_csv_formats(self):
        mpstat_text = """Linux 6.8.0 (node3)  08/25/2026  _x86_64_ (32 CPU)
12:00:00 PM CPU    %usr %nice %sys %iowait %irq %soft %steal %guest %gnice %idle
12:00:01 PM 56     1.00  0.00  1.00 0.00    0.00 0.00   0.00 0.00 0.00 90.00
12:00:01 PM 57     1.00  0.00  1.00 0.00    0.00 0.00   0.00 0.00 0.00 80.00
12:00:02 PM 56     1.00  0.00  1.00 0.00    0.00 0.00   0.00 0.00 0.00 70.00
12:00:02 PM 57     1.00  0.00  1.00 0.00    0.00 0.00   0.00 0.00 0.00 60.00
Average:   56     1.00  0.00  1.00 0.00    0.00 0.00   0.00 0.00 0.00 75.00
"""
        cpu = parse_mpstat_cpu_resource(mpstat_text, cores="56,57")
        self.assertAlmostEqual(cpu["avg"], 25.0)
        self.assertAlmostEqual(cpu["p95"], 38.5)
        self.assertEqual(cpu["sample_count"], 4)
        self.assertEqual(cpu["core_count"], 2)
        missing_text = "\n".join(
            line for line in mpstat_text.splitlines() if " 57 " not in line
        )
        with self.assertRaises(ValueError) as missing_error:
            parse_mpstat_cpu_resource(missing_text, cores="56,57")
        self.assertIn("missing cores=[57]", str(missing_error.exception))
        extra_text = mpstat_text.replace(
            "Average:",
            "12:00:03 PM 58     1.00  0.00  1.00 0.00    0.00 0.00   0.00 0.00 0.00 50.00\nAverage:",
            1,
        )
        with self.assertRaises(ValueError) as extra_error:
            parse_mpstat_cpu_resource(extra_text, cores="56,57")
        self.assertIn("extra cores=[58]", str(extra_error.exception))

        gpu_text = """2026/08/25 12:00:01.000, 1, GPU-test, 50 %, 10 %, 100 MiB, 900 MiB, 80.0 W
2026/08/25 12:00:02.000, 0, GPU-other, 99 %, 20 %, 200 MiB, 800 MiB, 90.0 W
2026/08/25 12:00:03.000, 1, GPU-test, 70 %, 30 %, 120 MiB, 880 MiB, 100.0 W
"""
        gpu = parse_nvidia_smi_csv_resource(
            gpu_text, gpu_index=1, gpu_csv_schema="full8"
        )
        self.assertAlmostEqual(gpu["utilization_gpu_pct"]["avg"], 60.0)
        self.assertAlmostEqual(gpu["utilization_gpu_pct"]["p95"], 69.0)
        self.assertEqual(gpu["utilization_gpu_pct"]["sample_count"], 2)
        self.assertEqual(gpu["sample_count"], 2)
        self.assertEqual(gpu["schema"], "full8")
        self.assertEqual(gpu["columns"][2], "name")

        inferred_gpu = parse_nvidia_smi_csv_resource(gpu_text, gpu_index=1)
        self.assertEqual(inferred_gpu["schema"], "full8")
        self.assertEqual(inferred_gpu["columns"], gpu["columns"])

        root = Path(__file__).resolve().parents[1]
        temporary_paths = []
        try:
            cpu_fd, cpu_name = tempfile.mkstemp(prefix="analysis_mpstat_", suffix=".txt", dir=root)
            gpu_fd, gpu_name = tempfile.mkstemp(prefix="analysis_nvidia_", suffix=".csv", dir=root)
            output_fd, output_name = tempfile.mkstemp(
                prefix="analysis_resource_metrics_", suffix=".json", dir=root
            )
            for descriptor in (cpu_fd, gpu_fd, output_fd):
                os.close(descriptor)
            cpu_path, gpu_path, output_path = Path(cpu_name), Path(gpu_name), Path(output_name)
            output_path.unlink()
            temporary_paths.extend((cpu_path, gpu_path, output_path))
            cpu_path.write_text(mpstat_text, encoding="utf-8")
            gpu_path.write_text(gpu_text, encoding="utf-8")
            self.assertEqual(
                resources(
                    str(root),
                    cpu_mpstat=str(cpu_path),
                    cpu_cores="56,57",
                    gpu_nvidia_csv=str(gpu_path),
                    gpu_index=1,
                    gpu_csv_schema="full8",
                    output=str(output_path),
                ),
                0,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["cpu"]["sample_count"], 4)
            self.assertEqual(payload["cpu"]["core_count"], 2)
            self.assertAlmostEqual(payload["cpu"]["p95"], 38.5)
            self.assertEqual(payload["gpu"]["gpu_index"], 1)
            self.assertEqual(payload["gpu"]["schema"], "full8")
            self.assertEqual(payload["gpu"]["column_count"], 8)
            self.assertAlmostEqual(
                payload["gpu"]["utilization_gpu_pct"]["p95"], 69.0
            )
        finally:
            for path in temporary_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def test_nvidia_cli_accepts_node3_compact_csv_and_normalizes_index(self):
        args = build_parser().parse_args(
            [
                "resources",
                "--method-dir",
                "method",
                "--gpu-nvidia-csv",
                "samples.csv",
                "--gpu-index",
                "1",
                "--gpu-csv-schema",
                "compact-name6",
            ]
        )
        self.assertIsInstance(args.gpu_index, int)
        self.assertEqual(args.gpu_csv_schema, "compact-name6")

        compact_gpu_text = (
            "2026/08/25 08:54:53.992, 1, NVIDIA RTX A6000, 0, 15911, 19.51\n"
        )
        gpu = parse_nvidia_smi_csv_resource(
            compact_gpu_text,
            gpu_index="1",
            gpu_csv_schema="compact-name6",
        )
        self.assertEqual(gpu["gpu_index"], 1)
        self.assertEqual(gpu["sample_count"], 1)
        self.assertEqual(gpu["schema"], "compact-name6")
        self.assertEqual(gpu["columns"], [
            "timestamp",
            "index",
            "name",
            "utilization.gpu",
            "memory.used",
            "power.draw",
        ])
        self.assertAlmostEqual(gpu["utilization_gpu_pct"]["avg"], 0.0)
        self.assertAlmostEqual(gpu["memory_used_mib"]["avg"], 15911.0)
        self.assertAlmostEqual(gpu["power_draw_w"]["avg"], 19.51)
        self.assertIsNone(gpu["utilization_memory_pct"])
        self.assertIsNone(gpu["memory_free_mib"])

    def test_nvidia_compact_no_name6_requires_schema_and_parses_temperature(self):
        compact_no_name_gpu_text = (
            "2026/08/25 08:54:53.992, 1, 12, 15911, 19.51, 43\n"
            "2026/08/25 08:54:54.992, 0, 88, 200, 90.50, 55\n"
        )
        with self.assertRaisesRegex(ValueError, "ambiguous six-column"):
            parse_nvidia_smi_csv_resource(compact_no_name_gpu_text, gpu_index=1)

        gpu = parse_nvidia_smi_csv_resource(
            compact_no_name_gpu_text,
            gpu_index="1",
            gpu_csv_schema="compact-no-name6",
        )
        self.assertEqual(gpu["gpu_index"], 1)
        self.assertEqual(gpu["sample_count"], 1)
        self.assertEqual(gpu["schema"], "compact-no-name6")
        self.assertEqual(
            gpu["columns"],
            [
                "timestamp",
                "index",
                "utilization.gpu",
                "memory.used",
                "power.draw",
                "temperature.gpu",
            ],
        )
        self.assertAlmostEqual(gpu["utilization_gpu_pct"]["avg"], 12.0)
        self.assertAlmostEqual(gpu["memory_used_mib"]["avg"], 15911.0)
        self.assertAlmostEqual(gpu["power_draw_w"]["avg"], 19.51)
        self.assertAlmostEqual(gpu["temperature_gpu_c"]["avg"], 43.0)
        self.assertIsNone(gpu["utilization_memory_pct"])
        self.assertIsNone(gpu["memory_free_mib"])

    def test_micro_resource_aggregation_weights_sidecar_samples_and_keeps_p95_na(self):
        entries = {
            ("gsm8k", "fastsd"): {
                "resources": {
                    "cpu": {"avg": 10.0, "p95": 20.0, "sample_count": 1},
                    "gpu": {
                        "utilization_gpu_pct": {
                            "avg": 20.0,
                            "p95": 30.0,
                            "sample_count": 2,
                        }
                    },
                }
            },
            ("mgsm", "fastsd"): {
                "resources": {
                    "cpu": {"avg": 90.0, "p95": 100.0, "sample_count": 9},
                    "gpu": {
                        "utilization_gpu_pct": {
                            "avg": 80.0,
                            "p95": 90.0,
                            "sample_count": 8,
                        }
                    },
                }
            },
        }
        aggregate = _analysis_aggregate_resources(entries, "fastsd")
        self.assertAlmostEqual(aggregate["cpu"]["avg"], 82.0)
        self.assertEqual(aggregate["cpu"]["sample_count"], 10)
        self.assertAlmostEqual(
            aggregate["gpu"]["utilization_gpu_pct"]["avg"], 68.0
        )
        self.assertEqual(
            aggregate["gpu"]["utilization_gpu_pct"]["sample_count"], 10
        )
        self.assertNotIn("p95", aggregate["cpu"])
        self.assertNotIn("p95", aggregate["gpu"]["utilization_gpu_pct"])


if __name__ == "__main__":
    unittest.main()
