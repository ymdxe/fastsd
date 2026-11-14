"""Benchmark windows, normalization, zero denominators and historical metrics."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

from scripts.zjh_ab import accept


class AcceptanceReportTests(unittest.TestCase):
    def paired_rows(self):
        with tempfile.TemporaryDirectory() as root:
            condition = self.condition(root, dict(prefetch_async_failed=0,
                prefetch_bytes=0, cache_transfers=dict(h2d_bytes=1024, d2h_bytes=1024, copies=2)))
            row = accept.summarize(condition, expected_requests=2)
        row.update(target_log_present=True, latency_samples=dict(ttft=2, tpot=2),
                   ttft_p50=100., ttft_p95=120., tpot_p50=10., tpot_p95=12.)
        for record in row["records"]:
            record["output_token_ids"] = [1, 2]
            record["output_text"] = "same decoded output"
        base, pref = deepcopy(row), deepcopy(row)
        base["dir"], pref["dir"] = "noprefetch", "prefetch"
        pref["prefetch_metrics"]["prefetch_bytes"] = 512
        pref["ready_hit"] = 1
        return [base, pref]

    def test_prefetch_pair_checks_complete_identity_output_latency_and_transfer(self):
        checks = accept.prefetch_pair_gates(self.paired_rows())
        self.assertTrue(all(ok for _, ok, _ in checks), checks)

    def test_pair_token_mismatch_fails_even_when_decoded_text_is_equal(self):
        rows = self.paired_rows()
        rows[1]["records"][0]["output_token_ids"] = [1, 3]
        check = next(c for c in accept.prefetch_pair_gates(rows) if c[0] == "paired output correctness")
        self.assertFalse(check[1])

    def test_pair_duplicate_tasks_and_missing_outputs_cannot_pass(self):
        rows = self.paired_rows()
        rows[1]["records"][1]["task_id"] = rows[1]["records"][0]["task_id"]
        checks = accept.prefetch_pair_gates(rows)
        self.assertFalse(next(c[1] for c in checks if c[0] == "prefetch: exact unique request count"))
        rows = self.paired_rows()
        rows[0]["records"][0].pop("output_token_ids")
        rows[0]["records"][0].pop("output_text")
        self.assertFalse(next(c[1] for c in accept.prefetch_pair_gates(rows) if c[0] == "paired output correctness"))

    def test_pair_performance_regressions_and_oom_fail(self):
        rows = self.paired_rows()
        rows[1].update(ttft_p95=130, tpot_p50=20, h2d_bytes=2048, oom_in_log=True)
        failures = [label for label, ok, _ in accept.prefetch_pair_gates(rows) if not ok]
        self.assertIn("prefetch ttft_p95 within tolerance", failures)
        self.assertIn("prefetch tpot_p50 within tolerance", failures)
        self.assertIn("total H2D within tolerance", failures)
        self.assertIn("prefetch: no CUDA OOM", failures)

    def test_incomplete_legacy_reference_never_certifies_readiness(self):
        rows = self.paired_rows()
        legacy = deepcopy(rows[1])
        legacy.update(incomplete=True, ready_hit=0)
        checks = accept.prefetch_pair_gates(rows, legacy=legacy)
        self.assertFalse(next(c[1] for c in checks if c[0] == "ready hits at least legacy"))

    def condition(self, root, scheduler=None, summaries=(), count=2):
        condition = Path(root) / "condition"
        (condition / "target").mkdir(parents=True)
        (condition / "edge").mkdir()
        (condition / "target" / "scheduler_metrics.json").write_text(
            json.dumps({"fastsd_scheduler": scheduler or {}}))
        records = [dict(task_id=str(i), generated_tokens=100,
            processing_start_epoch_s=110 + i * 10, first_token_epoch_s=112 + i * 10,
            completion_epoch_s=115 + i * 10, processing_start_monotonic_s=i * 10000,
            first_token_monotonic_s=i * 10000 + 2, completion_monotonic_s=i * 10000 + 5)
            for i in range(count)]
        (condition / "edge" / "edge_metrics_proc0.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))
        for index, summary in enumerate(summaries):
            (condition / "edge" / f"edge_metrics_summary_{index}.json").write_text(json.dumps(summary))
        return str(condition)

    def test_main_throughput_uses_multi_host_benchmark_epoch_window(self):
        summaries = [dict(benchmark_start_epoch_s=100, benchmark_end_epoch_s=130),
                     dict(benchmark_start_epoch_s=105, benchmark_end_epoch_s=140)]
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, summaries=summaries), expected_requests=2)
        self.assertEqual(row["span_s"], 40)
        self.assertEqual(row["span_basis"], "benchmark_epoch_window")
        self.assertEqual(row["tok_per_s"], 5)
        self.assertGreater(row["busy_tok_per_s"], row["tok_per_s"])
        self.assertFalse(row["incomplete"])

    def test_request_replay_metrics_are_collected_separately_from_kv_evictions(self):
        with tempfile.TemporaryDirectory() as root:
            condition = self.condition(root, {"kv_residency": {"residency_evictions": 7}})
            (Path(condition) / "target" / "idempotency_metrics.json").write_text(json.dumps({
                "idempotent_cache_hit": 3, "idempotent_inflight_wait": 2, "cache_evictions": 1}))
            path = Path(condition) / "edge" / "edge_metrics_proc0.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0].update(retry_count=2, replayed=True)
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            row = accept.summarize(condition, expected_requests=2)
        self.assertEqual((row["idempotent_cache_hit"], row["idempotent_inflight_wait"],
                          row["cache_evictions"], row["evictions"]), (3, 2, 1, 7))
        self.assertEqual((row["retry_count"], row["replayed_requests"]), (2, 1))

    def test_legacy_record_and_metric_schema_remains_readable(self):
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root))
        self.assertEqual(row["span_s"], 15)
        self.assertEqual(row["span_basis"], "legacy_request_span")
        self.assertTrue(row["incomplete"])
        self.assertIsNone(row["actual_plans"])
        self.assertIsNone(row["ready_per_actual_plan"])
        self.assertIsNone(row["h2d_bytes_per_ready"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            accept.print_table([row])
            accept.print_diagnostics([row])
        self.assertIn("incomplete", output.getvalue())
        self.assertIn("n/a", output.getvalue())

    def test_old_summary_duration_is_labelled_as_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, summaries=[dict(wallclock_s=50)]))
        self.assertEqual((row["span_s"], row["span_basis"]), (50, "legacy_wallclock_duration"))

    def test_per_plan_denominator_never_uses_snapshot_or_request_count(self):
        scheduler = dict(actual_plans=10, prediction_snapshots_consumed=24, prefetch_ready_hit=2,
            prefetch_refresh_count=8, lookahead_build_count=24,
            cache_transfers=dict(h2d_bytes=1024, copies=6, kv_transfer_prefetch_avg_s=.44),
            batch_compute_avg_s=.11, prefetch_ready_by_horizon={"1": {"ready": 2}},
            prefetch_prediction_by_horizon={"1": dict(plans=8, predicted=16, actual=20, hits=12)})
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, scheduler))
        self.assertEqual(row["ready_per_actual_plan"], .2)
        self.assertEqual(row["h2d_bytes_per_ready"], 512)
        self.assertEqual(row["copies_per_ready"], 3)
        self.assertEqual(row["transfer_per_batch_window"], 4)
        self.assertTrue(row["transfer_exceeds_batch_window"])
        h1 = row["prediction_by_horizon"]["1"]
        self.assertEqual((h1["precision"], h1["recall"], h1["coverage_per_actual_plan"]), (.75, .6, .8))

    def test_legacy_iterations_can_supply_actual_plan_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, dict(iterations=5, prefetch_ready_hit=1)))
        self.assertEqual(row["ready_per_actual_plan"], .2)

    def test_empty_records_and_zero_ready_do_not_divide_by_zero(self):
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, dict(actual_plans=0), count=0))
        self.assertEqual(row["tok_per_s"], 0)
        self.assertIsNone(row["copies_per_ready"])
        self.assertIsNone(row["transfer_exceeds_batch_window"])

    def test_incomplete_condition_fails_completion_gate(self):
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root))
        completion = [result for result in accept.gates([row], .05) if "complete benchmark" in result[0]]
        self.assertEqual(len(completion), 1)
        self.assertFalse(completion[0][1])

    def test_missing_transfer_counters_are_distinct_from_explicit_zero(self):
        for transfers, missing in (({}, True), (dict(h2d_bytes=0, d2h_bytes=0, copies=0), False)):
            with tempfile.TemporaryDirectory() as root:
                row = accept.summarize(self.condition(root, dict(cache_transfers=transfers)))
            self.assertEqual(bool(row["missing_transfer_counters"]), missing)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                accept.print_table([row])
            self.assertEqual("0.0B" in output.getvalue(), not missing)

    def test_vanilla_evictions_do_not_require_fastsd_activation_hook(self):
        scheduler = dict(kv_residency=dict(mode="active", residency_activations=0,
                                          residency_evictions=16))
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, scheduler))
        pressure = [r for r in accept.gates([row], .05) if "KV pressure exercised" in r[0]]
        self.assertTrue(pressure[0][1])

    def test_emergency_evictions_are_reported_separately(self):
        scheduler = dict(kv_residency=dict(residency_evictions=5,
            residency_emergency_eviction_rounds=2, residency_emergency_evictions=3,
            residency_emergency_blocked_rounds=1))
        with tempfile.TemporaryDirectory() as root:
            row = accept.summarize(self.condition(root, scheduler))
        self.assertEqual(row["evictions"], 5)
        self.assertEqual(row["emergency_eviction_rounds"], 2)
        self.assertEqual(row["emergency_evictions"], 3)
        self.assertEqual(row["emergency_blocked_rounds"], 1)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            accept.print_diagnostics([row])
        self.assertIn("emergency_evict=3", output.getvalue())


if __name__ == "__main__":
    unittest.main()
