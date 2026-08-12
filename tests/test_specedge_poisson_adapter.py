import asyncio
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baselines.specedge.adapter import poisson_client as adapter  # noqa: E402


class SpecEdgePoissonAdapterTests(unittest.TestCase):
    def _write_trace(self, directory: Path, records: list[dict]) -> Path:
        trace = directory / "arrival_trace.jsonl"
        with trace.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return trace

    def test_load_trace_rejects_non_monotonic_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = self._write_trace(
                Path(temp),
                [
                    {"request_id": "a", "scheduled_offset_s": 0.1},
                    {"request_id": "b", "scheduled_offset_s": 0.05},
                ],
            )
            with self.assertRaisesRegex(adapter.TraceFormatError, "non-decreasing"):
                adapter.load_trace(trace)

    def test_replay_uses_absolute_arrivals_and_records_queueing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root,
                [
                    {"request_id": "first", "scheduled_offset_s": 0.0, "prompt": "a"},
                    # Equal absolute deadlines ensure the first task obtains
                    # the one permit and the second queues without relying on
                    # host timer precision for a 10ms stagger.
                    {"request_id": "second", "scheduled_offset_s": 0.0, "prompt": "b"},
                ],
            )
            requests, trace_sha = adapter.load_trace(trace)
            output = root / "metrics" / "requests.jsonl"

            summary = asyncio.run(
                adapter.replay_trace(
                    requests,
                    trace_path=trace,
                    trace_sha256=trace_sha,
                    run_id="adapter-test",
                    client_factory=adapter.make_dry_run_factory(0.05),
                    output_path=output,
                    max_concurrency=1,
                )
            )

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            records_by_id = {record["request_id"]: record for record in records}
            first = records_by_id["first"]
            second = records_by_id["second"]

            self.assertEqual(summary["completed_count"], 2)
            self.assertEqual(summary["error_count"], 0)
            self.assertTrue(first["success"])
            self.assertTrue(second["success"])
            self.assertGreater(second["queue_wait_s"], 0.01)
            self.assertLess(
                second["actual_arrival_monotonic_s"], first["completion_monotonic_s"]
            )
            self.assertAlmostEqual(
                second["scheduled_deadline_monotonic_s"]
                - summary["started_monotonic_s"],
                0.0,
                places=6,
            )
            self.assertGreaterEqual(second["actual_arrival_offset_s"], 0.0)
            self.assertTrue((root / "metrics" / "requests.summary.json").is_file())

    def test_client_id_filter_keeps_global_trace_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = self._write_trace(
                Path(temp),
                [
                    {
                        "request_id": "a",
                        "arrival_index": 0,
                        "client_id": 0,
                        "scheduled_offset_s": 0.0,
                    },
                    {
                        "request_id": "b",
                        "arrival_index": 1,
                        "client_id": 1,
                        "scheduled_offset_s": 0.1,
                    },
                    {
                        "request_id": "c",
                        "arrival_index": 2,
                        "client_id": 0,
                        "scheduled_offset_s": 0.2,
                    },
                ],
            )
            requests, _ = adapter.load_trace(trace, max_requests=2, client_id=1)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].request_id, "b")
            self.assertEqual(requests[0].trace_index, 1)
            self.assertEqual(requests[0].client_id, 1)

    def test_summary_only_totals_explicit_complete_token_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root,
                [
                    {"request_id": "a", "scheduled_offset_s": 0.0},
                    {"request_id": "b", "scheduled_offset_s": 0.0},
                ],
            )
            requests, trace_sha = adapter.load_trace(trace)

            def factory(_request, _context):
                async def invoke():
                    return {
                        "text": "completion",
                        "output_includes_prompt": False,
                        "generated_token_count": 3,
                    }

                return invoke

            summary = asyncio.run(
                adapter.replay_trace(
                    requests,
                    trace_path=trace,
                    trace_sha256=trace_sha,
                    run_id="token-summary",
                    client_factory=factory,
                    output_path=root / "requests.jsonl",
                )
            )
            self.assertTrue(summary["generated_token_count_available"])
            self.assertEqual(summary["generated_token_count_record_count"], 2)
            self.assertEqual(summary["total_generated_tokens"], 6)

    def test_shared_start_barrier_uses_new_ready_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ready = root / "client-0.ready.json"
            start = root / "start.json"

            adapter._publish_ready_file(ready)
            ready_payload = json.loads(ready.read_text(encoding="utf-8"))
            self.assertEqual(ready_payload["event"], "factory_prepared")
            with self.assertRaisesRegex(adapter.AdapterConfigurationError, "overwrite"):
                adapter._publish_ready_file(ready)

            expected_start = time.monotonic() + 1.0
            start.write_text(
                json.dumps({"run_started_monotonic_s": expected_start}),
                encoding="utf-8",
            )
            self.assertEqual(
                adapter.wait_for_start_file(start, timeout_s=0.1), expected_start
            )

    def test_factory_error_is_recorded_without_guessing_client_api(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root, [{"request_id": "bad", "scheduled_offset_s": 0.0}]
            )
            requests, trace_sha = adapter.load_trace(trace)

            def broken_factory(_request, _context):
                return object()

            summary = asyncio.run(
                adapter.replay_trace(
                    requests,
                    trace_path=trace,
                    trace_sha256=trace_sha,
                    run_id="bad-factory",
                    client_factory=broken_factory,
                    output_path=root / "requests.jsonl",
                )
            )
            record = json.loads((root / "requests.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(summary["completed_count"], 0)
            self.assertEqual(summary["replay_status"], "failed")
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["error_type"], "AdapterConfigurationError")
            self.assertIn("will not guess", record["error_message"])

    def test_cli_requires_explicit_mode_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root, [{"request_id": "one", "scheduled_offset_s": 0.0}]
            )
            output = root / "requests.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "baselines" / "specedge" / "adapter" / "poisson_client.py"),
                    "--trace",
                    str(trace),
                    "--output",
                    str(output),
                    "--run-id",
                    "needs-mode",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("one of the arguments --dry-run --client-factory is required", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_request_error_preserves_artifacts_but_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root, [{"request_id": "bad", "scheduled_offset_s": 0.0}]
            )
            output = root / "requests.jsonl"

            def factory(_request, _context):
                async def invoke():
                    raise RuntimeError("simulated official RPC failure")

                return invoke

            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(adapter, "load_client_factory", return_value=factory):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    return_code = adapter.main(
                        [
                            "--trace",
                            str(trace),
                            "--output",
                            str(output),
                            "--run-id",
                            "request-error",
                            "--client-factory",
                            "test.factory:create_client",
                        ]
                    )

            self.assertEqual(return_code, 1)
            summary = json.loads(
                (root / "requests.summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["replay_status"], "failed")
            self.assertEqual(summary["error_count"], 1)
            self.assertIn("refusing to report a successful formal replay", stderr.getvalue())
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")

    def test_official_factory_requires_partition_and_env_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root,
                [
                    {
                        "request_id": "one",
                        "arrival_index": 0,
                        "client_id": 0,
                        "scheduled_offset_s": 0.0,
                    }
                ],
            )
            output = root / "requests.jsonl"
            base_command = [
                sys.executable,
                str(REPO / "baselines" / "specedge" / "adapter" / "poisson_client.py"),
                "--trace",
                str(trace),
                "--output",
                str(output),
                "--run-id",
                "official-gate",
                "--client-factory",
                "baselines.specedge.adapter.official_client_factory:create_client",
            ]
            no_partition = subprocess.run(
                base_command,
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(no_partition.returncode, 2)
            self.assertIn("requires --client-id", no_partition.stderr)
            self.assertFalse(output.exists())

            missing_env = subprocess.run(
                base_command + ["--client-id", "0", "--max-concurrency", "1"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing_env.returncode, 2)
            self.assertIn("SPECEDGE_OFFICIAL_ROOT", missing_env.stderr)
            self.assertFalse(output.exists())

    def test_refuses_to_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = self._write_trace(
                root, [{"request_id": "one", "scheduled_offset_s": 0.0}]
            )
            requests, trace_sha = adapter.load_trace(trace)
            output = root / "requests.jsonl"
            output.write_text("existing artifact\n", encoding="utf-8")
            with self.assertRaisesRegex(adapter.AdapterConfigurationError, "overwrite"):
                asyncio.run(
                    adapter.replay_trace(
                        requests,
                        trace_path=trace,
                        trace_sha256=trace_sha,
                        run_id="no-overwrite",
                        client_factory=adapter.make_dry_run_factory(),
                        output_path=output,
                    )
                )


if __name__ == "__main__":
    unittest.main()
