import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "merge_specedge_replay.py"
SPEC = importlib.util.spec_from_file_location("merge_specedge_replay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MERGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGER)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class MergeSpecEdgeReplayTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        trace = root / "trace.jsonl"
        _write_jsonl(
            trace,
            [
                {
                    "arrival_index": 0,
                    "client_id": 0,
                    "scheduled_offset_s": 0.1,
                },
                {
                    "arrival_index": 1,
                    "client_id": 1,
                    "scheduled_offset_s": 0.2,
                },
            ],
        )
        trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
        request0 = root / "requests0.jsonl"
        request1 = root / "requests1.jsonl"
        _write_jsonl(
            request0,
            [
                {
                    "run_id": "run-1",
                    "trace_index": 0,
                    "client_id": 0,
                    "status": "completed",
                    "actual_arrival_monotonic_s": 10.1,
                    "completion_monotonic_s": 10.4,
                    "e2e_from_arrival_s": 0.3,
                    "arrival_lag_s": 0.0,
                    "queue_wait_s": 0.0,
                    "client_result": {"generated_token_count": 3},
                }
            ],
        )
        _write_jsonl(
            request1,
            [
                {
                    "run_id": "run-1",
                    "trace_index": 1,
                    "client_id": 1,
                    "status": "completed",
                    "actual_arrival_monotonic_s": 10.2,
                    "completion_monotonic_s": 10.6,
                    "e2e_from_arrival_s": 0.4,
                    "arrival_lag_s": 0.0,
                    "queue_wait_s": 0.01,
                    "client_result": {"generated_token_count": 5},
                }
            ],
        )
        summary0 = root / "summary0.json"
        summary1 = root / "summary1.json"
        for path, client_id in ((summary0, 0), (summary1, 1)):
            path.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "trace_sha256": trace_sha,
                        "trace_client_ids": [client_id],
                    }
                ),
                encoding="utf-8",
            )
        return trace, request0, request1, summary0, summary1

    def test_merge_restores_trace_order_and_summarizes(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace, request0, request1, summary0, summary1 = self._fixture(root)
            output = root / "requests.jsonl"
            summary_output = root / "summary.json"
            result = MERGER.merge(
                trace_path=trace,
                request_paths=[request1, request0],
                summary_paths=[summary1, summary0],
                output_path=output,
                summary_output_path=summary_output,
                run_id="run-1",
                max_requests=None,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["trace_index"] for row in rows], [0, 1])
            self.assertEqual(result["completion_rate"], 1.0)
            self.assertAlmostEqual(result["measurement_window_s"], 0.5)
            self.assertTrue(result["generated_token_count_available"])
            self.assertEqual(result["total_generated_tokens"], 8)
            self.assertAlmostEqual(result["system_tok_per_s"], 16.0)
            self.assertTrue(summary_output.is_file())

    def test_merge_rejects_missing_trace_partition(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trace, request0, _, summary0, _ = self._fixture(root)
            with self.assertRaisesRegex(SystemExit, "does not cover"):
                MERGER.merge(
                    trace_path=trace,
                    request_paths=[request0],
                    summary_paths=[summary0],
                    output_path=root / "requests.jsonl",
                    summary_output_path=root / "summary.json",
                    run_id="run-1",
                    max_requests=None,
                )


if __name__ == "__main__":
    unittest.main()
