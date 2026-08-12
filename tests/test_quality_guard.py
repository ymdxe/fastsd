import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "experiments" / "quality_guard.py"
SPEC = importlib.util.spec_from_file_location("quality_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QUALITY_GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY_GUARD)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _completion_rows(*, benchmark: str) -> list[dict]:
    rows: list[dict] = []
    for index in range(QUALITY_GUARD.QUALITY_SAMPLE_COUNT):
        record = {
            "schema_version": 1,
            "status": "completed",
            "request_id": f"request-{index}",
            "dataset_index": index,
            "text": f"Reasoning; final answer is +{index:02d}.0.",
        }
        if benchmark == "humaneval":
            record["task_id"] = f"HumanEval/{index}"
            record["text"] = f"\n    return {index}\n"
        rows.append(record)
    return rows


class QualityGuardTests(unittest.TestCase):
    def _run_dir(self, root: Path) -> Path:
        run_dir = root / "run"
        (run_dir / "outputs").mkdir(parents=True)
        return run_dir

    def test_gsm8k_scores_fixed_first_32_with_explicit_final_number_rules(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_dir = self._run_dir(root)
            completions = _completion_rows(benchmark="gsm8k")
            # The first completion exercises comma stripping and decimal-zero
            # normalization; the rest use their raw integer answer.
            completions[0]["text"] = "there were 1,000 items, so the final result is 0.0."
            _write_jsonl(run_dir / "outputs" / "completions.jsonl", completions)
            dataset = root / "gsm8k.jsonl"
            _write_jsonl(
                dataset,
                [
                    {"question": f"q-{index}", "answer": f"work #### {index}"}
                    for index in range(40)
                ],
            )

            summary = QUALITY_GUARD.run_quality_guard(
                run_dir=run_dir,
                benchmark="gsm8k",
                dataset_path=dataset,
                argv=["quality_guard.py", "--benchmark", "gsm8k"],
            )

            quality_dir = run_dir / "quality"
            self.assertEqual(summary["result"]["correct"], 32)
            self.assertEqual(summary["result"]["final_number_exact_match"], 1.0)
            self.assertEqual(summary["fixed_sample_count"], 32)
            self.assertEqual(len(summary["parser_rules"]), 4)
            self.assertTrue((quality_dir / "summary.json").is_file())
            self.assertTrue((quality_dir / "scored_completions.jsonl").is_file())
            self.assertTrue((quality_dir / "manifest.json").is_file())
            self.assertIn("No generated code was executed", (quality_dir / "command_hints.txt").read_text(encoding="utf-8"))
            scored = [
                json.loads(line)
                for line in (quality_dir / "scored_completions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["dataset_index"] for row in scored], list(range(32)))
            self.assertEqual(scored[0]["predicted_final_number"], "0")

    def test_humaneval_materializes_official_sample_without_executing_it(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_dir = self._run_dir(root)
            _write_jsonl(run_dir / "outputs" / "completions.jsonl", _completion_rows(benchmark="humaneval"))
            dataset = root / "HumanEval.jsonl"
            _write_jsonl(
                dataset,
                [
                    {
                        "task_id": f"HumanEval/{index}",
                        "prompt": f"def f_{index}():",
                        "test": "assert True",
                        "entry_point": f"f_{index}",
                    }
                    for index in range(33)
                ],
            )

            summary = QUALITY_GUARD.run_quality_guard(
                run_dir=run_dir,
                benchmark="humaneval",
                dataset_path=dataset,
            )

            quality_dir = run_dir / "quality"
            samples = [
                json.loads(line)
                for line in (quality_dir / "humaneval_samples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(samples), 32)
            self.assertEqual(set(samples[0]), {"task_id", "completion"})
            self.assertEqual(samples[0]["task_id"], "HumanEval/0")
            self.assertEqual(samples[-1]["task_id"], "HumanEval/31")
            self.assertEqual(summary["result"]["evaluation_status"], "not_executed")
            self.assertFalse(summary["untrusted_code_executed"])
            hints = (quality_dir / "command_hints.txt").read_text(encoding="utf-8")
            self.assertIn("evaluate_functional_correctness", hints)
            self.assertIn("isolated", hints)

    def test_rejects_duplicate_or_missing_completion_before_creating_quality_dir(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_dir = self._run_dir(root)
            rows = _completion_rows(benchmark="gsm8k")
            rows[-1]["dataset_index"] = 0
            _write_jsonl(run_dir / "outputs" / "completions.jsonl", rows)
            dataset = root / "gsm8k.jsonl"
            _write_jsonl(
                dataset,
                [{"question": f"q-{index}", "answer": f"#### {index}"} for index in range(32)],
            )

            with self.assertRaisesRegex(QUALITY_GUARD.QualityGuardError, "duplicate completion"):
                QUALITY_GUARD.run_quality_guard(
                    run_dir=run_dir,
                    benchmark="gsm8k",
                    dataset_path=dataset,
                )
            self.assertFalse((run_dir / "quality").exists())

    def test_rejects_reusing_quality_directory_and_prompt_inclusive_output(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_dir = self._run_dir(root)
            rows = _completion_rows(benchmark="gsm8k")
            rows[0]["output_includes_prompt"] = True
            _write_jsonl(run_dir / "outputs" / "completions.jsonl", rows)
            dataset = root / "gsm8k.jsonl"
            _write_jsonl(
                dataset,
                [{"question": f"q-{index}", "answer": f"#### {index}"} for index in range(32)],
            )

            with self.assertRaisesRegex(QUALITY_GUARD.QualityGuardError, "includes its prompt"):
                QUALITY_GUARD.run_quality_guard(
                    run_dir=run_dir,
                    benchmark="gsm8k",
                    dataset_path=dataset,
                )
            self.assertFalse((run_dir / "quality").exists())

            (run_dir / "quality").mkdir()
            rows[0].pop("output_includes_prompt")
            _write_jsonl(run_dir / "outputs" / "completions.jsonl", rows)
            with self.assertRaisesRegex(QUALITY_GUARD.QualityGuardError, "reuse existing quality directory"):
                QUALITY_GUARD.run_quality_guard(
                    run_dir=run_dir,
                    benchmark="gsm8k",
                    dataset_path=dataset,
                )


if __name__ == "__main__":
    unittest.main()
