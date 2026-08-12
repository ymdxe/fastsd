import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baselines.specedge.adapter import official_client_factory as official  # noqa: E402
from baselines.specedge.adapter.poisson_client import ReplayContext  # noqa: E402


def _effective_config(host: str) -> dict:
    return {
        "version": 1,
        "opt": 2,
        "base": {
            "result_path": "/tmp/official-results",
            "exp_name": "factory-test",
            "dtype": "bf16",
            "seed": 7,
            "max_len": 128,
        },
        "client": {
            "host": host,
            "process_name": "edge-client",
            "draft_model": "/models/Qwen3-0.6B",
            "dataset": "mtbench",
            "reasoning": False,
            "sample_req_cnt": 1,
            "req_offset": 0,
            "max_n_beams": 32,
            "max_beam_len": 4,
            "max_branch_width": 16,
            "max_budget": 32,
            "proactive": {
                "type": "excluded",
                "max_n_beams": 32,
                "max_beam_len": 3,
                "max_branch_width": 16,
                "max_budget": 32,
            },
            "max_new_tokens": 64,
            "max_request_num": -1,
        },
    }


class _FakeTokenSequence:
    """Small tensor-shaped fixture; no torch dependency in adapter tests."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def numel(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FakeTokenSequence(self.token_ids[index])
        return self.token_ids[index]


class _FakeTokenBatch:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def numel(self) -> int:
        return sum(len(row) for row in self.rows)

    def __getitem__(self, index):
        if isinstance(index, int):
            return _FakeTokenSequence(self.rows[index])
        raise TypeError("only batch indexing is supported")


class _FakeTokenizer:
    def __init__(self) -> None:
        self.decode_calls: list[list[int]] = []

    def decode(self, token_ids: _FakeTokenSequence, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decode_calls.append(list(token_ids.token_ids))
        return " ".join(str(token_id) for token_id in token_ids.token_ids)


class OfficialSpecEdgeFactoryTests(unittest.TestCase):
    def _environment(self, root: Path) -> tuple[dict[str, str], Path, Path]:
        official_root = root / "official"
        (official_root / "src").mkdir(parents=True)
        (official_root / "src" / "config.py").write_text(
            "# intentionally not imported in this test\n", encoding="utf-8"
        )
        config_path = root / "effective.yaml"
        config_path.write_text(
            json.dumps(_effective_config("10.66.0.5:18000")), encoding="utf-8"
        )
        prompts_path = root / "prompts.jsonl"
        prompts_path.write_text(
            json.dumps(
                {
                    "arrival_index": 3,
                    "dataset_index": 17,
                    "task_id": "task-17",
                    "prompt": "saved Qwen prompt",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return (
            {
                "SPECEDGE_OFFICIAL_ROOT": str(official_root),
                "SPECEDGE_EFFECTIVE_CONFIG": str(config_path),
                "SPECEDGE_PROMPTS_PATH": str(prompts_path),
                "SPECEDGE_GRPC_ADDRESS": "10.66.0.5:18000",
                "SPECEDGE_CLIENT_ID": "1",
                "SPECEDGE_DEVICE": "cuda:0",
                "CUDA_VISIBLE_DEVICES": "2",
            },
            config_path,
            prompts_path,
        )

    def _context(self) -> ReplayContext:
        return ReplayContext(
            run_id="factory-test",
            trace_path="/tmp/arrival_trace.jsonl",
            trace_sha256="0" * 64,
            request_id="request-3",
            trace_index=3,
            client_id=1,
            scheduled_offset_s=0.3,
            run_started_monotonic_s=10.0,
            scheduled_deadline_monotonic_s=10.3,
            actual_arrival_monotonic_s=10.31,
            dispatch_monotonic_s=10.32,
        )

    def _request(self) -> dict:
        return {
            "arrival_index": 3,
            "client_id": 1,
            "dataset_index": 17,
            "task_id": "task-17",
        }

    def test_import_is_safe_without_official_runtime_dependencies(self):
        code = (
            "import sys; "
            "import baselines.specedge.adapter.official_client_factory; "
            "print(int(any(name in sys.modules for name in ('torch', 'grpc', 'yaml', 'config', 'util'))))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")

    def test_missing_explicit_paths_fail_before_runtime_import(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                official.OfficialSpecEdgeConfigurationError, "SPECEDGE_OFFICIAL_ROOT"
            ):
                official.read_environment_settings()

    def test_environment_and_saved_prompt_mapping_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            environment, _, _ = self._environment(Path(temp))
            with mock.patch.dict(os.environ, environment, clear=True):
                factory = official.OfficialSpecEdgeFactory.from_environment()

            self.assertIsNone(factory._components)
            self.assertEqual(factory.settings.client_id, 1)
            self.assertEqual(factory.config.host, "10.66.0.5:18000")
            rendered = factory.config.to_official_environment(factory.settings)
            self.assertEqual(rendered["SPECEDGE_HOST"], "10.66.0.5:18000")
            self.assertEqual(rendered["SPECEDGE_DEVICE"], "cuda:0")
            self.assertEqual(rendered["SPECEDGE_CLIENT_IDX"], "1")
            dataset_index, prompt = factory._prompt_for_request(self._request(), self._context())
            self.assertEqual(dataset_index, 17)
            self.assertEqual(prompt, "saved Qwen prompt")

    def test_endpoint_mismatch_is_rejected_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            environment, config_path, _ = self._environment(Path(temp))
            config_path.write_text(
                json.dumps(_effective_config("127.0.0.1:8000")), encoding="utf-8"
            )
            with mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(
                    official.OfficialSpecEdgeConfigurationError, "exactly match"
                ):
                    official.OfficialSpecEdgeFactory.from_environment()

    def test_edge_local_log_override_is_snapshotted_from_explicit_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment, _, _ = self._environment(root)
            local_result_path = root / "edge-official-logs"
            environment.update(
                {
                    "SPECEDGE_CLIENT_RESULT_PATH": str(local_result_path),
                    "SPECEDGE_CLIENT_EXP_NAME": "factory-test-client-1",
                }
            )
            with mock.patch.dict(os.environ, environment, clear=True):
                factory = official.OfficialSpecEdgeFactory.from_environment()

            # The factory carries an explicit snapshot, so later ambient
            # environment changes cannot redirect a live client's logs.
            with mock.patch.dict(
                os.environ,
                {"SPECEDGE_CLIENT_RESULT_PATH": "relative-untrusted"},
                clear=True,
            ):
                rendered = factory.config.to_official_environment(factory.settings)
            self.assertEqual(rendered["SPECEDGE_RESULT_PATH"], str(local_result_path))
            self.assertEqual(rendered["SPECEDGE_EXP_NAME"], "factory-test-client-1")
            self.assertEqual(rendered["SPECEDGE_PROCESS_NAME"], "edge-client_1")

    def test_factory_invokes_known_specexec_generate_with_saved_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            environment, _, _ = self._environment(Path(temp))
            with mock.patch.dict(os.environ, environment, clear=True):
                factory = official.OfficialSpecEdgeFactory.from_environment()

            class FakeClient:
                constructed = []
                generated = []
                instances = []

                def __init__(self, **kwargs):
                    self.__class__.constructed.append(kwargs)
                    self.__class__.instances.append(self)
                    self._prefix_tokens = _FakeTokenBatch([[11, 12, 21, 22, 23]])
                    self._num_original_tokens = 2
                    self._tokenizer = _FakeTokenizer()

                async def generate(self, request_index):
                    self.__class__.generated.append(request_index)

            factory._components = SimpleNamespace(
                spec_exec_client_type=FakeClient,
                client_config=SimpleNamespace(max_len=128),
            )
            factory._engine = "fake-engine"
            factory._tokenizer = "fake-tokenizer"
            factory.prepare = lambda: None

            result = asyncio.run(factory.create_client(self._request(), self._context())())
            self.assertEqual(FakeClient.generated, [17])
            self.assertEqual(FakeClient.constructed[0]["prompt"], "saved Qwen prompt")
            self.assertEqual(FakeClient.constructed[0]["engine"], "fake-engine")
            self.assertEqual(result["official_api"], "SpecExecClient.generate")
            self.assertEqual(result["dataset_index"], 17)
            self.assertEqual(result["text"], "21 22 23")
            self.assertFalse(result["output_includes_prompt"])
            self.assertEqual(result["generated_token_count"], 3)
            self.assertEqual(result["prompt_token_count"], 2)
            self.assertEqual(result["full_sequence_token_count"], 5)
            self.assertEqual(result["full_sequence_text"], "11 12 21 22 23")
            self.assertTrue(result["full_sequence_includes_prompt"])
            self.assertEqual(
                FakeClient.instances[0]._tokenizer.decode_calls,
                [[11, 12, 21, 22, 23], [21, 22, 23]],
            )

    def test_completion_extraction_rejects_impossible_pinned_client_state(self):
        bad_client = SimpleNamespace(
            _prefix_tokens=_FakeTokenBatch([[11, 12, 21]]),
            _num_original_tokens=4,
            _tokenizer=_FakeTokenizer(),
        )
        with self.assertRaisesRegex(
            official.OfficialSpecEdgeConfigurationError, "exceeds generated"
        ):
            official.OfficialSpecEdgeFactory._extract_official_completion(bad_client)


if __name__ == "__main__":
    unittest.main()
