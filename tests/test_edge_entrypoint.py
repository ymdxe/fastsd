import concurrent.futures
import importlib.util
import multiprocessing.spawn as mp_spawn
import os
import py_compile
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
EDGE_PATH = REPO_ROOT / "edge" / "edge.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _load_edge_with_auto_gptq_blocked():
    """Import edge.py with lightweight dependency stubs and no auto_gptq."""

    class FakeAutoModelForCausalLM:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))

            class LoadedModel:
                def eval(self):
                    return self

            return LoadedModel()

    class FakeDecoding:
        pass

    fake_torch = _module("torch", Tensor=object, long=object)
    fake_torch.no_grad = lambda function=None: function or (lambda wrapped: wrapped)
    fake_src = _module("src")
    fake_src.__path__ = []
    stubs = {
        "torch": fake_torch,
        "requests": _module("requests", Session=object),
        "transformers": _module(
            "transformers", AutoModelForCausalLM=FakeAutoModelForCausalLM
        ),
        "src": fake_src,
        "src.arrival": _module(
            "src.arrival",
            poisson_arrival_offsets=lambda *args, **kwargs: [],
            shard_samples=lambda samples, *args, **kwargs: samples,
        ),
        "src.engine": _module("src.engine", Decoding=FakeDecoding),
        "src.kvcache": _module("src.kvcache", KVCacheModel=object),
        "src.metrics": _module(
            "src.metrics", elapsed_ms=lambda *args, **kwargs: 0.0, tpot_ms=lambda *args, **kwargs: 0.0
        ),
        "src.runtime": _module(
            "src.runtime",
            configure_torch_threads=lambda *args, **kwargs: None,
            resolve_dtype=lambda *args, **kwargs: "float32",
        ),
        "src.util": _module(
            "src.util",
            parse_arguments=lambda *args, **kwargs: None,
            seed_everything=lambda *args, **kwargs: None,
        ),
        # A None entry makes any attempted top-level import fail immediately.
        "auto_gptq": None,
    }
    module_name = "edge_optional_dependency_test"
    spec = importlib.util.spec_from_file_location(module_name, EDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module, FakeAutoModelForCausalLM


class EdgeEntrypointTests(unittest.TestCase):
    def test_edge_module_compiles(self):
        py_compile.compile(str(EDGE_PATH), doraise=True)

    def test_mt_bench_uses_chat_template(self):
        source = EDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('self.args.dataset != "mt_bench"', source)
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn("configure_torch_threads", source)
        self.assertIn("resolve_dtype", source)
        self.assertIn("if torch.cuda.is_available()", source)
        self.assertIn("configure_spawn_executable()", source)
        self.assertIn("mp.set_executable(executable)", source)

    def test_cpu_non_gptq_import_and_load_do_not_require_auto_gptq(self):
        edge_module, fake_auto_model = _load_edge_with_auto_gptq_blocked()
        runner = object.__new__(edge_module.EdgeRunner)
        runner.args = types.SimpleNamespace(draft_dtype="float32")

        model_dir = str(REPO_ROOT / "non-gptq-test-model")
        loaded_model = runner._load_draft_model(model_dir, "cpu")

        self.assertIsNotNone(loaded_model)
        self.assertEqual(len(fake_auto_model.calls), 1)
        self.assertEqual(fake_auto_model.calls[0][1]["device_map"], {"": "cpu"})

    def test_gptq_request_reports_missing_auto_gptq_clearly(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        runner = object.__new__(edge_module.EdgeRunner)
        runner.args = types.SimpleNamespace(draft_dtype="float32")

        model_dir = str(REPO_ROOT / "gptq-test-model")
        with patch.dict(sys.modules, {"auto_gptq": None}):
            with patch.object(edge_module.os.path, "exists", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "GPTQ draft requested.*auto_gptq"):
                    runner._load_draft_model(model_dir, "cpu")

    def test_spawn_uses_explicit_python_wrapper(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        wrapper = sys.executable
        expected_wrapper = os.path.abspath(os.path.expanduser(wrapper))
        previous_wrapper = mp_spawn.get_executable()
        try:
            with patch.dict(os.environ, {"PYTHON_BIN": wrapper}):
                configured = edge_module.configure_spawn_executable()
            self.assertEqual(configured, expected_wrapper)
            self.assertEqual(os.fsdecode(mp_spawn.get_executable()), expected_wrapper)
        finally:
            edge_module.mp.set_executable(previous_wrapper)

    def test_invalid_python_wrapper_fails_clearly(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        invalid_wrapper = "/missing/fastsd/python314_glibc"
        with patch.dict(os.environ, {"PYTHON_BIN": invalid_wrapper}):
            with patch.object(edge_module.os.path, "isfile", return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, "PYTHON_BIN"):
                    edge_module.configure_spawn_executable()

    def test_missing_python_wrapper_preserves_multiprocessing_default(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHON_BIN", None)
            with patch.object(edge_module.mp, "set_executable") as set_executable:
                configured = edge_module.configure_spawn_executable()

        self.assertIsNone(configured)
        set_executable.assert_not_called()

    def test_edge_warmup_does_not_default_to_ten_task_cap(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        with patch.object(
            edge_module,
            "parse_arguments",
            return_value=types.SimpleNamespace(),
        ):
            with patch.object(sys, "argv", ["edge.py", "--warmup_requests", "10"]):
                args = edge_module.parse_edge_arguments()

        self.assertEqual(args.max_tasks_per_draft, 0)

    def test_edge_explicit_task_cap_remains_available_for_smoke_runs(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        with patch.object(
            edge_module,
            "parse_arguments",
            return_value=types.SimpleNamespace(),
        ):
            with patch.object(
                sys,
                "argv",
                ["edge.py", "--warmup_requests", "10", "--max_tasks_per_draft", "3"],
            ):
                args = edge_module.parse_edge_arguments()

        self.assertEqual(args.max_tasks_per_draft, 3)

    def test_first_draft_overlaps_prefill_and_gates_verify_on_prefill(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        prefill_started = threading.Event()
        release_prefill = threading.Event()
        prefill_returned = threading.Event()
        release_prefill_finish = threading.Event()
        events = []
        counts = {"prefill": 0, "draft": 0, "verify": 0}

        def prefill():
            counts["prefill"] += 1
            events.append("prefill_started")
            prefill_started.set()
            self.assertTrue(release_prefill.wait(timeout=2))
            events.append("prefill_returned")
            prefill_returned.set()
            self.assertTrue(release_prefill_finish.wait(timeout=2))
            return {"status": "prefill_ok", "session_id": "s"}

        def draft():
            self.assertTrue(prefill_started.wait(timeout=2))
            counts["draft"] += 1
            events.append("draft")
            release_prefill.set()
            self.assertTrue(prefill_returned.wait(timeout=2))
            release_prefill_finish.set()
            return [11, 12]

        def verify():
            counts["verify"] += 1
            events.append("verify")
            return {"accepted": 2, "final_token": 13}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            draft_output, response, timings = edge_module._run_first_draft_with_prefill(
                executor,
                prefill,
                draft,
                overlap=True,
                timeout=2,
                clock=time.perf_counter,
            )
            verify_future = executor.submit(verify)
            verify_future.result(timeout=2)
        finally:
            executor.shutdown(wait=True)

        self.assertEqual(counts, {"prefill": 1, "draft": 1, "verify": 1})
        self.assertEqual(draft_output, [11, 12])
        self.assertEqual(response["status"], "prefill_ok")
        self.assertLess(events.index("prefill_returned"), events.index("verify"))
        self.assertGreaterEqual(timings["prefill_ms"], 0.0)
        self.assertGreaterEqual(timings["first_draft_ms"], 0.0)
        self.assertGreater(
            timings["prefill_first_draft_overlap_ms"],
            0.0,
            msg=f"timings={timings!r}, events={events!r}",
        )

    def test_timing_update_precedes_long_poll_prefill_completion(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        prefill_started = threading.Event()
        timing_updated = threading.Event()
        events = []

        def prefill():
            events.append("prefill_started")
            prefill_started.set()
            self.assertTrue(timing_updated.wait(timeout=2))
            events.append("prefill_returned")
            return {"status": "prefill_ok"}

        def draft():
            self.assertTrue(prefill_started.wait(timeout=2))
            events.append("draft")
            return [11, 12], {
                "local_prefill_s": 0.2,
                "local_decode_per_token_s": 0.03,
            }

        def timing_update(local_timing):
            self.assertEqual(local_timing["local_prefill_s"], 0.2)
            self.assertEqual(local_timing["local_decode_per_token_s"], 0.03)
            events.append("timing_update")
            timing_updated.set()
            return {"status": "timing_ready", "push_s": 0.01}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            draft_output, response, timings = edge_module._run_first_draft_with_prefill(
                executor,
                prefill,
                draft,
                overlap=True,
                timeout=2,
                clock=time.perf_counter,
                timing_call=timing_update,
            )
            events.append("verify")
        finally:
            executor.shutdown(wait=True)

        self.assertEqual(draft_output, [11, 12])
        self.assertEqual(response["status"], "prefill_ok")
        self.assertEqual(
            events,
            ["prefill_started", "draft", "timing_update", "prefill_returned", "verify"],
        )
        self.assertEqual(timings["push_s"], 0.01)

    def test_arrival_protocol_uploads_tokens_only_after_local_round(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        events = []
        uploaded = []

        def arrival():
            events.append("arrival")
            return {"status": "arrival_recorded", "cloud_arrival_time_ns": 1}

        def draft():
            events.append("draft")
            return [11, 12, 13], {
                "local_prefill_s": 0.2,
                "local_decode_per_token_s": 0.03,
            }

        def upload(tokens):
            events.append("upload")
            uploaded.append(tokens)
            return {"status": "prefill_ok", "cloud_send_time_ns": 2}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            draft_output, response, timings = edge_module._run_first_draft_after_arrival(
                executor,
                arrival,
                draft,
                upload,
                timeout=2,
                clock=time.perf_counter,
            )
        finally:
            executor.shutdown(wait=True)

        self.assertEqual(draft_output, [11, 12, 13])
        self.assertEqual(uploaded, [[11, 12, 13]])
        self.assertEqual(response["status"], "prefill_ok")
        self.assertLess(events.index("draft"), events.index("upload"))
        self.assertEqual(timings["prefill_first_draft_overlap_ms"], 0.0)

    def test_task_executor_is_closed_when_task_impl_raises(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()

        class RecordingExecutor(concurrent.futures.ThreadPoolExecutor):
            def __init__(self):
                self.shutdown_calls = []
                super().__init__(max_workers=1)

            def shutdown(self, *args, **kwargs):
                self.shutdown_calls.append((args, kwargs))
                return super().shutdown(*args, **kwargs)

        executor = RecordingExecutor()
        runner = object.__new__(edge_module.EdgeRunner)

        def failing_impl(*args):
            args[-1]["executor"] = executor
            raise RuntimeError("local draft failure")

        runner._run_draft_process_http_impl = failing_impl
        try:
            with self.assertRaisesRegex(RuntimeError, "local draft failure"):
                runner.run_draft_process_http(None, 0)
            self.assertTrue(
                any(call_kwargs.get("wait") is True for _, call_kwargs in executor.shutdown_calls)
            )
        finally:
            executor.shutdown(wait=True)

    def test_prefill_failure_does_not_submit_verify(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        draft_calls = []
        verify_calls = []

        def prefill():
            return {"status": "prefill_failed"}

        def draft():
            draft_calls.append(1)
            return [11]

        def verify():
            verify_calls.append(1)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                edge_module._run_first_draft_with_prefill(
                    executor,
                    prefill,
                    draft,
                    overlap=True,
                    timeout=2,
                )
            self.assertEqual(len(draft_calls), 1)
            # A failed state-establishing RPC is a hard gate: callers must not
            # enqueue Verify after the helper raises.
            self.assertEqual(len(verify_calls), 0)
        finally:
            executor.shutdown(wait=True)

    def test_timing_draft_failure_cancels_blocked_prefill(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        prefill_started = threading.Event()
        cancel_received = threading.Event()
        events = []
        verify_calls = []

        def prefill():
            events.append("prefill_started")
            prefill_started.set()
            self.assertTrue(cancel_received.wait(timeout=2))
            events.append("prefill_returned")
            return {"status": "prefill_ok"}

        def draft():
            self.assertTrue(prefill_started.wait(timeout=2))
            events.append("draft")
            raise RuntimeError("local draft failed")

        def cancel():
            events.append("cancel")
            cancel_received.set()
            return {"status": "cancelled"}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with self.assertRaisesRegex(RuntimeError, "local draft failed"):
                edge_module._run_first_draft_with_prefill(
                    executor,
                    prefill,
                    draft,
                    overlap=True,
                    timeout=2,
                    cancel_call=cancel,
                )
            self.assertEqual(events, ["prefill_started", "draft", "cancel", "prefill_returned"])
            self.assertEqual(verify_calls, [])
        finally:
            executor.shutdown(wait=True)

    def test_timing_update_failure_cancels_prefill_and_blocks_verify(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        prefill_started = threading.Event()
        cancel_received = threading.Event()
        events = []

        def prefill():
            events.append("prefill_started")
            prefill_started.set()
            self.assertTrue(cancel_received.wait(timeout=2))
            events.append("prefill_returned")
            return {"status": "prefill_ok"}

        def draft():
            self.assertTrue(prefill_started.wait(timeout=2))
            events.append("draft")
            return [11], {
                "local_prefill_s": 0.2,
                "local_decode_per_token_s": 0.03,
            }

        def timing_update(_local_timing):
            events.append("timing_update")
            return {"status": "rejected"}

        def cancel():
            events.append("cancel")
            cancel_received.set()
            return {"status": "cancelled"}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with self.assertRaisesRegex(RuntimeError, "timing update failed"):
                edge_module._run_first_draft_with_prefill(
                    executor,
                    prefill,
                    draft,
                    overlap=True,
                    timeout=2,
                    timing_call=timing_update,
                    cancel_call=cancel,
                )
            self.assertEqual(
                events,
                ["prefill_started", "draft", "timing_update", "cancel", "prefill_returned"],
            )
        finally:
            executor.shutdown(wait=True)

if __name__ == "__main__":
    unittest.main()
