import importlib.util
import multiprocessing.spawn as mp_spawn
import os
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_POOL_PATH = REPO_ROOT / "benchmark" / "eval_draft_pool.py"


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _load_draft_pool_with_stubs():
    fake_torch = _module(
        "torch",
        Tensor=object,
        inference_mode=lambda: (lambda function: function),
    )
    fake_transformers = _module(
        "transformers",
        AutoModelForCausalLM=object,
        AutoTokenizer=object,
    )
    stubs = {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "auto_gptq": None,
        "src.common_metrics": _module(
            "src.common_metrics", summarize_requests=lambda *args, **kwargs: {}, write_json=lambda *args, **kwargs: None
        ),
        "src.evaluation": _module("src.evaluation", load_canonical_jsonl=lambda *args, **kwargs: []),
        "src.run_artifacts": _module(
            "src.run_artifacts",
            append_command=lambda *args, **kwargs: None,
            append_status=lambda *args, **kwargs: None,
            write_text_once=lambda *args, **kwargs: None,
        ),
        "src.runtime": _module(
            "src.runtime",
            configure_torch_threads=lambda *args, **kwargs: None,
            resolve_dtype=lambda *args, **kwargs: None,
            synchronize=lambda *args, **kwargs: None,
        ),
        "src.util": _module(
            "src.util",
            norm_logits=lambda *args, **kwargs: None,
            sample=lambda *args, **kwargs: None,
            seed_everything=lambda *args, **kwargs: None,
        ),
    }
    module_name = "draft_pool_spawn_test_module"
    spec = importlib.util.spec_from_file_location(module_name, DRAFT_POOL_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class DraftPoolSpawnTests(unittest.TestCase):
    def test_explicit_python_wrapper_is_used_by_spawn(self):
        draft_pool = _load_draft_pool_with_stubs()
        wrapper = sys.executable
        expected_wrapper = os.path.abspath(os.path.expanduser(wrapper))
        previous_wrapper = mp_spawn.get_executable()
        try:
            with patch.dict(os.environ, {"PYTHON_BIN": wrapper}):
                with patch.object(
                    draft_pool.mp,
                    "set_executable",
                    wraps=draft_pool.mp.set_executable,
                ) as set_executable:
                    configured = draft_pool.configure_spawn_executable()
            self.assertEqual(configured, expected_wrapper)
            set_executable.assert_called_once_with(expected_wrapper)
            self.assertEqual(os.fsdecode(mp_spawn.get_executable()), expected_wrapper)
        finally:
            draft_pool.mp.set_executable(previous_wrapper)

    def test_no_override_preserves_multiprocessing_default(self):
        draft_pool = _load_draft_pool_with_stubs()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHON_BIN", None)
            with patch.object(draft_pool.mp, "set_executable") as set_executable:
                configured = draft_pool.configure_spawn_executable()

        self.assertIsNone(configured)
        set_executable.assert_not_called()

    def test_command_name_override_resolves_for_ordinary_python(self):
        draft_pool = _load_draft_pool_with_stubs()
        command_name = Path(sys.executable).name
        resolved = shutil.which(command_name)
        if resolved is None:
            self.skipTest(f"{command_name} is not available on PATH")
        previous_wrapper = mp_spawn.get_executable()
        try:
            with patch.dict(os.environ, {"PYTHON_BIN": command_name}):
                configured = draft_pool.configure_spawn_executable()
            self.assertEqual(configured, os.path.abspath(resolved))
        finally:
            draft_pool.mp.set_executable(previous_wrapper)


if __name__ == "__main__":
    unittest.main()
