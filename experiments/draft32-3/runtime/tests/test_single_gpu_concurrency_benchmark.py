import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


class SingleGpuConcurrencyBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_torch = types.ModuleType("torch")
        fake_autogptq = types.ModuleType("auto_gptq")
        fake_transformers = types.ModuleType("transformers")

        fake_autogptq.AutoGPTQForCausalLM = object()
        fake_transformers.AutoModelForCausalLM = object()
        fake_transformers.AutoTokenizer = object()

        cls._old_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "auto_gptq", "transformers")
        }
        sys.modules["torch"] = fake_torch
        sys.modules["auto_gptq"] = fake_autogptq
        sys.modules["transformers"] = fake_transformers
        cls.bench = importlib.import_module("scripts.benchmark_single_gpu_concurrency")

    @classmethod
    def tearDownClass(cls):
        for name, module in cls._old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_build_prompt_uses_default_prompt(self):
        prompt = self.bench.build_prompt(None, 1)
        self.assertEqual(prompt, self.bench.PROMPT)

    def test_build_prompt_repeats_file_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompt.txt"
            prompt_file.write_text("alpha beta", encoding="utf-8")

            prompt = self.bench.build_prompt(str(prompt_file), 3)

        self.assertEqual(prompt, "alpha beta\n\nalpha beta\n\nalpha beta")


if __name__ == "__main__":
    unittest.main()
