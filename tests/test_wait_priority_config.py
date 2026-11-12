"""Environment settings exercise the production policy without reloading it globally."""
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


@contextmanager
def policy(**settings):
    env = {key: value for key, value in os.environ.items()
           if key not in {"FASTSD_WAIT_LAMBDA", "FASTSD_WAIT_LINEAR_UNTIL_S", "FASTSD_WAIT_LINEAR_ALPHA"}}
    env.update({"FASTSD_WAIT_" + key: str(value) for key, value in settings.items()})
    spec = importlib.util.spec_from_file_location("src._wait_config_test",
        Path(__file__).resolve().parents[1] / "src" / "fastsd_scheduler.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
        yield module


class WaitPriorityConfigurationTests(unittest.TestCase):
    def test_default_is_original_exponential(self):
        with policy() as module:
            for wait in (0, 5, 30, 300):
                self.assertAlmostEqual(module.wait_priority_term(None, wait), math.exp(.01 * wait))

    def test_environment_lambda_reaches_all_priority_entrypoints(self):
        for rate in (.005, .01, .02):
            with self.subTest(rate=rate), policy(LAMBDA=rate) as module:
                for kind in ("prefill", "verify"):
                    request = {"task_type": kind, "proc_id": "p", "W_i": 20}
                    expected = math.exp(rate * 20)
                    self.assertAlmostEqual(module.compute_priority_score(request, {}), expected)
                    score = (module.prefill_priority_score(request) if kind == "prefill"
                             else module.verify_priority_score(request, {}))
                    self.assertAlmostEqual(score, expected)

    def test_explicit_lambda_remains_supported(self):
        with policy(LAMBDA=.02) as module:
            request = {"task_type": "verify", "proc_id": "p", "W_i": 10}
            self.assertAlmostEqual(module.compute_priority_score(request, {}, lamda=.005), math.exp(.05))

    def test_piecewise_is_continuous_at_threshold(self):
        for threshold in (5, 30, 100):
            for alpha in (None, 0, .02):
                settings = dict(LAMBDA=.01, LINEAR_UNTIL_S=threshold)
                if alpha is not None:
                    settings["LINEAR_ALPHA"] = alpha
                with self.subTest(threshold=threshold, alpha=alpha), policy(**settings) as module:
                    edge = math.exp(.01 * threshold)
                    slope = .01 * edge if alpha is None else alpha
                    self.assertAlmostEqual(module.wait_priority_term(None, 0), edge - slope * threshold)
                    self.assertAlmostEqual(module.wait_priority_term(None, threshold), edge)
                    self.assertAlmostEqual(module.wait_priority_term(None, threshold - 1e-6), edge - slope * 1e-6)
                    self.assertAlmostEqual(module.wait_priority_term(None, threshold + 1e-6), math.exp(.01 * (threshold + 1e-6)))

    def test_zero_threshold_disables_piecewise_even_with_alpha(self):
        with policy(LINEAR_UNTIL_S=0, LINEAR_ALPHA=.5) as module:
            self.assertEqual(module.wait_priority_term(None, 0), 1)
            self.assertAlmostEqual(module.wait_priority_term(None, 10), math.exp(.1))

    def test_zero_lambda_and_negative_elapsed(self):
        with policy(LAMBDA=0, LINEAR_UNTIL_S=30, LINEAR_ALPHA=.02) as module:
            self.assertEqual(module.wait_priority_term(None, 300), 1)
        with policy() as module:
            self.assertEqual(module.wait_priority_term(None, -10), 1)

    def test_invalid_settings_and_nonfinite_elapsed_are_rejected(self):
        for setting in ("LAMBDA", "LINEAR_UNTIL_S", "LINEAR_ALPHA"):
            for value in (-1, "nan", "inf"):
                with self.subTest(setting=setting, value=value), self.assertRaises(ValueError):
                    with policy(**{setting: value}):
                        pass
        with policy() as module:
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    module.wait_priority_term(None, value)


if __name__ == "__main__":
    unittest.main()
