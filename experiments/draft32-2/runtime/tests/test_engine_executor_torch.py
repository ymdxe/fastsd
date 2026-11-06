"""Torch-context tests for the production executor wrappers.

These tests exercise the exact helpers the engine uses
(``make_forward_batch_runner`` / ``make_forward_job_runner`` and
``set_thread_device``) on a real CPU Linear layer and, when available, a real
CUDA device context.  No checkpoints or large models are loaded.
"""

import time
import types
import unittest
from collections import defaultdict

try:
    import torch
except ImportError:  # pragma: no cover - lightweight environments
    torch = None

from src.scheduler_runtime import (
    CacheResidency,
    ExecutionOutcome,
    KVPrefetchWorker,
    SinglePlanExecutor,
    make_forward_batch_runner,
    make_forward_job_runner,
    set_thread_device,
)


@unittest.skipIf(torch is None, "torch is required")
class ExecutorTorchContextTests(unittest.TestCase):
    def test_production_batch_runner_disables_grad_on_executor_thread(self):
        observed = {}
        model = torch.nn.Linear(2, 2)

        def handle_batch(batch):
            observed["grad_enabled"] = torch.is_grad_enabled()
            output = model(torch.ones(1, 2))
            observed["requires_grad"] = output.requires_grad
            return {"V": {"accepted": 4, "final_token": 0}}

        run_job = make_forward_job_runner(make_forward_batch_runner(handle_batch))
        executor = SinglePlanExecutor(
            run_job,
            thread_setup=lambda: set_thread_device(torch.device("cpu")),
        )
        executor.start()
        try:
            executor.submit(types.SimpleNamespace(verify_batch=[{}], prefill_batch=[]))
            stage = executor.wait_result(2.0)
            self.assertIsNotNone(stage)
            self.assertIsNone(stage.outcome.error)
            self.assertFalse(stage.final is False)
        finally:
            executor.stop()
            executor.join(2.0)

        self.assertFalse(observed["grad_enabled"])
        self.assertFalse(observed["requires_grad"])
        self.assertFalse(executor._thread.is_alive())

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA is required"
    )
    def test_executor_thread_binds_cuda_current_device(self):
        observed = {}

        def run_job(job):
            observed["device"] = torch.cuda.current_device()
            yield ExecutionOutcome(job=job)

        executor = SinglePlanExecutor(
            run_job,
            thread_setup=lambda: set_thread_device(torch.device("cuda:0")),
        )
        executor.start()
        try:
            executor.submit(object())
            stage = executor.wait_result(5.0)
            self.assertIsNotNone(stage)
            self.assertIsNone(stage.outcome.error)
        finally:
            executor.stop()
            executor.join(2.0)

        self.assertEqual(observed["device"], 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA is required"
    )
    def test_prefetch_thread_binds_cuda_current_device(self):
        observed = {}
        metrics = defaultdict(int)
        metrics["prefetch_async_last_error"] = None

        def setup_thread():
            set_thread_device(torch.device("cuda:0"))
            observed["device"] = torch.cuda.current_device()

        worker = KVPrefetchWorker(
            residency=CacheResidency(),
            get_cache=lambda pid: None,
            set_cache=lambda pid, cache: None,
            move_cache=lambda cache, device: cache,
            gpu_device=torch.device("cuda:0"),
            metrics=metrics,
            thread_setup=setup_thread,
        )
        worker.start()
        try:
            worker.submit(set())
            deadline = time.monotonic() + 3.0
            while metrics["prefetch_async_completed"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            worker.stop()
            worker.join(2.0)

        self.assertEqual(observed["device"], 0)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
