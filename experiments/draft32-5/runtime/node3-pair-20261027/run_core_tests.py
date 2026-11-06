"""Run relevant existing model-free/miniature-model regression tests on CPU."""
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT/'runtime'))
patterns=['test_fastsd_scheduler.py','test_request_validation.py',
          'test_scheduler_runtime.py','test_edge_entrypoint.py','test_cloud_ipc.py',
          'test_cloud_shutdown_gate.py','test_metrics.py','test_arrival.py',
          'test_engine_executor_torch.py','test_speculative_sampling.py',
          'test_kvcache_batching.py','test_kvcache_varlen.py','test_verify_accept.py',
          'test_cache_offload_policy.py','test_runtime_defaults.py']
suite=unittest.TestSuite()
for pattern in patterns:
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'runtime/tests'),pattern=pattern))
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
