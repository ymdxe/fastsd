"""Run required CPU checks in the task overlay before touching the old queue."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

root = Path(sys.argv[1]).resolve()
output = root.parent
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
sys.path[:0] = [str(root), str(root/'runtime'), str(root/'runtime/tests')]
status = {'state': 'running', 'started_epoch_s': time.time()}

def save():
    (output/'cpu-status.json').write_text(json.dumps(status, indent=2))

save()
try:
    import torch
    torch.set_num_threads(1)
    from promotion import content_digest
    from suite import verify_bundle
    status.update(bundle_id=verify_bundle(), validated_content_sha256=content_digest(root))
    for path in list(root.glob('*.py')) + list((root/'runtime/src').glob('*.py')):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    names = ['test_resident_varlen', 'test_probability_window', 'test_varlen_capability',
             'test_cache_offload_policy', 'test_round2_cache', 'test_scheduler_runtime',
             'test_review_engine', 'test_runtime_defaults', 'test_ar_contract', 'test_config1_contract', 'test_diagnostics_contract']
    tests = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    status.update(state='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  skipped=[(str(test), reason) for test, reason in result.skipped],
                  torch=torch.__version__, finished_epoch_s=time.time())
    save()
    if not result.wasSuccessful():
        raise SystemExit(1)
except BaseException as exc:
    if status.get('state') != 'failed':
        status.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        save()
    raise
