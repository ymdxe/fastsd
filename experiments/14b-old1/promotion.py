"""Require fresh GPU functional evidence for this exact small-experiment bundle."""
import json
from pathlib import Path


def require_promotion(root):
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    certificate = json.loads((root/'readiness.json').read_text())
    if (certificate['bundle_id'] != manifest['bundle_id']
            or certificate['source_sha'] != config['source_sha']
            or certificate['formal_output_equivalence'] is not False
            or certificate['passed'] is not True
            or certificate['cuda_tests']['run'] != 3
            or certificate['cuda_tests']['skipped'] != 0
            or certificate['real_model']['exact_probability_and_kv_match'] is not True):
        raise RuntimeError('Fresh GPU functional checks have not passed for this bundle')
    return certificate
