"""Content-bound gate for this isolated observation campaign only.

This certificate is not a formal output-equivalence certificate. The old
formal campaign and FA2 gates are deliberately kept in their original folders.
"""
import hashlib
import json
from pathlib import Path

GATES = ('cpu_contracts', 'trace_contracts', 'observer_contracts',
         'gpu_event_transfer_probe', 'real_model_prefetch_overlap')


def content_digest(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(p in ('__pycache__', '.git') for p in path.parts):
            continue
        if path.suffix in ('.pyc', '.pyo') or (path.parent == root and path.name in
                ('manifest.json', 'validation_certificate.json')):
            continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_promotion(root):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    if config.get('diagnostic_only') is not True or config.get('target_attention_backend') != 'sdpa_per_sequence':
        raise RuntimeError('Diagnostic gate is restricted to this SDPA observation campaign')
    certificate = json.loads((root / 'validation_certificate.json').read_text())
    if certificate.get('scope') != 'paper_wrr_prefetch_repair' or certificate.get('formal_output_equivalence') is not False:
        raise RuntimeError('Diagnostic certificate must explicitly disclaim formal output equivalence')
    if certificate.get('passed') is not True or any(certificate.get('gates', {}).get(g) is not True for g in GATES):
        raise RuntimeError('Diagnostic launch blocked: observation gates have not all passed')
    if certificate.get('validated_content_sha256') != content_digest(root):
        raise RuntimeError('Diagnostic launch blocked: package changed after validation')
    return certificate
