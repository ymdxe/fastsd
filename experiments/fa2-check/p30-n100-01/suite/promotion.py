"""Content-bound gate: changes after validation require a fresh validation."""
import hashlib
import json
from pathlib import Path

GATES = ('dependency_abi', 'cpu_contracts', 'gpu_attention_cache',
         'humaneval164_exact_greedy', 'performance')


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
    certificate = json.loads((root / 'validation_certificate.json').read_text())
    if certificate.get('passed') is not True or any(certificate.get('gates', {}).get(g) is not True for g in GATES):
        raise RuntimeError('Formal launch blocked: validation gates have not all passed')
    if certificate.get('validated_content_sha256') != content_digest(root):
        raise RuntimeError('Formal launch blocked: package changed after validation')
    return certificate
