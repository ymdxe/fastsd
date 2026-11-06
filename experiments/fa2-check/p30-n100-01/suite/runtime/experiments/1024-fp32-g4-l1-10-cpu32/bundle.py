"""Finalize hashes and pack code/data only; never pack model weights or results."""
import hashlib
import json
from pathlib import Path
import tarfile
from common import ROOT,write_json

if not json.loads((ROOT/'validation.json').read_text())['ok']:raise RuntimeError('Validate first')
paths=[]
for name in ('source','specedge','workloads'):
    paths += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
paths += list(ROOT.glob('*.py'))+[ROOT/'matrix.json',ROOT/'manifest.json',ROOT/'validation.json',ROOT/'README.zh.md']
paths += [ROOT/'node1-worker-map.json',ROOT/'node3-worker-map.json']+list((ROOT/'preflight').glob('*.json'))
paths=sorted(set(paths))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='manifest.json'}
manifest=json.loads((ROOT/'manifest.json').read_text());manifest['hashes']=hashes
write_json(ROOT/'manifest.json',manifest)
with tarfile.open(ROOT/'bundle.tar.gz','w:gz') as archive:
    for p in paths:archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
write_json(ROOT/'bundle.json',{'sha256':hashlib.sha256((ROOT/'bundle.tar.gz').read_bytes()).hexdigest(),'bytes':(ROOT/'bundle.tar.gz').stat().st_size,'files':len(paths)})
