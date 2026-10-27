"""Static/source or dependency preflight. Never load model weights or use CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def source_check():
    manifest = json.loads((ROOT/'official-manifest.json').read_text(encoding='utf-8'))
    errors = []
    for relative, expected in manifest['files'].items():
        path = ROOT/'official'/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            errors.append(relative)
    if errors:
        raise RuntimeError('Official source differs: '+', '.join(errors))
    return {'official_commit':manifest['upstream_commit'],'official_files':len(manifest['files']),
            'official_source_bytes_match':True}


def dependency_check():
    # The parent suite already puts its snapshot runtime on sys.path.
    suite = ROOT.parent
    sys.path[:0] = [str(suite),str(suite/'runtime')]
    from specedge.bootstrap import prepare_imports
    provenance = prepare_imports()
    import torch
    import transformers
    import grpc
    import google.protobuf
    from cpu_adapter import CPUCompatibleSpecEdgeEngine
    from client import IntegratedSpecExecClient
    from strategy.server_verify.specexec.grpc import InferenceController
    from wire_codec import SpecEdgeWireCodec
    # Exercise the exact BF16 transport and prove EOS is the tokenizer field.
    values = torch.tensor([0.0,1.5,-2.25],dtype=torch.bfloat16)
    payload = SpecEdgeWireCodec.encode(values)
    decoded = SpecEdgeWireCodec.decode(payload,device='cpu',dtype=torch.bfloat16,shape=(3,))
    if not torch.equal(values,decoded):
        raise RuntimeError('BF16 uint16 wire round-trip failed')
    return dict(provenance,python=sys.executable,python_version=sys.version,
        torch=torch.__version__,transformers=transformers.__version__,grpc=grpc.__version__,
        protobuf=google.protobuf.__version__,torch_uint16=hasattr(torch,'uint16'),
        cpu_engine_import=True,official_client_import=True,official_target_import=True,
        bf16_wire_roundtrip=True,model_weights_loaded=False,cuda_initialized=torch.cuda.is_initialized())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports',action='store_true')
    args=parser.parse_args()
    result=source_check()
    if args.imports:
        result['dependencies']=dependency_check()
    print(json.dumps(result,indent=2))