"""Vendored official SpecEdge with suite adapters in separate modules."""
from pathlib import Path
# Keep the original specedge.client/engine/tree imports resolving to official code.
__path__.append(str(Path(__file__).resolve().parent/'official'/'src'/'specedge'))