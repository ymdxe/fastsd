"""Adapters that make the pinned Official SpecEdge baseline replayable.

The modules in this package deliberately do not import the ``official``
submodule at import time.  This keeps trace and provenance tooling usable on
the edge host before the heavyweight Official SpecEdge environment is ready.
"""

from .poisson_client import (
    AdapterConfigurationError,
    ReplayContext,
    TraceFormatError,
    load_trace,
    replay_trace,
)

__all__ = [
    "AdapterConfigurationError",
    "ReplayContext",
    "TraceFormatError",
    "load_trace",
    "replay_trace",
]
