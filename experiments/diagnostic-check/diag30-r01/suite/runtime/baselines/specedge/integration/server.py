"""Launch the official SpecEdge server on a configurable loopback port."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parent
SPECEDGE_ROOT = INTEGRATION_ROOT.parent / "official"
sys.path.insert(0, str(SPECEDGE_ROOT / "src"))
sys.path.insert(0, str(SPECEDGE_ROOT / "src" / "script"))

import batch_server as official_server


async def serve(host: str, port: int) -> None:
    official_server.shutdown_event = asyncio.Event()
    controller = official_server.SpecExecBatchServer(
        shutdown_event=official_server.shutdown_event
    )
    server = official_server.grpc.aio.server()
    official_server.specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(
        controller, server
    )
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind SpecEdge gRPC server to {host}:{port}")
    try:
        await server.start()
        await official_server.shutdown_event.wait()
        await server.stop(grace=2.0)
        await controller.cleanup()
    except asyncio.CancelledError:
        await server.stop(0)
        raise
    except Exception:
        await server.stop(0)
        raise


def main(config_path: str, host: str, port: int) -> int:
    official_server._load_config(Path(config_path))
    official_server.util.set_seed(official_server.config.seed)
    signal.signal(signal.SIGINT, official_server.signal_handler)
    signal.signal(signal.SIGTERM, official_server.signal_handler)
    asyncio.run(serve(host, port))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.host, args.port))
