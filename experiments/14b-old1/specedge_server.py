"""Run the pinned official SpecEdge target with suite lifecycle instrumentation.

The upstream model, batching and CUDA Graph engine remain the execution core.
The integration verifier accepts gamma 2/4/8 chains and excludes graph padding.
"""
import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def _write_state(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _configure_logging(log, directory, name):
    configuration = log.get_default_log_config(directory, name)
    configuration["handlers"]["file"]["level"] = "INFO"
    configuration["loggers"]["root"]["level"] = "INFO"
    log.configure_logging(configuration)


def _runtime_dataset_hook(util):
    original = util.load_dataset

    def runtime_dataset(name, model_name=None, reasoning=False):
        # With cache_prefill=False the official target uses Validate.prefix.
        if name == "unused_at_runtime":
            return []
        return original(name, model_name=model_name, reasoning=reasoning)

    util.load_dataset = runtime_dataset


def _cuda_memory(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def inference_entry(num_clients, recv_queue, resp_queue):
    """Publish readiness only after weights, graph capture and KV allocation."""
    output = Path(os.environ["FASTSD_SPECEDGE_OUTPUT"])
    torch = None
    provenance = {}
    device = None
    try:
        from campaign import wait_gpu_memory
        experiment_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        wait_gpu_memory(experiment_config, output, "specedge")
        from specedge.bootstrap import prepare_imports
        provenance = prepare_imports()
        from config import SpecEdgeBatchServerConfig as config
        import log
        import torch
        import transformers
        import util
        from single_chain_server import SingleChainInferenceController

        _runtime_dataset_hook(util)
        _configure_logging(log, output / "specedge-raw", "server")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = config.device
        if device.type != "cuda" or config.dtype != torch.bfloat16:
            raise ValueError("This target requires CUDA and BF16")
        fraction = float(os.environ["FASTSD_SPECEDGE_GPU_MEMORY_FRACTION"])
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        util.set_seed(config.seed)
        started = time.monotonic()
        controller = SingleChainInferenceController(num_clients, recv_queue, resp_queue)
        from measurements import observe_gpu_call
        controller._engine.forward=observe_gpu_call(controller._engine.forward,"specedge_decode",device)
        controller._engine.prefill=observe_gpu_call(controller._engine.prefill,"specedge_prefill",device)
        torch.cuda.synchronize(device)
        parameters = list(controller._model.parameters())
        dtypes = sorted({str(parameter.dtype) for parameter in parameters})
        devices = sorted({str(parameter.device) for parameter in parameters})
        if dtypes != ["torch.bfloat16"] or any(not value.startswith("cuda:") for value in devices):
            raise RuntimeError("Actual Target dtype/device differs from CUDA BF16")
        metadata = {
            **provenance,
            "pid": os.getpid(), "ready_epoch_s": time.time(),
            "load_and_graph_s": time.monotonic() - started,
            "python": sys.executable, "python_version": sys.version,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "model": config.target_model, "dtype": str(config.dtype),
            "device": str(device), "parameter_dtypes": dtypes,
            "parameter_devices": devices,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "max_len": config.max_len, "tree_budget": config.max_budget,
            "single_chain": controller.single_chain_id,
            "graph_input_width": controller._max_n_beams,
            "client_gammas": controller._client_gammas,
            "logical_wire_inputs": [3, 5, 9],
            "padding_policy": "Distinct scratch KV slots; excluded from responses and retained KV",
            "num_clients": num_clients, "max_batch_size": config.max_batch_size,
            "gpu_memory_fraction": fraction,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "kv_policy": "Official GPU resident per-client cache plus BatchGraphEngine working cache",
            "cuda_graph": "official BatchGraphEngine enabled",
            "log_level": "INFO; structured cycle results retained",
            "cuda_memory": _cuda_memory(torch, device),
        }
        _write_state(output / "specedge-ready.json", metadata)
        controller.loop()
    except BaseException as exc:
        metadata = {
            **provenance, "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "epoch_s": time.time(),
        }
        if torch is not None and device is not None and torch.cuda.is_initialized():
            try:
                metadata["cuda_memory"] = _cuda_memory(torch, device)
            except Exception:
                pass
        _write_state(output / "specedge-failed.json", metadata)
        raise
    finally:
        metadata = {"pid": os.getpid(), "epoch_s": time.time()}
        if torch is not None and device is not None and torch.cuda.is_initialized():
            try:
                metadata["cuda_memory"] = _cuda_memory(torch, device)
            except Exception:
                pass
        from diagnostic_trace import trace
        trace.close()
        _write_state(output / "specedge-inference-exit.json", metadata)
        try:
            from specedge.bootstrap import stop_legacy_logging
            stop_legacy_logging()
        except Exception:
            pass


def configure_environment(config):
    output = Path(config["output_dir"])
    os.environ["FASTSD_DIAGNOSTIC_DIR"]=str(output/"specedge-diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    expected = config.get("expected_workers", [])
    num_clients = int(config.get("num_clients", len(expected) or 16))
    max_batch_size = int(config.get("batch_size", 16))
    if num_clients != 32 or not 1 <= max_batch_size <= 16:
        raise ValueError("This experiment requires 32 clients and 1..16 batch rows")
    if expected and (len(expected) != 32 or len(set(expected)) != 32):
        raise ValueError("Expected worker identities must contain 32 unique entries")
    if int(config.get("tree_budget", 8)) != 8:
        raise ValueError("This experiment fixes the maximum single-chain budget at 8")
    if not expected:
        raise ValueError("Single-chain target requires explicit expected worker identities")
    client_gammas = [int(config["nodes"][identity.rsplit("-w", 1)[0]]["gamma"])
                     for identity in expected]
    if client_gammas.count(4) != 16 or client_gammas.count(8) != 16:
        raise ValueError("Config 1 requires sixteen gamma8 and sixteen gamma4 clients")
    fraction = float(config.get("gpu_memory_fraction", 0.60))
    timeout = float(config.get("request_timeout", 3600))
    max_len = int(config.get("specedge_max_len", config.get("max_len", 1024)))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid GPU memory fraction")
    if not math.isfinite(timeout) or timeout <= 0 or max_len <= 64:
        raise ValueError("Invalid request timeout or context limit")
    for name in ("specedge-ready.json", "specedge-failed.json", "specedge-wrapper.json"):
        if (output / name).exists():
            raise FileExistsError("SpecEdge results already exist; use a new run ID: " + name)
    env = {
        "FASTSD_SPECEDGE_OUTPUT": str(output),
        "FASTSD_SPECEDGE_GPU_MEMORY_FRACTION": fraction,
        "FASTSD_SPECEDGE_CLIENT_GAMMAS": json.dumps(client_gammas),
        "SPECEDGE_RESULT_PATH": str(output / "specedge-raw"),
        "SPECEDGE_EXP_NAME": str(config.get("run_id", "run")),
        "SPECEDGE_PROCESS_NAME": "server",
        "SPECEDGE_SEED": int(config.get("seed", 1234)),
        "SPECEDGE_MAX_LEN": max_len,
        "SPECEDGE_BATCH_TYPE": "dynamic", "SPECEDGE_DATASET": "unused_at_runtime",
        "SPECEDGE_SAMPLE_REQ_CNT": 1, "SPECEDGE_REQ_OFFSET": 0,
        "SPECEDGE_TARGET_MODEL": config["model"],
        "SPECEDGE_SERVER_DEVICE": str(config.get("target_device", "cuda:0")),
        "SPECEDGE_DTYPE": "bf16", "SPECEDGE_TEMPERATURE": 0,
        "SPECEDGE_MAX_BATCH_SIZE": max_batch_size,
        "SPECEDGE_MAX_N_BEAMS": 1, "SPECEDGE_MAX_BUDGET": 8,
        "SPECEDGE_NUM_CLIENTS": num_clients, "SPECEDGE_CACHE_PREFILL": "False",
    }
    os.environ.update({key: str(value) for key, value in env.items()})
    return output


async def serve(config, host="127.0.0.1", port=19636):
    output = configure_environment(config)
    from specedge.bootstrap import prepare_imports
    provenance = prepare_imports()
    import grpc.aio
    import log
    import util
    from config import SpecEdgeBatchServerConfig as target_config
    from strategy.server_verify.specexec import grpc as official
    from specedge_grpc import specedge_pb2, specedge_pb2_grpc

    _runtime_dataset_hook(util)
    official._init_inference = inference_entry
    _configure_logging(log, output / "specedge-raw", "server-controller")
    util.set_seed(target_config.seed)
    shutdown_event = asyncio.Event()
    timeout = float(config.get("request_timeout", 3600))

    class SuiteServicer(official.SpecExecBatchServer):
        async def Validate(self, request, context):
            from measurements import trace
            arrived=time.monotonic_ns()
            trace.emit('grpc_cloud_receive',client_idx=request.client_idx,req_idx=request.req_idx)
            if not 0 <= request.client_idx < int(target_config.num_clients):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "client_idx outside configured range")
            future = asyncio.get_running_loop().create_future()
            with self._resp_lock:
                previous = self._resp_futures.get(request.client_idx)
                duplicate = previous is not None and not previous.done()
                if not duplicate:
                    self._resp_futures[request.client_idx] = future
            if duplicate:
                await context.abort(grpc.StatusCode.ALREADY_EXISTS, "client already has an in-flight verify")
            self._recv_queue.put(request.SerializeToString())
            try:
                selection, prefill_count = await asyncio.wait_for(future, timeout=timeout)
                return specedge_pb2.ValidateResponse(selection=selection, prefill=prefill_count)
            except asyncio.TimeoutError:
                await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "SpecEdge verification timeout")
            finally:
                trace.emit('grpc_cloud_complete',client_idx=request.client_idx,req_idx=request.req_idx,
                    start_ns=arrived,end_ns=time.monotonic_ns())
                with self._resp_lock:
                    if self._resp_futures.get(request.client_idx) is future:
                        self._resp_futures.pop(request.client_idx, None)

    controller = None
    server = None
    try:
        controller = SuiteServicer(shutdown_event=shutdown_event)
        server = grpc.aio.server(options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ])
        specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(controller, server)
        if server.add_insecure_port(f"{host}:{int(port)}") == 0:
            raise RuntimeError("SpecEdge gRPC bind failed")
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, shutdown_event.set)
        _write_state(output / "specedge-wrapper.json", {
            **provenance, "config": config, "host": host, "port": port,
            "pid": os.getpid(), "inference_pid": controller._inference_process.pid,
            "request_timeout": timeout,
        })
        await server.start()
        print(f"SpecEdge gRPC listening on {host}:{port}; await specedge-ready.json", flush=True)
        while not shutdown_event.is_set():
            if not controller._inference_process.is_alive():
                raise RuntimeError("Official SpecEdge inference process exited; inspect specedge-failed.json")
            await asyncio.sleep(0.5)
    except BaseException as exc:
        _write_state(output / "specedge-wrapper-failed.json", {
            "pid": os.getpid(), "error": repr(exc),
            "traceback": traceback.format_exc(), "epoch_s": time.time(),
        })
        raise
    finally:
        if server is not None:
            await server.stop(grace=2)
        shutdown_event.set()
        if controller is not None:
            # The official cleanup addresses only its own mp.Process and queues.
            await controller.cleanup()
        from specedge.bootstrap import stop_legacy_logging
        stop_legacy_logging()


def run(config, host="127.0.0.1", port=19636):
    mp.set_start_method("spawn", force=True)
    return asyncio.run(serve(config, host, port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19636)
    options = parser.parse_args()
    run(load_config(options.config), options.host, options.port)
