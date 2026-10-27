"""Lifecycle wrapper around the pinned official SpecEdge gRPC target.

The official InferenceController, model, CUDA Graph engine, tree verification
and cache operations are used directly. Only readiness/error reporting and the
request wait timeout are adapted for the suite's long CPU requests.
"""
import argparse
import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def load_config(path):
    text = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix.lower() == '.json':
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def _write_state(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def inference_entry(num_clients, recv_queue, resp_queue):
    """Same official target construction with an explicit post-load ready mark."""
    from specedge.bootstrap import prepare_imports
    provenance = prepare_imports()
    from config import SpecEdgeBatchServerConfig as config
    import log
    import torch
    import transformers
    import util
    original_load_dataset = util.load_dataset
    def runtime_dataset(name, model_name=None, reasoning=False):
        if name == 'unused_at_runtime':
            return []
        return original_load_dataset(name, model_name=model_name, reasoning=reasoning)
    util.load_dataset = runtime_dataset
    from strategy.server_verify.specexec.grpc import InferenceController
    output = Path(os.environ['FASTSD_SPECEDGE_OUTPUT'])
    log.configure_logging(log.get_default_log_config(output/'specedge-raw', 'server'))
    try:
        controller = InferenceController(num_clients, recv_queue, resp_queue)
        torch.cuda.synchronize(config.device)
        _write_state(output/'specedge-ready.json', dict(
            **provenance, pid=os.getpid(), ready_epoch_s=time.time(),
            python=sys.executable, torch=torch.__version__, transformers=transformers.__version__,
            model=config.target_model, dtype=str(config.dtype), device=str(config.device),
            max_len=config.max_len, tree_budget=config.max_budget,
            num_clients=num_clients, max_batch_size=config.max_batch_size,
            kv_policy='Official GPU resident per-client cache plus BatchGraphEngine working cache',
            cuda_graph='official BatchGraphEngine enabled'))
        controller.loop()
    except BaseException as exc:
        _write_state(output/'specedge-failed.json', dict(pid=os.getpid(), error=f'{type(exc).__name__}: {exc}',
            traceback=traceback.format_exc(), epoch_s=time.time()))
        raise
    finally:
        _write_state(output/'specedge-inference-exit.json',dict(pid=os.getpid(),epoch_s=time.time()))


def configure_environment(config):
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    num_clients = int(config.get('num_clients', len(config.get('expected_workers', [])) or 16))
    max_batch_size = int(config.get('batch_size', 16))
    if not 1 <= max_batch_size <= 16 or not 1 <= num_clients <= 16:
        raise ValueError('SpecEdge comparison requires 1..16 clients/batch rows')
    env = {
        'FASTSD_SPECEDGE_OUTPUT': str(output),
        'SPECEDGE_RESULT_PATH': str(output/'specedge-raw'),
        'SPECEDGE_EXP_NAME': str(config.get('run_id','run')),
        'SPECEDGE_PROCESS_NAME': 'server', 'SPECEDGE_SEED': 1234,
        'SPECEDGE_MAX_LEN': int(config.get('specedge_max_len',config.get('max_len',2048))),
        'SPECEDGE_BATCH_TYPE': 'dynamic', 'SPECEDGE_DATASET': 'unused_at_runtime',
        'SPECEDGE_SAMPLE_REQ_CNT': 1, 'SPECEDGE_REQ_OFFSET': 0,
        'SPECEDGE_TARGET_MODEL': config['model'],
        'SPECEDGE_SERVER_DEVICE': str(config.get('target_device','cuda:0')),
        'SPECEDGE_DTYPE': 'bf16', 'SPECEDGE_TEMPERATURE': 0,
        'SPECEDGE_MAX_BATCH_SIZE': max_batch_size,
        'SPECEDGE_MAX_N_BEAMS': 32, 'SPECEDGE_MAX_BUDGET': int(config.get('tree_budget',32)),
        'SPECEDGE_NUM_CLIENTS': num_clients, 'SPECEDGE_CACHE_PREFILL': 'False',
    }
    if int(env['SPECEDGE_MAX_BUDGET']) != 32:
        raise ValueError('This comparison fixes tree_budget=32')
    os.environ.update({key:str(value) for key,value in env.items()})
    return output


async def serve(config, host='127.0.0.1', port=19636):
    output = configure_environment(config)
    from specedge.bootstrap import prepare_imports
    provenance = prepare_imports()
    import grpc.aio
    import log
    import util
    from config import SpecEdgeBatchServerConfig as target_config
    from strategy.server_verify.specexec import grpc as official
    from specedge_grpc import specedge_pb2, specedge_pb2_grpc
    # Runtime prompts arrive in ValidateRequest.prefix. The only official
    # dataset read during controller construction is unused when cache_prefill
    # is false; avoid a fake on-disk dataset or precomputed prompt cache.
    original_load_dataset = util.load_dataset
    def runtime_dataset(name, model_name=None, reasoning=False):
        if name == 'unused_at_runtime':
            return []
        return original_load_dataset(name, model_name=model_name, reasoning=reasoning)
    util.load_dataset = runtime_dataset
    # Spawned children reconstruct imports, so identify the no-cache loader
    # through a top-level initialization hook in inference_entry as well.
    official._init_inference = inference_entry
    log.configure_logging(log.get_default_log_config(output/'specedge-raw','server-controller'))
    util.set_seed(target_config.seed)
    shutdown_event = asyncio.Event()
    class SuiteServicer(official.SpecExecBatchServer):
        async def Validate(self, request, context):
            if not 0 <= request.client_idx < int(target_config.num_clients):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'client_idx outside configured range')
            fut = asyncio.get_running_loop().create_future()
            with self._resp_lock:
                previous = self._resp_futures.get(request.client_idx)
                if previous is not None and not previous.done():
                    await context.abort(grpc.StatusCode.ALREADY_EXISTS,'client already has an in-flight verify')
                self._resp_futures[request.client_idx] = fut
            self._recv_queue.put(request.SerializeToString())
            try:
                selection, prefill_cnt = await asyncio.wait_for(fut,timeout=float(config.get('request_timeout',1800)))
                return specedge_pb2.ValidateResponse(selection=selection,prefill=prefill_cnt)
            finally:
                with self._resp_lock:
                    if self._resp_futures.get(request.client_idx) is fut:
                        self._resp_futures.pop(request.client_idx,None)
    controller = SuiteServicer(shutdown_event=shutdown_event)
    server = grpc.aio.server(options=[('grpc.max_receive_message_length',64*1024*1024),
                                      ('grpc.max_send_message_length',64*1024*1024)])
    specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(controller,server)
    if server.add_insecure_port(f'{host}:{int(port)}') == 0:
        raise RuntimeError('SpecEdge gRPC bind failed')
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(signum,shutdown_event.set)
    _write_state(output/'specedge-wrapper.json',dict(**provenance,config=config,host=host,port=port,pid=os.getpid()))
    try:
        await server.start()
        print(f'SpecEdge gRPC listening on {host}:{port}; readiness requires specedge-ready.json',flush=True)
        while not shutdown_event.is_set():
            if not controller._inference_process.is_alive():
                raise RuntimeError('Official SpecEdge inference process exited; inspect specedge-failed.json')
            await asyncio.sleep(0.5)
    finally:
        await server.stop(grace=2)
        shutdown_event.set()
        await controller.cleanup()


def run(config, host='127.0.0.1', port=19636):
    mp.set_start_method('spawn',force=True)
    return asyncio.run(serve(config,host,port))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=19636)
    args=parser.parse_args()
    run(load_config(args.config),args.host,args.port)