"""One CPU SpecEdge single-chain client with experiment-local tracing.

The official SpecExec/proactive/tree implementations remain the inference core.
This file supplies a non-graph CPU engine, FP32 protobuf transport, exact output
limits, input-template checks, and the shared experiment rendezvous.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
sys.path.insert(0,str(EXPERIMENT))
sys.path.insert(0, str(ROOT))

from bootstrap import install

install()

import grpc
import grpc.aio
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from model.cache import KVCache
import specedge.client.specexec as official_client
from specedge_grpc import specedge_pb2, specedge_pb2_grpc


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_json(path, payload):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def bind_cpu(cpu, numa):
    os.sched_setaffinity(0, set(cpu if isinstance(cpu,list) else [int(cpu)]))
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    lib.numa_allocate_nodemask.restype = ctypes.c_void_p
    lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.numa_bitmask_setbit.restype = ctypes.c_void_p
    lib.numa_set_membind.argtypes = [ctypes.c_void_p]
    lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    mask = lib.numa_allocate_nodemask()
    if not mask:
        raise RuntimeError("Cannot allocate NUMA mask")
    try:
        lib.numa_bitmask_setbit(mask, int(numa))
        lib.numa_set_membind(mask)
    finally:
        lib.numa_bitmask_free(mask)


class Rendezvous:
    def __init__(self, args):
        self.url = args.control_url.rstrip("/")
        self.token = args.token_file.read_text().strip()
        self.query = {"run": args.run, "worker": f"{args.node}-{args.worker}", "expected": args.expected}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, endpoint, **extra):
        query = dict(self.query, **extra)
        req = urllib.request.Request(self.url + "/" + endpoint + "?" + urllib.parse.urlencode(query), headers={"Authorization": "Bearer " + self.token})
        with self.opener.open(req, timeout=30) as response:
            return json.loads(response.read())

    def barrier(self, endpoint, timeout=7200):
        start = time.monotonic()
        while not self.call(endpoint).get("release"):
            if time.monotonic() - start > timeout:
                raise TimeoutError(endpoint + " rendezvous timed out")
            time.sleep(.15)


class CPUEngine:
    """The official GraphEngine contract using normal CPU forwards."""
    def __init__(self, model, max_len):
        self._model = model
        self._device = model.device
        self._dtype = model.dtype
        self.max_len = max_len
        if self._device.type != "cpu":
            raise ValueError("SpecEdge draft must stay on CPU")
        self._past_key_values = KVCache(config=model.config, batch_size=1,
                                      max_n_beams=1, max_len=max_len,
                                      device=self._device, dtype=self._dtype)

    @torch.inference_mode()
    def forward(self, input_ids, position_ids, cache_batch_indices, cache_seq_indices, attention_mask):
        self._past_key_values.seq_indices = torch.arange(input_ids.size(1), device=self._device, dtype=torch.long)
        return self._model.forward(input_ids=input_ids, position_ids=position_ids,
                                  cache_batch_indices=cache_batch_indices,
                                  cache_seq_indices=cache_seq_indices,
                                  attention_mask=util.invert_mask(attention_mask),
                                  past_key_values=self._past_key_values)[0]

    @torch.inference_mode()
    def prefill(self, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask):
        indices = torch.zeros(input_ids.size(-1), dtype=torch.long, device=self._device)
        with self._past_key_values.prefill_context(input_ids.size(-1), batch_idx):
            self._model.forward(input_ids=input_ids, position_ids=position_ids,
                                cache_batch_indices=indices, cache_seq_indices=cache_seq_indices,
                                attention_mask=util.invert_mask(attention_mask),
                                past_key_values=self._past_key_values)

    def gather(self, source, destination):
        self._past_key_values.gather(0, source, destination)

    def reset(self):
        self._past_key_values.clear()


class CPUTiming:
    def __init__(self, device=None, mode="no-sync", enabled=True):
        if device is not None and str(device).split(":")[0] != "cpu":
            raise ValueError("CPU timing adapter received a CUDA operation")
        self.enabled = enabled
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *unused):
        if self.enabled:
            self.elapsed = (time.perf_counter() - self.start) * 1000


@contextmanager
def cpu_timing():
    original = util.Timing
    util.Timing = CPUTiming
    try:
        yield
    finally:
        util.Timing = original


def encode(tensor, dtype=None):
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu().contiguous().numpy().tobytes()


class TracedSingleChainGrpcClient:
    """Keep official protobuf messages; require an actual unbranched chain."""
    def __init__(self, host, device):
        self.device = device
        self.gamma = int(os.environ["EXP_GAMMA"])
        self.path = Path(os.environ["EXP_RPC_LOG"])
        self.timeout = float(os.environ.get("EXP_REQUEST_TIMEOUT", "7200"))
        self.channel = grpc.aio.insecure_channel(host)
        self.stub = specedge_pb2_grpc.SpecEdgeServiceStub(self.channel)
        self.request_bytes = self.response_bytes = self.rpc_count = 0

    async def request(self, client_idx, req_idx, input_ids, position_ids, cache_seq_indices,
                      attention_mask, parent_indices, prefill=False, prefix=None):
        ids = input_ids.reshape(-1)
        positions = position_ids.reshape(-1)
        cache_indices = cache_seq_indices.reshape(-1)
        parents = parent_indices.reshape(-1)
        if ids.numel() != self.gamma + 1 or parents.numel() != self.gamma:
            raise RuntimeError(f"Expected root+{self.gamma} drafts; got {ids.numel()} inputs/{parents.numel()} parents")
        if not torch.equal(parents, cache_indices[:-1]):
            raise RuntimeError("Verify candidate parent pointers are not a single chain")
        if not torch.all(positions[1:] - positions[:-1] == 1):
            raise RuntimeError("Verify candidate positions are not a consecutive single chain")
        if prefill and not prefix:
            raise ValueError("First Verify needs the identical formatted prompt")
        request = specedge_pb2.ValidateRequest(
            client_idx=client_idx, req_idx=req_idx, input_ids=encode(input_ids),
            position_ids=encode(position_ids), cache_seq_indices=encode(cache_seq_indices),
            parent_indices=encode(parent_indices), attention_mask=encode(attention_mask, torch.float32),
            prefill=prefill, prefix=prefix)
        start = time.monotonic()
        row = {"endpoint": "/verify", "protocol": "grpc", "operation": "Validate",
               "node": os.environ["EXP_NODE"], "client_idx": client_idx, "req_idx": req_idx,
               "gamma": self.gamma, "input_tokens_including_root": int(ids.numel()),
               "candidate_tokens": int(parents.numel()), "chain_depth": int(parents.numel()),
               "max_children_per_parent": 1, "prefill": bool(prefill), "mask_dtype": "float32",
               "started_monotonic": start, "started_epoch": time.time(),
               "request_bytes": request.ByteSize()}
        self.request_bytes += request.ByteSize()
        self.rpc_count += 1
        try:
            response = await self.stub.Validate(request, timeout=self.timeout)
            result = torch.frombuffer(bytearray(response.selection), dtype=torch.long).clone()
            if result.numel() != ids.numel():
                raise RuntimeError(f"Target returned {result.numel()} selections for {ids.numel()} input tokens")
            self.response_bytes += response.ByteSize()
            row.update(ok=True, response_bytes=response.ByteSize())
            return result.to(self.device), response.prefill
        except BaseException as exc:
            row.update(ok=False, error=repr(exc))
            raise
        finally:
            row.update(finished_monotonic=time.monotonic(), rpc_ms=(time.monotonic() - start) * 1000)
            append_json(self.path, row)


# Preserve the official constructor and cycles while replacing its transport.
official_client.GrpcClientController = TracedSingleChainGrpcClient


class SingleChainClient(official_client.SpecExecClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_time = None
        self.first_draft_ms = None
        self.cycles = []

    def _grow_tree(self, prefill):
        start = time.monotonic()
        result = super()._grow_tree(prefill)
        if self.first_draft_ms is None:
            self.first_draft_ms = (time.monotonic() - start) * 1000
        return result

    def _trim_by_budget(self):
        # Official topk(sorted=False) can reorder a chain before Tree.gather.
        # Select exactly the same nodes, restoring their append/topology order.
        source = self._tree.logprobs[self._tree.prefix_len:self._tree.end].topk(k=self._max_budget, sorted=False).indices + self._tree.prefix_len
        source = source.sort().values
        destination = torch.arange(self._tree.prefix_len, self._tree.prefix_len + source.numel(), device=self._device)
        self._tree.gather(source, destination)
        self._engine.gather(source, destination)

    async def generate_exact(self, req_idx):
        cap = int(config.max_new_tokens)
        generated = 0
        step = 0
        while generated < cap:
            start = time.monotonic()
            with cpu_timing():
                fresh = await super()._cycle(req_idx, step, prefill=(step == 0))
            full_count = int(fresh.numel())
            if full_count < 1:
                raise RuntimeError("Official SpecExec cycle returned no verified output")
            accepted = full_count - 1  # Official metric includes its bonus token.
            if not 0 <= accepted <= self._max_budget:
                raise RuntimeError("Invalid single-chain accepted draft count")
            eos_positions = (fresh == self._tokenizer.eos_token_id).nonzero()
            hit_eos = bool(eos_positions.numel())
            if hit_eos:
                fresh = fresh[..., :int(eos_positions[0, -1].item()) + 1]
            fresh = fresh[..., :cap - generated]
            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh], dim=-1)
            generated += int(fresh.numel())
            if self.first_token_time is None and fresh.numel():
                self.first_token_time = time.monotonic()
            cycle = {"req_idx": req_idx, "step": step, "gamma": self._max_budget,
                     "accepted_draft_tokens": accepted, "verified_output_tokens_before_cap": full_count,
                     "retained_output_tokens": int(fresh.numel()), "proactive_reused": bool(self._proactive_draft),
                     "cycle_ms": (time.monotonic() - start) * 1000, "hit_eos": hit_eos}
            self.cycles.append(cycle)
            append_json(os.environ["EXP_CYCLE_LOG"], cycle)
            step += 1
            if hit_eos:
                break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=("node1", "node3"), required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--run", default="specedge", type=str)
    parser.add_argument("--expected", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-url", default="http://127.0.0.1:28634")
    parser.add_argument("--token-file", type=Path, default=EXPERIMENT / ".control-token")
    return parser.parse_args()


async def run(args, rendezvous):
    from common import annotate, origin_monotonic
    placement=json.loads((EXPERIMENT/f'{args.node}-worker-map.json').read_text())[args.worker]
    bind_cpu(placement['cpus'],placement['numa'])
    torch.set_num_threads(int(os.environ['SPECEDGE_THREADS']));torch.set_num_interop_threads(1)
    util.set_seed(int(config.seed))
    started=time.monotonic()
    model=util.load_graph_model(config.draft_model,config.device,config.dtype)
    if model.device.type!='cpu' or {p.dtype for p in model.parameters()}!={torch.float32}:raise RuntimeError('Draft device/dtype')
    engine=CPUEngine(model,int(config.max_len))
    tokenizer=util.load_tokenizer(config.draft_model)
    write_json(args.output/f'worker-{args.worker:02d}.json',{'pid':os.getpid(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'dtype':str(model.dtype),'torch':torch.__version__,'affinity':sorted(os.sched_getaffinity(0)),'load_seconds':time.monotonic()-started})
    rendezvous.barrier('model_ready')
    with grpc.insecure_channel(config.host) as channel:
        specedge_pb2_grpc.SpecEdgeServiceStub(channel).Sync(specedge_pb2.SyncRequest(),timeout=1800)
    # NTP-style calibration supplies a shared epoch; it is not used for priority.
    rendezvous.barrier('clock_acquire')
    try:
        probes=[]
        for _ in range(5):
            before=time.time_ns();r=rendezvous.call('health');after=time.time_ns()
            probes.append((after-before,int(r['epoch']*1e9)-(before+after)//2))
        uncertainty,offset=min(probes)
        write_json(args.output/f'clock-{args.worker:02d}.json',{'clock_offset_ns':offset,'clock_uncertainty_ns':uncertainty//2})
    finally:rendezvous.call('clock_release')
    rendezvous.barrier('clock_done')
    origin=origin_monotonic(offset)
    samples=[json.loads(s) for s in (EXPERIMENT/'workloads'/f'lambda-{os.environ["EXP_RATE"]}.jsonl').read_text().splitlines() if s.strip()]
    global_worker=int(os.environ['EXP_GLOBAL_WORKER'])
    for sample in samples:
        if sample['source_index']%32!=global_worker:continue
        scheduled=sample['scheduled_arrival_s']
        remaining=origin+scheduled-time.monotonic()
        if remaining>0:await asyncio.sleep(remaining)
        actual=time.monotonic()-origin
        request_start=time.monotonic()
        messages=[{'role':'user','content':sample['prompt']}]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        expected_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if len(expected_ids)+int(config.max_new_tokens)+2*int(config.max_budget)+4>int(config.max_len):raise ValueError('Tree capacity')
        client=SingleChainClient(engine=engine,tokenizer=tokenizer,prompt=prompt,max_len=int(config.max_len))
        if client._prefix_tokens.reshape(-1).tolist()!=expected_ids:raise RuntimeError('Prompt parity')
        try:
            await client.generate_exact(int(sample['source_index']))
            completed=time.monotonic()
            ids=client._prefix_tokens[0,client._num_original_tokens:].tolist();tokens=len(ids)
            if not tokens or tokens>int(config.max_new_tokens) or client.first_token_time is None:raise RuntimeError('Output cap/TTFT')
            e2e=(completed-request_start)*1000;ttft=(client.first_token_time-request_start)*1000
            accepted=sum(c['accepted_draft_tokens'] for c in client.cycles);drafted=sum(c['gamma'] for c in client.cycles)
            row={'method':'specedge_cpu_adapted_single_chain','task_id':sample['sample_id'],'sample_id':sample['sample_id'],
                 'global_sample_index':sample['source_index'],'proc_id':global_worker,'client_idx':int(config.client_idx),
                 'generated_tokens':tokens,'max_new_tokens':int(config.max_new_tokens),'request_e2e_ms':e2e,'ttft_ms':ttft,
                 'tpot_ms':(e2e-ttft)/(tokens-1) if tokens>1 else None,'first_draft_ms':client.first_draft_ms,
                 'accepted_total':accepted,'drafted_total':drafted,'rounds':len(client.cycles),
                 'accept_rate':accepted/drafted if drafted else None,'reuse_hit_rounds':sum(c['proactive_reused'] for c in client.cycles),
                 'output_token_ids':ids,'output_text':tokenizer.decode(ids,skip_special_tokens=True),'reference':sample['answer'],
                 'request_bytes':client._validator.request_bytes,'response_bytes':client._validator.response_bytes,'rpc_count':client._validator.rpc_count,
                 'scheduled_arrival_s':scheduled,'actual_arrival_s':actual,'arrival_lag_ms':max(0,actual-scheduled)*1000,
                 'completion_s':completed-origin}
            append_json(args.output/f'edge_metrics_proc{global_worker}.jsonl',annotate(row))
        finally:await client._validator.channel.close()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log.configure_logging(log.get_default_log_config(args.output, f"official-client-{args.worker:02d}"))
    rendezvous = Rendezvous(args)
    success = False
    error = None
    try:
        asyncio.run(run(args, rendezvous))
        success = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write_json(args.output / f"completion-{args.worker:02d}.json", {"worker_id": args.worker, "node": args.node, "success": success, "error": error,
                                                                     "completed_epoch": time.time(), "completed_monotonic": time.monotonic()})
        try:
            None  # Node agent reports completion and full streamed results.
        except Exception as exc:
            print("COMPLETION_REPORT_ERROR", repr(exc), flush=True)


if __name__ == "__main__":
    main()
