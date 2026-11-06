"""PyTorch CPU model-free tests of the real adapters and official tree cycles.

Run directly in the SpecEdge environment. --upstream-root can point to an
existing frozen suite when only the integration files were copied for tests.
No model is loaded, no inference server is created, and CUDA is not initialized.
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--upstream-root", type=Path, default=Path(__file__).resolve().parents[2])
options, unittest_args = parser.parse_known_args()
root = options.upstream_root.resolve()
integration = Path(__file__).resolve().parent
sys.path[:0] = [str(root), str(root / "runtime")]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FASTSD_EOS_TOKEN_IDS"] = "[151645,151643]"

from specedge.bootstrap import prepare_imports
prepare_imports()
sys.path.insert(0, str(integration))

import torch
import util
import log
from config import SpecEdgeClientConfig as client_config
from config import SpecEdgeBatchServerConfig as server_config
from specedge_grpc import specedge_pb2
from wire_codec import SpecEdgeWireCodec
from single_chain_server import SingleChainInferenceController

# Avoid configuring disk/multiprocessing loggers in a contract test.
class Recorder:
    def __init__(self):
        self.rows = []

    def log(self, row):
        self.rows.append(row)

log.get_logger = lambda: logging.getLogger("single_chain_test")
log.get_result_logger = Recorder

spec = importlib.util.spec_from_file_location("tested_single_chain_client", integration / "client.py")
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)
IntegratedSpecExecClient = client_module.IntegratedSpecExecClient
from specedge_correctness import SpecEdgeCorrectnessMixin


class Tokenizer:
    eos_token_id = 151645

    def encode(self, prompt, return_tensors=None):
        tokens = [int(value) for value in prompt.split()]
        return torch.tensor([tokens]) if return_tensors else tokens


class Cache:
    def __init__(self, batch, max_len):
        self.k_cache = torch.zeros(1, batch, 1, max_len, 1)
        self.v_cache = torch.zeros_like(self.k_cache)

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()


class GraphStub:
    def __init__(self, batch, max_len):
        self._past_key_values = Cache(batch, max_len)
        self.predictions = None

    def prefill(self, input_ids, batch_idx, **kwargs):
        count = input_ids.numel()
        self._past_key_values.k_cache[0, batch_idx, 0, :count, 0] = input_ids.flatten()
        self._past_key_values.v_cache[0, batch_idx, 0, :count, 0] = input_ids.flatten() + 1000

    def forward(self, input_ids, position_ids, cache_seq_indices, attention_mask, **kwargs):
        batch, width = input_ids.shape
        slots = cache_seq_indices.reshape(batch, width)
        for row in range(batch):
            assert slots[row].unique().numel() == width
            assert bool(attention_mask[row].bool().any(dim=-1).all())
            self._past_key_values.k_cache[0, row, 0, slots[row], 0] = input_ids[row].float()
            self._past_key_values.v_cache[0, row, 0, slots[row], 0] = input_ids[row].float() + 1000
        predictions = self.predictions if self.predictions is not None else input_ids + 1
        logits = torch.full((batch, width, 2048), -20.0)
        logits.scatter_(2, predictions.unsqueeze(-1), 20.0)
        return logits


def make_controller(max_len=80):
    controller = SingleChainInferenceController.__new__(SingleChainInferenceController)
    controller._num_clients = 2
    controller._client_gammas = [4, 8]
    controller._batch_size = 3
    controller._max_n_beams = 9
    controller._max_budget = 8
    controller._max_len = max_len
    controller._device = torch.device("cpu")
    controller._dtype = torch.bfloat16
    controller._temperature = 0
    controller._engine = GraphStub(3, max_len)
    controller.k_cache = torch.zeros(1, 2, 1, max_len, 1)
    controller.v_cache = torch.zeros_like(controller.k_cache)
    controller._client_indices = torch.full((3,), -1)
    controller._input_ids = torch.zeros(3, 9, dtype=torch.long)
    controller._position_ids = torch.zeros_like(controller._input_ids)
    controller._cache_seq_indices = torch.zeros_like(controller._input_ids)
    controller._parent_indices = torch.zeros(3, 8, dtype=torch.long)
    controller._cache_batch_indices = torch.arange(3).repeat_interleave(9)
    controller._attention_mask = torch.zeros(3, 1, 9, max_len, dtype=torch.bfloat16)
    controller._predefined_position_ids = torch.arange(max_len).unsqueeze(0)
    controller._predefined_attention_mask = torch.tril(torch.ones(max_len, max_len))[None, None]
    controller._iter_idx = torch.zeros(2, dtype=torch.long)
    controller._cache_lengths = [None, None]
    controller._active_requests = [None, None]
    controller._batch_number = 0
    controller._tokenizer = Tokenizer()
    controller._resp_queue = queue.Queue()
    controller._result_logger = Recorder()
    controller._logger = logging.getLogger("single_chain_server_test")
    return controller


def request(client_idx, gamma, prefix_len=4, root_token=4, prefill=True, req_idx=10):
    slots = list(range(prefix_len - 1, prefix_len + gamma))
    inputs = list(range(root_token, root_token + gamma + 1))
    mask = (torch.arange(80)[None, :] <= torch.tensor(slots)[:, None]).to(torch.bfloat16)
    encode = lambda value: SpecEdgeWireCodec.encode(torch.tensor(value, dtype=torch.long))
    prompt = " ".join(map(str, list(range(1, prefix_len)) + [root_token]))
    return specedge_pb2.ValidateRequest(
        client_idx=client_idx, req_idx=req_idx, prefill=prefill,
        prefix=prompt if prefill else "", input_ids=encode(inputs), position_ids=encode(slots),
        cache_seq_indices=encode(slots), parent_indices=encode(slots[:-1]),
        attention_mask=SpecEdgeWireCodec.encode(mask),
    )


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.controller = make_controller()
        self.original_timing = util.Timing
        class Timing:
            elapsed = 0
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        util.Timing = Timing

    def tearDown(self):
        util.Timing = self.original_timing

    def test_mixed_width_prefill_padding_and_next_round_kv(self):
        c = self.controller
        c._inference([request(0, 4), request(1, 8)])
        self.assertEqual(c._cache_lengths, [8, 12])
        for width in (5, 9):
            (payload, prefills), client = c._resp_queue.get_nowait()
            self.assertEqual(len(payload), width * 8)
            self.assertEqual(prefills, 2)
        self.assertEqual(c.k_cache[0, 0, 0, :8, 0].tolist(), list(range(1, 9)))
        self.assertEqual(c.k_cache[0, 1, 0, :12, 0].tolist(), list(range(1, 13)))
        self.assertEqual(c.k_cache[..., 12:, :].count_nonzero().item(), 0)
        c._inference([request(0, 4, prefix_len=9, root_token=9, prefill=False)])
        self.assertEqual(c._cache_lengths, [13, 12])
        self.assertEqual(c.k_cache[0, 0, 0, :13, 0].tolist(), list(range(1, 14)))

    def test_mismatch_discards_later_matches_and_padding(self):
        c = self.controller
        c._engine.predictions = torch.tensor([[5, 6, 77, 8, 9, 10, 11, 12, 13], [0] * 9, [0] * 9])
        c._inference([request(0, 4)])
        self.assertEqual(c._cache_lengths[0], 6)
        self.assertEqual(c.k_cache[0, 0, 0, :6, 0].tolist(), list(range(1, 7)))
        self.assertEqual(c.k_cache[0, 0, 0, 6:, 0].count_nonzero().item(), 0)

    def test_rejects_wrong_gamma_and_stale_request(self):
        c = self.controller
        with self.assertRaises(ValueError):
            c._inference([request(0, 8)])
        with self.assertRaises(ValueError):
            c._inference([request(0, 4, prefill=False)])

    def test_rejects_branch_and_noncausal_mask(self):
        c = self.controller
        req = request(0, 4)
        req.parent_indices = SpecEdgeWireCodec.encode(torch.tensor([3, 3, 5, 6]))
        with self.assertRaises(ValueError):
            c._inference([req])
        req = request(0, 4)
        req.attention_mask = SpecEdgeWireCodec.encode(torch.ones(5, 80, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            c._inference([req])


class DraftStub:
    max_len = 128

    def __init__(self):
        self.cache = torch.zeros(self.max_len, dtype=torch.long)

    def reset(self): self.cache.zero_()

    def prefill(self, input_ids, cache_seq_indices, **kwargs):
        self.cache[cache_seq_indices.flatten()] = input_ids.flatten()

    def forward(self, input_ids, cache_seq_indices, **kwargs):
        self.cache[cache_seq_indices.flatten()] = input_ids.flatten()
        logits = torch.full((1, input_ids.numel(), 2048), -20.0)
        logits.scatter_(2, (input_ids + 1).unsqueeze(-1), 20.0)
        return logits

    def gather(self, src_indices, dest_indices):
        self.cache[dest_indices] = self.cache[src_indices]
        self.cache[int(dest_indices.max()) + 1:] = 0


class Validator:
    def __init__(self, mismatch): self.mismatch = mismatch

    async def request(self, input_ids, **kwargs):
        result = input_ids.flatten() + 1
        if self.mismatch: result[0] += 1
        return result, int(kwargs["prefill"])


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        values = {
            "_initialized": True, "optimization": 2, "device": torch.device("cpu"),
            "dtype": torch.bfloat16, "max_n_beams": 1, "max_branch_width": 1,
            "max_new_tokens": 25, "client_idx": 0, "max_len": 128,
            "proactive_type": "included", "proactive_max_n_beams": 1,
            "proactive_max_branch_width": 1, "proactive_max_beam_len": 3,
            "host": "127.0.0.1:1",
        }
        for key, value in values.items(): setattr(client_config, key, value)

    async def test_official_chain_and_proactive_cycles_gamma4_gamma8(self):
        for gamma in (2, 4, 8):
            client_config.max_beam_len = gamma
            client_config.max_budget = gamma
            client_config.proactive_max_budget = gamma
            for mismatch in (False, True):
                with self.subTest(gamma=gamma, mismatch=mismatch):
                    engine = DraftStub()
                    client = IntegratedSpecExecClient(engine, Tokenizer(), "1 2 3 4", 128)
                    await client._validator._channel.close()
                    client._validator = Validator(mismatch)
                    await client.generate_exact(10)
                    self.assertEqual(client._prefix_tokens.numel() - 4, 25)
                    self.assertGreaterEqual(len(client.chain_audits), 3)
                    self.assertTrue(all(a["actual_candidates"] == gamma and a["wire_inputs"] == gamma + 1
                                        for a in client.chain_audits))
                    self.assertEqual(any(a["proactive_reused"] for a in client.chain_audits), not mismatch)
                    if not mismatch:
                        self.assertEqual(client._prefix_tokens.flatten().tolist(), list(range(1, 30)))
                    self.assertIs(super(IntegratedSpecExecClient, IntegratedSpecExecClient)._validate_tree,
                                  SpecEdgeCorrectnessMixin._validate_tree)

    async def test_client_to_target_wire_with_proactive_reuse(self):
        class InMemoryWire:
            def __init__(self, controller): self.controller = controller

            async def request(self, client_idx, req_idx, prefill=False, prefix=None, **tensors):
                payload = {key: SpecEdgeWireCodec.encode(value, dtype=torch.bfloat16 if key == "attention_mask" else None)
                           for key, value in tensors.items()}
                req = specedge_pb2.ValidateRequest(client_idx=client_idx, req_idx=req_idx,
                                                   prefill=prefill, prefix=prefix or "", **payload)
                self.controller._inference([req])
                (selection, prefills), identity = self.controller._resp_queue.get_nowait()
                assert identity == client_idx
                return SpecEdgeWireCodec.decode(selection, device="cpu", dtype=torch.long, shape=(-1,)), prefills

        # Server event timing is the only GPU-specific utility in this path.
        original_timing = util.Timing
        class Timing:
            elapsed = 0
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        util.Timing = Timing
        try:
            for gamma in (2, 4, 8):
                client_config.max_beam_len = client_config.max_budget = gamma
                client_config.proactive_max_budget = gamma
                controller = make_controller(max_len=128)
                controller._client_gammas[0] = gamma
                client = IntegratedSpecExecClient(DraftStub(), Tokenizer(), "1 2 3 4", 128)
                await client._validator._channel.close()
                client._validator = InMemoryWire(controller)
                await client.generate_exact(42)
                self.assertEqual(client._prefix_tokens.flatten().tolist(), list(range(1, 30)))
                self.assertTrue(any(a["proactive_reused"] for a in client.chain_audits))
                self.assertFalse(torch.cuda.is_initialized())
        finally:
            util.Timing = original_timing


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(argv=[sys.argv[0]] + unittest_args)
    assert not torch.cuda.is_initialized()
