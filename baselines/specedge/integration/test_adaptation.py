import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from .dataset_adapter import ADAPTERS
from .linear import TargetOnlyClient, grow_single_chain
from .memory import BudgetExceeded, DeviceBudget, cache_bytes, plan_budget
from .modeling import compare_token_signatures, token_signature
from .offload import LRUSlots
from .sampling import select_tokens
from .settings import ADAPTED, OFFICIAL, OFFLOAD, SINGLE_CHAIN, check_comparison_sampling, method_for, runtime_options, validate_config
from .tp import TPBatchEngine, ShardedCache, check_tp, parameter_bytes, shard_model
from src.common_metrics import IOStall
from src.metrics import request_latency


class DatasetTests(unittest.TestCase):
    def test_humaneval(self):
        record = dict(prompt="def inc(value):\n", reference=dict(entry_point="inc", test="def check(fn):\n    assert fn(3) == 4"))
        adapter = ADAPTERS["humaneval"]
        self.assertEqual(adapter.build_prompt(record, None, chat=False), record["prompt"])
        self.assertIsNone(adapter.evaluate(record, "    return value + 1")["score"])
        program = adapter.program(record, adapter.extract_answer("```python\n    return value + 1\n```"))
        exec(program, {})
        self.assertEqual(adapter.evaluate(record, "    return value + 1", executor=lambda code: code == program)["score"], 1)

    def test_mbpp(self):
        adapter = ADAPTERS["mbpp"]
        record = dict(prompt="Increment an integer", reference=dict(test_list=["assert inc(3) == 4"]))
        self.assertEqual(adapter.build_prompt(record, None, chat=False), record["prompt"])
        program = adapter.program(record, adapter.extract_answer("```python\ndef inc(value):\n    return value + 1\n```"))
        exec(program, {})
        self.assertEqual(adapter.evaluate(record, "def inc(value): return value + 1", executor=lambda code: "assert inc(3) == 4" in code)["score"], 1)

    def test_gsm8k(self):
        adapter = ADAPTERS["gsm8k"]
        self.assertEqual(adapter.evaluate({"reference": "3+4\n#### 7"}, "step 1: \\boxed{7.0}")["score"], 1)

    def test_mgsm(self):
        adapter = ADAPTERS["mgsm"]
        self.assertEqual(adapter.evaluate({"reference": "1,204"}, "答案：1204</s> 枝:10")["score"], 1)
        self.assertEqual(adapter.evaluate({"reference": "3"}, "unknown")["score"], 0)

    def test_mt_bench_history_uses_observed_reply(self):
        adapter = ADAPTERS["mt_bench"]
        record = dict(reference=dict(turns=["first?", "second?"]))
        calls = []
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)))
        adapter.build_prompt(record, tokenizer, chat=True, history=["actual answer"])
        messages, options = calls[0]
        self.assertEqual([message["content"] for message in messages], ["first?", "actual answer", "second?"])
        self.assertFalse(options["enable_thinking"])
        self.assertEqual(adapter.evaluate(record, ["a", "b"], judge=lambda _record, _outputs: 7)["score"], 7)


class ConfigTests(unittest.TestCase):
    def test_strict_budget_and_clamp_on_every_card(self):
        common = dict(instances=3, batch_size=3, max_len=10, max_n_beams=2,
                      verifier_width=4, reserve_bytes=100, element_size=2)
        devices = [DeviceBudget("cuda:0", 10000, 10000, 1000, 300), DeviceBudget("cuda:1", 10000, 4600, 1000, 300)]
        with self.assertRaisesRegex(MemoryError, r"cuda:1"):
            plan_budget(devices, **common)
        plan = plan_budget(devices, **common, allow_memory_clamp=True)
        self.assertTrue(plan["prealloc_clamped"])
        self.assertEqual(plan["batch_size"], 2)
        self.assertEqual(plan["prealloc_kv_bytes"], 2 * (3 + 2) * 300)
        self.assertEqual(plan["devices"][1]["kv_per_req"], 600)
        required = plan["devices"][1]["estimated_required_bytes"]
        with self.assertRaises(MemoryError):
            plan_budget([DeviceBudget("cuda:1", required + 100, required + 100, 1000, 300)], **{**common, "batch_size": 2})

    def test_qwen_token_signature_and_special_tokens(self):
        config = SimpleNamespace(vocab_size=151936, eos_token_id=151645, bos_token_id=None, pad_token_id=151643)
        generation = SimpleNamespace(eos_token_id=[151645, 151643], pad_token_id=151643)
        tokenizer = SimpleNamespace(get_vocab=lambda: {"<|im_end|>": 151645, "<|endoftext|>": 151643, "hello": 17},
                                    eos_token_id=151645, pad_token_id=151643, bos_token_id=None)
        signature = token_signature(tokenizer, config, generation)
        self.assertEqual(signature["eos"]["effective"], [151643, 151645])
        compare_token_signatures(signature, token_signature(tokenizer, config, generation))
        swapped = deepcopy(tokenizer)
        swapped.get_vocab = lambda: {**tokenizer.get_vocab(), "hello": 18}
        with self.assertRaisesRegex(ValueError, "vocab_sha256"):
            compare_token_signatures(signature, token_signature(swapped, config, generation))
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            token_signature(tokenizer, config, SimpleNamespace(eos_token_id=[151645, 151643], pad_token_id=151645))
        with self.assertRaisesRegex(ValueError, "bos"):
            compare_token_signatures(signature, token_signature(tokenizer, config, SimpleNamespace(bos_token_id=17)))

    def test_selection_and_method_labels(self):
        logits = torch.tensor([[[0.0, 3.0, 1.0], [4.0, 2.0, 1.0]]])
        for args in [(0, 1, 0), (0.7, 1, 1), (0.7, 0.01, 0)]:
            torch.testing.assert_close(select_tokens(logits, *args), torch.tensor([[1, 0]]))
        self.assertEqual(method_for("tree"), ADAPTED)
        self.assertEqual(method_for("single_chain"), SINGLE_CHAIN)
        self.assertEqual(method_for("tree", offload=True), OFFLOAD)

    def test_adapted_tp_chain_config_and_single_gpu_offload(self):
        config = {
            "base": dict(dtype="bf16", max_len=64),
            "server": dict(device="cuda:0", target_devices=["cuda:0", "cuda:1"], batch_type="dynamic", temperature=0.7,
                           num_clients=2, max_batch_size=1),
            "client": dict(draft_structure="single_chain", max_beam_len=4, max_n_beams=1, max_branch_width=1,
                           max_budget=4, max_new_tokens=8, proactive=dict(type="disabled")),
            "integration": dict(method=SINGLE_CHAIN),
        }
        self.assertEqual(validate_config(config), (SINGLE_CHAIN, ["cuda:0", "cuda:1"]))
        config["server"]["offload"] = True
        config["integration"]["method"] = OFFLOAD
        with self.assertRaisesRegex(ValueError, "single GPU"):
            validate_config(config)
        config["server"]["target_devices"] = ["cuda:0"]
        self.assertEqual(validate_config(config), (OFFLOAD, ["cuda:0"]))
        config["integration"]["method"] = OFFICIAL
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_config(config)

    def test_runtime_switches_and_sampling(self):
        self.assertEqual(runtime_options("single_chain", 4, environ={}), ("single_chain", 4, False))
        for tokens in (0, 4):
            structure, count, offload = runtime_options("single_chain", 4, environ={"SPECEDGE_DRAFT_TOKENS": str(tokens)})
            self.assertEqual(method_for(structure, offload, count), f"{SINGLE_CHAIN}_draft{tokens}")
        self.assertEqual(runtime_options("single_chain", 4, environ={"SPECEDGE_KV_OFFLOAD": "1"}), ("single_chain", 4, True))
        with self.assertRaisesRegex(ValueError, "only LRU"):
            runtime_options("single_chain", 4, environ={"SPECEDGE_KV_OFFLOAD_POLICY": "fifo"})
        check_comparison_sampling(0.7, 0.8, 0, "rejection")
        for sampling in ((0, 0.8, 0, "rejection"), (0.7, 1, 0, "rejection"), (0.7, 0.8, 1, "rejection"), (0.7, 0.8, 0, "greedy")):
            with self.assertRaises(ValueError):
                check_comparison_sampling(*sampling)

    def test_native_oom_uses_requested_batch(self):
        device = DeviceBudget("cuda:0", 10_000, 10_000, 3000, 2000)
        with self.assertRaises(BudgetExceeded) as failure:
            plan_budget([device], instances=2, batch_size=4, max_len=8, max_n_beams=1,
                        verifier_width=5, reserve_bytes=1000, element_size=2)
        self.assertEqual(failure.exception.requested_batch, 4)
        self.assertEqual(failure.exception.devices[0]["batch"], 4)

    def test_lru_evict_and_restore_preserves_both_kv_tensors(self):
        self._exercise_lru(torch.device("cpu"), lambda: None)

    @unittest.skipUnless(os.environ.get("SPECEDGE_TEST_CUDA") == "1" and torch.cuda.is_available(), "requires CUDA")
    def test_lru_transfers_both_kv_tensors_over_cuda(self):
        self._exercise_lru(torch.device("cuda:0"), lambda: torch.cuda.current_stream().synchronize())

    def _exercise_lru(self, device, synchronize):
        slots = LRUSlots(torch.empty(2, 3, device=device), 1, synchronize=synchronize)
        first = SimpleNamespace(client_idx=0, prefill=True)
        second = SimpleNamespace(client_idx=1, prefill=True)
        slots.begin([first])
        slots.writing = True
        expected = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 2, 3)
        slots.slot(0).copy_(expected)
        self.assertEqual(slots.stats[0]["io_stall_count"], 0)
        slots.begin([second])
        slots.writing = True
        slots.slot(1).fill_(19)
        self.assertEqual(slots.stats[1]["evictions"], 1)
        self.assertEqual(slots.stats[1]["d2h_bytes"], slots.slot_bytes)
        first.prefill = False
        slots.begin([first])
        torch.testing.assert_close(slots.slot(0), expected)
        for _component in range(2):
            slots.slot(0)
        stats = slots.stats[0]
        self.assertEqual(stats["restores"], 1)
        self.assertEqual(stats["io_stall_count"], 1)
        self.assertEqual(stats["h2d_bytes"], slots.slot_bytes)
        self.assertGreater(stats["exposed_h2d_stall_ms"], 0)

    def test_latency_fallback_and_zero_stall_denominator(self):
        self.assertEqual(request_latency(16.0, None, 5), (16.0, 0.0, True))
        self.assertEqual(request_latency(16.0, 4.0, 5), (4.0, 3.0, False))
        stall = IOStall()
        stall.record(6.0, restored=True)
        stall.record(0.0, restored=True)
        self.assertEqual(stall.summary(), dict(total_ms=6, count=1, avg_ms=3))


class TargetOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_draft0_sends_only_target_tokens_and_stops_at_eos(self):
        requests = []

        async def request(**payload):
            requests.append(payload)
            return torch.tensor([8 if len(requests) == 1 else 9]), 0

        async def close():
            pass

        tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: torch.tensor([[2, 3]]))
        vendor = SimpleNamespace(request=request, close=close)
        client = TargetOnlyClient(tokenizer, "prompt", [9], vendor, client_idx=3, max_len=12, max_new_tokens=4)
        await client.generate_exact(11)
        self.assertEqual(client._prefix_tokens.tolist(), [[2, 3, 8, 9]])
        self.assertEqual((client.drafted_tokens, client.accepted_tokens, client.verify_steps), (0, 0, 2))
        self.assertEqual([item["prefill"] for item in requests], [True, False])
        self.assertTrue(all(item["parent_indices"].numel() == 0 and item["input_ids"].numel() == 1 for item in requests))


class OfficialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = str(Path(__file__).resolve().parents[1] / "official" / "src")
        sys.path.insert(0, cls.source)
        cls.addClassCleanup(sys.path.remove, cls.source)

    def test_official_tree_has_a_single_contiguous_path(self):
        from specedge.tree import Tree

        tree = Tree(torch.tensor([[8, 9]]), torch.device("cpu"), torch.float32, 12)
        warmups = []

        def candidates(prefill):
            warmups.append(prefill)
            parent = torch.where(tree.status[:tree.end] == tree.CANDIDATE)[0]
            tree.status[parent] = tree.PROCESSED
            return torch.tensor([[0.0, 3.0]]), parent, tree.positions[parent], tree.logprobs[parent], 0.0

        def propose(logits):
            return torch.log_softmax(logits, dim=-1).max(dim=-1)

        grow_single_chain(tree, candidates, 4, True, propose=propose)
        self.assertEqual(warmups, [True, False, False, False, False])
        self.assertEqual(tree.end - tree.prefix_len, 4)
        self.assertEqual(tree.parents[tree.prefix_len:tree.end].tolist(), [1, 2, 3, 4])
        self.assertEqual(tree.tokens[tree.prefix_len:tree.end].tolist(), [1] * 4)

    def _exercise_tp(self, devices, dtype):
        from model.qwen3 import Qwen3ForCausalLM
        from model.cache import KVCache
        from transformers import Qwen3Config

        def invert_mask(mask):
            return mask.eq(0).to(mask.dtype) * torch.finfo(mask.dtype).min

        torch.manual_seed(8)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=4, attention_bias=True)
        full = Qwen3ForCausalLM(config).to(devices[0], dtype=dtype).eval()
        parallel = shard_model(deepcopy(full).cpu(), devices)
        engine = TPBatchEngine(parallel, 10, 2, 3, devices, invert_mask=invert_mask)
        reference_kv = KVCache(config, 3, 10, devices[0], dtype, 2)
        estimates = parameter_bytes(full, torch.empty((), dtype=dtype).element_size(), 2)
        self.assertLess(estimates[1], estimates[0])
        self.assertEqual(engine._past_key_values.shards[1].k_cache.size(2), 1)
        self.assertEqual(sum(cache.k_cache.numel() * cache.k_cache.element_size() * 2 for cache in engine._past_key_values.shards), 2 * cache_bytes(config, 10, torch.empty((), dtype=dtype).element_size()))
        for slot in range(2):
            prefill = dict(input_ids=torch.tensor([[1, 2]], device=devices[0]),
                           position_ids=torch.arange(2, device=devices[0]).unsqueeze(0),
                           cache_seq_indices=torch.arange(2, device=devices[0]),
                           attention_mask=torch.nn.functional.pad(torch.ones((1, 1, 2, 2), device=devices[0], dtype=dtype).tril(), (0, 8)))
            engine.prefill(slot, **prefill)
            with reference_kv.prefill_context(2, slot):
                full(**{**prefill, "attention_mask": invert_mask(prefill["attention_mask"])},
                     cache_batch_indices=torch.zeros(2, device=devices[0], dtype=torch.long), past_key_values=reference_kv)
        kwargs = dict(input_ids=torch.tensor([[3, 4, 5], [6, 7, 8]], device=devices[0]),
                      position_ids=torch.arange(2, 5, device=devices[0]).expand(2, -1),
                      cache_batch_indices=torch.arange(2, device=devices[0]).repeat_interleave(3),
                      cache_seq_indices=torch.arange(2, 5, device=devices[0]).repeat(2),
                      attention_mask=torch.nn.functional.pad(torch.ones((2, 1, 3, 5), device=devices[0], dtype=dtype).tril(diagonal=2), (0, 5)))
        expected = full(**{**kwargs, "attention_mask": invert_mask(kwargs["attention_mask"])}, past_key_values=reference_kv)[0]
        actual = engine.forward(**kwargs)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
        torch.testing.assert_close(actual.argmax(-1), expected.argmax(-1))
        indices = torch.tensor([0, 2, 4], device=devices[0])
        destinations = torch.arange(3, device=devices[0])
        reference_kv.gather(1, indices, destinations)
        engine.gather(1, indices, destinations)
        session = ShardedCache(config, 10, 3, 2, devices, dtype)
        session.k_cache[:, 0, ...].copy_(engine._past_key_values.k_cache[:, 1, ...])
        session.v_cache[:, 0, ...].copy_(engine._past_key_values.v_cache[:, 1, ...])
        for name in ("k_cache", "v_cache"):
            combined = torch.cat([getattr(cache, name)[:, 0].to(devices[0]) for cache in session.shards], dim=1)
            torch.testing.assert_close(combined, getattr(reference_kv, name)[:, 1], atol=0.002, rtol=0.02)

    @torch.inference_mode()
    def test_shards_match_official_prefill_forward_gather_and_session_copy_on_cpu(self):
        self._exercise_tp([torch.device("cpu"), torch.device("cpu")], torch.float32)

    @unittest.skipUnless(os.environ.get("SPECEDGE_TEST_CUDA") == "1", "enable explicit two-GPU smoke")
    @torch.inference_mode()
    def test_two_cuda_cards(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("two visible CUDA GPUs required")
        self._exercise_tp([torch.device("cuda:0"), torch.device("cuda:1")], torch.bfloat16)

    @unittest.skipUnless(os.environ.get("SPECEDGE_TEST_CUDA") == "1", "enable CUDA smoke")
    @torch.inference_mode()
    def test_cuda_math_with_one_card(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self._exercise_tp([torch.device("cuda:0")] * 2, torch.bfloat16)

    def test_tp_rejects_nondivisible_heads(self):
        config = SimpleNamespace(model_type="qwen3", num_attention_heads=4, num_key_value_heads=3, intermediate_size=16)
        with self.assertRaisesRegex(ValueError, "divide"):
            check_tp(config, 2)
