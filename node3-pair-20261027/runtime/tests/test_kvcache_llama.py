"""No-download Llama parity tests against independent Transformers forwards."""
import unittest
from unittest.mock import patch

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import supports_varlen_model, varlen_generate
    _DEPS = True
except ImportError:
    _DEPS = False


@unittest.skipUnless(_DEPS, "requires Torch and Transformers")
class LlamaVarlenTests(unittest.TestCase):
    def _run_case(self, kv_heads, rope_scaling=None):
        torch.manual_seed(19)
        config = LlamaConfig(
            vocab_size=97, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=kv_heads, max_position_embeddings=128,
            attention_dropout=0.0, rope_scaling=rope_scaling,
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).eval()
        self.assertTrue(supports_varlen_model(model))
        manager = KVCacheModel_batching(model, temperature=1.0)
        manager.vocab_size = config.vocab_size
        seqs = [torch.randint(3, config.vocab_size, (1, n)) for n in (5, 13)]
        pids = ["short", "long"]

        def forward(residuals, ids):
            # Neither padding nor the dense Transformers model forward may
            # be invoked by this path, including cached continuations.
            with patch.object(model, "forward", side_effect=AssertionError("dense forward")), \
                 patch("torch.nn.utils.rnn.pad_sequence", side_effect=AssertionError("padding")):
                varlen_generate(manager, model, residuals, ids, 0, sample_next=False)

        def compare():
            with torch.no_grad():
                for pid, seq in zip(pids, seqs):
                    ref = model(seq, use_cache=True)
                    expected = ref.logits.float().softmax(-1)
                    torch.testing.assert_close(manager._prob_history[pid], expected, atol=2e-6, rtol=2e-5)
                    actual_cache = manager._past_key_values[pid]
                    self.assertEqual(actual_cache.get_seq_length(), seq.shape[1])
                    for actual, expected_kv in zip(actual_cache.to_legacy_cache(), ref.past_key_values.to_legacy_cache()):
                        for a, b in zip(actual, expected_kv):
                            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                            self.assertFalse(a.requires_grad)

        forward(seqs, pids)
        compare()
        # Unequal verify suffixes: includes the one-query cached case.
        suffixes = [torch.randint(3, config.vocab_size, (1, n)) for n in (1, 4)]
        forward(suffixes, pids)
        seqs = [torch.cat((s, r), 1) for s, r in zip(seqs, suffixes)]
        compare()
        # Rejection rollback followed by another continuation.
        for i, pid in enumerate(pids):
            seqs[i] = seqs[i][:, :-2]
            manager.rollback(pid, seqs[i].shape[1])
        suffixes = [torch.randint(3, config.vocab_size, (1, n)) for n in (3, 2)]
        forward(suffixes, pids)
        seqs = [torch.cat((s, r), 1) for s, r in zip(seqs, suffixes)]
        compare()
        # A newly admitted request and an existing request in the same batch.
        manager.reset(pids[0])
        fresh = torch.randint(3, config.vocab_size, (1, 9))
        suffix = torch.randint(3, config.vocab_size, (1, 2))
        forward([fresh, suffix], pids)
        seqs = [fresh, torch.cat((seqs[1], suffix), 1)]
        compare()
        with self.assertRaisesRegex(ValueError, "unique"):
            forward([fresh, fresh], ["same", "same"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            forward([fresh[:, :0]], ["empty"])

    def test_llama_mha(self):
        self._run_case(4)

    def test_tinyllama_gqa(self):
        self._run_case(2)

    def test_linear_rope(self):
        self._run_case(2, {"rope_type": "linear", "factor": 2.0})


if __name__ == "__main__":
    unittest.main()
