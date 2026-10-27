import unittest

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.kvcache import KVCacheModel
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import varlen_generate
    from src.probability_window import append_rows
    DEPS = True
except ImportError:
    DEPS = False


@unittest.skipUnless(DEPS, "requires Torch and Transformers")
class ProbabilityWindowTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(27)
        cfg = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, attention_dropout=0.0)
        cfg._attn_implementation = "sdpa"
        return LlamaForCausalLM(cfg).eval()

    def test_window_owns_small_storage_and_checks_missing_boundaries(self):
        rows = torch.rand(1, 100, 41)
        live, start = append_rows(None, 0, rows, 0, 99)
        self.assertEqual(start, 99)
        self.assertEqual(live.untyped_storage().nbytes(), 41 * 4)
        with self.assertRaisesRegex(ValueError, "boundary"):
            append_rows(live, start, rows[:, :1], 101, 99)

    def test_target_window_matches_full_reference_through_rejection_and_bridge(self):
        for backend in ("varlen", "padded"):
            model = self.model()
            manager = KVCacheModel_batching(model, temperature=1)
            manager.vocab_size = 41
            manager.bounded_probabilities = True
            seq = torch.randint(1, 41, (1, 64))
            def forward(tail, prefill=False):
                if backend == "varlen":
                    varlen_generate(manager, model, [tail], ["p"], 0,
                                    is_prefill=prefill, sample_next=False)
                else:
                    manager.generate(seq, 1, ["p"], 0, prefill,
                                     input_lens=[seq.shape[1]], sample_next=False)
            forward(seq, True)
            self.assertEqual(manager._prob_history["p"].shape[1], 1)
            for accepted in (0, 2, 4):
                base = seq.shape[1]
                tail = torch.randint(1, 41, (1, 4))
                seq = torch.cat((seq, tail), 1)
                forward(tail)
                with torch.no_grad():
                    ref = model(seq).logits.float().softmax(-1)
                torch.testing.assert_close(manager.probability_rows("p", base - 1, base + 4),
                    ref[:, base - 1:], atol=2e-6, rtol=2e-5)
                manager.rollback("p", base + accepted)
                seq = seq[:, :base + accepted]
                self.assertEqual(manager._prob_history["p"].shape[1], 1)
            manager.reset("p")
            self.assertEqual(manager._prob_start, {})

    def test_draft_ahead_kv_reuse_avoids_recomputing_suffix(self):
        model = self.model()
        draft = KVCacheModel(model, temperature=0, bounded_probabilities=True)
        draft.vocab_size = 41
        prefix = torch.randint(1, 41, (1, 40))
        proposed, rows = draft.generate_with_probs(prefix, 4)
        self.assertEqual(draft._prob_history.shape[1], 4)
        torch.testing.assert_close(draft.probability_rows(39, 43)[0], rows)
        ahead, ahead_rows = draft.generate_with_probs(proposed, 4)
        # Full acceptance + bridge match: prefix includes the first ahead token.
        confirmed = ahead[:, :45]
        draft.rollback(ahead.shape[1] - 1)
        draft.retain_probabilities(confirmed.shape[1] - 1)
        before = draft.forward_tokens
        extended, _ = draft.generate_with_probs(ahead, 1)
        self.assertEqual(draft.forward_tokens - before, 1)
        with torch.no_grad():
            expected = model(ahead).logits[:, -1].argmax(-1)
        self.assertEqual(extended[0, -1].item(), expected.item())


if __name__ == "__main__":
    unittest.main()
