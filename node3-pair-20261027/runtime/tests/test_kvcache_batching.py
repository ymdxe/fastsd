import types
import unittest

try:
    import torch
    from transformers.cache_utils import DynamicCache
    from src.kvcache_batching import KVCacheModel_batching
except Exception:
    torch = None
    DynamicCache = None
    KVCacheModel_batching = None


if torch is not None:
    class KVCacheBatchingTests(unittest.TestCase):
        class FakeModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, use_cache=True):
                batch, width = input_ids.shape
                if past_key_values is None:
                    old = torch.zeros((batch, 1, 0, 1), dtype=torch.float32, device=input_ids.device)
                else:
                    old = past_key_values.to_legacy_cache()[0][0]
                if position_ids is None:
                    position_ids = torch.arange(width, device=input_ids.device).unsqueeze(0).expand(batch, -1)
                current = position_ids.to(torch.float32).view(batch, 1, width, 1)
                key = torch.cat((old, current), dim=-2)
                logits = torch.zeros((batch, width, 8), dtype=torch.float32, device=input_ids.device)
                logits[..., 1] = 1.0
                return types.SimpleNamespace(
                    logits=logits,
                    past_key_values=DynamicCache.from_legacy_cache([(key, key.clone())]),
                )

        def test_mixed_cached_lengths_do_not_persist_padding_gap(self):
            manager = KVCacheModel_batching(self.FakeModel(), temperature=0.0)
            manager.vocab_size = 8
            manager.generate(
                torch.tensor([[10, 11, 0], [20, 21, 22]]), 1,
                proc_ids=["a", "b"], pad_token_id=0, is_prefill=True,
                input_lens=[2, 3],
            )
            manager.forward_new_tokens(
                torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 23, 24]]),
                proc_ids=["a", "b"], pad_token_id=0, input_lens=[4, 5],
            )
            self.assertEqual(manager._past_key_values["a"].get_seq_length(), 4)
            self.assertEqual(manager._past_key_values["b"].get_seq_length(), 5)
            for pid, expected in (("a", [0, 1, 2, 3]), ("b", [0, 1, 2, 3, 4])):
                values = manager._past_key_values[pid].to_legacy_cache()[0][0][0, 0, :, 0].tolist()
                self.assertEqual(values, expected)
else:
    class KVCacheBatchingTests(unittest.TestCase):
        @unittest.skip("Torch/Transformers are required for KV cache tests")
        def test_mixed_cached_lengths_do_not_persist_padding_gap(self):
            pass


if __name__ == "__main__":
    unittest.main()
