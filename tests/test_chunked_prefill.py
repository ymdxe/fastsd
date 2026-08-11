from types import SimpleNamespace
import unittest

import torch
from transformers.cache_utils import DynamicCache

from src.kvcache_batching import KVCacheModel_batching


class FakeCausalLM:
    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size

    def __call__(
        self,
        input_ids,
        *,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
    ):
        batch_size, token_count = input_ids.shape
        logits = torch.zeros(batch_size, token_count, self.vocab_size)
        logits.scatter_(2, (input_ids % self.vocab_size).unsqueeze(-1), 5.0)
        appended = input_ids.to(torch.float32).reshape(batch_size, 1, token_count, 1)
        if past_key_values is None:
            key = appended
            value = appended.clone()
        else:
            legacy = past_key_values.to_legacy_cache()
            old_key, old_value = legacy[0]
            key = torch.cat((old_key, appended), dim=-2)
            value = torch.cat((old_value, appended.clone()), dim=-2)
        return SimpleNamespace(
            logits=logits,
            past_key_values=DynamicCache.from_legacy_cache([(key, value)]),
        )


class ChunkedPrefillTests(unittest.TestCase):
    def setUp(self):
        self.manager = KVCacheModel_batching(
            FakeCausalLM(), temperature=1.0, top_k=0, top_p=0.0
        )
        self.manager.vocab_size = 16

    def cache_values(self, proc_id):
        key, _ = self.manager._past_key_values[proc_id].to_legacy_cache()[0]
        return key[0, 0, :, 0].tolist()

    def test_chunked_prefill_preserves_variable_length_kv_and_probability_history(self):
        first_chunks = torch.tensor([[1, 2, 0], [5, 6, 7]], dtype=torch.long)
        first = self.manager.prefill_chunks(
            first_chunks,
            proc_ids=["p1", "p2"],
            pad_token_id=0,
            input_lens=[2, 3],
            cursors=[0, 0],
        )

        self.assertEqual([item["cache_len"] for item in first], [2, 3])
        self.assertEqual(self.cache_values("p1"), [1.0, 2.0])
        self.assertEqual(self.cache_values("p2"), [5.0, 6.0, 7.0])

        next_chunks = torch.tensor([[3, 4], [8, 0]], dtype=torch.long)
        second = self.manager.prefill_chunks(
            next_chunks,
            proc_ids=["p1", "p2"],
            pad_token_id=0,
            input_lens=[2, 1],
            cursors=[2, 3],
        )

        self.assertEqual([item["cache_len"] for item in second], [4, 4])
        self.assertEqual(self.cache_values("p1"), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.cache_values("p2"), [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(self.manager._prob_history["p1"].shape[1], 4)
        self.assertEqual(self.manager._prob_history["p2"].shape[1], 4)

    def test_continuation_requires_existing_state_at_exact_cursor(self):
        with self.assertRaisesRegex(ValueError, "missing Prefill continuation state"):
            self.manager.prefill_chunks(
                torch.tensor([[1]], dtype=torch.long),
                proc_ids=["missing"],
                pad_token_id=0,
                input_lens=[1],
                cursors=[1],
            )

        self.manager.prefill_chunks(
            torch.tensor([[1, 2]], dtype=torch.long),
            proc_ids=["p1"],
            pad_token_id=0,
            input_lens=[2],
            cursors=[0],
        )
        with self.assertRaisesRegex(ValueError, "does not match cursor"):
            self.manager.prefill_chunks(
                torch.tensor([[3]], dtype=torch.long),
                proc_ids=["p1"],
                pad_token_id=0,
                input_lens=[1],
                cursors=[1],
            )

    def test_variable_length_verify_drops_padded_cache_gap(self):
        self.manager.prefill_chunks(
            torch.tensor([[1, 2, 0], [5, 6, 7]], dtype=torch.long),
            proc_ids=["p1", "p2"],
            pad_token_id=0,
            input_lens=[2, 3],
            cursors=[0, 0],
        )

        self.manager.generate(
            [
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            ],
            gamma=1,
            proc_ids=["p1", "p2"],
            pad_token_id=0,
            is_prefill=False,
        )

        self.assertEqual(self.cache_values("p1"), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.cache_values("p2"), [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(self.manager._prob_history["p1"].shape[1], 4)
        self.assertEqual(self.manager._prob_history["p2"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
