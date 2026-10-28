import unittest
import gc

try:
    import torch
    from transformers.cache_utils import DynamicCache
    from src.cache_transfer import CacheTransferManager
    DEPS = True
except ImportError:
    DEPS = False


@unittest.skipUnless(DEPS and torch.cuda.is_available(), "requires CUDA")
class CacheTransferTests(unittest.TestCase):
    def test_cuda_varlen_gqa_and_rollback_match_dense(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from src.kvcache_batching import KVCacheModel_batching
        from src.kvcache_varlen import varlen_generate
        torch.manual_seed(3)
        config = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to("cuda").eval()
        manager = KVCacheModel_batching(model, temperature=1.0)
        manager.vocab_size = 97
        manager.bounded_probabilities = True
        seq = torch.randint(1, 97, (1, 64), device="cuda")
        varlen_generate(manager, model, [seq], ["p"], 0, is_prefill=True, sample_next=False)
        pointer = manager._past_key_values["p"]._fastsd_buffers[0][0].data_ptr()
        for n in (4, 1, 3):
            start = seq.shape[1]
            tail = torch.randint(1, 97, (1, n), device="cuda")
            seq = torch.cat((seq, tail), 1)
            varlen_generate(manager, model, [tail], ["p"], 0, sample_next=False)
            with torch.no_grad():
                expected = model(seq).logits[:, start - 1:].softmax(-1)
            torch.testing.assert_close(manager.probability_rows("p", start - 1, start + n),
                                       expected, atol=2e-5, rtol=2e-4)
            self.assertEqual(manager._past_key_values["p"]._fastsd_buffers[0][0].data_ptr(), pointer)
            self.assertEqual(manager._past_key_values["p"].to_legacy_cache()[0][0].data_ptr(), pointer)
            manager.rollback("p", seq.shape[1] - 1)
            seq = seq[:, :-1]

    def test_async_roundtrip_event_and_source_lifetime(self):
        transfers = CacheTransferManager("cuda:0", pinned_pool_bytes=1024 * 1024)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            data = torch.arange(128, device="cuda", dtype=torch.float32).reshape(1, 2, 16, 4)
            cache = DynamicCache.from_legacy_cache([(data * 2, data + 3)])
            transfers.mark_computed(cache, producer_stream=stream)
            host = transfers.move(cache, "cpu")
        del cache
        gc.collect()
        restored = transfers.move(host, "cuda:0")
        actual = restored.to_legacy_cache()
        torch.testing.assert_close(actual[0][0], data * 2)
        torch.testing.assert_close(actual[0][1], data + 3)
        self.assertTrue(host.to_legacy_cache()[0][0].is_pinned())
        self.assertEqual(transfers.metrics["copies"], 2)
        del host, restored, actual
        torch.cuda.synchronize()
        transfers.collect()
        gc.collect()
        transfers.collect()
        self.assertLessEqual(transfers.pool.bytes, 1024 * 1024)
        self.assertFalse(transfers.pending)
        transfers.close()


if __name__ == "__main__":
    unittest.main()
