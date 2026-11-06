"""CPU contracts for persistent storage, rollback and explicit backends."""
import gc
import unittest
import weakref
from unittest import mock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from src.attention_backend import require_attention_backend, validate_attention_backend
from src.cache_views import dynamic_cache_from_views, reusable_cache_buffers
from src.kvcache_batching import KVCacheModel_batching
from src.kvcache_varlen import varlen_generate


class ResidentCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=97, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=512, attention_dropout=0.0,
        )).eval()

    def manager(self):
        manager = KVCacheModel_batching(self.model, temperature=0.7, top_p=1)
        manager.vocab_size = 97
        return manager

    def forward(self, manager, ids, pid='a'):
        return varlen_generate(manager, self.model, [torch.tensor([ids])], [pid], 0,
                               sample_next=False)[0]

    def test_capacity_reuse_and_128_token_growth(self):
        m = self.manager()
        ids = [x % 96 for x in range(127)]
        self.forward(m, ids)
        cache = m._past_key_values['a']
        buffers = reusable_cache_buffers(cache)
        ptrs = [k.data_ptr() for k, v in buffers]
        prefix = [k.clone() for k, v in cache.to_legacy_cache()]
        self.assertTrue(all(k.shape[-2] == 128 for k, v in buffers))
        self.forward(m, [23])
        self.assertEqual(ptrs, [k.data_ptr() for k, v in reusable_cache_buffers(m._past_key_values['a'])])
        self.forward(m, [24])
        grown = m._past_key_values['a']
        self.assertEqual(grown.get_seq_length(), 129)
        self.assertTrue(all(k.shape[-2] == 256 for k, v in reusable_cache_buffers(grown)))
        self.assertTrue(all(k.data_ptr() != p for (k, v), p in zip(reusable_cache_buffers(grown), ptrs)))
        for expected, (key, value) in zip(prefix, grown.to_legacy_cache()):
            torch.testing.assert_close(expected, key[..., :127, :], atol=0, rtol=0)

    def test_rollback_overwrites_rejected_tokens_without_reading_capacity(self):
        m, reference = self.manager(), self.manager()
        prefix = [1, 7, 11, 19, 27, 31]
        self.forward(m, prefix + [80, 81, 82, 83, 84, 85, 86, 87])
        ptrs = [k.data_ptr() for k, v in reusable_cache_buffers(m._past_key_values['a'])]
        m.rollback('a', len(prefix))
        for key, value in reusable_cache_buffers(m._past_key_values['a']):
            key[..., len(prefix):, :] = float('nan')
            value[..., len(prefix):, :] = float('nan')
        actual = self.forward(m, [3, 4, 5, 6])
        expected = self.forward(reference, prefix + [3, 4, 5, 6])
        self.assertEqual(actual[-1].item(), expected[-1].item())
        torch.testing.assert_close(m._prob_history['a'][:, -4:], reference._prob_history['a'][:, -4:],
                                   atol=1e-5, rtol=1e-5)
        self.assertEqual(ptrs, [k.data_ptr() for k, v in reusable_cache_buffers(m._past_key_values['a'])])
        self.assertEqual(m._past_key_values['a'].get_seq_length(), 10)

    def test_chunked_prefill_and_multi_request_isolation(self):
        a, b = [1, 2, 3, 7, 9], [83, 70, 42]
        m, ref = self.manager(), self.manager()
        varlen_generate(m, self.model, [torch.tensor([a[:2]]), torch.tensor([b])], ['a', 'b'], 0,
                        sample_next=False)
        actual = self.forward(m, a[2:])
        expected = self.forward(ref, a)
        self.assertEqual(actual[-1].item(), expected[-1].item())
        torch.testing.assert_close(m._prob_history['a'][:, -3:], ref._prob_history['a'][:, -3:],
                                   atol=1e-5, rtol=1e-5)
        self.assertEqual(m._past_key_values['b'].get_seq_length(), len(b))

    def test_reset_releases_storage_and_probability_state(self):
        m = self.manager()
        self.forward(m, [2, 3, 4])
        buffer_ref = weakref.ref(reusable_cache_buffers(m._past_key_values['a'])[0][0])
        m.reset('a')
        gc.collect()
        self.assertIsNone(buffer_ref())
        self.assertFalse(m._past_key_values)
        self.assertFalse(m._prob_history)
        self.assertFalse(m._prob_start)

    def test_transfer_copy_drops_old_buffer_references(self):
        m = self.manager()
        self.forward(m, [2, 3, 4])
        original = m._past_key_values['a']
        moved = dynamic_cache_from_views([(k.clone(), v.clone()) for k, v in original.to_legacy_cache()])
        self.assertFalse(hasattr(moved, '_fastsd_buffers'))
        # Simulate a stale cache-version attachment: reject it immediately.
        moved._fastsd_buffers = original._fastsd_buffers
        self.assertIsNone(reusable_cache_buffers(moved))
        self.assertFalse(hasattr(moved, '_fastsd_buffers'))


class BackendContractTests(unittest.TestCase):
    def test_legacy_default_and_invalid_values(self):
        self.assertEqual(KVCacheModel_batching(None).attention_backend, 'sdpa_per_sequence')
        self.assertEqual(require_attention_backend('sdpa_per_sequence')['flash_attn_version'], None)
        with self.assertRaises(ValueError):
            validate_attention_backend('auto')

    def test_missing_dependency_and_wrong_version_never_fallback(self):
        with mock.patch('src.attention_backend.importlib.import_module', side_effect=ImportError('missing')):
            with self.assertRaises(ImportError):
                require_attention_backend('flash_attn_varlen')
        module = mock.Mock(__version__='2.7.0')
        with mock.patch('src.attention_backend.importlib.import_module', return_value=module):
            with self.assertRaises(RuntimeError):
                require_attention_backend('flash_attn_varlen')


if __name__ == '__main__':
    unittest.main()
