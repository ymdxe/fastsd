"""GPU gates for native GQA, causal alignment, isolation and cache lifecycle."""
import gc
import unittest
import weakref
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from src.attention_backend import PackedAttentionWorkspace, require_attention_backend
from src.cache_views import reusable_cache_buffers
from src.cache_transfer import CacheTransferManager
from src.kvcache_batching import KVCacheModel_batching
from src.kvcache_varlen import varlen_generate


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class FlashAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        require_attention_backend('flash_attn_varlen', device='cuda:0', dtype=torch.bfloat16)

    def tensors(self, histories, queries):
        ranges, offset, caches = [], 0, []
        for history, length in zip(histories, queries):
            ranges.append((offset, offset + length, history))
            offset += length
            shape = (1, 8, history + length, 128)
            caches.append((torch.randn(shape, device='cuda', dtype=torch.bfloat16),
                           torch.randn(shape, device='cuda', dtype=torch.bfloat16)))
        q = torch.randn((1, 32, offset, 128), device='cuda', dtype=torch.bfloat16)
        return ranges, q, caches

    def attend(self, workspace, ranges, q, caches):
        workspace.prepare(ranges, device=q.device, dtype=q.dtype, query_heads=32, kv_heads=8, head_dim=128)
        try:
            return workspace.attend(q, caches, scale=128**-0.5)
        finally:
            workspace.record_consumed()

    def test_fp32_reference_batch1_8_16_prefill_decode_verify(self):
        for batch in (1, 8, 16):
            for q_len in (1, 4, 5, 8, 9, 33):
                histories = [(0, 1, 7, 127, 128, 511)[i % 6] for i in range(batch)]
                queries = [q_len if i % 2 == 0 else max(1, q_len - 1) for i in range(batch)]
                ranges, q, caches = self.tensors(histories, queries)
                workspace = PackedAttentionWorkspace()
                actual = self.attend(workspace, ranges, q, caches)
                with self.subTest(batch=batch, queries=queries):
                    for (start, end, history), (k, v) in zip(ranges, caches):
                        qi = q[0, :, start:end].float()
                        ki, vi = k[0].float().repeat_interleave(4, 0), v[0].float().repeat_interleave(4, 0)
                        mask = torch.arange(k.shape[-2], device='cuda')[None, :] <= (history + torch.arange(end-start, device='cuda')[:, None])
                        scores = (qi @ ki.transpose(-2, -1)) * (128**-0.5)
                        reference = (scores.masked_fill(~mask, float('-inf')).softmax(-1) @ vi).transpose(0, 1).reshape(end-start, -1)
                        torch.testing.assert_close(actual[start:end].float(), reference, atol=2e-2, rtol=2e-2)
                    self.assertEqual(workspace.stats['attention_calls'], 1)
                    self.assertEqual(workspace.k.shape[1], 8)

    def test_no_future_or_cross_request_reads(self):
        ranges, q, caches = self.tensors([17, 9], [8, 4])
        workspace = PackedAttentionWorkspace()
        reference = self.attend(workspace, ranges, q, caches)
        caches[0][1][:, :, 18:, :].fill_(1024)
        caches[1][0].fill_(-512)
        caches[1][1].fill_(512)
        modified = self.attend(workspace, ranges, q, caches)
        torch.testing.assert_close(reference[0], modified[0], atol=0, rtol=0)
        self.assertFalse(torch.equal(reference[7], modified[7]))

    def test_workspace_capacity_and_cross_stream_fence(self):
        workspace = PackedAttentionWorkspace()
        ranges, q, caches = self.tensors([127]*16, [9]*16)
        reference = self.attend(workspace, ranges, q, caches)
        pointers = [t.data_ptr() for t in (workspace.q, workspace.k, workspace.v, workspace.cu_q, workspace.cu_k)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = self.attend(workspace, ranges, q, caches)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        self.assertEqual(pointers, [t.data_ptr() for t in (workspace.q, workspace.k, workspace.v, workspace.cu_q, workspace.cu_k)])


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class FlashModelCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(4321)
        cls.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=256,
            intermediate_size=512, num_hidden_layers=2, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128, max_position_embeddings=2048,
            attention_dropout=0.0)).to(device='cuda', dtype=torch.bfloat16).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def manager(self, backend):
        manager = KVCacheModel_batching(self.model, temperature=1, top_p=1, attention_backend=backend)
        manager.vocab_size = 97
        return manager

    def run_both(self, managers, parts, pids, prefill=False):
        for manager in managers:
            varlen_generate(manager, self.model, parts, pids, 0, is_prefill=prefill, sample_next=False)
        for pid in pids:
            torch.testing.assert_close(managers[0]._prob_history[pid], managers[1]._prob_history[pid], atol=2e-2, rtol=2e-2)
            self.assertEqual(managers[0]._past_key_values[pid].get_seq_length(), managers[1]._past_key_values[pid].get_seq_length())

    def test_chunked_prefill_decode_gamma4_8_reject_and_cleanup(self):
        for batch in (1, 8, 16):
            managers = [self.manager('sdpa_per_sequence'), self.manager('flash_attn_varlen')]
            pids = list(range(batch))
            prefixes = [torch.randint(1, 97, (1, 120+i%5), device='cuda') for i in pids]
            self.run_both(managers, [p[:, :61] for p in prefixes], pids, True)
            self.run_both(managers, [p[:, 61:] for p in prefixes], pids, True)
            for q_len in (1, 4, 5, 8, 9, 1):
                tails = [torch.randint(1, 97, (1, q_len), device='cuda') for _ in pids]
                self.run_both(managers, tails, pids)
                # Reject all but the first candidate. Poison the unused capacity.
                if q_len > 1:
                    for manager in managers:
                        for pid in pids:
                            length = manager._past_key_values[pid].get_seq_length() - q_len + 1
                            manager.rollback(pid, length)
                            for k, v in reusable_cache_buffers(manager._past_key_values[pid]):
                                k[..., length:, :].fill_(float('nan'))
                                v[..., length:, :].fill_(float('nan'))
            for manager in managers:
                refs = [weakref.ref(reusable_cache_buffers(manager._past_key_values[pid])[0][0]) for pid in pids]
                for pid in pids:
                    manager.reset(pid)
                torch.cuda.synchronize()
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
                self.assertFalse(manager._prob_history)

    def test_real_transfer_drops_gpu_storage_and_fences_new_forward(self):
        manager = self.manager('flash_attn_varlen')
        transfers = CacheTransferManager('cuda:0', pinned_pool_bytes=16*1024*1024)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            varlen_generate(manager, self.model, [torch.tensor([[1,2,3,4]], device='cuda')], ['p'], 0, sample_next=False)
            cache = manager._past_key_values['p']
            refs = [weakref.ref(k) for k, v in reusable_cache_buffers(cache)]
            transfers.mark_computed(cache, producer_stream=stream)
            host = transfers.move(cache, 'cpu')
        manager._past_key_values['p'] = host
        del cache
        torch.cuda.synchronize()
        transfers.collect()
        gc.collect()
        self.assertFalse(hasattr(host, '_fastsd_buffers'))
        self.assertTrue(all(ref() is None for ref in refs))
        restored = transfers.move(host, 'cuda:0')
        self.assertFalse(hasattr(restored, '_fastsd_buffers'))
        with torch.cuda.stream(stream):
            transfers.wait_ready(restored, 'cuda:0', compute_stream=stream)
            manager._past_key_values['p'] = restored
            varlen_generate(manager, self.model, [torch.tensor([[5,6]], device='cuda')], ['p'], 0, sample_next=False)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertEqual(manager._past_key_values['p'].get_seq_length(), 6)
        self.assertTrue(torch.isfinite(manager._prob_history['p']).all())
        manager.reset('p')
        transfers.close()

    def test_full_and_chunked_prefill(self):
        for batch in (1,8,16):
            complete = self.manager('flash_attn_varlen')
            chunked = self.manager('flash_attn_varlen')
            pids = list(range(batch))
            parts = [torch.randint(1,97,(1,133+i),device='cuda') for i in pids]
            varlen_generate(complete,self.model,parts,pids,0,sample_next=False)
            for left,right in ((0,64),(64,128),(128,None)):
                varlen_generate(chunked,self.model,[p[:,left:right] for p in parts],pids,0,sample_next=False)
            for pid in pids:
                torch.testing.assert_close(complete._prob_history[pid],chunked._prob_history[pid],atol=2e-2,rtol=2e-2)


if __name__ == '__main__':
    unittest.main()
