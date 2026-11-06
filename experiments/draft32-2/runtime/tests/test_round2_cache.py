import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace

from src.cache_protocol import CacheSnapshot, KVTransferTicket
from src.scheduler_runtime import CacheResidency, KVPrefetchWorker


class CacheProtocolTests(unittest.TestCase):
    def cache(self, length=8):
        return SimpleNamespace(get_seq_length=lambda: length)

    def test_reservation_claim_keeps_content_version_and_epoch_reuse_invalidates(self):
        registry, cache = CacheResidency(), self.cache()
        with registry.lock:
            snapshot = registry.snapshot_locked('p', cache)
            ticket = KVTransferTicket(snapshot, 'prefetch')
            registry.candidates['p'] = ticket
            registry.reserve(['p'])
            claimed = registry.claim_locked('p', cache, 'job')
            registry.activate(['p'])
            self.assertIs(claimed, ticket)
            self.assertTrue(ticket.matches(registry.snapshot_locked('p', cache)))
            registry.retire_session_locked('p')
            self.assertFalse(ticket.matches(registry.snapshot_locked('p', cache)))

    def test_pending_event_is_not_ready_until_all_chunks_recorded(self):
        ticket = KVTransferTicket(CacheSnapshot.capture(self.cache()), 'prefetch')
        ticket.last_copy = SimpleNamespace(query=lambda: True)
        ticket.copy_state = 'copying'
        self.assertFalse(ticket.ready())
        ticket.copy_done = ticket.last_copy
        self.assertTrue(ticket.ready())

    def test_slow_preparation_does_not_hold_registry_lock_or_replace_formal(self):
        registry, cache = CacheResidency(), self.cache()
        store = {'p': cache}
        entered, release = threading.Event(), threading.Event()
        def move(source, device):
            entered.set()
            self.assertTrue(release.wait(2))
            return self.cache()
        worker = KVPrefetchWorker(residency=registry, get_cache=store.get,
            set_cache=lambda *args: self.fail('background replaced formal cache'),
            move_cache=move, gpu_device='cuda', metrics=defaultdict(int))
        worker._latest_intent = frozenset({'p'})
        thread = threading.Thread(target=lambda: worker._execute(frozenset({'p'})))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertTrue(registry.lock.acquire(timeout=0.1))
            registry.reserve(['p'])
            registry.lock.release()
        finally:
            release.set()
            thread.join(2)
        self.assertIs(store['p'], cache)
        self.assertNotIn('p', registry.candidates)


try:
    import torch
    from src.cache_transfer import CacheTransferManager
    from src.cache_views import dynamic_cache_from_views
    from src.kvcache_varlen import require_full_attention, supports_varlen_model
    from transformers import LlamaConfig, LlamaForCausalLM
    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False


@unittest.skipUnless(CUDA, 'requires CUDA')
class TransferIsolationTests(unittest.TestCase):
    def host(self, n=8192):
        data = torch.arange(n, dtype=torch.float32).reshape(1, 1, n, 1).pin_memory()
        return dynamic_cache_from_views([(data, data + 1)])

    def test_partial_ticket_is_claimed_once_and_cancelled_storage_drains(self):
        transfers = CacheTransferManager('cuda:0', mode='overlap', chunk_bytes=1024)
        compute = torch.cuda.Stream()
        try:
            host = self.host()
            ticket = transfers.submit_h2d(host, purpose='prefetch')
            self.assertIsNone(ticket.copy_done)
            transfers.advance_prefetch(ticket)
            self.assertIsNone(ticket.copy_done)
            ticket.claimed_by = 'job'
            lease = transfers.bind_to_consumer(ticket, compute_stream=compute)
            compute.synchronize()
            torch.testing.assert_close(lease.cache.to_legacy_cache()[0][0].cpu(), host.to_legacy_cache()[0][0])
            self.assertEqual(transfers.metrics['copies'], 1)
            self.assertEqual(ticket.submitted_bytes, ticket.bytes_reserved)
            transfers.release_consumer(lease, compute_stream=compute)
            compute.synchronize()
            transfers.collect()
            self.assertEqual(transfers.speculative_bytes, 0)
            self.assertEqual(transfers.metrics['prefetch_wasted_bytes'], 0)
        finally:
            transfers.close()

    def test_collector_cannot_drop_a_demand_before_its_first_submission(self):
        transfers = CacheTransferManager('cuda:0')
        try:
            # Exercise the prepare/publish -> submit scheduling gap directly.
            cache = self.host(64)
            ticket = transfers._prepare(CacheSnapshot.capture(cache), torch.device('cuda:0'), 'demand')
            transfers.poll_completed()
            self.assertIn(ticket, transfers.pending)
            self.assertTrue(ticket.chunks)
            lease = transfers.bind_to_consumer(ticket, compute_stream=torch.cuda.current_stream())
            torch.testing.assert_close(lease.cache.to_legacy_cache()[0][0].cpu(), cache.to_legacy_cache()[0][0])
        finally:
            transfers.close()

    def test_measured_overlap_disturbance_falls_back_after_three_observations(self):
        transfers = CacheTransferManager('cuda:0', mode='overlap', disturbance_budget=0.1)
        try:
            transfers.observe_forward(('verify', 128, 4), 10, speculative=False)
            for _ in range(3):
                transfers.observe_forward(('verify', 128, 4), 12, speculative=True)
            self.assertEqual(transfers.mode, 'protect_forward')
            self.assertEqual(transfers.metrics['overlap_fallbacks'], 1)
        finally:
            transfers.close()

    def test_prefetch_b_does_not_add_dependency_to_compute_a(self):
        transfers = CacheTransferManager('cuda:0', mode='overlap', chunk_bytes=1024)
        compute = torch.cuda.Stream()
        try:
            with torch.cuda.stream(compute):
                a = torch.ones(64, device='cuda')
                a.add_(0)
            compute.synchronize()
            ticket = transfers.submit_h2d(self.host(), purpose='prefetch')
            with torch.cuda.stream(transfers.prefetch_stream):
                torch.cuda._sleep(100_000_000)
            transfers.advance_prefetch(ticket)
            with torch.cuda.stream(compute):
                a.add_(1)
            compute.synchronize()
            self.assertFalse(ticket.last_copy.query(), 'A unexpectedly waited for B')
            transfers.retire(ticket)
            transfers.collect()
            self.assertIn(ticket, transfers.pending)
            self.assertIsNotNone(ticket.source_lease)
            ticket.last_copy.synchronize()
            self.assertEqual(a.sum().item(), 128)
            transfers.collect()
            self.assertNotIn(ticket, transfers.pending)
        finally:
            transfers.close()

    def test_protect_forward_drains_only_submitted_chunk_and_blocks_more_dma(self):
        transfers = CacheTransferManager('cuda:0', chunk_bytes=1024)
        compute = torch.cuda.Stream()
        try:
            ticket = transfers.submit_h2d(self.host(), purpose='prefetch')
            with torch.cuda.stream(transfers.prefetch_stream):
                torch.cuda._sleep(20_000_000)
            transfers.advance_prefetch(ticket)
            transfers.admit()
            delay = transfers.begin_execution()
            self.assertTrue(ticket.last_copy.query())
            self.assertGreater(delay, 0)
            chunks = ticket.next_chunk
            transfers.advance_prefetch(ticket)
            self.assertEqual(ticket.next_chunk, chunks)
            self.assertIsNone(transfers.submit_h2d(self.host(64), purpose='prefetch'))
            with torch.cuda.stream(compute):
                torch.cuda._sleep(20_000_000)
            transfers.end_execution(compute_stream=compute)
            transfers.release_admission()
            transfers.advance_prefetch(ticket)
            self.assertEqual(ticket.next_chunk, chunks)
        finally:
            compute.synchronize()
            transfers.close()

    def test_budget_and_cpu_prepare_skip_speculation_but_demand_works(self):
        for kwargs in ({'prefetch_bytes': 1}, {'mode': 'cpu_prepare'}):
            transfers = CacheTransferManager('cuda:0', **kwargs)
            try:
                cache = self.host(64)
                self.assertIsNone(transfers.submit_h2d(cache, purpose='prefetch'))
                ticket = transfers.submit_h2d(cache)
                lease = transfers.bind_to_consumer(ticket, compute_stream=torch.cuda.current_stream())
                torch.testing.assert_close(lease.cache.to_legacy_cache()[0][0].cpu(), cache.to_legacy_cache()[0][0])
            finally:
                transfers.close()


class AttentionCompatibilityTests(unittest.TestCase):
    def test_dormant_window_is_allowed_actual_sliding_is_rejected(self):
        from tests.test_varlen_capability import VarlenCapabilityTests
        from src.kvcache_varlen import require_full_attention, supports_varlen_model
        model = VarlenCapabilityTests()._model('llama')
        model.config.sliding_window = 4096
        model.config.use_sliding_window = False
        self.assertTrue(supports_varlen_model(model))
        model.config.layer_types = ['sliding_attention']
        self.assertFalse(supports_varlen_model(model))
        with self.assertRaisesRegex(ValueError, 'sliding_attention'):
            require_full_attention(model)
        model.config.layer_types = ['full_attention']
        model.model.layers[0].self_attn.sliding_window = 128
        with self.assertRaises(ValueError):
            require_full_attention(model)


if __name__ == '__main__':
    unittest.main()
