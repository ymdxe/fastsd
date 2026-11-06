"""Regression checks for paper WRR prediction and real CUDA prefetch progress."""
from collections import defaultdict
from dataclasses import asdict
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from src.cache_protocol import CacheSnapshot, KVTransferTicket
from src.fastsd_scheduler import WorkItem, TaskLengthClassifier, plan_iteration
from src.scheduler_runtime import CacheResidency, KVPrefetchWorker, PrefetchCoordinator


class PaperSchedulingTests(unittest.TestCase):
    def item(self, pid, kind='verify', length=8, category='short', wait=0):
        return WorkItem(pid, pid, kind, category, total_tokens=length,
                        request={'task_type': kind, 'proc_id': pid, 'W_i': wait})

    def test_separate_draft_and_prompt_distributions(self):
        classifier = TaskLengthClassifier()
        for n in range(100):
            item = self.item(str(n), length=4 if n < 80 else 8)
            item.base_prefix_len = 2000
            classifier.observe(item)
        self.assertEqual(classifier.thresholds['verify'], (4, 8))
        self.assertEqual(classifier.thresholds['prefill'], (128, 512))
        self.assertEqual(classifier.category(self.item('v', length=8)), 'mid')
        self.assertEqual(classifier.observe(self.item('p', 'prefill', 300)), 'mid')

    def test_prediction_excludes_current_device_before_spending_slots(self):
        current = self.item('current', length=400, wait=100)
        future = self.item('future', wait=2)
        queues = {'verify': {'short': [current, future]}}
        before = [asdict(current), asdict(future)]
        plan = plan_iteration(queues, 16, max_num_seqs=1,
                              excluded_proc_ids=['current'], wrr_cursor=3)
        self.assertEqual(plan.verify_proc_ids, ['future'])
        self.assertEqual([asdict(current), asdict(future)], before)

    def test_wrr_priority_and_one_device_per_plan(self):
        short_low = self.item('a', wait=1)
        short_high = self.item('z', wait=10)
        mid = self.item('m', category='mid')
        long = self.item('l', category='long')
        duplicate = self.item('z', wait=8)
        duplicate.work_id = 'z-second'
        queues = {'verify': {'short': [short_low, short_high, duplicate], 'mid': [mid], 'long': [long]}}
        plan = plan_iteration(queues, 128, max_num_seqs=4, wrr_cursor=5)
        self.assertEqual(plan.verify_proc_ids, ['z', 'm', 'a', 'l'])
        self.assertEqual(len(set(plan.verify_proc_ids)), len(plan.verify_proc_ids))

    def test_coordinator_preserves_order_and_updates_existing_candidates(self):
        sent, available = [], set()
        coordinator = PrefetchCoordinator(submit=lambda p: sent.append(tuple(p)) or True,
            resident_snapshot=lambda: available, protected_snapshot=lambda: {'active'},
            pending_snapshot=lambda: True, metrics=defaultdict(int))
        coordinator.refresh(['z', 'active', 'a', 'z'])
        available.update(['a', 'z'])
        coordinator.refresh(['a', 'z'])
        self.assertEqual(sent, [('z', 'a'), ('a', 'z')])

    def test_worker_memory_budget_follows_prediction_not_pid(self):
        registry, prepared = CacheResidency(), []
        store = {pid: SimpleNamespace(get_seq_length=lambda: 4) for pid in ['a', 'z']}
        class OneTicketBudget:
            def submit_h2d(self, snapshot, purpose):
                if prepared:
                    return None
                prepared.append(snapshot.cache)
                return KVTransferTicket(snapshot, purpose, destination_cache=snapshot.cache,
                                        copy_state='ready')
            def poll_completed(self):
                return []
            def advance_prefetch(self, ticket):
                raise AssertionError('already ready')
        worker = KVPrefetchWorker(residency=registry, get_cache=store.get, set_cache=None,
            move_cache=None, gpu_device='cuda', metrics=defaultdict(int), transfers=OneTicketBudget())
        worker._latest_intent = ('z', 'a')
        worker._execute(('z', 'a'))
        self.assertEqual(list(registry.candidates), ['z'])
        self.assertIs(prepared[0], store['z'])


class PrefetchPollingTests(unittest.TestCase):
    def worker(self, transfers, clock):
        cache = SimpleNamespace(get_seq_length=lambda: 4)
        # The engine supplies an ordinary dict. Deliberately omit the new
        # retry counter: the worker owns its initialization, including callers
        # created before that counter was added.
        metrics = {name: 0 for name in (
            'prefetch_async_submitted', 'prefetch_async_dropped',
            'prefetch_async_completed', 'prefetch_async_moved',
            'prefetch_async_failed', 'prefetch_async_skipped_stale',
            'prefetch_async_superseded', 'prefetch_async_missing_cache',
            'prefetch_pending_underflow')}
        metrics['prefetch_async_last_error'] = None
        return KVPrefetchWorker(residency=CacheResidency(), get_cache=lambda _: cache,
            set_cache=None, move_cache=None, gpu_device='cuda', metrics=metrics,
            transfers=transfers, clock=clock)

    def test_plain_metrics_background_retry_has_no_worker_error(self):
        worker = self.worker(None, lambda: 10.0)
        worker._latest_intent = ('a',)
        worker._execute = lambda desired: 0
        worker._background_progress()
        self.assertIs(type(worker.metrics), dict)
        self.assertEqual(worker.metrics['prefetch_preparation_retries'], 1)
        self.assertEqual(worker.metrics['prefetch_async_failed'], 0)

    def test_existing_retry_metric_is_preserved(self):
        metrics = {'prefetch_preparation_retries': 17}
        KVPrefetchWorker(residency=CacheResidency(), get_cache=lambda _: None,
            set_cache=None, move_cache=None, gpu_device='cuda', metrics=metrics)
        self.assertEqual(metrics['prefetch_preparation_retries'], 17)

    def test_unchanged_budget_rejection_is_bounded_and_recovers(self):
        now = [10.0]
        class Transfers:
            attempts = 0
            available = False
            def poll_completed(self):
                return []
            def submit_h2d(self, snapshot, purpose):
                self.attempts += 1
                if not self.available:
                    return None
                return KVTransferTicket(snapshot, purpose, destination_cache=snapshot.cache,
                                        copy_state='ready')
        transfers = Transfers()
        worker = self.worker(transfers, lambda: now[0])
        worker._latest_intent = ('a',)
        worker._background_progress()
        now[0] += .01
        for _ in range(100):
            worker._background_progress()
        self.assertEqual(transfers.attempts, 1)
        self.assertGreater(worker._poll_timeout(), .03)
        transfers.available = True
        now[0] += .05
        worker._background_progress()
        self.assertEqual(transfers.attempts, 2)
        self.assertIn('a', worker.residency.candidates)

    def test_copy_progress_does_not_wait_for_preparation_retry(self):
        now = [10.0]
        class Transfers:
            advances = 0
            def poll_completed(self):
                return []
            def advance_prefetch(self, ticket):
                self.advances += 1
        transfers = Transfers()
        worker = self.worker(transfers, lambda: now[0])
        worker._latest_intent = ('a',)
        ticket = KVTransferTicket(CacheSnapshot.capture(worker._get_cache('a')), 'prefetch')
        worker.residency.candidates['a'] = ticket
        worker._next_preparation_retry_at = 20.0
        for _ in range(5):
            worker._background_progress()
        self.assertEqual(transfers.advances, 5)
        self.assertEqual(worker._poll_timeout(), .001)
        ticket.copy_state = 'ready'
        self.assertEqual(worker._poll_timeout(), .05)

    def test_new_intent_is_executed_despite_old_retry_deadline(self):
        worker = self.worker(None, lambda: 10.0)
        worker._next_preparation_retry_at = 100.0
        seen = []
        def execute(desired):
            seen.append(desired)
            worker.stop()
            return 0
        worker._execute = execute
        worker.submit(['new'])
        worker._run()
        self.assertEqual(seen, [('new',)])
        self.assertFalse(worker.has_pending())
        self.assertEqual(worker.metrics['prefetch_pending_underflow'], 0)


try:
    import torch
    from src.cache_views import dynamic_cache_from_views
    from src.cache_transfer import CacheTransferManager, _SlabArena
    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False


@unittest.skipUnless('CacheTransferManager' in globals(), 'requires PyTorch and transformers')
class PrefetchBudgetTests(unittest.TestCase):
    def test_full_local_budget_does_not_query_cuda(self):
        manager = CacheTransferManager('cpu', prefetch_bytes=64)
        manager.speculative_bytes = 48
        with patch('torch.cuda.mem_get_info', side_effect=AssertionError('unneeded driver query')):
            self.assertFalse(manager._reserve_speculation(32))
        self.assertEqual(manager.speculative_bytes, 48)
        self.assertEqual(manager.metrics['prefetch_local_budget_skips'], 1)

    def test_preallocated_reservation_is_atomic(self):
        manager = CacheTransferManager('cpu', prefetch_bytes=64, headroom_bytes=0)
        manager.gpu_arena = _SlabArena(torch.empty(64, dtype=torch.uint8))
        barrier, results = threading.Barrier(3), []
        def reserve():
            barrier.wait()
            results.append(manager._reserve_speculation(48))
        workers = [threading.Thread(target=reserve) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join()
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(manager.speculative_bytes, 48)


@unittest.skipUnless(CUDA, 'requires CUDA')
class PaperTransferTests(unittest.TestCase):
    def cache(self, layers=8):
        return dynamic_cache_from_views([
            (torch.full((1, 1, 8, 128), float(i), pin_memory=True),
             torch.full((1, 1, 8, 128), float(-i), pin_memory=True)) for i in range(layers)])

    def test_new_candidate_can_prepare_after_admission_and_execution(self):
        manager, compute = CacheTransferManager('cuda:0'), torch.cuda.Stream()
        source = self.cache()
        try:
            manager.install_probe('new-arrival', group_bytes=256 * 1024,
                rate_bytes_per_second=4 * 1024 ** 2,
                evidence={'purpose': 'bounded concurrency regression'})
            manager.admit()
            manager.begin_execution(shape='new-arrival')
            with torch.cuda.stream(compute):
                torch.cuda._sleep(30_000_000)
            ticket = manager.submit_h2d(source, purpose='prefetch')
            self.assertIsNotNone(ticket, 'B was denied because A was already executing')
            manager.advance_prefetch(ticket)
            self.assertGreater(manager.metrics['speculative_submitted_bytes'], 0)
            ticket.claimed_by = 'next-batch'
            lease = manager.bind_to_consumer(ticket, compute_stream=compute)
            compute.synchronize()
            for expected, actual in zip(source.to_legacy_cache(), lease.cache.to_legacy_cache()):
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b.cpu(), rtol=0, atol=0)
            manager.release_consumer(lease, compute_stream=compute)
            manager.end_execution(compute_stream=compute)
            manager.release_admission()
        finally:
            manager.close()

    def test_chunk_budget_coalesces_small_layers_without_exceeding_bytes(self):
        manager = CacheTransferManager('cuda:0', chunk_bytes=16 * 1024)
        try:
            ticket = manager.submit_h2d(self.cache(), purpose='prefetch')
            manager.advance_prefetch(ticket)
            self.assertEqual(ticket.next_chunk, 4)
            self.assertEqual(ticket.submitted_bytes, 16 * 1024)
            self.assertIsNone(ticket.copy_done)
            ticket.claimed_by = 'demand'
            stream = torch.cuda.current_stream()
            lease = manager.bind_to_consumer(ticket, compute_stream=stream)
            stream.synchronize()
            self.assertEqual(ticket.submitted_bytes, ticket.bytes_reserved)
            manager.release_consumer(lease, compute_stream=stream)
        finally:
            manager.close()
        self.assertEqual(manager.speculative_bytes, 0)

    def test_paper_overlap_never_silently_changes_method(self):
        manager = CacheTransferManager('cuda:0')
        try:
            manager.observe_forward('shape', 10, speculative=False)
            for _ in range(5):
                manager.observe_forward('shape', 20, speculative=True)
            self.assertEqual(manager.mode, 'overlap')
            self.assertEqual(manager.metrics['overlap_fallbacks'], 0)
            self.assertGreater(manager.metrics['forward_disturbance_ratio_max'], 0)
        finally:
            manager.close()


if __name__ == '__main__':
    unittest.main()
