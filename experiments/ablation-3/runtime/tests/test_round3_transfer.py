"""Resource-control tests run without a CUDA driver; CUDA tests use real storage."""
import importlib.util
from pathlib import Path
import sys
import json
import io
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from src.cache_protocol import CacheSnapshot, KVTransferTicket


try:
    from src.cache_transfer import CacheTransferManager, OverlapController, _SlabArena
except ImportError:
    # Only dependency boundaries are stubbed. Every method under test is loaded
    # from the production module, rather than copying its control logic.
    fake_torch = ModuleType('torch')
    fake_torch.device = lambda device: SimpleNamespace(type=str(device).split(':')[0])
    fake_views = ModuleType('src.cache_views')
    fake_views.dynamic_cache_from_views = lambda views: None
    spec = importlib.util.spec_from_file_location('src._round3_transfer_nocuda',
        Path(__file__).resolve().parents[1] / 'src' / 'cache_transfer.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': fake_torch, 'src.cache_views': fake_views}):
        spec.loader.exec_module(module)
    CacheTransferManager, OverlapController, _SlabArena = (
        module.CacheTransferManager, module.OverlapController, module._SlabArena)


class FakeEvent:
    def __init__(self, ready=False):
        self.done = ready

    def query(self):
        return self.done


class FakeTensor:
    def __init__(self, count, *, offset=0, base=4096):
        self.count, self.offset, self.base = count, offset, base
        self.dtype, self.shape = 'uint8', (count,)

    def numel(self):
        return self.count

    def element_size(self):
        return 1

    def data_ptr(self):
        return self.base + self.offset

    def stride(self):
        return (1,)

    def narrow(self, dim, offset, count):
        return FakeTensor(count, offset=self.offset + offset, base=self.base)

    def view(self, shape_or_dtype):
        return self


class ControllerTests(unittest.TestCase):
    def controller(self):
        now = [0.0]
        return OverlapController(1024, 4096, 16384, clock=lambda: now[0]), now

    def profile(self, controller, shape='shape', samples=None):
        return controller.install_calibration(shape,
            baseline_ms=[10] * 8, sample_ms=samples or [10] * 8,
            group_bytes=512, rate_bytes_per_second=4096,
            evidence={'paired_replay': True, 'output_parity': True})

    def test_unknown_context_pauses_only_compute_window(self):
        controller, now = self.controller()
        controller.context_for(('verify', 1, (8,), (2048,), 'bfloat16', 'sdpa'))
        self.assertEqual(controller.allowance(computing=True), 0)
        self.assertEqual(controller.status()['reason'], 'uncalibrated_context')
        now[0] += 1
        self.assertEqual(controller.allowance(computing=False), 1024)

    def test_group_and_rate_limits_apply_independently(self):
        controller, now = self.controller()
        self.assertTrue(self.profile(controller))
        controller.context_for('shape')
        self.assertEqual(controller.allowance(computing=True), 512)
        controller.consume(512)
        self.assertEqual(controller.allowance(computing=True), 0)
        now[0] += .03125
        self.assertEqual(controller.allowance(computing=True), 128)
        now[0] += 1
        self.assertEqual(controller.allowance(computing=True), 512)

    def test_demand_and_d2h_pause_new_dma_and_record_reasons(self):
        controller, _ = self.controller()
        self.profile(controller)
        controller.context_for('shape')
        for reason in ('demand_h2d_pending', 'required_d2h_pending'):
            self.assertEqual(controller.allowance(computing=True, blocked_reason=reason), 0)
            self.assertEqual(controller.status()['reason'], reason)
        transitions = controller.status()['transitions']
        self.assertEqual([row['reason'] for row in transitions],
                         ['demand_h2d_pending', 'required_d2h_pending'])
        self.assertTrue(all('duration_s' in row and 'group_bytes' in row for row in transitions))

    def test_failed_calibration_stays_paused_and_requires_real_samples(self):
        controller, _ = self.controller()
        self.assertFalse(self.profile(controller, samples=[10.3] * 8))
        controller.context_for('shape')
        self.assertEqual(controller.allowance(computing=True), 0)
        self.assertEqual(controller.status()['reason'], 'paired_replay_failed')
        with self.assertRaises(ValueError):
            controller.install_calibration('empty', baseline_ms=[], sample_ms=[],
                group_bytes=512, rate_bytes_per_second=4096, evidence={})

    def test_explicit_probe_then_runtime_throttle_and_pause(self):
        controller, now = self.controller()
        controller.install_probe('probe', group_bytes=256, rate_bytes_per_second=1024,
                                 evidence={'purpose': 'paired calibration'})
        controller.context_for('probe')
        self.assertEqual(controller.allowance(computing=True), 256)
        self.assertEqual(controller.status()['state'], 'probe')
        self.profile(controller, shape='shape')
        controller.context_for('shape')
        for _ in range(3):
            controller.observe('shape', 11, speculative=True)
        now[0] += 1
        self.assertEqual(controller.allowance(computing=True), 256)
        self.assertEqual(controller.status()['state'], 'throttled')
        for _ in range(3):
            controller.observe('shape', 11, speculative=True)
        self.assertEqual(controller.allowance(computing=True), 0)
        self.assertEqual(controller.status()['state'], 'paused')

    def test_shape_schema_includes_backend_dtype_and_context(self):
        controller, _ = self.controller()
        shape = ('verify', 1, (8,), (2048,), 'bfloat16', 'sdpa')
        self.profile(controller, shape=shape)
        controller.context_for(list(shape))
        self.assertGreater(controller.allowance(computing=True), 0)
        controller.context_for(('verify', 1, (8,), (4096,), 'bfloat16', 'sdpa'))
        self.assertEqual(controller.allowance(computing=True), 0)

    def test_new_admission_revokes_previous_phase_context(self):
        manager = CacheTransferManager('cpu')
        manager.install_calibration('old-phase', baseline_ms=[10] * 8, sample_ms=[10] * 8,
            group_bytes=256 * 1024, rate_bytes_per_second=4 * 1024 ** 2,
            evidence={'paired_replay': True})
        manager.set_forward_context('old-phase')
        self.assertGreater(manager.controller.allowance(computing=True), 0)
        manager.admit()
        self.assertEqual(manager.controller.allowance(computing=bool(manager.admitted)), 0)
        self.assertEqual(manager.controller_status()['reason'], 'uncalibrated_context')
        manager.begin_execution(shape='old-phase')
        self.assertEqual(manager.controller.context, 'old-phase')
        manager.release_admission()

    def test_public_disable_shape_records_replay_rejection(self):
        manager = CacheTransferManager('cpu')
        manager.install_probe('shape', evidence={'test': 'probe'})
        manager.disable_shape('shape', reason='paired_confidence_interval_inconclusive')
        manager.set_forward_context('shape')
        self.assertEqual(manager.controller.allowance(computing=True), 0)
        self.assertEqual(manager.controller_status()['reason'], 'paired_confidence_interval_inconclusive')
        self.assertEqual(manager.requested_mode, 'overlap')

    def test_pageable_speculation_is_deferred_without_staging_or_idle_race(self):
        for admitted, executing in ((0, 0), (1, 0), (1, 1)):
            with self.subTest(admitted=admitted, executing=executing):
                manager = CacheTransferManager('cpu', prefetch_bytes=512)
                manager.enabled, manager.admitted, manager.executing = True, admitted, executing
                source = FakeTensor(128)
                source.device = SimpleNamespace(type='cpu')
                source.is_pinned = lambda: False
                source.copy_ = lambda other: self.fail('pageable source was staged')
                cache = SimpleNamespace(to_legacy_cache=lambda: [(source, source)])
                with patch.object(manager, '_reserve_speculation',
                                  side_effect=AssertionError('pageable source reserved GPU memory')):
                    self.assertIsNone(manager._prepare(CacheSnapshot(cache),
                        SimpleNamespace(type='cuda'), 'prefetch'))
                self.assertEqual(manager.metrics['prefetch_deferred_cpu_staging'], 1)
                self.assertEqual(manager.speculative_bytes, 0)


class CalibrationContextTests(unittest.TestCase):
    def context(self):
        return dict(gpu_uuid='GPU-device-a', model='/models/target',
                    dtype='torch.bfloat16', attention_backend='sdpa_per_sequence',
                    forward_primitive='varlen_generate', kv_batch_mode='varlen')

    def artifact(self):
        return dict(context=self.context(), profiles=[dict(shape='shape',
            baseline_ms=[10] * 8, sample_ms=[10] * 8,
            group_bytes=256 * 1024, rate_bytes_per_second=4 * 1024 ** 2,
            evidence={'paired_replay': True, 'output_parity': True})])

    def load(self, manager, artifact, **kwargs):
        with patch('builtins.open', return_value=io.StringIO(json.dumps(artifact))):
            return manager.load_calibration('calibration.json', **kwargs)

    def test_matching_hardware_model_and_primitive_context_is_accepted(self):
        manager = CacheTransferManager('cpu')
        self.assertEqual(self.load(manager, self.artifact(), context=self.context()), 1)
        manager.set_forward_context('shape')
        self.assertGreater(manager.controller.allowance(computing=True), 0)
        self.assertEqual(manager.controller_status()['calibration_context'], self.context())

    def test_each_mismatched_context_field_rejects_before_installing(self):
        for key in self.context():
            with self.subTest(field=key):
                manager = CacheTransferManager('cpu')
                expected = self.context()
                expected[key] += '-different'
                with self.assertRaisesRegex(ValueError, key):
                    self.load(manager, self.artifact(), context=expected)
                self.assertFalse(manager.controller.profiles)

    def test_runtime_context_and_saved_context_are_required(self):
        manager = CacheTransferManager('cpu')
        with self.assertRaisesRegex(ValueError, 'runtime calibration context'):
            self.load(manager, self.artifact())
        artifact = self.artifact()
        del artifact['context']['gpu_uuid']
        with self.assertRaisesRegex(ValueError, 'saved calibration context'):
            self.load(manager, artifact, context=self.context())
        self.assertFalse(manager.controller.profiles)

    def test_invalid_later_profile_does_not_partially_authorize_file(self):
        manager = CacheTransferManager('cpu')
        artifact = self.artifact()
        invalid = dict(artifact['profiles'][0], shape='invalid', baseline_ms=[])
        artifact['profiles'].append(invalid)
        with self.assertRaises(ValueError):
            self.load(manager, artifact, context=self.context())
        self.assertFalse(manager.controller.profiles)

    def test_gpu_uuid_case_does_not_change_device_identity(self):
        manager = CacheTransferManager('cpu')
        expected = self.context()
        expected['gpu_uuid'] = expected['gpu_uuid'].lower()
        self.assertEqual(self.load(manager, self.artifact(), context=expected), 1)


class SlabLeaseTests(unittest.TestCase):
    def test_dma_consumer_and_registry_all_protect_same_storage(self):
        arena = _SlabArena(FakeTensor(512))
        lease, _ = arena.acquire([FakeTensor(512)])
        dma, consumer = FakeEvent(), FakeEvent()
        lease.events[:] = [dma, consumer]
        lease.owners.add('registry-a')
        lease.retired = True
        self.assertEqual(arena.acquire([FakeTensor(512)]), (None, None))
        dma.done = True
        self.assertEqual(arena.acquire([FakeTensor(512)]), (None, None))
        consumer.done = True
        self.assertEqual(arena.acquire([FakeTensor(512)]), (None, None))
        lease.owners.clear()
        recycled, _ = arena.acquire([FakeTensor(512)])
        self.assertIsNotNone(recycled)
        self.assertEqual(recycled.offset, lease.offset)

    def test_storage_alias_wrapper_retains_lease_and_its_final_event(self):
        manager = CacheTransferManager('cpu', prefetch_bytes=512)
        manager.gpu_arena = _SlabArena(FakeTensor(512))
        lease, views = manager.gpu_arena.acquire([FakeTensor(512)])
        original = SimpleNamespace(to_legacy_cache=lambda: [(views[0], views[0])])
        alias_tensor = views[0].narrow(0, 32, 64)
        alias = SimpleNamespace(to_legacy_cache=lambda: [(alias_tensor, alias_tensor)])
        manager.retain_cache(original, owner='original')
        manager.retain_cache(alias, owner='alias')
        event_released = []
        lease.on_recycle = lambda: event_released.append(True)
        lease.retired = True
        manager.release_cache(owner='original')
        self.assertEqual(manager.gpu_arena.acquire([FakeTensor(512)]), (None, None))
        self.assertFalse(event_released)
        manager.release_cache(owner='alias')
        self.assertIsNotNone(manager.gpu_arena.acquire([FakeTensor(512)])[0])
        self.assertEqual(event_released, [True])

    def test_replace_registry_owner_releases_old_storage(self):
        manager = CacheTransferManager('cpu', prefetch_bytes=512)
        manager.gpu_arena = _SlabArena(FakeTensor(512))
        lease, views = manager.gpu_arena.acquire([FakeTensor(256)])
        manager.retain_cache(SimpleNamespace(to_legacy_cache=lambda: [(views[0], views[0])]), owner='pid')
        lease.retired = True
        external = FakeTensor(256, base=8192)
        manager.retain_cache(SimpleNamespace(to_legacy_cache=lambda: [(external, external)]), owner='pid')
        manager.gpu_arena.collect()
        self.assertFalse(manager.gpu_arena.leases)

    def test_active_consumer_keeps_aliased_storage_after_registry_release(self):
        manager = CacheTransferManager('cpu', prefetch_bytes=512)
        manager.gpu_arena = _SlabArena(FakeTensor(512))
        storage, views = manager.gpu_arena.acquire([FakeTensor(512)])
        cache = SimpleNamespace(to_legacy_cache=lambda: [(views[0], views[0])])
        manager.retain_cache(cache, owner='registry')
        storage.retired = True
        ticket = KVTransferTicket(CacheSnapshot(cache), 'demand',
                                  destination_cache=cache, copy_state='ready')
        ticket._source_storage_leases = [storage]
        consumer = manager.bind_to_consumer(ticket, compute_stream=None)
        manager.release_cache(owner='registry')
        manager.gpu_arena.collect()
        self.assertIn(storage, manager.gpu_arena.leases)
        manager.release_consumer(consumer, compute_stream=None)
        manager.gpu_arena.collect()
        self.assertNotIn(storage, manager.gpu_arena.leases)

    def test_retirement_publishes_event_recycle_callback_atomically(self):
        manager = CacheTransferManager('cpu', prefetch_bytes=512)
        manager.gpu_arena = _SlabArena(FakeTensor(512))
        storage, views = manager.gpu_arena.acquire([FakeTensor(512)])
        storage.owners.add('registry')
        cache = SimpleNamespace(to_legacy_cache=lambda: [(views[0], views[0])])
        ticket = KVTransferTicket(CacheSnapshot(cache), 'prefetch',
                                  destination_cache=cache, copy_state='invalid-draining',
                                  retired=True, bytes_reserved=512)
        final_event = FakeEvent(ready=True)
        ticket._storage_lease, ticket._pooled_events = storage, [final_event]
        manager.pending, manager.speculative_bytes = [ticket], 512
        def guard(target, name, value):
            if target is storage and name == 'retired' and value:
                self.assertTrue(manager.lock._is_owned())
                self.assertIsNotNone(storage.on_recycle)
            object.__setattr__(target, name, value)
        with patch.object(type(storage), '__setattr__', new=guard):
            manager.poll_completed()
        self.assertFalse(manager._events)
        storage.owners.clear()
        manager.poll_completed()
        self.assertEqual(manager._events, [final_event])

    def test_source_write_protection_fences_alias_and_claimed_remainder(self):
        manager = CacheTransferManager('cpu')
        tensor = FakeTensor(512)
        alias = SimpleNamespace(to_legacy_cache=lambda: [(tensor, tensor)])
        waited, advanced = [], []
        stream = SimpleNamespace(wait_event=lambda event: waited.append(event))
        for claimed in (None, 'consumer'):
            with self.subTest(claimed=claimed):
                event = FakeEvent()
                ticket = KVTransferTicket(CacheSnapshot(object()), 'prefetch',
                    last_copy=event, claimed_by=claimed)
                ticket._source_ranges = [(4096, 4608)]
                manager.pending = [ticket]
                with patch.object(manager, '_advance',
                    side_effect=lambda value, **kw: advanced.append((value, kw))):
                    manager.protect_source_write(alias, producer_stream=stream)
                self.assertIs(waited[-1], event)
                if claimed is None:
                    self.assertTrue(ticket.retired)
                else:
                    self.assertFalse(ticket.retired)
                    self.assertEqual(advanced[-1], (ticket, {'all_chunks': True}))


try:
    import torch
    from src.cache_views import dynamic_cache_from_views
    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False


@unittest.skipUnless(CUDA, 'requires CUDA')
class CudaSlabTests(unittest.TestCase):
    def source(self, tokens=1024):
        return dynamic_cache_from_views([(
            torch.arange(tokens * 128, dtype=torch.float32, pin_memory=True).reshape(1, 1, tokens, 128),
            torch.ones((1, 1, tokens, 128), dtype=torch.float32, pin_memory=True))])

    def test_active_prepare_and_copy_use_preallocated_resources_and_same_ticket(self):
        manager = CacheTransferManager('cuda:0', prefetch_bytes=4 * 1024 ** 2,
                                       chunk_bytes=256 * 1024, headroom_bytes=0)
        source, compute = self.source(), torch.cuda.Stream()
        manager.install_probe('verify-probe', group_bytes=256 * 1024,
            rate_bytes_per_second=64 * 1024 ** 2, evidence={'test': 'allocation-free speculative window'})
        try:
            manager.begin_execution(shape='verify-probe')
            with patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('driver query in forward')), \
                 patch.object(torch, 'empty', side_effect=AssertionError('allocation in forward')), \
                 patch.object(torch, 'empty_like', side_effect=AssertionError('allocation in forward')), \
                 patch.object(torch.cuda, 'Event', side_effect=AssertionError('event allocation in speculative window')):
                ticket = manager.submit_h2d(source, purpose='prefetch')
                self.assertIsNotNone(ticket)
                manager.advance_prefetch(ticket)
            self.assertGreater(ticket.submitted_bytes, 0)
            already_submitted = ticket.submitted_bytes
            ticket.claimed_by = 'same-ticket-demand'
            lease = manager.bind_to_consumer(ticket, compute_stream=compute)
            self.assertIs(lease.ticket, ticket)
            self.assertEqual(ticket.submitted_bytes, ticket.bytes_reserved)
            self.assertGreater(ticket.submitted_bytes, already_submitted)
            manager.retain_cache(lease.cache, owner='formal-registry')
            compute.synchronize()
            for expected, actual in zip(source.to_legacy_cache(), lease.cache.to_legacy_cache()):
                for cpu, gpu in zip(expected, actual):
                    torch.testing.assert_close(cpu, gpu.cpu(), rtol=0, atol=0)
            storage = ticket._storage_lease
            manager.release_consumer(lease, compute_stream=compute)
            compute.synchronize()
            manager.collect()
            self.assertIn(storage, manager.gpu_arena.leases)
            manager.release_cache(owner='formal-registry')
            manager.collect()
            self.assertNotIn(storage, manager.gpu_arena.leases)
        finally:
            manager.end_execution(compute_stream=compute)
            manager.close()

    def test_unknown_shape_permits_preparation_but_pauses_new_compute_dma(self):
        manager = CacheTransferManager('cuda:0', prefetch_bytes=4 * 1024 ** 2, headroom_bytes=0)
        compute = torch.cuda.Stream()
        try:
            manager.begin_execution(shape=('verify', 'uncalibrated'))
            ticket = manager.submit_h2d(self.source(), purpose='prefetch')
            self.assertIsNotNone(ticket)
            manager.advance_prefetch(ticket)
            self.assertEqual(ticket.submitted_bytes, 0)
            self.assertEqual(manager.controller_status()['reason'], 'uncalibrated_context')
            self.assertEqual(manager.requested_mode, 'overlap')
        finally:
            manager.end_execution(compute_stream=compute)
            manager.close()

    def test_d2h_source_holds_gpu_slab_after_registry_replacement(self):
        manager = CacheTransferManager('cuda:0', prefetch_bytes=4 * 1024 ** 2, headroom_bytes=0)
        compute = torch.cuda.Stream()
        try:
            ticket = manager.submit_h2d(self.source(), purpose='prefetch')
            lease = manager.bind_to_consumer(ticket, compute_stream=compute)
            manager.retain_cache(lease.cache, owner='registry')
            compute.synchronize()
            manager.release_consumer(lease, compute_stream=compute)
            with torch.cuda.stream(manager.d2h_stream):
                torch.cuda._sleep(50_000_000)
            offload = manager.submit_d2h(lease.cache)
            storage = ticket._storage_lease
            self.assertIn(storage, offload._source_storage_leases)
            self.assertIn(('dma-source', id(offload)), storage.owners)
            manager.retain_cache(offload.destination_cache, owner='registry')
            self.assertIn(('dma-source', id(offload)), storage.owners)
            if not offload.last_copy.query():
                manager.collect()
                self.assertIn(storage, manager.gpu_arena.leases)
            manager.d2h_stream.synchronize()
            manager.collect()
            self.assertNotIn(storage, manager.gpu_arena.leases)
        finally:
            manager.close()


if __name__ == '__main__':
    unittest.main()
