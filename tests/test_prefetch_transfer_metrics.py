"""Actual byte submission and completion accounting without a CUDA driver."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.cache_protocol import CacheSnapshot, KVTransferTicket
from tests import test_round3_transfer as transfer_checks


CacheTransferManager = transfer_checks.CacheTransferManager


class Event(transfer_checks.FakeEvent):
    def record(self, stream):
        pass


class TransferMetricTests(unittest.TestCase):
    def ticket(self, **kwargs):
        return KVTransferTicket(CacheSnapshot(object()), "prefetch", bytes_reserved=128,
            submitted_bytes=128, prepared_at=7, copy_started_at=8,
            transfer_horizon=2, horizon=1, **kwargs)

    def clock(self):
        return patch.dict(CacheTransferManager._poll_completed.__globals__,
                          time=SimpleNamespace(perf_counter=lambda: 10.0))

    def test_ready_ticket_is_reported_once_with_bytes_time_and_rate(self):
        manager = CacheTransferManager("cpu")
        ticket = self.ticket(copy_state="copying", copy_done=Event(True))
        manager.pending.append(ticket)
        with self.clock():
            manager.poll_completed()
            manager.poll_completed()
        self.assertEqual(manager.metrics["kv_transfer_samples"], 1)
        self.assertEqual(ticket.transfer_elapsed_s, 2)
        self.assertEqual(ticket.transfer_rate_bytes_per_s, 64)
        bucket = manager.metrics["kv_transfer_by_horizon"]["2"]
        self.assertEqual((bucket["tickets"], bucket["bytes"], bucket["avg_s"]), (1, 128, 2))
        detail = manager.metrics["kv_transfer_recent_tickets"][0]
        self.assertEqual((detail["horizon"], detail["preparation_to_observed_s"]), (2, 3))

    def test_cancelled_partial_dma_is_measured_only_after_drain(self):
        manager = CacheTransferManager("cpu")
        event = Event(False)
        ticket = self.ticket(copy_state="copying", last_copy=event)
        manager.pending.append(ticket)
        manager.speculative_bytes = 128
        manager.retire(ticket)
        with self.clock():
            manager.poll_completed()
            self.assertFalse(ticket.reported)
            event.done = True
            manager.poll_completed()
            manager.poll_completed()
        bucket = manager.metrics["kv_transfer_by_horizon"]["2"]
        self.assertEqual((bucket["tickets"], bucket["retired"], bucket["completed"]), (1, 1, 0))
        self.assertEqual(bucket["bytes"], 128)
        self.assertEqual(manager.metrics["kv_transfer_samples"], 0)
        self.assertEqual(manager.speculative_bytes, 0)

    def test_consumer_retirement_does_not_lose_completed_transfer_timing(self):
        manager = CacheTransferManager("cpu")
        event = Event(True)
        ticket = self.ticket(copy_state="copying", copy_done=event, last_copy=event, claimed_by="job")
        manager.pending.append(ticket)
        manager.speculative_bytes = 128
        manager.retire(ticket)
        with self.clock():
            manager.poll_completed()
        self.assertEqual(manager.metrics["kv_transfer_h2d_avg_s"], 2)
        self.assertEqual(manager.metrics["kv_transfer_by_horizon"]["2"]["retired"], 0)

    def test_ticket_detail_history_is_bounded_while_totals_keep_accumulating(self):
        manager = CacheTransferManager("cpu")
        with self.clock():
            for _ in range(260):
                manager._record_transfer(self.ticket(copy_state="ready"))
        self.assertEqual(len(manager.metrics["kv_transfer_recent_tickets"]), 256)
        self.assertEqual(manager.metrics["kv_transfer_ticket_details_dropped"], 4)
        self.assertEqual(manager.metrics["kv_transfer_by_horizon"]["2"]["tickets"], 260)

    def test_no_dma_ticket_does_not_create_a_transfer_sample(self):
        manager = CacheTransferManager("cpu")
        ticket = KVTransferTicket(CacheSnapshot(object()), "prefetch", copy_state="ready")
        manager.pending.append(ticket)
        with self.clock():
            manager.poll_completed()
        self.assertEqual(manager.metrics["kv_transfer_samples"], 0)
        self.assertEqual(manager.metrics["kv_transfer_recent_tickets"], [])

    def test_actual_submitted_bytes_keep_birth_horizon_after_demand_claim(self):
        manager = CacheTransferManager("cpu")
        copies = []
        source = SimpleNamespace(numel=lambda: 64, element_size=lambda: 1, is_cuda=False)
        target = SimpleNamespace(copy_=lambda src, **kw: copies.append(src), is_cuda=True,
                                 record_stream=lambda stream: None)
        stream = SimpleNamespace(wait_event=lambda event: None)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext(),
                                                         Event=lambda **kw: Event(True)))
        ticket = KVTransferTicket(CacheSnapshot(object()), "prefetch", bytes_reserved=128,
            destination_cache=SimpleNamespace(), chunks=[(source, target), (source, target)],
            horizon=1, transfer_horizon=2, claimed_by="job")
        with patch.object(manager, "_stream", return_value=stream), patch.dict(
                CacheTransferManager._advance.__globals__, torch=fake_torch):
            manager._advance(ticket, all_chunks=True)
        self.assertEqual(len(copies), 2)
        self.assertEqual(manager.metrics["h2d_bytes"], 128)
        self.assertEqual(manager.metrics["prefetch_bytes_by_horizon"], {"2": 128})
        self.assertEqual(manager.metrics["speculative_bytes_by_horizon"], {})
        self.assertEqual(ticket.submitted_bytes, 128)

    def test_online_rate_honors_both_configured_limits(self):
        controller = transfer_checks.OverlapController(1024, 128, 1024, online_fraction=.5)
        self.assertEqual(controller.online_rate, 128)
        controller = transfer_checks.OverlapController(1024, 4096, 1024, online_fraction=.25)
        self.assertEqual(controller.online_rate, 256)


if __name__ == "__main__":
    unittest.main()
