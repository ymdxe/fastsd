import threading
import unittest
from types import SimpleNamespace

from src.cache_protocol import CacheSnapshot
from src.cache_residency import CPU_RESIDENT, EMERGENCY_FREE_BYTES
from src.direct_prefetch import DirectPrefetcher


class FakeEvent:
    def __init__(self, done=False):
        self.done_flag = done

    def query(self):
        return self.done_flag


class FakeCache:
    def __init__(self, length=4):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeTicket:
    def __init__(self, cache, done=False):
        self.destination_cache = cache
        self.copy_done = FakeEvent(done)
        self.copy_start = None
        self.submitted = False
        self.released = False


class FakeDriver:
    def __init__(self, *, free=2 * 1024 ** 3, auto_complete=True, nbytes=64):
        self.free = free
        self.auto_complete = auto_complete
        self.nbytes = nbytes
        self.tickets = []

    @staticmethod
    def source_ready(cache):
        return True

    def free_bytes(self):
        return self.free

    def describe(self, cache):
        return self.nbytes

    def allocate(self, pid, snapshot, nbytes, prepared_at):
        ticket = FakeTicket(FakeCache(snapshot.valid_len), self.auto_complete)
        self.tickets.append(ticket)
        return ticket

    def submit(self, ticket):
        ticket.submitted = True

    @staticmethod
    def ready(ticket):
        return ticket.copy_done.query()

    @staticmethod
    def copy_ms(ticket):
        return 1.0

    @staticmethod
    def consume(ticket, compute_stream):
        return ticket.destination_cache

    @staticmethod
    def release(ticket):
        ticket.released = True

    @staticmethod
    def finish(ticket):
        ticket.copy_done.done_flag = True

    @staticmethod
    def is_oom(error):
        return False


class FakeResidency:
    def __init__(self, cache):
        self.lock = threading.RLock()
        self.active = set()
        self.reserved = set()
        self.candidates = {}
        self.cache = cache
        self.version = 0

    def snapshot_locked(self, pid, cache):
        return CacheSnapshot.capture(cache, session_epoch=1, content_version=self.version)


class DirectPrefetchTests(unittest.TestCase):
    def make(self, *, free=2 * 1024 ** 3, auto_complete=True, nbytes=64):
        cache = FakeCache()
        residency = FakeResidency(cache)
        manager = SimpleNamespace(get=lambda pid: SimpleNamespace(
            state=CPU_RESIDENT, in_transfer=False,
            protected=lambda tick: False,
        ))
        metrics = {}
        driver = FakeDriver(free=free, auto_complete=auto_complete, nbytes=nbytes)
        worker = DirectPrefetcher(
            residency=residency, manager=manager,
            get_cache=lambda pid: cache, driver=driver, metrics=metrics,
            max_bytes=1024, reclaim=lambda needed: None,
        )
        snapshot = residency.snapshot_locked("p", cache)
        return worker, driver, residency, cache, snapshot, metrics

    def test_prepare_then_take_ready_entry(self):
        worker, driver, residency, cache, snapshot, metrics = self.make()
        self.assertTrue(worker.prepare("p", snapshot))
        entry = worker.take("p", snapshot)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.cache.length, cache.length)
        self.assertEqual(metrics["prefetch_ready_hit"], 1)
        worker.retire(entry)
        self.assertEqual(worker.stats()["reserved_bytes"], 0)

    def test_snapshot_mismatch_is_stale(self):
        worker, driver, residency, cache, snapshot, metrics = self.make()
        self.assertTrue(worker.prepare("p", snapshot))
        residency.version += 1
        current = residency.snapshot_locked("p", cache)
        self.assertIsNone(worker.take("p", current))
        self.assertEqual(metrics["prefetch_skipped_stale"], 1)

    def test_pending_take_never_waits(self):
        worker, driver, residency, cache, snapshot, metrics = self.make(auto_complete=False)
        self.assertTrue(worker.prepare("p", snapshot))
        self.assertIsNone(worker.take("p", snapshot))
        self.assertEqual(metrics["prefetch_pending_hit"], 1)
        driver.tickets[0].copy_done.done_flag = True
        worker.collect()
        self.assertEqual(worker.stats()["reserved_bytes"], 0)

    def test_free_memory_guard_skips(self):
        worker, driver, residency, cache, snapshot, metrics = self.make(
            free=EMERGENCY_FREE_BYTES + 63,
            nbytes=64,
        )
        self.assertFalse(worker.prepare("p", snapshot))
        self.assertEqual(metrics["prefetch_skipped_no_memory"], 1)

    def test_release_drains_and_does_not_leak(self):
        worker, driver, residency, cache, snapshot, metrics = self.make(auto_complete=False)
        self.assertTrue(worker.prepare("p", snapshot))
        worker.release("p")
        self.assertEqual(worker.stats()["prefetch_candidates"], 0)
        driver.tickets[0].copy_done.done_flag = True
        worker.collect()
        self.assertEqual(worker.stats()["reserved_bytes"], 0)
        self.assertTrue(driver.tickets[0].released)

    def test_demand_fallback_is_counted(self):
        worker, driver, residency, cache, snapshot, metrics = self.make()
        worker.record_demand_fallback()
        self.assertEqual(metrics["prefetch_demand_miss"], 1)
        self.assertEqual(metrics["prefetch_fallback_to_demand"], 1)


if __name__ == "__main__":
    unittest.main()

