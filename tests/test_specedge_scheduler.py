"""Deterministic ingress / execution boundaries. No threads, sleeps or model."""
import itertools
import unittest

from src.specedge_scheduler import WorkerInbox, run_dynamic_fifo


def task(pid, kind='verify', **fields):
    return dict(proc_id=pid, task_type=kind, **fields)


class DynamicFIFO(unittest.TestCase):
    def setUp(self):
        self.clock = itertools.count().__next__
        self.inbox = WorkerInbox(self.clock)
        self.seals, self.phases, self.errors, self.controls = [], [], [], []
        self.reserved = set()
        self.on_seal = self.on_execute = self.on_reserve = lambda: None

    def run_fifo(self, limit, *messages):
        for msg in messages:
            self.inbox.put(msg)

        def sealed(batch_id, rows, reason, at):
            self.seals.append(([r['proc_id'] for r in rows], reason))
            self.on_seal()

        def execute(rows):
            self.phases.append([r['proc_id'] for r in rows])
            self.assertTrue(self.reserved.issuperset(self.phases[-1]))
            self.on_execute()

        def reserve(pids):
            self.reserved.update(pids)
            self.on_reserve()

        run_dynamic_fifo(self.inbox, limit, admit=lambda m: m, execute=execute,
            reserve=reserve, release=self.reserved.difference_update,
            reject=lambda r, e: self.errors.append((r['proc_id'], e)), control=self.controls.append,
            maintenance=lambda: None, sealed=sealed, phase_done=lambda *args: None, clock=self.clock)
        self.assertFalse(self.reserved)

    def test_fifo_and_partial_shutdown(self):
        self.run_fifo(3, *(task(p, arrival_monotonic_s=100-i, priority=i, prefix_len=i+1)
                           for i, p in enumerate('ABCDE')), None)
        self.assertEqual(self.seals, [(['A','B','C'], 'full'), (['D','E'], 'shutdown')])
        self.assertEqual(self.phases, [['A','B','C'], ['D','E']])

    def test_arrivals_during_collection_then_during_forward(self):
        def during_collection():
            if self.reserved == {'A'}:
                self.inbox.put(task('B'))
        def during_forward():
            if len(self.phases) == 1:
                for msg in [task('C'), task('D'), None]:
                    self.inbox.put(msg)
        self.on_reserve, self.on_execute = during_collection, during_forward
        self.run_fifo(4, task('A'))
        self.assertEqual(self.seals, [(['A','B'], 'queue_empty'), (['C','D'], 'shutdown')])
        self.assertEqual(self.phases, [['A','B'], ['C','D']])

    def test_singleton_seals_immediately(self):
        self.on_seal = lambda: (self.inbox.put(task('B')), self.inbox.put(None)) if len(self.seals) == 1 else None
        self.run_fifo(4, task('A'))
        self.assertEqual(self.seals, [(['A'], 'queue_empty'), (['B'], 'shutdown')])

    def test_mixed_admission(self):
        self.run_fifo(4, task('A'), task('B','prefill'), task('C'), task('D','prefill'), task('E'), None)
        self.assertEqual(self.phases, [['B'], ['D'], ['A','C'], ['E']])

    def test_control_conflict_and_cancel_during_collection(self):
        self.run_fifo(3, task('A'), task('A'), dict(proc_id='A', control_type='prefill_cancel'),
                      task('B'), dict(proc_id='B', control_type='prefill_timing'), None)
        self.assertEqual(self.phases, [['B']]); self.assertEqual(len(self.errors), 2)
        self.assertEqual(len(self.controls), 2)

    def test_sealed_cancel_never_refills(self):
        def release_after_prefill():
            if self.phases == [['B']]:
                self.inbox.put(task('C'))
                self.inbox.put(dict(proc_id='A', control_type='session_release'))
                self.inbox.put(None)
        self.on_execute = release_after_prefill
        self.run_fifo(2, task('A'), task('B','prefill'))
        self.assertEqual(self.phases, [['B'], ['C']]); self.assertEqual(len(self.errors), 1)

    def test_fatal_forward_replies_and_drops_leases(self):
        def fail():
            raise RuntimeError('out of memory')
        self.on_execute = fail
        with self.assertRaisesRegex(RuntimeError, 'out of memory'):
            self.run_fifo(3, task('A'), task('B'), None)
        self.assertEqual(self.errors, [('A','out of memory'), ('B','out of memory')])
        self.assertFalse(self.reserved)

    def test_idle_sweep_distinguishes_ready_from_reserved(self):
        from src.session_lifecycle import SessionLifecycle
        from src.scheduler_runtime import CacheResidency
        memory = CacheResidency()
        cleaned = []
        life = SessionLifecycle(cleanup=cleaned.append, cancel=lambda pid: None,
            protected=memory.protected_snapshot, busy=self.inbox.busy, reply=lambda *args: None,
            idle_timeout=2, clock=self.clock)
        for pid in ('ready','selected','old'):
            life.touch(pid)
        self.inbox.put(task('ready'))
        memory.reserve(['selected'])
        for _ in range(4):
            self.clock()
        life.sweep()
        self.assertEqual(cleaned, ['old'])
        self.assertEqual(memory.protected_snapshot(), {'selected'})
        self.inbox.get_nowait()
        memory.release(['selected'])
        life.sweep()
        self.assertEqual(set(cleaned), {'ready','selected','old'})
