"""Model-free Target-only request, scheduling and streaming contract checks."""
import json
from pathlib import Path
import queue
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_backend import ARJob, AREngine, completion_reason, input_ids_digest, plan_batch
from ar_worker import consume_stream, metric_record, validate_canonical_rows, wait_scheduled


EOS = {151645, 151643}


def row(sample_id='c1-node1-w0-0', ids=None, arrival=2.0):
    ids = [4, 8, 15] if ids is None else ids
    return {'sample_id': sample_id, 'prompt': 'already rendered', 'input_token_ids': ids,
            'input_ids_sha256': input_ids_digest(ids), 'scheduled_arrival_s': arrival,
            'node': 'node1', 'worker': 0}


def stream(tokens, reason='eos', count=None, warmup=False):
    events = [{'token_id': token} for token in tokens]
    events.append({'done': True, 'reason': 'completed', 'stop_reason': reason,
                   'generated_tokens': len(tokens) if count is None else count,
                   'warmup': warmup})
    return [json.dumps(event) for event in events]


class BatchTests(unittest.TestCase):
    def test_real_token_and_row_budget_with_mixed_lengths(self):
        jobs = [ARJob(str(i), [1] * (20 + i * 41), 400) for i in range(32)]
        for job in jobs[:7]:
            job.prompt_position = len(job.input_ids)
            job.next_token = 123
        for budget in (1, 8, 16, 512):
            with self.subTest(budget=budget):
                plans = plan_batch(jobs, 16, budget)
                self.assertLessEqual(len(plans), 16)
                self.assertLessEqual(sum(length for _, _, length in plans), budget)
                self.assertEqual(len({job.session_id for job, _, _ in plans}), len(plans))
                for job, kind, length in plans:
                    self.assertGreater(length, 0)
                    self.assertLessEqual(length, 1 if kind == 'decode' else len(job.input_ids) - job.prompt_position)

    def test_prefill_rotation_and_cancel(self):
        jobs = [ARJob(str(i), [1] * 100, 400) for i in range(3)]
        jobs[0].cancelled.set()
        self.assertIs(plan_batch(jobs, 1, 7, 0)[0][0], jobs[1])
        self.assertIs(plan_batch(jobs, 1, 7, 1)[0][0], jobs[2])

    def test_decode_needs_previous_token(self):
        job = ARJob('x', [1], 400, prompt_position=1)
        with self.assertRaises(ValueError):
            plan_batch([job], 16, 512)

    def test_dual_eos_precedes_budget(self):
        for token in EOS:
            self.assertEqual(completion_reason(token, 400, 400, EOS), 'eos')
        self.assertEqual(completion_reason(4, 400, 400, EOS), 'max_tokens')
        self.assertIsNone(completion_reason(4, 399, 400, EOS))


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.row = row()
        self.engine = AREngine({'max_tokens': 400, 'trace_rows': [self.row]})
        self.engine.status = 'ok'
        self.engine.vocab_size = 160000

    def submit(self, **kwargs):
        options = {'sample_id': self.row['sample_id'], 'input_ids': self.row['input_token_ids'],
                   'max_tokens': 400, 'input_ids_sha256': self.row['input_ids_sha256']}
        options.update(kwargs)
        return self.engine.submit(**options)

    def test_exact_canonical_ids_hash_and_budget_enforced(self):
        for patch in ({'input_ids': [4, 8, 16]}, {'input_ids_sha256': 'bad'},
                      {'sample_id': 'unknown'}, {'max_tokens': 32}, {'max_tokens': True},
                      {'input_ids': [True]}, {'input_ids': [-1]}, {'input_ids': [160000]}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.submit(**patch)
        self.assertEqual(self.engine.stats['submitted'], 0)

    def test_warmup_and_formal_are_distinct_and_not_replayed(self):
        warm = self.submit(warmup=True, max_tokens=1)
        formal = self.submit()
        self.assertTrue(warm.warmup)
        self.assertFalse(formal.warmup)
        self.assertNotEqual(warm.session_id, formal.session_id)
        self.assertEqual(self.engine.stats['warmup_submitted'], 1)
        self.assertEqual(self.engine.stats['formal_submitted'], 1)
        with self.assertRaises(ValueError):
            self.submit()
        with self.assertRaises(ValueError):
            self.submit(warmup=True, max_tokens=2)

    def test_cancelled_and_failed_jobs_release_all_cache_state(self):
        class Manager:
            def __init__(self):
                self._past_key_values, self._prob_history = {}, {}

            def reset(self, session_id):
                self._past_key_values.pop(session_id, None)
                self._prob_history.pop(session_id, None)

        self.engine._manager = Manager()
        job = self.submit()
        self.engine._admit()
        self.engine._manager._past_key_values[job.session_id] = object()
        self.engine._manager._prob_history[job.session_id] = object()
        self.engine._finish(job, 'failed', 'simulated request failure')
        terminal = job.events.get_nowait()
        self.assertEqual(terminal['reason'], 'failed')
        self.assertEqual(self.engine.stats['active'], 0)
        self.assertEqual(self.engine.stats['cache_count'], 0)
        self.assertEqual(self.engine.stats['prob_history_count'], 0)
        self.assertEqual(self.engine.stats['failed'], 1)
        with self.assertRaises(queue.Empty):
            job.events.get_nowait()

    def test_pending_cancellation_is_completed_once(self):
        job = self.submit()
        job.cancelled.set()
        self.engine._admit()
        self.assertEqual(self.engine.stats['pending'], 0)
        self.assertEqual(self.engine.stats['cancelled'], 1)
        self.assertEqual(job.events.get_nowait()['reason'], 'cancelled')
        self.assertFalse(self.engine._active)


class StreamTests(unittest.TestCase):
    def parse(self, lines, max_tokens=400, **kwargs):
        return consume_stream(lines, max_tokens=max_tokens, eos_token_ids=EOS,
                              warmup=False, vocab_size=160000, **kwargs)

    def test_both_eos_preserved_with_exact_count(self):
        for eos in EOS:
            result = self.parse(stream([4, eos]))
            self.assertEqual(result['output_token_ids'], [4, eos])
            self.assertEqual(result['generated_tokens'], 2)
            self.assertEqual(result['stop_reason'], 'eos')

    def test_output_budget_without_eos(self):
        result = self.parse(stream([4, 5], reason='max_tokens'), max_tokens=2)
        self.assertEqual(result['stop_reason'], 'max_tokens')

    def test_malformed_incomplete_and_overrun_streams_fail(self):
        invalid = [[], [json.dumps({'token_id': 4})], stream([4], reason='max_tokens'),
                   stream([4, 151645], count=1), stream([151645, 4]),
                   stream([4, 5, 6], reason='max_tokens'),
                   stream([True, 151645]), stream([-1, 151645]), stream([160000, 151645]),
                   ['{invalid'], ['[]'], stream([151645]) + ['{}'],
                   [json.dumps({'done': True, 'reason': 'failed', 'error': 'OOM'})],
                   stream([151645], warmup=True), stream([151645], reason='max_tokens')]
        for lines in invalid:
            with self.subTest(lines=lines), self.assertRaises(RuntimeError):
                self.parse(lines, max_tokens=2)

    def test_total_deadline_enforced(self):
        with self.assertRaises(TimeoutError):
            self.parse(stream([151645]), deadline=5, now=lambda: 6)


class SchedulingTests(unittest.TestCase):
    def test_no_delay_added_for_fifo_backlog(self):
        waits = []
        wait_scheduled(100, 20, now=lambda: 150, sleep=waits.append)
        self.assertEqual(waits, [])

    def test_fixed_schedule_waits_only_until_original_arrival(self):
        clock = [102.0]
        waits = []

        def sleep(seconds):
            waits.append(seconds)
            clock[0] += seconds

        wait_scheduled(100, 4.5, now=lambda: clock[0], sleep=sleep)
        self.assertEqual(clock[0], 104.5)
        self.assertEqual(sum(waits), 2.5)

    def test_canonical_order_count_and_hash(self):
        rows = [row('one', arrival=2), row('two', arrival=9)]
        validate_canonical_rows(rows, 2)
        for invalid, count in ((rows, 1), (rows[::-1], 2), ([rows[0], rows[0]], 2),
                               ([{**rows[0], 'input_ids_sha256': 'wrong'}], 1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_canonical_rows(invalid, count)

    def test_metric_timing_uses_client_visibility_and_frozen_origin(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return 'output'

        result = {'request_start_monotonic_s': 110.0, 'completion_monotonic_s': 114.1,
                  'token_visibility_monotonic_s': [112.0, 114.0], 'generated_tokens': 2,
                  'output_token_ids': [4, 151645], 'stop_reason': 'eos'}
        info = {'run_id': 'run', 'bundle_id': 'bundle', 'trace_sha256': 'hash'}
        record = metric_record(row(arrival=2), result, 100, info, Tokenizer())
        self.assertEqual(record['arrival_lag_ms'], 8000)
        self.assertEqual(record['ttft_ms'], 2000)
        self.assertEqual(record['tpot_ms'], 2000)
        self.assertAlmostEqual(record['request_e2e_ms'], 4100)
        self.assertEqual(record['token_visibility_s'], [12, 14])
        self.assertFalse(record['client_model_loaded'])
        self.assertEqual(record['input_ids_sha256'], row()['input_ids_sha256'])


if __name__ == '__main__':
    unittest.main()
