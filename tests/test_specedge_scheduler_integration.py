"""Run the production target with a seeded in-memory Llama, no weights download."""
from collections import defaultdict
from contextlib import ExitStack
import queue
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cloud.cloud_service import CloudTargetWorker
from src import engine
from src.util import parse_arguments


class TargetIntegration(unittest.TestCase):
    def test_real_varlen_batch_and_ar_parity(self):
        torch.manual_seed(42)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
        prompts = [[2,3,4], [7,6], [5,8,9,3]]
        proposals, expected = [], []
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                ids = list(prompt)
                for _ in range(3):
                    ids.append(int(model(torch.tensor([ids])).logits[0,-1].argmax()))
                suffix = ids[len(prompt):]
                accepted = i  # zero, partial, all (two submitted candidates)
                proposed = suffix[:2]
                if i < 2:
                    proposed[i] = (proposed[i] + 1) % 32
                proposals.append(proposed)
                expected.append((len(prompt) + accepted, suffix[i]))

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['test', '--server_sched_mode','specedge',
                '--exp_name', temp, '--batch_size','3', '--token_budget','1','--min_prefill_chunk_tokens','1',
                '--target_device','cpu','--kv_residency_budget_mb','1', '--target_dtype','float32']))
            args = parse_arguments()
            args.vocab_size = 32
            self.assertFalse(args.pipeline_gamma_adapt)
            worker = CloudTargetWorker(args)
            stack.enter_context(patch.object(worker, '_load_target_model_for_service', return_value=model))
            stack.enter_context(patch.object(worker, '_maybe_start_energy_service', return_value=None))
            for name in ['plan_iteration','TaskLengthClassifier','KVPrefetchWorker','H1Prefetcher','GammaController']:
                stack.enter_context(patch.object(engine, name, side_effect=AssertionError(name)))
            # Synchronous test publisher drives the production destination at exact RPC boundaries.
            inbox = None
            def publish(relay):
                nonlocal inbox
                inbox = relay.queue
                for i, prompt in enumerate(prompts):
                    inbox.put(dict(proc_id=i, task_type='prefill', draft_output=prompt, prefix_len=len(prompt)))
            stack.enter_context(patch.object(engine.IngressRelay, 'start', publish))
            stack.enter_context(patch.object(engine.IngressRelay, 'close'))
            batches = stack.enter_context(patch.object(engine, 'varlen_generate', wraps=engine.varlen_generate))
            replies = defaultdict(list)
            def reply(pid, payload):
                replies[pid].append(payload)
                if payload.get('prefill_ok') or payload.get('status') == 'prefill_ok':
                    inbox.put(dict(proc_id=pid, task_type='verify', tail_only=True, prefix_len=len(prompts[pid]),
                                   gamma=2, draft_output=proposals[pid]))
                    if pid == 2:
                        inbox.put(dict(proc_id='bad',task_type='verify',prefix_len=1,gamma=0,draft_output=[1]))
                        inbox.put(None)
            class Routes:
                def __getitem__(self, key):
                    return SimpleNamespace(put=lambda payload: reply(key, payload))
            worker.run_target_process_batching(SimpleNamespace(pad_token_id=0), queue.Queue(), Routes())
            self.assertEqual([list(call.args[3]) for call in batches.call_args_list], [[0],[1],[2],[0,1,2]])
            for i, pair in enumerate(expected):
                result = replies[i][-1]
                self.assertEqual((result['accepted'], result['final_token']), pair)
                self.assertNotIn('suggested_gamma', result)
            self.assertIn('error', replies['bad'][0])
            metrics = worker.last_generation_metrics['fastsd_scheduler']
            self.assertEqual(metrics['kv_residency']['victim_policy'], 'lru')
            self.assertEqual(metrics['query_tokens'], 15)
            self.assertEqual([metrics[k] for k in ['prefetch_candidates','lookahead_build_count','prefetch_bytes','cache_sessions']], [0]*4)
    def test_existing_schedulers(self):
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).eval()
        for mode in ('fastsd', 'vanilla', 'pipeline'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', ['test','--server_sched_mode',mode,
                    '--exp_name',temp,'--target_device','cpu','--no-enable_latency_priority','--no-kv_prefetch']))
                args = parse_arguments()
                args.vocab_size = 32
                worker = CloudTargetWorker(args)
                stack.enter_context(patch.object(worker,'_load_target_model_for_service',return_value=model))
                stack.enter_context(patch.object(worker,'_maybe_start_energy_service', return_value=None))
                planner = stack.enter_context(patch.object(engine,'plan_iteration',wraps=engine.plan_iteration))
                forwards = stack.enter_context(patch.object(engine,'varlen_generate',wraps=engine.varlen_generate))
                source, replies = queue.Queue(), defaultdict(queue.Queue)
                for pid in range(2):
                    source.put(dict(proc_id=pid, task_type='prefill', draft_output=[1,2],prefix_len=2))
                source.put(None)
                worker.run_target_process_batching(SimpleNamespace(pad_token_id=0), source, replies)
                self.assertEqual([replies[i].get_nowait()['status'] for i in range(2)], ['prefill_ok']*2)
                self.assertEqual(planner.called, mode == 'fastsd')
                if mode != 'fastsd':
                    self.assertEqual([list(call.args[3]) for call in forwards.call_args_list], [[0],[1]])

    def test_rejection_and_eos(self):
        from edge.edge import EdgeRunner
        from src.speculative_sampling import rejection_sample
        ids = torch.tensor([[0],[1]])
        probabilities = torch.ones((2,1))
        for accepted in range(3):
            rows = torch.eye(3)
            if accepted < 2:
                rows[accepted] = torch.tensor([0.,0.,1.])
            count, token = rejection_sample([0,1], rows, ids, probabilities,
                generator=torch.Generator().manual_seed(600 + accepted), uniforms=[0.5,0.5])
            self.assertEqual((count,token.item()), (accepted,2))
        truncated, hit = EdgeRunner._truncate_at_eos(torch.tensor([[2,5,2,4]]), 2, 1)
        self.assertTrue(hit)
        self.assertEqual(truncated.tolist(), [[2,5,2]])

class HttpSessionIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_replay_and_conflicting_session(self):
        import asyncio
        import json
        from cloud import cloud_service as cloud
        from fastapi import HTTPException
        from src.idempotency import IdempotencyCache

        published = asyncio.Event()
        messages = []
        def publish(message):
            messages.append(message)
            published.set()

        def request(seq):
            key = json.dumps(['POST','/prefill','session','task',seq], separators=(',',':'))
            return cloud.PrefillRequest(session_id='session', task_id='task', request_seq=seq,
                request_id=key, draft_output=[1,2], prefix_len=2, lag=0, current_time=0)

        with patch.object(cloud, 'request_queue', SimpleNamespace(put=publish)), \
                patch.object(cloud, '_idempotency_cache', IdempotencyCache()):
            first = asyncio.create_task(cloud.prefill(request(1)))
            await asyncio.wait_for(published.wait(), timeout=2)
            replay = asyncio.create_task(cloud.prefill(request(1)))
            with self.assertRaises(HTTPException) as conflict:
                await cloud.prefill(request(2))
            self.assertEqual(conflict.exception.status_code, 409)
            self.assertEqual(len(messages), 1)
            cloud._dispatch_worker_result(dict(request_id=messages[0]['response_key'], payload={'status':'prefill_ok'}))
            first_result, replay_result = await asyncio.gather(first, replay)
            self.assertFalse(first_result['replayed'])
            self.assertTrue(replay_result['replayed'])
            self.assertNotIn('session', cloud._pending_sessions)
