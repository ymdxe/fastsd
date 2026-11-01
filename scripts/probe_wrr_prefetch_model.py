"""Real-model transport/forward probe; not a throughput or HumanEval score."""
import hashlib
import threading
import time


def run_probe(model_path):
    import torch
    from transformers import AutoModelForCausalLM
    from diagnostic_trace import trace
    from src.cache_transfer import CacheTransferManager
    from src.fastsd_scheduler import WorkItem, plan_iteration
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import varlen_generate
    from src.scheduler_runtime import CacheResidency, KVPrefetchWorker

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    compute = torch.cuda.Stream()
    transfers = CacheTransferManager('cuda:0', mode='overlap', record_timeline=True)
    registry = CacheResidency()
    # Use the engine's plain-dict contract, so missing metric initialization
    # cannot be hidden by a defaultdict during validation.
    metrics = {name: 0 for name in (
        'prefetch_async_submitted', 'prefetch_async_dropped',
        'prefetch_async_completed', 'prefetch_async_moved',
        'prefetch_async_failed', 'prefetch_async_skipped_stale',
        'prefetch_async_superseded', 'prefetch_async_missing_cache',
        'prefetch_pending_underflow')}
    metrics['prefetch_async_last_error'] = None
    worker_errors = []
    def manager():
        result = KVCacheModel_batching(model, temperature=0, attention_backend='sdpa_per_sequence')
        result.vocab_size = model.config.vocab_size
        result.bounded_probabilities = True
        return result
    reference, actual, active = manager(), manager(), manager()
    prefix = torch.arange(128, device='cuda:0').remainder(model.config.vocab_size).reshape(1, -1)
    tail = torch.arange(128, 136, device='cuda:0').reshape(1, -1)
    with torch.no_grad(), torch.cuda.stream(compute):
        varlen_generate(reference, model, [prefix], ['future'], 0, is_prefill=True, sample_next=False)
        source = reference._past_key_values['future']
        transfers.mark_computed(source, producer_stream=compute)
        offload = transfers.submit_d2h(source)
        offload.copy_done.synchronize()
        host = offload.destination_cache
        actual._prob_history['future'] = reference._prob_history['future'].clone()
        actual._prob_start['future'] = reference._prob_start['future']
        expected_tokens = varlen_generate(reference, model, [tail], ['future'], 0, sample_next=False)[0]
        expected_prob = reference._prob_history['future'].clone()
        expected_kv = [(k.clone(), v.clone()) for k, v in reference._past_key_values['future'].to_legacy_cache()]
        pids = ['active-' + str(i) for i in range(16)]
        varlen_generate(active, model, [prefix[:, :64]] * 16, pids, 0,
                        is_prefill=True, sample_next=False)
    compute.synchronize()
    store = {'future': host}
    worker = KVPrefetchWorker(residency=registry, get_cache=store.get, set_cache=None,
        move_cache=transfers.move, transfers=transfers, gpu_device='cuda:0', metrics=metrics,
        thread_setup=lambda: torch.cuda.set_device(0), on_error=worker_errors.append)
    worker.start()
    registry.reserve(pids)
    registry.activate(pids)
    transfers.admit()
    transfers.begin_execution()
    item = WorkItem('future', 'future', 'verify', 'short', total_tokens=8,
                    request={'task_type': 'verify', 'proc_id': 'future'})
    plan = plan_iteration({'verify': {'short': [item]}}, 512, max_num_seqs=16,
                          excluded_proc_ids=pids, selection_policy='legacy')
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        with torch.no_grad(), torch.cuda.stream(compute):
            start.record(compute)
            worker.submit(plan.verify_proc_ids)
            varlen_generate(active, model, [tail] * 16, pids, 0, sample_next=False)
            end.record(compute)
        end.synchronize()
        trace.gpu_span('model_probe_forward', start, end, batch_size=16, gamma=8,
                       attention_backend='sdpa_per_sequence')
        deadline = time.monotonic() + 30
        ticket = None
        while time.monotonic() < deadline:
            with registry.lock:
                ticket = registry.candidates.get('future')
            if worker_errors:
                raise AssertionError('Production prefetch worker failed: ' + repr(worker_errors))
            if ticket is not None and ticket.ready() and metrics.get('prefetch_preparation_retries', 0) > 0:
                break
            time.sleep(.001)
        if ticket is None or not ticket.ready():
            raise AssertionError('Production worker did not complete the predicted KV')
        worker.stop()
        worker.join(timeout=5)
        if worker.is_alive() or worker_errors or metrics['prefetch_async_failed']:
            raise AssertionError('Prefetch worker did not finish cleanly: ' + repr(metrics))
        if metrics['prefetch_preparation_retries'] <= 0:
            raise AssertionError('Plain-dict background retry path was not exercised')
        ticket_id = id(ticket)
        with registry.lock:
            registry.reserve(['future'])
            ticket = registry.claim_locked('future', host, 'probe-next-batch')
        lease = transfers.bind_to_consumer(ticket, compute_stream=compute)
        actual._past_key_values['future'] = lease.cache
        with torch.no_grad(), torch.cuda.stream(compute):
            actual_tokens = varlen_generate(actual, model, [tail], ['future'], 0, sample_next=False)[0]
        compute.synchronize()
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)
        # Last eight rows are the identical target verification computation.
        torch.testing.assert_close(actual._prob_history['future'][:, -8:], expected_prob[:, -8:], rtol=0, atol=0)
        for expected, observed in zip(expected_kv, actual._past_key_values['future'].to_legacy_cache()):
            for a, b in zip(expected, observed):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        transfers.release_consumer(lease, compute_stream=compute)
        return {'model': model_path, 'dtype': 'bfloat16', 'attention_backend': 'sdpa_per_sequence',
                'probe_batch_size': 16, 'probe_gamma': 8, 'prefetch_ticket_id': ticket_id,
                'target_forward_ms': start.elapsed_time(end), 'worker_metrics': dict(metrics),
                'transfer_metrics': dict(transfers.metrics), 'exact_probability_and_kv_match': True,
                'probe_scope': 'One controlled real-model batch; full campaign validates actual traffic separately.'}
    finally:
        worker.stop()
        worker.join()
        transfers.end_execution(compute_stream=compute)
        transfers.release_admission()
        transfers.close()
