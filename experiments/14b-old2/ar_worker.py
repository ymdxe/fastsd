"""CPU-only sequential sender for the Target-only canonical-input baseline."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from ar_backend import completion_reason, input_ids_digest
from common import trace_hash, valid_id, write_json
from experiment_hooks import finish_worker, rpc, wait_for_start


def validate_canonical_rows(rows, expected_count):
    if len(rows) != expected_count or not rows:
        raise ValueError('Canonical assignment has an incorrect request count')
    seen = set()
    previous = -1.0
    for row in rows:
        sample_id = row.get('sample_id')
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError('Missing or duplicate canonical sample ID')
        seen.add(sample_id)
        if input_ids_digest(row['input_token_ids']) != row['input_ids_sha256']:
            raise ValueError('Canonical input token ID hash mismatch')
        if not isinstance(row.get('prompt'), str):
            raise ValueError('Canonical rendered prompt is missing')
        arrival = row.get('scheduled_arrival_s')
        if isinstance(arrival, bool) or not isinstance(arrival, (int, float)) or not math.isfinite(arrival) or arrival < 0 or arrival < previous:
            raise ValueError('Canonical arrivals must be finite, nonnegative and ordered')
        previous = arrival


def wait_scheduled(origin, scheduled_arrival_s, *, now=time.monotonic, sleep=time.sleep):
    """Use the frozen arrival clock; finished requests never shift future times."""
    deadline = origin + scheduled_arrival_s
    while True:
        remaining = deadline - now()
        if remaining <= 0:
            return
        sleep(min(remaining, 1.0))


def consume_stream(lines, *, max_tokens, eos_token_ids, warmup, now=time.monotonic,
                   deadline=None, vocab_size=None):
    """Require an explicit successful terminal event and an exact EOS/budget stop."""
    token_ids, visible = [], []
    terminal = None
    for raw in lines:
        if deadline is not None and now() > deadline:
            raise TimeoutError('Target-only request exceeded its total deadline')
        if not raw:
            continue
        if terminal is not None:
            raise RuntimeError('Target-only stream contains data after its terminal event')
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError('Target-only stream contains invalid JSON') from exc
        if not isinstance(event, dict):
            raise RuntimeError('Target-only stream event must be an object')
        if event.get('done') is True:
            if 'token_id' in event:
                raise RuntimeError('Target-only terminal event also contains a token')
            if event.get('error') or event.get('reason') != 'completed':
                raise RuntimeError('Target-only request failed: ' + str(event))
            count = event.get('generated_tokens')
            if isinstance(count, bool) or not isinstance(count, int) or count != len(token_ids):
                raise RuntimeError('Target-only terminal token count differs from stream')
            if not token_ids:
                raise RuntimeError('Target-only stream ended without an output token')
            expected_reason = completion_reason(token_ids[-1], len(token_ids), max_tokens, eos_token_ids)
            if expected_reason is None or event.get('stop_reason') != expected_reason:
                raise RuntimeError('Target-only stream ended before EOS/output budget or with a wrong reason')
            if event.get('warmup') is not warmup:
                raise RuntimeError('Target-only response warmup flag differs from request')
            terminal = event
            continue
        if 'done' in event or 'error' in event or 'token_id' not in event:
            raise RuntimeError('Unexpected Target-only stream event: ' + str(event))
        token = event['token_id']
        if isinstance(token, bool) or not isinstance(token, int) or token < 0 or (vocab_size is not None and token >= vocab_size):
            raise RuntimeError('Target-only stream contains an invalid token ID')
        if len(token_ids) >= max_tokens or (token_ids and token_ids[-1] in eos_token_ids):
            raise RuntimeError('Target-only stream contains tokens after EOS/output budget')
        token_ids.append(token)
        visible.append(now())
    if terminal is None:
        raise RuntimeError('Target-only stream ended without a terminal event')
    return {'output_token_ids': token_ids, 'token_visibility_monotonic_s': visible,
            'stop_reason': terminal['stop_reason'], 'generated_tokens': len(token_ids),
            'completion_monotonic_s': now()}


def generate(session, url, row, max_tokens, eos_token_ids, timeout, *, warmup=False,
             vocab_size=None):
    started = time.monotonic()
    from measurements import trace,set_task
    set_task(row['sample_id'])
    trace.emit('ar_wire_send',task_id=row['sample_id'],warmup=warmup)
    if input_ids_digest(row['input_token_ids']) != row['input_ids_sha256']:
        raise RuntimeError('Canonical input changed immediately before transmission')
    payload = {'sample_id': row['sample_id'], 'input_token_ids': row['input_token_ids'],
               'input_ids_sha256': row['input_ids_sha256'], 'max_tokens': max_tokens,
               'warmup': warmup}
    with session.post(url.rstrip('/') + '/ar/generate', json=payload, stream=True,
                      timeout=(min(30.0, timeout), timeout)) as response:
        response.raise_for_status()
        result = consume_stream(response.iter_lines(chunk_size=1), max_tokens=max_tokens,
                                eos_token_ids=eos_token_ids, warmup=warmup,
                                deadline=started + timeout, vocab_size=vocab_size)
    result['request_start_monotonic_s'] = started
    trace.emit('ar_wire_complete',task_id=row['sample_id'],warmup=warmup,
        start_ns=int(started*1e9),end_ns=int(result['completion_monotonic_s']*1e9))
    return result


def metric_record(row, result, origin, info, tokenizer):
    started = result['request_start_monotonic_s']
    finished = result['completion_monotonic_s']
    visibility = result['token_visibility_monotonic_s']
    first, last = visibility[0], visibility[-1]
    count = result['generated_tokens']
    arrival = row['scheduled_arrival_s']
    return {
        **row, 'task_id': row['sample_id'], 'method': 'target_only', 'success': True,
        'run_id': info['run_id'], 'bundle_id': info['bundle_id'],
        'trace_sha256': info['trace_sha256'],
        'generated_tokens': count, 'output_token_ids': result['output_token_ids'],
        'output_text': tokenizer.decode(result['output_token_ids'], skip_special_tokens=True),
        'stop_reason': result['stop_reason'], 'truncated': result['stop_reason'] == 'max_tokens',
        'warmup': False, 'warmup_requests_excluded': 1,
        'actual_arrival_s': started - origin,
        'arrival_lag_ms': max(0.0, (started - origin - arrival) * 1000),
        'processing_start_s': started - origin, 'latency_basis': 'request_processing_start',
        'task_e2e_ms': (finished - started) * 1000,
        'request_e2e_ms': (finished - started) * 1000,
        'ttft_ms': (first - started) * 1000,
        'tpot_ms': (last - first) * 1000 / (count - 1) if count > 1 else None,
        'completion_s': finished - origin,
        'request_start_monotonic_s': started,
        'first_token_monotonic_s': first, 'completion_monotonic_s': finished,
        'experiment_t0_monotonic_s': origin,
        'token_visibility_s': [stamp - origin for stamp in visibility],
        'accepted_total': 0, 'drafted_total': 0, 'rounds': 0,
        'decode_output_tokens': count,
        'input_token_count': len(row['input_token_ids']),
        'input_ids_sha256_verified': True, 'client_model_loaded': False,
    }


def run(options):
    from suite import verify_bundle
    bundle_id = verify_bundle()
    run_id = valid_id(os.environ['FASTSD_SUITE_RUN_ID'])
    identity = os.environ['FASTSD_SUITE_WORKER']
    node = platform.node().split('.')[0]
    if identity != f'{node}-w{options.worker_id}':
        raise RuntimeError('Worker identity does not match this host/worker ID')
    info = rpc(options.server_url, '/experiment/info')
    if options.run_info and json.loads(Path(options.run_info).read_text(encoding='utf-8')) != info:
        raise RuntimeError('Local run info differs from the live Target configuration')
    if info['run_id'] != run_id or info['bundle_id'] != bundle_id:
        raise RuntimeError('Target run ID or bundle differs from this worker')
    if options.method != 'target_only' or info['method'] != options.method:
        raise RuntimeError('Target method differs from this Target-only worker')
    if bundle_id != os.environ['FASTSD_SUITE_BUNDLE_ID']:
        raise RuntimeError('Launcher bundle differs from this worker')
    if info['trace_sha256'] != os.environ['FASTSD_SUITE_TRACE_SHA256']:
        raise RuntimeError('Launcher trace differs from Target')
    if trace_hash(info['trace_rows']) != info['trace_sha256']:
        raise RuntimeError('Target trace hash does not match the supplied rows')
    node_config = info['nodes'][node]
    if options.gamma != int(node_config['gamma']) or int(node_config['threads']) != 6:
        raise RuntimeError('Worker gamma/thread settings differ from requested configuration')
    if not 0 <= options.worker_id < int(node_config['workers']):
        raise RuntimeError('Unexpected worker number')
    affinity = [int(value) for value in options.affinity.split(',')]
    if len(affinity) != 6 or len(set(affinity)) != 6:
        raise RuntimeError('Expected exactly six distinct logical CPUs')
    if not set(affinity).issubset(os.sched_getaffinity(0)):
        raise RuntimeError('Assigned CPUs are outside this process allowed affinity')
    os.sched_setaffinity(0, affinity)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU sender must be launched with CUDA_VISIBLE_DEVICES empty')
    output = Path(options.exp_name).resolve()
    if not output.is_dir():
        raise RuntimeError('Launcher must create the dedicated worker output directory')
    if (output / 'worker-status.json').exists() or list(output.glob('edge_metrics_proc*.jsonl')):
        raise FileExistsError('Worker results already exist; use a new run ID')
    rows = [json.loads(line) for line in Path(options.inputs).read_text(encoding='utf-8').splitlines() if line.strip()]
    expected = [row for row in info['trace_rows'] if row['node'] == node and int(row['worker']) == options.worker_id]
    if rows != expected:
        raise RuntimeError('Worker inputs differ from the full frozen assignment')
    validate_canonical_rows(rows, int(info['worker_request_counts'][identity]))
    timeout = float(info['request_timeout'])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Request timeout must be finite and positive')
    eos_token_ids = frozenset(info['eos_token_ids'])
    status = {'worker': identity, 'run_id': run_id, 'state': 'loading',
              'method': 'target_only', 'started_epoch_s': time.time()}
    os.environ['FASTSD_DIAGNOSTIC_DIR']=str(output/'diagnostics')
    write_json(output / 'worker-status.json', status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('worker signal ' + str(signum))

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        import requests
        import transformers
        from transformers import AutoTokenizer
        started = time.monotonic()
        tokenizer_path = node_config['tokenizer_model']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        target_state = rpc(options.server_url, '/ar/stats')
        if not target_state.get('model_ready') or target_state.get('method') != 'target_only':
            raise RuntimeError('Target-only engine is not ready')
        vocab_size = int(target_state['vocab_size'])
        if frozenset(target_state['eos_token_ids']) != eos_token_ids:
            raise RuntimeError('Target-only engine EOS settings differ from run configuration')
        loaded_torch = sys.modules.get('torch')
        cuda_initialized = bool(loaded_torch and loaded_torch.cuda.is_initialized())
        write_json(output / 'actual-model.json', {
            'method': 'target_only', 'model_loaded': False, 'parameter_count': 0,
            'tokenizer_model': tokenizer_path, 'tokenizer_use': 'output decoding only',
            'tokenizer_load_s': time.monotonic() - started,
            'transformers_version': transformers.__version__,
            'python_version': platform.python_version(), 'affinity': sorted(os.sched_getaffinity(0)),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_initialized': cuda_initialized, 'epoch_s': time.time(),
        })
        if cuda_initialized:
            raise RuntimeError('CPU sender unexpectedly initialized CUDA')
        if tokenizer.eos_token_id not in eos_token_ids:
            raise RuntimeError('CPU tokenizer EOS is outside configured EOS IDs')
        write_json(output / 'runtime-arguments.json', {
            'resolved_parameters': vars(options), 'max_tokens': info['max_tokens'],
            'eos_token_ids': sorted(eos_token_ids), 'request_timeout': timeout,
            'max_inflight_per_sender': 1, 'arrival_source': 'frozen canonical trace',
            'rate_per_worker': info['rate_per_worker'], 'tasks_per_worker': len(rows),
            'draft_model_loaded': False,
        })
        with requests.Session() as session:
            session.trust_env = False
            status['state'] = 'warmup'
            write_json(output / 'worker-status.json', status)
            warmup = generate(session, options.server_url, rows[0], 1, eos_token_ids,
                              timeout, warmup=True, vocab_size=vocab_size)
            write_json(output / 'warmup.json', {'sample_id': rows[0]['sample_id'],
                       'warmup_requests': 1, 'max_tokens': 1, 'excluded_from_formal_metrics': True,
                       'input_ids_sha256': rows[0]['input_ids_sha256'], **warmup})
            status['state'] = 'ready'
            write_json(output / 'worker-status.json', status)
            origin = wait_for_start(options)
            status.update(state='running', local_t0_monotonic_s=origin, completed_requests=0)
            write_json(output / 'worker-status.json', status)
            actual_ids = []
            with (output / 'edge_metrics_proc0.jsonl').open('x', encoding='utf-8') as metrics:
                for row in rows:
                    wait_scheduled(origin, row['scheduled_arrival_s'])
                    from conversations import resolve_sample
                    processing_started=time.monotonic()
                    row=resolve_sample(row,options.server_url,output)
                    result = generate(session, options.server_url, row, int(info['max_tokens']),
                                      eos_token_ids, timeout, vocab_size=vocab_size)
                    result["request_start_monotonic_s"]=processing_started
                    record = metric_record(row, result, origin, info, tokenizer)
                    metrics.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                    metrics.flush()
                    actual_ids.append(row['sample_id'])
                    status['completed_requests'] = len(actual_ids)
                    write_json(output / 'worker-status.json', status)
            if actual_ids != [row['sample_id'] for row in expected]:
                raise RuntimeError('Incomplete or duplicate worker outputs')
        finish_worker(options)
        status.update(state='complete', requests=len(actual_ids))
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        from diagnostic_trace import trace
        trace.close()
        status['finished_epoch_s'] = time.time()
        write_json(output / 'worker-status.json', status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--method', required=True, choices=('target_only',))
    parser.add_argument('--gamma', required=True, type=int)
    parser.add_argument('--affinity', required=True)
    parser.add_argument('--worker-id', required=True, type=int)
    parser.add_argument('--run-info')
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
