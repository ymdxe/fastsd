"""Target-only greedy streaming with bounded variable-length batches."""
import asyncio
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import sys
import threading
import time
import traceback
import uuid


@dataclass
class ARJob:
    sample_id: str
    input_ids: list
    max_tokens: int
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    prompt_position: int = 0
    next_token: object = None
    emitted: int = 0
    warmup: bool = False


def input_ids_digest(input_ids):
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError('input_token_ids must be a nonempty list')
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
           for token in input_ids):
        raise ValueError('input_token_ids contains an invalid token')
    return hashlib.sha256(json.dumps(input_ids, separators=(',', ':')).encode('utf-8')).hexdigest()


def completion_reason(token, emitted, max_tokens, eos_token_ids):
    if token in eos_token_ids:
        return 'eos'
    if emitted >= max_tokens:
        return 'max_tokens'
    return None


def plan_batch(jobs, batch_size, token_budget, cursor=0):
    """Plan one mixed forward; each row appears once and budgets count real tokens.

    Decode rows take one token each. Remaining rows share available prefill
    capacity, so a long prompt cannot repeatedly monopolize the entire budget.
    The helper is independent of Torch and can be checked without loading a model.
    """
    if batch_size < 1 or token_budget < 1:
        raise ValueError('batch_size and token_budget must be positive')
    live = [job for job in jobs if not job.cancelled.is_set()]
    decode = [job for job in live if job.prompt_position == len(job.input_ids)]
    prefill = [job for job in live if job.prompt_position < len(job.input_ids)]
    if prefill:
        shift = cursor % len(prefill)
        prefill = prefill[shift:] + prefill[:shift]
    plans = []
    remaining = int(token_budget)
    for job in decode[:min(batch_size, remaining)]:
        if job.next_token is None:
            raise ValueError('decode row has no previous output token')
        plans.append((job, 'decode', 1))
        remaining -= 1
    rows = min(batch_size - len(plans), len(prefill), remaining)
    for index, job in enumerate(prefill[:rows]):
        length = min(len(job.input_ids) - job.prompt_position,
                     max(1, remaining // (rows - index)))
        plans.append((job, 'prefill', length))
        remaining -= length
    return plans


class AREngine:
    def __init__(self, config):
        self.config = dict(config)
        self.batch_size = int(config.get('batch_size', 16))
        self.token_budget = int(config.get('token_budget', 512))
        self.max_tokens = int(config.get('max_tokens', 400))
        if min(self.batch_size, self.token_budget, self.max_tokens) < 1:
            raise ValueError('AR budgets must be positive')
        self.pending = queue.Queue()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.status = 'not_started'
        self.error = None
        self.stats = dict(pending=0, active=0, peak_active=0, cache_count=0,
                          prob_history_count=0, peak_cache_count=0,
                          submitted=0, completed=0, failed=0, cancelled=0,
                          emitted_tokens=0, batches=0, prefill_tokens=0,
                          decode_tokens=0, max_batch_rows=0,
                          max_batch_input_tokens=0, last_batch=None,
                          warmup_submitted=0, warmup_completed=0,
                          formal_submitted=0, formal_completed=0)
        self.eos_token_ids = frozenset(config.get('eos_token_ids', [151645, 151643]))
        if not self.eos_token_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                         for x in self.eos_token_ids):
            raise ValueError('eos_token_ids must contain nonnegative integers')
        self._expected = {row['sample_id']: row for row in config.get('trace_rows', [])}
        self._seen = set()
        self.vocab_size = None
        self._manager = None
        self._torch = None
        self._active = {}
        self._log = None

    def start(self):
        with self.lock:
            if self.thread is not None:
                return
            self.status = 'loading'
            self.thread = threading.Thread(target=self._run, name='ar-target', daemon=True)
            self.thread.start()

    def stop(self, timeout=30):
        self.stopping.set()
        self.pending.put(None)
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return {'thread_alive': bool(thread is not None and thread.is_alive())}

    def snapshot(self):
        with self.lock:
            return dict(self.stats, status=self.status, error=self.error,
                        model_ready=self.status == 'ok',
                        worker_alive=bool(self.thread and self.thread.is_alive()),
                        method='target_only', kv_batch_mode='varlen',
                        kv_policy='CPU offload after every mixed forward; no prefetch',
                        batch_size=self.batch_size, token_budget=self.token_budget,
                        vocab_size=self.vocab_size, eos_token_ids=sorted(self.eos_token_ids))

    def submit(self, sample_id, input_ids, max_tokens, input_ids_sha256=None, warmup=False):
        actual_hash = input_ids_digest(input_ids)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError('sample_id must be a nonempty string')
        if input_ids_sha256 is not None and actual_hash != input_ids_sha256:
            raise ValueError('input_ids_sha256 does not match supplied token IDs')
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.max_tokens:
            raise ValueError('max_tokens outside configured range')
        if not isinstance(warmup, bool) or (warmup and max_tokens != 1):
            raise ValueError('Warmup must generate exactly one token')
        if self._expected:
            row = self._expected.get(sample_id)
            if row is None or row['input_token_ids'] != input_ids or row['input_ids_sha256'] != actual_hash:
                raise ValueError('Request differs from frozen canonical input')
            if not warmup and max_tokens != self.max_tokens:
                raise ValueError('Formal request must use the configured output budget')
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
               or (self.vocab_size is not None and token >= self.vocab_size)
               for token in input_ids):
            raise ValueError('input_token_ids contains an invalid token')
        with self.lock:
            if self.status != 'ok' or self.stopping.is_set():
                raise RuntimeError('AR worker is not ready')
            key = (sample_id, warmup)
            if key in self._seen:
                raise ValueError('Duplicate request for this frozen sample')
            self._seen.add(key)
            job = ARJob(sample_id, list(input_ids), max_tokens, warmup=warmup)
            self.stats['submitted'] += 1
            self.stats['warmup_submitted' if warmup else 'formal_submitted'] += 1
            self.stats['pending'] += 1
            self.pending.put(job)
        return job

    def _record(self, event):
        if self._log is not None:
            self._log.write(json.dumps(dict(time_ns=time.time_ns(), **event)) + '\n')
            self._log.flush()

    def _update_state(self):
        manager = self._manager
        with self.lock:
            active = len(self._active)
            caches = len(manager._past_key_values) if manager is not None else 0
            histories = len(manager._prob_history) if manager is not None else 0
            self.stats.update(active=active, cache_count=caches,
                              prob_history_count=histories)
            self.stats['peak_active'] = max(self.stats['peak_active'], active)
            self.stats['peak_cache_count'] = max(self.stats['peak_cache_count'], caches)
        if active > self.batch_size or caches > self.batch_size or histories > self.batch_size:
            raise RuntimeError('AR state exceeded configured active-session budget')

    def _finish(self, job, reason='completed', error=None, stop_reason=None):
        if self._manager is not None:
            self._manager.reset(job.session_id)
        self._active.pop(job.session_id, None)
        with self.lock:
            self.stats[reason] += 1
            if reason == 'completed':
                self.stats['warmup_completed' if job.warmup else 'formal_completed'] += 1
        self._update_state()
        message = {'done': True, 'generated_tokens': job.emitted, 'reason': reason,
                   'stop_reason': stop_reason, 'warmup': job.warmup,
                   'server_time_ns': time.time_ns()}
        if error is not None:
            message['error'] = error
        job.events.put(message)
        self._record(dict(event='request_end', session_id=job.session_id,
                          sample_id=job.sample_id, generated_tokens=job.emitted,
                          reason=reason, stop_reason=stop_reason, warmup=job.warmup,
                          cache_count=len(self._manager._past_key_values)
                          if self._manager is not None else 0))

    def _admit(self):
        while len(self._active) < self.batch_size:
            try:
                job = self.pending.get_nowait()
            except queue.Empty:
                break
            if job is None:
                continue
            with self.lock:
                self.stats['pending'] -= 1
            if job.cancelled.is_set():
                self._finish(job, 'cancelled')
                continue
            self._active[job.session_id] = job
        self._update_state()

    def _move_cache(self, session_id, device):
        from src.cache_views import dynamic_cache_from_views
        cache = self._manager._past_key_values.get(session_id)
        if cache is not None:
            legacy = cache.to_legacy_cache()
            moved = tuple((key.to(device), value.to(device)) for key, value in legacy)
            self._manager._past_key_values[session_id] = dynamic_cache_from_views(moved)

    def _run(self):
        model = None
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # The selected GPU is admitted before any CUDA initialization.
            from campaign import wait_gpu_memory
            from common import write_json
            imported_torch = sys.modules.get('torch')
            if imported_torch is not None and imported_torch.cuda.is_initialized():
                raise RuntimeError('Target-only CUDA was initialized before the resource gate')
            with self.lock:
                self.status = 'waiting_for_memory'
            wait_gpu_memory(self.config, output_dir, 'target_only')
            with self.lock:
                self.status = 'loading'
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from src.kvcache_batching import KVCacheModel_batching
            from src.kvcache_varlen import supports_varlen_model, varlen_generate
            self._torch = torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.cuda.set_device(0)
            fraction = float(self.config['gpu_memory_fraction'])
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError('Invalid configured GPU memory fraction')
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.manual_seed(int(self.config.get('seed', 1234)))
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            tokenizer_path = self.config.get('tokenizer_model', self.config.get('tokenizer'))
            model_path = self.config.get('target_model', self.config.get('model'))
            started = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                device_map={'': 'cuda:0'}, local_files_only=True).eval()
            dtypes = sorted({str(p.dtype) for p in model.parameters()})
            devices = sorted({str(p.device) for p in model.parameters()})
            if dtypes != ['torch.bfloat16'] or devices != ['cuda:0']:
                raise RuntimeError('Actual Target dtype/device mismatch: ' + str((dtypes, devices)))
            if not supports_varlen_model(model):
                raise ValueError('AR baseline requires supported varlen decoder')
            self.vocab_size = int(model.config.vocab_size)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if tokenizer.eos_token_id not in self.eos_token_ids:
                raise RuntimeError('Target tokenizer EOS is outside configured EOS IDs')
            self._manager = KVCacheModel_batching(model, temperature=0, top_k=0, top_p=1)
            self._manager.vocab_size = self.vocab_size
            # AR only needs each row's last boundary probability, including
            # intermediate prompt chunks. No full prompt probability history.
            self._manager.bounded_probabilities = True
            self._log = (output_dir / 'ar-batches.jsonl').open('x', encoding='utf-8')
            metadata = dict(epoch_s=time.time(), load_s=time.monotonic() - started,
                            method='target_only', model_path=model_path, tokenizer_path=tokenizer_path,
                            source_sha=self.config.get('source_sha'), torch=torch.__version__,
                            transformers=transformers.__version__, python=platform.python_version(),
                            dtypes=dtypes, device=str(model.device), devices=devices,
                            parameters=sum(p.numel() for p in model.parameters()),
                            vocab_size=self.vocab_size, eos_token_ids=sorted(self.eos_token_ids),
                            seed=int(self.config.get('seed', 1234)), gpu_memory_fraction=fraction,
                            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                            allocated_bytes=torch.cuda.memory_allocated(0),
                            reserved_bytes=torch.cuda.memory_reserved(0),
                            attention_implementation=getattr(model.config, '_attn_implementation', None),
                            probability_policy='last boundary only',
                            kv_policy='CPU offload after every mixed forward; no prefetch')
            write_json(output_dir / 'target-model-ready.json', metadata)
            self._record(dict(event='loaded', **metadata))
            with self.lock:
                self.status = 'ok'
            cursor = 0
            with torch.no_grad():
                while not self.stopping.is_set():
                    for job in list(self._active.values()):
                        if job.cancelled.is_set():
                            self._finish(job, 'cancelled')
                    self._admit()
                    plans = plan_batch(list(self._active.values()), self.batch_size,
                                       self.token_budget, cursor)
                    cursor += 1
                    if not plans:
                        # Brief event wait; accepted jobs are kept for admission on next loop.
                        try:
                            job = self.pending.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if job is not None:
                            with self.lock:
                                self.stats['pending'] -= 1
                            if job.cancelled.is_set():
                                self._finish(job, 'cancelled')
                            else:
                                self._active[job.session_id] = job
                                self._update_state()
                        continue
                    residuals, ids, lengths = [], [], []
                    for job, kind, length in plans:
                        self._move_cache(job.session_id, 'cuda:0')
                        values = (job.input_ids[job.prompt_position:job.prompt_position + length]
                                  if kind == 'prefill' else [job.next_token])
                        residuals.append(torch.tensor([values], dtype=torch.long, device='cuda:0'))
                        ids.append(job.session_id)
                        lengths.append(length)
                    outputs = varlen_generate(self._manager, model, residuals, ids,
                                              pad_token_id, is_prefill=True, sample_next=False)
                    if len(outputs) != len(plans):
                        raise RuntimeError('Target forward returned the wrong number of rows')
                    batch_tokens = 0
                    for (job, kind, length), output in zip(plans, outputs):
                        self._move_cache(job.session_id, 'cpu')
                        if kind == 'prefill':
                            job.prompt_position += length
                        if kind == 'prefill' and job.prompt_position < len(job.input_ids):
                            continue
                        token = int(output[-1].item())
                        if not 0 <= token < self.vocab_size:
                            raise RuntimeError('Target produced an invalid output token')
                        job.next_token = token
                        job.emitted += 1
                        batch_tokens += 1
                        job.events.put({'token_id': token, 'server_time_ns': time.time_ns()})
                        stop_reason = completion_reason(token, job.emitted, job.max_tokens, self.eos_token_ids)
                        if stop_reason is not None:
                            self._finish(job, stop_reason=stop_reason)
                    self._update_state()
                    prefill_count = sum(length for _, kind, length in plans if kind == 'prefill')
                    decode_count = sum(length for _, kind, length in plans if kind == 'decode')
                    memory = {'allocated_bytes': torch.cuda.memory_allocated(0),
                              'reserved_bytes': torch.cuda.memory_reserved(0)}
                    batch = dict(rows=len(plans), input_lengths=lengths,
                                 input_tokens=sum(lengths), output_tokens=batch_tokens,
                                 prefill_tokens=prefill_count, decode_tokens=decode_count)
                    with self.lock:
                        self.stats['batches'] += 1
                        self.stats['prefill_tokens'] += prefill_count
                        self.stats['decode_tokens'] += decode_count
                        self.stats['emitted_tokens'] += batch_tokens
                        self.stats['max_batch_rows'] = max(self.stats['max_batch_rows'], len(plans))
                        self.stats['max_batch_input_tokens'] = max(self.stats['max_batch_input_tokens'], sum(lengths))
                        self.stats['last_batch'] = batch
                    self._record(dict(event='batch', **batch, **memory,
                                      active=len(self._active),
                                      cache_count=len(self._manager._past_key_values),
                                      prob_history_count=len(self._manager._prob_history)))
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            with self.lock:
                self.status = 'failed'
                self.error = error
            (output_dir / 'target-only-failed.json').write_text(json.dumps({
                'epoch_s': time.time(), 'error': error, 'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
            for job in list(self._active.values()):
                self._finish(job, 'failed', error)
            self._drain_pending('failed', error)
        finally:
            for job in list(self._active.values()):
                self._finish(job, 'cancelled')
            self._drain_pending('cancelled', 'AR service is stopping')
            with self.lock:
                if self.status != 'failed':
                    self.status = 'stopped'
            self._record(dict(event='worker_exit', snapshot=self.snapshot()))
            if self._log is not None:
                self._log.close()
            if self._torch is not None and self._torch.cuda.is_initialized():
                memory = {'peak_allocated_bytes': self._torch.cuda.max_memory_allocated(0),
                          'peak_reserved_bytes': self._torch.cuda.max_memory_reserved(0),
                          'epoch_s': time.time(), 'status': self.status}
                (output_dir / 'target-memory.json').write_text(json.dumps(memory, indent=2), encoding='utf-8')
            self._manager = None
            model = None
            gc.collect()
            if self._torch is not None and self._torch.cuda.is_initialized():
                self._torch.cuda.empty_cache()

    def _drain_pending(self, reason, error):
        while True:
            try:
                job = self.pending.get_nowait()
            except queue.Empty:
                return
            if job is not None:
                with self.lock:
                    self.stats['pending'] -= 1
                self._finish(job, reason, error)


def install_ar(app, config):
    """Install the AR routes/lifecycle and return the single-model engine."""
    from fastapi import HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, StrictBool, StrictInt, StrictStr

    engine = AREngine(config)

    class GenerateRequest(BaseModel):
        input_token_ids: list[StrictInt]
        input_ids_sha256: StrictStr
        sample_id: StrictStr
        max_tokens: StrictInt = int(config.get('max_tokens', 400))
        warmup: StrictBool = False

    @app.on_event('startup')
    def start_ar():
        engine.start()

    @app.on_event('shutdown')
    def stop_ar():
        engine.stop()

    @app.get('/health')
    def health():
        state = engine.snapshot()
        return {key: state[key] for key in ('status', 'model_ready', 'worker_alive', 'error', 'method')}

    @app.get('/clock/sync')
    def clock_sync():
        return {'cloud_time_ns': time.time_ns()}

    @app.get('/ar/stats')
    def stats():
        return engine.snapshot()

    @app.post('/ar/generate')
    async def generate(body: GenerateRequest, request: Request):
        try:
            job = engine.submit(body.sample_id, body.input_token_ids, body.max_tokens,
                                body.input_ids_sha256, body.warmup)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(job.events.get, True, 0.25)
                    except queue.Empty:
                        continue
                    yield json.dumps(event, separators=(',', ':')) + '\n'
                    if event.get('done'):
                        break
            finally:
                job.cancelled.set()

        return StreamingResponse(stream(), media_type='application/x-ndjson',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return engine
