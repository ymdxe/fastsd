"""Explicit target attention selection and reusable packed GPU workspaces."""
from contextlib import contextmanager
import importlib
import math
import time
import torch

BACKENDS = ('sdpa_per_sequence', 'flash_attn_varlen')
_flash_kernel = None


def validate_attention_backend(name):
    if name not in BACKENDS:
        raise ValueError('Unknown target attention backend: ' + str(name))
    return name


def require_attention_backend(name, *, device=None, dtype=None):
    """Fail explicitly; production never silently substitutes another backend."""
    global _flash_kernel
    validate_attention_backend(name)
    if name == 'sdpa_per_sequence':
        return {'target_attention_backend': name, 'flash_attn_version': None}
    module = importlib.import_module('flash_attn')
    if module.__version__ != '2.8.3':
        raise RuntimeError('Expected flash-attn 2.8.3, found ' + module.__version__)
    importlib.import_module('flash_attn_2_cuda')
    _flash_kernel = module.flash_attn_varlen_func
    if device is not None:
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('flash_attn_varlen requires CUDA')
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError('flash_attn_varlen requires an Ampere-or-newer device')
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('flash_attn_varlen requires FP16 or BF16 tensors')
    return {'target_attention_backend': name, 'flash_attn_version': module.__version__}


class PhaseTimer:
    """Opt-in diagnostic timing; disabled in formal experiments."""
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.wall_ms = {}
        self.events = []

    @contextmanager
    def measure(self, phase):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        began = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((phase, start, end))
            self.wall_ms[phase] = self.wall_ms.get(phase, 0.0) + (time.perf_counter() - began) * 1000

    def collect(self):
        result = {'host_wall_ms': dict(self.wall_ms), 'stream_elapsed_ms': {}}
        if self.events:
            self.events[-1][2].synchronize()
        for phase, start, end in self.events:
            result['stream_elapsed_ms'][phase] = result['stream_elapsed_ms'].get(phase, 0.0) + start.elapsed_time(end)
        self.events.clear()
        self.wall_ms.clear()
        return result


class PackedAttentionWorkspace:
    """Single-compute-owner scratch, reused across layers and forwards.

    Source caches retain their own storage and transfer lifecycle. Scratch
    contains only the current layer's valid prefixes and is not a residency
    copy of the per-session cache. A stream fence protects subsequent reuse.
    """
    def __init__(self):
        self.q = self.k = self.v = None
        self.cu_q = self.cu_k = None
        self.signature = None
        self.last_event = None
        self.last_stream = None
        self.stats = {'attention_calls': 0, 'workspace_allocations': 0,
                      'packed_query_tokens': 0, 'packed_kv_tokens': 0}

    def prepare(self, ranges, *, device, dtype, query_heads, kv_heads, head_dim):
        if query_heads % kv_heads:
            raise ValueError('Query heads must be divisible by KV heads')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Packed attention workspace requires CUDA')
        stream = torch.cuda.current_stream(device)
        if self.last_event is not None and self.last_stream != stream.cuda_stream:
            stream.wait_event(self.last_event)
        self.query_lengths = [end - start for start, end, cached in ranges]
        self.key_lengths = [cached + end - start for start, end, cached in ranges]
        if not ranges or any(x <= 0 for x in self.query_lengths) or any(x < 0 for _, _, x in ranges):
            raise ValueError('Packed attention requires positive queries and nonnegative history')
        self.total_q, self.total_k = sum(self.query_lengths), sum(self.key_lengths)
        if max(self.total_q, self.total_k) >= 2 ** 31:
            raise ValueError('Packed lengths exceed int32 bounds')
        signature = (str(device), dtype, query_heads, kv_heads, head_dim)
        if self.signature != signature:
            self.q = self.k = self.v = None
            self.cu_q = self.cu_k = None
            self.signature = signature
        q_capacity = math.ceil(self.total_q / 128) * 128
        k_capacity = math.ceil(self.total_k / 128) * 128
        if self.q is None or self.q.shape[0] < self.total_q:
            self.q = torch.empty((q_capacity, query_heads, head_dim), device=device, dtype=dtype)
            self.stats['workspace_allocations'] += 1
        if self.k is None or self.k.shape[0] < self.total_k:
            self.k = torch.empty((k_capacity, kv_heads, head_dim), device=device, dtype=dtype)
            self.v = torch.empty_like(self.k)
            self.stats['workspace_allocations'] += 2
        boundaries_q, boundaries_k = [0], [0]
        for q_len, k_len in zip(self.query_lengths, self.key_lengths):
            boundaries_q.append(boundaries_q[-1] + q_len)
            boundaries_k.append(boundaries_k[-1] + k_len)
        self.batch_size = len(ranges)
        if self.cu_q is None or self.cu_q.numel() < self.batch_size + 1:
            self.cu_q = torch.empty(self.batch_size + 1, device=device, dtype=torch.int32)
            self.cu_k = torch.empty_like(self.cu_q)
            self.stats['workspace_allocations'] += 2
        self.cu_q[:self.batch_size + 1].copy_(torch.tensor(boundaries_q, dtype=torch.int32))
        self.cu_k[:self.batch_size + 1].copy_(torch.tensor(boundaries_k, dtype=torch.int32))
        for tensor in (self.q, self.k, self.v, self.cu_q, self.cu_k):
            tensor.record_stream(stream)
        self.last_stream = stream.cuda_stream

    def attend(self, q, caches, *, scale, timer=None):
        if _flash_kernel is None:
            raise RuntimeError('FlashAttention capability check was not performed')
        if len(caches) != len(self.key_lengths):
            raise ValueError('Packed cache count differs from query count')
        timer = timer or PhaseTimer(False)
        with timer.measure('kv_pack'):
            self.q[:self.total_q].copy_(q.squeeze(0).transpose(0, 1))
            offset = 0
            for (key, value), length in zip(caches, self.key_lengths):
                if key.shape[-2] != length or value.shape[-2] != length:
                    raise ValueError('Cache includes invalid or missing tokens')
                self.k[offset:offset + length].copy_(key.squeeze(0).transpose(0, 1))
                self.v[offset:offset + length].copy_(value.squeeze(0).transpose(0, 1))
                offset += length
        with timer.measure('attention'):
            output = _flash_kernel(self.q[:self.total_q], self.k[:self.total_k], self.v[:self.total_k],
                self.cu_q[:self.batch_size + 1], self.cu_k[:self.batch_size + 1],
                max(self.query_lengths), max(self.key_lengths),
                dropout_p=0.0, softmax_scale=scale, causal=True)
        self.stats['attention_calls'] += 1
        self.stats['packed_query_tokens'] += self.total_q
        self.stats['packed_kv_tokens'] += self.total_k
        return output.reshape(self.total_q, -1)

    def record_consumed(self):
        self.last_event = torch.cuda.Event()
        self.last_event.record(torch.cuda.current_stream(self.q.device))

    def snapshot(self):
        tensors = [self.q, self.k, self.v, self.cu_q, self.cu_k]
        return {**self.stats, 'workspace_bytes': sum(t.numel() * t.element_size() for t in tensors if t is not None)}
