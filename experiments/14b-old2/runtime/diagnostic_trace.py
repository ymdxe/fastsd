"""Buffered observational events for the isolated Config1 diagnostic campaign."""
import atexit
import gzip
import bisect
import json
import os
from pathlib import Path
import queue
import threading
import time

class DiagnosticTrace:
    def __init__(self):
        self.enabled = bool(os.environ.get("FASTSD_DIAGNOSTIC_DIR"))
        self.path = None
        self.events = queue.Queue(maxsize=200000)
        self.emit_count = self.dropped = self.emit_ns = self.writer_cpu_ns = 0
        self.written = self.writer_errors = 0
        self.closing = False
        self.thread = None
        self.start_lock = threading.Lock()
        self.context = threading.local()
        self.gpu_lock = threading.Lock()
        self.origin = None
        self.gpu_pending = []
        self.gpu_limit_drops = 0

    def _start(self):
        if self.thread is not None:
            return
        with self.start_lock:
            if self.thread is not None:
                return
            root = Path(os.environ["FASTSD_DIAGNOSTIC_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / ("events-" + str(os.getpid()) + ".jsonl.gz")
            self.thread = threading.Thread(target=self._writer, name="diagnostic-writer", daemon=True)
            self.thread.start()
            atexit.register(self.close)

    def emit(self, kind, **fields):
        if not self.enabled or self.closing:
            return
        start = time.perf_counter_ns()
        self._start()
        event = {"kind": kind, "mono_ns": time.monotonic_ns(), "epoch_ns": time.time_ns(),
                 "pid": os.getpid(), "tid": threading.get_ident(), **fields}
        try:
            self.events.put_nowait(event)
            self.emit_count += 1
        except queue.Full:
            self.dropped += 1
        self.emit_ns += time.perf_counter_ns() - start

    def _memory(self, handle):
        try:
            status = {}
            for line in Path("/proc/self/status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM", "VmSize", "VmSwap", "VmLck", "Threads"):
                    status[key] = int(value.split()[0])
            host = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("MemAvailable", "MemTotal", "SwapFree", "SwapTotal", "Mlocked", "Unevictable"):
                    host[key] = int(value.split()[0])
            vm = {}
            for line in Path("/proc/vmstat").read_text().splitlines():
                key, value = line.split()
                if key in ("pswpin", "pswpout", "pgmajfault", "oom_kill"):
                    vm[key] = int(value)
            handle.write(json.dumps({"kind":"memory_sample","pid":os.getpid(),
                "mono_ns":time.monotonic_ns(),"epoch_ns":time.time_ns(),
                "process":status,"host_kib":host,"host_vmstat":vm}) + "\n")
            self.written += 1
        except OSError:
            pass

    def _writer(self):
        try:
            with gzip.open(self.path,"xt",encoding="utf-8",compresslevel=1) as handle:
                next_sample = 0
                while True:
                    cpu = time.thread_time_ns()
                    try:
                        event = self.events.get(timeout=0.2)
                    except queue.Empty:
                        event = None
                    if event is False:
                        break
                    if event is not None:
                        handle.write(json.dumps(event, ensure_ascii=False, separators=(",",":"))+"\n")
                        self.written += 1
                    if time.monotonic() >= next_sample:
                        self._memory(handle)
                        handle.flush()
                        next_sample = time.monotonic() + 1
                    self.writer_cpu_ns += time.thread_time_ns() - cpu
        except BaseException as exc:
            self.writer_errors += 1
            self.writer_error = repr(exc)

    def gpu_init(self, torch, device):
        if not self.enabled or self.origin is not None or str(device) == "cpu":
            return
        # Called before warmup or measured requests, never in the forward loop.
        origin = torch.cuda.Event(enable_timing=True)
        origin.record(torch.cuda.current_stream(device))
        origin.synchronize()
        self.origin = origin
        self.emit("gpu_clock_origin", device=str(device), scope="one CUDA device; intervals compare without cross-host clock conversion")

    def gpu_span(self, kind, start, end, **fields):
        if not self.enabled or self.origin is None:
            return
        with self.gpu_lock:
            if len(self.gpu_pending) >= 100000:
                self.gpu_limit_drops += 1
                return
            self.gpu_pending.append((kind, start, end, fields))
        self.gpu_collect()

    def gpu_collect(self):
        if not self.enabled or self.origin is None:
            return
        with self.gpu_lock:
            remaining = []
            for kind, start, end, fields in self.gpu_pending:
                if end.query():
                    self.emit("cuda_span", operation=kind,
                              start_ms=self.origin.elapsed_time(start),
                              end_ms=self.origin.elapsed_time(end), **fields)
                else:
                    remaining.append((kind, start, end, fields))
            self.gpu_pending = remaining

    def close(self):
        if self.closing or self.thread is None:
            return
        self.gpu_collect()
        self.emit("trace_closing", unresolved_cuda_spans=len(self.gpu_pending))
        self.closing = True
        if self.thread.is_alive():
            try:
                self.events.put(False, timeout=5)
            except queue.Full:
                self.dropped += self.events.qsize()
            self.thread.join(timeout=15)
        summary = {"emitted":self.emit_count,"written":self.written,"dropped":self.dropped,
                   "emit_host_ns":self.emit_ns,"writer_cpu_ns":self.writer_cpu_ns,
                   "writer_errors":self.writer_errors,"writer_error":getattr(self,"writer_error",None),
                   "writer_alive_at_close":self.thread.is_alive(),
                   "unresolved_cuda_spans":len(self.gpu_pending),"cuda_limit_drops":self.gpu_limit_drops,
                   "scope":"Observer overhead is recorded, not subtracted from experiment latency."}
        self.path.with_name("trace-summary-"+str(os.getpid())+".json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

def request_meta(req):
    return {k:req.get(k) for k in ("task_id","proc_id","work_id","task_type","prefix_len","gamma",
        "server_enqueue_monotonic","arrival_monotonic_s","_diag_http_queue_mono_ns",
        "timing_ready","timing_required","_plan_ready_at") if k in req}

def item_meta(item):
    return {**request_meta(item.request),"work_id":item.work_id,"proc_id":item.proc_id,
        "task_type":item.task_type,"category":item.category,"cursor":item.cursor,
        "remaining_tokens":item.remaining_tokens,"missed_cycles":item.missed_cycles,
        "finished":item.finished}

def queue_rows(queues):
    state = queues.snapshot() if hasattr(queues,"snapshot") else queues
    return [item_meta(x) for categories in state.values() for rows in categories.values() for x in rows]

def backlog(arrivals, actual, index):
    return max(0, bisect.bisect_right(arrivals, actual) - index - 1)

trace = DiagnosticTrace()

