"""SpecEdge dynamic FIFO admission on the shared separate-RPC transport."""
from collections import Counter
import queue
import time


class WorkerInbox(queue.Queue):
    """The relay publishes here. Only the consumer owns sessions and KV.

    Queue calls _put/_get under its mutex. Cancellation watermarks allow a
    sealed phase to observe a release behind unrelated, still-queued data.
    """
    def __init__(self, clock=time.monotonic):
        super().__init__()
        self.clock = clock
        self.sequence = 0
        self.ready = Counter()
        self.cancel_seq = {}

    def _put(self, msg):
        if msg is not None:
            self.sequence += 1
            msg['worker_enqueue_seq'] = self.sequence
            msg['worker_enqueue_monotonic'] = self.clock()
            pid = msg['proc_id']
            if msg.get('control_type') in ('session_release', 'prefill_cancel'):
                self.cancel_seq[pid] = self.sequence
            self.ready[pid] += 1
        super()._put(msg)

    def _get(self):
        msg = super()._get()
        if msg is not None:
            pid = msg['proc_id']
            self.ready[pid] -= 1
            if not self.ready[pid]:
                del self.ready[pid]
            if self.cancel_seq.get(pid) == msg['worker_enqueue_seq']:
                del self.cancel_seq[pid]
        return msg

    def busy(self):
        with self.mutex:
            return set(self.ready)

    def cancelled(self, req):
        with self.mutex:
            return self.cancel_seq.get(req['proc_id'], 0) > req['worker_enqueue_seq']


def run_dynamic_fifo(inbox, limit, *, admit, execute, reserve, release, reject,
                     control, maintenance, sealed, phase_done, clock=time.monotonic):
    """One owner, one sealed batch. execute delivers responses synchronously."""
    selected = {}
    batch_id = 0
    stopping = False

    def discard(pid, error):
        req = selected.pop(pid)
        release([pid])
        reject(req, error)

    try:
        while not stopping:
            while len(selected) < limit:
                try:
                    msg = inbox.get_nowait() if selected else inbox.get(timeout=0.25)
                except queue.Empty:
                    if selected:
                        reason = 'queue_empty'
                        break
                    maintenance()
                    continue
                if msg is None:
                    stopping = True
                    reason = 'shutdown'
                    break
                pid = msg['proc_id']
                kind = msg.get('control_type')
                if kind:
                    if kind in ('prefill_cancel', 'session_release') and pid in selected:
                        discard(pid, 'session released')
                    control(msg)
                else:
                    req = admit(msg)
                    if req is None:
                        continue
                    if pid in selected:
                        reject(req, 'session already has an in-flight request')
                        continue
                    selected[pid] = req
                    reserve([pid])
            else:
                reason = 'full'
            if not selected:
                continue
            batch_id += 1
            at = clock()
            sealed(batch_id, list(selected.values()), reason, at)
            prefills = [[req] for req in selected.values() if req['task_type'] == 'prefill']
            verifies = [req for req in selected.values() if req['task_type'] == 'verify']
            for phase in [*prefills, verifies]:
                live = []
                for req in phase:
                    if inbox.cancelled(req):
                        discard(req['proc_id'], 'session released')
                    else:
                        live.append(req)
                        req['_plan_ready_at'] = at
                if live:
                    started = clock()
                    execute(live)
                    for req in live:
                        pid = req['proc_id']
                        del selected[pid]
                        release([pid])
                    phase_done(batch_id, live, started, clock())
            maintenance()
    except Exception as exc:
        # The cloud dispatcher rejects remaining queued RPCs on worker death.
        # Return the actual model/resource error to all unfinished selections.
        for pid in list(selected):
            discard(pid, str(exc))
        raise
