"""Control-owner session retirement; never free a cache owned by a forward."""

import time


class SessionLifecycle:
    def __init__(self, *, cleanup, cancel, protected, busy, reply, idle_timeout=300.0,
                 clock=time.monotonic):
        self.cleanup = cleanup
        self.cancel = cancel
        self.protected = protected
        self.busy = busy
        self.reply = reply
        self.idle_timeout = float(idle_timeout)
        self.clock = clock
        self.last_seen = {}
        self.pending = {}
        self.released = 0

    def touch(self, pid):
        self.last_seen[pid] = self.clock()

    def release(self, pid, response_key=None):
        keys = self.pending.setdefault(pid, [])
        if response_key is not None:
            keys.append(response_key)
        self.cancel(pid)

    def sweep(self):
        now = self.clock()
        protected = self.protected()
        busy = self.busy()
        if self.idle_timeout > 0:
            for pid, seen in list(self.last_seen.items()):
                if pid not in protected and pid not in busy and now - seen >= self.idle_timeout:
                    self.release(pid)
        for pid, response_keys in list(self.pending.items()):
            if pid in protected:
                continue
            self.cleanup(pid)
            self.last_seen.pop(pid, None)
            self.pending.pop(pid, None)
            self.released += 1
            for key in response_keys:
                self.reply(key, {"status": "released"})
