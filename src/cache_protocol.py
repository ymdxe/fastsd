"""Storage ownership and transfer metadata; no CUDA calls in this module."""

from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class CacheSnapshot:
    cache: object
    session_epoch: int = 0
    content_version: int = 0
    valid_len: int = 0
    source_identity: int = 0

    @classmethod
    def capture(cls, cache, *, session_epoch=0, content_version=0):
        return cls(cache, session_epoch, content_version,
                   int(cache.get_seq_length()), id(cache))


@dataclass(eq=False)
class KVTransferTicket:
    snapshot: CacheSnapshot
    purpose: str
    destination_cache: object = None
    source_lease: object = None
    producer_ready: object = None
    copy_done: object = None
    copy_state: str = "queued"
    claimed_by: object = None
    bytes_reserved: int = 0
    last_copy: object = None
    consumer_done: object = None
    chunks: list = field(default_factory=list)
    next_chunk: int = 0
    submitted_bytes: int = 0
    direction: str = "h2d"
    lock: object = field(default_factory=threading.RLock, repr=False)
    retired: bool = False
    reported: bool = False
    budget_released: bool = False

    @property
    def session_epoch(self):
        return self.snapshot.session_epoch

    @property
    def content_version(self):
        return self.snapshot.content_version

    @property
    def valid_len(self):
        return self.snapshot.valid_len

    @property
    def source_identity(self):
        return self.snapshot.source_identity

    def matches(self, snapshot):
        return (self.copy_state != "invalid-draining" and not self.retired
                and self.session_epoch == snapshot.session_epoch
                and self.content_version == snapshot.content_version
                and self.valid_len == snapshot.valid_len
                and self.source_identity == snapshot.source_identity)

    def ready(self):
        # An unrecorded event is never a promise for future submissions.
        return self.copy_state == "ready" or (
            self.copy_state == "copying" and self.copy_done is not None
            and self.copy_done.query())


@dataclass
class CacheLease:
    ticket: KVTransferTicket
    cache: object
    wait_start: object = None
    wait_end: object = None

    def exposed_stall_ms(self):
        if self.wait_end is None:
            return 0.0
        self.wait_end.synchronize()
        return self.wait_start.elapsed_time(self.wait_end)
