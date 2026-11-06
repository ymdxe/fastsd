"""Task-scoped framed CPU/GPU reverse relay with fixed experiment endpoints.

CPU runs ``hub --bind-ip IB_ADDRESS``. GPU runs ``agent --hub CPU_IB``.
Only the GPU initiates inter-host TCP connections. Neither side can request an
arbitrary destination: model maps to GPU loopback 19660 (gRPC), and coordinator
maps to GPU loopback 19659 (HTTP). CPU listeners are loopback 29660/29659,
and the CPU IB channel port is 19661. Authentication is read only from a private
--auth-file; its contents are never printed or put in command arguments.
"""
from __future__ import annotations

import argparse
import collections
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import secrets
import selectors
import signal
import socket
import stat
import struct
import threading
import time

SERVICES = {"model": 19676, "coordinator": 19675}
LOCAL_PORTS = {"model": 29676, "coordinator": 29675}
CHANNEL_PORT = 19677
DEFAULT_LIFETIME_S = 48 * 60 * 60
MAX_LIFETIME_S = 21 * 24 * 60 * 60 + 60 * 60
PROTOCOL_VERSION = 2
FRAME_DATA = 1
FRAME_EOF = 2
FRAME_HEADER = struct.Struct("!BI")
MAX_FRAME_SIZE = 256 * 1024
BUFFER_LIMIT = 4 * 1024 * 1024


def recv_line(sock, limit=1024):
    data = bytearray()
    while len(data) < limit:
        byte = sock.recv(1)
        if not byte:
            raise ConnectionError("EOF during channel authentication")
        if byte == b"\n":
            return bytes(data)
        data.extend(byte)
    raise ValueError("oversized channel authentication")


def digest(token, nonce, service):
    message = ("framed-v" + str(PROTOCOL_VERSION) + ":" + nonce + ":" + service).encode("ascii")
    return hmac.new(token, message, hashlib.sha256).hexdigest()


def load_auth_file(path):
    """Read bounded authentication bytes from a private regular file.

    On POSIX, reject symlinks, another owner's file, and group/other permissions.
    The operator provisions this task file; this function never changes its mode.
    """
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("auth file must be a regular file")
        if os.name == "posix":
            if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ValueError("auth file must be owned by this user with no group/other permissions")
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("auth file exceeds the bounded input size")
    token = raw.strip()
    if not 32 <= len(token) <= 256:
        raise ValueError("auth file must contain 32 to 256 bytes")
    return token


class FrameProtocolError(ValueError):
    """Malformed framing or transport termination without a logical EOF."""


class FrameDecoder:
    def __init__(self):
        self.pending = bytearray()
        self.eof = False

    def room(self, output):
        return BUFFER_LIMIT - len(output) - len(self.pending)

    def feed(self, data, output):
        if self.eof and data:
            raise FrameProtocolError("bytes after logical EOF")
        if len(data) > self.room(output):
            raise FrameProtocolError("inbound buffer limit exceeded")
        self.pending.extend(data)
        while len(self.pending) >= FRAME_HEADER.size:
            kind, size = FRAME_HEADER.unpack_from(self.pending)
            if kind == FRAME_DATA:
                if not 1 <= size <= MAX_FRAME_SIZE:
                    raise FrameProtocolError("invalid DATA frame size")
            elif kind == FRAME_EOF:
                if size != 0:
                    raise FrameProtocolError("EOF frame must be empty")
            else:
                raise FrameProtocolError("unknown frame type")
            frame_size = FRAME_HEADER.size + size
            if len(self.pending) < frame_size:
                return
            if kind == FRAME_EOF:
                del self.pending[:frame_size]
                self.eof = True
                if self.pending:
                    raise FrameProtocolError("bytes after logical EOF")
                return
            output.extend(self.pending[FRAME_HEADER.size:frame_size])
            del self.pending[:frame_size]

    def transport_eof(self):
        if not self.eof or self.pending:
            raise FrameProtocolError("channel closed before complete logical EOF")


class Runtime:
    def __init__(self, *, idle_timeout=900, status_file=None):
        self.stop = threading.Event()
        self.idle_timeout = idle_timeout
        self.status_file = Path(status_file) if status_file else None
        self.lock = threading.Lock()
        self.sockets = set()
        self.threads = []
        self.thread_lock = threading.Lock()
        self.threads_reaped = 0
        self.counters = collections.Counter()
        self.started = time.time()

    def add(self, sock):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and sock.type == socket.SOCK_STREAM:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self.lock:
            self.sockets.add(sock)
        return sock

    def close(self, sock):
        with self.lock:
            self.sockets.discard(sock)
        try:
            sock.close()
        except OSError:
            pass

    def count(self, key, number=1):
        with self.lock:
            self.counters[key] += number

    def _reap_threads_locked(self):
        remaining = [thread for thread in self.threads if thread.is_alive()]
        self.threads_reaped += len(self.threads) - len(remaining)
        self.threads = remaining

    def reap_threads(self):
        with self.thread_lock:
            self._reap_threads_locked()
            return len(self.threads), self.threads_reaped

    def spawn(self, fn, *args):
        thread = threading.Thread(target=fn, args=args, daemon=True)
        with self.thread_lock:
            self._reap_threads_locked()
            # Start and register under one lock so an unstarted thread is never reaped.
            thread.start()
            self.threads.append(thread)
        return thread

    def snapshot(self):
        active_threads, threads_reaped = self.reap_threads()
        with self.lock:
            return {"started_epoch_s": self.started, "updated_epoch_s": time.time(),
                    "stopping": self.stop.is_set(), "open_sockets": len(self.sockets),
                    "active_threads": active_threads, "threads_reaped": threads_reaped,
                    "protocol_version": PROTOCOL_VERSION,
                    **dict(self.counters)}

    def save_status(self):
        self.reap_threads()
        if self.status_file:
            temporary = self.status_file.with_suffix(self.status_file.suffix + ".tmp")
            temporary.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
            os.replace(temporary, self.status_file)

    def shutdown(self):
        self.stop.set()
        with self.lock:
            sockets = list(self.sockets)
        for sock in sockets:
            self.close(sock)
        until = time.monotonic() + 5
        with self.thread_lock:
            threads = list(self.threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0, until - time.monotonic()))
        self.save_status()

    def pump(self, raw, channel):
        """Bridge a raw local socket and a framed inter-host channel.

        DATA carries unchanged application bytes; EOF explicitly closes one logical
        direction. Never use an inter-host FIN as the only completion signal.
        """
        outgoing = bytearray()
        incoming = bytearray()
        decoder = FrameDecoder()
        raw_read_open = True
        raw_write_open = True
        channel_read_open = True
        local_eof_sent = False
        last_activity = time.monotonic()
        selector = selectors.DefaultSelector()
        self.count("active_connections")
        try:
            raw.setblocking(False)
            channel.setblocking(False)
            while not self.stop.is_set():
                if decoder.eof and not incoming and raw_write_open:
                    try:
                        raw.shutdown(socket.SHUT_WR)
                    except OSError as exc:
                        if exc.errno == errno.ENOTCONN:
                            self.count("raw_shutdown_already_closed")
                        else:
                            self.count("raw_shutdown_errors")
                            raise
                    raw_write_open = False
                if not raw_read_open and not outgoing and not local_eof_sent:
                    local_eof_sent = True
                    self.count("eof_frames_sent")
                if local_eof_sent and decoder.eof and not incoming and not decoder.pending:
                    self.count("completed_connections")
                    return
                masks = {
                    raw: ((selectors.EVENT_READ if raw_read_open and
                           len(outgoing) + FRAME_HEADER.size < BUFFER_LIMIT else 0) |
                          (selectors.EVENT_WRITE if incoming and raw_write_open else 0)),
                    channel: ((selectors.EVENT_READ if channel_read_open and
                               decoder.room(incoming) > 0 else 0) |
                              (selectors.EVENT_WRITE if outgoing else 0)),
                }
                for sock, mask in masks.items():
                    try:
                        selector.unregister(sock)
                    except KeyError:
                        pass
                    if mask:
                        selector.register(sock, mask)
                if time.monotonic() - last_activity > self.idle_timeout:
                    self.count("idle_timeouts")
                    return
                for key, events in selector.select(timeout=0.5):
                    sock = key.fileobj
                    if events & selectors.EVENT_READ:
                        if sock is raw:
                            room = BUFFER_LIMIT - len(outgoing) - FRAME_HEADER.size
                            try:
                                data = raw.recv(min(MAX_FRAME_SIZE, room))
                            except BlockingIOError:
                                data = None
                            if data == b"":
                                raw_read_open = False
                                outgoing.extend(FRAME_HEADER.pack(FRAME_EOF, 0))
                                last_activity = time.monotonic()
                            elif data:
                                outgoing.extend(FRAME_HEADER.pack(FRAME_DATA, len(data)))
                                outgoing.extend(data)
                                self.count("payload_bytes_from_raw", len(data))
                                last_activity = time.monotonic()
                            with self.lock:
                                self.counters["outbound_buffer_peak"] = max(
                                    self.counters["outbound_buffer_peak"], len(outgoing))
                        else:
                            room = decoder.room(incoming)
                            try:
                                data = channel.recv(min(MAX_FRAME_SIZE, room))
                            except BlockingIOError:
                                data = None
                            if data == b"":
                                decoder.transport_eof()
                                channel_read_open = False
                            elif data:
                                was_eof = decoder.eof
                                with self.lock:
                                    self.counters["inbound_buffer_peak"] = max(
                                        self.counters["inbound_buffer_peak"],
                                        len(incoming) + len(decoder.pending) + len(data))
                                decoder.feed(data, incoming)
                                self.count("wire_bytes_received", len(data))
                                if decoder.eof and not was_eof:
                                    self.count("eof_frames_received")
                                last_activity = time.monotonic()
                    if events & selectors.EVENT_WRITE:
                        pending = incoming if sock is raw else outgoing
                        if not pending:
                            continue
                        try:
                            sent = sock.send(pending)
                        except BlockingIOError:
                            sent = None
                        if sent == 0:
                            raise ConnectionError("zero-byte send on active channel")
                        if sent:
                            del pending[:sent]
                            self.count("payload_bytes_to_raw" if sock is raw else "wire_bytes_sent", sent)
                            last_activity = time.monotonic()
        except FrameProtocolError:
            if not self.stop.is_set():
                self.count("protocol_errors")
                self.count("relay_errors")
        except (OSError, ValueError):
            if not self.stop.is_set():
                self.count("relay_errors")
        finally:
            selector.close()
            self.close(raw)
            self.close(channel)
            self.count("active_connections", -1)



class Hub(Runtime):
    def __init__(self, token, bind_ip, *, channel_port=CHANNEL_PORT, local_ports=None,
                 acquire_timeout=30, **kwargs):
        super().__init__(**kwargs)
        self.token = token
        self.bind_ip = bind_ip
        self.channel_port = channel_port
        self.local_ports = dict(LOCAL_PORTS if local_ports is None else local_ports)
        self.acquire_timeout = acquire_timeout
        self.available = {service: queue.Queue(maxsize=128) for service in SERVICES}
        self.listeners = {}

    def listen(self, name, address, callback):
        listener = self.add(socket.socket())
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
        listener.listen(128)
        listener.settimeout(0.5)
        self.listeners[name] = listener.getsockname()

        def accept():
            while not self.stop.is_set():
                try:
                    sock, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                self.spawn(callback, self.add(sock))

        self.spawn(accept)

    def start(self):
        self.listen("channels", (self.bind_ip, self.channel_port), self.authenticate)
        for service, port in self.local_ports.items():
            self.listen(service, ("127.0.0.1", port),
                        lambda sock, service=service: self.serve_client(service, sock))
        return self

    def authenticate(self, sock):
        queued = False
        try:
            sock.settimeout(5)
            nonce = secrets.token_hex(32)
            sock.sendall(json.dumps({"version": PROTOCOL_VERSION, "nonce": nonce}).encode() + b"\n")
            response = json.loads(recv_line(sock))
            if not isinstance(response, dict):
                raise ValueError("authentication response must be an object")
            service = response.get("service")
            if response.get("version") != PROTOCOL_VERSION or service not in SERVICES or not hmac.compare_digest(
                    str(response.get("digest", "")), digest(self.token, nonce, service)):
                self.count("authentication_failures")
                return
            sock.sendall(b"R")
            sock.settimeout(None)
            self.available[service].put_nowait(sock)
            queued = True
            self.count(service + "_authenticated_channels")
        except (OSError, ValueError, queue.Full, TypeError):
            self.count("channel_rejections")
        finally:
            if not queued:
                self.close(sock)

    @staticmethod
    def usable(sock):
        # Idle channels must not have bytes or EOF before the one-byte GO marker.
        readable, _, errors = __import__("select").select([sock], [], [sock], 0)
        return not errors and not readable

    def serve_client(self, service, client):
        channel = None
        started = time.monotonic()
        self.count(service + "_local_connections")
        try:
            while not self.stop.is_set() and time.monotonic() - started < self.acquire_timeout:
                try:
                    candidate = self.available[service].get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    if not self.usable(candidate):
                        self.close(candidate)
                        self.count("stale_channels")
                        continue
                    candidate.settimeout(10)
                    candidate.sendall(b"G")
                    if candidate.recv(1) != b"C":
                        self.close(candidate)
                        self.count("backend_connect_failures")
                        # Fail this local connection before reading its request.
                        # Never replay request bytes onto another backend.
                        return
                    channel = candidate
                    break
                except (OSError, ValueError):
                    self.close(candidate)
                    self.count("stale_channels")
            if channel is None:
                self.count("channel_acquire_failures")
                return
            self.pump(client, channel)
        finally:
            self.close(client)
            if channel:
                self.close(channel)

    def snapshot(self):
        return {**super().snapshot(), "role": "hub", "listeners": self.listeners,
                "available_channels": {key: value.qsize() for key, value in self.available.items()}}


class Agent(Runtime):
    def __init__(self, token, hub, *, channel_port=CHANNEL_PORT, backend_ports=None,
                 channels=32, control_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.token = token
        self.hub = hub
        self.channel_port = channel_port
        self.backend_ports = dict(SERVICES if backend_ports is None else backend_ports)
        self.pool_sizes = {"model": channels, "coordinator": control_channels}

    def start(self):
        for service, count in self.pool_sizes.items():
            for _ in range(count):
                self.spawn(self.slot, service)
        return self

    def slot(self, service):
        while not self.stop.is_set():
            channel = backend = None
            try:
                channel = self.add(socket.create_connection((self.hub, self.channel_port), timeout=5))
                channel.settimeout(5)
                challenge = json.loads(recv_line(channel))
                if not isinstance(challenge, dict):
                    raise ValueError("authentication challenge must be an object")
                nonce = challenge.get("nonce", "")
                if challenge.get("version") != PROTOCOL_VERSION or len(nonce) != 64:
                    raise ValueError("invalid hub challenge")
                response = {"version": PROTOCOL_VERSION, "service": service, "digest": digest(self.token, nonce, service)}
                channel.sendall(json.dumps(response).encode() + b"\n")
                if channel.recv(1) != b"R":
                    raise ConnectionError("hub rejected channel")
                self.count(service + "_ready_channels")
                channel.settimeout(0.5)
                try:
                    while not self.stop.is_set():
                        try:
                            marker = channel.recv(1)
                            break
                        except socket.timeout:
                            continue
                    else:
                        return
                finally:
                    self.count(service + "_ready_channels", -1)
                if marker != b"G":
                    raise ConnectionError("hub channel closed")
                backend = self.add(socket.create_connection(("127.0.0.1", self.backend_ports[service]), timeout=10))
                channel.sendall(b"C")
                self.pump(backend, channel)
                channel = backend = None
            except (OSError, ValueError, TypeError):
                if not self.stop.is_set():
                    self.count("channel_reconnects")
                    self.stop.wait(0.5)
            finally:
                if channel:
                    self.close(channel)
                if backend:
                    self.close(backend)

    def snapshot(self):
        return {**super().snapshot(), "role": "agent", "hub": self.hub,
                "channel_port": self.channel_port, "pool_sizes": self.pool_sizes}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("hub", "agent"))
    parser.add_argument("--auth-file", required=True, help="private file containing shared authentication bytes")
    parser.add_argument("--bind-ip", help="CPU IB address; required by hub")
    parser.add_argument("--hub", help="CPU IB address; required by GPU agent")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--control-channels", type=int, default=64)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--lifetime", type=float, default=DEFAULT_LIFETIME_S, help="lifetime seconds, 30 to 1818000; default 172800 (48 hours)")
    parser.add_argument("--idle-timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if not 1 <= args.channels <= 64 or not 1 <= args.control_channels <= 64:
        parser.error("channel counts outside permitted range")
    if not 30 <= args.lifetime <= MAX_LIFETIME_S or not 10 <= args.idle_timeout <= 3600:
        parser.error("invalid bounded lifetime or idle timeout")
    if args.role == "hub" and not args.bind_ip:
        parser.error("hub requires --bind-ip")
    if args.role == "agent" and not args.hub:
        parser.error("agent requires --hub")
    return args


def main():
    args = parse_args()
    try:
        token = load_auth_file(args.auth_file)
    except (OSError, ValueError):
        raise SystemExit("Cannot read auth file: require a private, current-user-owned regular file containing 32 to 256 bytes") from None
    kwargs = {"idle_timeout": args.idle_timeout, "status_file": args.status_file}
    if args.role == "hub":
        runtime = Hub(token, args.bind_ip, **kwargs)
    else:
        runtime = Agent(token, args.hub, channels=args.channels,
                        control_channels=args.control_channels, **kwargs)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: runtime.stop.set())
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *_: runtime.stop.set())
    try:
        runtime.start()
        print(json.dumps({"event": "relay-started", "pid": os.getpid(),
                          "lifetime_s": args.lifetime, **runtime.snapshot()}), flush=True)
        deadline = time.monotonic() + args.lifetime
        while not runtime.stop.wait(2):
            runtime.save_status()
            if time.monotonic() >= deadline:
                runtime.count("lifetime_expired")
                break
    finally:
        runtime.shutdown()
        print(json.dumps({"event": "relay-stopped", **runtime.snapshot()}), flush=True)


if __name__ == "__main__":
    main()
