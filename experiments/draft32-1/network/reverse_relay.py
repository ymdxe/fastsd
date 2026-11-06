"""Task-scoped CPU/GPU reverse relay with fixed long-run experiment endpoints.

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
import threading
import time

SERVICES = {"model": 19660, "coordinator": 19659}
LOCAL_PORTS = {"model": 29660, "coordinator": 29659}
CHANNEL_PORT = 19661
MAX_LIFETIME_S = 48 * 60 * 60


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
    return hmac.new(token, (nonce + ":" + service).encode("ascii"), hashlib.sha256).hexdigest()


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

    def pump(self, first, second):
        """Relay full duplex with bounded buffering and TCP half-close semantics."""
        pair = (first, second)
        pending = {first: bytearray(), second: bytearray()}
        read_open = {first: True, second: True}
        write_open = {first: True, second: True}
        last_activity = time.monotonic()
        selector = selectors.DefaultSelector()
        self.count("active_connections")
        try:
            for sock in pair:
                sock.setblocking(False)
            while not self.stop.is_set():
                for source, destination in ((first, second), (second, first)):
                    if not read_open[source] and not pending[destination] and write_open[destination]:
                        try:
                            destination.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        write_open[destination] = False
                if not any(read_open.values()) and not any(pending.values()):
                    self.count("completed_connections")
                    return
                for sock in pair:
                    other = second if sock is first else first
                    mask = 0
                    if read_open[sock] and len(pending[other]) < 4 * 1024 * 1024:
                        mask |= selectors.EVENT_READ
                    if pending[sock] and write_open[sock]:
                        mask |= selectors.EVENT_WRITE
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
                    other = second if sock is first else first
                    if events & selectors.EVENT_READ:
                        try:
                            data = sock.recv(256 * 1024)
                        except BlockingIOError:
                            data = None
                        if data == b"":
                            read_open[sock] = False
                        elif data:
                            pending[other].extend(data)
                            last_activity = time.monotonic()
                            self.count("received_bytes", len(data))
                    if events & selectors.EVENT_WRITE and pending[sock]:
                        try:
                            sent = sock.send(pending[sock])
                        except BlockingIOError:
                            sent = 0
                        if sent:
                            del pending[sock][:sent]
                            last_activity = time.monotonic()
        except (OSError, ValueError):
            if not self.stop.is_set():
                self.count("relay_errors")
        finally:
            selector.close()
            self.close(first)
            self.close(second)
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
            sock.sendall(json.dumps({"version": 1, "nonce": nonce}).encode() + b"\n")
            response = json.loads(recv_line(sock))
            service = response.get("service")
            if service not in SERVICES or not hmac.compare_digest(
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
                nonce = challenge.get("nonce", "")
                if challenge.get("version") != 1 or len(nonce) != 64:
                    raise ValueError("invalid hub challenge")
                response = {"service": service, "digest": digest(self.token, nonce, service)}
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
                self.pump(channel, backend)
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
    parser.add_argument("--lifetime", type=float, default=MAX_LIFETIME_S, help="lifetime in seconds, 30 to 172800 (48 hours)")
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
