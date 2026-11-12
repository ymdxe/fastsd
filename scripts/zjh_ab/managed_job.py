#!/usr/bin/env python3
"""Small Linux job supervisor for runall; no third-party dependencies.

A random, persisted environment marker identifies descendants even if timeout
or multiprocessing creates another process group. No port/name-based killing.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

MARKER = 'FASTSD_MANAGED_JOB'


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return raw[19] if raw[0] != 'Z' else None  # field 22: starttime
    except (OSError, IndexError):
        return None


def owned_pids(token):
    key = (MARKER + '=' + token).encode()
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            if key in (entry / 'environ').read_bytes().split(b'\0') and identity(int(entry.name)):
                found.append(int(entry.name))
        except (OSError, ProcessLookupError):
            pass
    return found


def signal_owned(token, sig):
    for pid in owned_pids(token):
        # Recheck membership just before signalling; never use a saved PID alone.
        try:
            key = (MARKER + '=' + token).encode()
            if key in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0'):
                os.kill(pid, sig)
        except (OSError, ProcessLookupError):
            pass


def drain(token, grace):
    signal_owned(token, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while owned_pids(token) and time.monotonic() < deadline:
        time.sleep(.1)
    signal_owned(token, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while owned_pids(token) and time.monotonic() < deadline:
        time.sleep(.1)
    return not owned_pids(token)


def start(args):
    path = Path(args.state).resolve()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        raise ValueError('missing command')
    with locked(path):
        if path.exists():
            raise ValueError('job state already exists; use a new RUN_TAG: ' + str(path))
        data = dict(token=uuid.uuid4().hex, command=command, cwd=args.cwd,
                    phase='starting', stop_requested=False, created=time.time())
        write(path, data)
        log = Path(args.log).resolve()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab', buffering=0) as output:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                     'supervise', str(path)],
                                    env=dict(os.environ, **{MARKER: data['token']}),
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                    start_new_session=True, close_fds=True)
        data.update(supervisor_pid=proc.pid, supervisor_start=identity(proc.pid))
        write(path, data)
    print(proc.pid)
    return 0


def supervise(args):
    path = Path(args.state)
    stopping = [False]
    def stop_signal(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    child = None
    code = 143
    token = os.environ[MARKER]
    try:
        with locked(path):
            data = read(path)
            if not data['stop_requested'] and not stopping[0]:
                child = subprocess.Popen(data['command'], cwd=data['cwd'], close_fds=True)
                data.update(phase='running', child_pid=child.pid)
                write(path, data)
        if child:
            while child.poll() is None and not stopping[0]:
                time.sleep(.1)
            if not stopping[0]:
                code = child.returncode
    except BaseException as exc:
        print('managed job failed:', repr(exc), flush=True)
        code = 1
    finally:
        clean = drain(token, 5)
        if child:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                clean = False
        with locked(path):
            data = read(path)
            data.update(phase='finished' if clean else 'cleanup_failed', exit_code=code,
                        finished=time.time())
            write(path, data)
    return 0 if clean else 1


def stop(args):
    path = Path(args.state)
    with locked(path):
        if not path.exists():
            # A cancelled launch may still be in flight over SSH. A tombstone
            # prevents that delayed start from creating an untracked process.
            write(path, dict(token=uuid.uuid4().hex, phase='finished',
                             stop_requested=True, exit_code=143))
            return 0
        data = read(path)
        data['stop_requested'] = True
        write(path, data)
    if 'token' in data:
        clean = drain(data['token'], args.grace)
        with locked(path):
            data = read(path)
            if clean and data['phase'] != 'finished':
                data.update(phase='finished', exit_code=143, finished=time.time())
                write(path, data)
        return 0 if clean else 1
    # Controller shell: PID + kernel starttime prevents reuse killing.
    pid, birth = data['pid'], data['start']
    if birth and identity(pid) == birth:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + args.grace
        while identity(pid) == birth and time.monotonic() < deadline:
            time.sleep(.1)
        if identity(pid) == birth:
            os.kill(pid, signal.SIGKILL)
    return 0


def status(args):
    data = read(Path(args.state))
    if data['phase'] == 'finished':
        print(data.get('exit_code', 1))
    elif owned_pids(data['token']):
        print('running')
    else:
        print('lost')  # not success: supervisor was killed before recording exit
    return 0


def register(args):
    path = Path(args.state)
    with locked(path):
        if path.exists():
            raise ValueError('controller state already exists')
        write(path, dict(pid=args.pid, start=identity(args.pid)))
    return 0


def check_port(args):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', args.port))
        except OSError as exc:
            print('port %d unavailable: %s; stop the owning run before restarting' %
                  (args.port, exc), file=sys.stderr)
            return 1
    return 0


def owns_port(args):
    data = read(Path(args.state))
    inodes = set()
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        for row in Path(table).read_text().splitlines()[1:]:
            fields = row.split()
            if fields[3] == '0A' and int(fields[1].split(':')[1], 16) == args.port:
                inodes.add('socket:[' + fields[9] + ']')
    for pid in owned_pids(data['token']):
        try:
            for fd in Path('/proc', str(pid), 'fd').iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return 0
                except OSError:
                    pass
        except OSError:
            pass
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('start')
    p.add_argument('--state', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--cwd', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    for name in ('supervise', 'status', 'stop', 'register'):
        p = sub.add_parser(name)
        p.add_argument('state')
        if name == 'stop':
            p.add_argument('--grace', type=float, default=10)
        if name == 'register':
            p.add_argument('--pid', type=int, required=True)
    p = sub.add_parser('check-port')
    p.add_argument('port', type=int)
    p = sub.add_parser('owns-port')
    p.add_argument('state')
    p.add_argument('port', type=int)
    args = parser.parse_args()
    return globals()[args.action.replace('-', '_')](args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print('managed_job:', exc, file=sys.stderr)
        sys.exit(2)
