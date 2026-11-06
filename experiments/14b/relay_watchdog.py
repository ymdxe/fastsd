#!/usr/bin/env python3
"""Reap this campaign's explicit relays after its manager process exits.

Linux only. Never signals the campaign, a process group, or an unverified PID.
SIGTERM/SIGINT/SIGHUP to this watchdog stop observation without initiating cleanup.
"""
import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import threading
import time

TERMINAL_STATES = {"complete", "failed", "cancelled", "aborted"}


class IdentityError(RuntimeError):
    pass


class ProcessTransition(IdentityError):
    """An exiting/execing task temporarily has no command line; never signal it."""


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise IdentityError("JSON input exceeds 4 MiB: " + str(path))
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise IdentityError("Expected JSON object: " + str(path))
    return result


def proc_snapshot(pid):
    """Read only identity metadata, never environment or auth-file contents."""
    root = Path("/proc") / str(pid)
    try:
        first = (root / "stat").read_text()
        tail = first[first.rfind(")") + 2:].split()
        uids = next(line.split()[1:] for line in (root / "status").read_text().splitlines()
                    if line.startswith("Uid:"))
        if root.stat().st_uid != os.getuid() or any(int(uid) != os.getuid() for uid in uids):
            raise IdentityError("Refusing to read another UID's command line")
        command = [os.fsdecode(item) for item in (root / "cmdline").read_bytes().split(b"\0") if item]
        second = (root / "stat").read_text()
        check = second[second.rfind(")") + 2:].split()
        if tail[19] != check[19] or tail[2] != check[2]:
            raise IdentityError("Process identity changed during /proc read")
        return {"pid": pid, "pgid": int(check[2]), "start_ticks": int(check[19]),
                "state": check[0], "uids": [int(uid) for uid in uids],
                "proc_uid": root.stat().st_uid, "cmdline": command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def boot_epoch():
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return int(line.split()[1])
    raise IdentityError("Cannot read Linux boot time")


class Binding:
    """A trusted explicit launch record plus its pinned Linux process identity."""
    def __init__(self, path, kind):
        self.path = str(Path(path).resolve())
        record = read_json(path)
        self.pid = record.get("pid")
        self.pgid = record.get("pgid")
        if (type(self.pid) is not int or self.pid <= 1 or self.pid == os.getpid()
                or type(self.pgid) is not int or self.pgid <= 1):
            raise IdentityError("Unsafe or missing PID/PGID in " + self.path)
        if record.get("uid", os.getuid()) != os.getuid():
            raise IdentityError("Launch record UID differs from watchdog UID")
        argv = record.get("argv")
        observed = record.get("observed_cmdline", argv)
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise IdentityError("Invalid launch argv")
        if not isinstance(observed, list) or not observed or not all(isinstance(x, str) for x in observed):
            raise IdentityError("Invalid observed_cmdline")
        scripts = [x for x in argv if x.endswith(".py") and Path(x).is_absolute()]
        expected_name = "campaign.py" if kind == "campaign" else "reverse_relay.py"
        self.script = next((x for x in scripts if Path(x).name == expected_name), None)
        if self.script is None or any(x not in observed for x in scripts):
            raise IdentityError("Explicit launch script is absent from observed command")
        self.role = kind
        if kind == "relay":
            index = argv.index(self.script)
            self.role = argv[index + 1] if index + 1 < len(argv) else None
            if self.role not in {"hub", "agent"}:
                raise IdentityError("Relay launch does not name hub or agent role")
        self.command = observed
        self.script_paths = scripts
        self.start_ticks = record.get("start_ticks")
        if self.start_ticks is not None and (type(self.start_ticks) is not int or self.start_ticks <= 0):
            raise IdentityError("Invalid start_ticks")
        self.record_epoch = record.get("epoch_s")
        if self.start_ticks is None and (not isinstance(self.record_epoch, (int, float))
                                        or not math.isfinite(self.record_epoch)):
            raise IdentityError("Legacy record lacks a finite launch epoch")
        self.seen_alive = False
        self.legacy_epoch_validation = self.start_ticks is None

    def check(self):
        observed = proc_snapshot(self.pid)
        if observed is None:
            return "exited", None
        if observed["proc_uid"] != os.getuid() or any(uid != os.getuid() for uid in observed["uids"]):
            raise IdentityError("Process UID mismatch for PID " + str(self.pid))
        if observed["pgid"] != self.pgid:
            raise IdentityError("Process group mismatch for PID " + str(self.pid))
        if self.start_ticks is not None and observed["start_ticks"] != self.start_ticks:
            raise IdentityError("Process start_ticks mismatch (possible PID reuse): " + str(self.pid))
        if self.start_ticks is None:
            start_epoch = boot_epoch() + observed["start_ticks"] / os.sysconf("SC_CLK_TCK")
            delta = self.record_epoch - start_epoch
            # btime is integer seconds; control writes epoch immediately after Popen.
            if not -1.5 <= delta <= 10.0:
                raise IdentityError("Legacy launch epoch does not match process start")
            self.start_ticks = observed["start_ticks"]
        if observed["state"] in {"Z", "X", "x"}:
            return "exited", observed
        if not observed["cmdline"]:
            raise ProcessTransition("Process command line temporarily empty during exit/exec")
        if observed["cmdline"] != self.command or any(x not in observed["cmdline"] for x in self.script_paths):
            raise IdentityError("Exact command-line mismatch for PID " + str(self.pid))
        self.seen_alive = True
        return "alive", observed

    def public(self):
        return {"record": self.path, "pid": self.pid, "pgid": self.pgid,
                "start_ticks": self.start_ticks, "role": self.role,
                "legacy_epoch_validation": self.legacy_epoch_validation,
                "command_sha256": hashlib.sha256(json.dumps(self.command).encode()).hexdigest()}


def campaign_status(path, pid):
    try:
        status = read_json(path)
        if status.get("pid") != pid:
            return {"state": "unknown", "read_error": "status PID mismatch"}
        return {key: status.get(key) for key in ("state", "finished_epoch_s", "active_run_id")}
    except (OSError, ValueError, IdentityError) as exc:
        return {"state": "unknown", "read_error": type(exc).__name__}


def stop_relay(binding, grace, stop):
    result = {**binding.public(), "signals": [], "signal_binding": None,
              "signal_events": [], "identity_rechecks": 0}
    descriptor = None
    waiter = None
    unsupported = {errno.ENOSYS, errno.ENOTSUP}

    def alive():
        result["identity_rechecks"] += 1
        state, _ = binding.check()
        return state != "exited"

    def use_legacy(reason):
        nonlocal descriptor, waiter
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        waiter = None
        result.update(signal_binding="verified_pid_legacy_kernel", legacy_reason=reason,
                      binding_note="Legacy kernel has no pidfd atomic binding; full identity is rechecked immediately before each single-PID signal.")

    def record_signal(signum):
        name = signal.Signals(signum).name
        result["signals"].append(name)
        result["signal_events"].append({"signal": name, "binding": result["signal_binding"]})

    def send_verified(signum):
        if stop.is_set():
            return "stopped"
        if descriptor is not None:
            if not alive():
                return "exited"
            try:
                signal.pidfd_send_signal(descriptor, signum, None, 0)
            except OSError as exc:
                if exc.errno not in unsupported:
                    raise
                use_legacy("pidfd_send_signal_errno_" + str(exc.errno))
            else:
                record_signal(signum)
                return "sent"
        if stop.is_set():
            return "stopped"
        # Only an explicitly unsupported API/kernel reaches this branch. Do not
        # insert other work between this full identity check and the single PID.
        if not alive():
            return "exited"
        os.kill(binding.pid, signum)
        record_signal(signum)
        return "sent"

    def exited():
        if waiter is not None:
            return bool(waiter.poll(0))
        try:
            return not alive()
        except ProcessTransition:
            result["identity_transition_observations"] = result.get("identity_transition_observations", 0) + 1
            return False

    try:
        if stop.is_set():
            result["state"] = "manual_stop"
            return result
        if not alive():
            result["state"] = "already_exited"
            return result
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            use_legacy("pidfd_api_unavailable")
        else:
            try:
                descriptor = os.pidfd_open(binding.pid, 0)
            except OSError as exc:
                if exc.errno not in unsupported:
                    raise
                use_legacy("pidfd_open_errno_" + str(exc.errno))
            else:
                result["signal_binding"] = "pidfd"
                waiter = select.poll()
                waiter.register(descriptor, select.POLLIN)
        sent = send_verified(signal.SIGTERM)
        if sent != "sent":
            result["state"] = "already_exited" if sent == "exited" else "manual_stop"
            return result
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if exited():
                result["state"] = "terminated"
                return result
            if stop.wait(min(0.1, max(0, deadline-time.monotonic()))):
                result["state"] = "manual_stop_after_sigterm"
                return result
        if exited():
            result["state"] = "terminated"
            return result
        sent = send_verified(signal.SIGKILL)
        if sent != "sent":
            result["state"] = "terminated" if sent == "exited" else "manual_stop_after_sigterm"
            return result
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if exited():
                result["state"] = "killed"
                return result
            if stop.wait(min(0.1, max(0, deadline-time.monotonic()))):
                result["state"] = "manual_stop_after_sigkill"
                return result
        result["state"] = "kill_exit_unconfirmed"
    except ProcessLookupError:
        result["state"] = "already_exited"
    except (OSError, IdentityError) as exc:
        result.update(state="refused", error=str(exc))
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return result


def run_watchdog(campaign_record, status_path, relay_records, output, poll_seconds=15.0,
                 term_grace_seconds=30.0, stop=None):
    stop = stop if stop is not None else threading.Event()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A fresh explicit output is required; subsequent writes update only this file.
    with output.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")
    report = {"state": "starting", "pid": os.getpid(), "uid": os.getuid(),
              "started_epoch_s": time.time(), "campaign_status_path": str(status_path),
              "poll_seconds": poll_seconds, "term_grace_seconds": term_grace_seconds,
              "relays": [], "cleanup_started": False}
    def save():
        report["updated_epoch_s"] = time.time()
        temporary = output.with_name(output.name + ".writing-" + str(os.getpid()))
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    try:
        if os.getuid() != os.geteuid():
            raise IdentityError("Real and effective watchdog UID differ")
        manager = Binding(campaign_record, "campaign")
        relays = [Binding(path, "relay") for path in relay_records]
        if len(relays) not in (1, 3) or len({r.pid for r in relays} | {manager.pid}) != len(relays) + 1:
            raise IdentityError("Expected one hub or three unique agents, distinct from campaign")
        if (len(relays) == 1 and relays[0].role != "hub") or (len(relays) == 3 and any(r.role != "agent" for r in relays)):
            raise IdentityError("Relay topology must be one hub or three agents")
        for relay in relays:
            relay.check()
        report.update(campaign=manager.public(), relays=[r.public() for r in relays], state="watching")
        save()
        last_save = time.monotonic()
        while not stop.is_set():
            try:
                state, _ = manager.check()
            except ProcessTransition:
                # Do not infer exit from an empty cmdline before /proc reports it.
                stop.wait(min(poll_seconds, 1.0))
                continue
            status = campaign_status(status_path, manager.pid)
            report.update(campaign=manager.public(), campaign_status=status,
                          campaign_process_state=state, last_checked_epoch_s=time.time())
            if state == "exited":
                report["exit_reason"] = "terminal_and_exited" if status.get("state") in TERMINAL_STATES else "unexpected_process_exit"
                break
            if time.monotonic() - last_save >= 60:
                save()
                last_save = time.monotonic()
            stop.wait(poll_seconds)
        if stop.is_set():
            report["state"] = "manual_stop"
            report["finished_epoch_s"] = time.time()
            save()
            return 0, report
        # A failed/complete status alone never reaches this block while manager lives.
        report.update(state="cleaning", cleanup_started=True)
        save()
        results = []
        for relay in relays:
            results.append(stop_relay(relay, term_grace_seconds, stop))
            report["relays"] = results + [r.public() for r in relays[len(results):]]
            save()
            if stop.is_set():
                break
        success = not stop.is_set() and len(results) == len(relays) and all(
            result["state"] in {"already_exited", "terminated", "killed"} for result in results)
        report["state"] = "complete" if success else ("manual_stop" if stop.is_set() else "cleanup_incomplete")
        code = 0 if success or stop.is_set() else 2
    except (OSError, ValueError, IdentityError) as exc:
        report.update(state="refused", error=str(exc))
        code = 2
    report["finished_epoch_s"] = time.time()
    save()
    return code, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-record", required=True)
    parser.add_argument("--campaign-status", required=True)
    parser.add_argument("--relay-record", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--term-grace-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not 10 <= args.poll_seconds <= 30:
        parser.error("poll-seconds must be between 10 and 30")
    if not math.isfinite(args.term_grace_seconds) or not 1 <= args.term_grace_seconds <= 120:
        parser.error("term-grace-seconds must be between 1 and 120")
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda _signum, _frame: stop.set())
    code, report = run_watchdog(args.campaign_record, args.campaign_status, args.relay_record,
                               args.output, args.poll_seconds, args.term_grace_seconds, stop)
    print(json.dumps({"state": report["state"], "output": args.output}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
