"""No-listener, public-fixture validation for the task relay.

Does not create an auth file, generate credentials, or open network sockets.
"""
from __future__ import annotations
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task_reverse_relay", ROOT / "reverse_relay.py")
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)
PUBLIC_FIXTURE = b"public-test-fixture-never-a-deployed-credential"


class PublicMemoryFile(io.BytesIO):
    def fileno(self):
        return 123


def read_public_fixture(payload=PUBLIC_FIXTURE, mode=stat.S_IFREG | 0o600, owner=1000):
    stream = PublicMemoryFile(payload)
    metadata = types.SimpleNamespace(st_mode=mode, st_uid=owner)
    with mock.patch.object(relay.os, "open", return_value=123) as opened, \
         mock.patch.object(relay.os, "fdopen", return_value=stream), \
         mock.patch.object(relay.os, "fstat", return_value=metadata), \
         mock.patch.object(relay.os, "name", "posix"), \
         mock.patch.object(relay.os, "geteuid", return_value=1000, create=True):
        value = relay.load_auth_file("public-fixture-memory-only")
        assert opened.call_args.args[1] & relay.os.O_RDONLY == relay.os.O_RDONLY
        if hasattr(relay.os, "O_NOFOLLOW"):
            assert opened.call_args.args[1] & relay.os.O_NOFOLLOW
        return value


class RelayValidation(unittest.TestCase):
    def test_fixed_endpoints_and_no_port_cli_override(self):
        self.assertEqual(relay.SERVICES, {"model": 19660, "coordinator": 19659})
        self.assertEqual(relay.LOCAL_PORTS, {"model": 29660, "coordinator": 29659})
        self.assertEqual(relay.CHANNEL_PORT, 19661)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            relay.parse_args(["agent", "--hub", "10.66.0.3", "--auth-file", "unused", "--status-file", "unused", "--backend-port", "22"])

    def test_lifetime_48h_and_bounded_pool_idle_limits(self):
        base = ["agent", "--hub", "10.66.0.3", "--auth-file", "unused", "--status-file", "unused"]
        self.assertEqual(relay.parse_args(base).lifetime, 172800)
        self.assertEqual(relay.parse_args(base + ["--lifetime", "172800", "--control-channels", "64"]).lifetime, 172800)
        for extra in (["--lifetime", "172801"], ["--lifetime", "29"], ["--lifetime", "nan"], ["--lifetime", "inf"], ["--channels", "65"], ["--control-channels", "65"], ["--idle-timeout", "3601"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                relay.parse_args(base + extra)

    def test_private_auth_file_constraints_with_memory_fixture(self):
        self.assertEqual(read_public_fixture(), PUBLIC_FIXTURE)
        for kwargs in ({"mode": stat.S_IFREG | 0o644}, {"owner": 999}, {"mode": stat.S_IFDIR | 0o700}, {"payload": b"short"}, {"payload": b"x" * 257}, {"payload": b"x" * 4097}):
            with self.subTest(kwargs={key: value if key != "payload" else len(value) for key, value in kwargs.items()}), self.assertRaises(ValueError):
                read_public_fixture(**kwargs)

    def test_concurrent_short_threads_reaped_and_live_thread_retained(self):
        runtime = relay.Runtime()
        def producer():
            for _ in range(64):
                runtime.spawn(lambda: None).join()
        producers = [threading.Thread(target=producer) for _ in range(4)]
        for item in producers:
            item.start()
        for item in producers:
            item.join()
        first = runtime.snapshot()
        self.assertEqual(first["active_threads"], 0)
        self.assertEqual(first["threads_reaped"], 256)
        release = threading.Event()
        worker = runtime.spawn(release.wait)
        self.assertEqual(runtime.snapshot()["active_threads"], 1)
        release.set()
        worker.join()
        runtime.save_status()
        self.assertEqual(runtime.threads, [])
        runtime.shutdown()
        self.assertEqual(runtime.snapshot()["threads_reaped"], 257)

    def test_snapshot_excludes_auth_content(self):
        for runtime in (relay.Hub(PUBLIC_FIXTURE, "10.66.0.3"), relay.Agent(PUBLIC_FIXTURE, "10.66.0.3")):
            snapshot = json.dumps(runtime.snapshot())
            self.assertNotIn(PUBLIC_FIXTURE.decode(), snapshot)
            self.assertNotIn('"token"', snapshot)
            runtime.shutdown()

    def test_original_auth_and_transport_bodies_unchanged(self):
        tree = ast.parse((ROOT / "reverse_relay.py").read_text(encoding="utf-8"))
        found = {}
        for item in tree.body:
            if isinstance(item, ast.FunctionDef):
                found[item.name] = item
            elif isinstance(item, ast.ClassDef):
                for method in item.body:
                    if isinstance(method, ast.FunctionDef):
                        found[item.name + "." + method.name] = method
        provenance = json.loads((ROOT / "relay-provenance.json").read_text(encoding="utf-8"))
        for name, expected in provenance["preserved_ast_sha256"].items():
            with self.subTest(function=name):
                actual = hashlib.sha256(ast.dump(found[name], include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, expected)
        self.assertEqual(hashlib.sha256((ROOT / "reverse_relay.py").read_bytes()).hexdigest(), provenance["relay_sha256"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RelayValidation)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(), "network_listeners_started": 0, "auth_files_created": 0, "secrets_generated": 0, "fixture": "public in-memory constant"}
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
