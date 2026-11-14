"""Replay and disconnect checks without loading models or using a GPU."""

import ast
import asyncio
from http.client import RemoteDisconnected
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List
import unittest
from unittest.mock import AsyncMock, Mock, call, patch
import uuid

import requests
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from src.idempotency import IdempotencyCache, IdempotencyConflict


def edge_client_class():
    # Exercise the production HTTP client without importing the model engine.
    path = Path(__file__).resolve().parents[1] / "edge" / "edge.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "EdgeClient")
    namespace = dict(requests=requests, json=json, threading=threading, time=time,
        uuid=uuid, os=__import__("os"), Any=Any, Dict=Dict, List=List,
        DIAG=SimpleNamespace(emit=lambda *args, **kwargs: None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["EdgeClient"]


EdgeClient = edge_client_class()


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    result.url = "http://target/verify"
    return result


class EdgeRetryTests(unittest.TestCase):
    def client(self):
        client = EdgeClient("http://target")
        client.session.close()
        client.timing_session.close()
        client.session = Mock()
        client.timing_session = Mock()
        return client

    def test_disconnect_reuses_body_sequence_and_rebuilds_connection(self):
        client = self.client()
        old = client.session
        old.post.side_effect = requests.ConnectionError("lost acknowledgement")
        replacement = Mock()
        replacement.post.return_value = response({"accepted": 3, "replayed": True})
        payload = dict(session_id="s", task_id="t", draft_output=[1, 2, 3])
        with patch.object(requests, "Session", return_value=replacement), patch.object(time, "sleep") as sleep:
            result = client._post("/verify", payload)
        old.close.assert_called_once()
        sleep.assert_called_once_with(0.5)
        self.assertFalse(replacement.trust_env)
        self.assertEqual(old.post.call_args.kwargs["data"], replacement.post.call_args.kwargs["data"])
        self.assertEqual(payload["request_seq"], 1)
        self.assertEqual(result["retry_count"], 1)
        self.assertIn("ConnectionError", result["last_error"])
        self.assertTrue(result["replayed"])
        client._post("/verify", dict(session_id="s", task_id="t", draft_output=[4]))
        self.assertEqual(json.loads(replacement.post.call_args.kwargs["data"])["request_seq"], 2)
        self.assertEqual(client.snapshot_transport_stats()["retry_count"], 1)

    def test_three_retries_use_exact_backoffs(self):
        client = self.client()
        sessions = [client.session, Mock(), Mock(), Mock()]
        for session in sessions:
            session.post.side_effect = ProtocolError("broken wire")
        with patch.object(requests, "Session", side_effect=sessions[1:]), patch.object(time, "sleep") as sleep:
            with self.assertRaises(ProtocolError):
                client._post("/verify", dict(session_id="s", task_id="t"))
        self.assertEqual(sleep.call_args_list, [call(0.5), call(1.0), call(2.0)])
        self.assertEqual(sum(session.post.call_count for session in sessions), 4)
        bodies = [session.post.call_args.kwargs["data"] for session in sessions]
        self.assertEqual(len(set(bodies)), 1)

    def test_only_connection_errors_are_retryable(self):
        for error in (requests.ConnectionError("reset"), RemoteDisconnected("closed"),
                      requests.exceptions.ChunkedEncodingError("truncated"), ProtocolError("closed")):
            self.assertTrue(EdgeClient._is_connection_failure(error))
        for error in (requests.Timeout("timeout"), requests.ConnectTimeout("timeout"),
                      requests.ReadTimeout("timeout"), TimeoutError(), ValueError("business"),
                      requests.ConnectionError(ReadTimeoutError(None, "/", "timeout"))):
            self.assertFalse(EdgeClient._is_connection_failure(error))

    def test_http_and_business_errors_and_timeouts_never_retry(self):
        for status in (400, 409, 422, 500, 503, 504):
            with self.subTest(status=status):
                client = self.client()
                client.session.post.return_value = response({"detail": {
                    "code": "negative_upload_pre_enqueue", "route": "/verify", "enqueued": False}}, status)
                with patch.object(requests, "Session") as new_session, patch.object(time, "sleep") as sleep:
                    with self.assertRaises(requests.HTTPError):
                        client._post("/verify", {})
                self.assertEqual(client.session.post.call_count, 1)
                new_session.assert_not_called()
                sleep.assert_not_called()
        client = self.client()
        client.session.post.side_effect = requests.ConnectionError(ReadTimeoutError(None, "/", "read timeout"))
        with patch.object(requests, "Session") as new_session:
            with self.assertRaises(requests.ConnectionError):
                client._post("/verify", {})
        new_session.assert_not_called()
        client.session.post.side_effect = None
        client.session.post.return_value = response({"error": "business failure"})
        self.assertEqual(client._post("/verify", {})["error"], "business failure")

    def test_verify_does_not_resubmit_a_422_or_regenerate_inputs(self):
        client = self.client()
        client._ensure_clock = lambda: False
        client.session.post.return_value = response({"detail": {
            "code": "negative_upload_pre_enqueue", "route": "/verify", "enqueued": False}}, 422)
        with self.assertRaises(requests.HTTPError):
            client.verify("s", "t", [1, 2], 1, .1, 0., 1)
        self.assertEqual(client.session.post.call_count, 1)

    def test_control_routes_share_sequence_without_sharing_connection(self):
        client = self.client()
        client.session.post.return_value = response({"status": "ok"})
        client.timing_session.post.return_value = response({"status": "cancelled"})
        client._post("/prefill", dict(session_id="s", task_id="t"))
        client._post("/prefill/cancel", dict(session_id="s", task_id="t"), session_attr="timing_session")
        body = json.loads(client.timing_session.post.call_args.kwargs["data"])
        self.assertEqual(body["request_seq"], 2)
        self.assertEqual(json.loads(body["request_id"]), ["POST", "/prefill/cancel", "s", "t", 2])


class ReplayCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_duplicates_execute_once(self):
        cache = IdempotencyCache()
        gate = asyncio.Event()
        async def work():
            await gate.wait()
            return {"tokens": [7]}
        operation = AsyncMock(side_effect=work)
        first = asyncio.create_task(cache.execute("key", "body", operation))
        await asyncio.sleep(0)
        duplicate = asyncio.create_task(cache.execute("key", "body", operation))
        await asyncio.sleep(0)
        gate.set()
        (value, replayed), (other, replayed_duplicate) = await asyncio.gather(first, duplicate)
        operation.assert_awaited_once()
        self.assertEqual(value, other)
        self.assertFalse(replayed)
        self.assertTrue(replayed_duplicate)
        value["tokens"].append(8)
        cached, _ = await cache.execute("key", "body", operation)
        self.assertEqual(cached, {"tokens": [7]})
        self.assertEqual(cache.metrics["idempotent_inflight_wait"], 1)
        self.assertEqual(cache.metrics["idempotent_cache_hit"], 1)

    async def test_disconnected_owner_does_not_cancel_inflight_work(self):
        cache = IdempotencyCache()
        gate = asyncio.Event()
        async def work():
            await gate.wait()
            return {"accepted": 4}
        operation = AsyncMock(side_effect=work)
        owner = asyncio.create_task(cache.execute("key", "body", operation))
        await asyncio.sleep(0)
        owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await owner
        duplicate = asyncio.create_task(cache.execute("key", "body", operation))
        await asyncio.sleep(0)
        gate.set()
        self.assertEqual(await duplicate, ({"accepted": 4}, True))
        operation.assert_awaited_once()

    async def test_failures_are_shared_but_not_cached(self):
        cache = IdempotencyCache()
        gate = asyncio.Event()
        async def work():
            await gate.wait()
            raise RuntimeError("worker failed")
        operation = AsyncMock(side_effect=work)
        first = asyncio.create_task(cache.execute("key", "body", operation))
        await asyncio.sleep(0)
        other = asyncio.create_task(cache.execute("key", "body", operation))
        await asyncio.sleep(0)
        gate.set()
        errors = await asyncio.gather(first, other, return_exceptions=True)
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors))
        operation.assert_awaited_once()
        self.assertEqual(cache.completed, {})
        self.assertEqual(cache.inflight, {})
        failed = AsyncMock(return_value={"error": "business failure"})
        await cache.execute("key", "body", failed)
        await cache.execute("key", "body", failed)
        self.assertEqual(failed.await_count, 2)

    async def test_lru_ttl_bounds_and_payload_conflicts(self):
        now = [0.]
        cache = IdempotencyCache(max_entries=2, ttl_s=10., clock=lambda: now[0])
        work = AsyncMock(return_value={"ok": True})
        await cache.execute("a", "body", work)
        await cache.execute("b", "body", work)
        now[0] = 1.
        await cache.execute("a", "body", work)
        await cache.execute("c", "body", work)
        self.assertEqual(list(cache.completed), ["a", "c"])
        with self.assertRaises(IdempotencyConflict):
            await cache.execute("a", "changed-body", work)
        now[0] = 12.
        await cache.execute("d", "body", work)
        self.assertEqual(list(cache.completed), ["d"])
        self.assertEqual(cache.metrics["cache_evictions"], 3)


class CloudReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import cloud.cloud_service as cs
        import httpx
        self.cs = cs
        self.patches = [
            patch.object(cs, "_idempotency_cache", IdempotencyCache()),
            patch.object(cs, "request_queue", object()),
        ]
        for patched in self.patches:
            patched.start()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=cs.app), base_url="http://target")

    async def asyncTearDown(self):
        await self.http.aclose()
        for patched in reversed(self.patches):
            patched.stop()

    def identity(self, route, sequence, **payload):
        payload.setdefault("session_id", "s")
        payload.setdefault("task_id", "t")
        payload.update(request_seq=sequence, request_id=json.dumps(
            ("POST", route, payload["session_id"], payload["task_id"], sequence), separators=(",", ":")))
        return payload

    def verify_payload(self):
        return self.identity("/verify", 2, draft_output=[1, 2, 3], prefix_len=2,
                             lag=.1, current_time=0., gamma=1, verify_method="greedy",
                             edge_send_time_ns=1)

    async def test_completed_verify_replays_after_session_release(self):
        live = {"s"}
        mutations = []
        async def worker(req, sid):
            if req.get("control_type") == "session_release":
                live.discard(sid)
                return {"status": "released"}
            if sid not in live:
                return {"error": "session has no target cache; prefill required"}
            mutations.append(req["draft_output"])
            return {"accepted": 3, "final_token": 9}
        with patch.object(self.cs, "_enqueue_and_wait", side_effect=worker):
            payload = self.verify_payload()
            first = await self.http.post("/verify", json=payload)
            released = await self.http.post("/session/release", json=self.identity("/session/release", 1, task_id=""))
            replay = await self.http.post("/verify", json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(released.json()["status"], "released")
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertTrue(replay.json()["replayed"])
        self.assertEqual({**first.json(), "replayed": True}, replay.json())
        self.assertEqual(mutations, [[1, 2, 3]])

    async def test_duplicate_waits_for_one_cloud_operation(self):
        gate = asyncio.Event()
        async def worker(*args):
            await gate.wait()
            return {"accepted": 3, "final_token": 9}
        work = AsyncMock(side_effect=worker)
        with patch.object(self.cs, "_enqueue_and_wait", work):
            first = asyncio.create_task(self.http.post("/verify", json=self.verify_payload()))
            await asyncio.sleep(0)
            duplicate = asyncio.create_task(self.http.post("/verify", json=self.verify_payload()))
            await asyncio.sleep(0)
            gate.set()
            replies = await asyncio.gather(first, duplicate)
        work.assert_awaited_once()
        self.assertEqual([reply.json()["replayed"] for reply in replies], [False, True])
        self.assertEqual(self.cs._idempotency_cache.metrics["idempotent_inflight_wait"], 1)

    async def test_init_replays_same_session_and_bad_identity_is_rejected(self):
        payload = self.identity("/session/init", 1)
        first = await self.http.post("/session/init", json=payload)
        replay = await self.http.post("/session/init", json=payload)
        self.assertEqual(first.json()["session_id"], replay.json()["session_id"])
        self.assertTrue(replay.json()["replayed"])
        payload["request_seq"] = 2
        self.assertEqual((await self.http.post("/session/init", json=payload)).status_code, 422)

    async def test_failed_cloud_response_is_not_cached(self):
        work = AsyncMock(side_effect=[{"error": "temporary business error"},
                                     {"accepted": 3, "final_token": 9}])
        with patch.object(self.cs, "_enqueue_and_wait", work):
            first = await self.http.post("/verify", json=self.verify_payload())
            later = await self.http.post("/verify", json=self.verify_payload())
        self.assertEqual(first.status_code, 422)
        self.assertEqual(later.status_code, 200)
        self.assertFalse(later.json()["replayed"])
        self.assertEqual(work.await_count, 2)

    async def test_edge_lost_response_retries_through_real_cloud_routes_once(self):
        loop = asyncio.get_running_loop()
        bodies = []
        class Connection:
            trust_env = False
            def close(self):
                pass
            def post(_connection, url, *, data, **kwargs):
                bodies.append(data)
                result = asyncio.run_coroutine_threadsafe(
                    self.http.post("/verify", content=data, headers={"Content-Type": "application/json"}), loop).result(10)
                if len(bodies) == 1:
                    raise requests.ConnectionError(RemoteDisconnected("response lost after completion"))
                return response(result.json(), result.status_code)
        client = EdgeClient("http://target")
        client.session.close()
        client.session = Connection()
        work = AsyncMock(return_value={"accepted": 3, "final_token": 9})
        payload = self.verify_payload()
        payload.pop("request_id")
        payload.pop("request_seq")
        with patch.object(self.cs, "_enqueue_and_wait", work), patch.object(requests, "Session", Connection), patch.object(time, "sleep"):
            result = await asyncio.to_thread(client._post, "/verify", payload)
        self.assertEqual(result["accepted"], 3)
        self.assertTrue(result["replayed"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(bodies[0], bodies[1])
        work.assert_awaited_once()
