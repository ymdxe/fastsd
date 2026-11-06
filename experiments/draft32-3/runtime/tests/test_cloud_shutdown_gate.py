"""Direct unit checks for the /exit admission gate ordering.

These exercise the real ``cloud.cloud_service`` handlers with a recording
queue; they do not start a worker, load a model, or run requests end-to-end.
"""

import asyncio
import threading
import unittest
from unittest.mock import patch

try:
    from fastapi import HTTPException

    import cloud.cloud_service as cs
except Exception:  # pragma: no cover - lightweight environments without deps
    cs = None
    HTTPException = None


class _AliveWorker:
    def is_alive(self):
        return True


class _RecordingQueue:
    def __init__(self):
        self.items = []
        self.fail = False
        self.request_published = threading.Event()

    def put(self, item):
        if self.fail:
            raise RuntimeError("queue broken")
        self.items.append(item)
        if isinstance(item, dict):
            self.request_published.set()


@unittest.skipIf(cs is None, "cloud service dependencies unavailable")
class CloudShutdownGateTests(unittest.TestCase):
    def test_release_interrupts_pending_request_and_waits_for_worker_ack(self):
        async def scenario():
            pending = asyncio.create_task(cs._enqueue_and_wait({"task_type": "verify"}, "session-1"))
            await asyncio.sleep(0)
            release = asyncio.create_task(cs.session_release(cs.SessionReleaseRequest(session_id="session-1")))
            await asyncio.sleep(0)
            self.assertEqual(await pending, {"error": "session released"})
            self.assertFalse(release.done())
            message = cs.request_queue.items[-1]
            self.assertEqual(message["control_type"], "session_release")
            cs._pending[message["response_key"]].set_result({"status": "released"})
            self.assertEqual((await release)["status"], "released")
        asyncio.run(scenario())

    def test_request_timeout_releases_abandoned_session(self):
        with patch.dict("os.environ", {"FASTSD_REQUEST_TIMEOUT_S": "0.01"}):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(cs._enqueue_and_wait({"task_type": "verify"}, "session-1"))
        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(cs._pending, {})
        self.assertEqual(cs._pending_sessions, {})
        self.assertEqual(cs.request_queue.items[-1],
                         {"control_type": "session_release", "proc_id": "session-1"})

    def setUp(self):
        self._saved = {
            "request_queue": cs.request_queue,
            "worker_proc": cs.worker_proc,
            "_admission_closed": cs._admission_closed,
            "_exit_sent": cs._exit_sent,
        }
        with cs._pending_lock:
            cs._pending.clear()
            cs._pending_sessions.clear()
            cs._pending_meta.clear()
            cs._pending_task_ids.clear()
        cs.request_queue = _RecordingQueue()
        cs.worker_proc = _AliveWorker()
        cs._admission_closed = False
        cs._exit_sent = False

    def tearDown(self):
        with cs._pending_lock:
            cs._pending.clear()
            cs._pending_sessions.clear()
            cs._pending_meta.clear()
            cs._pending_task_ids.clear()
        for name, value in self._saved.items():
            setattr(cs, name, value)

    def test_new_request_after_exit_is_rejected_without_enqueue_or_pending(self):
        self.assertEqual(cs.exit_worker()["status"], "sent")
        self.assertEqual(cs.request_queue.items, [None])

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                cs._enqueue_and_wait({"task_type": "verify"}, "session-1")
            )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(cs.request_queue.items, [None])
        self.assertEqual(cs._pending, {})
        self.assertNotIn("session-1", cs._pending_sessions)

    def test_repeated_exit_is_idempotent(self):
        self.assertEqual(cs.exit_worker()["status"], "sent")
        self.assertEqual(cs.exit_worker()["status"], "already_sent")
        self.assertEqual(cs.request_queue.items, [None])

    def test_admitted_request_precedes_sentinel_and_response_still_resolves(self):
        outcome = {}

        def run_request():
            try:
                outcome["value"] = asyncio.run(
                    cs._enqueue_and_wait({"task_type": "verify"}, "session-1")
                )
            except BaseException as exc:  # pragma: no cover - test diagnostic
                outcome["error"] = exc

        thread = threading.Thread(target=run_request)
        thread.start()
        try:
            # The registration and queue publication happen under the gate
            # lock; once observed, the request is guaranteed to precede None.
            self.assertTrue(cs.request_queue.request_published.wait(2.0))
            self.assertEqual(cs.exit_worker()["status"], "sent")

            self.assertEqual(len(cs.request_queue.items), 2)
            self.assertIsInstance(cs.request_queue.items[0], dict)
            self.assertIsNone(cs.request_queue.items[1])

            # /exit closes admission but must not cancel already-pending work.
            with cs._pending_lock:
                response_key = cs._pending_sessions["session-1"]
                future = cs._pending[response_key]
            future.get_loop().call_soon_threadsafe(
                future.set_result, {"status": "prefill_ok"}
            )
            thread.join(2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcome.get("value"), {"status": "prefill_ok"})
        finally:
            thread.join(2.0)

    def test_enqueue_failure_cleans_pending_and_returns_503(self):
        cs.request_queue.fail = True

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                cs._enqueue_and_wait({"task_type": "verify"}, "session-1")
            )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(cs.request_queue.items, [])
        self.assertEqual(cs._pending, {})
        self.assertNotIn("session-1", cs._pending_sessions)
        self.assertNotIn("session-1", cs._pending_task_ids)

    def test_exit_put_failure_reopens_admission(self):
        cs.request_queue.fail = True

        with self.assertRaises(HTTPException) as raised:
            cs.exit_worker()

        self.assertEqual(raised.exception.status_code, 503)
        self.assertFalse(cs._admission_closed)
        self.assertFalse(cs._exit_sent)
        self.assertEqual(cs.request_queue.items, [])

        cs.request_queue.fail = False
        self.assertEqual(cs.exit_worker()["status"], "sent")
        self.assertEqual(cs.request_queue.items, [None])

    def test_health_reports_draining_while_worker_is_alive(self):
        cs.exit_worker()
        status = cs.health()
        self.assertEqual(status["status"], "draining")
        self.assertTrue(status["draining"])
        self.assertTrue(status["worker_alive"])


if __name__ == "__main__":
    unittest.main()
