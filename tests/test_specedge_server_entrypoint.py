import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from baselines.specedge.adapter import server_entrypoint as entrypoint


class SpecEdgeServerEntrypointTests(unittest.TestCase):
    @staticmethod
    def _official_batch_config() -> dict:
        return {
            "base": {
                "result_path": "/tmp/specedge-results",
                "exp_name": "adapter-test",
                "seed": 7,
                "max_len": 128,
                "dtype": "bf16",
            },
            "server": {
                "target_model": "/models/Qwen3-8B",
                "device": "cuda:0",
                "temperature": 0.0,
                "max_batch_size": 2,
                "num_clients": 2,
                "batch_type": "static",
                "cache_prefill": False,
            },
            "client": {
                "dataset": "mtbench",
                "sample_req_cnt": 1,
                "req_offset": 0,
                "max_n_beams": 4,
                "max_budget": 32,
            },
        }

    @staticmethod
    def _unified_config() -> dict:
        return {
            "experiment": {
                "name": "qwen3-cloud-edge-mtbench-poisson",
                "dataset": "mt_bench",
                "run_root": "/tmp/fastsd-results",
            },
            "models": {
                "draft_model": "/models/Qwen3-0.6B",
                "target_model": "/models/Qwen3-8B",
                "dtype": "bfloat16",
            },
            "decoding": {"max_new_tokens": 256, "temperature": 0.0},
            "workload": {"num_edge_clients": 2, "max_requests": 80},
            "specedge": {
                "tree_budget": 32,
                "max_beam_len": 4,
                "max_branch_width": 16,
                "max_batch_size": 2,
                "num_clients": 2,
                "cache_prefill": False,
            },
        }

    def _write_config(self, directory: Path) -> Path:
        config_path = directory / "official.yaml"
        config_path.write_text(
            yaml.safe_dump(self._official_batch_config(), sort_keys=False), encoding="utf-8"
        )
        return config_path

    def test_explicit_ib_address_formats_as_expected(self):
        host = entrypoint.validate_bind_host("10.66.0.5")
        self.assertEqual(entrypoint.format_bind_address(host, 18000), "10.66.0.5:18000")
        self.assertEqual(
            entrypoint.format_bind_address(entrypoint.validate_bind_host("::1"), 18000),
            "[::1]:18000",
        )

    def test_rejects_wildcard_or_nonliteral_listener(self):
        for value in ("0.0.0.0", "::", "node2", ""):
            with self.subTest(value=value):
                with self.assertRaises(entrypoint.ServerEntrypointError):
                    entrypoint.validate_bind_host(value)

    def test_dry_run_validates_config_without_creating_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._write_config(root)
            run_dir = root / "would-not-exist"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                return_code = entrypoint.main(
                    [
                        "--config",
                        str(config),
                        "--bind-host",
                        "10.66.0.5",
                        "--port",
                        "18000",
                        "--run-dir",
                        str(run_dir),
                        "--dry-run",
                    ]
                )

            report = json.loads(captured.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(report["mode"], "dry-run")
            self.assertEqual(report["bind_address"], "10.66.0.5:18000")
            self.assertTrue(report["would_write_effective_config"])
            self.assertEqual(
                report["server_events_output"],
                str(run_dir / "metrics" / entrypoint.SERVER_EVENTS_FILE_NAME),
            )
            self.assertFalse(run_dir.exists())

    def test_unified_config_renders_official_batch_server_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "unified.yaml"
            config_path.write_text(
                yaml.safe_dump(self._unified_config(), sort_keys=False), encoding="utf-8"
            )
            rendered = entrypoint.load_and_validate_config(
                config_path,
                run_id="specedge-test",
                target_model="/override/Qwen3-8B",
                draft_model="/override/Qwen3-0.6B",
                client_host="10.66.0.5:18000",
            )

            self.assertEqual(rendered.source_format, "fastsd-unified")
            self.assertEqual(rendered.document["base"]["dtype"], "bf16")
            self.assertEqual(rendered.document["base"]["exp_name"], "specedge-test")
            self.assertEqual(rendered.document["server"]["device"], "cuda:0")
            self.assertEqual(rendered.document["server"]["target_model"], "/override/Qwen3-8B")
            self.assertEqual(rendered.document["server"]["batch_type"], "dynamic")
            self.assertEqual(rendered.document["server"]["max_batch_size"], 2)
            self.assertEqual(rendered.document["server"]["num_clients"], 2)
            self.assertEqual(rendered.document["client"]["draft_model"], "/override/Qwen3-0.6B")
            self.assertEqual(rendered.document["client"]["host"], "10.66.0.5:18000")
            self.assertEqual(rendered.document["client"]["dataset"], "mtbench")
            self.assertEqual(rendered.document["client"]["max_budget"], 32)

    def test_unified_dry_run_accepts_gpu_zero_without_creating_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "unified.yaml"
            config_path.write_text(
                yaml.safe_dump(self._unified_config(), sort_keys=False), encoding="utf-8"
            )
            run_dir = root / "run"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                return_code = entrypoint.main(
                    [
                        "--config",
                        str(config_path),
                        "--run-id",
                        "specedge-test",
                        "--bind-host",
                        "10.66.0.5",
                        "--cloud-physical-gpu",
                        "0",
                        "--run-dir",
                        str(run_dir),
                        "--dry-run",
                    ]
                )

            report = json.loads(captured.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(report["config_source_format"], "fastsd-unified")
            self.assertFalse(run_dir.exists())

    def test_smoke_token_override_is_reported_and_leaves_source_config_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "unified.yaml"
            source = self._unified_config()
            config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                return_code = entrypoint.main(
                    [
                        "--config",
                        str(config_path),
                        "--run-id",
                        "specedge-smoke",
                        "--bind-host",
                        "10.66.0.5",
                        "--max-new-tokens",
                        "32",
                        "--dry-run",
                    ]
                )
            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(captured.getvalue())["max_new_tokens"], 32)
            self.assertEqual(yaml.safe_load(config_path.read_text(encoding="utf-8"))["decoding"]["max_new_tokens"], 256)

    def test_effective_config_is_under_run_dir_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = entrypoint.load_and_validate_config(self._write_config(root))
            run_dir = root / "run"
            result_dir = root / "results"

            effective = entrypoint.write_effective_config(
                config, run_dir=run_dir, result_path=result_dir
            )
            saved = yaml.safe_load(effective.read_text(encoding="utf-8"))
            self.assertEqual(effective, run_dir / "config" / entrypoint.EFFECTIVE_CONFIG_NAME)
            self.assertEqual(saved["base"]["result_path"], str(result_dir.resolve()))
            self.assertEqual(config.document["base"]["result_path"], "/tmp/specedge-results")
            with self.assertRaisesRegex(entrypoint.ServerEntrypointError, "overwrite"):
                entrypoint.write_effective_config(config, run_dir=run_dir)

    def test_config_validation_reports_missing_official_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data = self._official_batch_config()
            del data["server"]["target_model"]
            config = root / "invalid.yaml"
            config.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(entrypoint.ServerEntrypointError, "target_model"):
                entrypoint.load_and_validate_config(config)

    def test_shutdown_file_requires_absolute_path_and_dry_run_never_creates_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._write_config(root)
            marker = root / "control" / "graceful-shutdown.json"
            with self.assertRaisesRegex(entrypoint.ServerEntrypointError, "absolute"):
                entrypoint.resolve_shutdown_file("control/graceful-shutdown.json")

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                return_code = entrypoint.main(
                    [
                        "--config",
                        str(config),
                        "--run-id",
                        "shutdown-dry-run",
                        "--bind-host",
                        "10.66.0.5",
                        "--shutdown-file",
                        str(marker),
                        "--dry-run",
                    ]
                )

            report = json.loads(captured.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(report["shutdown_file"], str(marker))
            self.assertFalse(marker.exists())
            self.assertFalse(marker.parent.exists())

    def test_live_shutdown_file_must_be_new_and_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._write_config(root)
            marker = root / "graceful-shutdown.json"
            original = '{"run_id":"stale-run"}\n'
            marker.write_text(original, encoding="utf-8")
            run_dir = root / "live-run"
            captured_error = io.StringIO()
            with contextlib.redirect_stderr(captured_error):
                return_code = entrypoint.main(
                    [
                        "--config",
                        str(config),
                        "--run-id",
                        "new-run",
                        "--bind-host",
                        "10.66.0.5",
                        "--run-dir",
                        str(run_dir),
                        "--shutdown-file",
                        str(marker),
                    ]
                )

            self.assertEqual(return_code, 2)
            self.assertIn("must not already exist", captured_error.getvalue())
            self.assertEqual(marker.read_text(encoding="utf-8"), original)
            self.assertFalse(run_dir.exists())

    def test_shutdown_marker_matches_only_its_run_and_async_watcher_sets_event(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mismatched = root / "mismatched.json"
            matched = root / "matched.json"
            mismatched.write_text('{"run_id":"other-run"}\n', encoding="utf-8")
            matched.write_text('{"run_id":"target-run"}\n', encoding="utf-8")

            self.assertFalse(
                entrypoint.shutdown_marker_matches(mismatched, run_id="target-run")
            )
            self.assertTrue(
                entrypoint.shutdown_marker_matches(matched, run_id="target-run")
            )

            async def exercise_watcher() -> None:
                mismatched_event = asyncio.Event()
                mismatched_task = asyncio.create_task(
                    entrypoint.watch_shutdown_file(
                        mismatched_event,
                        shutdown_file=mismatched,
                        run_id="target-run",
                        poll_interval_s=0.001,
                    )
                )
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(mismatched_event.wait(), timeout=0.02)
                self.assertFalse(mismatched_event.is_set())
                mismatched_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await mismatched_task

                matched_event = asyncio.Event()
                watcher_result = await asyncio.wait_for(
                    entrypoint.watch_shutdown_file(
                        matched_event,
                        shutdown_file=matched,
                        run_id="target-run",
                        poll_interval_s=0.001,
                    ),
                    timeout=0.1,
                )
                self.assertTrue(watcher_result)
                self.assertTrue(matched_event.is_set())

            with contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(exercise_watcher())

    def test_matching_shutdown_marker_uses_server_graceful_cleanup_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            marker = root / "graceful-shutdown.json"
            events_path = root / "metrics" / entrypoint.SERVER_EVENTS_FILE_NAME
            marker.write_text('{"run_id":"target-run"}\n', encoding="utf-8")

            class FakeServer:
                def __init__(self) -> None:
                    self.started = False
                    self.stop_grace: list[float] = []

                def add_insecure_port(self, _bind_address: str) -> int:
                    return 18000

                async def start(self) -> None:
                    self.started = True

                async def stop(self, grace: float) -> None:
                    self.stop_grace.append(grace)

            class FakeController:
                instances: list["FakeController"] = []

                def __init__(self, *, shutdown_event: asyncio.Event) -> None:
                    self.shutdown_event = shutdown_event
                    self.cleaned = False
                    self.__class__.instances.append(self)

                async def cleanup(self) -> None:
                    self.cleaned = True

            class FakeGrpcAio:
                def __init__(self, server: FakeServer) -> None:
                    self._server = server

                def server(self) -> FakeServer:
                    return self._server

            class FakeService:
                registered = False
                servicer = None

                @classmethod
                def add_SpecEdgeServiceServicer_to_server(cls, controller, _server):
                    cls.registered = True
                    cls.servicer = controller

            fake_server = FakeServer()
            bindings = entrypoint.OfficialBindings(
                batch_server=object(),
                controller_type=FakeController,
                grpc_aio=FakeGrpcAio(fake_server),
                grpc_service=FakeService,
            )
            captured = io.StringIO()
            with mock.patch.object(entrypoint, "_install_shutdown_handlers", return_value={}):
                with contextlib.redirect_stdout(captured):
                    asyncio.run(
                        entrypoint.serve_official(
                            bindings,
                            bind_address="10.66.0.5:18000",
                            shutdown_file=marker,
                            run_id="target-run",
                            server_events_output=events_path,
                        )
                    )

            self.assertTrue(FakeService.registered)
            self.assertIsInstance(FakeService.servicer, entrypoint.EventRecordingSpecEdgeService)
            self.assertTrue(fake_server.started)
            self.assertEqual(fake_server.stop_grace, [2.0])
            self.assertTrue(FakeController.instances[0].cleaned)
            self.assertTrue(events_path.is_file())
            events = [json.loads(line) for line in captured.getvalue().splitlines()]
            self.assertIn("shutdown_marker_matched", [event["event"] for event in events])
            self.assertIn("server_stopped", [event["event"] for event in events])
            stopped = next(event for event in events if event["event"] == "server_stopped")
            self.assertTrue(stopped["cleanup_completed"])

    def test_event_proxy_records_server_monotonic_enter_leave_and_safe_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            events_path = Path(temporary_directory) / "specedge_server_events.jsonl"
            writer = entrypoint.ServerGrpcEventWriter.open_new(
                events_path, run_id="event-run"
            )

            class Request:
                client_idx = 1
                req_idx = 9
                prefill = True
                # This deliberately must not appear in the event artifact.
                prefix = "sensitive prompt text"

            class Controller:
                async def Validate(self, _request, _context):
                    return "validate-result"

                async def Sync(self, _request, _context):
                    raise RuntimeError("simulated server failure")

            service = entrypoint.EventRecordingSpecEdgeService(Controller(), writer)

            async def exercise_service() -> None:
                self.assertEqual(
                    await service.Validate(Request(), object()), "validate-result"
                )
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    await service.Sync(object(), object())

            asyncio.run(exercise_service())
            writer.close()

            rows = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(row["method"], row["phase"]) for row in rows],
                [("Validate", "enter"), ("Validate", "leave"), ("Sync", "enter"), ("Sync", "leave")],
            )
            self.assertEqual(rows[0]["request_identity"], {"client_idx": 1, "prefill": True, "req_idx": 9})
            self.assertNotIn("request_identity", rows[2])
            self.assertEqual(rows[1]["outcome"], "completed")
            self.assertEqual(rows[3]["outcome"], "error")
            self.assertEqual(rows[3]["error_type"], "RuntimeError")
            self.assertTrue(
                all("server_monotonic_s" in row for row in rows)
            )
            self.assertNotIn("sensitive prompt text", events_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(entrypoint.ServerEntrypointError, "overwrite"):
                entrypoint.ServerGrpcEventWriter.open_new(events_path, run_id="event-run")


if __name__ == "__main__":
    unittest.main()
