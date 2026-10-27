import py_compile
from pathlib import Path
import unittest


class CloudIPCPayloadTests(unittest.TestCase):
    def test_http_queue_payloads_do_not_send_torch_storages(self):
        repo = Path(__file__).resolve().parents[1]
        cloud_path = repo / "cloud" / "cloud_service.py"
        engine_path = repo / "src" / "engine.py"
        cloud_text = cloud_path.read_text(encoding="utf-8")
        engine_text = engine_path.read_text(encoding="utf-8")

        self.assertNotIn("torch.tensor(req.draft_output", cloud_text)
        self.assertGreaterEqual(
            cloud_text.count('"draft_output": [int(token) for token in req.draft_output]'),
            2,
        )
        self.assertNotIn('"final_token": new_token', engine_text)
        self.assertIn("if not torch.is_tensor(draft_output):", engine_text)
        self.assertIn("req = tensorize_draft_output(req)", engine_text)
        self.assertGreaterEqual(
            engine_text.count("req = canonicalize_ingress(req)"),
            2,
        )

        py_compile.compile(str(cloud_path), doraise=True)
        py_compile.compile(str(engine_path), doraise=True)

    def test_timing_prefill_cancel_has_cloud_and_worker_handlers(self):
        repo = Path(__file__).resolve().parents[1]
        cloud_text = (repo / "cloud" / "cloud_service.py").read_text(encoding="utf-8")
        engine_text = (repo / "src" / "engine.py").read_text(encoding="utf-8")

        self.assertIn('@app.post("/prefill/cancel")', cloud_text)
        self.assertIn('"control_type": "prefill_cancel"', cloud_text)
        self.assertIn('req.get("control_type") == "prefill_cancel"', engine_text)
        self.assertIn("pending_prefill_cancel", engine_text)

    def test_arrival_protocol_records_before_token_upload(self):
        repo = Path(__file__).resolve().parents[1]
        cloud_text = (repo / "cloud" / "cloud_service.py").read_text(encoding="utf-8")
        edge_text = (repo / "edge" / "edge.py").read_text(encoding="utf-8")
        engine_text = (repo / "src" / "engine.py").read_text(encoding="utf-8")

        self.assertIn('@app.post("/prefill/arrival")', cloud_text)
        self.assertIn("class PrefillArrivalRequest", cloud_text)
        self.assertIn("_prefill_arrivals", cloud_text)
        self.assertIn("def prefill_arrival(", edge_text)
        self.assertIn("_run_first_draft_after_arrival", edge_text)
        self.assertIn('"arrival_monotonic_s"', cloud_text)
        self.assertIn('req.get("arrival_monotonic_s")', engine_text)


if __name__ == "__main__":
    unittest.main()
