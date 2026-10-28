"""Exercise the production worker, slicing, cache windows and session release."""
import queue
import threading
import types
import unittest
from collections import defaultdict

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.engine import Decoding
    DEPS = True
except ImportError:
    DEPS = False


@unittest.skipUnless(DEPS, "requires model runtime dependencies")
class ReviewEngineTests(unittest.TestCase):
    def run_worker_case(self, backend, mode="fastsd", device="cpu", cycles=16, adaptive=False):
        class Runner(Decoding):
            def load_data(self): pass
            def preprocess(self, x): return x
            def postprocess(self, x, y): return y
            def eval(self): pass
        torch.manual_seed(77)
        config = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=128, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to(device).eval()
        engine = object.__new__(Runner)
        engine.args = types.SimpleNamespace(target_model="tiny-test", vocab_size=31,
            temp=0.0, top_k=0, top_p=0.0, kv_batch_mode=backend, max_tokens=32,
            gamma=8, token_budget=4, batch_size=2, min_prefill_chunk_tokens=1,
            prefill_chunk_quantum=2, server_sched_mode=mode, pipeline_gamma_adapt=adaptive,
            enable_pipeline=False,
            session_idle_timeout=30, async_kv_transfer=device != "cpu")
        engine.seed = 1
        engine.last_generation_metrics = {}
        engine._load_target_model_for_service = lambda *args: model
        engine._service_target_device = lambda: device
        engine._maybe_start_energy_service = lambda: None
        engine.color_print = lambda *args: None
        ingress, responses, errors = queue.Queue(), defaultdict(queue.Queue), []
        def run():
            try:
                engine.run_target_process_batching(types.SimpleNamespace(pad_token_id=0), ingress, responses)
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        counter = 0
        def request(payload):
            nonlocal counter
            counter += 1
            key = str(counter)
            payload = dict(payload, proc_id="session", response_key=key)
            ingress.put(payload)
            try:
                result = responses[key].get(timeout=10)
            except queue.Empty:
                if errors:
                    raise errors[0]
                raise
            self.assertNotIn("error", result)
            return result
        try:
            for _ in range(cycles):
                prefix = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
                self.assertEqual(request({"task_type": "prefill", "draft_output": prefix[0].tolist(),
                    "prefix_len": prefix.shape[1], "gamma": 0})["status"], "prefill_ok")
                seq = prefix.clone()
                with torch.no_grad():
                    for _ in range(9):
                        token = model(seq).logits[:, -1].argmax(-1).reshape(1, 1)
                        seq = torch.cat((seq, token), 1)
                result = request({"task_type": "verify", "draft_output": seq[0, 7:15].tolist(),
                    "prefix_len": 7, "gamma": 8, "tail_only": True, "verify_method": "greedy"})
                self.assertEqual("suggested_gamma" in result, adaptive)
                for metric in ("forward_gpu_ms", "batch_start_delay_ms", "exposed_h2d_stall_ms"):
                    self.assertIn(metric, result)
                    self.assertGreaterEqual(result[metric], 0)
                self.assertEqual(result["accepted"], 15)
                self.assertEqual(result["final_token"], seq[0, 15].item())
                self.assertEqual(request({"control_type": "session_release"})["status"], "released")
        finally:
            ingress.put(None)
            worker.join(10)
        self.assertFalse(worker.is_alive())
        if errors:
            raise errors[0]
        metrics = engine.last_generation_metrics["fastsd_scheduler"]
        self.assertEqual(metrics["cache_sessions"], 0)
        self.assertEqual(metrics["probability_sessions"], 0)
        self.assertGreaterEqual(metrics["released_sessions"], cycles)
        self.assertLessEqual(metrics["peak_cache_sessions"], 1)
        self.assertEqual(metrics["gamma_rounds"], cycles)

    def test_varlen_chunked_rounds_and_repeated_sessions(self):
        self.run_worker_case("varlen")

    def test_padded_chunked_rounds_and_repeated_sessions(self):
        self.run_worker_case("padded")

    def test_all_service_modes_share_round_controller_without_pipeline(self):
        for mode in ("fastsd", "vanilla", "pipeline"):
            with self.subTest(mode=mode):
                self.run_worker_case("varlen", mode, cycles=2, adaptive=True)

    def test_baseline_release(self):
        self.run_worker_case("varlen", "vanilla")

    @unittest.skipUnless(DEPS and torch.cuda.is_available(), "requires CUDA")
    def test_cuda_worker_with_async_offload_and_reload(self):
        self.run_worker_case("varlen", device="cuda:0", cycles=2)


if __name__ == "__main__":
    unittest.main()
