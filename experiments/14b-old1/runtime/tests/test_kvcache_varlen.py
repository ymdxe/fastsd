"""Parity tests: padding-free varlen forward vs legacy padded forward.

Uses a local Llama checkpoint on GPU. Both paths must produce matching per-proc
prob histories, KV caches and sampled next tokens. Skipped when Torch /
Transformers / GPU / the model checkpoint are unavailable.
"""

import unittest
import os
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.kvcache_batching import KVCacheModel_batching
    from src.kvcache_varlen import varlen_generate

    _DEPS = True
except Exception:  # pragma: no cover - dependency gate
    _DEPS = False

MODEL_PATH = os.environ.get("FASTSD_LLAMA_MODEL_PATH", "")


def _cuda_available():
    return _DEPS and bool(MODEL_PATH) and torch.cuda.is_available() and Path(MODEL_PATH).exists()


@unittest.skipUnless(_cuda_available(), "requires Torch + GPU + FASTSD_LLAMA_MODEL_PATH")
class KVCacheVarlenParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        cls.pad_id = cls.tokenizer.pad_token_id
        cls.model = (
            AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
            .to("cuda")
            .eval()
        )
        if cls.model.config.model_type != "llama":
            raise ValueError("FASTSD_LLAMA_MODEL_PATH must refer to a Llama checkpoint")
        cls.vocab_size = len(cls.tokenizer)

    def _fresh_manager(self):
        manager = KVCacheModel_batching(self.model, temperature=0.0)
        manager.vocab_size = self.vocab_size
        return manager

    def _assert_state_equal(self, m1, m2, pids, atol=1e-4, rtol=1e-4):
        for pid in pids:
            self.assertEqual(
                m1._past_key_values[pid].get_seq_length(),
                m2._past_key_values[pid].get_seq_length(),
                f"cache seq_len mismatch for {pid}",
            )
            p1 = m1._prob_history[pid]
            p2 = m2._prob_history[pid]
            self.assertEqual(p1.shape, p2.shape, f"prob shape mismatch for {pid}")
            self.assertTrue(
                torch.allclose(p1, p2, atol=atol, rtol=rtol),
                f"prob history mismatch for {pid}: max abs diff {(p1 - p2).abs().max().item():.3e}",
            )
            for layer_idx, ((k1, v1), (k2, v2)) in enumerate(
                zip(
                    m1._past_key_values[pid].to_legacy_cache(),
                    m2._past_key_values[pid].to_legacy_cache(),
                )
            ):
                self.assertTrue(
                    torch.allclose(k1, k2, atol=atol, rtol=rtol),
                    f"KV layer {layer_idx} key mismatch for {pid}: {(k1 - k2).abs().max().item():.3e}",
                )
                self.assertTrue(
                    torch.allclose(v1, v2, atol=atol, rtol=rtol),
                    f"KV layer {layer_idx} value mismatch for {pid}: {(v1 - v2).abs().max().item():.3e}",
                )

    def _encode(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    def test_prefill_parity_mixed_lengths(self):
        prompts = [
            "def fib(n):",
            "Write a function that returns the sum of two integers.",
            "What is the capital of France? Explain briefly.",
        ]
        encs = [self._encode(p) for p in prompts]
        lens = [e.shape[1] for e in encs]
        pids = ["a", "b", "c"]

        # ---- legacy padded path ----
        m_pad = self._fresh_manager()
        max_T = max(lens)
        padded = []
        for e in encs:
            pad = torch.full((1, max_T - e.shape[1]), self.pad_id, dtype=torch.long, device="cuda")
            padded.append(torch.cat([e, pad], dim=1))
        x_batch = torch.cat(padded, dim=0)
        new_x_pad = m_pad.generate(
            x_batch, 1, proc_ids=pids, pad_token_id=self.pad_id,
            is_prefill=True, input_lens=lens,
        )

        # ---- padding-free varlen path ----
        m_var = self._fresh_manager()
        for pid in pids:
            m_var.reset(pid)  # engine prefill resets fresh rows before admission
        new_x_var = varlen_generate(
            m_var, self.model, encs, pids, self.pad_id, is_prefill=True
        )

        self.assertEqual([t.tolist() for t in new_x_pad], [t.tolist() for t in new_x_var])
        self._assert_state_equal(m_pad, m_var, pids)

    def test_verify_parity_tail_only_different_cached_lengths(self):
        prompts = [
            "def fib(n):",
            "What is the capital of France?",
        ]
        pids = ["x", "y"]
        drafts = [
            self._encode(" return 0"),
            self._encode(" Paris is"),
        ]

        def build_prefill(manager):
            encs = [self._encode(p) for p in prompts]
            lens = [e.shape[1] for e in encs]
            max_T = max(lens)
            padded = []
            for e in encs:
                pad = torch.full((1, max_T - e.shape[1]), self.pad_id, dtype=torch.long, device="cuda")
                padded.append(torch.cat([e, pad], dim=1))
            manager.generate(
                torch.cat(padded, dim=0), 1, proc_ids=pids, pad_token_id=self.pad_id,
                is_prefill=True, input_lens=lens,
            )
            # engine rolls back after prefill; keep the full prompt cache here
            # so both managers start verify from identical state.

        # ---- legacy padded verify (tail_only: fake prefix + tail) ----
        m_pad = self._fresh_manager()
        build_prefill(m_pad)
        x_batch = []
        verify_input_lens = []
        for pid, d in zip(pids, drafts):
            cached_len = m_pad._past_key_values[pid].get_seq_length()
            pad = torch.full((1, cached_len), self.pad_id, dtype=torch.long, device="cuda")
            x_batch.append(torch.cat((pad, d), dim=1))
            verify_input_lens.append(cached_len + d.shape[1])
        max_T = max(x.shape[1] for x in x_batch)
        padded_verify = []
        for x in x_batch:
            pad = torch.full((1, max_T - x.shape[1]), self.pad_id, dtype=torch.long, device="cuda")
            padded_verify.append(torch.cat((x, pad), dim=1))
        nt_pad = m_pad.generate(
            torch.cat(padded_verify, dim=0), 1, proc_ids=pids,
            pad_token_id=self.pad_id, is_prefill=False, input_lens=verify_input_lens,
        )

        # ---- padding-free varlen verify (tail-only: pass the tail as-is) ----
        m_var = self._fresh_manager()
        build_prefill(m_var)
        nt_var = varlen_generate(
            m_var, self.model, drafts, pids, self.pad_id, is_prefill=False
        )

        # NOTE: the padded path samples its returned token from
        # ``not_cached_q[i, -1, :]`` which may fall on a padding position
        # (the engine ignores this return value for verify), so sampled
        # tokens are not compared here; prob history and KV cache are the
        # authoritative parity checks.
        self._assert_state_equal(m_pad, m_var, pids)

    def test_prefill_continuation_parity(self):
        prompt = "Write a Python function that computes the greatest common divisor of two positive integers."
        pid = "cont"
        full = self._encode(prompt)
        full_len = full.shape[1]
        chunk1_len = 12  # < full_len

        # ---- legacy padded path: prefill chunk1 then continuation chunk2 ----
        m_pad = self._fresh_manager()
        m_pad.generate(
            full[:, :chunk1_len], 1, proc_ids=[pid], pad_token_id=self.pad_id,
            is_prefill=True, input_lens=[chunk1_len],
        )
        m_pad.forward_new_tokens(
            full, proc_ids=[pid], pad_token_id=self.pad_id, input_lens=[full_len],
        )

        # ---- varlen path: same two steps with residuals ----
        m_var = self._fresh_manager()
        m_var.reset(pid)
        varlen_generate(
            m_var, self.model, [full[:, :chunk1_len]], [pid], self.pad_id, is_prefill=True
        )
        cached = m_var._past_key_values[pid].get_seq_length()
        self.assertEqual(cached, chunk1_len)  # prefill KV covers the prompt only
        varlen_generate(
            m_var, self.model, [full[:, cached:]], [pid], self.pad_id, is_prefill=True
        )

        self._assert_state_equal(m_pad, m_var, [pid], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
