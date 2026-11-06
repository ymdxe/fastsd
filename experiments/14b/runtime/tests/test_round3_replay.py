import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('replay_prefetch_pair',
    Path(__file__).resolve().parents[1] / 'scripts/replay_prefetch_pair.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class ReplayEvidenceTests(unittest.TestCase):
    def test_pair_bootstrap_preserves_covariance(self):
        pairs = [(10 + index, 10 + index + .1) for index in range(40)]
        result = replay.paired_summary(pairs, resamples=200)
        self.assertAlmostEqual(result['mean_delta_ms'], .1)
        for endpoint in result['ci95']['mean_delta_ms']:
            self.assertAlmostEqual(endpoint, .1)
        self.assertEqual(result['pairs'], 40)

    def test_acceptance_requires_evidence_beyond_forward_percentiles(self):
        aa = replay.paired_summary([(10, 10)] * 30, resamples=20)
        ad = replay.paired_summary([(10, 10.05)] * 30, resamples=20)
        kwargs = dict(actual_overlap=True, output_equal=True,
                      h2d_wait_reduced=True, total_host_wait_reduced=True, startup_ok=True)
        self.assertEqual(replay.assess_forward(aa, ad, **kwargs), 'passed_measured_shape')
        for key, expected in [('actual_overlap', 'failed_no_measured_memcpy_kernel_overlap'),
                              ('output_equal', 'failed_output_mismatch'),
                              ('h2d_wait_reduced', 'failed_no_exposed_h2d_reduction'),
                              ('total_host_wait_reduced', 'failed_no_total_next_batch_host_wait_reduction'),
                              ('startup_ok', 'failed_forward_start_delay')]:
            self.assertEqual(replay.assess_forward(aa, ad, **dict(kwargs, **{key: False})), expected)

    def test_baseline_noise_cannot_be_reported_as_success(self):
        aa = replay.paired_summary([(10, 10.5)] * 30, resamples=20)
        ad = replay.paired_summary([(10, 10)] * 30, resamples=20)
        self.assertEqual(replay.assess_forward(aa, ad, actual_overlap=True,
            output_equal=True, h2d_wait_reduced=True, total_host_wait_reduced=True,
            startup_ok=True), 'inconclusive_baseline_noise')

    def test_trace_uses_memcpy_kernel_activities_and_unions_overlap(self):
        data = {'traceEvents': [
            {'ph': 'X', 'cat': 'user_annotation', 'name': 'round3_target_forward', 'pid': 1, 'tid': 10, 'ts': 0, 'dur': 4, 'args': {'External id': 1}},
            {'ph': 'X', 'cat': 'cpu_op', 'name': 'aten::mm', 'pid': 1, 'tid': 10, 'ts': 1, 'dur': 2, 'args': {'External id': 2}},
            {'ph': 'X', 'cat': 'user_annotation', 'name': 'round3_speculative_h2d', 'pid': 1, 'tid': 20, 'ts': 0, 'dur': 4, 'args': {'External id': 3}},
            {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaMemcpyAsync', 'pid': 1, 'tid': 20, 'ts': 1, 'dur': 2, 'args': {'correlation': 4}},
            {'ph': 'X', 'cat': 'kernel', 'name': 'matmul', 'ts': 0, 'dur': 10, 'args': {'External id': 2}},
            {'ph': 'X', 'cat': 'kernel', 'name': 'attention', 'ts': 5, 'dur': 10, 'args': {'External id': 2}},
            {'ph': 'X', 'cat': 'kernel', 'name': 'next_forward', 'ts': 5, 'dur': 50, 'args': {'External id': 99}},
            {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD', 'ts': 7, 'dur': 4, 'args': {'correlation': 4}},
            {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD', 'ts': 0, 'dur': 50, 'args': {'correlation': 99}},
            {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'Memcpy DtoH', 'ts': 0, 'dur': 30},
            {'ph': 'X', 'cat': 'user_annotation', 'name': 'large_copy_span', 'ts': 0, 'dur': 30}]}
        with patch.object(Path, 'read_text', return_value=json.dumps(data)):
            result = replay.trace_overlap('profiler.json')
        self.assertEqual(result['memcpy_kernel_overlap_us'], 4)
        self.assertEqual(result['h2d_activities'], 1)
        self.assertEqual(result['kernel_activities'], 2)
        self.assertEqual(result['attribution'], 'correlated_labeled_activities')

    def test_unattributed_background_copy_cannot_establish_overlap(self):
        events = [{'ph': 'X', 'cat': 'kernel', 'name': 'kernel', 'ts': 0, 'dur': 10},
                  {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD', 'ts': 0, 'dur': 10}]
        with patch.object(Path, 'read_text', return_value=json.dumps({'traceEvents': events})):
            result = replay.trace_overlap('profiler.json')
        self.assertEqual(result['memcpy_kernel_overlap_us'], 0)
        self.assertEqual(result['attribution'], 'incomplete_activity_attribution')

    def test_production_forward_prefill_and_continuation_match_llama(self):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
            from src.kvcache_batching import KVCacheModel_batching
        except ImportError:
            self.skipTest('requires Torch and Transformers')

        torch.manual_seed(31)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, attention_dropout=0.0)
        config._attn_implementation = 'sdpa'
        model = LlamaForCausalLM(config).eval()
        manager = KVCacheModel_batching(model, temperature=0,
                                        attention_backend='sdpa_per_sequence')
        manager.vocab_size = config.vocab_size
        manager.bounded_probabilities = True
        prefix = torch.tensor([[3, 7, 9, 4, 2]])
        tail = torch.tensor([[6, 11]])
        sequence = prefix

        for residual, prefill in ((prefix, True), (tail, False)):
            if not prefill:
                sequence = torch.cat((sequence, residual), dim=1)
            with torch.no_grad():
                reference = model(sequence, use_cache=True)
            output = replay.production_forward(manager, model, [residual], ['request'],
                                               is_prefill=prefill)
            expected_next = reference.logits[:, -1].argmax(-1)
            torch.testing.assert_close(output[0][:-1], residual[0])
            torch.testing.assert_close(output[0][-1:], expected_next)
            self.assertEqual(manager._prob_start['request'], prefix.shape[1] - 1)
            expected_probabilities = torch.nn.functional.one_hot(
                reference.logits[:, prefix.shape[1] - 1:].argmax(-1),
                num_classes=config.vocab_size).float()
            torch.testing.assert_close(manager._prob_history['request'], expected_probabilities)
            actual_cache = manager._past_key_values['request']
            self.assertEqual(actual_cache.get_seq_length(), sequence.shape[1])
            for actual, expected in zip(actual_cache.to_legacy_cache(),
                                        reference.past_key_values.to_legacy_cache()):
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(actual_tensor, expected_tensor,
                                               atol=2e-6, rtol=2e-5)

    def test_shape_parser_accepts_prefill_continuation_and_rejects_zero_tokens(self):
        self.assertEqual(replay.parse_shape('verify:16:8:128'), ('verify', 16, 8, 128))
        self.assertEqual(replay.parse_shape('prefill:1:128:128'), ('prefill', 1, 128, 128))
        with self.assertRaises(Exception):
            replay.parse_shape('prefill:1:0:128')


if __name__ == '__main__':
    unittest.main()
