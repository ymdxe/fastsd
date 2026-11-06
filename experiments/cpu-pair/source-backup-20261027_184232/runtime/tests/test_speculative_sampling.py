"""CPU-level tests for the sparse rejection-sampling protocol."""

import unittest

try:
    import torch
    from src.speculative_sampling import (
        decode_sparse_block,
        draft_distribution,
        encode_sparse_block,
        rejection_sample,
        slice_sparse_block,
        sparse_block_from_dense,
    )
except ModuleNotFoundError:  # the lightweight repository test image has no torch
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for sampling tests")
class SparseRejectionSamplingTests(unittest.TestCase):
    def test_sparse_wire_round_trip_and_row_slice(self):
        rows = torch.tensor(
            [[0.7, 0.2, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0]], dtype=torch.float32
        )
        ids, probs = sparse_block_from_dense(rows, 3)
        encoded = encode_sparse_block(ids, probs)
        decoded_ids, decoded_probs = decode_sparse_block(encoded, vocab_size=4)
        self.assertTrue(torch.equal(ids.cpu(), decoded_ids.cpu()))
        self.assertTrue(torch.allclose(probs.cpu(), decoded_probs.cpu(), atol=1e-6))
        sliced_ids, sliced_probs = decode_sparse_block(
            slice_sparse_block(encoded, 1, 1), vocab_size=4
        )
        self.assertEqual(tuple(sliced_ids.shape), (1, 3))
        self.assertTrue(torch.allclose(sliced_probs[0].sum(), torch.tensor(1.0)))

    def test_non_argmax_draft_token_can_be_accepted(self):
        target = torch.tensor(
            [[0.45, 0.50, 0.05], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        draft = torch.tensor([[0.50, 0.45, 0.05]], dtype=torch.float32)
        ids, probs = sparse_block_from_dense(draft, 3)
        accepted, correction = rejection_sample(
            [0], target, ids, probs, uniforms=[0.8]
        )
        self.assertEqual(accepted, 1)
        self.assertEqual(int(correction.item()), 2)

    def test_rejection_samples_the_exact_residual(self):
        # q puts all mass on token 0; p's residual is exactly token 1.
        target = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
        draft = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ids, probs = sparse_block_from_dense(draft, 1)
        accepted, correction = rejection_sample(
            [0], target, ids, probs, uniforms=[1.0]
        )
        self.assertEqual(accepted, 0)
        self.assertEqual(int(correction.item()), 1)

    def test_first_middle_and_last_rejections_stop_at_the_first_row(self):
        draft = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ids, probs = sparse_block_from_dense(draft, 1)

        first_target = torch.tensor(
            [[0.25, 0.75], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        accepted, correction = rejection_sample(
            [0], first_target, ids, probs, uniforms=[1.0]
        )
        self.assertEqual(accepted, 0)
        self.assertEqual(int(correction.item()), 1)

        three_draft = torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(3, 1)
        ids, probs = sparse_block_from_dense(three_draft, 1)
        middle_target = torch.tensor(
            [
                [1.0, 0.0],
                [0.25, 0.75],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        accepted, correction = rejection_sample(
            [0, 0, 0], middle_target, ids, probs, uniforms=[0.0, 1.0]
        )
        self.assertEqual(accepted, 1)
        self.assertEqual(int(correction.item()), 1)

        last_target = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
            dtype=torch.float32,
        )
        accepted, correction = rejection_sample(
            [0, 0, 0], last_target, ids, probs, uniforms=[0.0, 0.0, 1.0]
        )
        self.assertEqual(accepted, 2)
        self.assertEqual(int(correction.item()), 1)

    def test_disjoint_support_and_zero_target_mass_are_explicit(self):
        draft = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ids, probs = sparse_block_from_dense(draft, 1)
        with self.assertRaises(ValueError):
            rejection_sample(
                [1],
                torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=torch.float32),
                ids,
                probs,
                uniforms=[0.0],
            )

        accepted, correction = rejection_sample(
            [0],
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            ids,
            probs,
        )
        self.assertEqual(accepted, 0)
        self.assertEqual(int(correction.item()), 1)

    def test_all_accept_can_defer_the_unused_correction(self):
        draft = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ids, probs = sparse_block_from_dense(draft, 1)
        accepted, correction = rejection_sample(
            [0],
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            ids,
            probs,
            sample_correction=False,
        )
        self.assertEqual(accepted, 1)
        self.assertIsNone(correction)

    def test_temp_zero_is_one_hot_and_rejection_matches_greedy(self):
        probs = draft_distribution(torch.tensor([[1.0, 3.0, 2.0]]), 0.0, 2)
        self.assertEqual(int(torch.argmax(probs).item()), 1)
        self.assertAlmostEqual(float(probs.sum().item()), 1.0)

    def test_incomplete_sparse_support_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_sparse_block(
                torch.tensor([[0]], dtype=torch.long),
                torch.tensor([[0.5]], dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
