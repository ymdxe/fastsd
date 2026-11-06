"""Pure-Python contracts; no packages, model files or server are required."""

import unittest

from single_chain_contract import (
    accepted_chain_length, clipped_chunk_length, scratch_slots, validate_chain_layout,
)


class SingleChainContractTests(unittest.TestCase):
    def test_gamma4_and_gamma8_layouts(self):
        for gamma in (4, 8):
            with self.subTest(gamma=gamma):
                positions = list(range(17, 18 + gamma))
                root = validate_chain_layout(list(range(gamma + 1)), positions, positions,
                                             positions[:-1], gamma=gamma, max_len=2048)
                self.assertEqual(root, 17)
                scratch = scratch_slots(root=root, logical_width=gamma + 1,
                                        physical_width=9, max_len=2048)
                self.assertFalse(set(scratch) & set(range(root + gamma + 1)))
                self.assertEqual(len(set(scratch)), 8 - gamma)

    def test_branches_wrong_width_and_position_are_rejected(self):
        for parents, positions, inputs in [
            ([6, 6, 8, 9], list(range(6, 11)), [0] * 5),
            (list(range(6, 10)), [6, 7, 8, 10, 11], [0] * 5),
            (list(range(6, 10)), list(range(6, 11)), [0] * 9),
        ]:
            with self.assertRaises(ValueError):
                validate_chain_layout(inputs, positions, list(range(6, 11)), parents,
                                      gamma=4, max_len=2048)

    def test_scratch_cannot_overlap_live_chain(self):
        with self.assertRaises(ValueError):
            scratch_slots(root=2040, logical_width=5, physical_width=9, max_len=2048)

    def test_first_mismatch_stops_acceptance(self):
        self.assertEqual(accepted_chain_length([1, 2, 3, 4], [9, 2, 3, 4, 5]), 0)
        self.assertEqual(accepted_chain_length([1, 2, 3, 4], [1, 2, 9, 4, 5]), 2)
        self.assertEqual(accepted_chain_length([1, 2, 3, 4], [1, 2, 3, 4, 5]), 4)

    def test_padded_predictions_are_not_accepted_as_real_responses(self):
        with self.assertRaises(ValueError):
            accepted_chain_length([1, 2, 3, 4], list(range(9)))

    def test_both_common_eos_tokens_and_cap(self):
        for eos in (151645, 151643):
            self.assertEqual(clipped_chunk_length([4, eos, 6], 10, [151645, 151643]), (2, True))
            self.assertEqual(clipped_chunk_length([4, eos, 6], 1, [151645, 151643]), (1, False))
        self.assertEqual(clipped_chunk_length([1, 2, 3], 2, [151645, 151643]), (2, False))


if __name__ == "__main__":
    unittest.main()
