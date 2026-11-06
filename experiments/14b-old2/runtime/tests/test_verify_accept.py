"""Acceptance-rule tests for the optional probability-based verify path.

The greedy mode must stay bit-for-bit equivalent to the loop it replaces in
``Decoding.run_target_process_batching`` (engine.py:1186-1213); the prob mode
implements standard speculative-sampling rejection with an injectable uniform
stream so the decision is deterministic under test.
"""

import math
import random
import unittest

from src.fastsd_scheduler import (
    accept_count,
    plan_draft_prob_sources,
)


def _reference_greedy(draft_tokens, target_top1):
    """Literal transcription of the legacy cloud verify loop's accept rule."""
    for i in range(len(draft_tokens)):
        if int(draft_tokens[i]) != int(target_top1[i]):
            return i
    return len(draft_tokens)


class AcceptCountGreedyTests(unittest.TestCase):
    def test_all_match_accepts_everything(self):
        self.assertEqual(
            accept_count([1, 2, 3, 4], [1, 2, 3, 4], mode="greedy"), 4
        )

    def test_first_mismatch_truncates_prefix(self):
        self.assertEqual(accept_count([1, 2, 3], [1, 9, 3], mode="greedy"), 1)
        self.assertEqual(accept_count([9, 2, 3], [1, 2, 3], mode="greedy"), 0)
        self.assertEqual(accept_count([1, 2, 9], [1, 2, 3], mode="greedy"), 2)

    def test_greedy_ignores_probability_inputs(self):
        self.assertEqual(
            accept_count(
                [1, 2],
                [1, 2],
                None,
                None,
                mode="greedy",
                threshold=0.0,
                uniforms=None,
            ),
            2,
        )

    def test_empty_input_accepts_nothing(self):
        self.assertEqual(accept_count([], [], mode="greedy"), 0)

    def test_matches_reference_loop_on_random_cases(self):
        rng = random.Random(1234)
        vocab = 8
        for _ in range(200):
            length = rng.randint(0, 6)
            draft = [rng.randrange(vocab) for _ in range(length)]
            top1 = [rng.randrange(vocab) for _ in range(length)]
            self.assertEqual(
                accept_count(draft, top1, mode="greedy"),
                _reference_greedy(draft, top1),
                msg=f"draft={draft} top1={top1}",
            )

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            accept_count([1, 2], [1], mode="greedy")


class AcceptCountProbTests(unittest.TestCase):
    def test_ratio_at_least_one_always_accepts(self):
        self.assertEqual(
            accept_count(
                [5],
                [7],
                [0.6],
                [0.5],
                mode="prob",
                uniforms=[0.999],
            ),
            1,
        )

    def test_uniform_equal_to_ratio_accepts(self):
        # Boundary: the rule is ``r <= ratio``, so r == ratio must accept.
        self.assertEqual(
            accept_count([5], [7], [0.25], [0.5], mode="prob", uniforms=[0.5]),
            1,
        )

    def test_first_rejection_truncates_prefix(self):
        # ratios: 1.0 (accept), 0.4 (r=0.9 rejects), third never inspected.
        self.assertEqual(
            accept_count(
                [1, 2, 3],
                [1, 4, 3],
                [0.5, 0.2, 0.5],
                [0.5, 0.5, 0.5],
                mode="prob",
                uniforms=[0.1, 0.9, 0.1],
            ),
            1,
        )

    def test_zero_draft_probability_rejects(self):
        self.assertEqual(
            accept_count(
                [5], [5], [1.0], [0.0], mode="prob", uniforms=[0.0]
            ),
            0,
        )

    def test_zero_target_probability_rejects(self):
        self.assertEqual(
            accept_count(
                [5], [7], [0.0], [0.5], mode="prob", uniforms=[0.0]
            ),
            0,
        )

    def test_nan_probabilities_reject_without_raising(self):
        self.assertEqual(
            accept_count(
                [5], [5], [float("nan")], [0.5], mode="prob", uniforms=[0.0]
            ),
            0,
        )
        self.assertEqual(
            accept_count(
                [5], [5], [0.5], [float("nan")], mode="prob", uniforms=[0.0]
            ),
            0,
        )

    def test_accepts_token_that_is_not_the_target_argmax(self):
        # The core behavioural difference from greedy: a non-top-1 draft token
        # survives when the probability ratio is high enough.
        self.assertEqual(
            accept_count(
                [5], [7], [0.45], [0.5], mode="prob", uniforms=[0.5]
            ),
            1,
        )

    def test_one_hot_distributions_reproduce_greedy(self):
        # --temp 0 degenerates both models to one-hot rows: p_draft == 1 and
        # p_target in {0, 1}.  The prob rule must then agree with greedy for
        # every uniform draw, including r == 0.0.
        draft = [3, 4, 5]
        top1 = [3, 9, 5]
        target_probs = [1.0, 0.0, 1.0]
        draft_probs = [1.0, 1.0, 1.0]
        for r in (0.0, 1e-9, 0.5, 0.9999999):
            self.assertEqual(
                accept_count(
                    draft,
                    top1,
                    target_probs,
                    draft_probs,
                    mode="prob",
                    uniforms=[r, r, r],
                ),
                _reference_greedy(draft, top1),
                msg=f"r={r}",
            )

    def test_uniforms_injection_is_deterministic(self):
        kwargs = dict(
            draft_tokens=[1, 2, 3],
            target_top1=[1, 2, 3],
            target_probs=[0.3, 0.3, 0.9],
            draft_probs=[0.6, 0.6, 0.6],
            mode="prob",
        )
        uniforms = [0.1, 0.9, 0.3]
        first = accept_count(uniforms=uniforms, **kwargs)
        second = accept_count(uniforms=uniforms, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first, 1)

    def test_uniforms_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            accept_count(
                [1, 2], [1, 2], [0.5, 0.5], [0.5, 0.5], mode="prob", uniforms=[0.1]
            )

    def test_prob_mode_requires_probability_vectors(self):
        with self.assertRaises(ValueError):
            accept_count([1], [1], None, None, mode="prob", uniforms=[0.5])
        with self.assertRaises(ValueError):
            accept_count([1, 2], [1, 2], [0.5], [0.5], mode="prob", uniforms=[0.5, 0.5])

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            accept_count([1], [1], mode="uniform")


class AcceptCountThresholdTests(unittest.TestCase):
    def test_threshold_blocks_low_ratio(self):
        # ratio = 0.2, r = 0.1 would accept without the gate.
        self.assertEqual(
            accept_count(
                [5],
                [7],
                [0.1],
                [0.5],
                mode="prob",
                threshold=0.5,
                uniforms=[0.1],
            ),
            0,
        )
        self.assertEqual(
            accept_count(
                [5],
                [7],
                [0.1],
                [0.5],
                mode="prob",
                threshold=0.1,
                uniforms=[0.1],
            ),
            1,
        )

    def test_zero_threshold_disables_gate(self):
        self.assertEqual(
            accept_count(
                [5], [7], [0.01], [0.5], mode="prob", threshold=0.0, uniforms=[0.001]
            ),
            1,
        )

    def test_negative_or_nan_threshold_raises(self):
        with self.assertRaises(ValueError):
            accept_count([5], [7], [0.5], [0.5], mode="prob", threshold=-0.1)
        with self.assertRaises(ValueError):
            accept_count(
                [5], [7], [0.5], [0.5], mode="prob", threshold=float("nan")
            )


class PlanDraftProbSourcesTests(unittest.TestCase):
    def test_full_history_rows_cover_every_token(self):
        # generate(prefix, gamma) leaves prefix_len + gamma - 1 rows.
        self.assertEqual(plan_draft_prob_sources(10, 4, 13), (4, 0))

    def test_full_reuse_has_no_history_rows(self):
        # Reuse round: prefix_len == history_rows + 1 and no forward ran.
        self.assertEqual(
            plan_draft_prob_sources(11, 4, 10, reused_prob_count=4), (0, 4)
        )

    def test_reuse_tail_is_used_when_history_runs_out(self):
        # Rows for the first 2 tokens exist, the rest must come from reuse.
        self.assertEqual(
            plan_draft_prob_sources(11, 4, 12, reused_prob_count=2), (2, 2)
        )

    def test_incomplete_vector_reports_partial_counts(self):
        history_count, reused_count = plan_draft_prob_sources(11, 4, 10)
        self.assertLess(history_count + reused_count, 4)

    def test_zero_tokens(self):
        self.assertEqual(plan_draft_prob_sources(5, 0, 0), (0, 0))


if __name__ == "__main__":
    unittest.main()
