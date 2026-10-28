import random
import unittest
from src.progress_control import AcceptanceProfile, AheadController, GammaController, draft_seconds_per_token


class RoundControlTests(unittest.TestCase):
    def test_mixed_gamma_conditional_risk_sets(self):
        p = AcceptanceProfile()
        for a, n in [(0, 1)] * 2 + [(1, 1)] * 3 + [(1, 2)] + [(2, 2)] * 4:
            p.observe(a, n)
        self.assertEqual(p.risk, {1: 10, 2: 5})
        self.assertAlmostEqual(p.survival(1), 0.8)
        self.assertAlmostEqual(p.survival(2), 0.64)
        self.assertAlmostEqual(p.expected_progress(2), 2.44)

    def test_incremental_counts_match_independent_window_recomputation(self):
        p, history = AcceptanceProfile(window=7), []
        rng = random.Random(13)
        for _ in range(100):
            n = rng.randint(1, 16)
            a = rng.randint(0, n)
            history.append((a, n))
            p.observe(a, n)
            survival = 1
            for j in range(1, 17):
                risk = [(a, n) for a, n in history[-7:] if n >= j and a >= j - 1]
                survival *= sum(a >= j for a, _ in risk) / len(risk) if risk else 0.8
                self.assertAlmostEqual(p.survival(j), survival)
            self.assertLessEqual(len(p.rounds), 7)

    def test_residual_slice_conditions_on_accepted_prefix_and_defers_bonus(self):
        p = AcceptanceProfile()
        for _ in range(5):
            p.observe(4, 8)
            p.observe(8, 8)
        self.assertAlmostEqual(p.slice_progress(4, 2, False), 1.5)
        self.assertAlmostEqual(p.slice_progress(4, 2, True), 2.0)
        self.assertNotEqual(p.expected_progress(2), p.slice_progress(4, 2, True))

    def test_full_reuse_never_updates_decode_ema_or_falls_back_to_lag(self):
        c = GammaController()
        valid = dict(local_decode_per_token_s=0.02, fresh_decode_tokens=4, draft_timing_valid=True)
        c.finish('p', accepted=4, proposed=4, request=valid, verify_s=0.1)
        for _ in range(20):
            reused = dict(lag=0.00001, local_decode_per_token_s=None,
                          fresh_decode_tokens=0, draft_timing_valid=False)
            self.assertEqual(draft_seconds_per_token(reused), 0)
            c.finish('p', accepted=4, proposed=4, request=reused, verify_s=0.1)
        self.assertEqual(c.draft_s['p'], 0.02)
        self.assertEqual(len(c.profiles['p'].rounds), 21)
        self.assertEqual(draft_seconds_per_token(dict(lag=0.08, gamma=4, draft_timing_valid=None)), 0.02)

    def test_cooldown_does_not_invent_observations_and_probe_can_recover(self):
        c = AheadController(cooldown_rounds=3)
        for _ in range(4):
            c.observe(4, 0, 0.01, 0.01)
        self.assertEqual(c.state, 'cooldown')
        for _ in range(2):
            c.begin_round(cap=4, expected_wait_s=1, draft_s=0.01)
            self.assertFalse(c.allow(steps=0, cap=4, elapsed_s=0, expected_wait_s=1, draft_s=0.01))
            c.observe(0, 0, 0.01, 0)
        self.assertEqual(len(c.observations), 4)
        c.begin_round(cap=1, expected_wait_s=1, draft_s=0.01)
        self.assertEqual(c.state, 'probe')
        self.assertFalse(c.allow(steps=0, cap=1, elapsed_s=0, expected_wait_s=1, draft_s=0.01))
        c.begin_round(cap=4, expected_wait_s=1, draft_s=0.01)
        self.assertTrue(c.allow(steps=1, cap=4, elapsed_s=0.01, expected_wait_s=1, draft_s=0.01))
        self.assertFalse(c.allow(steps=2, cap=4, elapsed_s=0.02, expected_wait_s=1, draft_s=0.01))
        c.observe(2, 1, 0.01, 0)
        self.assertEqual(c.state, 'enabled')
        self.assertEqual(len(c.observations), 1)


if __name__ == '__main__':
    unittest.main()
