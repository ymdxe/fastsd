"""Empirical acceptance survival and conservative speculative-length control.

Queue delay is never interpreted as spare compute capacity. The estimates are
heuristics and expose observations for calibration against measured batches.
"""

from collections import deque
import math


class AcceptanceProfile:
    def __init__(self, window=128):
        self.rounds = deque(maxlen=window)

    def observe(self, accepted, proposed):
        accepted, proposed = int(accepted), int(proposed)
        if not 0 <= accepted <= proposed or proposed <= 0:
            raise ValueError("invalid acceptance observation")
        self.rounds.append((accepted, proposed))

    def expected_progress(self, gamma):
        survival, progress = 1.0, 1.0
        for position in range(1, int(gamma) + 1):
            # All-accept short rounds are right-censored. A rejection at an
            # earlier position is still an observed failure for this position.
            observed = [(a, n) for a, n in self.rounds if n >= position or a < n]
            if observed:
                probability = sum(a >= position for a, _ in observed) / len(observed)
            else:
                probability = survival * 0.8
            survival = min(survival, probability)
            progress += survival
        return progress


def draft_seconds_per_token(request):
    def nonnegative(value):
        try:
            number = float(value)
            return max(0.0, number) if math.isfinite(number) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0
    direct = nonnegative(request.get("local_decode_per_token_s"))
    if direct > 0:
        return direct
    gamma = max(1, nonnegative(request.get("logical_gamma", request.get("gamma", 1))))
    return nonnegative(request.get("lag", 0.0)) / gamma


def choose_gamma(profile, *, current, draft_s, verify_s, rtt_s=0.0, queue_s=0.0,
                 minimum=1, maximum=16, step=2, hysteresis=0.1):
    current = max(minimum, min(maximum, int(current)))
    if draft_s <= 0 or verify_s <= 0:
        return current
    # This first online model has a measured fixed/linear split. It deliberately
    # excludes queue delay from the reward denominator to prevent queue growth
    # from encouraging larger speculative rounds.
    def rate(gamma):
        gpu_s = verify_s * (0.5 + 0.5 * gamma / current)
        return profile.expected_progress(gamma) / max(1e-9, draft_s * gamma + gpu_s + rtt_s)
    candidates = {minimum, maximum, current} | {g for g in (1, 2, 4, 8, 16) if minimum <= g <= maximum}
    best = max(candidates, key=lambda g: (rate(g), -g))
    if rate(best) < rate(current) * (1 + hysteresis):
        return current
    if queue_s > verify_s and best > current:
        return current
    return max(minimum, min(maximum, max(current - step, min(current + step, best))))


class BatchCostModel:
    """Small online profile table keyed by operation and shape buckets."""

    def __init__(self):
        self.table = {}

    @staticmethod
    def work(rows):
        return sum(tokens * (1.0 + context / 4096.0) for context, tokens in rows)

    @staticmethod
    def key(kind, rows):
        bucket = lambda n: max(0, int(n)).bit_length()
        return kind, bucket(max((l for l, _ in rows), default=0)), bucket(sum(n for _, n in rows)), bucket(len(rows))

    def observe(self, kind, rows, seconds):
        if seconds <= 0 or not math.isfinite(seconds) or not rows:
            return
        key = self.key(kind, rows)
        value = seconds / max(1.0, self.work(rows))
        previous, count = self.table.get(key, (value, 0))
        self.table[key] = (0.8 * previous + 0.2 * value, count + 1)

    def estimate(self, kind, rows):
        if not rows:
            return 0.0
        key = self.key(kind, rows)
        value = self.table.get(key)
        if value is None:
            candidates = [(k, v) for k, v in self.table.items() if k[0] == kind]
            value = min(candidates, key=lambda kv: sum(abs(a-b) for a,b in zip(kv[0][1:],key[1:])))[1] if candidates else (0.001, 0)
        return max(1e-6, value[0] * self.work(rows))


class AheadController:
    def __init__(self):
        self.observations = deque(maxlen=32)

    def allow(self, *, steps, cap, elapsed_s, expected_wait_s, draft_s):
        if steps >= cap:
            return False
        if expected_wait_s > 0 and draft_s > 0 and elapsed_s + draft_s > expected_wait_s:
            return False
        if len(self.observations) >= 4:
            saved = sum(reused * cost for generated, reused, cost, stall in self.observations)
            waste = sum((generated - reused) * cost + stall for generated, reused, cost, stall in self.observations)
            if saved <= waste:
                return False
        return True

    def observe(self, generated, reused, draft_s, response_stall_s):
        self.observations.append((generated, reused, draft_s, response_stall_s))
