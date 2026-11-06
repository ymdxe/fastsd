"""Empirical acceptance survival and conservative speculative-length control.

Queue delay is never interpreted as spare compute capacity. The estimates are
heuristics and expose observations for calibration against measured batches.
"""

from collections import deque
import math


class AcceptanceProfile:
    """Conditional risk sets with right censoring, updated once per round."""

    def __init__(self, window=128):
        self.rounds = deque(maxlen=window)
        self.risk, self.success = {}, {}
        self._survival = [1.0]
        self._progress = [1.0]
        self._slices = {}

    def _count(self, accepted, proposed, delta):
        # At position j a round is at risk iff j was proposed and the prior
        # j-1 candidates survived. Earlier rejections do not re-enter risk.
        for j in range(1, min(proposed, accepted + 1) + 1):
            self.risk[j] = self.risk.get(j, 0) + delta
            self.success[j] = self.success.get(j, 0) + delta * (accepted >= j)
            if self.risk[j] == 0:
                self.risk.pop(j)
                self.success.pop(j)

    def observe(self, accepted, proposed):
        accepted, proposed = int(accepted), int(proposed)
        if not 0 <= accepted <= proposed or proposed <= 0:
            raise ValueError("invalid acceptance observation")
        if len(self.rounds) == self.rounds.maxlen:
            self._count(*self.rounds[0], -1)
        self.rounds.append((accepted, proposed))
        self._count(accepted, proposed, 1)
        self._survival, self._progress, self._slices = [1.0], [1.0], {}

    def conditional(self, position):
        n = self.risk.get(position, 0)
        return self.success[position] / n if n else 0.8

    def survival(self, gamma):
        while len(self._survival) <= int(gamma):
            j = len(self._survival)
            self._survival.append(self._survival[-1] * self.conditional(j))
            self._progress.append(self._progress[-1] + self._survival[-1])
        return self._survival[int(gamma)]

    def expected_progress(self, gamma):
        self.survival(gamma)
        return self._progress[int(gamma)]

    def slice_progress(self, offset, count, final):
        key = int(offset), int(count), bool(final)
        if key not in self._slices:
            survival, accepted = 1.0, 0.0
            for j in range(key[0] + 1, key[0] + key[1] + 1):
                survival *= self.conditional(j)
                accepted += survival
            # Nonfinal all-accept slices defer the bonus token. A rejection
            # terminates the round and still commits one correction token.
            self._slices[key] = accepted + (1.0 if final else 1.0 - survival)
        return self._slices[key]


def draft_seconds_per_token(request):
    def nonnegative(value):
        try:
            number = float(value)
            return max(0.0, number) if math.isfinite(number) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0
    if request.get("draft_timing_valid") is False:
        return 0.0
    fresh = request.get("fresh_decode_tokens")
    if fresh is not None and nonnegative(fresh) <= 0:
        return 0.0
    direct = nonnegative(request.get("local_decode_per_token_s"))
    if direct > 0:
        return direct
    if fresh is not None or request.get("draft_timing_valid") is not None:
        return 0.0  # new protocol never infers physical decode speed from lag
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


class GammaController:
    """One update at the logical-round boundary, shared by every service mode."""

    def __init__(self, *, policy="progress", adaptive=True, minimum=1,
                 maximum=16, step=2, alpha=0.2):
        self.policy, self.adaptive = policy, adaptive
        self.minimum, self.maximum, self.step, self.alpha = minimum, maximum, step, alpha
        self.profiles, self.draft_s, self.verify_s, self.rtt_s = {}, {}, {}, {}
        self.rounds = 0

    def _ema(self, store, pid, value):
        if math.isfinite(value) and value >= 0:
            store[pid] = (1 - self.alpha) * store.get(pid, value) + self.alpha * value
        return store.get(pid, 0.0)

    def forget(self, pid):
        for store in (self.profiles, self.draft_s, self.verify_s, self.rtt_s):
            store.pop(pid, None)

    def finish(self, pid, *, accepted, proposed, request, verify_s, queue_s=0):
        profile = self.profiles.setdefault(pid, AcceptanceProfile())
        profile.observe(accepted, proposed)
        self.rounds += 1
        speed = draft_seconds_per_token(request)
        if speed > 0:
            self._ema(self.draft_s, pid, speed)
        service = self._ema(self.verify_s, pid, max(0, verify_s))
        try:
            rtt = max(0, float(request.get("transport_rtt", 0) or 0))
        except (ValueError, TypeError, OverflowError):
            rtt = 0
        rtt = self._ema(self.rtt_s, pid, rtt)
        output = {"expected_progress": profile.expected_progress(proposed)}
        if not self.adaptive:
            return output
        speed = self.draft_s.get(pid, 0)
        if self.policy == "progress":
            suggested = choose_gamma(profile, current=proposed, draft_s=speed,
                verify_s=service, rtt_s=rtt, queue_s=queue_s, minimum=self.minimum,
                maximum=self.maximum, step=self.step)
        elif self.policy == "legacy":
            suggested = round(min(1e9, (service + rtt) / speed)) if speed > 0 else proposed
            suggested = max(proposed - self.step, min(proposed + self.step, suggested))
            suggested = max(self.minimum, min(self.maximum, suggested))
        else:
            raise ValueError("unknown gamma policy")
        output["suggested_gamma"] = suggested
        return output


class AheadController:
    def __init__(self, cooldown_rounds=4):
        self.observations = deque(maxlen=32)
        self.state = "enabled"
        self.cooldown_rounds = int(cooldown_rounds)
        self.remaining = 0
        self._round_allowed = None

    def begin_round(self, *, cap, expected_wait_s, draft_s):
        self._round_allowed = True
        if self.state == "cooldown":
            self.remaining = max(0, self.remaining - 1)
            if self.remaining:
                self._round_allowed = False
                return
            self.state = "probe"
        # Probe must include a bridge AND a candidate; a bridge-only attempt
        # cannot observe candidate reuse. Keep waiting for an adequate window.
        if self.state == "probe" and (cap < 2 or
                (expected_wait_s > 0 and draft_s > 0 and 2 * draft_s > expected_wait_s)):
            self._round_allowed = False

    def allow(self, *, steps, cap, elapsed_s, expected_wait_s, draft_s):
        if self._round_allowed is None:
            self.begin_round(cap=cap, expected_wait_s=expected_wait_s, draft_s=draft_s)
        if not self._round_allowed:
            return False
        limit = min(cap, 2) if self.state == "probe" else cap
        if steps >= limit:
            return False
        return not (expected_wait_s > 0 and draft_s > 0 and elapsed_s + draft_s > expected_wait_s)

    def observe(self, generated, reused, draft_s, response_stall_s):
        self._round_allowed = None
        if generated <= 0:
            return  # no invented zero-yield observation while paused
        value = (generated, reused, draft_s, response_stall_s)
        if self.state == "probe":
            if reused * draft_s > (generated - reused) * draft_s + response_stall_s:
                self.state = "enabled"
                self.observations.clear()
            elif reused > 0 and response_stall_s <= draft_s:
                # A minimal bridge+candidate probe has unavoidable bridge
                # overhead. One real reuse with bounded stall earns a full trial.
                self.state = "enabled"
                self.observations.clear()
            else:
                self.state, self.remaining = "cooldown", self.cooldown_rounds
        self.observations.append(value)
        if self.state == "enabled" and len(self.observations) >= 4:
            saved = sum(r * c for g, r, c, s in self.observations)
            waste = sum((g - r) * c + s for g, r, c, s in self.observations)
            if saved <= waste:
                self.state, self.remaining = "cooldown", self.cooldown_rounds
