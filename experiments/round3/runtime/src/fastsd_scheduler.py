"""Pure FastSD admission policy.

The cloud executor owns model/KV operations; this module only decides which
logical work items may advance in one scheduling iteration.  Keeping the
planner side-effect free is deliberate: the same code is used by the real
admission path and by the KV prefetch predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import deque
import copy
import math
import time
from diagnostic_trace import trace as DIAG, request_meta as diag_req, item_meta as diag_item, queue_rows as diag_queues, backlog as diag_backlog
import random
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


FASTSD_DEFAULT_R1 = 128
FASTSD_DEFAULT_R2 = 512
FASTSD_DYNAMIC_WINDOW = 100
FASTSD_VERIFY_PREFILL_SWITCH_THRESHOLD = 6  # legacy API compatibility
FASTSD_VERIFY_WRR_ORDER = (
    "short", "short", "short", "short", "short", "short",
    "mid", "mid", "mid", "long",
)
FASTSD_QUEUE_ORDER = ("SV", "SP", "MV", "MP", "LV", "LP")
FASTSD_VISIT_QUEUE_ORDER = {
    "short": ("SV", "SP", "MV", "MP", "LV", "LP"),
    "mid": ("MV", "SV", "SP", "MP", "LV", "LP"),
    "long": ("LV", "SV", "SP", "MV", "MP", "LP"),
}
_QUEUE_COORDINATES = {
    category[0].upper() + kind[0].upper(): (kind, category)
    for category in ("short", "mid", "long")
    for kind in ("verify", "prefill")
}


def _finite_nonnegative(value: Any, field: str) -> float:
    """Return a finite duration in seconds or reject an invalid sample."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _timing_value(req: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    """Read one canonical seconds field, accepting migration aliases.

    The aliases keep old experiment manifests readable while all new wire
    requests use the explicit ``*_s`` names.  Millisecond aliases are only
    accepted when their name explicitly ends in ``_ms``.
    """

    for name in names:
        if name not in req or req[name] is None:
            continue
        value = _finite_nonnegative(req[name], name)
        return value / 1000.0 if name.endswith("_ms") else value
    return float(default)


def _has_explicit_latency_timing(req: Mapping[str, Any]) -> bool:
    # Presence of optional fields with ``None`` is common on the cloud HTTP
    # model and must not opt a legacy request into the new formula.  A caller
    # can still explicitly opt in before all measurements are available.
    if req.get("timing_priority", False) or req.get("latency_priority_enabled", False):
        return True
    return any(
        name in req and req[name] is not None
        for name in (
            "local_prefill_s", "local_prefill_ms", "prefill_first_token_s",
            "local_decode_per_token_s", "local_decode_per_token_ms",
            "push_s", "push_ms", "pull_s", "pull_ms", "last_pull_s",
            "T_i_p", "T_i_d", "T_i_push", "T_i_pull", "Tp", "Td",
        )
    )


def _wait_seconds(req: Mapping[str, Any], now: float) -> float:
    """Compute W from one clock domain, preferring recorded cloud arrival."""

    if "W_i" in req:
        return _finite_nonnegative(req["W_i"], "W_i")
    if "wait_s" in req:
        return _finite_nonnegative(req["wait_s"], "wait_s")
    if "arrival_monotonic_s" in req:
        start = _finite_nonnegative(req["arrival_monotonic_s"], "arrival_monotonic_s")
    elif "cloud_arrival_monotonic_s" in req:
        start = _finite_nonnegative(req["cloud_arrival_monotonic_s"], "cloud_arrival_monotonic_s")
    elif "server_enqueue_monotonic" in req:
        start = _finite_nonnegative(req["server_enqueue_monotonic"], "server_enqueue_monotonic")
    elif "queue_wait_start_s" in req:
        start = _finite_nonnegative(req["queue_wait_start_s"], "queue_wait_start_s")
    else:
        # Legacy requests use epoch ``current_time`` and tests pass a matching
        # synthetic ``now``.  The engine always writes the monotonic field.
        start = _finite_nonnegative(req.get("current_time", 0.0), "current_time")
    return max(0.0, float(now) - start)


def full_prefix_bridge_tokens(prefix_len: int, cached_len: int) -> int:
    """Return the uncached correction-token count for a full-prefix verify.

    The target deliberately rolls back before the correction/final token, so
    the next non-pipeline request may have a logical prefix exactly one token
    longer than its target KV cache.  That token is a real forward token and
    must be included in both admission cost and the internal tail-only slice.
    """
    missing = int(prefix_len) - int(cached_len)
    if missing not in (0, 1):
        raise ValueError(
            "full-prefix verify expects target cache to trail logical prefix "
            f"by at most one token, got prefix_len={prefix_len}, cached_len={cached_len}"
        )
    return missing


def verify_logit_position(prefix_len: int, draft_index: int) -> int:
    """Logical causal-logit position that predicts one draft token.

    A bridge/correction token may be physically appended in the same forward,
    but it already occupies logical position ``prefix_len - 1``.  Its logits
    therefore predict draft token zero; physical bridge presence must never
    shift this logical index.
    """
    if int(prefix_len) <= 0 or int(draft_index) < 0:
        raise ValueError("prefix_len must be positive and draft_index non-negative")
    return int(prefix_len) + int(draft_index) - 1


def build_fixed_wrr_order():
    return list(FASTSD_VERIFY_WRR_ORDER)


def compute_quantile(sorted_values, q: float) -> int:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    idx = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return int(sorted_values[idx])


def update_length_thresholds(
    recent_prefix_lens,
    default_r1: int = FASTSD_DEFAULT_R1,
    default_r2: int = FASTSD_DEFAULT_R2,
    min_samples: int = FASTSD_DYNAMIC_WINDOW,
    q1: float = 0.7,
    q2: float = 0.9,
):
    if len(recent_prefix_lens) < min_samples:
        return default_r1, default_r2
    sorted_lens = sorted(int(v) for v in recent_prefix_lens)
    r1 = compute_quantile(sorted_lens, q1)
    r2 = compute_quantile(sorted_lens, q2)
    if r2 < r1:
        r2 = r1
    return r1, r2


def length_category(prefix_len: int, r1: int, r2: int) -> str:
    if prefix_len <= r1:
        return "short"
    if prefix_len <= r2:
        return "mid"
    return "long"


class TaskLengthClassifier:
    """Paper Sec. 3.2: independent 100-arrival prefill/verify distributions."""

    def __init__(self):
        self.defaults = {"prefill": (128, 512), "verify": (20, 100)}
        self.history = {kind: deque(maxlen=FASTSD_DYNAMIC_WINDOW) for kind in self.defaults}
        self.thresholds = dict(self.defaults)

    def observe(self, item):
        kind = item.task_type
        self.history[kind].append(item.total_tokens)
        self.thresholds[kind] = update_length_thresholds(
            self.history[kind], *self.defaults[kind])
        return self.category(item)

    def category(self, item):
        return length_category(item.remaining_tokens, *self.thresholds[item.task_type])


def prefill_priority_score(req, now=None, lamda=0.01):
    if now is None:
        now = 0.0
    elapsed = _wait_seconds(req, float(now))
    wait_term = math.exp(float(lamda) * elapsed)
    # A timing-required Prefill remains queued, but cannot be selected before
    # the Edge timing update atomically marks it ready.  The planner also
    # filters this state; -inf makes direct score callers obey the same rule.
    if req.get("timing_required", False) and not req.get("timing_ready", False):
        return -math.inf
    if _has_explicit_latency_timing(req):
        m_i = max(
            1,
            int(req.get("m_i", req.get("prefill_gamma", req.get("gamma", 1))) or 1),
        )
        local_prefill = _timing_value(
            req, "local_prefill_s", "prefill_first_token_s", "local_prefill_ms",
            "T_i_p", "Tp",
        )
        decode_per_token = _timing_value(
            req, "local_decode_per_token_s", "local_decode_per_token_ms",
            "T_i_d", "Td",
        )
        push = _timing_value(req, "push_s", "push_ms", "T_i_push", default=0.0)
        # Prefill has no acceptance probability yet; per the protocol its
        # P_n is fixed at one and cannot be overridden by request data.
        return -(local_prefill + decode_per_token * (m_i - 1) + push) + wait_term
    # Legacy vanilla/pipeline requests had no local timing update.
    return wait_term


def verify_priority_score(req, accept_stats, now=None, lamda=0.01):
    if now is None:
        now = 0.0
    elapsed = _wait_seconds(req, float(now))
    wait_term = math.exp(float(lamda) * elapsed)

    pid = req["proc_id"]
    accepted_sum, total_sum = accept_stats.get(pid, (0, 0))
    if total_sum <= 0:
        acc_prob = 1.0
    else:
        acc_prob = (accepted_sum + 1.0) / (total_sum + 1.0)
    acc_prob = max(acc_prob, 1e-6)
    draft_len = max(1, int(req.get("m_i", req.get("gamma", 1)) or 1))
    if _has_explicit_latency_timing(req):
        decode_per_token = _timing_value(
            req, "local_decode_per_token_s", "local_decode_per_token_ms", "T_i_d", "Td",
        )
        push = _timing_value(req, "push_s", "push_ms", "T_i_push", default=0.0)
        pull = _timing_value(
            req, "pull_s", "last_pull_s", "pull_ms", "T_i_pull", default=0.0,
        )
        denominator = _timing_value(req, "P_n", "p_n", default=acc_prob)
        if denominator <= 0.0:
            raise ValueError("P_n must be positive")
        return -((decode_per_token * draft_len + push + pull) / denominator) + wait_term

    # Legacy request compatibility: do not reinterpret historical RTT fields
    # for baseline runs that have not opted into latency-aware scoring.
    draft_total_time = _timing_value(req, "lag", default=0.0)
    if draft_total_time <= 0.0 and "draft_time_per_token" in req:
        draft_total_time = draft_len * _finite_nonnegative(req["draft_time_per_token"], "draft_time_per_token")
    transport_rtt = _timing_value(req, "transport_rtt", "edge_rtt", default=0.0)
    return -((draft_total_time + transport_rtt) / acc_prob) + wait_term


def compute_priority_score(req, accept_stats, now=None, lamda=0.01):
    """Compatibility dispatcher; scores are compared within one queue."""
    if req["task_type"] == "prefill":
        return prefill_priority_score(req, now=now, lamda=lamda)
    return verify_priority_score(req, accept_stats, now=now, lamda=lamda)


def should_switch_to_prefill(verify_underutilized_flags, has_prefill_tasks: bool) -> bool:
    """Legacy helper retained for callers outside the new FastSD planner."""
    return bool(has_prefill_tasks) and sum(int(flag) for flag in verify_underutilized_flags) >= FASTSD_VERIFY_PREFILL_SWITCH_THRESHOLD


@dataclass
class WorkItem:
    """Persistent logical request; it is never split into independent HTTP requests."""

    work_id: str
    proc_id: Any
    task_type: str
    category: str
    request: Mapping[str, Any] = field(default_factory=dict, repr=False)
    total_tokens: int = 0
    cursor: int = 0
    enqueue_cycle: int = 0
    last_served_cycle: int = -1
    missed_cycles: int = 0
    base_prefix_len: int = 0
    original_gamma: int = 0
    accepted_so_far: int = 0
    bridge_pending: int = 0
    response_key: str | None = None
    finished: bool = False
    state: str = "ready"  # ready -> reserved -> running -> ready/finished

    @property
    def timing_ready(self) -> bool:
        """Whether a timing-required Prefill may enter a forward plan."""

        return bool(
            self.task_type != "prefill"
            or not self.request.get("timing_required", False)
            or self.request.get("timing_ready", False)
        )

    @property
    def remaining_tokens(self) -> int:
        return max(0, int(self.total_tokens) - int(self.cursor))

    @property
    def is_overdue_prefill(self) -> bool:
        return self.task_type == "prefill" and self.missed_cycles >= 2

    @classmethod
    def from_request(cls, request: Mapping[str, Any], category: str, cycle: int = 0, work_id: str | None = None):
        task_type = str(request["task_type"])
        if task_type == "prefill":
            prompt_len = request.get("prompt_len")
            if prompt_len is None:
                prompt_len = request["draft_output"].shape[1]
            total = int(prompt_len)
            gamma = int(request.get("prefill_gamma", request.get("gamma", 0)) or 0)
        else:
            total = int(request.get("gamma", 0) or 0)
            gamma = total
        if total <= 0:
            raise ValueError("WorkItem total_tokens must be positive")
        pid = request["proc_id"]
        return cls(
            work_id=work_id or str(request.get("work_id", f"{pid}:{id(request)}")),
            proc_id=pid,
            task_type=task_type,
            category=category,
            request=request,
            total_tokens=total,
            enqueue_cycle=cycle,
            base_prefix_len=int(request.get("prefix_len", 0) or 0),
            original_gamma=gamma,
            bridge_pending=(
                1 if request.get("has_bridge_token", False)
                else int(request.get("bridge_tokens", 0) or 0)
            ),
            response_key=request.get("response_key"),
        )


@dataclass(frozen=True)
class ExecutionSlice:
    work_id: str
    proc_id: Any
    task_type: str
    category: str
    offset: int
    draft_token_count: int
    forward_token_count: int
    includes_bridge: bool = False

    @property
    def num_tokens(self) -> int:
        """Compatibility alias: admission cost is real forward tokens."""
        return self.forward_token_count


@dataclass
class SchedulerState:
    wrr_cursor: int = 0
    current_cycle: int = 0


@dataclass
class AdmissionPlan:
    verify_slices: list[ExecutionSlice]
    prefill_slices: list[ExecutionSlice]
    used_tokens: int
    token_budget: int
    max_num_seqs: int
    next_wrr_cursor: int
    next_cycle: int
    slots_consumed: int
    selected_work_ids: list[str] = field(default_factory=list)
    _selected_items: list[WorkItem] = field(default_factory=list, repr=False)
    _all_items: list[WorkItem] = field(default_factory=list, repr=False)
    start_cycle: int = 0
    plan_ms: float = 0.0
    seal_reason: str = ""
    prediction_assumptions: dict[str, str] = field(default_factory=dict)
    _committed: bool = field(default=False, repr=False)

    @property
    def ordered_slices(self):
        """Slices in successful WRR visit order, before execution phase grouping."""
        by_id = {sl.work_id: sl for sl in self.verify_slices + self.prefill_slices}
        return [by_id[work_id] for work_id in self.selected_work_ids]

    @property
    def verify_proc_ids(self):
        return [s.proc_id for s in self.verify_slices]

    @property
    def prefill_proc_ids(self):
        return [s.proc_id for s in self.prefill_slices]

    @property
    def remaining_tokens(self) -> int:
        return self.token_budget - self.used_tokens



def cost_features(item, sl=None):
    """(cached context, actual forward tokens), shared by ranking/execution."""
    if isinstance(item, Mapping):
        if "_cost_features" in item:
            return tuple(item["_cost_features"])
        context = int(item.get("_cached_context", 0))
        if item["task_type"] == "prefill":
            return context, int(item["draft_output"].shape[1]) - context
        bridge = max(0, int(item["prefix_len"]) - context)
        return context, int(item["gamma"]) + bridge
    if sl is None:
        raise ValueError("WorkItem cost features require its actual slice")
    context = sl.offset if item.task_type == "prefill" else item.base_prefix_len + sl.offset - int(sl.includes_bridge)
    return max(0, context), sl.forward_token_count


def transition_work_item(item, sl, *, start_cycle, next_cycle,
                         progress=None, completed=False, category_for=None):
    """Pure, idempotent state transition used by projection and commit."""
    result = copy.copy(item)
    result.request = dict(item.request)
    if getattr(item, "cancel_pending", False) or item.state in {"cancelled", "canceled", "released"}:
        result.finished = True
        result.state = "finished"
        return result
    if sl is not None:
        advance = sl.draft_token_count if progress is None else int(progress)
        target = sl.offset + max(0, min(sl.draft_token_count, advance))
        result.cursor = max(item.cursor, target)
        result.last_served_cycle, result.missed_cycles = next_cycle, 0
        if item.task_type == "verify":
            result.accepted_so_far = max(item.accepted_so_far, target)
            result.bridge_pending = 0
        result.finished = (item.finished or completed or result.remaining_tokens <= 0
                           or (item.task_type == "verify" and advance < sl.draft_token_count))
        result.state = "finished" if result.finished else "ready"
    elif item.task_type == "prefill" and not item.finished:
        result.missed_cycles += max(0, next_cycle - max(start_cycle, item.enqueue_cycle))
    if category_for is not None:
        result.category = category_for(result)
    return result


def _normalise_queues(queues: Mapping[str, Any]) -> dict[str, dict[str, list[WorkItem]]]:
    """Accept either {task_type: {category: items}} or {category: {task_type: items}}."""
    out = {"verify": {c: [] for c in ("short", "mid", "long")}, "prefill": {c: [] for c in ("short", "mid", "long")}}
    if set(queues).issubset({"verify", "prefill"}):
        for task_type, by_cat in queues.items():
            for cat, items in by_cat.items():
                out[task_type][cat] = list(items)
    else:
        for cat, by_type in queues.items():
            for task_type, items in by_type.items():
                out[task_type][cat] = list(items)
    return out


def _as_work_item(item: WorkItem | Mapping[str, Any], task_type: str, category: str, cycle: int) -> WorkItem:
    if isinstance(item, WorkItem):
        return item
    return WorkItem.from_request(item, category=category, cycle=cycle)


def _priority(item: WorkItem, accept_stats: Mapping[Any, Sequence[float]], now: float) -> float:
    return compute_priority_score(item.request, accept_stats, now=now)


def _eligible(item, selected, reserved):
    return (
        item.work_id not in selected and item.work_id not in reserved
        and not item.finished and not getattr(item, "cancel_pending", False)
        and item.state not in {"reserved", "running", "cancelled", "canceled", "released", "finished"}
        and item.remaining_tokens > 0 and item.timing_ready
    )


def take_for_visit(state, category, selected, accept_stats, now, reserved):
    """Choose a queue first, then its highest original score, without fit filtering.

    A chosen candidate that does not fit seals the batch. Testing a later queue
    for a smaller candidate would silently spend the current queue's visit.
    """
    overdue = [item for item in state["prefill"][category]
               if _eligible(item, selected, reserved) and item.is_overdue_prefill]
    if overdue:
        return max(overdue, key=lambda item: _priority(item, accept_stats, now))
    for queue_name in FASTSD_VISIT_QUEUE_ORDER[category]:
        kind, bucket = _QUEUE_COORDINATES[queue_name]
        candidates = [item for item in state[kind][bucket]
                      if _eligible(item, selected, reserved)]
        if candidates:
            return max(candidates, key=lambda item: _priority(item, accept_stats, now))
    return None


def _all_candidates(state, selected, reserved):
    for task_type in ("verify", "prefill"):
        for category in ("short", "mid", "long"):
            for item in state[task_type][category]:
                if _eligible(item, selected, reserved):
                    yield item


def _make_slice(item: WorkItem, budget: int, quantum: int, min_prefill: int) -> ExecutionSlice | None:
    if budget <= 0 or item.remaining_tokens <= 0:
        return None
    # bridge_pending is explicit WorkItem state so admission charges the
    # external pipeline bridge exactly once.
    bridge = item.bridge_pending if item.task_type == "verify" else 0
    if item.task_type == "verify":
        draft = min(item.remaining_tokens, budget - bridge)
        if draft <= 0:
            return None
    else:
        if budget < min(min_prefill, item.remaining_tokens):
            return None
        draft = min(item.remaining_tokens, budget, quantum)
        if draft <= 0:
            return None
    return ExecutionSlice(
        work_id=item.work_id,
        proc_id=item.proc_id,
        task_type=item.task_type,
        category=item.category,
        offset=item.cursor,
        draft_token_count=draft,
        forward_token_count=draft + bridge,
        includes_bridge=bool(bridge),
    )


def plan_iteration(
    queues: Mapping[str, Any],
    token_budget: int,
    current_cycle: int = 0,
    wrr_cursor: int = 0,
    consume: bool = False,
    max_num_seqs: int = 4,
    min_prefill_chunk_tokens: int = 16,
    prefill_chunk_quantum: int = 128,
    accept_stats: Mapping[Any, Sequence[float]] | None = None,
    now: float = 0.0,
    reserved_work_ids: Iterable[str] = (),
    excluded_proc_ids: Iterable[Any] = (),
    selection_policy: str = "legacy",
    candidate_value=None,
    max_wait_s: float = 1.0,
) -> AdmissionPlan:
    """Build an admission plan without mutating queues or WorkItems.

    ``consume`` is accepted for compatibility with early callers but is
    intentionally ignored.  Call :func:`commit_admission_plan` only after the
    corresponding forward succeeds.
    """
    plan_started = time.perf_counter()
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if min_prefill_chunk_tokens <= 0 or prefill_chunk_quantum <= 0:
        raise ValueError("prefill chunk sizes must be positive")
    if min_prefill_chunk_tokens > token_budget:
        raise ValueError("min_prefill_chunk_tokens must not exceed token_budget")
    if prefill_chunk_quantum < min_prefill_chunk_tokens:
        raise ValueError("prefill_chunk_quantum must not be smaller than min_prefill_chunk_tokens")
    if selection_policy not in {"legacy", "fcfs", "progress"}:
        raise ValueError("unknown scheduler selection policy")
    state_raw = _normalise_queues(queues)
    state = {t: {c: [_as_work_item(x, t, c, current_cycle) for x in state_raw[t][c]] for c in state_raw[t]} for t in state_raw}
    excluded = set(excluded_proc_ids)
    state = {t: {c: [item for item in items if item.proc_id not in excluded]
                 for c, items in categories.items()} for t, categories in state.items()}
    accept_stats = accept_stats or {}
    selected: set[str] = set()
    reserved = set(reserved_work_ids)
    verify: list[ExecutionSlice] = []
    prefill: list[ExecutionSlice] = []
    used = 0
    slots = 0
    cursor = int(wrr_cursor) % len(FASTSD_VERIFY_WRR_ORDER)
    selected_items = []
    visit_ids = []
    seal_reason = ""

    while used < token_budget and len(selected) < max_num_seqs:
        category = FASTSD_VERIFY_WRR_ORDER[cursor]
        item = take_for_visit(state, category, selected, accept_stats, now, reserved)
        diag_wrr_item = item
        diag_values = []
        if selection_policy != "legacy":
            candidates = [x for x in _all_candidates(state, selected, reserved)
                          if _make_slice(x, token_budget - used, prefill_chunk_quantum, min_prefill_chunk_tokens) is not None]
            if selection_policy == "fcfs":
                item = min(candidates, key=lambda x: x.request.get("server_enqueue_monotonic", 0), default=None)
            else:
                def progress_key(candidate):
                    wait = _wait_seconds(candidate.request, now)
                    urgent = candidate.is_overdue_prefill or wait >= max_wait_s
                    candidate_slice = _make_slice(candidate, token_budget - used, prefill_chunk_quantum, min_prefill_chunk_tokens)
                    value = (candidate_value(candidate, candidate_slice, verify + prefill) if candidate_value is not None
                             else _priority(candidate, accept_stats, now))
                    if DIAG.enabled and getattr(DIAG.context, "plan_kind", "") == "admission":
                        diag_values.append([candidate.work_id, urgent, wait if urgent else 0.0, value])
                    return urgent, wait if urgent else 0.0, value
                item = max(candidates, key=progress_key, default=None)
        if DIAG.enabled and getattr(DIAG.context, "plan_kind", "") == "admission":
            DIAG.emit("scheduler_choice", plan_id=getattr(DIAG.context,"plan_id",None), policy=selection_policy, slot=slots, wrr_category=category, wrr_proposed=diag_wrr_item.work_id if diag_wrr_item else None, chosen=item.work_id if item else None, chosen_category=item.category if item else None, candidate_values=diag_values, used_tokens=used, token_budget=token_budget)
        if item is None:
            seal_reason = "no_runnable_task"
            break
        sl = _make_slice(item, token_budget - used, prefill_chunk_quantum, min_prefill_chunk_tokens)
        if sl is None:
            if not selected:
                raise ValueError(
                    f"work item {item.work_id} cannot fit a legal slice in full token_budget={token_budget}"
                )
            seal_reason = "insufficient_remaining_budget"
            break
        selected.add(item.work_id)
        selected_items.append(item)
        visit_ids.append(item.work_id)
        # The KV belongs to a device; each plan can reserve that device once.
        reserved.update(candidate.work_id for candidate in
                        _all_candidates(state, selected, reserved)
                        if candidate.proc_id == item.proc_id)
        if sl.task_type == "verify":
            verify.append(sl)
        else:
            prefill.append(sl)
        used += sl.forward_token_count
        slots += 1
        cursor = (cursor + 1) % len(FASTSD_VERIFY_WRR_ORDER)
    else:
        seal_reason = "token_budget" if used >= token_budget else "max_num_seqs"

    # If a single selected Prefill is the only remaining work, fill the tail of
    # the budget while retaining the one-slice-per-WorkItem invariant.
    live = list(_all_candidates(state, set(), reserved))
    if used < token_budget and len(live) == 1 and len(prefill) == 1 and not verify:
        item = live[0]
        sl = prefill[0]
        extra = min(token_budget - used, item.remaining_tokens - sl.draft_token_count)
        if extra > 0:
            prefill[0] = replace(sl, draft_token_count=sl.draft_token_count + extra,
                                 forward_token_count=sl.forward_token_count + extra)
            used += extra

    total_steps = int(wrr_cursor) % len(FASTSD_VERIFY_WRR_ORDER) + slots
    next_cycle = int(current_cycle) + total_steps // len(FASTSD_VERIFY_WRR_ORDER)
    return AdmissionPlan(
        verify_slices=verify,
        prefill_slices=prefill,
        used_tokens=used,
        token_budget=token_budget,
        max_num_seqs=max_num_seqs,
        next_wrr_cursor=total_steps % len(FASTSD_VERIFY_WRR_ORDER),
        next_cycle=next_cycle,
        slots_consumed=slots,
        selected_work_ids=visit_ids,
        _selected_items=selected_items,
        _all_items=[item for cats in state.values() for items in cats.values() for item in items],
        start_cycle=int(current_cycle),
        plan_ms=(time.perf_counter() - plan_started) * 1000,
        seal_reason=seal_reason,
    )


def project_after_plan(queues, plan, *, category_for=None, progress=None,
                       completed_work_ids=(), applied_work_ids=()):
    """Project metadata only; known outcomes replace the all-accept assumption.

    ``applied_work_ids`` are phases already reflected in the queue snapshot.
    They still receive the plan's service bookkeeping, without moving their
    cursor again. New arrivals are reclassified but never aged retroactively.
    """
    state = _normalise_queues(queues)
    slices = {s.work_id: s for s in plan.verify_slices + plan.prefill_slices}
    completed, applied = set(completed_work_ids), set(applied_work_ids)
    participants = {item.work_id for item in plan._all_items}
    result = {t: {c: [] for c in cats} for t, cats in state.items()}
    for task_type, categories in state.items():
        for items in categories.values():
            for item in items:
                sl = slices.get(item.work_id)
                if item.work_id in applied and sl is not None:
                    projected = copy.copy(item)
                    projected.request = dict(item.request)
                    projected.last_served_cycle, projected.missed_cycles = plan.next_cycle, 0
                    projected.finished = (item.finished or item.work_id in completed
                                          or item.remaining_tokens <= 0
                                          or getattr(item, "cancel_pending", False))
                    projected.state = "finished" if projected.finished else "ready"
                    if category_for is not None:
                        projected.category = category_for(projected)
                else:
                    actual = progress.get(item.work_id, sl.draft_token_count) if progress is not None and sl is not None else None
                    projected = transition_work_item(item, sl,
                        start_cycle=plan.start_cycle,
                        next_cycle=plan.next_cycle if item.work_id in participants or sl is not None else plan.start_cycle,
                        progress=actual, completed=item.work_id in completed,
                        category_for=category_for)
                if not projected.finished and projected.remaining_tokens > 0:
                    result[task_type][projected.category].append(projected)
    return result


def predict_next_plan(queues, inflight_plan=None, *, known_phase_results=None,
                      applied_work_ids=(), completed_work_ids=(), category_for=None,
                      planning_time=0.0, **planner_kwargs):
    """Plan the complete next batch before filtering safe KV transfers.

    Results map work IDs to an actual per-slice progress count, or a mapping
    containing ``progress``, ``completed`` and ``applied``. The caller freezes
    classifier thresholds and acceptance statistics together with its queue
    snapshot. No real WorkItem, WRR state or classifier history is modified.
    """
    progress = {}
    completed, applied = set(completed_work_ids), set(applied_work_ids)
    for work_id, outcome in (known_phase_results or {}).items():
        if isinstance(outcome, Mapping):
            if outcome.get("progress") is not None:
                progress[work_id] = int(outcome["progress"])
            if outcome.get("completed", False):
                completed.add(work_id)
            if outcome.get("applied", False):
                applied.add(work_id)
        else:
            progress[work_id] = int(outcome)
    assumptions = {}
    if inflight_plan is not None:
        queues = project_after_plan(queues, inflight_plan, category_for=category_for,
            progress=progress, completed_work_ids=completed, applied_work_ids=applied)
        planner_kwargs["current_cycle"] = inflight_plan.next_cycle
        planner_kwargs["wrr_cursor"] = inflight_plan.next_wrr_cursor
        assumptions = {sl.work_id: "all_accept" for sl in inflight_plan.verify_slices
                       if sl.work_id not in progress and sl.work_id not in completed and sl.work_id not in applied}
    else:
        snapshot = _normalise_queues(queues)
        queues = {kind: {category: [] for category in categories}
                  for kind, categories in snapshot.items()}
        for kind, categories in snapshot.items():
            for items in categories.values():
                for item in items:
                    cloned = copy.copy(item)
                    cloned.request = dict(item.request)
                    if category_for is not None:
                        cloned.category = category_for(cloned)
                    queues[kind][cloned.category].append(cloned)
    planner_kwargs["now"] = planning_time
    prediction = plan_iteration(queues, **planner_kwargs)
    prediction.prediction_assumptions = assumptions
    return prediction


def reserve_admission_plan(plan: AdmissionPlan) -> None:
    for item in plan._selected_items:
        if item.state not in {"ready", "reserved"}:
            raise RuntimeError(f"work item {item.work_id} is already {item.state}")
        item.state = "reserved"


def commit_admission_plan(plan, completed_work_ids=(), progress=None, *,
                          all_items=None, category_for=None):
    if plan._committed:
        return
    completed = set(completed_work_ids)
    slices = {s.work_id: s for s in plan.verify_slices + plan.prefill_slices}
    participants = {item.work_id for item in plan._all_items or plan._selected_items}
    items = list(all_items) if all_items is not None else plan._all_items or plan._selected_items
    for item in items:
        if item.work_id not in participants:
            continue
        sl = slices.get(item.work_id)
        actual = progress.get(item.work_id, sl.draft_token_count) if progress is not None and sl is not None else None
        updated = transition_work_item(item, sl, start_cycle=plan.start_cycle,
            next_cycle=plan.next_cycle, progress=actual,
            completed=item.work_id in completed, category_for=category_for)
        for name in ("cursor", "last_served_cycle", "missed_cycles", "state", "finished",
                     "accepted_so_far", "bridge_pending", "category"):
            setattr(item, name, getattr(updated, name))
    plan._committed = True


def abort_admission_plan(plan: AdmissionPlan) -> None:
    for item in plan._selected_items:
        if item.state == "reserved":
            item.state = "ready"


def predict_next_verify_proc_ids(
    verify_queues,
    order,
    batch_size: int,
    start_idx: int = 0,
    pinned_gpu_pids=None,
    max_slots: int | None = None,
):
    if batch_size <= 0 or not order:
        return []
    pinned = set(pinned_gpu_pids or ())
    total_slots = len(order)
    slots_to_scan = total_slots if max_slots is None else max(0, min(int(max_slots), total_slots))
    predicted = []
    seen = set()
    for offset in range(slots_to_scan):
        cat = order[(start_idx + offset) % total_slots]
        for item in verify_queues.get(cat, ()):
            pid = item["proc_id"] if isinstance(item, Mapping) else item.proc_id
            if pid in pinned or pid in seen:
                continue
            predicted.append(pid)
            seen.add(pid)
            if len(predicted) >= batch_size:
                return predicted
    return predicted


def accept_count(
    draft_tokens,
    target_top1,
    target_probs=None,
    draft_probs=None,
    *,
    mode="greedy",
    threshold=0.0,
    uniforms=None,
):
    """Compatibility helper for testing the legacy and scalar-ratio rules.

    Runtime verification uses :func:`src.speculative_sampling.rejection_sample`
    so it can sample the complete residual distribution.  This small list-based
    helper remains useful for protocol tests that only need the prefix length.
    ``threshold`` is intentionally a test-only gate and is not used by the
    production verifier.
    """

    draft = list(draft_tokens)
    target = list(target_top1)
    if len(draft) != len(target):
        raise ValueError("draft and target token vectors must have equal length")
    mode = str(mode).lower()
    if mode == "greedy":
        for index, token in enumerate(draft):
            if int(token) != int(target[index]):
                return index
        return len(draft)
    if mode != "prob":
        raise ValueError("mode must be 'greedy' or 'prob'")
    if target_probs is None or draft_probs is None:
        raise ValueError("prob mode requires target_probs and draft_probs")
    if len(target_probs) != len(draft) or len(draft_probs) != len(draft):
        raise ValueError("probability vectors must match draft length")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite and non-negative") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    random_values = iter(uniforms) if uniforms is not None else None
    for index, (p_value, q_value) in enumerate(zip(target_probs, draft_probs)):
        try:
            p_value = float(p_value)
            q_value = float(q_value)
        except (TypeError, ValueError, OverflowError):
            return index
        if not math.isfinite(p_value) or not math.isfinite(q_value) or p_value <= 0.0 or q_value <= 0.0:
            return index
        ratio = min(1.0, p_value / q_value)
        if ratio < threshold:
            return index
        if random_values is None:
            draw = random.random()
        else:
            try:
                draw = float(next(random_values))
            except StopIteration as exc:
                raise ValueError("uniforms must contain one value per draft token") from exc
        if not math.isfinite(draw) or draw < 0.0 or draw > 1.0:
            raise ValueError("uniform values must lie in [0, 1]")
        if draw > ratio:
            return index
    return len(draft)


def plan_draft_prob_sources(
    prefix_len: int,
    gamma: int,
    history_rows: int,
    reused_prob_count: int = 0,
):
    """Report how many draft probability rows come from history and reuse."""

    prefix_len = int(prefix_len)
    gamma = int(gamma)
    history_rows = int(history_rows)
    reused_prob_count = max(0, int(reused_prob_count))
    if gamma <= 0:
        return 0, 0
    history_available = max(0, history_rows - prefix_len + 1)
    history_count = min(gamma, history_available)
    reused_count = min(gamma - history_count, reused_prob_count)
    return history_count, reused_count
