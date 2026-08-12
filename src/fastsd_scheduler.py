import math
import queue
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


FASTSD_DEFAULT_R1 = 128
FASTSD_DEFAULT_R2 = 512
FASTSD_DYNAMIC_WINDOW = 100
FASTSD_VERIFY_WRR_ORDER = (
    "short",
    "short",
    "short",
    "short",
    "short",
    "short",
    "mid",
    "mid",
    "mid",
    "long",
)
FASTSD_CATEGORIES = ("short", "mid", "long")


class OversizeVerifyError(ValueError):
    """Raised when an atomic Verify request cannot fit an empty tick budget."""

    def __init__(self, request: dict, cost: int, token_budget: int) -> None:
        self.request = request
        self.cost = int(cost)
        self.token_budget = int(token_budget)
        request_id = request.get("proc_id", request.get("task_id", "unknown"))
        super().__init__(
            f"Verify request {request_id!r} costs {self.cost} effective tokens, "
            f"which exceeds token_budget={self.token_budget}; Verify is atomic"
        )


@dataclass(frozen=True)
class ScheduleEntry:
    """One admitted request (Verify) or request slice (Prefill)."""

    request: dict = field(repr=False, compare=False)
    request_id: Any
    task_type: str
    category: str
    effective_tokens: int
    chunk_start: int | None = None
    chunk_end: int | None = None


@dataclass
class SchedulePlan:
    """A tick-wide admission plan; model forwards remain task-type homogeneous."""

    verify_entries: list[ScheduleEntry] = field(default_factory=list)
    prefill_entries: list[ScheduleEntry] = field(default_factory=list)
    used_tokens: int = 0
    token_budget: int = 0
    start_cursor: int = 0
    end_cursor: int = 0
    completed_cycles: int = 0

    @property
    def entries(self) -> list[ScheduleEntry]:
        return [*self.verify_entries, *self.prefill_entries]


@dataclass
class UnifiedSchedulerState:
    """Persistent state shared by consecutive FastSD scheduling ticks."""

    order: tuple[str, ...] = FASTSD_VERIFY_WRR_ORDER
    cursor: int = 0
    prefill_max_wait_cycles: int = 2
    prefill_wait_cycles: dict[str, int] = field(
        default_factory=lambda: {category: 0 for category in FASTSD_CATEGORIES}
    )
    prefill_admitted_in_cycle: set[str] = field(default_factory=set)
    completed_cycles: int = 0

    def __post_init__(self) -> None:
        self.order = tuple(self.order)
        if not self.order:
            raise ValueError("scheduler order must not be empty")
        if int(self.prefill_max_wait_cycles) <= 0:
            raise ValueError("prefill_max_wait_cycles must be positive")
        self.prefill_max_wait_cycles = int(self.prefill_max_wait_cycles)
        self.cursor = int(self.cursor) % len(self.order)
        for category in set(self.order) | set(FASTSD_CATEGORIES):
            self.prefill_wait_cycles.setdefault(category, 0)


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


def compute_priority_score(req, accept_stats, now=None, lamda=0.01):
    if now is None:
        now = 0.0

    enqueue_time = req.get("server_enqueue_monotonic", req.get("current_time", now))
    elapsed = max(0.0, float(now) - float(enqueue_time))
    wait_term = math.exp(lamda * elapsed)
    if req["task_type"] == "prefill":
        return wait_term

    pid = req["proc_id"]
    accepted_sum, total_sum = accept_stats.get(pid, (0, 0))

    if total_sum <= 0:
        acc_prob = 1.0
    else:
        # Laplace smoothing prevents cold-start tasks from being catastrophically deprioritized.
        acc_prob = (accepted_sum + 1.0) / (total_sum + 1.0)
    acc_prob = max(acc_prob, 1e-6)

    draft_len = max(1, int(req.get("gamma", 1) or 1))
    draft_total_time = float(req.get("lag", 0.0))
    if draft_total_time <= 0.0 and "draft_time_per_token" in req:
        draft_total_time = draft_len * max(0.0, float(req["draft_time_per_token"]))

    transport_rtt = max(0.0, float(req.get("transport_rtt", req.get("edge_rtt", 0.0))))
    return -((draft_total_time + transport_rtt) / acc_prob) + wait_term


def _payload_token_length(draft_output) -> int:
    shape = getattr(draft_output, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(draft_output, (list, tuple)):
        if draft_output and isinstance(draft_output[0], (list, tuple)):
            return len(draft_output[0])
        return len(draft_output)
    raise TypeError("draft_output must expose shape or be a token sequence")


def request_budget_tokens(
    req: Mapping[str, Any],
    *,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
) -> int:
    """Return effective tokens consumed by this admission decision."""

    payload_len = _payload_token_length(req["draft_output"])
    if payload_len <= 0:
        raise ValueError("draft_output must contain at least one token")

    task_type = req["task_type"]
    if task_type == "prefill":
        if chunk_start is None:
            chunk_start = int(req.get("prefill_cursor", 0))
        if chunk_end is None:
            chunk_end = int(req.get("prefill_chunk_end", payload_len))
        if not 0 <= int(chunk_start) < int(chunk_end) <= payload_len:
            raise ValueError(
                f"invalid Prefill chunk [{chunk_start}:{chunk_end}) for payload length {payload_len}"
            )
        return int(chunk_end) - int(chunk_start)

    if task_type != "verify":
        raise ValueError(f"unsupported task_type: {task_type}")
    if req.get("tail_only", False):
        return payload_len
    return max(1, payload_len - int(req["prefix_len"]))


def _queue_peek(items):
    return items[0] if items else None


def _queue_pop_left(items):
    if hasattr(items, "popleft"):
        return items.popleft()
    return items.pop(0)


def _request_identity(req: Mapping[str, Any]):
    return req.get("proc_id", req.get("task_id", id(req)))


def _prefill_total_tokens(req: Mapping[str, Any]) -> int:
    payload_len = _payload_token_length(req["draft_output"])
    total = int(req.get("prefix_len", payload_len))
    if total <= 0 or total > payload_len:
        raise ValueError(
            f"Prefill prefix_len={total} must be within payload length {payload_len}"
        )
    cursor = int(req.get("prefill_cursor", 0))
    if cursor < 0 or cursor >= total:
        raise ValueError(f"Prefill cursor {cursor} must be within [0, {total})")
    return total


def _iter_queue_items(queues: Mapping[str, Sequence[dict]]) -> Iterable[dict]:
    for items in queues.values():
        yield from items


def _validate_pending_requests(verify_queues, prefill_queues, token_budget: int) -> None:
    for req in _iter_queue_items(verify_queues):
        cost = request_budget_tokens(req)
        if cost > token_budget:
            raise OversizeVerifyError(req, cost, token_budget)
    for req in _iter_queue_items(prefill_queues):
        _prefill_total_tokens(req)


def _complete_cycle(state: UnifiedSchedulerState, prefill_queues) -> None:
    for category in state.prefill_wait_cycles:
        if category in state.prefill_admitted_in_cycle:
            state.prefill_wait_cycles[category] = 0
        elif prefill_queues.get(category):
            state.prefill_wait_cycles[category] += 1
        else:
            state.prefill_wait_cycles[category] = 0
    state.prefill_admitted_in_cycle.clear()
    state.completed_cycles += 1


def build_unified_schedule_plan(
    verify_queues: MutableMapping[str, Any],
    prefill_queues: MutableMapping[str, Any],
    state: UnifiedSchedulerState,
    token_budget: int,
) -> SchedulePlan:
    """Build one global-budget plan while mutating the supplied pending queues."""

    budget = int(token_budget)
    if budget <= 0:
        raise ValueError("token_budget must be positive")
    _validate_pending_requests(verify_queues, prefill_queues, budget)

    plan = SchedulePlan(
        token_budget=budget,
        start_cursor=state.cursor,
        end_cursor=state.cursor,
    )
    start_completed_cycles = state.completed_cycles
    remaining = budget
    no_progress_slots = 0
    selected_prefills = set()

    while remaining > 0 and no_progress_slots < len(state.order):
        category = state.order[state.cursor]
        verify_items = verify_queues.get(category, ())
        prefill_items = prefill_queues.get(category, ())
        if not prefill_items:
            state.prefill_wait_cycles[category] = 0

        entry = None
        prefill_due = (
            bool(prefill_items)
            and state.prefill_wait_cycles[category] >= state.prefill_max_wait_cycles
        )

        if not prefill_due:
            verify_req = _queue_peek(verify_items)
            if verify_req is not None:
                verify_cost = request_budget_tokens(verify_req)
                if verify_cost <= remaining:
                    verify_req = _queue_pop_left(verify_items)
                    entry = ScheduleEntry(
                        request=verify_req,
                        request_id=_request_identity(verify_req),
                        task_type="verify",
                        category=category,
                        effective_tokens=verify_cost,
                    )

        if entry is None:
            prefill_req = _queue_peek(prefill_items)
            request_identity = _request_identity(prefill_req) if prefill_req is not None else None
            if prefill_req is not None and request_identity not in selected_prefills:
                total_tokens = _prefill_total_tokens(prefill_req)
                chunk_start = int(prefill_req.get("prefill_cursor", 0))
                chunk_end = min(total_tokens, chunk_start + remaining)
                prefill_req = _queue_pop_left(prefill_items)
                chunk_tokens = request_budget_tokens(
                    prefill_req,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                entry = ScheduleEntry(
                    request=prefill_req,
                    request_id=request_identity,
                    task_type="prefill",
                    category=category,
                    effective_tokens=chunk_tokens,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                selected_prefills.add(request_identity)
                state.prefill_admitted_in_cycle.add(category)
                if prefill_due:
                    state.prefill_wait_cycles[category] = 0

        if entry is not None:
            if entry.task_type == "verify":
                plan.verify_entries.append(entry)
            else:
                plan.prefill_entries.append(entry)
            plan.used_tokens += entry.effective_tokens
            remaining -= entry.effective_tokens
            no_progress_slots = 0
        else:
            no_progress_slots += 1

        state.cursor = (state.cursor + 1) % len(state.order)
        if state.cursor == 0:
            _complete_cycle(state, prefill_queues)

    plan.end_cursor = state.cursor
    plan.completed_cycles = state.completed_cycles - start_completed_cycles
    if plan.used_tokens > budget:
        raise AssertionError("unified scheduler exceeded its global token budget")
    return plan


def split_plan_microbatches(
    entries: Sequence[ScheduleEntry], max_num_seqs: int
) -> list[list[ScheduleEntry]]:
    limit = int(max_num_seqs)
    if limit <= 0:
        raise ValueError("max_num_seqs must be positive")
    return [list(entries[index:index + limit]) for index in range(0, len(entries), limit)]


def ordered_plan_microbatches(
    plan: SchedulePlan, max_num_seqs: int
) -> list[tuple[str, list[ScheduleEntry]]]:
    """Return the execution order: every Verify batch, then every Prefill batch."""

    return [
        *(("verify", batch) for batch in split_plan_microbatches(
            plan.verify_entries, max_num_seqs
        )),
        *(("prefill", batch) for batch in split_plan_microbatches(
            plan.prefill_entries, max_num_seqs
        )),
    ]


def advance_prefill_cursor(req: dict, chunk_end: int) -> bool:
    """Commit one completed Prefill chunk and report whether the prompt is done."""

    total_tokens = _prefill_total_tokens(req)
    old_cursor = int(req.get("prefill_cursor", 0))
    new_cursor = int(chunk_end)
    if not old_cursor < new_cursor <= total_tokens:
        raise ValueError(
            f"Prefill cursor must advance within ({old_cursor}, {total_tokens}], got {new_cursor}"
        )
    req["prefill_cursor"] = new_cursor
    return new_cursor == total_tokens


def uses_strict_fcfs(server_sched_mode: str) -> bool:
    return server_sched_mode in {"pipeline", "vanilla"}


def drain_queue_bounded(source_queue, max_items: int) -> tuple[list[Any], bool]:
    """Perform at most ``max_items`` non-blocking reads from an ingress queue."""

    limit = int(max_items)
    if limit <= 0:
        raise ValueError("max_items must be positive")
    items = []
    stop_received = False
    for _ in range(limit):
        try:
            item = source_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            stop_received = True
            break
        items.append(item)
    return items, stop_received


# Compatibility helpers for callers outside the current cloud worker. The V1
# FastSD path uses ``UnifiedSchedulerState`` and ``build_unified_schedule_plan``.
def should_switch_to_prefill(verify_underutilized_flags, has_prefill_tasks: bool) -> bool:
    return bool(has_prefill_tasks) and sum(int(flag) for flag in verify_underutilized_flags) >= 6


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
        category = order[(start_idx + offset) % total_slots]
        for item in verify_queues.get(category, ()):
            pid = item["proc_id"]
            if pid in pinned or pid in seen:
                continue
            predicted.append(pid)
            seen.add(pid)
            if len(predicted) >= batch_size:
                return predicted
    return predicted
