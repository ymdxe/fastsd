import math
from dataclasses import dataclass


FASTSD_DEFAULT_R1 = 128
FASTSD_DEFAULT_R2 = 512
FASTSD_DYNAMIC_WINDOW = 100
FASTSD_VERIFY_PREFILL_SWITCH_THRESHOLD = 6
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

    arrival = req["arrival_time"] if "arrival_time" in req else req["current_time"]
    elapsed = max(0.0, float(now) - float(arrival))
    wait_term = math.exp(lamda * elapsed)
    if req["task_type"] == "prefill":
        if "prefill_time" in req:
            cost = (req["prefill_time"] + req["draft_time_per_token"] *
                    (req["gamma"] - 1) + req["push_time"])
            return -cost + wait_term  # Initial acceptance probability is one.
        return wait_term

    pid = req["proc_id"]
    accepted_sum, total_sum = accept_stats.get(pid, (0, 0))

    if total_sum <= 0:
        acc_prob = 1.0
    else:
        # Laplace smoothing prevents cold-start tasks from being catastrophically deprioritized.
        acc_prob = (accepted_sum + 1.0) / (total_sum + 1.0)
    acc_prob = max(acc_prob, 1e-6)

    if "push_time" in req:
        cost = (req["draft_time_per_token"] * req["gamma"] +
                req["push_time"] + req["pull_time"])
        return -cost / acc_prob + wait_term

    draft_len = max(1, int(req.get("gamma", 1) or 1))
    draft_total_time = float(req.get("lag", 0.0))
    if draft_total_time <= 0.0 and "draft_time_per_token" in req:
        draft_total_time = draft_len * max(0.0, float(req["draft_time_per_token"]))

    transport_rtt = max(0.0, float(req.get("transport_rtt", req.get("edge_rtt", 0.0))))
    return -((draft_total_time + transport_rtt) / acc_prob) + wait_term


def should_switch_to_prefill(verify_underutilized_flags, has_prefill_tasks: bool) -> bool:
    return bool(has_prefill_tasks) and sum(int(flag) for flag in verify_underutilized_flags) >= FASTSD_VERIFY_PREFILL_SWITCH_THRESHOLD


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
            pid = item["proc_id"]
            if pid in pinned or pid in seen:
                continue
            predicted.append(pid)
            seen.add(pid)
            if len(predicted) >= batch_size:
                return predicted

    return predicted


@dataclass
class Task:
    """One logical request; only execution advances its offset."""

    request: dict
    length: int
    offset: int = 0
    bridge: int = 0
    waited_cycles: int = 0
    last_served_cycle: int = -1
    verify_ms: float = 0.0

    @property
    def pid(self):
        return self.request["proc_id"]

    @property
    def kind(self):
        return self.request["task_type"]

    @property
    def remaining(self):
        return self.length - self.offset


@dataclass(frozen=True)
class Allocation:
    task: Task
    offset: int
    tokens: int
    bridge: int
    cycle: int = 0

    @property
    def cost(self):
        return self.tokens + self.bridge


@dataclass(frozen=True)
class BatchPlan:
    allocations: tuple
    cursor: int
    cycle: int


def rebuild_queues(tasks, thresholds, accept_stats, now):
    """Algorithm 1 QueueUpdate: preserve arrivals and execution progress."""
    queues = {kind: {c: [] for c in ("short", "mid", "long")}
              for kind in ("verify", "prefill")}
    for task in tasks:
        category = length_category(task.length, *thresholds[task.kind])
        queues[task.kind][category].append(task)
    for categories in queues.values():
        for items in categories.values():
            items.sort(key=lambda task: compute_priority_score(
                task.request, accept_stats, now), reverse=True)
    return queues


def plan_batch(queues, cursor, cycle, token_budget, max_devices, excluded=()):
    """Read-only WRR plan, reused verbatim for predictive preloading.

    Each class visit consumes one WRR slot. Soft fill borrows that visit and
    never charges another slot. Budget feasibility is checked before choosing
    a task, including any uncomputed bridge between its prefix and KV.
    """
    order = FASTSD_VERIFY_WRR_ORDER
    selected = set(excluded)
    allocations = []
    budget = token_budget

    def choose(category):
        overdue = [t for t in queues["prefill"][category] if t.waited_cycles >= 2]
        candidates = overdue + queues["verify"][category] + queues["prefill"][category]
        for task in candidates:
            bridge = task.bridge if task.offset == 0 else 0
            if task.pid not in selected and task.remaining > 0 and budget > bridge:
                return Allocation(task, task.offset, min(task.remaining, budget - bridge), bridge, visit_cycle)
        return None

    while budget > 0 and len(allocations) < max_devices:
        visit_cycle = cycle
        category = order[cursor]
        cursor = (cursor + 1) % len(order)
        if cursor == 0:
            cycle += 1
        allocation = choose(category)
        if allocation is None:
            for other in ("short", "mid", "long"):
                if other != category:
                    allocation = choose(other)
                    if allocation is not None:
                        break
        if allocation is None:
            break
        allocations.append(allocation)
        selected.add(allocation.task.pid)
        budget -= allocation.cost
    return BatchPlan(tuple(allocations), cursor, cycle)
