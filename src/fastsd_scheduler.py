import math


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

    elapsed = max(0.0, float(now) - float(req["current_time"]))
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
