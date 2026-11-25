"""Compute a conservative verifier startup budget per participating GPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceBudget:
    device: str
    total: int
    free: int
    model_est: int
    kv_per_req: int
    logits_per_req: int = 0


def cache_bytes(config, max_len, element_size, tp_size=1):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if config.num_key_value_heads % tp_size:
        raise ValueError("KV heads must divide target_tp_size")
    return 2 * config.num_hidden_layers * (config.num_key_value_heads // tp_size) * head_dim * max_len * element_size


class BudgetExceeded(MemoryError):
    def __init__(self, rows, requested_batch, reserve_bytes):
        self.devices = rows
        self.requested_batch = requested_batch
        super().__init__(f"SpecEdge KV preallocation exceeds GPU budget (reserve={reserve_bytes}): {rows}")


def plan_budget(devices, *, instances, batch_size, max_len, max_n_beams, verifier_width,
                reserve_bytes, element_size, allow_memory_clamp=False, resident_slots=None):
    if not all(value > 0 for value in (instances, batch_size, max_len, max_n_beams, verifier_width, element_size)) or reserve_bytes < 0:
        raise ValueError("invalid SpecEdge memory dimensions or reserve")

    slots = instances if resident_slots is None else resident_slots
    if not 1 <= slots <= instances:
        raise ValueError("invalid resident slot count")

    def per_device(device, batch):
        kv_per_req = device.kv_per_req * max_n_beams
        resident = (slots + batch) * device.kv_per_req
        prefill_scratch = device.kv_per_req
        masks = element_size * (max_len**2 + 3 * batch * verifier_width * max_len)
        need = device.model_est + batch * kv_per_req + slots * device.kv_per_req + prefill_scratch + masks + batch * device.logits_per_req
        return dict(device=device.device, total=device.total, free=device.free, model_est=device.model_est,
                    kv_per_req=kv_per_req, actual_kv_per_slot=device.kv_per_req, batch=batch,
                    max_len=max_len, max_n_beams=max_n_beams, verifier_width=verifier_width,
                    prealloc_kv_bytes=resident, estimated_required_bytes=need,
                    available_bytes=min(device.total - reserve_bytes, device.free - reserve_bytes))

    requested_batch = batch_size
    while batch_size > 0:
        rows = [per_device(device, batch_size) for device in devices]
        if all(row["estimated_required_bytes"] < row["available_bytes"] for row in rows):
            return dict(batch_size=batch_size, requested_batch_size=requested_batch,
                        prealloc_clamped=batch_size != requested_batch, devices=rows,
                        prealloc_kv_bytes=sum(row["prealloc_kv_bytes"] for row in rows))
        if not allow_memory_clamp or batch_size == 1:
            raise BudgetExceeded(rows, requested_batch, reserve_bytes)
        batch_size -= 1
