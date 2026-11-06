import os


def should_offload_target_cache(task_type: str, pid=None, pinned_gpu_pids=None) -> bool:
    if os.environ.get("FASTSD_DISABLE_TARGET_CACHE_OFFLOAD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    if task_type == "prefill":
        return True
    if task_type == "verify":
        return pid not in (pinned_gpu_pids or set())
    return False
