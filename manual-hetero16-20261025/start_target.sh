#!/usr/bin/env bash
# User-invoked only. Does not stop or replace another service.
set -euo pipefail
REPO=/home/hdd3/zhangh/workspace/fastsd
PYTHON=/home/hdd/zhangh/envs/new_fastsd/bin/python
cd "$REPO"
if ss -ltn 'sport = :19635' | tail -n +2 | grep -q LISTEN; then
  echo 'Port 19635 is already listening. Inspect the existing service before starting another.' >&2
  exit 1
fi
nvidia-smi
GPU_ID="${1:-}"
if [[ -z "$GPU_ID" ]]; then
  read -r -p 'Physical GPU ID allocated to this experiment: ' GPU_ID
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo 'Expected one numeric physical GPU ID.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export FASTSD_TARGET_DEVICE=cuda:0
export CLOUD_SERVICE_HOST=219.216.65.42 CLOUD_SERVICE_PORT=19635
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$REPO/exp"
TARGET_RUN=$(mktemp -d "$REPO/exp/hetero16_target_$(date +%Y%m%d_%H%M%S)_XXXXXX")
cmd=("$PYTHON" -u cloud/cloud_service.py
  --exp_name "$TARGET_RUN"
  --draft_model /home/hdd/zhangh/models/Qwen3-0.6B
  --target_model /home/hdd/zhangh/models/Qwen3-8B
  --target_device cuda:0 --target_dtype bfloat16
  --profile custom --server_sched_mode fastsd --kv_batch_mode varlen
  --batch_size 16 --token_budget 512 --gamma 8 --max_tokens 32
  --verify_method greedy --temp 0 --top_k 0 --top_p 1
  --no-enable_pipeline --no-enable_proactive_draft
  --no-enable_latency_priority --no-pipeline_gamma_adapt --no-measure_energy)
{
  printf 'CUDA_VISIBLE_DEVICES=%q FASTSD_TARGET_DEVICE=cuda:0 CLOUD_SERVICE_HOST=219.216.65.42 CLOUD_SERVICE_PORT=19635 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ' "$GPU_ID"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$TARGET_RUN/commands.txt"
git rev-parse HEAD > "$TARGET_RUN/git-head.txt"
git status --short > "$TARGET_RUN/git-status.txt"
sha256sum cloud/cloud_service.py src/engine.py src/util.py src/fastsd_scheduler.py > "$TARGET_RUN/source-sha256.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free --format=csv > "$TARGET_RUN/gpu-before.csv"
df -h /home/hdd /home/hdd3 > "$TARGET_RUN/disk-before.txt"
printf 'Target output: %s\n' "$TARGET_RUN"
set +e
"${cmd[@]}" 2>&1 | tee "$TARGET_RUN/target.log"
statuses=("${PIPESTATUS[@]}")
printf 'target_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$TARGET_RUN/exit-status.txt"
exit "${statuses[0]}"
