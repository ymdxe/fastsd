#!/usr/bin/env bash
# serve.sh — run the cloud target on node2. Foreground; Ctrl-C stops it.
#
#   ./serve.sh sd-cpu                  # FCFS, no speculation, no prefetch
#   ./serve.sh fastsd                  # FastSD default configuration
#   KV_RESIDENCY_BUDGET_MB=256 RUN_NAME=pressure-fastsd ./serve.sh fastsd
#
# Nothing on this host talks to node3.  It just serves and, on exit, writes
# exp/zjh_ab/<run>/target/scheduler_metrics.json.
set -euo pipefail

CASE="${1:?usage: serve.sh <sd-cpu|fastsd>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=params.env
source "$HERE/params.env"

RUN="${RUN_NAME:-$CASE}"
EXP_DIR="$EXP_ROOT/$RUN/target"
mkdir -p "$EXP_DIR"

export CLOUD_SERVICE_HOST CLOUD_SERVICE_PORT
export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"

RESIDENCY=(
  --kv_residency_budget_mb "$KV_RESIDENCY_BUDGET_MB"
  --kv_residency_high_ratio "$KV_HIGH"
  --kv_residency_low_ratio "$KV_LOW"
  --kv_prefetch_online_fraction "$KV_PREFETCH_ONLINE_FRACTION"
  --kv_prefetch_mb "$KV_PREFETCH_MB"
  --kv_prefetch_rate_mb_s "${KV_PREFETCH_RATE_MB_S:-64}"
  --kv_prefetch_idle_rate_mb_s "${KV_PREFETCH_IDLE_RATE_MB_S:-512}"
  --no-measure_forward_gpu
)

COMMON=(
  --target_model "$TARGET_MODEL_NODE2"
  --draft_model "$TARGET_MODEL_NODE2"          # tokenizer only
  --target_dtype bfloat16 --draft_dtype bfloat16
  --target_device cuda:0
  --batch_size "$BATCH_SIZE" --token_budget "$TOKEN_BUDGET"
  --max_tokens "$MAX_TOKENS" --num_drafts "$NUM_DRAFTS" --gamma "$GAMMA"
  --seed "$SEED"
  --verify_method greedy --temp 0 --top_k 0 --top_p 1 --stop_policy eos
  --no-measure_energy --no-pipeline_gamma_adapt
  --target_attention_backend sdpa_per_sequence --kv_batch_mode varlen
  --session_idle_timeout 1800
  --exp_name "$EXP_DIR"
  "${RESIDENCY[@]}"
)

case "$CASE" in
  sd-cpu)
    exec "$NODE2_PY" cloud/cloud_service.py "${COMMON[@]}" \
      --profile vanilla --server_sched_mode vanilla --scheduler_policy fcfs \
      --no-enable_pipeline --no-enable_proactive_draft \
      --no-enable_latency_priority --no-overlap_prefill_first_draft \
      --no-kv_prefetch --kv_prefetch_mode overlap
    ;;
  fastsd)
    exec "$NODE2_PY" cloud/cloud_service.py "${COMMON[@]}" \
      --profile custom --server_sched_mode fastsd --scheduler_policy legacy \
      --enable_pipeline --enable_proactive_draft --enable_latency_priority \
      --overlap_prefill_first_draft --kv_prefetch --kv_prefetch_mode overlap
    ;;
  *)
    echo "unknown case: $CASE (expected sd-cpu|fastsd)" >&2
    exit 2
    ;;
esac
