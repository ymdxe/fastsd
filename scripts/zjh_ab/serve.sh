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
  --kv_prefetch_chunk_kb "${KV_PREFETCH_CHUNK_KB:-4096}"
  --kv_prefetch_rate_mb_s "${KV_PREFETCH_RATE_MB_S:-0}"
  --kv_prefetch_idle_rate_mb_s "${KV_PREFETCH_IDLE_RATE_MB_S:-0}"
  --no-measure_forward_gpu
)

ENERGY=(--no-measure_energy)
if [[ "${MEASURE_ENERGY:-0}" == "1" || "${MEASURE_ENERGY:-0}" == "true" || "${MEASURE_ENERGY:-0}" == "yes" ]]; then
  ENERGY=(--measure_energy)
fi

COMMON=(
  --target_model "$TARGET_MODEL_NODE2"
  --draft_model "$TARGET_MODEL_NODE2"          # tokenizer only
  --target_dtype bfloat16 --draft_dtype bfloat16
  --target_device cuda:0
  --batch_size "$BATCH_SIZE" --token_budget "$TOKEN_BUDGET"
  --max_tokens "$MAX_TOKENS" --num_drafts "$NUM_DRAFTS" --gamma "$GAMMA"
  --seed "$SEED"
  --verify_method "${VERIFY_METHOD:-greedy}" --temp 0 --top_k 0 --top_p 1 --stop_policy eos
  "${ENERGY[@]}" --no-pipeline_gamma_adapt
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
  fastsd*)
    FASTSD_FLAGS=(--profile custom --server_sched_mode fastsd --scheduler_policy legacy)
    if [[ "$CASE" == *nopipeline* ]]; then
      FASTSD_FLAGS+=(--no-enable_pipeline --no-overlap_prefill_first_draft)
    else
      FASTSD_FLAGS+=(--enable_pipeline --overlap_prefill_first_draft)
    fi
    if [[ "$CASE" == *noproactive* ]]; then
      FASTSD_FLAGS+=(--no-enable_proactive_draft)
    else
      FASTSD_FLAGS+=(--enable_proactive_draft)
    fi
    if [[ "$CASE" == *nolatency* ]]; then
      FASTSD_FLAGS+=(--no-enable_latency_priority)
    else
      FASTSD_FLAGS+=(--enable_latency_priority)
    fi
    if [[ "$CASE" == *noprefetch* ]]; then
      FASTSD_FLAGS+=(--no-kv_prefetch)
    else
      FASTSD_FLAGS+=(--kv_prefetch)
    fi
    exec "$NODE2_PY" cloud/cloud_service.py "${COMMON[@]}" \
      "${FASTSD_FLAGS[@]}" --kv_prefetch_mode overlap
    ;;
  *)
    echo "unknown case: $CASE (expected sd-cpu or a fastsd* variant)" >&2
    exit 2
    ;;
esac
