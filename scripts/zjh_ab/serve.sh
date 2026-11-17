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

CASE="${1:?usage: serve.sh <ar|sd-cpu|fastsd>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=params.env
source "$HERE/params.env"

if [[ -n "${TARGET_TP_GPUS:-}" ]]; then
  TARGET_VISIBLE_GPUS="$TARGET_TP_GPUS"
else
  TARGET_VISIBLE_GPUS="${TARGET_GPU}"
  # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to visible ordinal 0.
  TARGET_TP_GPUS="0"
fi

RUN="${RUN_NAME:-$CASE}"
EXP_DIR="$EXP_ROOT/$RUN/target"
mkdir -p "$EXP_DIR"

POWER_MONITOR_PID=""
stop_power_monitor() {
  if [[ -n "$POWER_MONITOR_PID" ]]; then
    kill -TERM "$POWER_MONITOR_PID" 2>/dev/null || true
    wait "$POWER_MONITOR_PID" 2>/dev/null || true
    POWER_MONITOR_PID=""
  fi
}
start_power_monitor() {
  case "${POWER_MONITOR:-0}" in
    1|true|yes|on)
      "$NODE2_PY" -m src.power_monitor \
        --role target \
        --experiment-id "${POWER_EXPERIMENT_ID:-$RUN}" \
        --output-dir "$EXP_DIR" \
        --sample-interval-ms "${POWER_SAMPLE_INTERVAL_MS:-250}" \
        --root-pid "$$" \
        --gpu-index 0 \
        >"$EXP_DIR/power_monitor.log" 2>&1 &
      POWER_MONITOR_PID=$!
      echo "$POWER_MONITOR_PID" >"$EXP_DIR/power_monitor.pid"
      ;;
  esac
}
trap 'stop_power_monitor' EXIT
trap 'stop_power_monitor; exit 143' INT TERM

export CLOUD_SERVICE_HOST CLOUD_SERVICE_PORT
export CUDA_VISIBLE_DEVICES="$TARGET_VISIBLE_GPUS"
export TARGET_TP_SIZE TARGET_TP_GPUS

RESIDENCY=(
  --kv_residency_budget_mb "$KV_RESIDENCY_BUDGET_MB"
  --kv_residency_high_ratio "$KV_HIGH"
  --kv_residency_low_ratio "$KV_LOW"
  --kv_prefetch_online_fraction "$KV_PREFETCH_ONLINE_FRACTION"
  --kv_prefetch_mb "$KV_PREFETCH_MB"
  --kv_prefetch_implementation "${KV_PREFETCH_IMPLEMENTATION:-h1}"
  --kv_prefetch_chunk_kb "${KV_PREFETCH_CHUNK_KB:-4096}"
  --kv_prefetch_rate_mb_s "${KV_PREFETCH_RATE_MB_S:-100000}"
  --kv_prefetch_idle_rate_mb_s "${KV_PREFETCH_IDLE_RATE_MB_S:-100000}"
  --no-measure_forward_gpu
)
if [[ "${KV_OVERLAP_AUTO_FALLBACK:-1}" == "0" ]]; then
  RESIDENCY+=(--no-kv_overlap_auto_fallback)
fi

ENERGY=(--no-measure_energy)
if [[ "${MEASURE_ENERGY:-0}" == "1" || "${MEASURE_ENERGY:-0}" == "true" || "${MEASURE_ENERGY:-0}" == "yes" ]]; then
  ENERGY=(--measure_energy)
fi

COMMON=(
  --target_model "$TARGET_MODEL_NODE2"
  --target_dtype bfloat16 --draft_dtype bfloat16
  --target_device cuda:0
  --target_tp_size "${TARGET_TP_SIZE:-1}"
  --target_tp_gpus "${TARGET_TP_GPUS:-0}"
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
  ar|target-only|target_only)
    export FASTSD_TARGET_MODE=ar
    start_power_monitor
    "$NODE2_PY" cloud/cloud_service.py "${COMMON[@]}" \
      --profile vanilla --server_sched_mode vanilla --scheduler_policy fcfs \
      --no-enable_pipeline --no-enable_proactive_draft \
      --no-enable_latency_priority --no-overlap_prefill_first_draft \
      --no-kv_prefetch --kv_prefetch_mode overlap \
      --kv_residency_victim_policy lru
    ;;
  sd-cpu)
    export FASTSD_TARGET_MODE=speculative
    COMMON+=(--draft_model "${DRAFT_MODEL:-$N2_DRAFT_MODEL}")
    start_power_monitor
    "$NODE2_PY" cloud/cloud_service.py "${COMMON[@]}" \
      --profile vanilla --server_sched_mode vanilla --scheduler_policy fcfs \
      --no-enable_pipeline --no-enable_proactive_draft \
      --no-enable_latency_priority --no-overlap_prefill_first_draft \
      --no-kv_prefetch --kv_prefetch_mode overlap \
      --kv_residency_victim_policy lru
    ;;
  fastsd*)
    export FASTSD_TARGET_MODE=speculative
    COMMON+=(--draft_model "${DRAFT_MODEL:-$N2_DRAFT_MODEL}")
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
    start_power_monitor
    "$NODE2_PY" cloud/cloud_service.py "${COMMON[@]}" \
      "${FASTSD_FLAGS[@]}" --kv_prefetch_mode overlap
    ;;
  *)
    echo "unknown case: $CASE (expected sd-cpu or a fastsd* variant)" >&2
    exit 2
    ;;
esac
