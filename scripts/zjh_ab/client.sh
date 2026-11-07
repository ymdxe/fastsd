#!/usr/bin/env bash
# client.sh — run the draft edge on node3 against an already-running target.
#
#   ./client.sh sd-cpu
#   ./client.sh fastsd
#   KV_RESIDENCY_BUDGET_MB=256 RUN_NAME=pressure-fastsd ./client.sh fastsd
#
# This host never starts or stops the target.  It waits for /health, then runs
# the edge and writes exp/zjh_ab/<run>/edge/.  If node3 cannot reach node2 over
# HTTP directly it falls back to an SSH tunnel for its own connection only.
set -euo pipefail

CASE="${1:?usage: client.sh <sd-cpu|fastsd>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=params.env
source "$HERE/params.env"

# Dual-edge / Poisson overrides (safe defaults preserve the old single-edge run).
DATA_PATH="${DATA_PATH:-data}"
ARRIVAL_DISTRIBUTION="${ARRIVAL_DISTRIBUTION:-immediate}"
ARRIVAL_RATE="${ARRIVAL_RATE:-1.0}"
ARRIVAL_SEED="${ARRIVAL_SEED:-1234}"

RUN="${RUN_NAME:-$CASE}"
EXP_DIR="$EXP_ROOT/$RUN/edge"
mkdir -p "$EXP_DIR"

RUN() { if [ "$DRY_RUN" = "1" ]; then printf '  [dry] %s\n' "$*"; else "$@"; fi; }

# edge.py imports diagnostic_trace before it fixes sys.path.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# HuggingFace tokenizers/Rayon creates a Rust global thread pool sized by the
# host CPU count.  With many draft workers that alone can exhaust the pids
# limit, so keep it tiny; torch threads are controlled by --edge_threads.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

TUNNEL_PID=""
cleanup() { [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT

SERVER_URL="$TARGET_URL"
say() { printf '\n=== %s ===\n' "$*"; }

say "check target at $TARGET_URL"
if [ "$DRY_RUN" != "1" ]; then
  if curl -sf -m 3 "$TARGET_URL/health" >/dev/null 2>&1; then
    echo "direct HTTP reachable"
  else
    echo "direct HTTP blocked -> opening an SSH tunnel for this host only"
    ssh -o BatchMode=yes -N -L "$TUNNEL_LOCAL_PORT:127.0.0.1:$CLOUD_SERVICE_PORT" \
        "$SSH_NODE2" &
    TUNNEL_PID=$!
    SERVER_URL="http://127.0.0.1:$TUNNEL_LOCAL_PORT"
    sleep 2
  fi

  say "wait for /health"
  waited=0
  until curl -sf -m 3 "$SERVER_URL/health" >/dev/null 2>&1; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$HEALTH_TIMEOUT_S" ]; then
      echo "target not healthy after ${HEALTH_TIMEOUT_S}s at $SERVER_URL" >&2
      echo "start it on node2 first: ./scripts/zjh_ab/serve.sh $CASE" >&2
      exit 1
    fi
  done
  echo "healthy after ${waited}s"
fi

DEVICE=cpu
EXTRA=()
if [ "$CASE" = "fastsd" ]; then
  EXTRA=(--server_sched_mode fastsd --enable_pipeline --enable_proactive_draft
         --enable_latency_priority --overlap_prefill_first_draft)
else
  EXTRA=(--server_sched_mode vanilla --no-enable_pipeline
         --no-enable_proactive_draft --no-enable_latency_priority
         --no-overlap_prefill_first_draft)
fi

ARGS=(
  --dataset "$DATASET" --data_path "$DATA_PATH"
  --arrival_distribution "$ARRIVAL_DISTRIBUTION"
  --arrival_rate "$ARRIVAL_RATE"
  --arrival_seed "$ARRIVAL_SEED"
  --draft_model "$DRAFT_MODEL" --target_model "$TARGET_MODEL"
  --draft_dtype bfloat16 --num_drafts "$NUM_DRAFTS" --edge_gpus 1
  --max_tasks_per_draft $(( (REQUESTS + NUM_DRAFTS - 1) / NUM_DRAFTS ))
  --max_tokens "$MAX_TOKENS" --gamma "$GAMMA" --seed "$SEED"
  --edge_threads "$EDGE_THREADS"
  --batch_size "$BATCH_SIZE" --token_budget "$TOKEN_BUDGET" --kv_batch_mode varlen
  --edge_use_cpu
  --request_timeout 600
  --timing_max_clock_uncertainty_ms 100
  --server_url "$SERVER_URL"
  "${EXTRA[@]}"
  --exp_name "$EXP_DIR"
)

say "run edge [$RUN] draft_device=$DEVICE"
if [ "$DRY_RUN" = "1" ]; then
  RUN env CUDA_VISIBLE_DEVICES="$DRAFT_GPU" "$NODE3_PY" edge/edge.py "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$DRAFT_GPU" timeout "$RUN_TIMEOUT_S" \
    "$NODE3_PY" edge/edge.py "${ARGS[@]}"
fi
say "edge finished -> $EXP_DIR"
