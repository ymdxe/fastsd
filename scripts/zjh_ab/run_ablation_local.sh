#!/usr/bin/env bash
# Fast local ablation on the node1 GPU account.
#
# target + one 32-draft CPU edge both run in this account, so there is no
# cross-account tunnel and no node2 dependency.  Same parameters for every
# round; only the component under test changes.
set -euo pipefail

REPO="${REPO:-$HOME/new_new_fastsd}"
PY="${PY:-$(command -v python)}"
PORT="${PORT:-8003}"
TARGET_GPU="${TARGET_GPU:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

cd "$REPO"

log() { printf '%s %s\n' "$(date -Ins)" "$*"; }

wait_target() {
    local logfile="$1" pid="$2" waited=0
    while true; do
        if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "FATAL: target exited before ready"
            tail -n 80 "$logfile" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$READY_TIMEOUT" ]; then
            log "FATAL: target not ready after ${READY_TIMEOUT}s"
            return 1
        fi
    done
}

cleanup() {
    if [ -n "${TARGET_PID:-}" ]; then
        curl -s -X POST "http://127.0.0.1:${PORT}/exit" >/dev/null 2>&1 || true
        kill -INT "$TARGET_PID" 2>/dev/null || true
    fi
    pkill -f 'edge[.]py' || true
}

run_round() {
    local run="$1" budget_mb="$2" wait_t0="$3" rounds="$4" case_name="$5" edge_case="$6"
    local target_log="/tmp/local_${run}_target.log"
    local edge_log="/tmp/local_${run}_edge.log"

    log "=== $run START budget=${budget_mb} t0=${wait_t0} K=${rounds} case=${case_name} ==="
    pkill -f 'edge[.]py' || true
    pkill -f 'cloud_service[.]py' || true

    env \
      TARGET_GPU="$TARGET_GPU" \
      TARGET_MODEL_NODE2=/mnt/nfs0/zhangh/models/Qwen3-8B \
      FASTSD_WAIT_LINEAR_UNTIL_S="$wait_t0" \
      KV_PREFETCH_LOOKAHEAD_ROUNDS="$rounds" \
      NUM_DRAFTS=32 BATCH_SIZE=8 TOKEN_BUDGET=32 MAX_TOKENS=400 GAMMA=8 \
      KV_RESIDENCY_BUDGET_MB="$budget_mb" \
      KV_PREFETCH_MB=256 KV_PREFETCH_RATE_MB_S=256 KV_PREFETCH_IDLE_RATE_MB_S=1024 \
      NODE2_PY="$PY" \
      RUN_NAME="$run" \
      ./scripts/zjh_ab/serve.sh "$case_name" >"$target_log" 2>&1 &
    TARGET_PID=$!
    wait_target "$target_log" "$TARGET_PID"

    env \
      NODE3_PY="$PY" \
      MODELS=/mnt/nfs0/zhangh/models \
      TARGET_MODEL=/mnt/nfs0/zhangh/models/Qwen3-8B \
      DRAFT_MODEL=/mnt/nfs0/zhangh/models/Qwen3-0.6B \
      DATA_PATH=data REQUESTS=164 NUM_DRAFTS=32 EDGE_THREADS=2 \
      BATCH_SIZE=8 TOKEN_BUDGET=32 MAX_TOKENS=400 GAMMA=8 \
      ARRIVAL_DISTRIBUTION=poisson ARRIVAL_RATE=0.1 ARRIVAL_SEED=1234 \
      TARGET_URL="http://127.0.0.1:${PORT}" \
      RUN_NAME="$run" \
      ./scripts/zjh_ab/client.sh "$edge_case" >"$edge_log" 2>&1 || true

    curl -s -X POST "http://127.0.0.1:${PORT}/exit" >/dev/null 2>&1 || true
    sleep 3
    kill -INT "$TARGET_PID" 2>/dev/null || true
    wait "$TARGET_PID" 2>/dev/null || true
    TARGET_PID=""
    log "=== $run DONE ==="
}

trap cleanup EXIT

run_round ablation_local_exp_la1        2048 0  1 fastsd             fastsd
run_round ablation_local_exp_la3        2048 0  3 fastsd             fastsd
run_round ablation_local_piecewise_la3  2048 30 3 fastsd             fastsd
run_round ablation_local_noprefetch_la1 2048 0  1 fastsd-noprefetch  fastsd
run_round ablation_local_sd_32d         0    0  1 sd-cpu             sd-cpu

trap - EXIT
log "=== LOCAL ABLATION DONE ==="
