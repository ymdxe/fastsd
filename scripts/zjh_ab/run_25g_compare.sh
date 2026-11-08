#!/usr/bin/env bash
# Overnight budget / wait-base sweep for the dual-edge setup.
#
# Run on node1 inside tmux:
#   ./scripts/zjh_ab/run_budget_sweep.sh
#
# node2: target on GPU1 + one edge (16 drafts)
# node1: one edge (16 drafts)
# Same full-HumanEval split (82/82) for every round.
set -euo pipefail

NODE2="${NODE2:-zhangh@node2.sanzo.top}"
NODE1="${NODE1:-zhangh@202.199.13.149}"
N2_REPO="${N2_REPO:-/home/hdd/zhangh/workspace/new_new_fastsd}"
N1_REPO="${N1_REPO:-/home/zhangh/workspace/new_new_fastsd}"
N2_PY="${N2_PY:-/home/hdd/zhangh/envs/new_fastsd/bin/python}"
N1_PY="${N1_PY:-/home/zhangh/venvs/fastsd_cpu/bin/python}"
PORT="${PORT:-8003}"
TUNNEL_PORT="${TUNNEL_PORT:-18003}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

log() { printf '%s %s\n' "$(date -Ins)" "$*"; }

CURRENT_TARGET_PID=""

cleanup_round() {
    if [ -n "$CURRENT_TARGET_PID" ]; then
        ssh "$NODE2" "kill -INT '$CURRENT_TARGET_PID' 2>/dev/null || true" || true
    fi
    ssh "$NODE2" "pkill -f 'cloud_service[.]py' || true; pkill -f 'edge[.]py' || true" || true
    pkill -f 'edge[.]py' || true
    CURRENT_TARGET_PID=""
}

ensure_tunnel() {
    if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/${TUNNEL_PORT}" 2>/dev/null; then
        return 0
    fi
    ssh -f -N -o BatchMode=yes -L "${TUNNEL_PORT}:127.0.0.1:${PORT}" "$NODE2"
}

wait_target() {
    local logfile="$1" pid="$2"
    local waited=0
    while true; do
        if ssh "$NODE2" "grep -q 'Loading checkpoint shards: 100%' '$logfile' 2>/dev/null"; then
            return 0
        fi
        if ! ssh "$NODE2" "kill -0 '$pid' 2>/dev/null"; then
            log "FATAL: target exited before ready"
            ssh "$NODE2" "tail -n 80 '$logfile'" || true
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

run_round() {
    local run="$1" budget_mb="$2" wait_base="$3"
    local target_log="/tmp/sweep_${run}_target.log"
    local n2_edge_log="/tmp/sweep_${run}_edge_node2.log"
    local n1_edge_log="/tmp/sweep_${run}_edge_node1.log"
    local target_pid n2_edge_pid

    log "=== $run START budget=${budget_mb}MB wait_base=${wait_base} ==="
    pkill -f 'edge[.]py' || true
    ssh "$NODE2" "pkill -f 'edge[.]py' || true; pkill -f 'cloud_service[.]py' || true" || true

    target_pid=$(ssh "$NODE2" "cd '$N2_REPO'; rm -f '$target_log'; \
        nohup env \
          FASTSD_WAIT_PRIORITY_BASE='$wait_base' \
          NUM_DRAFTS=32 BATCH_SIZE=8 TOKEN_BUDGET=64 MAX_TOKENS=400 GAMMA=4 \
          KV_RESIDENCY_BUDGET_MB='$budget_mb' \
          KV_PREFETCH_MB=256 KV_PREFETCH_RATE_MB_S=256 KV_PREFETCH_IDLE_RATE_MB_S=1024 \
          RUN_NAME='$run' \
          ./scripts/zjh_ab/serve.sh fastsd >'$target_log' 2>&1 & echo \$!")
    wait_target "$target_log" "$target_pid"

    CURRENT_TARGET_PID="$target_pid"
    trap cleanup_round EXIT

    n2_edge_pid=$(ssh "$NODE2" "cd '$N2_REPO'; rm -f '$n2_edge_log'; \
        nohup env \
          NODE3_PY='$N2_PY' \
          MODELS=/mnt/nfs0/zhangh/models \
          TARGET_MODEL=/mnt/nfs0/zhangh/models/Qwen3-8B \
          DRAFT_MODEL=/mnt/nfs0/zhangh/models/Qwen3-0.6B \
          DATA_PATH=data_dual_node2 REQUESTS=82 NUM_DRAFTS=16 EDGE_THREADS=4 \
          BATCH_SIZE=8 TOKEN_BUDGET=64 MAX_TOKENS=400 GAMMA=4 \
          ARRIVAL_DISTRIBUTION=poisson ARRIVAL_RATE=0.64 ARRIVAL_SEED=5678 \
          TARGET_URL=http://127.0.0.1:${PORT} RUN_NAME='$run' \
          ./scripts/zjh_ab/client.sh fastsd >'$n2_edge_log' 2>&1 & echo \$!")

    log "node2 edge pid=$n2_edge_pid; running node1 edge"
    local edge1_ok=1 edge2_ok=1
    if ! (cd "$N1_REPO" && env \
        NODE3_PY="$N1_PY" \
        MODELS=/mnt/nfs0/zhangh/models \
        TARGET_MODEL=/mnt/nfs0/zhangh/models/Qwen3-8B \
        DRAFT_MODEL=/mnt/nfs0/zhangh/models/Qwen3-0.6B \
        DATA_PATH=data_dual_node1 REQUESTS=82 NUM_DRAFTS=16 EDGE_THREADS=4 \
        BATCH_SIZE=8 TOKEN_BUDGET=64 MAX_TOKENS=400 GAMMA=4 \
        ARRIVAL_DISTRIBUTION=poisson ARRIVAL_RATE=0.64 ARRIVAL_SEED=1234 \
        TARGET_URL=http://127.0.0.1:${TUNNEL_PORT} RUN_NAME="$run" \
        ./scripts/zjh_ab/client.sh fastsd >"$n1_edge_log" 2>&1); then
        log "WARN: node1 edge failed; continuing to collect target metrics"
        tail -n 40 "$n1_edge_log" || true
        edge1_ok=0
    fi

    if ! ssh "$NODE2" "while kill -0 '$n2_edge_pid' 2>/dev/null; do sleep 2; done"; then
        log "WARN: node2 edge failed; continuing to collect target metrics"
        ssh "$NODE2" "tail -n 40 '$n2_edge_log'" || true
        edge2_ok=0
    fi

    ssh "$NODE2" "curl -s -X POST http://127.0.0.1:${PORT}/exit >/dev/null 2>&1 || true"
    local waited=0
    while ! ssh "$NODE2" "test -s '$N2_REPO/exp/zjh_ab/$run/target/scheduler_metrics.json'"; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge 60 ]; then
            log "WARN: scheduler metrics missing for $run"
            break
        fi
    done
    cleanup_round
    trap - EXIT
    if [ "$edge1_ok" -eq 0 ] || [ "$edge2_ok" -eq 0 ]; then
        log "=== $run DONE WITH EDGE FAILURE ==="
        return 1
    fi
    log "=== $run DONE ==="
}

ensure_tunnel

# Same 2.5GB pressure, only the wait-priority base changes.
run_round dual-25g-p2 2560 2.0               || log "WARN: dual-25g-p2 failed"
run_round dual-25g-pe 2560 2.718281828459045 || log "WARN: dual-25g-pe failed"

log "=== SWEEP DONE ==="
