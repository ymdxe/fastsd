#!/usr/bin/env bash
# Seven-condition ablation:
#   target: node1 GPU account, GPU1, port 8003
#   edge A: node1 CPU account
#   edge B: node2 CPU account
#
# Requires two already-open localhost:18003 -> GPU:8003 tunnels:
#   node1 CPU: ssh -N -L 18003:127.0.0.1:8003 zhangh@node1.sanzo.top
#   node2 CPU: ssh -N -L 18003:127.0.0.1:8003 zhangh@node1.sanzo.top
set -euo pipefail

GPU="${GPU:-zhangh@node1.sanzo.top}"
N2="${N2:-zhangh@node2.sanzo.top}"
GPU_REPO="${GPU_REPO:-\$HOME/new_new_fastsd}"
N1_REPO="${N1_REPO:-/home/zhangh/workspace/new_new_fastsd}"
N2_REPO="${N2_REPO:-/home/hdd/zhangh/workspace/new_new_fastsd}"
N1_PY="${N1_PY:-/home/zhangh/venvs/fastsd_cpu/bin/python}"
N2_PY="${N2_PY:-/home/hdd/zhangh/envs/new_fastsd/bin/python}"
NFS_BASE="${NFS_BASE:-/mnt/nfs0/zhangh/fastsd_exp}"
GPU_TMP="${GPU_TMP:-/dev/shm}"
N2_TMP="${N2_TMP:-/dev/shm}"
GPU_EXP_ROOT="${GPU_EXP_ROOT:-$GPU_TMP/fastsd_exp/target}"
N1_EXP_ROOT="${N1_EXP_ROOT:-$NFS_BASE/node1}"
N2_EXP_ROOT="${N2_EXP_ROOT:-$N2_TMP/fastsd_exp/node2}"
TARGET_GPU="${TARGET_GPU:-1}"
PORT="${PORT:-8003}"
TUNNEL_PORT="${TUNNEL_PORT:-18003}"
# Node1 CPU reaches the target through its tunnel.  When the target and the
# node2 edge are the same host, node2 can use 8003 directly.
N1_TARGET_URL="${N1_TARGET_URL:-http://127.0.0.1:${TUNNEL_PORT}}"
N2_TARGET_URL="${N2_TARGET_URL:-http://127.0.0.1:${TUNNEL_PORT}}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

log() { printf '%s %s\n' "$(date -Ins)" "$*"; }

wait_target() {
    local logfile="$1" pid="$2" waited=0
    while true; do
        if ssh "$GPU" "curl -sf -m 2 http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ssh "$GPU" "kill -0 '$pid' 2>/dev/null"; then
            log "FATAL: target exited before ready"
            ssh "$GPU" "tail -n 80 '$logfile'" || true
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
    local run="$1" budget_mb="$2" wait_t0="$3" rounds="$4" case_name="$5" edge_case="$6"
    local target_log="${GPU_TMP}/ablation_${run}_target.log"
    local n1_log="${NFS_BASE}/logs/ablation_${run}_edge_node1.log"
    local n2_log="${N2_TMP}/ablation_${run}_edge_node2.log"
    local target_pid n2_pid edge1_ok=1 edge2_ok=1

    mkdir -p "${NFS_BASE}/logs"
    log "=== $run START budget=${budget_mb} t0=${wait_t0} K=${rounds} case=${case_name} edge=${edge_case} ==="
    # Re-running the same condition must not append to stale edge/target files.
    ssh "$GPU" "rm -rf \"$GPU_EXP_ROOT/$run\"" || true
    rm -rf "$N1_EXP_ROOT/$run" || true
    ssh "$N2" "rm -rf '$N2_EXP_ROOT/$run'" || true
    pkill -f 'edge[.]py' || true
    ssh "$N2" "pkill -f 'edge[.]py' || true" || true
    ssh "$GPU" "pkill -f 'cloud_service[.]py' || true" || true

    target_pid=$(ssh "$GPU" "cd \"$GPU_REPO\"; rm -f '$target_log'; \
        nohup env \
          TARGET_GPU='$TARGET_GPU' \
          PYTHONDONTWRITEBYTECODE=1 \
          EXP_ROOT='$GPU_EXP_ROOT' \
          TARGET_MODEL_NODE2=/mnt/nfs0/zhangh/models/Qwen3-8B \
          FASTSD_WAIT_LINEAR_UNTIL_S='$wait_t0' \
          KV_PREFETCH_LOOKAHEAD_ROUNDS='$rounds' \
          NUM_DRAFTS=32 BATCH_SIZE=8 TOKEN_BUDGET=32 MAX_TOKENS=400 GAMMA=8 \
          KV_RESIDENCY_BUDGET_MB='$budget_mb' \
          KV_PREFETCH_MB=256 KV_PREFETCH_RATE_MB_S=256 KV_PREFETCH_IDLE_RATE_MB_S=1024 \
          RUN_NAME='$run' \
          ./scripts/zjh_ab/serve.sh '$case_name' >'$target_log' 2>&1 & echo \$!")
    wait_target "$target_log" "$target_pid"

    if ! curl -sf -m 3 "http://127.0.0.1:${TUNNEL_PORT}/health" >/dev/null 2>&1; then
        log "FATAL: node1 tunnel localhost:${TUNNEL_PORT} is not healthy"
        ssh "$GPU" "curl -s -X POST http://127.0.0.1:${PORT}/exit >/dev/null 2>&1 || true"
        return 1
    fi
    if ! ssh "$N2" "curl -sf -m 3 '$N2_TARGET_URL/health'" >/dev/null 2>&1; then
        log "FATAL: node2 target URL $N2_TARGET_URL is not healthy"
        ssh "$GPU" "curl -s -X POST http://127.0.0.1:${PORT}/exit >/dev/null 2>&1 || true"
        return 1
    fi

    n2_pid=$(ssh "$N2" "cd '$N2_REPO'; rm -f '$n2_log'; \
        nohup env \
          NODE3_PY='$N2_PY' \
          PYTHONDONTWRITEBYTECODE=1 \
          MODELS=/mnt/nfs0/zhangh/models \
          TARGET_MODEL=/mnt/nfs0/zhangh/models/Qwen3-8B \
          DRAFT_MODEL=/mnt/nfs0/zhangh/models/Qwen3-0.6B \
          EXP_ROOT='$N2_EXP_ROOT' \
          DATA_PATH=data_dual_node2 REQUESTS=82 NUM_DRAFTS=16 EDGE_THREADS=4 \
          BATCH_SIZE=8 TOKEN_BUDGET=32 MAX_TOKENS=400 GAMMA=8 \
          ARRIVAL_DISTRIBUTION=poisson ARRIVAL_RATE=1.6 ARRIVAL_SEED=5678 \
          TARGET_URL='$N2_TARGET_URL' RUN_NAME='$run' \
          ./scripts/zjh_ab/client.sh '$edge_case' >'$n2_log' 2>&1 & echo \$!")

    log "node2 edge pid=$n2_pid; running node1 edge"
    if ! (cd "$N1_REPO" && env \
        NODE3_PY="$N1_PY" \
        PYTHONDONTWRITEBYTECODE=1 \
        MODELS=/mnt/nfs0/zhangh/models \
        TARGET_MODEL=/mnt/nfs0/zhangh/models/Qwen3-8B \
        DRAFT_MODEL=/mnt/nfs0/zhangh/models/Qwen3-0.6B \
        EXP_ROOT="$N1_EXP_ROOT" \
        DATA_PATH=data_dual_node1 REQUESTS=82 NUM_DRAFTS=16 EDGE_THREADS=4 \
        BATCH_SIZE=8 TOKEN_BUDGET=32 MAX_TOKENS=400 GAMMA=8 \
        ARRIVAL_DISTRIBUTION=poisson ARRIVAL_RATE=1.6 ARRIVAL_SEED=1234 \
        TARGET_URL="$N1_TARGET_URL" RUN_NAME="$run" \
        ./scripts/zjh_ab/client.sh "$edge_case" >"$n1_log" 2>&1); then
        log "WARN: node1 edge failed"
        tail -n 40 "$n1_log" || true
        edge1_ok=0
    fi

    if ! ssh "$N2" "while kill -0 '$n2_pid' 2>/dev/null; do sleep 2; done"; then
        log "WARN: node2 edge failed"
        ssh "$N2" "tail -n 40 '$n2_log'" || true
        edge2_ok=0
    fi

    ssh "$GPU" "curl -s -X POST http://127.0.0.1:${PORT}/exit >/dev/null 2>&1 || true"

    # Do not kill the target before its finalizer has written metrics.
    local metrics_path="$GPU_EXP_ROOT/$run/target/scheduler_metrics.json"
    local waited=0
    while ! ssh "$GPU" "test -s \"$metrics_path\"" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge 60 ]; then
            log "WARN: scheduler metrics missing for $run after ${waited}s"
            break
        fi
    done
    ssh "$GPU" "kill -INT '$target_pid' 2>/dev/null || true" || true
    ssh "$GPU" "pkill -f 'cloud_service[.]py' || true" || true
    pkill -f 'edge[.]py' || true
    ssh "$N2" "pkill -f 'edge[.]py' || true" || true

    if [ "$edge1_ok" -eq 0 ] || [ "$edge2_ok" -eq 0 ]; then
        log "=== $run DONE WITH EDGE FAILURE ==="
        return 1
    fi
    log "=== $run DONE ==="
}

# Tunnels are checked after each target starts: a tunnel can exist while the
# target is intentionally down, so a preflight /health check would be wrong.

# Seven conditions.  T0=30 new piecewise, T0=0 old exp.
run_round ablation_piecewise_la1_t5 2048 5 1 fastsd             fastsd || log "WARN: ablation_piecewise_la1 failed"
run_round ablation_piecewise_la3_t5 2048 5 3 fastsd             fastsd || log "WARN: ablation_piecewise_la3 failed"
run_round ablation_exp_la1       2048 0  1 fastsd             fastsd || log "WARN: ablation_exp_la1 failed"
run_round ablation_exp_la3       2048 0  3 fastsd             fastsd || log "WARN: ablation_exp_la3 failed"
run_round ablation_noprefetch_piecewise 2048 30 1 fastsd-noprefetch fastsd || log "WARN: ablation_noprefetch_piecewise failed"
run_round ablation_noprefetch_exp       2048 0  1 fastsd-noprefetch fastsd || log "WARN: ablation_noprefetch_exp failed"
run_round ablation_sd_32d               0    0  1 sd-cpu             sd-cpu  || log "WARN: ablation_sd_32d failed"
run_round ablation_sd_offload_32d    2048 0  1 sd-cpu             sd-cpu  || log "WARN: ablation_sd_offload_32d failed"

log "=== ABLATION DONE ==="
