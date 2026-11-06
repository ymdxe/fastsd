#!/usr/bin/env bash
# Queue the three zjh_ab rounds from node1.
#
# node1 can reach both node2 (target) and node3 (edge), so it starts the
# target on node2, runs the edge on node3, stops the target so that
# scheduler_metrics.json is flushed, then moves to the next round.
#
# Usage (on node1, inside tmux):
#   ./scripts/zjh_ab/run_queue.sh
#
# Override SSH targets if needed:
#   NODE2=zhangh@node2.sanzo.top NODE3=zhangh@10.66.0.2 ./scripts/zjh_ab/run_queue.sh
set -euo pipefail

NODE2="${NODE2:-zhangh@node2.sanzo.top}"
NODE3="${NODE3:-zhangh@10.66.0.2}"
NODE2_REPO="${NODE2_REPO:-/home/hdd/zhangh/workspace/new_new_fastsd}"
NODE3_REPO="${NODE3_REPO:-/home/zhangh/workspace/new_new_fastsd}"
PORT="${PORT:-8003}"
TUNNEL_PORT="${TUNNEL_PORT:-18003}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"

log() { printf '%s %s\n' "$(date -Ins)" "$*"; }

# A single persistent tunnel is enough for all three rounds.
ensure_tunnel() {
    # The remote non-login shell may not have ss in PATH.  Test the TCP
    # listener directly with bash's /dev/tcp and only create a tunnel when
    # the port is actually free.  Never report success after ssh -f fails.
    if ssh "$NODE3" "timeout 2 bash -c 'exec 3<>/dev/tcp/127.0.0.1/${TUNNEL_PORT}' 2>/dev/null"; then
        echo 'tunnel already listening'
        return 0
    fi
    ssh "$NODE3" "ssh -f -N -o BatchMode=yes -L ${TUNNEL_PORT}:127.0.0.1:${PORT} ${NODE2}" || {
        echo 'failed to start tunnel' >&2
        return 1
    }
    echo 'tunnel started'
}

run_round() {
    local run="$1" case_name="$2" batch="$3" drafts="$4" budget="$5"
    local target_log="/tmp/ab_target_${run}.log"
    local edge_log="/tmp/ab_edge_${run}.log"
    local target_pid

    log "=== $run START batch=$batch drafts=$drafts budget=$budget ==="

    target_pid=$(ssh "$NODE2" "rm -f '$target_log'; cd '$NODE2_REPO'; \
        nohup env RUN_NAME='$run' BATCH_SIZE='$batch' NUM_DRAFTS='$drafts' \
        KV_RESIDENCY_BUDGET_MB='$budget' \
        ./scripts/zjh_ab/serve.sh '$case_name' >'$target_log' 2>&1 & echo \$!")

    if [ -z "$target_pid" ]; then
        log "FATAL: could not start target for $run"
        return 1
    fi
    log "$run target pid=$target_pid, waiting for checkpoint"

    local waited=0
    while true; do
        if ssh "$NODE2" "grep -q 'Loading checkpoint shards: 100%' '$target_log' 2>/dev/null"; then
            break
        fi
        if ! ssh "$NODE2" "kill -0 '$target_pid' 2>/dev/null"; then
            log "FATAL: target exited early for $run"
            ssh "$NODE2" "tail -n 80 '$target_log'" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$READY_TIMEOUT" ]; then
            log "FATAL: target not ready after ${READY_TIMEOUT}s for $run"
            return 1
        fi
    done
    log "$run target ready after ${waited}s"

    log "$run running edge"
    if ! ssh "$NODE3" "cd '$NODE3_REPO'; \
        RUN_NAME='$run' BATCH_SIZE='$batch' NUM_DRAFTS='$drafts' \
        ./scripts/zjh_ab/client.sh '$case_name' >'$edge_log' 2>&1"; then
        log "FATAL: edge failed for $run"
        ssh "$NODE3" "tail -n 120 '$edge_log'" || true
        ssh "$NODE2" "kill -INT '$target_pid' 2>/dev/null || true"
        return 1
    fi
    log "$run edge finished"

    log "$run draining target via /exit"
    ssh "$NODE2" "curl -s -X POST http://127.0.0.1:${PORT}/exit >/dev/null 2>&1 || true"

    # /exit makes the target worker return through its finally block and
    # write scheduler_metrics.json.  Give it time before stopping uvicorn.
    waited=0
    while ! ssh "$NODE2" "test -s '$NODE2_REPO/exp/zjh_ab/$run/target/scheduler_metrics.json'"; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge 60 ]; then
            log "WARN: scheduler_metrics.json not written after /exit"
            break
        fi
    done

    log "$run stopping target"
    ssh "$NODE2" "kill -INT '$target_pid' 2>/dev/null || true"

    waited=0
    while ssh "$NODE2" "kill -0 '$target_pid' 2>/dev/null"; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$STOP_TIMEOUT" ]; then
            log "WARN: target did not stop in ${STOP_TIMEOUT}s, sending KILL"
            ssh "$NODE2" "kill -KILL '$target_pid' 2>/dev/null || true"
            break
        fi
    done

    if ! ssh "$NODE2" "test -s '$NODE2_REPO/exp/zjh_ab/$run/target/scheduler_metrics.json'"; then
        log "FATAL: scheduler_metrics.json missing for $run"
        ssh "$NODE2" "tail -n 100 '$target_log'" || true
        return 1
    fi

    log "=== $run DONE ==="
}

ensure_tunnel

run_round sd-cpu           sd-cpu 8 4 0
run_round fastsd           fastsd 8 4 0
run_round pressure-fastsd  fastsd 4 8 256

log "=== ALL THREE ROUNDS DONE ==="
log "target metrics: $NODE2:$NODE2_REPO/exp/zjh_ab/*/target/scheduler_metrics.json"
log "edge metrics:   $NODE3:$NODE3_REPO/exp/zjh_ab/*/edge/"
