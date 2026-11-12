#!/usr/bin/env bash
# Dual-edge FastSD/prefetch experiment runner.
#
# Run this on node1 CPU after copying the repository there:
#   cd /home/zhangh/workspace/new_new_fastsd
#   bash scripts/zjh_ab/runall.sh
#
# The runner runs on node1 CPU and syncs runtime files to node2.
# A persistent supervisor owns each target, edge and tunnel process tree.
# A dropped control connection therefore does not kill the remote jobs.  The
# tunnel guard uses ordinary ssh keepalives and reconnects without autossh.
# Existing result directories are never removed; every invocation gets a new
# timestamped prefix unless RUN_TAG is supplied.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_HOST="${GPU_HOST:?set GPU_HOST, e.g. zhangh@node1.sanzo.top}"
N2_HOST="${N2_HOST:?set N2_HOST, e.g. zhangh@node2.sanzo.top}"
GPU_REPO="${GPU_REPO:?set GPU_REPO, e.g. /home/zhangh/new_new_fastsd}"
N2_REPO="${N2_REPO:?set N2_REPO, e.g. /home/hdd/zhangh/workspace/new_new_fastsd}"
N2_PY="${N2_PY:-/home/hdd/zhangh/envs/new_fastsd/bin/python}"
N1_PY="${N1_PY:-/home/zhangh/venvs/fastsd_cpu/bin/python}"
LOCAL_PY="${LOCAL_PY:-/home/zhangh/venvs/fastsd_cpu/bin/python}"
GPU_PY="${GPU_PY:-/home/hdd/zhangh/envs/new_fastsd/bin/python}"
TARGET_GPU="${TARGET_GPU:?set TARGET_GPU explicitly}"
TARGET_ON_N2="${TARGET_ON_N2:?set TARGET_ON_N2=0 or 1}"
TARGET_MIN_FREE_MB="${TARGET_MIN_FREE_MB:-20000}"
COLLECT_RETRIES="${COLLECT_RETRIES:-5}"
COLLECT_RETRY_DELAY="${COLLECT_RETRY_DELAY:-2}"
PORT="${PORT:-8003}"
TUNNEL_PORT="${TUNNEL_PORT:-18003}"
N2_TUNNEL_PORT="${N2_TUNNEL_PORT:-18003}"
PORT_BASE="${PORT_BASE:-40003}"
TUNNEL_BASE="${TUNNEL_BASE:-40004}"
NFS_ROOT="${NFS_ROOT:-/mnt/nfs0/zhangh/fastsd_exp}"
GPU_EXP_ROOT="${GPU_EXP_ROOT:-/dev/shm/fastsd_exp/target}"
N2_EXP_ROOT="${N2_EXP_ROOT:-/dev/shm/fastsd_exp/node2}"
N1_EXP_ROOT="${N1_EXP_ROOT:-$NFS_ROOT/node1}"
N2_DRAFT_MODEL="${N2_DRAFT_MODEL:-/mnt/nfs0/zhangh/models/Qwen3-0.6B}"
N1_DRAFT_MODEL="${N1_DRAFT_MODEL:-/mnt/nfs0/zhangh/models/Qwen3-0.6B}"
N2_DATA_PATH="${N2_DATA_PATH:-data_dual_node2}"
N1_DATA_PATH="${N1_DATA_PATH:-data_dual_node1}"
REPORT_ROOT="${REPORT_ROOT:-$NFS_ROOT/reports}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-14400}"
STOP_TIMEOUT="${STOP_TIMEOUT:-180}"
TUNNEL_TIMEOUT="${TUNNEL_TIMEOUT:-45}"
MANUAL_TUNNEL="${MANUAL_TUNNEL:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SYNC_ONLY="${SYNC_ONLY:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"
# Optional stress knob: set TOKEN_BUDGET_OVERRIDE=512 to run every round with
# a 512-token target batch budget.  Defaults preserve the staged 32/128/256
# budgets used by the gamma=8/16/32 matrix below.
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE:-}"
VERIFY_METHOD="${VERIFY_METHOD:-greedy}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-64}"
if (( TOTAL_REQUESTS < 2 || TOTAL_REQUESTS % 2 != 0 )); then
  printf '%s
' "TOTAL_REQUESTS must be a positive even number" >&2
  exit 2
fi
REQUESTS_PER_EDGE=$((TOTAL_REQUESTS / 2))
WAIT_LAMBDA="${WAIT_LAMBDA:-0.01}"
WAIT_LINEAR_UNTIL_S="${WAIT_LINEAR_UNTIL_S:-0}"
WAIT_LINEAR_ALPHA="${WAIT_LINEAR_ALPHA:-}"


SSH_OPTS=(
  -o BatchMode=yes
  -o TCPKeepAlive=yes
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
  -o ConnectTimeout=20
)

log() { printf '%s %s\n' "$(date -Ins)" "$*" >&2; }

retry() {
  local attempts="${COLLECT_RETRIES:-5}" delay="${COLLECT_RETRY_DELAY:-2}" n=1
  until "$@"; do
    if (( n >= attempts )); then
      log "retry exhausted (${attempts}): $*"
      return 1
    fi
    log "retry ${n}/${attempts} in ${delay}s: $*"
    sleep "$delay"
    n=$((n + 1))
    delay=$((delay * 2))
  done
}
[[ "$TARGET_ON_N2" == 0 || "$TARGET_ON_N2" == 1 ]] || { log "TARGET_ON_N2 must be 0 or 1"; exit 2; }
[[ "$TARGET_GPU" =~ ^[0-9]+$ ]] || { log "TARGET_GPU must be numeric"; exit 2; }
budget_for() {
  local default="$1"
  printf '%s\n' "${TOKEN_BUDGET_OVERRIDE:-$default}"
}
ssh_gpu() { ssh "${SSH_OPTS[@]}" "$GPU_HOST" "$@"; }
ssh_n2() { ssh "${SSH_OPTS[@]}" "$N2_HOST" "$@"; }

sync_host() {
  local host="$1" dest="$2"
  log "sync runtime files -> $host:$dest"
  rsync -az --exclude='params.env' --exclude='__pycache__/' --exclude='*.pyc' \
    "$ROOT/src/" "$host:$dest/src/"
  rsync -az --exclude='params.env' --exclude='__pycache__/' --exclude='*.pyc' \
    "$ROOT/edge/" "$host:$dest/edge/"
  rsync -az --exclude='params.env' --exclude='__pycache__/' --exclude='*.pyc' \
    "$ROOT/cloud/" "$host:$dest/cloud/"
  rsync -az "$ROOT/diagnostic_trace.py" "$host:$dest/diagnostic_trace.py"
  rsync -az --exclude='params.env' --exclude='__pycache__/' --exclude='*.pyc' \
    "$ROOT/scripts/zjh_ab/" "$host:$dest/scripts/zjh_ab/"
}

sync_all() {
  [[ "$SKIP_SYNC" == 1 ]] && { log "SKIP_SYNC=1; no remote sync"; return; }
  sync_host "$GPU_HOST" "$GPU_REPO"
  if [[ "$GPU_HOST:$GPU_REPO" != "$N2_HOST:$N2_REPO" ]]; then
    sync_host "$N2_HOST" "$N2_REPO"
  fi
}

remote_command() {
  local host="$1" command
  shift
  printf -v command '%q ' "$@"
  ssh "${SSH_OPTS[@]}" "$host" "$command"
}

# Record the job BEFORE attempting SSH. A lost SSH reply cannot hide a job
# that already started on the remote machine.
launch_job() {
  local host="$1" py="$2" repo="$3" state="$4" logfile="$5"
  shift 5
  local helper="$repo/scripts/zjh_ab/managed_job.py"
  printf '%s\t%s\t%s\t%s\n' "$host" "$py" "$helper" "$state" >>"$STATE_DIR/jobs.tsv"
  if [[ "$host" == local ]]; then
    "$py" "$helper" start --state "$state" --log "$logfile" --cwd "$repo" -- "$@"
  else
    remote_command "$host" "$py" "$helper" start --state "$state" --log "$logfile" --cwd "$repo" -- "$@"
  fi
}

job_status() {
  local host="$1" py="$2" repo="$3" state="$4"
  if [[ "$host" == local ]]; then
    "$py" "$repo/scripts/zjh_ab/managed_job.py" status "$state"
  else
    remote_command "$host" "$py" "$repo/scripts/zjh_ab/managed_job.py" status "$state"
  fi
}

cleanup_jobs() {
  bash "$ROOT/scripts/zjh_ab/stop_runall.sh" --jobs-only "$STATE_DIR"
}

runner_exit() {
  local code=$?
  trap - EXIT INT TERM
  if ! cleanup_jobs; then
    log "cleanup incomplete; retry: bash scripts/zjh_ab/stop_runall.sh $RUN_TAG"
    code=1
  fi
  printf '%s\n' "$code" >"$STATE_DIR/runner.exit"
  exit "$code"
}

check_target_gpu() {
  local free
  free=$(ssh_gpu "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i '$TARGET_GPU'" 2>/dev/null | head -n1 | tr -d '[:space:]')
  if [[ ! "$free" =~ ^[0-9]+$ ]]; then
    log "cannot read free memory for $GPU_HOST GPU $TARGET_GPU"
    return 1
  fi
  if (( free < TARGET_MIN_FREE_MB )); then
    log "GPU $TARGET_GPU has only ${free}MiB free; need at least ${TARGET_MIN_FREE_MB}MiB"
    return 1
  fi
  log "GPU $TARGET_GPU free=${free}MiB"
}

check_output_dirs() {
  ssh_gpu "mkdir -p '$GPU_EXP_ROOT' && touch '$GPU_EXP_ROOT/.write_test' && rm -f '$GPU_EXP_ROOT/.write_test'"     || { log "GPU output root not writable: $GPU_EXP_ROOT"; return 1; }
  mkdir -p "$N1_EXP_ROOT" && touch "$N1_EXP_ROOT/.write_test" && rm -f "$N1_EXP_ROOT/.write_test"     || { log "node1 output root not writable: $N1_EXP_ROOT"; return 1; }
  ssh_n2 "mkdir -p '$N2_EXP_ROOT' && touch '$N2_EXP_ROOT/.write_test' && rm -f '$N2_EXP_ROOT/.write_test'"     || { log "node2 output root not writable: $N2_EXP_ROOT"; return 1; }
}

ensure_target_port_free() {
  local attempt
  for attempt in 1 2 3; do
    if remote_command "$GPU_HOST" "$GPU_PY" "$GPU_REPO/scripts/zjh_ab/managed_job.py" check-port "$PORT" >/dev/null 2>&1; then
      return 0
    fi
    log "target port $PORT busy (attempt $attempt); inspecting owner"
    local owner
    owner=$(ssh_gpu "ss -ltnp 2>/dev/null | grep ':$PORT ' | head -n1") || true
    log "port owner: ${owner:-unknown}"
    local pid
    pid=$(printf '%s\n' "$owner" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1)
    if [[ -n "$pid" ]]; then
      local cmd
      cmd=$(ssh_gpu "ps -p '$pid' -o cmd= 2>/dev/null") || true
      log "port pid=$pid cmd=$cmd"
      if [[ "$cmd" == *cloud_service.py* ]]; then
        log "killing stale cloud_service pid=$pid on $GPU_HOST"
        ssh_gpu "kill -TERM '$pid' 2>/dev/null || true"
        sleep 3
        ssh_gpu "kill -KILL '$pid' 2>/dev/null || true" || true
      else
        log "FATAL: port $PORT is owned by non-target process; refusing to kill"
        return 1
      fi
    else
      log "FATAL: cannot identify owner of port $PORT"
      return 1
    fi
  done
  log "FATAL: target port $PORT still busy after cleanup"
  return 1
}

start_target() {
  local run="$1" gamma="$2" budget="$3" case_name="$4" rate="$5" idle="$6" online="$7"
  TARGET_STATE="$GPU_EXP_ROOT/$run/target.job.json"
  TARGET_LOG="$GPU_EXP_ROOT/logs/${run}.target.log"
  launch_job "$GPU_HOST" "$GPU_PY" "$GPU_REPO" "$TARGET_STATE" "$TARGET_LOG" env \
    NODE2_PY="$GPU_PY" TARGET_GPU="$TARGET_GPU" \
    TARGET_MODEL_NODE2=/mnt/nfs0/zhangh/models/Qwen3-8B \
    EXP_ROOT="$GPU_EXP_ROOT" RUN_NAME="$run" \
    NUM_DRAFTS=32 BATCH_SIZE=8 GAMMA="$gamma" TOKEN_BUDGET="$budget" MAX_TOKENS=400 SEED=1234 \
    VERIFY_METHOD="$VERIFY_METHOD" FASTSD_WAIT_LAMBDA="$WAIT_LAMBDA" \
    FASTSD_WAIT_LINEAR_UNTIL_S="$WAIT_LINEAR_UNTIL_S" FASTSD_WAIT_LINEAR_ALPHA="$WAIT_LINEAR_ALPHA" \
    KV_PREFETCH_LOOKAHEAD_ROUNDS=3 KV_PREFETCH_HORIZON_LIMIT=1 KV_PREFETCH_H2_MAX_PIDS=0 \
    KV_RESIDENCY_BUDGET_MB=2048 KV_PREFETCH_MB=256 \
    KV_PREFETCH_BEFORE_SUBMIT="${KV_PREFETCH_BEFORE_SUBMIT:-0}" \
    KV_PREFETCH_RATE_MB_S="$rate" KV_PREFETCH_IDLE_RATE_MB_S="$idle" \
    KV_PREFETCH_ONLINE_FRACTION="$online" KV_PREFETCH_CHUNK_KB=4096 \
    KV_PREFETCH_SPECULATIVE_CHUNK_KB="${SPEC_CHUNK_KB:-256}" \
    MEASURE_ENERGY="${MEASURE_ENERGY:-0}" \
    CLOUD_SERVICE_HOST=127.0.0.1 CLOUD_SERVICE_PORT="$PORT" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    bash scripts/zjh_ab/serve.sh "$case_name"
  local deadline=$((SECONDS + READY_TIMEOUT)) status
  while (( SECONDS < deadline )); do
    status=$(job_status "$GPU_HOST" "$GPU_PY" "$GPU_REPO" "$TARGET_STATE") || return 1
    if [[ "$status" != running ]]; then
      log "target stopped before ready: $status"
      remote_command "$GPU_HOST" tail -n 80 "$TARGET_LOG" || true
      return 1
    fi
    if remote_command "$GPU_HOST" "$GPU_PY" "$GPU_REPO/scripts/zjh_ab/managed_job.py" owns-port "$TARGET_STATE" "$PORT" &&
       ssh_gpu "curl --noproxy '*' -sf -m 3 http://127.0.0.1:${PORT}/health" >/dev/null; then
      return 0
    fi
    sleep 3
  done
  log "target readiness timeout"
  remote_command "$GPU_HOST" tail -n 80 "$TARGET_LOG" || true
  return 1
}

start_guard() {
  local host="$1" py="$2" repo="$3" state="$4" logfile="$5" port="$6"
  launch_job "$host" "$py" "$repo" "$state" "$logfile" \
    bash scripts/zjh_ab/ssh_tunnel_guard.sh "$GPU_HOST" "$port" "$PORT" 2
}

start_edge() {
  local host="$1" py="$2" repo="$3" exp="$4" model="$5" data="$6" url="$7" seed="$8"
  shift 8
  local run="$1" gamma="$2" budget="$3" case_name="$4"
  launch_job "$host" "$py" "$repo" "$exp/$run/edge.job.json" "$exp/logs/${run}.edge.log" env \
    NODE3_PY="$py" MODELS=/mnt/nfs0/zhangh/models \
    TARGET_MODEL=/mnt/nfs0/zhangh/models/Qwen3-8B DRAFT_MODEL="$model" \
    EXP_ROOT="$exp" RUN_NAME="$run" DATASET=humaneval DATA_PATH="$data" REQUESTS="$REQUESTS_PER_EDGE" NUM_DRAFTS=16 EDGE_THREADS=4 \
    BATCH_SIZE=8 GAMMA="$gamma" TOKEN_BUDGET="$budget" MAX_TOKENS=400 SEED=1234 \
    VERIFY_METHOD="$VERIFY_METHOD" \
    ARRIVAL_DISTRIBUTION=poisson ARRIVAL_RATE=1.6 ARRIVAL_SEED="$seed" \
    TARGET_URL="$url" RUN_TIMEOUT_S="$ROUND_TIMEOUT" HEALTH_TIMEOUT_S=900 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    bash scripts/zjh_ab/client.sh "$case_name"
}

wait_edges() {
  local run="$1" deadline=$((SECONDS + ROUND_TIMEOUT)) a b target
  while (( SECONDS < deadline )); do
    a=$(job_status local "$N1_PY" "$ROOT" "$N1_EXP_ROOT/$run/edge.job.json") || return 1
    b=$(job_status "$N2_HOST" "$N2_PY" "$N2_REPO" "$N2_EXP_ROOT/$run/edge.job.json") || return 1
    for status in "$a" "$b"; do
      if [[ "$status" != running && "$status" != 0 ]]; then
        log "edge failed: node1=$a node2=$b; stopping round"
        return 1
      fi
    done
    [[ "$a" == 0 && "$b" == 0 ]] && return 0
    target=$(job_status "$GPU_HOST" "$GPU_PY" "$GPU_REPO" "$TARGET_STATE") || return 1
    [[ "$target" == running ]] || { log "target exited during edge work: $target"; return 1; }
    sleep 3
  done
  log "edge round timed out"
  return 1
}

force_cleanup_target() {
  log "forcing cleanup of target on $GPU_HOST"
  ssh_gpu "pkill -f '[c]loud_service.py' || true" || true
  ssh_gpu "pkill -f 'managed_job.py supervise .*target.job.json' || true" || true
  sleep 3
  ssh_gpu "pkill -9 -f '[c]loud_service.py' || true" || true
  ssh_gpu "pkill -9 -f 'managed_job.py supervise .*target.job.json' || true" || true
}

stop_round() {
  # Only send /exit when the listener belongs to this registered target.
  if remote_command "$GPU_HOST" "$GPU_PY" "$GPU_REPO/scripts/zjh_ab/managed_job.py" owns-port "$TARGET_STATE" "$PORT"; then
    ssh_gpu "curl --noproxy '*' -sf -m 5 -X POST http://127.0.0.1:${PORT}/exit" >/dev/null || true
    local deadline=$((SECONDS + STOP_TIMEOUT))
    while (( SECONDS < deadline )); do
      remote_command "$GPU_HOST" test -s "$GPU_EXP_ROOT/$CURRENT_RUN/target/scheduler_metrics.json" && break
      sleep 2
    done
  fi
  cleanup_jobs
  if ! remote_command "$GPU_HOST" "$GPU_PY" "$GPU_REPO/scripts/zjh_ab/managed_job.py" check-port "$PORT" >/dev/null 2>&1; then
    log "WARN: target port $PORT still busy after stop; forcing stale target cleanup"
    force_cleanup_target
  fi
  if ! remote_command "$GPU_HOST" "$GPU_PY" "$GPU_REPO/scripts/zjh_ab/managed_job.py" check-port "$PORT" >/dev/null 2>&1; then
    log "FATAL: target port $PORT is still busy after forced cleanup"
    return 1
  fi
}

collect_round() {
  local run="$1" report="$2"
  mkdir -p "$report/$run/target" "$report/$run/edge" "$report/$run/raw/node2" "$report/$run/raw/node1"
  retry scp "${SSH_OPTS[@]}" "$GPU_HOST:$GPU_EXP_ROOT/$run/target/scheduler_metrics.json" "$report/$run/target/" || return 1
  # Energy is optional; preserve it whenever MEASURE_ENERGY was enabled.
  retry scp "${SSH_OPTS[@]}" "$GPU_HOST:$GPU_EXP_ROOT/$run/target/energy_metrics.json" "$report/$run/target/" || true
  retry rsync -az "$N2_HOST:$N2_EXP_ROOT/$run/edge/" "$report/$run/raw/node2/" || return 1
  retry rsync -az "$N1_EXP_ROOT/$run/edge/" "$report/$run/raw/node1/" || return 1
  retry cp "$N1_EXP_ROOT/logs/${run}.edge.log" "$report/$run/raw/node1/edge.log" || return 1
  retry scp "${SSH_OPTS[@]}" "$N2_HOST:$N2_EXP_ROOT/logs/${run}.edge.log" "$report/$run/raw/node2/edge.log" || return 1
  retry scp "${SSH_OPTS[@]}" "$GPU_HOST:$GPU_EXP_ROOT/logs/${run}.target.log" "$report/$run/target/target.log" || return 1
  # Keep per-host filenames distinct for the two edge summaries/process logs.
  local f base ext node
  for node in node1 node2; do
    for f in "$report/$run/raw/$node"/edge_metrics*; do
      [[ -f "$f" ]] || continue
      base="${f##*/}"
      ext="${base##*.}"
      cp "$f" "$report/$run/edge/${base%.*}_${node}.${ext}"
    done
  done
}

run_round() {
  local run="$1" gamma="$2" budget="$3" case_name="$4" rate="$5" idle="$6" online="$7" slot="${8:-0}"
  PORT=$((PORT_BASE + slot * 2))
  TUNNEL_PORT=$((TUNNEL_BASE + slot * 2))
  N2_TUNNEL_PORT="$TUNNEL_PORT"
  CURRENT_RUN="$run"
  log "=== $run gamma=$gamma budget=$budget case=$case_name rate=$rate idle=$idle online=$online port=$PORT tunnel=$TUNNEL_PORT ==="
  check_target_gpu || return 1
  ensure_target_port_free || return 1
  # Check BEFORE model allocation. A conflict belongs to someone else (or an
  # older runner); report it, never kill an arbitrary port owner.
  if [[ "$MANUAL_TUNNEL" != 1 ]]; then
    "$LOCAL_PY" "$ROOT/scripts/zjh_ab/managed_job.py" check-port "$TUNNEL_PORT"
    remote_command "$GPU_HOST" "$GPU_PY" "$GPU_REPO/scripts/zjh_ab/managed_job.py" check-port "$PORT"
    if [[ "$TARGET_ON_N2" != 1 ]]; then
      remote_command "$N2_HOST" "$N2_PY" "$N2_REPO/scripts/zjh_ab/managed_job.py" check-port "$N2_TUNNEL_PORT"
    fi
    start_guard local "$N1_PY" "$ROOT" "$N1_EXP_ROOT/$run/tunnel.job.json" "$N1_EXP_ROOT/logs/${run}.tunnel.log" "$TUNNEL_PORT"
  fi
  local n2_url="http://127.0.0.1:$PORT"
  if [[ "$TARGET_ON_N2" != 1 ]]; then
    if [[ "$MANUAL_TUNNEL" != 1 ]]; then
      start_guard "$N2_HOST" "$N2_PY" "$N2_REPO" "$N2_EXP_ROOT/$run/tunnel.job.json" "$N2_EXP_ROOT/logs/${run}.tunnel.log" "$N2_TUNNEL_PORT"
    fi
    n2_url="http://127.0.0.1:$N2_TUNNEL_PORT"
  fi
  start_target "$run" "$gamma" "$budget" "$case_name" "$rate" "$idle" "$online"
  local deadline=$((SECONDS + TUNNEL_TIMEOUT))
  until curl --noproxy '*' -sf -m 3 "http://127.0.0.1:$TUNNEL_PORT/health" >/dev/null &&
        remote_command "$N2_HOST" curl --noproxy '*' -sf -m 3 "$n2_url/health" >/dev/null; do
    if (( SECONDS >= deadline )); then
      log "tunnel readiness failed; cleaning this round"
      tail -n 60 "$N1_EXP_ROOT/logs/${run}.tunnel.log" || true
      return 1
    fi
    sleep 2
  done
  local edge_case="$case_name"
  start_edge "$N2_HOST" "$N2_PY" "$N2_REPO" "$N2_EXP_ROOT" "$N2_DRAFT_MODEL" "$N2_DATA_PATH" "$n2_url" 5678 "$run" "$gamma" "$budget" "$edge_case"
  start_edge local "$N1_PY" "$ROOT" "$N1_EXP_ROOT" "$N1_DRAFT_MODEL" "$N1_DATA_PATH" "http://127.0.0.1:$TUNNEL_PORT" 1234 "$run" "$gamma" "$budget" "$edge_case"

  local edge_ok=1
  if ! wait_edges "$run"; then
    log "WARN: edge failed for $run; stopping target and collecting partial metrics"
    edge_ok=0
  fi

  stop_round
  collect_round "$run" "$REPORT_ROOT/$RUN_TAG" || log "WARN: collect failed for $run"
  if [[ "$edge_ok" == 0 ]]; then
    log "=== $run DONE WITH EDGE FAILURE ==="
    return 1
  fi
  log "=== $run collected; request completion requires acceptance report ==="
}

cd "$ROOT"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ && "$RUN_TAG" != . && "$RUN_TAG" != .. ]] || { log "invalid RUN_TAG"; exit 2; }
RUNALL_STATE_ROOT="${RUNALL_STATE_ROOT:-$ROOT/runtime/runall}"
mkdir -p "$RUNALL_STATE_ROOT"
exec 9>"$RUNALL_STATE_ROOT/controller.lock"
flock -n 9 || { log "another runall is active; stop it with scripts/zjh_ab/stop_runall.sh"; exit 1; }
STATE_DIR="$RUNALL_STATE_ROOT/$RUN_TAG"
mkdir "$STATE_DIR" || { log "RUN_TAG already exists; choose a new tag"; exit 1; }
printf 'LOCAL_PY=%q\n' "$LOCAL_PY" >"$STATE_DIR/config.env"
"$LOCAL_PY" "$ROOT/scripts/zjh_ab/managed_job.py" register "$STATE_DIR/controller.json" --pid "$$"
printf '%s\n' "$RUN_TAG" >"$RUNALL_STATE_ROOT/latest"
trap runner_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ -x "$N1_PY" ]] || { log "node1 Python missing: N1_PY=$N1_PY"; exit 1; }
[[ -f "$N1_DATA_PATH/humaneval.jsonl" ]] || { log "missing node1 dataset: $N1_DATA_PATH/humaneval.jsonl"; exit 1; }
ssh_n2 "test -x '$N2_PY' && test -f '$N2_REPO/$N2_DATA_PATH/humaneval.jsonl'" || { log "node2 SSH/Python/dataset preflight failed"; exit 1; }
log "topology: node1 CPU + node2 CPU; target=$GPU_HOST GPU=$TARGET_GPU; verify=$VERIFY_METHOD"
log "stop command: bash scripts/zjh_ab/stop_runall.sh $RUN_TAG"
sync_all
check_output_dirs || exit 1
[[ "$SYNC_ONLY" == 1 ]] && { log "SYNC_ONLY=1; sync complete"; exit 0; }

REPORT="$REPORT_ROOT/$RUN_TAG"
mkdir -p "$REPORT"

# The first six runs isolate prefetch's net effect at increasing target work.
# The seventh compares software-rate throttling at gamma=16.  The final run
# is standard server/edge speculative decoding with the same KV pressure.
if [[ "${DIAGNOSTIC_ONLY:-0}" == "1" ]]; then
  # Focused attribution matrix: gamma=8, token budget=32, 2 GiB residency,
  # unlimited prefetch rate and one fixed arrival schedule. Each ablation
  # changes one FastSD mechanism so scheduler and edge metrics are comparable.
  DIAG_BUDGET="${DIAG_BUDGET:-$(budget_for 32)}"
  DIAG_GAMMA="${DIAG_GAMMA:-8}"
  SPEC_CHUNK_KB=4096
  run_round "${RUN_TAG}_sd_offload" "$DIAG_GAMMA" "$DIAG_BUDGET" sd-cpu 0 0 0.0 0 || log "WARN: diagnostic sd failed"
  run_round "${RUN_TAG}_fastsd_noprefetch" "$DIAG_GAMMA" "$DIAG_BUDGET" fastsd-noprefetch 0 0 1.0 1 || log "WARN: diagnostic no-prefetch failed"
  run_round "${RUN_TAG}_fastsd" "$DIAG_GAMMA" "$DIAG_BUDGET" fastsd 0 0 1.0 2 || log "WARN: diagnostic fastsd failed"
  run_round "${RUN_TAG}_fastsd_nopipeline" "$DIAG_GAMMA" "$DIAG_BUDGET" fastsd-nopipeline 0 0 1.0 3 || log "WARN: diagnostic no-pipeline failed"
  run_round "${RUN_TAG}_fastsd_noproactive" "$DIAG_GAMMA" "$DIAG_BUDGET" fastsd-noproactive 0 0 1.0 4 || log "WARN: diagnostic no-proactive failed"
  run_round "${RUN_TAG}_fastsd_nolatency" "$DIAG_GAMMA" "$DIAG_BUDGET" fastsd-nolatency 0 0 1.0 5 || log "WARN: diagnostic no-latency failed"
  conditions=("$REPORT"/*)
  "$LOCAL_PY" "$ROOT/scripts/zjh_ab/accept.py" "${conditions[@]}" \
    --expected-requests "$TOTAL_REQUESTS" --json-output "$REPORT/report.json" \
    2>&1 | tee "$REPORT/report.txt" || true
  log "DIAGNOSTIC MATRIX DONE: $REPORT"
  exit 0
fi
B32="$(budget_for 32)"
B128="$(budget_for 128)"
B256="$(budget_for 256)"
run_round "${RUN_TAG}_g8_b${B32}_off" 8 "$B32" fastsd-noprefetch 256 1024 0.25 0 || log "WARN: round failed"
run_round "${RUN_TAG}_g8_b${B32}_on" 8 "$B32" fastsd 256 1024 1.0 1 || log "WARN: round failed"
run_round "${RUN_TAG}_g16_b${B128}_off" 16 "$B128" fastsd-noprefetch 256 1024 0.25 2 || log "WARN: round failed"
run_round "${RUN_TAG}_g16_b${B128}_on" 16 "$B128" fastsd 256 1024 1.0 3 || log "WARN: round failed"
run_round "${RUN_TAG}_g32_b${B256}_off" 32 "$B256" fastsd-noprefetch 256 1024 0.25 4 || log "WARN: round failed"
run_round "${RUN_TAG}_g32_b${B256}_on" 32 "$B256" fastsd 256 1024 1.0 5 || log "WARN: round failed"
SPEC_CHUNK_KB=4096 run_round "${RUN_TAG}_g16_b${B128}_on_uncapped" 16 "$B128" fastsd 0 0 1.0 6 || log "WARN: round failed"
run_round "${RUN_TAG}_sd_offload_g8_b${B32}" 8 "$B32" sd-cpu 256 1024 0.0 7 || log "WARN: round failed"

conditions=("$REPORT"/*)
"$LOCAL_PY" "$ROOT/scripts/zjh_ab/accept.py" "${conditions[@]}" \
  --expected-requests "$TOTAL_REQUESTS" --json-output "$REPORT/report.json" \
  2>&1 | tee "$REPORT/report.txt" || true

if [[ -d "$ROOT/baselines/specedge/official" ]] && [[ -n "$(find "$ROOT/baselines/specedge/official" -type f -print -quit)" ]] && [[ -n "${SPECEDGE_CONFIG:-}" ]]; then
  log "SpecEdge prerequisites detected; run its prepared suite separately with SPECEDGE_CONFIG=$SPECEDGE_CONFIG"
else
  log "SpecEdge skipped: official submodule/runtime/config is not ready for Qwen3-8B on this topology"
fi

log "ALL DONE: $REPORT"
