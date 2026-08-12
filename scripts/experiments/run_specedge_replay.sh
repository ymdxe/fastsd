#!/usr/bin/env bash
# Replay one shared trace through two one-GPU Official SpecEdge client processes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  EDGE_PHYSICAL_GPUS=0,1 bash scripts/experiments/run_specedge_replay.sh \
      --config configs/experiments/mtbench_poisson.yaml \
      --trace /home/hdd/.../arrival_trace.jsonl \
      --grpc-address 10.66.0.5:18000 --run-id ID --run-dir /home/hdd/.../edge

The default factory is the checked-in bridge to pinned Official SpecEdge.  It
starts one Python process per physical edge GPU, waits until both have loaded
and completed Official Sync, then publishes one future node1 monotonic start
timestamp.  Each client therefore replays its own trace client_id partition
against the same global Poisson origin.

Options:
  --cloud-host HOST             SSH alias hosting the SpecEdge server (default: node2)
  --cloud-run-dir PATH          Cloud component directory (default: sibling cloud/)
  --client-factory MODULE:NAME  Explicit alternative reviewed factory
  --dry-run                     Metadata/timing smoke; sends no gRPC requests
EOF
}

CONFIG=""
TRACE=""
GRPC_ADDRESS=""
RUN_ID=""
RUN_DIR=""
EDGE_GPUS="${EDGE_PHYSICAL_GPUS:-}"
PYTHON_BIN="${SPECEDGE_PYTHON:-/home/hdd/zhangh/envs/specedge/bin/python}"
DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"
CLIENT_FACTORY="${SPECEDGE_CLIENT_FACTORY:-}"
DRY_RUN=false
MAX_REQUESTS=""
REQUEST_TIMEOUT_S=""
GPU_SAMPLE_INTERVAL_MS="${FASTSD_GPU_SAMPLE_INTERVAL_MS:-50}"
CLOUD_HOST="${SPECEDGE_CLOUD_HOST:-node2}"
CLOUD_RUN_DIR="${SPECEDGE_CLOUD_RUN_DIR:-}"
REMOTE_REPO_ROOT="${SPECEDGE_REMOTE_REPO_ROOT:-/home/hdd/zhangh/workspace/fastsd}"
SSH_BIN="${FASTSD_SSH_BIN:-ssh}"
SCP_BIN="${FASTSD_SCP_BIN:-scp}"
START_BARRIER_TIMEOUT_S="${SPECEDGE_START_BARRIER_TIMEOUT_S:-900}"
SYNC_TIMEOUT_S="${SPECEDGE_SYNC_TIMEOUT_S:-600}"
DRAIN_TIMEOUT_S="${SPECEDGE_DRAIN_TIMEOUT_S:-600}"
CLOUD_SHUTDOWN_TIMEOUT_S="${SPECEDGE_CLOUD_SHUTDOWN_TIMEOUT_S:-120}"
CROP_TOOL="${SCRIPT_DIR}/crop_gpu_samples.py"
CLOUD_MARKER_TOOL="${REPO_ROOT}/scripts/experiments/write_control_marker.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2 ;;
    --trace) TRACE="${2:-}"; shift 2 ;;
    --grpc-address) GRPC_ADDRESS="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --edge-physical-gpus) EDGE_GPUS="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --draft-model) DRAFT_MODEL="${2:-}"; shift 2 ;;
    --client-factory) CLIENT_FACTORY="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --max-requests) MAX_REQUESTS="${2:-}"; shift 2 ;;
    --request-timeout-s) REQUEST_TIMEOUT_S="${2:-}"; shift 2 ;;
    --cloud-host) CLOUD_HOST="${2:-}"; shift 2 ;;
    --cloud-run-dir) CLOUD_RUN_DIR="${2:-}"; shift 2 ;;
    --gpu-sample-interval-ms) GPU_SAMPLE_INTERVAL_MS="${2:-}"; shift 2 ;;
    --drain-timeout-s) DRAIN_TIMEOUT_S="${2:-}"; shift 2 ;;
    --cloud-shutdown-timeout-s) CLOUD_SHUTDOWN_TIMEOUT_S="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${CONFIG}" && -n "${TRACE}" && -n "${GRPC_ADDRESS}" && -n "${RUN_ID}" && -n "${RUN_DIR}" && -n "${EDGE_GPUS}" ]] || { usage >&2; die "Missing required arguments."; }
validate_run_id "${RUN_ID}"
validate_hdd_path "${RUN_DIR}"
validate_hdd_path "${TRACE}"
assert_fresh_component_run_path "${RUN_DIR}" edge
validate_gpu_list "${EDGE_GPUS}" 2
[[ "${GRPC_ADDRESS}" == "10.66.0.5:18000" ]] || die "SpecEdge replay must use the IB endpoint 10.66.0.5:18000."
validate_host_alias "${CLOUD_HOST}"
[[ "${START_BARRIER_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "SPECEDGE_START_BARRIER_TIMEOUT_S must be positive."
[[ "${SYNC_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "SPECEDGE_SYNC_TIMEOUT_S must be positive."
[[ "${DRAIN_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--drain-timeout-s must be positive."
[[ "${CLOUD_SHUTDOWN_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || die "SPECEDGE_CLOUD_SHUTDOWN_TIMEOUT_S must be a positive integer number of seconds."
awk -v value="${START_BARRIER_TIMEOUT_S}" 'BEGIN { exit !(value > 0) }' || die "SPECEDGE_START_BARRIER_TIMEOUT_S must be positive."
awk -v value="${SYNC_TIMEOUT_S}" 'BEGIN { exit !(value > 0) }' || die "SPECEDGE_SYNC_TIMEOUT_S must be positive."
awk -v value="${DRAIN_TIMEOUT_S}" 'BEGIN { exit !(value > 0) }' || die "--drain-timeout-s must be positive."
if [[ "${DRY_RUN}" == true && -n "${CLIENT_FACTORY}" ]]; then
  die "Choose exactly one of --dry-run or --client-factory/SPECEDGE_CLIENT_FACTORY."
fi
if [[ "${DRY_RUN}" == false && -z "${CLIENT_FACTORY}" ]]; then
  CLIENT_FACTORY="baselines.specedge.adapter.official_client_factory:create_client"
fi
if [[ "${DRY_RUN}" == false ]]; then
  [[ "${CLIENT_FACTORY}" =~ ^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$ ]] || die "A real run requires a module:callable factory."
fi
if [[ -n "${MAX_REQUESTS}" ]]; then
  [[ "${MAX_REQUESTS}" =~ ^[1-9][0-9]*$ ]] || die "--max-requests must be a positive integer."
fi
if [[ -n "${REQUEST_TIMEOUT_S}" ]]; then
  [[ "${REQUEST_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--request-timeout-s must be a positive number."
fi
if [[ "${CONFIG}" != /* ]]; then CONFIG="${REPO_ROOT}/${CONFIG}"; fi
if [[ "${DRAFT_MODEL}" != /* ]]; then DRAFT_MODEL="${REPO_ROOT}/${DRAFT_MODEL}"; fi
if [[ -z "${CLOUD_RUN_DIR}" ]]; then CLOUD_RUN_DIR="$(dirname "${RUN_DIR}")/cloud"; fi
validate_hdd_path "${CLOUD_RUN_DIR}"
validate_hdd_path "${REMOTE_REPO_ROOT}"
require_file "${CONFIG}"
require_file "${TRACE}"
require_executable_or_command "${PYTHON_BIN}"
require_executable_or_command "${SSH_BIN}"
require_executable_or_command "${SCP_BIN}"
require_file "${DRAFT_MODEL}/config.json"
ADAPTER="${REPO_ROOT}/baselines/specedge/adapter/poisson_client.py"
FACTORY_MODULE="${REPO_ROOT}/baselines/specedge/adapter/official_client_factory.py"
MERGER="${SCRIPT_DIR}/merge_specedge_replay.py"
COMPLETION_TOOL="${SCRIPT_DIR}/materialize_specedge_completions.py"
require_file "${ADAPTER}"
require_file "${FACTORY_MODULE}"
require_file "${MERGER}"
require_file "${COMPLETION_TOOL}"
require_file "${CROP_TOOL}"
require_file "${CLOUD_MARKER_TOOL}"
OFFICIAL_ROOT="${REPO_ROOT}/baselines/specedge/official"
if [[ "${DRY_RUN}" != true ]]; then
  require_file "${OFFICIAL_ROOT}/src/config.py"
  # Official SpecEdge has no generic health RPC.  A TCP connection is a
  # transport-only preflight; each explicit factory then performs its own Sync.
  "${PYTHON_BIN}" - "${GRPC_ADDRESS}" <<'PY'
import socket
import sys

host, separator, port_text = sys.argv[1].rpartition(":")
if not separator or not host or not port_text.isdigit():
    raise SystemExit("invalid --grpc-address")
with socket.create_connection((host, int(port_text)), timeout=3):
    pass
PY
fi

metadata_python init \
  --run-dir "${RUN_DIR}" --run-id "${RUN_ID}" --role specedge-edge --method specedge \
  --config "${CONFIG}" --repo-root "${REPO_ROOT}" --physical-gpus "${EDGE_GPUS}" \
  --peer-ip "10.66.0.5" --model-path "${DRAFT_MODEL}" --provenance-name "${SPECEDGE_EDGE_PROVENANCE_NAME:-node1}"
SOURCE_TRACE_SHA="$(safe_sha256 "${TRACE}")"
metadata_python attach-trace --run-dir "${RUN_DIR}" --trace "${TRACE}"
TRACE="${RUN_DIR}/workload/arrival_trace.jsonl"
require_file "${TRACE}"
[[ "$(safe_sha256 "${TRACE}")" == "${SOURCE_TRACE_SHA}" ]] || die "run-local trace checksum verification failed"

CLIENT_CONFIG="${RUN_DIR}/config/specedge-server-effective.yaml"
REMOTE_CONFIG="${CLOUD_RUN_DIR}/config/specedge-server-effective.yaml"
if [[ "${DRY_RUN}" != true ]]; then
  assert_fresh_path "${CLIENT_CONFIG}"
  note "copying immutable Official effective config from ${CLOUD_HOST}:${REMOTE_CONFIG}"
  "${SSH_BIN}" "${CLOUD_HOST}" "test -f -- '${REMOTE_CONFIG}'"
  REMOTE_CONFIG_SHA="$("${SSH_BIN}" "${CLOUD_HOST}" "sha256sum -- '${REMOTE_CONFIG}'" | awk '{print $1}')"
  [[ "${REMOTE_CONFIG_SHA}" =~ ^[0-9a-fA-F]{64}$ ]] || die "could not obtain SHA-256 for remote effective config"
  "${SCP_BIN}" -p -- "${CLOUD_HOST}:${REMOTE_CONFIG}" "${CLIENT_CONFIG}"
  require_file "${CLIENT_CONFIG}"
  LOCAL_CONFIG_SHA="$(safe_sha256 "${CLIENT_CONFIG}")"
  [[ "${LOCAL_CONFIG_SHA}" == "${REMOTE_CONFIG_SHA}" ]] || die "copied effective config SHA-256 differs from node2 source"
fi

SYNC_DIR="${RUN_DIR}/sync"
READY_0="${SYNC_DIR}/client-0.ready.json"
READY_1="${SYNC_DIR}/client-1.ready.json"
START_FILE="${SYNC_DIR}/common-start.json"
mkdir -- "${SYNC_DIR}"
assert_fresh_path "${READY_0}"
assert_fresh_path "${READY_1}"
assert_fresh_path "${START_FILE}"

LOG_SAMPLER="${RUN_DIR}/logs/gpu_sampler.stderr.log"
LOG_SHUTDOWN="${RUN_DIR}/logs/cloud_shutdown.json"
COMMAND_FILE="${RUN_DIR}/logs/launch_command.sh"
assert_fresh_path "${LOG_SAMPLER}"
assert_fresh_path "${LOG_SHUTDOWN}"
assert_fresh_path "${COMMAND_FILE}"
for client_id in 0 1; do
  assert_fresh_path "${RUN_DIR}/logs/specedge_edge_client${client_id}.stdout.log"
  assert_fresh_path "${RUN_DIR}/logs/specedge_edge_client${client_id}.stderr.log"
  assert_fresh_path "${RUN_DIR}/metrics/requests_proc${client_id}.jsonl"
  assert_fresh_path "${RUN_DIR}/metrics/summary_proc${client_id}.json"
done

IFS=, read -r -a GPU_ARRAY <<< "${EDGE_GPUS}"

write_client_command() {
  local client_id="$1"
  local gpu="$2"
  printf 'env '
  printf '%q ' \
    "PYTHONNOUSERSITE=1" \
    "CUDA_VISIBLE_DEVICES=${gpu}" \
    "SPECEDGE_GRPC_ADDRESS=${GRPC_ADDRESS}" \
    "SPECEDGE_OFFICIAL_ROOT=${OFFICIAL_ROOT}" \
    "SPECEDGE_EFFECTIVE_CONFIG=${CLIENT_CONFIG}" \
    "SPECEDGE_PROMPTS_PATH=${RUN_DIR}/workload/prompts.jsonl" \
    "SPECEDGE_CLIENT_ID=${client_id}" \
    "SPECEDGE_DEVICE=cuda:0" \
    "SPECEDGE_CLIENT_RESULT_PATH=${RUN_DIR}/logs" \
    "SPECEDGE_CLIENT_EXP_NAME=official-client-${client_id}" \
    "SPECEDGE_SYNC_TIMEOUT_S=${SYNC_TIMEOUT_S}"
  printf '%q ' "${PYTHON_BIN}" "${ADAPTER}" \
    --trace "${TRACE}" \
    --output "${RUN_DIR}/metrics/requests_proc${client_id}.jsonl" \
    --summary-output "${RUN_DIR}/metrics/summary_proc${client_id}.json" \
    --run-id "${RUN_ID}" --client-id "${client_id}" --max-concurrency 1 \
    --ready-file "${SYNC_DIR}/client-${client_id}.ready.json" \
    --start-file "${START_FILE}" --start-barrier-timeout-s "${START_BARRIER_TIMEOUT_S}"
  if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' --dry-run
  else
    printf '%q ' --client-factory "${CLIENT_FACTORY}"
  fi
  if [[ -n "${MAX_REQUESTS}" ]]; then printf '%q ' --max-requests "${MAX_REQUESTS}"; fi
  if [[ -n "${REQUEST_TIMEOUT_S}" ]]; then printf '%q ' --request-timeout-s "${REQUEST_TIMEOUT_S}"; fi
  printf '\n'
}

{
  printf '# Generated by FastSD SpecEdge replay wrapper at %s\n' "$(date --iso-8601=seconds)"
  write_client_command 0 "${GPU_ARRAY[0]}"
  write_client_command 1 "${GPU_ARRAY[1]}"
  printf '# The wrapper starts both commands in the background, waits for both ready files, then creates %q.\n' "${START_FILE}"
} > "${COMMAND_FILE}"
metadata_python attach-command --run-dir "${RUN_DIR}" --command-file "${COMMAND_FILE}"

run_client() {
  local client_id="$1"
  local gpu="$2"
  local stdout_path="${RUN_DIR}/logs/specedge_edge_client${client_id}.stdout.log"
  local stderr_path="${RUN_DIR}/logs/specedge_edge_client${client_id}.stderr.log"
  local -a command=(
    env
    "PYTHONNOUSERSITE=1"
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "SPECEDGE_GRPC_ADDRESS=${GRPC_ADDRESS}"
    "SPECEDGE_OFFICIAL_ROOT=${OFFICIAL_ROOT}"
    "SPECEDGE_EFFECTIVE_CONFIG=${CLIENT_CONFIG}"
    "SPECEDGE_PROMPTS_PATH=${RUN_DIR}/workload/prompts.jsonl"
    "SPECEDGE_CLIENT_ID=${client_id}"
    "SPECEDGE_DEVICE=cuda:0"
    "SPECEDGE_CLIENT_RESULT_PATH=${RUN_DIR}/logs"
    "SPECEDGE_CLIENT_EXP_NAME=official-client-${client_id}"
    "SPECEDGE_SYNC_TIMEOUT_S=${SYNC_TIMEOUT_S}"
    "${PYTHON_BIN}" "${ADAPTER}"
    --trace "${TRACE}"
    --output "${RUN_DIR}/metrics/requests_proc${client_id}.jsonl"
    --summary-output "${RUN_DIR}/metrics/summary_proc${client_id}.json"
    --run-id "${RUN_ID}" --client-id "${client_id}" --max-concurrency 1
    --ready-file "${SYNC_DIR}/client-${client_id}.ready.json"
    --start-file "${START_FILE}" --start-barrier-timeout-s "${START_BARRIER_TIMEOUT_S}"
  )
  if [[ "${DRY_RUN}" == true ]]; then
    command+=(--dry-run)
  else
    command+=(--client-factory "${CLIENT_FACTORY}")
  fi
  if [[ -n "${MAX_REQUESTS}" ]]; then command+=(--max-requests "${MAX_REQUESTS}"); fi
  if [[ -n "${REQUEST_TIMEOUT_S}" ]]; then command+=(--request-timeout-s "${REQUEST_TIMEOUT_S}"); fi
  "${command[@]}" > >(tee "${stdout_path}") 2> >(tee "${stderr_path}" >&2)
}

CLIENT_PIDS=()
RUN_EXIT_CODE=1
SAMPLER_PID=""

terminate_client_children() {
  local pid
  for pid in "${CLIENT_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      # Exact child process launched by this wrapper; no discovered or foreign
      # process is ever stopped.
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}

finish_wrapper() {
  local status="failed"
  local sampler_rc=0
  terminate_client_children
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    # Exact sampler child started below; no unrelated process is touched.
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SAMPLER_PID}" ]]; then
    set +e
    wait "${SAMPLER_PID}" 2>/dev/null
    sampler_rc=$?
    set -e
  fi
  if [[ "${RUN_EXIT_CODE}" -eq 0 && "${sampler_rc}" -ne 0 ]]; then
    RUN_EXIT_CODE="${sampler_rc}"
  fi
  if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
    set +e
    "${PYTHON_BIN}" "${CROP_TOOL}" \
      --raw "${RUN_DIR}/metrics/gpu_samples_raw.csv" \
      --records "${RUN_DIR}/metrics/requests.jsonl" \
      --start-field actual_arrival_monotonic_s \
      --end-field completion_monotonic_s \
      --output "${RUN_DIR}/metrics/gpu_samples.csv" \
      >>"${LOG_SAMPLER}" 2>&1
    crop_rc=$?
    set -e
    if [[ "${crop_rc}" -ne 0 ]]; then
      RUN_EXIT_CODE="${crop_rc}"
      note "could not create measurement-window SpecEdge edge GPU samples"
    fi
  fi
  if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
    status="complete"
  elif [[ "${RUN_EXIT_CODE}" -eq 130 || "${RUN_EXIT_CODE}" -eq 143 ]]; then
    status="cancelled"
  fi
  metadata_python finish --run-dir "${RUN_DIR}" --status "${status}" --exit-code "${RUN_EXIT_CODE}" || note "could not finalize SpecEdge edge manifest"
  trap - EXIT
  exit "${RUN_EXIT_CODE}"
}
trap finish_wrapper EXIT

# The sampler covers the same component process lifetime and uses only the two
# explicitly allocated physical GPUs.  Arrival/E2E measurement starts later at
# the common barrier, after both clients have prepared.
# Retain cold-start samples separately; only the cropped canonical CSV enters
# J/token analysis after the shared Poisson barrier has completed.
sampler_python --output "${RUN_DIR}/metrics/gpu_samples_raw.csv" --physical-gpus "${EDGE_GPUS}" --interval-ms "${GPU_SAMPLE_INTERVAL_MS}" >>"${LOG_SAMPLER}" 2>&1 &
SAMPLER_PID=$!

note "starting two SpecEdge replay clients with common Poisson barrier; factory=${CLIENT_FACTORY:-dry-run}, grpc=${GRPC_ADDRESS}"
run_client 0 "${GPU_ARRAY[0]}" &
CLIENT_PIDS+=("$!")
run_client 1 "${GPU_ARRAY[1]}" &
CLIENT_PIDS+=("$!")

if ! "${PYTHON_BIN}" - "${READY_0}" "${READY_1}" "${START_FILE}" "${START_BARRIER_TIMEOUT_S}" "${CLIENT_PIDS[0]}" "${CLIENT_PIDS[1]}" <<'PY'
import json
import os
import pathlib
import sys
import time

ready_paths = [pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])]
start_path = pathlib.Path(sys.argv[3])
timeout_s = float(sys.argv[4])
pids = [int(sys.argv[5]), int(sys.argv[6])]
deadline = time.monotonic() + timeout_s
while not all(path.is_file() for path in ready_paths):
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"SpecEdge child exited before readiness: {pid}")
    if time.monotonic() >= deadline:
        raise SystemExit("timed out waiting for both SpecEdge client ready artifacts")
    time.sleep(0.05)
if start_path.exists() or start_path.is_symlink():
    raise SystemExit(f"refusing to overwrite start barrier: {start_path}")
payload = {"schema_version": 1, "run_started_monotonic_s": time.monotonic() + 2.0}
with start_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
then
  RUN_EXIT_CODE=1
else
  RUN_EXIT_CODE=0
fi

if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
  # Absolute client deadlines are relative to the shared node1 monotonic
  # barrier.  Enforce the requested 10-minute maximum drain after the final
  # scheduled arrival, while retaining any partial child artifacts on failure.
  set +e
  "${PYTHON_BIN}" - "${TRACE}" "${START_FILE}" "${DRAIN_TIMEOUT_S}" "${MAX_REQUESTS}" "${CLIENT_PIDS[@]}" <<'PY'
import json
import os
import pathlib
import sys
import time

trace_path = pathlib.Path(sys.argv[1])
start_path = pathlib.Path(sys.argv[2])
drain_s = float(sys.argv[3])
max_requests = int(sys.argv[4]) if sys.argv[4] else None
pids = [int(value) for value in sys.argv[5:]]
trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if max_requests is not None:
    trace = trace[:max_requests]
if not trace:
    raise SystemExit("empty selected trace")
barrier = json.loads(start_path.read_text(encoding="utf-8"))["run_started_monotonic_s"]
last_scheduled = max(float(item["scheduled_offset_s"]) for item in trace)
deadline = float(barrier) + last_scheduled + drain_s
while True:
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        # A completed child can remain a zombie until the Bash parent reaps it
        # below.  Treat that as finished rather than falsely burning the whole
        # post-arrival drain allowance.
        stat_path = pathlib.Path(f"/proc/{pid}/stat")
        try:
            state = stat_path.read_text(encoding="utf-8").rsplit(") ", 1)[1][:1]
        except (FileNotFoundError, IndexError):
            continue
        if state == "Z":
            continue
        alive.append(pid)
    if not alive:
        break
    if time.monotonic() >= deadline:
        raise SystemExit(
            "SpecEdge post-arrival drain exceeded --drain-timeout-s=" + str(drain_s)
        )
    time.sleep(0.05)
PY
  watchdog_rc=$?
  set -e
  if [[ "${watchdog_rc}" -ne 0 ]]; then
    RUN_EXIT_CODE="${watchdog_rc}"
    terminate_client_children
  fi
fi

for pid in "${CLIENT_PIDS[@]}"; do
  set +e
  wait "${pid}"
  client_rc=$?
  set -e
  if [[ "${client_rc}" -ne 0 ]]; then
    RUN_EXIT_CODE="${client_rc}"
  fi
done
# Both client children have been reaped.  Clear their exact PIDs before the
# EXIT trap so a later PID reuse can never be mistaken for one of our children.
CLIENT_PIDS=()

if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
  merge_command=(
    "${PYTHON_BIN}" "${MERGER}"
    --trace "${TRACE}"
    --requests "${RUN_DIR}/metrics/requests_proc0.jsonl" "${RUN_DIR}/metrics/requests_proc1.jsonl"
    --summaries "${RUN_DIR}/metrics/summary_proc0.json" "${RUN_DIR}/metrics/summary_proc1.json"
    --output "${RUN_DIR}/metrics/requests.jsonl"
    --summary-output "${RUN_DIR}/metrics/summary.json"
    --run-id "${RUN_ID}"
  )
  if [[ -n "${MAX_REQUESTS}" ]]; then merge_command+=(--max-requests "${MAX_REQUESTS}"); fi
  set +e
  "${merge_command[@]}" >>"${RUN_DIR}/logs/specedge_edge_client0.stdout.log" 2>>"${RUN_DIR}/logs/specedge_edge_client0.stderr.log"
  RUN_EXIT_CODE=$?
  set -e
fi
if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
  set +e
  "${PYTHON_BIN}" "${COMPLETION_TOOL}" \
    --trace "${TRACE}" --requests "${RUN_DIR}/metrics/requests.jsonl" \
    --output "${RUN_DIR}/outputs/completions.jsonl" >>"${RUN_DIR}/logs/specedge_edge_client0.stdout.log" 2>>"${RUN_DIR}/logs/specedge_edge_client0.stderr.log"
  RUN_EXIT_CODE=$?
  set -e
fi
if [[ "${RUN_EXIT_CODE}" -eq 0 && "${DRY_RUN}" != true ]]; then
  CLOUD_SHUTDOWN_FILE="${CLOUD_RUN_DIR}/control/graceful-shutdown.json"
  note "requesting graceful shutdown of the matching SpecEdge cloud run"
  set +e
  "${SSH_BIN}" "${CLOUD_HOST}" \
    "python3 '${REMOTE_REPO_ROOT}/scripts/experiments/write_control_marker.py' --path '${CLOUD_SHUTDOWN_FILE}' --run-id '${RUN_ID}'" \
    >"${LOG_SHUTDOWN}" 2>&1
  shutdown_rc=$?
  set -e
  if [[ "${shutdown_rc}" -ne 0 ]]; then
    RUN_EXIT_CODE="${shutdown_rc}"
    note "cloud shutdown marker could not be created; edge result is not eligible for finalization"
  else
    note "waiting for the matching SpecEdge cloud manifest to reach complete"
    shutdown_deadline=$((SECONDS + CLOUD_SHUTDOWN_TIMEOUT_S))
    while true; do
      cloud_status="$("${SSH_BIN}" "${CLOUD_HOST}" "python3 - '${CLOUD_RUN_DIR}/manifest.json' <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding='utf-8'))
print(payload.get('status', ''))
PY" 2>/dev/null || true)"
      if [[ "${cloud_status}" == "complete" ]]; then
        break
      fi
      if (( SECONDS >= shutdown_deadline )); then
        RUN_EXIT_CODE=1
        note "SpecEdge cloud did not finalize complete within ${CLOUD_SHUTDOWN_TIMEOUT_S}s"
        break
      fi
      sleep 1
    done
  fi
fi
