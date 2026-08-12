#!/usr/bin/env bash
# Foreground server wrapper for a transport-only SpecEdge adapter.
# The pinned upstream batch_server.py binds [::]:8000, so this wrapper refuses
# to start it directly: doing so would violate the IB-only experiment contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  CLOUD_PHYSICAL_GPU=0 bash scripts/experiments/start_specedge_server.sh \
      --config configs/experiments/mtbench_poisson.yaml --run-id ID \
      --run-dir /home/hdd/.../cloud --bind-host 10.66.0.5 --port 18000

An explicit transport adapter is required because the pinned Official SpecEdge
server hard-codes [::]:8000.  Set --server-entrypoint (or
SPECEDGE_SERVER_ENTRYPOINT) to an adapter that accepts the displayed arguments
and binds only the requested IB host and port.  This wrapper never falls back
to the unsafe official server command.

--server-events-output defaults to
<run-dir>/metrics/specedge_server_events.jsonl and is passed to the adapter.
EOF
}

CONFIG=""
RUN_ID=""
RUN_DIR=""
BIND_HOST=""
PORT=""
SERVER_EVENTS_OUTPUT="${SPECEDGE_SERVER_EVENTS_OUTPUT:-}"
SHUTDOWN_FILE="${SPECEDGE_SHUTDOWN_FILE:-}"
MAX_TOKENS=""
CLOUD_GPU="${CLOUD_PHYSICAL_GPU:-}"
PYTHON_BIN="${SPECEDGE_PYTHON:-/home/hdd/zhangh/envs/specedge/bin/python}"
TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"
SERVER_ENTRYPOINT="${SPECEDGE_SERVER_ENTRYPOINT:-${REPO_ROOT}/baselines/specedge/adapter/server_entrypoint.py}"
GPU_SAMPLE_INTERVAL_MS="${FASTSD_GPU_SAMPLE_INTERVAL_MS:-50}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --bind-host) BIND_HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --server-events-output) SERVER_EVENTS_OUTPUT="${2:-}"; shift 2 ;;
    --shutdown-file) SHUTDOWN_FILE="${2:-}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:-}"; shift 2 ;;
    --cloud-physical-gpu) CLOUD_GPU="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --target-model) TARGET_MODEL="${2:-}"; shift 2 ;;
    --draft-model) DRAFT_MODEL="${2:-}"; shift 2 ;;
    --server-entrypoint) SERVER_ENTRYPOINT="${2:-}"; shift 2 ;;
    --gpu-sample-interval-ms) GPU_SAMPLE_INTERVAL_MS="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${CONFIG}" && -n "${RUN_ID}" && -n "${RUN_DIR}" && -n "${BIND_HOST}" && -n "${PORT}" && -n "${CLOUD_GPU}" ]] || { usage >&2; die "Missing required arguments."; }
validate_run_id "${RUN_ID}"
validate_hdd_path "${RUN_DIR}"
assert_fresh_component_run_path "${RUN_DIR}" cloud
validate_ipv4 "${BIND_HOST}"
validate_port "${PORT}"
validate_gpu_list "${CLOUD_GPU}" 1
if [[ -n "${MAX_TOKENS}" ]]; then
  [[ "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "--max-tokens must be a positive integer."
fi
if [[ "${CONFIG}" != /* ]]; then CONFIG="${REPO_ROOT}/${CONFIG}"; fi
if [[ "${TARGET_MODEL}" != /* ]]; then TARGET_MODEL="${REPO_ROOT}/${TARGET_MODEL}"; fi
if [[ "${DRAFT_MODEL}" != /* ]]; then DRAFT_MODEL="${REPO_ROOT}/${DRAFT_MODEL}"; fi
if [[ "${SERVER_ENTRYPOINT}" != /* ]]; then SERVER_ENTRYPOINT="${REPO_ROOT}/${SERVER_ENTRYPOINT}"; fi
if [[ -z "${SERVER_EVENTS_OUTPUT}" ]]; then SERVER_EVENTS_OUTPUT="${RUN_DIR}/metrics/specedge_server_events.jsonl"; fi
if [[ -z "${SHUTDOWN_FILE}" ]]; then SHUTDOWN_FILE="${RUN_DIR}/control/graceful-shutdown.json"; fi
validate_hdd_path "${SERVER_EVENTS_OUTPUT}"
validate_hdd_path "${SHUTDOWN_FILE}"
[[ "${SHUTDOWN_FILE}" == "${RUN_DIR}/control/graceful-shutdown.json" ]] || die "--shutdown-file must be this run's control/graceful-shutdown.json."
require_file "${CONFIG}"
require_executable_or_command "${PYTHON_BIN}"
require_file "${TARGET_MODEL}/config.json"
require_file "${SERVER_ENTRYPOINT}"

metadata_python init \
  --run-dir "${RUN_DIR}" --run-id "${RUN_ID}" --role specedge-cloud --method specedge \
  --config "${CONFIG}" --repo-root "${REPO_ROOT}" --physical-gpus "${CLOUD_GPU}" \
  --bind-host "${BIND_HOST}" --port "${PORT}" --peer-ip "10.66.0.4" \
  --model-path "${TARGET_MODEL}" --provenance-name "${SPECEDGE_CLOUD_PROVENANCE_NAME:-node2}"

LOG_STDOUT="${RUN_DIR}/logs/specedge_cloud.stdout.log"
LOG_STDERR="${RUN_DIR}/logs/specedge_cloud.stderr.log"
LOG_SAMPLER="${RUN_DIR}/logs/gpu_sampler.stderr.log"
COMMAND_FILE="${RUN_DIR}/logs/launch_command.sh"
assert_fresh_path "${LOG_STDOUT}"
assert_fresh_path "${LOG_STDERR}"
assert_fresh_path "${LOG_SAMPLER}"
assert_fresh_path "${COMMAND_FILE}"
assert_fresh_path "${SERVER_EVENTS_OUTPUT}"
mkdir -- "${RUN_DIR}/control"
assert_fresh_path "${SHUTDOWN_FILE}"

LAUNCH=(
  env
  "PYTHONNOUSERSITE=1"
  "TOKENIZERS_PARALLELISM=false"
  "CUDA_VISIBLE_DEVICES=${CLOUD_GPU}"
  "SPECEDGE_BIND_HOST=${BIND_HOST}"
  "SPECEDGE_PORT=${PORT}"
  "SPECEDGE_RESULTS_DIR=${RUN_DIR}"
  "${PYTHON_BIN}" "${SERVER_ENTRYPOINT}"
  --config "${CONFIG}" --run-id "${RUN_ID}" --run-dir "${RUN_DIR}"
  --bind-host "${BIND_HOST}" --port "${PORT}"
  --server-events-output "${SERVER_EVENTS_OUTPUT}"
  --shutdown-file "${SHUTDOWN_FILE}"
  --result-path "${RUN_DIR}/logs"
  --target-model "${TARGET_MODEL}" --draft-model "${DRAFT_MODEL}" --cloud-physical-gpu "${CLOUD_GPU}"
)
if [[ -n "${MAX_TOKENS}" ]]; then LAUNCH+=(--max-new-tokens "${MAX_TOKENS}"); fi
write_command_file "${COMMAND_FILE}" "${LAUNCH[@]}"
metadata_python attach-command --run-dir "${RUN_DIR}" --command-file "${COMMAND_FILE}"

# Keep startup/warmup samples immutable.  node1 finalization later crops a
# canonical gpu_samples.csv against server-side adapter events on node2.
sampler_python --output "${RUN_DIR}/metrics/gpu_samples_raw.csv" --physical-gpus "${CLOUD_GPU}" --interval-ms "${GPU_SAMPLE_INTERVAL_MS}" >>"${LOG_SAMPLER}" 2>&1 &
SAMPLER_PID=$!
RUN_EXIT_CODE=1

finish_wrapper() {
  local status="failed"
  local sampler_rc=0
  if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
    status="complete"
  elif [[ "${RUN_EXIT_CODE}" -eq 130 || "${RUN_EXIT_CODE}" -eq 143 ]]; then
    status="cancelled"
  fi
  if kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
  fi
  set +e
  wait "${SAMPLER_PID}" 2>/dev/null
  sampler_rc=$?
  set -e
  if [[ "${RUN_EXIT_CODE}" -eq 0 && "${sampler_rc}" -ne 0 ]]; then
    RUN_EXIT_CODE="${sampler_rc}"
    status="failed"
  fi
  metadata_python finish --run-dir "${RUN_DIR}" --status "${status}" --exit-code "${RUN_EXIT_CODE}" || note "could not finalize SpecEdge cloud manifest"
  trap - EXIT
  exit "${RUN_EXIT_CODE}"
}
trap finish_wrapper EXIT

note "starting SpecEdge transport adapter on ${BIND_HOST}:${PORT}; GPU ${CLOUD_GPU} is cuda:0"
set +e
"${LAUNCH[@]}" > >(tee "${LOG_STDOUT}") 2> >(tee "${LOG_STDERR}" >&2)
RUN_EXIT_CODE=$?
set -e
