#!/usr/bin/env bash
# Foreground edge replay launcher for FastSD or its strict Vanilla baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  EDGE_PHYSICAL_GPUS=0,1 SERVER_URL=http://10.66.0.5:1597 \
  bash scripts/experiments/run_fastsd_replay.sh \
      --config configs/experiments/mtbench_poisson.yaml \
      --method fastsd|vanilla --trace /home/hdd/.../arrival_trace.jsonl \
      --run-id ID --run-dir /home/hdd/.../edge

The trace is copied into the new run directory and its hash becomes part of the
manifest before any model is loaded.

For the 16-request calibration pilot, use --arrival-mode closed_loop together
with --max-requests 16 (no --trace).  Closed-loop calibration is recorded in
the component run but is intentionally not eligible for the Poisson matrix.
EOF
}

CONFIG=""
METHOD=""
TRACE=""
ARRIVAL_MODE="poisson"
MAX_REQUESTS=""
MAX_TOKENS="256"
DATASET="mt_bench"
DATA_PATH="${REPO_ROOT}/data/mt_bench.jsonl"
RUN_ID=""
RUN_DIR=""
EDGE_GPUS="${EDGE_PHYSICAL_GPUS:-}"
SERVER_URL="${SERVER_URL:-http://10.66.0.5:1597}"
PYTHON_BIN="${FASTSD_PYTHON:-/home/hdd/zhangh/envs/fastsd/bin/python}"
DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"
TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
GPU_SAMPLE_INTERVAL_MS="${FASTSD_GPU_SAMPLE_INTERVAL_MS:-50}"
DRAIN_TIMEOUT_S="${FASTSD_DRAIN_TIMEOUT_S:-600}"
CLOUD_READY_TIMEOUT_S="${FASTSD_CLOUD_READY_TIMEOUT_S:-600}"
CROP_TOOL="${SCRIPT_DIR}/crop_gpu_samples.py"
CLOSED_LOOP_PLAN_TOOL="${SCRIPT_DIR}/write_closed_loop_plan.py"
CLOUD_HOST="${FASTSD_CLOUD_HOST:-node2}"
CLOUD_RUN_DIR="${FASTSD_CLOUD_RUN_DIR:-}"
SSH_BIN="${FASTSD_SSH_BIN:-ssh}"
CLOUD_SHUTDOWN_TIMEOUT_S="${FASTSD_CLOUD_SHUTDOWN_TIMEOUT_S:-120}"
SMOKE_TUNNEL_MODE="${FASTSD_SMOKE_TUNNEL_MODE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2 ;;
    --method) METHOD="${2:-}"; shift 2 ;;
    --trace) TRACE="${2:-}"; shift 2 ;;
    --arrival-mode) ARRIVAL_MODE="${2:-}"; shift 2 ;;
    --max-requests) MAX_REQUESTS="${2:-}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:-}"; shift 2 ;;
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --data) DATA_PATH="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --edge-physical-gpus) EDGE_GPUS="${2:-}"; shift 2 ;;
    --server-url) SERVER_URL="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --draft-model) DRAFT_MODEL="${2:-}"; shift 2 ;;
    --target-model) TARGET_MODEL="${2:-}"; shift 2 ;;
    --gpu-sample-interval-ms) GPU_SAMPLE_INTERVAL_MS="${2:-}"; shift 2 ;;
    --drain-timeout-s) DRAIN_TIMEOUT_S="${2:-}"; shift 2 ;;
    --cloud-host) CLOUD_HOST="${2:-}"; shift 2 ;;
    --cloud-run-dir) CLOUD_RUN_DIR="${2:-}"; shift 2 ;;
    --cloud-shutdown-timeout-s) CLOUD_SHUTDOWN_TIMEOUT_S="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${CONFIG}" && -n "${METHOD}" && -n "${RUN_ID}" && -n "${RUN_DIR}" && -n "${EDGE_GPUS}" ]] || { usage >&2; die "Missing required arguments."; }
[[ "${METHOD}" == "fastsd" || "${METHOD}" == "vanilla" ]] || die "--method must be fastsd or vanilla."
[[ "${ARRIVAL_MODE}" == "poisson" || "${ARRIVAL_MODE}" == "closed_loop" ]] || die "--arrival-mode must be poisson or closed_loop."
[[ "${DATASET}" == "mt_bench" || "${DATASET}" == "gsm8k" || "${DATASET}" == "humaneval" ]] || die "--dataset must be mt_bench, gsm8k, or humaneval."
[[ "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "--max-tokens must be a positive integer."
if [[ -n "${MAX_REQUESTS}" ]]; then
  [[ "${MAX_REQUESTS}" =~ ^[1-9][0-9]*$ ]] || die "--max-requests must be a positive integer."
fi
if [[ "${ARRIVAL_MODE}" == "poisson" ]]; then
  [[ -n "${TRACE}" ]] || die "--trace is required for --arrival-mode poisson."
else
  [[ -z "${TRACE}" ]] || die "--trace is not used for --arrival-mode closed_loop."
  [[ -n "${MAX_REQUESTS}" ]] || die "--max-requests is required for closed_loop calibration."
fi
[[ "${DRAIN_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--drain-timeout-s must be a positive number."
awk -v timeout="${DRAIN_TIMEOUT_S}" 'BEGIN { exit !(timeout > 0) }' || die "--drain-timeout-s must be greater than zero."
[[ "${CLOUD_READY_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || die "FASTSD_CLOUD_READY_TIMEOUT_S must be a positive integer number of seconds."
[[ "${CLOUD_SHUTDOWN_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || die "FASTSD_CLOUD_SHUTDOWN_TIMEOUT_S must be a positive integer number of seconds."
[[ "${SMOKE_TUNNEL_MODE}" == "0" || "${SMOKE_TUNNEL_MODE}" == "1" ]] || die "FASTSD_SMOKE_TUNNEL_MODE must be 0 or 1."
validate_run_id "${RUN_ID}"
validate_hdd_path "${RUN_DIR}"
if [[ "${ARRIVAL_MODE}" == "poisson" ]]; then validate_hdd_path "${TRACE}"; fi
assert_fresh_component_run_path "${RUN_DIR}" edge
validate_gpu_list "${EDGE_GPUS}" 2
if [[ "${SERVER_URL}" != "http://10.66.0.5:1597" ]]; then
  if [[ "${SMOKE_TUNNEL_MODE}" != "1" || ! "${SERVER_URL}" =~ ^http://127[.]0[.]0[.]1:[0-9]+$ ]]; then
    die "FastSD replay must use the IB cloud endpoint http://10.66.0.5:1597; loopback is allowed only with FASTSD_SMOKE_TUNNEL_MODE=1 for non-performance smoke runs."
  fi
  note "smoke tunnel mode enabled; network timing from this run is not eligible for performance analysis"
fi
validate_host_alias "${CLOUD_HOST}"
if [[ -z "${CLOUD_RUN_DIR}" ]]; then CLOUD_RUN_DIR="$(dirname "${RUN_DIR}")/cloud"; fi
validate_hdd_path "${CLOUD_RUN_DIR}"
if [[ "${CONFIG}" != /* ]]; then CONFIG="${REPO_ROOT}/${CONFIG}"; fi
if [[ "${DRAFT_MODEL}" != /* ]]; then DRAFT_MODEL="${REPO_ROOT}/${DRAFT_MODEL}"; fi
if [[ "${TARGET_MODEL}" != /* ]]; then TARGET_MODEL="${REPO_ROOT}/${TARGET_MODEL}"; fi
if [[ "${DATA_PATH}" != /* ]]; then DATA_PATH="${REPO_ROOT}/${DATA_PATH}"; fi
require_file "${CONFIG}"
if [[ "${ARRIVAL_MODE}" == "poisson" ]]; then require_file "${TRACE}"; fi
require_file "${DATA_PATH}"
require_executable_or_command "${PYTHON_BIN}"
require_file "${DRAFT_MODEL}/config.json"
require_file "${DRAFT_MODEL}/tokenizer.json"
require_file "${CROP_TOOL}"
require_file "${CLOSED_LOOP_PLAN_TOOL}"
require_executable_or_command "${SSH_BIN}"
if [[ "${ARRIVAL_MODE}" == "closed_loop" ]]; then
  (( MAX_REQUESTS % 2 == 0 )) || die "closed_loop --max-requests must divide evenly across two edge clients."
  MAX_TASKS_PER_DRAFT=$(( MAX_REQUESTS / 2 ))
else
  MAX_TASKS_PER_DRAFT=40
fi

if command -v curl >/dev/null 2>&1; then
  note "waiting for the matching cloud run to finish target/KV initialization before allocating edge models"
  health_deadline=$((SECONDS + CLOUD_READY_TIMEOUT_S))
  while true; do
    health_payload="$(curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 "${SERVER_URL}/health" 2>/dev/null || true)"
    if [[ -n "${health_payload}" ]] && "${PYTHON_BIN}" - "${RUN_ID}" "${health_payload}" <<'PY'
import json
import sys

expected_run_id, raw = sys.argv[1:]
payload = json.loads(raw)
if payload.get("status") != "ok" or payload.get("run_id") != expected_run_id:
    raise SystemExit(1)
PY
    then
      break
    fi
    if (( SECONDS >= health_deadline )); then
      die "cloud did not become ready for run_id=${RUN_ID} within ${CLOUD_READY_TIMEOUT_S}s; no edge model was started."
    fi
    sleep 2
  done
else
  die "curl is required for the non-mutating cloud health check."
fi

if [[ "${METHOD}" == "fastsd" ]]; then
  PROFILE_ARGS=(--profile custom --server_sched_mode fastsd --enable_proactive_draft --no-enable_pipeline)
else
  PROFILE_ARGS=(--profile vanilla --server_sched_mode vanilla --no-enable_proactive_draft --no-enable_pipeline)
fi

metadata_python init \
  --run-dir "${RUN_DIR}" --run-id "${RUN_ID}" --role edge --method "${METHOD}" \
  --config "${CONFIG}" --repo-root "${REPO_ROOT}" --physical-gpus "${EDGE_GPUS}" \
  --peer-ip "10.66.0.5" --model-path "${DRAFT_MODEL}" --model-path "${TARGET_MODEL}" \
  --provenance-name "${FASTSD_EDGE_PROVENANCE_NAME:-node1}"
if [[ "${ARRIVAL_MODE}" == "poisson" ]]; then
  SOURCE_TRACE_SHA="$(safe_sha256 "${TRACE}")"
  metadata_python attach-trace --run-dir "${RUN_DIR}" --trace "${TRACE}"
  TRACE="${RUN_DIR}/workload/arrival_trace.jsonl"
  require_file "${TRACE}"
  [[ "$(safe_sha256 "${TRACE}")" == "${SOURCE_TRACE_SHA}" ]] || die "run-local trace checksum verification failed"
else
  "${PYTHON_BIN}" "${CLOSED_LOOP_PLAN_TOOL}" \
    --run-dir "${RUN_DIR}" --dataset "${DATASET}" --data "${DATA_PATH}" \
    --max-requests "${MAX_REQUESTS}" --num-clients 2
  metadata_python attach-closed-loop-plan --run-dir "${RUN_DIR}"
fi

LOG_STDOUT="${RUN_DIR}/logs/edge.stdout.log"
LOG_STDERR="${RUN_DIR}/logs/edge.stderr.log"
LOG_SAMPLER="${RUN_DIR}/logs/gpu_sampler.stderr.log"
LOG_SHUTDOWN="${RUN_DIR}/logs/cloud_shutdown.json"
COMMAND_FILE="${RUN_DIR}/logs/launch_command.sh"
assert_fresh_path "${LOG_STDOUT}"
assert_fresh_path "${LOG_STDERR}"
assert_fresh_path "${LOG_SAMPLER}"
assert_fresh_path "${LOG_SHUTDOWN}"
assert_fresh_path "${COMMAND_FILE}"

LAUNCH=(
  env
  "PYTHONNOUSERSITE=1"
  "TOKENIZERS_PARALLELISM=false"
  "NO_PROXY=10.66.0.4,10.66.0.5,127.0.0.1,localhost"
  "no_proxy=10.66.0.4,10.66.0.5,127.0.0.1,localhost"
  "CUDA_VISIBLE_DEVICES=${EDGE_GPUS}"
  "FASTSD_RESULTS_DIR=${RUN_DIR}"
  "${PYTHON_BIN}" "${REPO_ROOT}/edge/edge.py"
  --config "${CONFIG}" --method "${METHOD}" --run-id "${RUN_ID}" --run-dir "${RUN_DIR}"
  --edge-physical-gpus "${EDGE_GPUS}"
  --arrival-mode "${ARRIVAL_MODE}" --server_url "${SERVER_URL}"
  --draft_model "${DRAFT_MODEL}" --target_model "${TARGET_MODEL}" --tokenizer-model "${DRAFT_MODEL}" --model-dtype bfloat16
  --edge_gpu_start 0 --edge_gpus 2 --num_drafts 2 --max_tasks_per_draft "${MAX_TASKS_PER_DRAFT}"
  --dataset "${DATASET}" --data_path "${DATA_PATH}"
  --gamma 6 --max_tokens "${MAX_TOKENS}" --temp 0 --top_k 0 --top_p 1.0 --token_budget 256 --max_num_seqs 2 --prefill_max_wait_cycles 2
  --post-arrival-drain-timeout-s "${DRAIN_TIMEOUT_S}"
  "${PROFILE_ARGS[@]}"
)
if [[ "${ARRIVAL_MODE}" == "poisson" ]]; then
  LAUNCH+=(--arrival-trace-in "${TRACE}")
  if [[ -n "${MAX_REQUESTS}" ]]; then LAUNCH+=(--max-requests "${MAX_REQUESTS}"); fi
fi
write_command_file "${COMMAND_FILE}" "${LAUNCH[@]}"
metadata_python attach-command --run-dir "${RUN_DIR}" --command-file "${COMMAND_FILE}"

# Preserve complete cold-start evidence separately.  The EXIT handler creates
# the canonical gpu_samples.csv only after clipping it to actual arrivals and
# completions, so J/token never includes model load or service startup.
sampler_python --output "${RUN_DIR}/metrics/gpu_samples_raw.csv" --physical-gpus "${EDGE_GPUS}" --interval-ms "${GPU_SAMPLE_INTERVAL_MS}" >>"${LOG_SAMPLER}" 2>&1 &
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
    # This is only the sampler child started by this wrapper.
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
  if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
    if [[ "${ARRIVAL_MODE}" != "poisson" ]]; then
      # Closed-loop pilot has the same local request-window semantics, but no
      # shared trace.  It remains outside root finalization/matrix analysis.
      :
    fi
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
      status="failed"
      note "could not create measurement-window edge GPU samples"
    fi
  fi
  metadata_python finish --run-dir "${RUN_DIR}" --status "${status}" --exit-code "${RUN_EXIT_CODE}" || note "could not finalize edge manifest"
  trap - EXIT
  exit "${RUN_EXIT_CODE}"
}
trap finish_wrapper EXIT

note "starting ${METHOD} edge replay; physical GPUs ${EDGE_GPUS} map to cuda:0,cuda:1"
set +e
"${LAUNCH[@]}" > >(tee "${LOG_STDOUT}") 2> >(tee "${LOG_STDERR}" >&2)
RUN_EXIT_CODE=$?
set -e
if [[ "${RUN_EXIT_CODE}" -eq 0 ]]; then
  note "requesting graceful shutdown of the matching cloud run"
  set +e
  curl --noproxy '*' --fail --silent --show-error \
    -X POST -H "X-FastSD-Run-ID: ${RUN_ID}" \
    "${SERVER_URL}/shutdown" > "${LOG_SHUTDOWN}"
  shutdown_rc=$?
  set -e
  if [[ "${shutdown_rc}" -ne 0 ]]; then
    RUN_EXIT_CODE="${shutdown_rc}"
    note "cloud shutdown request failed; edge result is not eligible for finalization"
  elif [[ "${SMOKE_TUNNEL_MODE}" == "1" ]]; then
    note "smoke tunnel shutdown accepted; cloud manifest completion must be verified by the external tunnel controller"
  else
    note "waiting for the matching cloud manifest to reach complete"
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
        note "cloud did not finalize complete within ${CLOUD_SHUTDOWN_TIMEOUT_S}s after its matching shutdown request"
        break
      fi
      sleep 1
    done
  fi
fi
