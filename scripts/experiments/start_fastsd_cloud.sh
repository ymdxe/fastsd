#!/usr/bin/env bash
# Foreground FastSD cloud service launcher with a fresh, auditable component run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  CLOUD_PHYSICAL_GPU=0 bash scripts/experiments/start_fastsd_cloud.sh \
      --config configs/experiments/mtbench_poisson.yaml \
      --method fastsd|vanilla --run-id ID --run-dir /home/hdd/.../cloud \
      --bind-host 10.66.0.5 --port 1597

The service remains in the foreground.  Ctrl-C is handled as a cancelled run;
only the sampler child created by this wrapper receives a graceful SIGTERM.
EOF
}

CONFIG=""
METHOD=""
RUN_ID=""
RUN_DIR=""
BIND_HOST=""
PORT=""
CLOUD_GPU="${CLOUD_PHYSICAL_GPU:-${FASTSD_CLOUD_GPU:-}}"
PYTHON_BIN="${FASTSD_PYTHON:-/home/hdd/zhangh/envs/fastsd/bin/python}"
TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
TOKENIZER_MODEL="${FASTSD_CLOUD_TOKENIZER_MODEL:-${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}}"
GPU_SAMPLE_INTERVAL_MS="${FASTSD_GPU_SAMPLE_INTERVAL_MS:-50}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2 ;;
    --method) METHOD="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --bind-host) BIND_HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --cloud-physical-gpu) CLOUD_GPU="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --target-model) TARGET_MODEL="${2:-}"; shift 2 ;;
    --draft-model|--tokenizer-model) TOKENIZER_MODEL="${2:-}"; shift 2 ;;
    --gpu-sample-interval-ms) GPU_SAMPLE_INTERVAL_MS="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${CONFIG}" && -n "${METHOD}" && -n "${RUN_ID}" && -n "${RUN_DIR}" && -n "${BIND_HOST}" && -n "${PORT}" && -n "${CLOUD_GPU}" ]] || { usage >&2; die "Missing required arguments."; }
[[ "${METHOD}" == "fastsd" || "${METHOD}" == "vanilla" ]] || die "--method must be fastsd or vanilla."
validate_run_id "${RUN_ID}"
validate_hdd_path "${RUN_DIR}"
assert_fresh_component_run_path "${RUN_DIR}" cloud
validate_ipv4 "${BIND_HOST}"
validate_port "${PORT}"
validate_gpu_list "${CLOUD_GPU}" 1
[[ "${GPU_SAMPLE_INTERVAL_MS}" =~ ^[0-9]+$ ]] || die "--gpu-sample-interval-ms must be a positive integer."
if [[ "${CONFIG}" != /* ]]; then CONFIG="${REPO_ROOT}/${CONFIG}"; fi
if [[ "${TARGET_MODEL}" != /* ]]; then TARGET_MODEL="${REPO_ROOT}/${TARGET_MODEL}"; fi
if [[ "${TOKENIZER_MODEL}" != /* ]]; then TOKENIZER_MODEL="${REPO_ROOT}/${TOKENIZER_MODEL}"; fi
require_file "${CONFIG}"
require_executable_or_command "${PYTHON_BIN}"
require_file "${TARGET_MODEL}/config.json"
require_file "${TOKENIZER_MODEL}/tokenizer.json"

if [[ "${METHOD}" == "fastsd" ]]; then
  PROFILE_ARGS=(--profile custom --server_sched_mode fastsd --no-enable_pipeline)
else
  PROFILE_ARGS=(--profile vanilla --server_sched_mode vanilla --no-enable_pipeline --no-enable_proactive_draft)
fi

MODEL_METADATA_ARGS=(--model-path "${TARGET_MODEL}")
if [[ "${TOKENIZER_MODEL}" != "${TARGET_MODEL}" ]]; then
  MODEL_METADATA_ARGS+=(--model-path "${TOKENIZER_MODEL}")
fi
metadata_python init \
  --run-dir "${RUN_DIR}" --run-id "${RUN_ID}" --role cloud --method "${METHOD}" \
  --config "${CONFIG}" --repo-root "${REPO_ROOT}" --physical-gpus "${CLOUD_GPU}" \
  --bind-host "${BIND_HOST}" --port "${PORT}" --peer-ip "10.66.0.4" \
  "${MODEL_METADATA_ARGS[@]}" --provenance-name "${FASTSD_CLOUD_PROVENANCE_NAME:-node2}"

LOG_STDOUT="${RUN_DIR}/logs/cloud.stdout.log"
LOG_STDERR="${RUN_DIR}/logs/cloud.stderr.log"
LOG_SAMPLER="${RUN_DIR}/logs/gpu_sampler.stderr.log"
COMMAND_FILE="${RUN_DIR}/logs/launch_command.sh"
assert_fresh_path "${LOG_STDOUT}"
assert_fresh_path "${LOG_STDERR}"
assert_fresh_path "${LOG_SAMPLER}"
assert_fresh_path "${COMMAND_FILE}"

LAUNCH=(
  env
  "PYTHONNOUSERSITE=1"
  "TOKENIZERS_PARALLELISM=false"
  "CUDA_VISIBLE_DEVICES=${CLOUD_GPU}"
  "FASTSD_TARGET_DEVICE=cuda:0"
  "FASTSD_RESULTS_DIR=${RUN_DIR}"
  "CLOUD_SERVICE_PORT=${PORT}"
  "CLOUD_BIND_HOST=${BIND_HOST}"
  "${PYTHON_BIN}" "${REPO_ROOT}/cloud/cloud_service.py"
  --config "${CONFIG}" --method "${METHOD}" --run-id "${RUN_ID}" --run-dir "${RUN_DIR}"
  --cloud-physical-gpu "${CLOUD_GPU}"
  --bind-host "${BIND_HOST}" --port "${PORT}"
  --draft_model "${TOKENIZER_MODEL}" --target_model "${TARGET_MODEL}" --tokenizer-model "${TOKENIZER_MODEL}" --model-dtype bfloat16
  --token_budget 256 --max_num_seqs 2 --prefill_max_wait_cycles 2
  "${PROFILE_ARGS[@]}"
)
write_command_file "${COMMAND_FILE}" "${LAUNCH[@]}"
metadata_python attach-command --run-dir "${RUN_DIR}" --command-file "${COMMAND_FILE}"

# The finalizer on node1 crops this raw evidence using cloud-local timestamps
# returned in the canonical request metrics.  Keep the raw file immutable.
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
    # This is the exact sampler child PID created above; no unrelated process is touched.
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
  metadata_python finish --run-dir "${RUN_DIR}" --status "${status}" --exit-code "${RUN_EXIT_CODE}" || note "could not finalize cloud manifest"
  trap - EXIT
  exit "${RUN_EXIT_CODE}"
}
trap finish_wrapper EXIT

note "starting ${METHOD} cloud service on ${BIND_HOST}:${PORT}; physical GPU ${CLOUD_GPU} is cuda:0 in this process"
set +e
"${LAUNCH[@]}" > >(tee "${LOG_STDOUT}") 2> >(tee "${LOG_STDERR}" >&2)
RUN_EXIT_CODE=$?
set -e
