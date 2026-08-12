#!/usr/bin/env bash
# Generate one immutable arrival trace which every method reuses.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/build_workload.sh \
      --dataset mt_bench --data PATH --tokenizer-model PATH \
      --max-requests N --arrival-mode poisson --request-rate-rps RATE \
      --arrival-seed SEED --output /home/hdd/...

The output path must not exist.  The generated trace, rendered prompts, and
workload manifest are immutable inputs for FastSD, Vanilla, and SpecEdge.
EOF
}

DATASET=""
DATA_PATH=""
TOKENIZER_MODEL=""
MAX_REQUESTS=""
ARRIVAL_MODE=""
REQUEST_RATE=""
ARRIVAL_SEED=""
NUM_CLIENTS=2
OUTPUT=""
TRACE_PYTHON="${FASTSD_TRACE_PYTHON:-${FASTSD_PYTHON:-python3}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --data) DATA_PATH="${2:-}"; shift 2 ;;
    --tokenizer-model) TOKENIZER_MODEL="${2:-}"; shift 2 ;;
    --max-requests) MAX_REQUESTS="${2:-}"; shift 2 ;;
    --arrival-mode) ARRIVAL_MODE="${2:-}"; shift 2 ;;
    --request-rate-rps) REQUEST_RATE="${2:-}"; shift 2 ;;
    --arrival-seed) ARRIVAL_SEED="${2:-}"; shift 2 ;;
    --num-clients) NUM_CLIENTS="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --python) TRACE_PYTHON="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "${DATASET}" =~ ^(mt_bench|humaneval|gsm8k)$ ]] || die "--dataset must be mt_bench, humaneval, or gsm8k."
[[ -n "${DATA_PATH}" && -n "${TOKENIZER_MODEL}" && -n "${MAX_REQUESTS}" && -n "${ARRIVAL_MODE}" && -n "${ARRIVAL_SEED}" && -n "${OUTPUT}" ]] || { usage >&2; die "Missing required arguments."; }
[[ "${MAX_REQUESTS}" =~ ^[1-9][0-9]*$ ]] || die "--max-requests must be a positive integer."
[[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]] || die "--num-clients must be a positive integer."
[[ "${ARRIVAL_SEED}" =~ ^[0-9]+$ ]] || die "--arrival-seed must be a non-negative integer."
[[ "${ARRIVAL_MODE}" == "poisson" ]] || die "build_workload.sh creates immutable Poisson traces only; use the edge CLI directly for closed_loop pilots."
[[ "${REQUEST_RATE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--request-rate-rps must be a positive number for Poisson arrivals."
awk -v rate="${REQUEST_RATE}" 'BEGIN { exit !(rate > 0) }' || die "--request-rate-rps must be greater than zero."

if [[ "${DATA_PATH}" != /* ]]; then
  DATA_PATH="${REPO_ROOT}/${DATA_PATH}"
fi
if [[ "${TOKENIZER_MODEL}" != /* ]]; then
  TOKENIZER_MODEL="${REPO_ROOT}/${TOKENIZER_MODEL}"
fi
require_file "${DATA_PATH}"
require_file "${TOKENIZER_MODEL}/config.json"
require_file "${TOKENIZER_MODEL}/tokenizer.json"
validate_hdd_path "${OUTPUT}"
assert_fresh_path "${OUTPUT}"
require_executable_or_command "${TRACE_PYTHON}"
TRACE_TOOL="${SCRIPT_DIR}/make_poisson_trace.py"
require_file "${TRACE_TOOL}"

case "${DATASET}" in
  mt_bench) DATASET_FORMAT="mt_bench_first_turn_qwen3" ;;
  humaneval) DATASET_FORMAT="humaneval_prompt" ;;
  gsm8k) DATASET_FORMAT="gsm8k_question" ;;
esac

command=(
  "${TRACE_PYTHON}" "${TRACE_TOOL}"
  --dataset "${DATA_PATH}"
  --dataset-format "${DATASET_FORMAT}"
  --tokenizer-model "${TOKENIZER_MODEL}"
  --max-requests "${MAX_REQUESTS}"
  --rate-rps "${REQUEST_RATE}"
  --seed "${ARRIVAL_SEED}"
  --num-clients "${NUM_CLIENTS}"
  --output "${OUTPUT}"
)

note "generating immutable ${ARRIVAL_MODE} trace under ${OUTPUT}"
"${command[@]}"

require_file "${OUTPUT}/arrival_trace.jsonl"
require_file "${OUTPUT}/prompts.jsonl"
require_file "${OUTPUT}/manifest.json"
echo "WORKLOAD_READY output=${OUTPUT} trace_sha256=$(safe_sha256 "${OUTPUT}/arrival_trace.jsonl")"
