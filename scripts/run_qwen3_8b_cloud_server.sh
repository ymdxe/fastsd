#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FASTSD_PYTHON_BIN="${FASTSD_PYTHON:-/home/hdd/zhangh/envs/fastsd/bin/python}"
TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
DRAFT_TOKENIZER_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"
CLOUD_GPU="${FASTSD_CLOUD_GPU:-2}"
CLOUD_PORT="${CLOUD_SERVICE_PORT:-1597}"
CACHE_ROOT="${FASTSD_CACHE_ROOT:-/home/hdd/zhangh/cache}"
PIP_CACHE_DIR="${FASTSD_PIP_CACHE:-${CACHE_ROOT}/pip}"
HF_HOME="${FASTSD_HF_HOME:-${CACHE_ROOT}/huggingface}"
HUGGINGFACE_HUB_CACHE="${FASTSD_HF_HUB_CACHE:-${HF_HOME}/hub}"
TORCH_HOME="${FASTSD_TORCH_HOME:-${CACHE_ROOT}/torch}"
FASTSD_TMPDIR="${FASTSD_TMPDIR:-/home/hdd/zhangh/tmp/fastsd}"
RESULTS_DIR="${FASTSD_RESULTS_DIR:-${REPO_ROOT}/results}"

for required_path in "${FASTSD_PYTHON_BIN}" "${TARGET_MODEL}/config.json" "${DRAFT_TOKENIZER_MODEL}/tokenizer.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TORCH_HOME}" \
  "${FASTSD_TMPDIR}" \
  "${RESULTS_DIR}"

echo "Cloud target model: ${TARGET_MODEL}"
echo "Draft tokenizer: ${DRAFT_TOKENIZER_MODEL}"
echo "Physical GPU: ${CLOUD_GPU} (visible as cuda:0)"
echo "Listening port: ${CLOUD_PORT}"

exec env \
  PYTHONNOUSERSITE=1 \
  TOKENIZERS_PARALLELISM=false \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  HF_HOME="${HF_HOME}" \
  HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
  TORCH_HOME="${TORCH_HOME}" \
  TMPDIR="${FASTSD_TMPDIR}" \
  FASTSD_RESULTS_DIR="${RESULTS_DIR}" \
  CUDA_VISIBLE_DEVICES="${CLOUD_GPU}" \
  FASTSD_TARGET_DEVICE=cuda:0 \
  CLOUD_SERVICE_PORT="${CLOUD_PORT}" \
  "${FASTSD_PYTHON_BIN}" cloud/cloud_service.py \
    --profile custom \
    --server_sched_mode fastsd \
    --draft_model "${DRAFT_TOKENIZER_MODEL}" \
    --target_model "${TARGET_MODEL}" \
    --token_budget 128 \
    --max_num_seqs 1 \
    --prefill_max_wait_cycles 2 \
    --no-enable_pipeline \
    --exp_name qwen3_8b_cloud \
    "$@"
