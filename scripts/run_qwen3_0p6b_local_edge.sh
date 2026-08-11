#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FASTSD_PYTHON_BIN="${FASTSD_PYTHON:-python}"
DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-${REPO_ROOT}/../models/Qwen3-0.6B}"
CLOUD_URL="${SERVER_URL:-http://39.102.209.27:1597}"

if [[ ! -f "${DRAFT_MODEL}/config.json" ]]; then
  echo "Qwen3-0.6B was not found at ${DRAFT_MODEL}." >&2
  echo "Set FASTSD_DRAFT_MODEL to the local model directory." >&2
  exit 1
fi

echo "Local draft model: ${DRAFT_MODEL}"
echo "Cloud URL: ${CLOUD_URL}"

exec env PYTHONNOUSERSITE=1 \
  "${FASTSD_PYTHON_BIN}" edge/edge.py \
    --server_url "${CLOUD_URL}" \
    --profile custom \
    --server_sched_mode fastsd \
    --draft_model "${DRAFT_MODEL}" \
    --target_model qwen3-8b \
    --edge_gpu_start 0 \
    --edge_gpus 1 \
    --num_drafts 1 \
    --max_tasks_per_draft 1 \
    --dataset humaneval \
    --data_path ./data \
    --max_tokens 32 \
    --gamma 6 \
    --token_budget 128 \
    --max_num_seqs 1 \
    --prefill_max_wait_cycles 2 \
    --no-enable_pipeline \
    --no-enable_proactive_draft \
    --exp_name qwen3_0p6b_8b_smoke \
    "$@"
