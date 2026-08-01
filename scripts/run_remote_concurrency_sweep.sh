#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <model_path> <output_jsonl> [gpu]"
  exit 1
fi

MODEL_PATH="$1"
OUTPUT_JSONL="$2"
GPU="${3:-0}"

CONCURRENCIES=(1 8 16 24 32 40 48 56 64)
REMOTE_ENV="/vepfs_hyh/hyh/.venvs/fastsd-throughput-env"
REMOTE_REPO="/vepfs_hyh/hyh/FastSD"

rm -f "$OUTPUT_JSONL"

for n in "${CONCURRENCIES[@]}"; do
  ssh voc_gpu1 "cd ${REMOTE_REPO} && source ${REMOTE_ENV}/bin/activate && python scripts/benchmark_single_gpu_concurrency.py --model-path ${MODEL_PATH} --gpu ${GPU} --concurrency ${n} --measure-runs 2 --max-new-tokens 64" >> "$OUTPUT_JSONL"
done
