#!/bin/bash
set -euo pipefail

ENV_PREFIX="${FASTSD_CONDA_ENV:-/home/hdd/zhangh/envs/fastsd}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
CACHE_ROOT="${FASTSD_CACHE_ROOT:-/home/hdd/zhangh/cache}"
PIP_CACHE_DIR="${FASTSD_PIP_CACHE:-${CACHE_ROOT}/pip}"
HF_HOME="${FASTSD_HF_HOME:-${CACHE_ROOT}/huggingface}"
HUGGINGFACE_HUB_CACHE="${FASTSD_HF_HUB_CACHE:-${HF_HOME}/hub}"
TORCH_HOME="${FASTSD_TORCH_HOME:-${CACHE_ROOT}/torch}"
TMPDIR="${FASTSD_TMPDIR:-/home/hdd/zhangh/tmp/fastsd}"
RESULTS_DIR="${FASTSD_RESULTS_DIR:-/home/hdd/zhangh/workspace/fastsd/results}"
TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"
CLOUD_GPU="${FASTSD_CLOUD_GPU:-2}"
CLOUD_PORT="${CLOUD_SERVICE_PORT:-1597}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Conda environment Python was not found: ${PYTHON_BIN}" >&2
  echo "Create it with:" >&2
  echo "  /home/hdd/zhangh/miniforge3/bin/conda create --prefix ${ENV_PREFIX} python=3.10 pip -y" >&2
  exit 1
fi

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TORCH_HOME}" \
  "${TMPDIR}" \
  "${RESULTS_DIR}"
export \
  PIP_CACHE_DIR \
  HF_HOME \
  HUGGINGFACE_HUB_CACHE \
  TORCH_HOME \
  TMPDIR \
  PYTHONNOUSERSITE=1

/home/hdd/zhangh/miniforge3/bin/conda env config vars set \
  -p "${ENV_PREFIX}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  HF_HOME="${HF_HOME}" \
  HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
  TORCH_HOME="${TORCH_HOME}" \
  TMPDIR="${TMPDIR}" \
  FASTSD_RESULTS_DIR="${RESULTS_DIR}" \
  FASTSD_TARGET_MODEL="${TARGET_MODEL}" \
  FASTSD_DRAFT_MODEL="${DRAFT_MODEL}" \
  FASTSD_CLOUD_GPU="${CLOUD_GPU}" \
  CLOUD_SERVICE_PORT="${CLOUD_PORT}" \
  PYTHONNOUSERSITE=1 \
  TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m pip install \
  torch==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121

"${PYTHON_BIN}" -m pip install \
  transformers==4.57.6 \
  accelerate==1.12.0 \
  numpy==1.26.4 \
  requests==2.32.5 \
  fastapi==0.128.0 \
  uvicorn==0.40.0 \
  nvidia-ml-py3 \
  pytest

TARGET_MODEL="${TARGET_MODEL}" DRAFT_MODEL="${DRAFT_MODEL}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import torch
import transformers

target = Path(os.environ["TARGET_MODEL"])
draft = Path(os.environ["DRAFT_MODEL"])
for path in (target / "config.json", draft / "tokenizer.json"):
    if not path.is_file():
        raise SystemExit(f"missing required model file: {path}")

if not torch.cuda.is_available():
    raise SystemExit("PyTorch is installed, but CUDA is not available")

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
print("FastSD cloud environment is ready")
PY

echo "Conda env: ${ENV_PREFIX}"
echo "Cache root: ${CACHE_ROOT}"
echo "Temporary files: ${TMPDIR}"
echo "Experiment results: ${RESULTS_DIR}"
