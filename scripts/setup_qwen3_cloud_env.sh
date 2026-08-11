#!/bin/bash
set -euo pipefail

ENV_PREFIX="${FASTSD_CONDA_ENV:-/home/hdd/zhangh/envs/fastsd}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
PIP_CACHE_DIR="${FASTSD_PIP_CACHE:-/home/hdd/zhangh/cache/pip}"
TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Conda environment Python was not found: ${PYTHON_BIN}" >&2
  echo "Create it with:" >&2
  echo "  /home/hdd/zhangh/miniforge3/bin/conda create --prefix ${ENV_PREFIX} python=3.10 pip -y" >&2
  exit 1
fi

mkdir -p "${PIP_CACHE_DIR}"
export PIP_CACHE_DIR PYTHONNOUSERSITE=1

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
