#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NODE=$(hostname -s)
case "$NODE" in
  gpunode2*)
    PY=/home/hdd/zhangh/envs/specedge/bin/python
    export SPECEDGE_ROLE=gpu
    export SPECEDGE_RESULT_ROOT=/home/hdd3/zhangh/workspace/fastsd/exp/specedge_manual_20261026
    export HF_HOME=/home/hdd/zhangh/cache/huggingface TORCH_HOME=/home/hdd/zhangh/cache/torch
    ;;
  node1)
    PY=/home/zhangh/miniconda3/envs/fastsd/bin/python
    export SPECEDGE_ROLE=cpu SPECEDGE_DRAFT_MODEL_PATH=/mnt/nfs0/zhangh/models/Qwen3-0.6B
    export SPECEDGE_RESULT_ROOT=/home/zhangh/workspace/fastsd/exp/specedge_manual_20261026
    ;;
  node2)
    PY=/home/zhangh/venvs/fastsd_cpu/bin/python
    export SPECEDGE_ROLE=cpu SPECEDGE_DRAFT_MODEL_PATH=/vol1/zhangh/models/Qwen3-0.6B
    export SPECEDGE_RESULT_ROOT=/home/zhangh/workspace/fastsd/exp/specedge_manual_20261026
    export PYTHONPATH=/home/zhangh/workspace/fastsd/tmp/specedge-cpu-gpu-20261026-deps
    ;;
  node3)
    PY=/home/zhangh/envs/fastsd_cpu_py314/bin/python314_glibc
    export SPECEDGE_ROLE=cpu SPECEDGE_DRAFT_MODEL_PATH=/mnt/nfs0/zhangh/models/Qwen3-1.7B
    export SPECEDGE_RESULT_ROOT=/home/zhangh/workspace/fastsd/exp/specedge_manual_20261026
    ;;
  *) echo "Unconfigured host: $NODE" >&2; exit 1 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$BUNDLE:$BUNDLE/runtime${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
ACTION=${1:-check}
if [[ "$ACTION" == check || "$ACTION" == status || "$ACTION" == --help ]]; then
  exec "$PY" -B -u "$BUNDLE/run.py" "${@:-check}"
fi
[[ "$ACTION" == server || "$ACTION" == client ]] || { echo 'Actions: check, server, client, status' >&2; exit 1; }
[[ $# -ge 2 && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || { echo 'Specify a new run ID.' >&2; exit 1; }
mkdir -p "$SPECEDGE_RESULT_ROOT"
CONSOLE="$SPECEDGE_RESULT_ROOT/$2.$ACTION.$NODE.console.log"
[[ ! -e "$CONSOLE" ]] || { echo "Log already exists: $CONSOLE. Use a new run ID." >&2; exit 1; }
set +e
"$PY" -B -u "$BUNDLE/run.py" "$@" 2>&1 | tee "$CONSOLE"
STATUSES=("${PIPESTATUS[@]}")
set -e
if [[ "${STATUSES[0]}" != 0 ]]; then exit "${STATUSES[0]}"; fi
exit "${STATUSES[1]}"
