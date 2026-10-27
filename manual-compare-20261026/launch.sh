#!/usr/bin/env bash
set -euo pipefail
SUITE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NODE=$(hostname -s)
case "$NODE" in
  gpunode2*) PY=/home/hdd/zhangh/envs/new_fastsd/bin/python; REPO=/home/hdd3/zhangh/workspace/fastsd; GPU_HOST=1 ;;
  node1) PY=/home/zhangh/miniconda3/envs/fastsd/bin/python; REPO=/home/zhangh/workspace/fastsd; GPU_HOST=0 ;;
  node2) PY=/home/zhangh/venvs/fastsd_cpu/bin/python; REPO=/home/zhangh/workspace/fastsd; GPU_HOST=0 ;;
  node3) PY=/home/zhangh/envs/fastsd_cpu/bin/python; REPO=/home/zhangh/workspace/fastsd; GPU_HOST=0 ;;
  *) echo "Unconfigured host: $NODE" >&2; exit 1 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$SUITE:$SUITE/runtime${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=127.0.0.1,localhost,219.216.65.42
export no_proxy="$NO_PROXY"
if [[ "$GPU_HOST" == 1 ]]; then
  export HF_HOME=/home/hdd/zhangh/cache/huggingface
  export TORCH_HOME=/home/hdd/zhangh/cache/torch
fi
ACTION=${1:-check}
shift || true
case "$ACTION" in
  tunnel)
    [[ "$GPU_HOST" == 0 ]] || { echo 'Run tunnel on a CPU host.' >&2; exit 1; }
    exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L 127.0.0.1:29635:219.216.65.42:19635 zhangh@219.216.65.42
    ;;
  server|oracle)
    [[ "$GPU_HOST" == 1 ]] || { echo 'Run this command on gpunode2.' >&2; exit 1; }
    if [[ "$ACTION" == server ]]; then
      [[ $# -ge 3 ]] || { echo 'Usage: launch.sh server ar|sd|fastsd single|parity|pilot|low_rep1 RUN_ID [GPU_ID]' >&2; exit 1; }
      GPU_ID=${4:-}
    else
      GPU_ID=${1:-}
    fi
    "$PY" -B -c 'import socket; s=socket.socket(); s.bind(("219.216.65.42",19635)); s.close()' || {
      echo 'Port 19635 is occupied. Inspect its owner; stop only your experiment in its original terminal.' >&2; exit 1; }
    nvidia-smi
    if [[ -z "$GPU_ID" ]]; then
      read -r -p '请输入已分配给本实验的物理 GPU 编号: ' GPU_ID
    fi
    [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'Expected one numeric GPU ID.' >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    mkdir -p "$REPO/exp/manual_compare_20261026"
    if [[ "$ACTION" == oracle ]]; then
      LOGROOT=$(mktemp -d "$REPO/exp/manual_compare_20261026/oracle_$(date +%Y%m%d_%H%M%S)_XXXXXX")
      "$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["bundle_id"])' "$SUITE/manifest.json" > "$LOGROOT/bundle-id.txt"
      cmd=("$PY" -B -u "$SUITE/reference/check_oracle.py" --repo "$SUITE/runtime" --output "$LOGROOT/results")
    else
      [[ "$3" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || { echo 'Invalid run ID.' >&2; exit 1; }
      LOGROOT="$REPO/exp/manual_compare_20261026/$3"
      [[ ! -e "$LOGROOT" ]] || { echo 'Run ID already exists; use a new ID.' >&2; exit 1; }
      cmd=("$PY" -B -u "$SUITE/suite.py" server "$1" "$2" "$3")
    fi
    echo "Output: $LOGROOT"
    # Console logs live beside run directories; the Python entry creates the run exclusively.
    CONSOLE="$LOGROOT.console.log"
    [[ ! -e "$CONSOLE" ]] || { echo 'Console log already exists; use a new run ID.' >&2; exit 1; }
    set +e
    "${cmd[@]}" 2>&1 | tee "$CONSOLE"
    statuses=("${PIPESTATUS[@]}")
    set -e
    if [[ -d "$LOGROOT" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q NVIDIA_TF32_OVERRIDE=0 ' "$GPU_ID" > "$LOGROOT/commands.txt"
      printf '%q ' "${cmd[@]}" >> "$LOGROOT/commands.txt"
      printf '\n' >> "$LOGROOT/commands.txt"
      printf 'python_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$LOGROOT/exit-status.txt"
    fi
    exit "${statuses[0]}"
    ;;
  status)
    if [[ "$GPU_HOST" == 1 ]]; then
      exec "$PY" -B "$SUITE/suite.py" status --server-url http://219.216.65.42:19635 "$@"
    fi
    exec "$PY" -B "$SUITE/suite.py" status "$@"
    ;;
  check|clients|summary|gate)
    exec "$PY" -B -u "$SUITE/suite.py" "$ACTION" "$@"
    ;;
  *) echo 'Actions: check, tunnel, oracle, server, clients, status, summary, gate' >&2; exit 1 ;;
esac