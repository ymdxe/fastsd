#!/usr/bin/env bash
set -euo pipefail
SUITE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/home/hdd3/zhangh/workspace/fastsd
cd "$REPO"
if ss -ltn 'sport = :19635' | tail -n +2 | grep -q LISTEN; then
  echo 'Your experiment Target on port 19635 is still running. Stop it in its own terminal with Ctrl+C first.' >&2
  exit 1
fi
nvidia-smi
GPU_ID="${1:-}"
if [[ -z "$GPU_ID" ]]; then
  read -r -p 'Physical GPU ID allocated to this experiment: ' GPU_ID
fi
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'Expected one numeric physical GPU ID.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/home/hdd/zhangh/cache/huggingface TORCH_HOME=/home/hdd/zhangh/cache/torch
mkdir -p "$REPO/exp"
LOGROOT=$(mktemp -d "$REPO/exp/oracle8b_gate_$(date +%Y%m%d_%H%M%S)_XXXXXX")
cmd=(/home/hdd/zhangh/envs/new_fastsd/bin/python -B -u "$SUITE/check_oracle.py" --output "$LOGROOT/results")
printf 'CUDA_VISIBLE_DEVICES=%q HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ' "$GPU_ID" > "$LOGROOT/commands.txt"
printf '%q ' "${cmd[@]}" >> "$LOGROOT/commands.txt"
printf '\n' >> "$LOGROOT/commands.txt"
git rev-parse HEAD > "$LOGROOT/git-head.txt"
git status --short > "$LOGROOT/git-status.txt"
printf 'Diagnostic output: %s\n' "$LOGROOT"
set +e
"${cmd[@]}" 2>&1 | tee "$LOGROOT/diagnostic.log"
statuses=("${PIPESTATUS[@]}")
printf 'python_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$LOGROOT/exit-status.txt"
exit "${statuses[0]}"
