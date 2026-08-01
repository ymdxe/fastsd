#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed 's/ //g' | sort -u)
  for pid in "${pids[@]}"; do
    [[ -n "$pid" ]] || continue
    kill -9 "$pid" 2>/dev/null || true
  done
fi

sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
