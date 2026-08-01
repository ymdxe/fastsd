#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXP_NAME="${1:-fastsd_run}"
shift 1 || true

COMMON_ARGS=(
  --profile custom
  --server_sched_mode fastsd
  --enable_proactive_draft
  --no-enable_pipeline
  --exp_name "$EXP_NAME"
  "$@"
)

python edge/edge.py --server_url "${SERVER_URL:-http://127.0.0.1:8001}" "${COMMON_ARGS[@]}"

echo "Done. Metrics summary: $(pwd)/exp/${EXP_NAME}/edge_metrics_summary.json"
