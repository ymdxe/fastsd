#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="${1:-both}"
EXP_NAME="${2:-profile_${PROFILE}}"
shift 2 || true

case "$PROFILE" in
  vanilla|proactive_only|pipeline_only|both)
    ;;
  *)
    echo "Unsupported profile: $PROFILE"
    echo "Usage: bash scripts/run_vanilla_profile.sh {vanilla|proactive_only|pipeline_only|both} [exp_name] [extra args...]"
    exit 1
    ;;
esac

COMMON_ARGS=(--profile "$PROFILE" --exp_name "$EXP_NAME" --server_sched_mode "vanilla" "$@")

python edge/edge.py --server_url "${SERVER_URL:-http://127.0.0.1:8001}" "${COMMON_ARGS[@]}"

echo "Done. Metrics summary: $(pwd)/exp/${EXP_NAME}/edge_metrics_summary.json"
