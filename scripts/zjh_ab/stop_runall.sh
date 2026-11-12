#!/usr/bin/env bash
# Run on node1 CPU: bash scripts/zjh_ab/stop_runall.sh [RUN_TAG]
# Saved job records allow cleanup even after the runner was killed with SIGKILL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_ROOT="${RUNALL_STATE_ROOT:-$ROOT/runtime/runall}"
JOBS_ONLY=0
if [[ "${1:-}" == --jobs-only ]]; then
  JOBS_ONLY=1
  STATE_DIR="${2:?state directory required}"
else
  TAG="${1:-}"
  if [[ -z "$TAG" ]]; then
    [[ -f "$STATE_ROOT/latest" ]] || {
      echo "No managed run recorded; old runners need one-time manual cleanup." >&2
      echo "Local residual candidates:" >&2
      pgrep -af 'new_new_fastsd.*(cloud_service\.py|edge\.py|managed_job\.py|runall\.sh|ssh_tunnel_guard\.sh)|ssh .*18003' || echo "  none" >&2
      exit 1
    }
    TAG=$(cat "$STATE_ROOT/latest")
  fi
  [[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ && "$TAG" != . && "$TAG" != .. ]] || exit 2
  STATE_DIR="$STATE_ROOT/$TAG"
fi
[[ -f "$STATE_DIR/config.env" ]] || {
  echo "No job registry: $STATE_DIR" >&2
  echo "Local residual candidates:" >&2
  pgrep -af 'new_new_fastsd.*(cloud_service\.py|edge\.py|managed_job\.py|runall\.sh|ssh_tunnel_guard\.sh)|ssh .*18003' || echo "  none" >&2
  exit 1
}
source "$STATE_DIR/config.env"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
# Stop the controller first so it cannot launch another round during cleanup.
if [[ "$JOBS_ONLY" == 0 ]]; then
  "$LOCAL_PY" "$ROOT/scripts/zjh_ab/managed_job.py" stop "$STATE_DIR/controller.json" --grace 3
fi
failed=0
if [[ -f "$STATE_DIR/jobs.tsv" ]]; then
  while IFS=$'\t' read -r host py helper state; do
    [[ -n "$host" ]] || continue
    echo "stop $host $state" >&2
    if [[ "$host" == local ]]; then
      "$py" "$helper" stop "$state" --grace 8 || failed=1
    else
      printf -v command '%q ' "$py" "$helper" stop "$state" --grace 8
      ssh "${SSH_OPTS[@]}" "$host" "$command" || failed=1
    fi
  done < <(tac "$STATE_DIR/jobs.tsv")
fi
if [[ "$failed" == 0 ]]; then
  echo "Recorded jobs stopped: $STATE_DIR"
else
  echo "Cleanup incomplete (host unreachable or process still alive); rerun this stop command. Records retained." >&2
fi

# Managed jobs do not cover processes started manually or by older runners.
pattern='new_new_fastsd.*(cloud_service\.py|edge\.py|managed_job\.py|runall\.sh|ssh_tunnel_guard\.sh)|ssh .*18003'
echo "Local residual candidates:" >&2
pgrep -af "$pattern" || echo "  none" >&2
for host in ${CHECK_HOSTS:-}; do
  echo "Residual candidates on $host:" >&2
  ssh "${SSH_OPTS[@]}" "$host" "pgrep -af '$pattern' || echo '  none'" || true
done
exit "$failed"
