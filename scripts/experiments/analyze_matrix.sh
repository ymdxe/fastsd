#!/usr/bin/env bash
# Aggregate only finalized complete runs into a new analysis directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  echo "Usage: bash scripts/experiments/analyze_matrix.sh --run-root /home/hdd/... --output /home/hdd/... [--slo-e2e-ms N]" >&2
}

RUN_ROOT=""
OUTPUT=""
PYTHON_BIN="${FASTSD_METADATA_PYTHON:-${FASTSD_PYTHON:-python3}}"
SLO_E2E_MS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --slo-e2e-ms) SLO_E2E_MS="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
[[ -n "${RUN_ROOT}" && -n "${OUTPUT}" ]] || { usage; die "--run-root and --output are required."; }
validate_hdd_path "${RUN_ROOT}"
validate_hdd_path "${OUTPUT}"
require_directory "${RUN_ROOT}"
assert_fresh_path "${OUTPUT}"
require_executable_or_command "${PYTHON_BIN}"
TOOL="${SCRIPT_DIR}/analyze_matrix.py"
require_file "${TOOL}"
if [[ -n "${SLO_E2E_MS}" ]]; then
  [[ "${SLO_E2E_MS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "--slo-e2e-ms must be positive."
  awk -v value="${SLO_E2E_MS}" 'BEGIN { exit !(value > 0) }' || die "--slo-e2e-ms must be positive."
fi
COMMAND=("${PYTHON_BIN}" "${TOOL}" --run-root "${RUN_ROOT}" --output "${OUTPUT}")
if [[ -n "${SLO_E2E_MS}" ]]; then COMMAND+=(--slo-e2e-ms "${SLO_E2E_MS}"); fi
"${COMMAND[@]}"
