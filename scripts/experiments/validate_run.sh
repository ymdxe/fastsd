#!/usr/bin/env bash
# Read-only validation wrapper.  It intentionally writes no marker file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  echo "Usage: bash scripts/experiments/validate_run.sh --run-dir /home/hdd/.../RUN_ID" >&2
}

RUN_DIR=""
PYTHON_BIN="${FASTSD_METADATA_PYTHON:-${FASTSD_PYTHON:-python3}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
[[ -n "${RUN_DIR}" ]] || { usage; die "--run-dir is required."; }
validate_hdd_path "${RUN_DIR}"
require_directory "${RUN_DIR}"
require_executable_or_command "${PYTHON_BIN}"
TOOL="${SCRIPT_DIR}/validate_experiment_run.py"
require_file "${TOOL}"
"${PYTHON_BIN}" "${TOOL}" --run-dir "${RUN_DIR}"
