#!/usr/bin/env bash
# Create the independently locked Official SpecEdge runtime environment.
#
# This entrypoint deliberately performs every eligibility check before it asks
# uv to create anything.  It never replaces an existing environment and has
# no cleanup path: a failed uv invocation is left for its owner to inspect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

readonly DEFAULT_ENV_DIR="/home/hdd/zhangh/envs/specedge"
readonly OFFICIAL_DIR="${REPO_ROOT}/baselines/specedge/official"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/setup_specedge_locked_env.sh \
      --python /absolute/path/to/python3.14 [--uv /absolute/path/to/uv]

Creates only /home/hdd/zhangh/envs/specedge using the pinned Official
baselines/specedge/official/uv.lock.  The supplied Python must be an explicit
executable whose version is exactly 3.14.  The script refuses a missing uv,
missing/incorrect Python, dirty or uninitialised Official submodule, or an
existing environment before it invokes uv sync.  It never deletes or replaces
an environment.  The runtime environment excludes the optional dev group.
EOF
}

PYTHON_BIN=""
UV_BIN="uv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --uv) UV_BIN="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${PYTHON_BIN}" ]] || { usage >&2; die "--python must explicitly name a Python 3.14 executable."; }
[[ "${PYTHON_BIN}" == /* ]] || die "--python must be an absolute executable path, not a shell lookup."

# ---- All checks below are read-only.  Do not move uv sync above this line. ----
require_repo_root
require_directory "${OFFICIAL_DIR}"
require_file "${OFFICIAL_DIR}/pyproject.toml"
require_file "${OFFICIAL_DIR}/uv.lock"
require_executable_or_command "${UV_BIN}"
require_executable_or_command "${PYTHON_BIN}"
validate_hdd_path "${DEFAULT_ENV_DIR}"
assert_fresh_path "${DEFAULT_ENV_DIR}"
require_directory "$(dirname "${DEFAULT_ENV_DIR}")"
[[ -w "$(dirname "${DEFAULT_ENV_DIR}")" ]] || die "Environment parent is not writable: $(dirname "${DEFAULT_ENV_DIR}")"

official_status="$(git -C "${OFFICIAL_DIR}" status --short)"
[[ -z "${official_status}" ]] || die "Official SpecEdge worktree is dirty; refuse to lock an unreproducible environment."
submodule_state="$(git -C "${REPO_ROOT}" submodule status -- baselines/specedge/official)"
[[ "${submodule_state}" == " "* ]] || die "Official SpecEdge submodule is not initialized at its pinned commit: ${submodule_state:-missing}"

python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${python_version}" == "3.14" ]] || die "--python must be Python 3.14.x; found ${python_version}."

# `uv lock --check` is a read-only lock consistency gate.  `--frozen` below
# guarantees that uv sync cannot refresh or rewrite the committed lock file.
# Run it from the project so this works across supported uv CLI versions.
(
  cd "${OFFICIAL_DIR}"
  "${UV_BIN}" lock --check
)

note "all read-only gates passed; creating ${DEFAULT_ENV_DIR} from the pinned uv.lock"
(
  cd "${OFFICIAL_DIR}"
  UV_PROJECT_ENVIRONMENT="${DEFAULT_ENV_DIR}" "${UV_BIN}" sync --frozen --no-dev --python "${PYTHON_BIN}"
)

require_executable_or_command "${DEFAULT_ENV_DIR}/bin/python"
"${DEFAULT_ENV_DIR}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 14); import grpc, torch, transformers'
note "SpecEdge locked runtime is ready at ${DEFAULT_ENV_DIR}; set SPECEDGE_PYTHON=${DEFAULT_ENV_DIR}/bin/python when needed."
