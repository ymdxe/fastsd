#!/usr/bin/env bash
# Collect immutable cloud artifacts without overwriting or deleting anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/finalize_run.sh \
      --run-dir /home/hdd/.../RUN_ID --cloud-host node2 \
      --cloud-dir /home/hdd/.../RUN_ID/cloud

The local run root must already contain edge/.  The cloud/ destination and
root-level manifest/config/workload/provenance directories must not exist.  A
Rsync is invoked with --ignore-existing only; this script does not remove
remote or local experiment artifacts.
EOF
}

RUN_DIR=""
CLOUD_HOST=""
CLOUD_DIR=""
SSH_BIN="${FASTSD_SSH_BIN:-ssh}"
RSYNC_BIN="${FASTSD_RSYNC_BIN:-rsync}"
PYTHON_BIN="${FASTSD_METADATA_PYTHON:-${FASTSD_PYTHON:-python3}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --cloud-host) CLOUD_HOST="${2:-}"; shift 2 ;;
    --cloud-dir) CLOUD_DIR="${2:-}"; shift 2 ;;
    --ssh) SSH_BIN="${2:-}"; shift 2 ;;
    --rsync) RSYNC_BIN="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${RUN_DIR}" && -n "${CLOUD_HOST}" && -n "${CLOUD_DIR}" ]] || { usage >&2; die "Missing required arguments."; }
validate_hdd_path "${RUN_DIR}"
validate_hdd_path "${CLOUD_DIR}"
validate_host_alias "${CLOUD_HOST}"
require_executable_or_command "${SSH_BIN}"
require_executable_or_command "${RSYNC_BIN}"
require_executable_or_command "${PYTHON_BIN}"
CROP_TOOL="${SCRIPT_DIR}/crop_gpu_samples.py"
require_file "${CROP_TOOL}"
require_directory "${RUN_DIR}"
require_file "${RUN_DIR}/edge/manifest.json"
assert_fresh_path "${RUN_DIR}/cloud"
assert_fresh_path "${RUN_DIR}/manifest.json"
assert_fresh_path "${RUN_DIR}/config"
assert_fresh_path "${RUN_DIR}/workload"
assert_fresh_path "${RUN_DIR}/provenance"
assert_fresh_path "${RUN_DIR}/metrics"
assert_fresh_path "${RUN_DIR}/outputs"

note "checking remote cloud artifacts on ${CLOUD_HOST}"
"${SSH_BIN}" "${CLOUD_HOST}" "test -d -- '${CLOUD_DIR}' && test -f -- '${CLOUD_DIR}/manifest.json'"
REMOTE_MANIFEST_SHA="$(${SSH_BIN} "${CLOUD_HOST}" "sha256sum -- '${CLOUD_DIR}/manifest.json' | awk '{print \$1}'")"
[[ "${REMOTE_MANIFEST_SHA}" =~ ^[0-9a-fA-F]{64}$ ]] || die "Remote cloud manifest did not produce a SHA-256 checksum."

mkdir -- "${RUN_DIR}/cloud"
note "copying cloud component with rsync --ignore-existing"
"${RSYNC_BIN}" -a --ignore-existing -- "${CLOUD_HOST}:${CLOUD_DIR}/" "${RUN_DIR}/cloud/"
require_file "${RUN_DIR}/cloud/manifest.json"
LOCAL_MANIFEST_SHA="$(safe_sha256 "${RUN_DIR}/cloud/manifest.json")"
[[ "${LOCAL_MANIFEST_SHA}" == "${REMOTE_MANIFEST_SHA}" ]] || die "Copied cloud manifest SHA does not match the remote source."

require_file "${RUN_DIR}/edge/metrics/requests.jsonl"
require_file "${RUN_DIR}/edge/metrics/summary.json"
require_file "${RUN_DIR}/edge/metrics/gpu_samples_raw.csv"
require_file "${RUN_DIR}/edge/outputs/completions.jsonl"
require_file "${RUN_DIR}/cloud/metrics/gpu_samples_raw.csv"

mkdir -- "${RUN_DIR}/config" "${RUN_DIR}/workload" "${RUN_DIR}/provenance" "${RUN_DIR}/metrics" "${RUN_DIR}/outputs"
require_file "${RUN_DIR}/edge/config/resolved.yaml"
require_file "${RUN_DIR}/edge/workload/arrival_trace.jsonl"
require_file "${RUN_DIR}/edge/workload/manifest.json"
require_file "${RUN_DIR}/edge/workload/prompts.jsonl"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/config/resolved.yaml" "${RUN_DIR}/config/resolved.yaml"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/workload/arrival_trace.jsonl" "${RUN_DIR}/workload/arrival_trace.jsonl"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/workload/manifest.json" "${RUN_DIR}/workload/manifest.json"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/workload/prompts.jsonl" "${RUN_DIR}/workload/prompts.jsonl"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/metrics/requests.jsonl" "${RUN_DIR}/metrics/requests.jsonl"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/metrics/summary.json" "${RUN_DIR}/metrics/summary.json"
cp --preserve=mode,timestamps --no-clobber -- "${RUN_DIR}/edge/outputs/completions.jsonl" "${RUN_DIR}/outputs/completions.jsonl"

# A completed edge wrapper normally creates this canonical file before its
# manifest is finalized.  Recreate it only for a crash-recovered component
# that has a complete request artifact plus immutable raw samples; never
# overwrite an existing result.
if [[ ! -e "${RUN_DIR}/edge/metrics/gpu_samples.csv" && ! -L "${RUN_DIR}/edge/metrics/gpu_samples.csv" ]]; then
  "${PYTHON_BIN}" "${CROP_TOOL}" \
    --raw "${RUN_DIR}/edge/metrics/gpu_samples_raw.csv" \
    --records "${RUN_DIR}/edge/metrics/requests.jsonl" \
    --start-field actual_arrival_monotonic_s \
    --end-field completion_monotonic_s \
    --output "${RUN_DIR}/edge/metrics/gpu_samples.csv"
fi
require_file "${RUN_DIR}/edge/metrics/gpu_samples.csv"

# Component samplers keep startup/warmup observations in gpu_samples_raw.csv.
# Create the cloud canonical CSV from cloud-host-local request/event timestamps
# only; no node1/node2 monotonic values are ever compared.
METHOD="$("${PYTHON_BIN}" - "${RUN_DIR}/edge/manifest.json" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
method = payload.get("method")
if method not in {"fastsd", "vanilla", "specedge"}:
    raise SystemExit(f"unsupported experiment method in edge manifest: {method!r}")
print(method)
PY
)"
if [[ "${METHOD}" == "fastsd" || "${METHOD}" == "vanilla" ]]; then
  "${PYTHON_BIN}" "${CROP_TOOL}" \
    --raw "${RUN_DIR}/cloud/metrics/gpu_samples_raw.csv" \
    --records "${RUN_DIR}/edge/metrics/requests.jsonl" \
    --start-field cloud_measurement_start_monotonic_s \
    --end-field cloud_measurement_end_monotonic_s \
    --output "${RUN_DIR}/cloud/metrics/gpu_samples.csv"
else
  # The transport-only SpecEdge adapter records these cloud-local RPC bounds;
  # a missing event artifact is a hard failure rather than a guessed J/token.
  require_file "${RUN_DIR}/cloud/metrics/specedge_server_events.jsonl"
  "${PYTHON_BIN}" "${CROP_TOOL}" \
    --raw "${RUN_DIR}/cloud/metrics/gpu_samples_raw.csv" \
    --records "${RUN_DIR}/cloud/metrics/specedge_server_events.jsonl" \
    --event-timestamp-field server_monotonic_s \
    --event-method Validate \
    --output "${RUN_DIR}/cloud/metrics/gpu_samples.csv"
fi

MERGE_TOOL="${SCRIPT_DIR}/merge_gpu_samples.py"
require_file "${MERGE_TOOL}"
"${PYTHON_BIN}" "${MERGE_TOOL}" \
  --edge "${RUN_DIR}/edge/metrics/gpu_samples.csv" \
  --cloud "${RUN_DIR}/cloud/metrics/gpu_samples.csv" \
  --output "${RUN_DIR}/metrics/gpu_samples.csv"

shopt -s nullglob
EDGE_PROVENANCE=("${RUN_DIR}/edge/provenance/"*.json)
CLOUD_PROVENANCE=("${RUN_DIR}/cloud/provenance/"*.json)
[[ ${#EDGE_PROVENANCE[@]} -eq 1 ]] || die "Expected exactly one edge provenance JSON file."
[[ ${#CLOUD_PROVENANCE[@]} -eq 1 ]] || die "Expected exactly one cloud provenance JSON file."
EDGE_PROVENANCE_DEST="${RUN_DIR}/provenance/$(basename "${EDGE_PROVENANCE[0]}")"
CLOUD_PROVENANCE_DEST="${RUN_DIR}/provenance/$(basename "${CLOUD_PROVENANCE[0]}")"
[[ "${EDGE_PROVENANCE_DEST}" != "${CLOUD_PROVENANCE_DEST}" ]] || die "Edge and cloud provenance use the same file name; refuse to overwrite either."
cp --preserve=mode,timestamps --no-clobber -- "${EDGE_PROVENANCE[0]}" "${EDGE_PROVENANCE_DEST}"
cp --preserve=mode,timestamps --no-clobber -- "${CLOUD_PROVENANCE[0]}" "${CLOUD_PROVENANCE_DEST}"

metadata_python finalize-root --run-dir "${RUN_DIR}"
echo "FINALIZE_OK run_dir=${RUN_DIR}"
