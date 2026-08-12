#!/usr/bin/env bash
# Read-only admission check for a single experiment allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/preflight.sh --role edge|cloud --physical-gpus IDS \
      --run-root /home/hdd/... --min-free-gib N [options]

Required:
  --role ROLE                 edge requires exactly two GPUs; cloud exactly one.
  --physical-gpus IDS        Comma-separated physical GPU indexes.
  --run-root PATH            Proposed result root, explicitly under /home/hdd.
  --min-free-gib N           Minimum available GiB on the result filesystem.

Options:
  --peer-ip IPV4             Verify route and three ICMP probes to this peer.
  --bind-host IPV4           Require this local IPv4 address (cloud role).
  --ports P1[,P2...]         Require all listed TCP ports to be unused.
  --expected-ref REF         Require HEAD to equal the supplied local Git ref.
  --expected-specedge-ref SHA Require pinned SpecEdge submodule commit.
  --python PATH              Python to CUDA/import smoke-test.
  --require-exclusive        Fail if any selected GPU has a compute process.
  --help

The script only reads state.  It never creates files, removes files, or stops
processes.
EOF
}

ROLE=""
PHYSICAL_GPUS=""
RUN_ROOT=""
MIN_FREE_GIB=""
PEER_IP=""
BIND_HOST=""
PORTS=""
EXPECTED_REF=""
EXPECTED_SPECEDGE_REF=""
PYTHON_BIN="${FASTSD_PYTHON:-/home/hdd/zhangh/envs/fastsd/bin/python}"
REQUIRE_EXCLUSIVE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="${2:-}"; shift 2 ;;
    --physical-gpus) PHYSICAL_GPUS="${2:-}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --min-free-gib) MIN_FREE_GIB="${2:-}"; shift 2 ;;
    --peer-ip) PEER_IP="${2:-}"; shift 2 ;;
    --bind-host) BIND_HOST="${2:-}"; shift 2 ;;
    --ports) PORTS="${2:-}"; shift 2 ;;
    --expected-ref) EXPECTED_REF="${2:-}"; shift 2 ;;
    --expected-specedge-ref) EXPECTED_SPECEDGE_REF="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --require-exclusive) REQUIRE_EXCLUSIVE=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "${ROLE}" == "edge" || "${ROLE}" == "cloud" ]] || die "--role must be edge or cloud."
[[ -n "${PHYSICAL_GPUS}" && -n "${RUN_ROOT}" && -n "${MIN_FREE_GIB}" ]] || { usage >&2; die "--physical-gpus, --run-root, and --min-free-gib are required."; }
[[ "${MIN_FREE_GIB}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--min-free-gib must be a non-negative number."
validate_hdd_path "${RUN_ROOT}"
if [[ "${ROLE}" == "edge" ]]; then
  validate_gpu_list "${PHYSICAL_GPUS}" 2
else
  validate_gpu_list "${PHYSICAL_GPUS}" 1
  [[ -n "${BIND_HOST}" ]] || die "cloud preflight requires --bind-host."
fi

if [[ -n "${PEER_IP}" ]]; then
  validate_ipv4 "${PEER_IP}"
fi
if [[ -n "${BIND_HOST}" ]]; then
  validate_ipv4 "${BIND_HOST}"
fi
if [[ -n "${PORTS}" ]]; then
  [[ "${PORTS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--ports must be comma-separated port numbers."
  IFS=, read -r -a PORT_ARRAY <<< "${PORTS}"
  for port in "${PORT_ARRAY[@]}"; do
    validate_port "${port}"
  done
fi

require_repo_root
need_command git
need_command df
need_command nvidia-smi
require_executable_or_command "${PYTHON_BIN}"

note "checking clean Git state"
GIT_STATUS="$(git -C "${REPO_ROOT}" status --porcelain --ignore-submodules=none)"
[[ -z "${GIT_STATUS}" ]] || die "Git worktree is not clean; resolve it before allocating GPUs."
if [[ -n "${EXPECTED_REF}" ]]; then
  EXPECTED_SHA="$(git -C "${REPO_ROOT}" rev-parse "${EXPECTED_REF}^{commit}")" || die "Cannot resolve expected ref: ${EXPECTED_REF}"
  HEAD_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  [[ "${HEAD_SHA}" == "${EXPECTED_SHA}" ]] || die "HEAD ${HEAD_SHA} does not equal ${EXPECTED_REF} (${EXPECTED_SHA})."
fi
SPECEDGE_STATUS="$(git -C "${REPO_ROOT}" submodule status -- baselines/specedge/official)" || die "Cannot inspect pinned SpecEdge submodule."
[[ "${SPECEDGE_STATUS}" != -* && "${SPECEDGE_STATUS}" != +* ]] || die "SpecEdge submodule is uninitialized or differs from the recorded revision."
if [[ -n "${EXPECTED_SPECEDGE_REF}" ]]; then
  [[ "${SPECEDGE_STATUS}" == *"${EXPECTED_SPECEDGE_REF}"* ]] || die "SpecEdge does not match expected revision ${EXPECTED_SPECEDGE_REF}."
fi

note "checking result filesystem"
FREE_GIB="$(available_gib "${RUN_ROOT}")"
awk -v actual="${FREE_GIB}" -v required="${MIN_FREE_GIB}" 'BEGIN { exit !(actual >= required) }' || die "Only ${FREE_GIB} GiB available under ${RUN_ROOT}; need at least ${MIN_FREE_GIB} GiB."

case "${ROLE}" in
  edge)
    DRAFT_MODEL="${FASTSD_DRAFT_MODEL:-/home/hdd/zhangh/models/Qwen3-0.6B}"
    require_file "${DRAFT_MODEL}/config.json"
    require_file "${DRAFT_MODEL}/tokenizer.json"
    ;;
  cloud)
    TARGET_MODEL="${FASTSD_TARGET_MODEL:-/home/hdd/zhangh/models/Qwen3-8B}"
    CLOUD_TOKENIZER="${FASTSD_CLOUD_TOKENIZER_MODEL:-${TARGET_MODEL}}"
    require_file "${TARGET_MODEL}/config.json"
    require_file "${CLOUD_TOKENIZER}/tokenizer.json"
    ;;
esac

note "checking selected GPU indexes and exclusive allocation"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
IFS=, read -r -a GPU_ARRAY <<< "${PHYSICAL_GPUS}"
COMPUTE_APPS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
for gpu in "${GPU_ARRAY[@]}"; do
  ((10#${gpu} < GPU_COUNT)) || die "Selected GPU ${gpu} does not exist (detected ${GPU_COUNT})."
  GPU_UUID="$(nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader,nounits)"
  if [[ "${REQUIRE_EXCLUSIVE}" == true ]] && grep -Fq "${GPU_UUID}" <<< "${COMPUTE_APPS}"; then
    die "Selected GPU ${gpu} (${GPU_UUID}) has a compute process; no process was touched."
  fi
done

note "checking Python imports and visible CUDA devices"
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - "${#GPU_ARRAY[@]}" <<'PY'
import sys
import torch
import transformers

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the configured Python interpreter")
if torch.cuda.device_count() != expected:
    raise SystemExit(
        f"CUDA_VISIBLE_DEVICES mapping mismatch: expected {expected}, got {torch.cuda.device_count()}"
    )
print(f"python={sys.executable} torch={torch.__version__} transformers={transformers.__version__}")
PY

if [[ -n "${BIND_HOST}" ]]; then
  need_command ip
  ip -o -4 addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "${BIND_HOST}" || die "--bind-host ${BIND_HOST} is not assigned to this machine."
fi
if [[ -n "${PEER_IP}" ]]; then
  need_command ip
  need_command ping
  note "checking route and ping to ${PEER_IP}"
  ip route get "${PEER_IP}" >/dev/null
  ping -c 3 -W 2 "${PEER_IP}" >/dev/null
fi
if [[ -n "${PORTS}" ]]; then
  if command -v ss >/dev/null 2>&1; then
    LISTENING_TCP="$(ss -ltnH)"
  elif command -v netstat >/dev/null 2>&1; then
    LISTENING_TCP="$(netstat -ltn)"
  else
    die "Need ss or netstat to check requested ports."
  fi
  for port in "${PORT_ARRAY[@]}"; do
    if grep -Eq "[:.]${port}[[:space:]]" <<< "${LISTENING_TCP}"; then
      die "TCP port ${port} is already listening; no process was touched."
    fi
  done
fi

HEAD_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "PRECHECK_OK role=${ROLE} head=${HEAD_SHA} free_gib=${FREE_GIB} physical_gpus=${PHYSICAL_GPUS}"
