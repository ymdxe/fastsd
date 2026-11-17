#!/usr/bin/env bash
# Run the separately managed official SpecEdge baseline.
# No model/data/package download is performed here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${DATASET:-humaneval}"
MT_BENCH_TURNS="${MT_BENCH_TURNS:-1}"
DATA_PATH="${DATA_PATH:-$ROOT/data}"
SPECEDGE_OFFICIAL_ROOT="${SPECEDGE_OFFICIAL_ROOT:-$ROOT/baselines/specedge/official/source}"
SPECEDGE_CONFIG="${SPECEDGE_CONFIG:-$ROOT/baselines/specedge/configs/final/specedge_qwen3_8b_0.6b.yaml}"
SPECEDGE_RESULT_ROOT="${SPECEDGE_RESULT_ROOT:-$ROOT/runtime/specedge}"
SPECEDGE_MODE="${SPECEDGE_MODE:-official}"
SPECEDGE_REMOTE_ROOT="${SPECEDGE_REMOTE_ROOT:-}"
SPECEDGE_HOST="${SPECEDGE_HOST:-}"
SPECEDGE_SSH_KEY="${SPECEDGE_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SPECEDGE_INPUT_PATH="${SPECEDGE_INPUT_PATH:-$SPECEDGE_RESULT_ROOT/input/${DATASET}.jsonl}"
SPECEDGE_SERVER_HOST="${SPECEDGE_SERVER_HOST:-$SPECEDGE_HOST}"
SPECEDGE_CLIENT_HOST="${SPECEDGE_CLIENT_HOST:-localhost}"
SPECEDGE_TARGET_MODEL_8B="${SPECEDGE_TARGET_MODEL_8B:-Qwen3-8B}"
SPECEDGE_DRAFT_MODEL_0_6B="${SPECEDGE_DRAFT_MODEL_0_6B:-Qwen3-0.6B}"
SPECEDGE_TARGET_MODEL_32B="${SPECEDGE_TARGET_MODEL_32B:-Qwen3-32B}"
SPECEDGE_DRAFT_MODEL_1_7B="${SPECEDGE_DRAFT_MODEL_1_7B:-Qwen3-1.7B}"

case "$SPECEDGE_MODE" in
  official|cpu-adapted) ;;
  *) echo "SPECEDGE_MODE must be official or cpu-adapted" >&2; exit 2 ;;
esac
[[ -d "$SPECEDGE_OFFICIAL_ROOT" ]] || {
  echo "official SpecEdge source missing: $SPECEDGE_OFFICIAL_ROOT" >&2
  echo "set SPECEDGE_OFFICIAL_SOURCE and run baselines/specedge/link_official.sh" >&2
  exit 2
}
INPUT="$DATA_PATH/$DATASET.jsonl"
[[ -f "$INPUT" ]] || { echo "dataset missing: $INPUT" >&2; exit 2; }

python3 "$ROOT/baselines/specedge/integration/convert_dataset.py" \
  --dataset "$DATASET" --input "$INPUT" --output "$SPECEDGE_INPUT_PATH" \
  --mt-bench-turns "$MT_BENCH_TURNS"

export SPECEDGE_SERVER_HOST SPECEDGE_CLIENT_HOST \
  SPECEDGE_TARGET_MODEL_8B SPECEDGE_DRAFT_MODEL_0_6B \
  SPECEDGE_TARGET_MODEL_32B SPECEDGE_DRAFT_MODEL_1_7B \
  SPECEDGE_SSH_KEY SPECEDGE_INPUT_PATH SPECEDGE_MODE
RENDERED_CONFIG="$SPECEDGE_RESULT_ROOT/config/$(basename "$SPECEDGE_CONFIG")"
python3 - "$SPECEDGE_CONFIG" "$RENDERED_CONFIG" <<'PY'
import os, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
for key in (
    "SPECEDGE_SSH_KEY", "SPECEDGE_SERVER_HOST", "SPECEDGE_CLIENT_HOST",
    "SPECEDGE_TARGET_MODEL_8B", "SPECEDGE_DRAFT_MODEL_0_6B",
    "SPECEDGE_TARGET_MODEL_32B", "SPECEDGE_DRAFT_MODEL_1_7B", "SPECEDGE_INPUT_PATH",
):
    text = text.replace("${" + key + "}", os.environ.get(key, ""))
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(text, encoding="utf-8")
PY

echo "config=$RENDERED_CONFIG"
echo "official_root=$SPECEDGE_OFFICIAL_ROOT"
echo "result_root=$SPECEDGE_RESULT_ROOT"
echo "mode=$SPECEDGE_MODE official_tp=false offload=false"

if [[ -n "$SPECEDGE_HOST" ]]; then
  [[ -n "$SPECEDGE_REMOTE_ROOT" ]] || { echo "SPECEDGE_REMOTE_ROOT is required with SPECEDGE_HOST" >&2; exit 2; }
  tar -czf - --exclude='__pycache__' -C "$ROOT" baselines/specedge | ssh -i "$SPECEDGE_SSH_KEY" "$SPECEDGE_HOST" \
    "mkdir -p '$SPECEDGE_REMOTE_ROOT' && tar -xzf - -C '$SPECEDGE_REMOTE_ROOT'"
  echo "remote source/config staged at $SPECEDGE_HOST:$SPECEDGE_REMOTE_ROOT"
  echo "Run the official server/client commands from docs/specedge_server_todo.md on that host."
  exit 0
fi

if [[ "${SPECEDGE_EXECUTE:-0}" != 1 ]]; then
  echo "staging complete; set SPECEDGE_EXECUTE=1 to invoke the official scripts"
  exit 0
fi

[[ -x "$SPECEDGE_OFFICIAL_ROOT/script/batch_server.sh" ]] || {
  echo "official batch_server.sh missing or not executable" >&2; exit 2;
}
CONFIG_DEST="$SPECEDGE_OFFICIAL_ROOT/config/fastsd_final.yaml"
mkdir -p "$(dirname "$CONFIG_DEST")"
  cp "$RENDERED_CONFIG" "$CONFIG_DEST"
exec "$SPECEDGE_OFFICIAL_ROOT/script/batch_server.sh" -f "$CONFIG_DEST"
