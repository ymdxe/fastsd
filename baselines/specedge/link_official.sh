#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SPECEDGE_OFFICIAL_SOURCE:?set SPECEDGE_OFFICIAL_SOURCE to an existing official SpecEdge checkout}"
DEST="$ROOT/official/source"

[[ -d "$SOURCE" ]] || { echo "official source does not exist: $SOURCE" >&2; exit 2; }
if [[ -e "$DEST" || -L "$DEST" ]]; then
  [[ "${SPECEDGE_REPLACE_LINK:-0}" == 1 ]] || { echo "destination exists: $DEST (set SPECEDGE_REPLACE_LINK=1 to replace)" >&2; exit 2; }
  rm -rf "$DEST"
fi
ln -s "$SOURCE" "$DEST"
echo "linked official SpecEdge: $DEST -> $SOURCE"
