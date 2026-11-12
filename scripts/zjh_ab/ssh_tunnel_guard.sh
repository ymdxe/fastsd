#!/usr/bin/env bash
# One SSH local-forward attempt.  No restart loop: if the connection drops,
# the process exits and the owning round fails visibly instead of respawning.
set -u

HOST="${1:?host is required}"
LOCAL_PORT="${2:?local port is required}"
REMOTE_PORT="${3:?remote port is required}"

exec ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o TCPKeepAlive=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o ConnectTimeout=20 \
  -o BatchMode=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "$HOST"
