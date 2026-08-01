#!/usr/bin/env bash
set -euo pipefail

# Clear proxy-related variables to avoid broken streaming JSON responses.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

# Restore TLS verification defaults if previously disabled.
unset NODE_TLS_REJECT_UNAUTHORIZED

exec opencode "$@"
