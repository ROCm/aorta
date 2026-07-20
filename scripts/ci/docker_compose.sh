#!/usr/bin/env bash
# Wrapper for hosts with either Docker Compose v2 (`docker compose`) or the
# standalone v1 binary (`docker-compose`). Ubuntu packages often ship only one.
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
  exec docker compose "$@"
fi

if command -v docker-compose >/dev/null 2>&1; then
  exec docker-compose "$@"
fi

echo "ERROR: neither 'docker compose' nor 'docker-compose' is available" >&2
echo "Install Compose v2 plugin or the docker-compose package, then retry." >&2
exit 1
