#!/usr/bin/env bash
# Resolve docker/compose commands for self-hosted runners where the runner user
# may need either direct docker group access or passwordless sudo.
#
# Usage:
#   source scripts/ci/docker_env.sh
#   $AORTA_DOCKER info
#   bash scripts/ci/docker_compose.sh up -d
#
# This file is meant to be *sourced*, so it deliberately does NOT run
# `set -euo pipefail` (that would mutate the caller's shell options). The
# wrapper scripts (docker_cmd.sh / docker_compose.sh) set strict mode
# themselves before sourcing this helper.

_aorta_resolve_docker() {
  if [ -n "${AORTA_DOCKER:-}" ]; then
    return 0
  fi
  if docker info >/dev/null 2>&1; then
    AORTA_DOCKER=docker
  elif sudo -n docker info >/dev/null 2>&1; then
    AORTA_DOCKER="sudo docker"
  else
    echo "ERROR: cannot access Docker as $(whoami)" >&2
    echo "Fix on the runner host (pick one):" >&2
    echo "  sudo usermod -aG docker,video $(whoami)" >&2
    echo "  sudo setfacl -m user:$(whoami):rw /var/run/docker.sock" >&2
    echo "  passwordless sudo for docker (see docs/ci-testing-plan.md)" >&2
    id >&2
    ls -l /var/run/docker.sock >&2 || true
    getent group docker >&2 || true
    return 1
  fi
  export AORTA_DOCKER
}

_aorta_resolve_compose() {
  _aorta_resolve_docker
  if [ -n "${AORTA_COMPOSE:-}" ]; then
    return 0
  fi
  if ${AORTA_DOCKER} compose version >/dev/null 2>&1; then
    AORTA_COMPOSE="${AORTA_DOCKER} compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    if [ "${AORTA_DOCKER}" = "sudo docker" ]; then
      AORTA_COMPOSE="sudo docker-compose"
    else
      AORTA_COMPOSE=docker-compose
    fi
  else
    echo "ERROR: neither 'docker compose' nor 'docker-compose' is available" >&2
    return 1
  fi
  export AORTA_COMPOSE
}

_aorta_resolve_docker
_aorta_resolve_compose
