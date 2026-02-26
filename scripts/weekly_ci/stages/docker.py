"""
Docker management stages for Weekly CI Kickoff.

This module provides:
- Docker container setup (build and start)
- Docker container cleanup (stop and remove)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ..utils import check_docker_exists, check_docker_running, run_command


def stage_docker_setup(
    compose_file: str,
    container_name: str,
    repo_root: Path,
    logger: logging.Logger,
    registry_user: Optional[str] = None,
    registry_password: Optional[str] = None,
    skip_build: bool = True,
    force_restart: bool = False,
) -> None:
    """Set up Docker container for the pipeline.

    Steps:
    1. Check if container is already running (skip setup if so, unless force_restart)
    2. Login to Docker registry (if credentials provided)
    3. Stop and remove any existing container with the same name
    4. Build the Docker image using docker compose (if skip_build=False)
    5. Start the container in detached mode

    Args:
        compose_file: Path to docker-compose file (relative to repo root).
        container_name: Name of the container.
        repo_root: Path to repository root.
        logger: Logger instance.
        registry_user: Docker registry username (optional).
        registry_password: Docker registry password (optional, can use DOCKER_PASSWORD env).
        skip_build: If True, skip docker compose build (default: True).
        force_restart: If True, restart container even if already running (default: False).

    Raises:
        RuntimeError: If Docker setup fails.
    """
    compose_path = repo_root / compose_file

    # Verify compose file exists
    if not compose_path.exists():
        raise RuntimeError(f"Docker compose file not found: {compose_path}")

    logger.info(f"Using compose file: {compose_path}")

    # Check if container is already running - reuse if so (unless force_restart)
    if check_docker_running(container_name, logger) and not force_restart:
        logger.info(f"Container '{container_name}' is already running")
        logger.info("  Reusing existing container (use --force-restart to restart)")
        logger.info(f"  ✓ Container {container_name} is ready")
        return

    if force_restart:
        logger.info("Force restart requested, will restart container...")

    # Docker login (if credentials provided)
    password = registry_password or os.environ.get("DOCKER_PASSWORD", "")
    if registry_user and password:
        logger.info(f"Logging into Docker registry as {registry_user}...")
        result = run_command(
            f"docker login -u {registry_user} -p {password}",
            logger,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("  ✓ Docker login successful")
        else:
            logger.warning("  ⚠ Docker login failed, continuing anyway...")
    elif registry_user:
        logger.warning(
            f"  ⚠ Docker user '{registry_user}' provided but no password. "
            "Set DOCKER_PASSWORD env var or use --docker-password"
        )

    # Cleanup existing container
    logger.info(f"Cleaning up existing container: {container_name}...")

    if check_docker_exists(container_name, logger):
        if check_docker_running(container_name, logger):
            logger.info(f"  Stopping running container: {container_name}")
            run_command(
                f"docker stop {container_name}",
                logger,
                capture_output=True,
                check=False,
            )

        logger.info(f"  Removing container: {container_name}")
        run_command(
            f"docker rm {container_name}",
            logger,
            capture_output=True,
            check=False,
        )

    # Also try docker compose down
    logger.info("  Running docker compose down (cleanup)...")
    run_command(
        f"docker compose -f {compose_path} down",
        logger,
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    logger.info("  ✓ Cleanup complete")

    # Build the image (unless skip_build is True)
    if skip_build:
        logger.info("Skipping Docker image build (--docker-build to enable)")
    else:
        logger.info("Building Docker image...")
        logger.info("  This may take several minutes on first run...")
        run_command(
            f"docker compose -f {compose_path} build",
            logger,
            cwd=repo_root,
            check=True,
        )
        logger.info("  ✓ Docker image built successfully")

    # Start the container
    logger.info("Starting Docker container...")
    run_command(
        f"docker compose -f {compose_path} up -d",
        logger,
        cwd=repo_root,
        check=True,
    )

    # Verify container is running
    if not check_docker_running(container_name, logger):
        raise RuntimeError(f"Container {container_name} failed to start")

    logger.info(f"  ✓ Container {container_name} is running")


def stage_cleanup(
    compose_file: str,
    container_name: str,
    repo_root: Path,
    logger: logging.Logger,
) -> None:
    """Clean up Docker container after pipeline completion.

    Steps:
    1. Stop the container if running
    2. Remove the container
    3. Run docker compose down for full cleanup

    Args:
        compose_file: Path to docker-compose file (relative to repo root).
        container_name: Name of the container.
        repo_root: Path to repository root.
        logger: Logger instance.
    """
    compose_path = repo_root / compose_file

    logger.info(f"Cleaning up container: {container_name}...")

    # Stop container if running
    if check_docker_running(container_name, logger):
        logger.info(f"  Stopping container: {container_name}")
        run_command(
            f"docker stop {container_name}",
            logger,
            capture_output=True,
            check=False,
        )

    # Remove container
    if check_docker_exists(container_name, logger):
        logger.info(f"  Removing container: {container_name}")
        run_command(
            f"docker rm {container_name}",
            logger,
            capture_output=True,
            check=False,
        )

    # Docker compose down
    if compose_path.exists():
        logger.info("  Running docker compose down...")
        run_command(
            f"docker compose -f {compose_path} down",
            logger,
            cwd=repo_root,
            capture_output=True,
            check=False,
        )

    logger.info("  ✓ Cleanup complete")

