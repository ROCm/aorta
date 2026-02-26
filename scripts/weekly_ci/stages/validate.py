"""
Environment validation stage for Weekly CI Kickoff.

This module provides:
- Docker availability check
- Repository root validation
- Config file existence check
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..utils import get_repo_root, run_command


def stage_validate_environment(
    config_path: Path,
    logger: logging.Logger,
) -> Path:
    """Validate the execution environment.

    Checks:
    1. Docker is installed and accessible
    2. Docker Compose is available
    3. Repository root can be determined
    4. Config file exists (if specified)

    Args:
        config_path: Path to the configuration file.
        logger: Logger instance.

    Returns:
        Path to the repository root.

    Raises:
        EnvironmentError: If validation fails.
    """
    logger.info("Validating execution environment...")

    # Check Docker is available
    logger.info("Checking Docker availability...")
    if not shutil.which("docker"):
        raise EnvironmentError("Docker is not installed or not in PATH")

    # Verify Docker daemon is running
    result = run_command(
        "docker info",
        logger,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise EnvironmentError(
            "Docker daemon is not running. Please start Docker and try again."
        )
    logger.info("  ✓ Docker is available and running")

    # Check Docker Compose
    logger.info("Checking Docker Compose...")
    result = run_command(
        "docker compose version",
        logger,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        # Try legacy docker-compose
        result = run_command(
            "docker-compose version",
            logger,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise EnvironmentError(
                "Docker Compose is not available. Please install Docker Compose."
            )
    logger.info("  ✓ Docker Compose is available")

    # Find repository root
    logger.info("Finding repository root...")
    try:
        repo_root = get_repo_root()
        logger.info(f"  ✓ Repository root: {repo_root}")
    except FileNotFoundError as e:
        raise EnvironmentError(f"Could not find repository root: {e}") from e

    # Check config file exists
    if config_path.name != "weekly_ci.yaml" or config_path.exists():
        # Only check if a custom config was specified or default exists
        if config_path.exists():
            logger.info(f"  ✓ Config file exists: {config_path}")
        else:
            logger.warning(f"  ⚠ Config file not found: {config_path} (using defaults)")

    # Check required directories exist
    experiments_dir = repo_root / "experiments"
    if not experiments_dir.exists():
        logger.info(f"Creating experiments directory: {experiments_dir}")
        experiments_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Environment validation complete!")
    return repo_root

