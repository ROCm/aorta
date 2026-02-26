"""
Build stages for Weekly CI Kickoff.

This module provides:
- RCCL clone and build
- Python dependency installation
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..utils import docker_exec


def stage_build_rccl(
    container_name: str,
    rccl_branch: str,
    gpu_target: str,
    logger: logging.Logger,
) -> None:
    """Clone and build RCCL inside the Docker container.

    Steps:
    1. Create /rccl directory if it doesn't exist
    2. Clone or update the RCCL repository
    3. Checkout the specified branch
    4. Build RCCL with the specified GPU target

    Args:
        container_name: Name of the Docker container.
        rccl_branch: RCCL git branch to checkout.
        gpu_target: GPU architecture target (e.g., gfx950).
        logger: Logger instance.

    Raises:
        RuntimeError: If RCCL build fails.
    """
    logger.info(f"Building RCCL branch: {rccl_branch}")
    logger.info(f"GPU target: {gpu_target}")

    # Build script to run inside container
    build_script = f"""
        set -e

        mkdir -p /rccl && cd /rccl

        if [ -d 'rccl' ]; then
            echo "Updating existing RCCL repository..."
            cd rccl
            git fetch origin
            git checkout {rccl_branch}
            git pull origin {rccl_branch} || true
        else
            echo "Cloning RCCL repository..."
            git clone --recursive https://github.com/mustafabar/rccl.git
            cd rccl
            git checkout {rccl_branch}
        fi

        echo "Building RCCL with GPU target: {gpu_target}"
        echo "This may take 15-30 minutes..."
        ./install.sh -l --amdgpu_targets={gpu_target}

        echo "Verifying build..."
        if [ -d "/rccl/rccl/build/release" ]; then
            echo "RCCL build directory contents:"
            ls -la /rccl/rccl/build/release/ | head -20
        else
            echo "ERROR: Build directory not found!"
            exit 1
        fi

        echo "RCCL build completed successfully!"
    """

    logger.info("Starting RCCL build inside container...")
    logger.info("  (This may take 15-30 minutes on first build)")

    try:
        docker_exec(container_name, build_script, logger, check=True)
        logger.info("  ✓ RCCL build completed successfully")
    except Exception as e:
        raise RuntimeError(f"RCCL build failed: {e}") from e


def stage_install_dependencies(
    container_name: str,
    repo_root: Path,
    logger: logging.Logger,
) -> None:
    """Install Python dependencies inside the Docker container.

    Steps:
    1. Install requirements from requirements.txt
    2. Install additional analysis packages
    3. Install the current package in editable mode (pip install -e .)

    Args:
        container_name: Name of the Docker container.
        repo_root: Path to repository root (for reference).
        logger: Logger instance.

    Raises:
        RuntimeError: If dependency installation fails.
    """
    logger.info("Installing Python dependencies...")

    # Install script
    install_script = """
        set -e

        echo "Installing requirements.txt..."
        pip install -r requirements.txt

        echo "Installing additional analysis packages..."
        pip install pandas openpyxl matplotlib seaborn numpy

        echo "Installing current package in editable mode..."
        pip install -e .

        echo "Verifying installations..."
        python -c "import pandas; import matplotlib; import seaborn; print('Core packages OK')"

        echo "Dependencies installed successfully!"
    """

    try:
        docker_exec(
            container_name,
            install_script,
            logger,
            workdir="/workspace/aorta",
            check=True,
        )
        logger.info("  ✓ Dependencies installed successfully")
    except Exception as e:
        raise RuntimeError(f"Dependency installation failed: {e}") from e


def verify_rccl_installation(
    container_name: str,
    logger: logging.Logger,
) -> bool:
    """Verify RCCL is properly installed.

    Args:
        container_name: Name of the Docker container.
        logger: Logger instance.

    Returns:
        True if RCCL is installed, False otherwise.
    """
    try:
        result = docker_exec(
            container_name,
            "test -d /rccl/rccl/build/release && echo 'exists'",
            logger,
            capture_output=True,
            check=False,
        )
        return "exists" in result.stdout if result.stdout else False
    except Exception:
        return False

