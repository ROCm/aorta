"""
Utility functions for Weekly CI Kickoff.

This module provides:
- Shell command execution with logging
- Docker container command execution
- Path and file utilities
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


def extract_date_from_experiment_dir(experiment_dir: str, logger: logging.Logger) -> str:
    """Extract YYYY-MM-DD from experiment directory name.

    Format: rccl_warp_speed_YYYYMMDD_HHMMSS (e.g. experiments/rccl_warp_speed_20260224_065602)
    Fallback: today's date if parsing fails.

    Args:
        experiment_dir: Path to experiment directory (relative or absolute).
        logger: Logger instance for warnings.

    Returns:
        Date string in YYYY-MM-DD format.
    """
    exp_name = Path(experiment_dir).name
    try:
        timestamp_str = exp_name.replace("rccl_warp_speed_", "")
        date_obj = datetime.strptime(timestamp_str[:8], "%Y%m%d")
        return date_obj.strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        result = datetime.now().strftime("%Y-%m-%d")
        logger.warning(f"Could not parse date from {exp_name}, using today: {result}")
        return result


def get_repo_root() -> Path:
    """Get the root directory of the aorta repository.

    Returns:
        Path to the repository root.

    Raises:
        FileNotFoundError: If repository root cannot be determined.
    """
    # Try to find root by looking for pyproject.toml or .git
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Could not determine repository root")


def run_command(
    cmd: str,
    logger: logging.Logger,
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    check: bool = True,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Execute a shell command with logging.

    Args:
        cmd: Command string to execute.
        logger: Logger instance for output.
        cwd: Working directory for command execution.
        capture_output: If True, capture stdout/stderr instead of streaming.
        check: If True, raise exception on non-zero exit code.
        env: Optional environment variables dict (merged with current env).

    Returns:
        CompletedProcess instance with return code and captured output.

    Raises:
        subprocess.CalledProcessError: If check=True and command fails.
    """
    logger.debug(f"Running command: {cmd}")
    if cwd:
        logger.debug(f"Working directory: {cwd}")

    # Merge environment if provided
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)

    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=check,
                env=cmd_env,
            )
            if result.stdout:
                logger.debug(f"stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr}")
        else:
            # Stream output in real-time
            result = subprocess.run(
                cmd, shell=True, cwd=cwd, check=check, env=cmd_env, text=True
            )

        logger.debug(f"Command completed with return code: {result.returncode}")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        raise


def docker_exec(
    container_name: str,
    cmd: str,
    logger: logging.Logger,
    workdir: Optional[str] = None,
    capture_output: bool = False,
    check: bool = True,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Execute a command inside a Docker container.

    Args:
        container_name: Name of the Docker container.
        cmd: Command string to execute inside the container.
        logger: Logger instance for output.
        workdir: Working directory inside the container.
        capture_output: If True, capture stdout/stderr instead of streaming.
        check: If True, raise exception on non-zero exit code.
        env: Optional environment variables to set in container.

    Returns:
        CompletedProcess instance with return code and captured output.

    Raises:
        subprocess.CalledProcessError: If check=True and command fails.
    """
    # Build docker exec command
    docker_cmd = f"docker exec"

    # Add environment variables
    if env:
        for key, value in env.items():
            docker_cmd += f' -e {key}="{value}"'

    # Add working directory
    if workdir:
        docker_cmd += f" -w {workdir}"

    # Escape the command for bash -c
    escaped_cmd = cmd.replace("'", "'\"'\"'")
    docker_cmd += f" {container_name} bash -c '{escaped_cmd}'"

    logger.debug(f"Docker exec: {cmd[:100]}..." if len(cmd) > 100 else f"Docker exec: {cmd}")

    return run_command(docker_cmd, logger, capture_output=capture_output, check=check)


def check_docker_running(container_name: str, logger: logging.Logger) -> bool:
    """Check if a Docker container is running.

    Args:
        container_name: Name of the container to check.
        logger: Logger instance.

    Returns:
        True if container is running, False otherwise.
    """
    try:
        result = run_command(
            f"docker inspect -f '{{{{.State.Running}}}}' {container_name}",
            logger,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and "true" in result.stdout.lower()
    except Exception:
        return False


def check_docker_exists(container_name: str, logger: logging.Logger) -> bool:
    """Check if a Docker container exists (running or stopped).

    Args:
        container_name: Name of the container to check.
        logger: Logger instance.

    Returns:
        True if container exists, False otherwise.
    """
    try:
        result = run_command(
            f"docker inspect {container_name}",
            logger,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_latest_experiment_dir(
    experiments_dir: Path, prefix: str = "rccl_warp_speed_"
) -> Optional[Path]:
    """Find the most recently created experiment directory.

    Args:
        experiments_dir: Base experiments directory.
        prefix: Prefix to match experiment directories.

    Returns:
        Path to the most recent experiment directory, or None if not found.
    """
    if not experiments_dir.exists():
        return None

    matching_dirs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    return matching_dirs[0] if matching_dirs else None


def find_second_latest_experiment_dir(
    experiments_dir: Path, prefix: str = "rccl_warp_speed_"
) -> Optional[Path]:
    """Find the second most recently created experiment directory.

    Used for cross-timestamp baseline comparison.

    Args:
        experiments_dir: Base experiments directory.
        prefix: Prefix to match experiment directories.

    Returns:
        Path to the second most recent experiment directory, or None if not found.
    """
    if not experiments_dir.exists():
        return None

    matching_dirs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    return matching_dirs[1] if len(matching_dirs) > 1 else None


def parse_config_pairs(config_pairs: str) -> list[tuple[str, str]]:
    """Parse config pairs string into list of (cu, threads) tuples.

    Args:
        config_pairs: Space-separated CU,threads pairs (e.g., "56,256 37,384").

    Returns:
        List of (cu_count, threads) tuples.
    """
    pairs = []
    for pair in config_pairs.split():
        parts = pair.split(",")
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def get_config_dir_name(cu: str, threads: str) -> str:
    """Generate directory name for a configuration.

    Args:
        cu: CU count.
        threads: Thread count.

    Returns:
        Directory name in format "{cu}cu_{threads}threads".
    """
    return f"{cu}cu_{threads}threads"

