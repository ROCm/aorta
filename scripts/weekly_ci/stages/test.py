"""
Test execution stages for Weekly CI Kickoff.

This module provides:
- Performance test execution (RCCL warp speed comparison)
- Experiment directory discovery
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..utils import (
    docker_exec,
    find_latest_experiment_dir,
    find_second_latest_experiment_dir,
    parse_config_pairs,
)


def stage_run_performance_tests(
    container_name: str,
    config_pairs: str,
    training_config: str,
    logger: logging.Logger,
) -> None:
    """Run RCCL warp speed comparison tests inside the Docker container.

    This stage executes the run_rccl_warp_speed_comparison.sh script which:
    1. Runs training with different CU/thread configurations
    2. Collects performance traces for each configuration
    3. Stores results in experiments/rccl_warp_speed_<timestamp>/ directory

    Args:
        container_name: Name of the Docker container.
        config_pairs: Space-separated CU,threads pairs (e.g., "56,256 37,384 32,512").
        training_config: Path to training config YAML (inside container).
        logger: Logger instance.

    Raises:
        RuntimeError: If performance tests fail.
    """
    logger.info("Running RCCL warp speed comparison tests...")
    logger.info(f"  Config pairs: {config_pairs}")
    logger.info(f"  Training config: {training_config}")

    # Parse and display configurations
    pairs = parse_config_pairs(config_pairs)
    for cu, threads in pairs:
        logger.info(f"    - {cu} CUs, {threads} threads")

    # Build the test script to run inside container
    test_script = f"""
        set -e

        # Set RCCL library path
        export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH

        echo "Starting RCCL warp speed comparison tests..."
        echo "Config pairs: {config_pairs}"
        echo "Training config: {training_config}"

        # Run the RCCL warp speed comparison script
        bash ./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh \\
            -p "{config_pairs}" \\
            -c {training_config}

        echo "Performance tests completed!"
    """

    logger.info("  Starting performance tests...")
    logger.info("  (This may take 30-60 minutes depending on configurations)")

    try:
        docker_exec(
            container_name,
            test_script,
            logger,
            workdir="/workspace/aorta",
            check=True,
        )
        logger.info("  ✓ Performance tests completed successfully")
    except Exception as e:
        raise RuntimeError(f"Performance tests failed: {e}") from e


def stage_find_experiment_dir(
    repo_root: Path,
    logger: logging.Logger,
    explicit_experiment_dir: str = "",
) -> Optional[str]:
    """Find or validate the experiment directory.

    If explicit_experiment_dir is provided, validates it exists.
    Otherwise, searches for the most recent experiment directory matching:
    experiments/rccl_warp_speed_<timestamp>/

    Args:
        repo_root: Path to the repository root.
        logger: Logger instance.
        explicit_experiment_dir: Explicitly specified experiment directory (optional).

    Returns:
        Relative path to the experiment directory (e.g., "experiments/rccl_warp_speed_20260220_143000"),
        or None if no experiment directory found.

    Raises:
        RuntimeError: If no experiment directory is found or explicit path doesn't exist.
    """
    logger.info("Finding experiment directory...")

    if explicit_experiment_dir:
        # Use explicitly provided experiment directory
        exp_path = repo_root / explicit_experiment_dir
        if exp_path.exists():
            logger.info(f"  Using specified experiment directory: {explicit_experiment_dir}")
            return explicit_experiment_dir
        else:
            raise RuntimeError(
                f"Specified experiment directory not found: {explicit_experiment_dir}"
            )

    # Auto-detect most recent experiment
    experiments_dir = repo_root / "experiments"
    experiment_dir = find_latest_experiment_dir(experiments_dir)

    if experiment_dir is None:
        raise RuntimeError(
            f"No experiment directory found in {experiments_dir}. "
            "Make sure performance tests have been run first, or use --experiment-dir to specify one."
        )

    # Return relative path
    relative_path = experiment_dir.relative_to(repo_root)
    logger.info(f"  Found experiment directory: {relative_path}")

    return str(relative_path)


def _find_latest_aorta_report_date(aorta_report_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Find the most recent date directory in aorta-report containing rccl-warp-speed results.

    Args:
        aorta_report_dir: Path to aorta-report repository.
        logger: Logger instance.

    Returns:
        Path to the most recent rccl-warp-speed directory, or None if not found.
    """
    if not aorta_report_dir or not aorta_report_dir.exists():
        return None

    # Look for date directories (format: YYYY-MM-DD) containing rccl-warp-speed
    date_dirs = []
    for item in aorta_report_dir.iterdir():
        if item.is_dir() and len(item.name) == 10 and item.name[4] == '-' and item.name[7] == '-':
            rccl_path = item / "rccl-warp-speed"
            if rccl_path.exists() and rccl_path.is_dir():
                date_dirs.append((item.name, rccl_path))

    if not date_dirs:
        return None

    # Sort by date (descending) and return the most recent
    date_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_date, latest_path = date_dirs[0]
    logger.debug(f"  Found {len(date_dirs)} date directories in aorta-report, latest: {latest_date}")

    return latest_path


def stage_find_baseline_experiment_dir(
    repo_root: Path,
    baseline_experiment: str,
    logger: logging.Logger,
    baseline_date: str = "",
    aorta_report_dir: Optional[Path] = None,
) -> Optional[str]:
    """Find baseline experiment directory for cross-timestamp comparison.

    Priority order:
    1. baseline_experiment - explicit local experiment path
    2. baseline_date + aorta_report_dir - explicit date directory in aorta-report
    3. Auto-detect most recent date in aorta-report (if checked out)
    4. Auto-detect second most recent local experiment

    Args:
        repo_root: Path to the repository root.
        baseline_experiment: Explicit baseline experiment path (empty for auto-detect).
        logger: Logger instance.
        baseline_date: Date directory in aorta-report (e.g., "2026-02-19").
        aorta_report_dir: Path to aorta-report repository (for date-based lookup).

    Returns:
        Path to the baseline experiment directory (absolute for aorta-report, relative for local),
        or None if not found.
    """
    logger.info("Finding baseline experiment directory for cross-timestamp comparison...")

    # Option 1: Explicit local experiment path
    if baseline_experiment:
        baseline_path = repo_root / baseline_experiment
        if baseline_path.exists():
            logger.info(f"  Using provided baseline: {baseline_experiment}")
            return baseline_experiment
        else:
            logger.warning(f"  Provided baseline not found: {baseline_experiment}")
            logger.info("  Attempting other options...")

    # Option 2: Explicit date directory in aorta-report
    if baseline_date and aorta_report_dir:
        aorta_baseline_path = aorta_report_dir / baseline_date / "rccl-warp-speed"
        if aorta_baseline_path.exists():
            logger.info(f"  Using aorta-report baseline from {baseline_date}")
            # Return relative path if aorta-report is inside repo (for Docker compatibility)
            try:
                relative_path = aorta_baseline_path.relative_to(repo_root)
                logger.info(f"    Path (relative): {relative_path}")
                return str(relative_path)
            except ValueError:
                # aorta-report is outside repo, return absolute path
                logger.info(f"    Path (absolute): {aorta_baseline_path}")
                return str(aorta_baseline_path)
        else:
            logger.warning(f"  Baseline date not found in aorta-report: {baseline_date}")
            logger.info(f"    Expected path: {aorta_baseline_path}")
            logger.info("  Attempting auto-detection...")

    # Option 3: Auto-detect most recent date in aorta-report
    if aorta_report_dir and aorta_report_dir.exists():
        logger.info("  Searching aorta-report for most recent baseline...")
        aorta_baseline_path = _find_latest_aorta_report_date(aorta_report_dir, logger)
        if aorta_baseline_path:
            logger.info(f"  Found baseline in aorta-report: {aorta_baseline_path.parent.name}")
            # Return relative path if aorta-report is inside repo (for Docker compatibility)
            try:
                relative_path = aorta_baseline_path.relative_to(repo_root)
                logger.info(f"    Path (relative): {relative_path}")
                return str(relative_path)
            except ValueError:
                # aorta-report is outside repo, return absolute path
                logger.info(f"    Path (absolute): {aorta_baseline_path}")
                return str(aorta_baseline_path)
        else:
            logger.info("  No rccl-warp-speed results found in aorta-report")
            logger.info("  Attempting local auto-detection...")

    # Option 4: Auto-detect second most recent local experiment
    experiments_dir = repo_root / "experiments"
    baseline_dir = find_second_latest_experiment_dir(experiments_dir)

    if baseline_dir is None:
        logger.warning("  No baseline experiment directory found for cross-timestamp comparison")
        logger.warning("  (Need at least 2 experiment runs, aorta-report with results, or use --baseline-experiment)")
        return None

    # Return relative path for local experiments
    relative_path = baseline_dir.relative_to(repo_root)
    logger.info(f"  Found baseline experiment directory: {relative_path}")

    return str(relative_path)


def validate_experiment_configs(
    experiment_dir: Path,
    config_pairs: str,
    logger: logging.Logger,
) -> list[str]:
    """Validate that experiment directory contains expected configuration subdirectories.

    Args:
        experiment_dir: Path to the experiment directory.
        config_pairs: Space-separated CU,threads pairs.
        logger: Logger instance.

    Returns:
        List of found configuration directory names.
    """
    pairs = parse_config_pairs(config_pairs)
    found_configs = []
    missing_configs = []

    for cu, threads in pairs:
        config_dir_name = f"{cu}cu_{threads}threads"
        config_path = experiment_dir / config_dir_name

        if config_path.exists() and config_path.is_dir():
            found_configs.append(config_dir_name)
            logger.debug(f"  Found: {config_dir_name}")
        else:
            missing_configs.append(config_dir_name)
            logger.warning(f"  Missing: {config_dir_name}")

    if missing_configs:
        logger.warning(
            f"  Warning: {len(missing_configs)} configuration(s) missing: "
            f"{', '.join(missing_configs)}"
        )

    return found_configs

