"""
Analysis stages for Weekly CI Kickoff.

This module provides:
- Single-config analysis (aorta-report summary per config)
- Pairwise comparison (baseline vs each config)
- Compare-all-runs analysis (all configs compared together)
- Cross-timestamp comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..utils import docker_exec, get_config_dir_name, parse_config_pairs


def stage_single_config_analysis(
    container_name: str,
    experiment_dir: str,
    config_pairs: str,
    logger: logging.Logger,
    skip_tracelens: bool = False,
) -> None:
    """Run single-config summary analysis for each configuration.

    Generates individual summary reports using
    `aorta-report pipeline summary --test <config_dir>`.

    Args:
        container_name: Name of the Docker container.
        experiment_dir: Path to the experiment directory (relative to workspace).
        config_pairs: Space-separated CU,threads pairs.
        logger: Logger instance.
        skip_tracelens: If True, pass --skip-tracelens (when TraceLens already run).
    """
    logger.info("Running single-config analysis...")
    logger.info(f"  Experiment directory: {experiment_dir}")

    pairs = parse_config_pairs(config_pairs)
    skip_tracelens_flag = " --skip-tracelens" if skip_tracelens else ""
    single_config_output_dir = f"{experiment_dir}/single_config_results"

    for cu, threads in pairs:
        config_dir_name = get_config_dir_name(cu, threads)
        test_dir = f"{experiment_dir}/{config_dir_name}"
        single_output = f"{single_config_output_dir}/single_config_{config_dir_name}"

        logger.info(f"  Processing {config_dir_name}...")

        summary_script = f"""
            set -e
            export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH

            if [ ! -d "{test_dir}" ]; then
                echo "Warning: Directory not found: {test_dir}"
                exit 0
            fi

            mkdir -p "{single_output}"

            echo "Generating summary for {config_dir_name}..."
            aorta-report pipeline summary --test "{test_dir}" --output "{single_output}"{skip_tracelens_flag}
        """

        try:
            docker_exec(
                container_name,
                summary_script,
                logger,
                workdir="/workspace/aorta",
                check=True,
            )
            logger.info(f"    ✓ Summary generated for {config_dir_name}")
        except Exception as e:
            logger.warning(f"    ⚠ Summary generation failed for {config_dir_name}: {e}")

    logger.info("  ✓ Single-config analysis completed")


def stage_pairwise_comparison(
    container_name: str,
    experiment_dir: str,
    config_pairs: str,
    baseline: str,
    logger: logging.Logger,
    baseline_label: str = "",
    test_label: str = "",
) -> None:
    """Run pairwise comparison analysis (baseline vs each config).

    Uses `aorta-report pipeline summary --baseline --test --skip-tracelens`.
    Always passes --skip-tracelens (expects tracelens from single-config stage).

    Args:
        container_name: Name of the Docker container.
        experiment_dir: Path to the experiment directory (relative to workspace).
        config_pairs: Space-separated CU,threads pairs.
        baseline: Baseline configuration (CU,threads format, e.g., "56,256").
        logger: Logger instance.
        baseline_label: Optional label for baseline in reports.
        test_label: Optional label for test in reports.
    """
    logger.info("Running pairwise comparison analysis...")
    logger.info(f"  Experiment directory: {experiment_dir}")
    logger.info(f"  Baseline configuration: {baseline}")

    pairs = parse_config_pairs(config_pairs)
    baseline_parts = baseline.split(",")
    if len(baseline_parts) != 2:
        raise RuntimeError(f"Invalid baseline format: {baseline}. Expected 'CU,threads'")

    baseline_cu, baseline_threads = baseline_parts
    baseline_dir_name = get_config_dir_name(baseline_cu, baseline_threads)
    baseline_path = f"{experiment_dir}/{baseline_dir_name}"
    comparison_output_dir = f"{experiment_dir}/comparison_results"

    for cu, threads in pairs:
        if cu == baseline_cu and threads == baseline_threads:
            logger.debug(f"  Skipping baseline: {baseline_cu},{baseline_threads}")
            continue

        config_dir_name = get_config_dir_name(cu, threads)
        test_dir = f"{experiment_dir}/{config_dir_name}"
        comparison_output = f"{comparison_output_dir}/{baseline_dir_name}_vs_{config_dir_name}"

        logger.info(f"  Comparing: {baseline_dir_name} vs {config_dir_name}...")

        label_args = ""
        if baseline_label:
            label_args += f' --baseline-label "{baseline_label}"'
        if test_label:
            label_args += f' --test-label "{test_label}"'

        comparison_script = f"""
            set -e
            export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH

            if [ ! -d "{test_dir}" ]; then
                echo "Warning: Test directory not found: {test_dir}"
                exit 0
            fi

            if [ ! -d "{baseline_path}" ]; then
                echo "Warning: Baseline directory not found: {baseline_path}"
                exit 0
            fi

            mkdir -p "{comparison_output}"

            echo "========================================"
            echo "Comparing baseline ({baseline_cu} cu, {baseline_threads} threads) vs test ({cu} cu, {threads} threads)"
            echo "========================================"

            aorta-report pipeline summary \\
                --baseline "{baseline_path}" \\
                --test "{test_dir}" \\
                --skip-tracelens \\
                --output "{comparison_output}"{label_args}

            echo "Comparison complete!"
        """

        try:
            docker_exec(
                container_name,
                comparison_script,
                logger,
                workdir="/workspace/aorta",
                check=True,
            )
            logger.info(f"    ✓ Comparison complete: {baseline_dir_name} vs {config_dir_name}")
        except Exception as e:
            logger.warning(f"    ⚠ Comparison failed for {baseline_dir_name} vs {config_dir_name}: {e}")

    logger.info("  ✓ Pairwise comparison completed")


def stage_compare_all_analysis(
    container_name: str,
    experiment_dir: str,
    config_pairs: str,
    baseline: str,
    logger: logging.Logger,
) -> None:
    """Run compare-all-runs analysis across all configurations.

    This stage compares all configurations together using the
    `run_full_analysis.py --compare-all-runs` command.

    Args:
        container_name: Name of the Docker container.
        experiment_dir: Path to the experiment directory (relative to workspace).
        config_pairs: Space-separated CU,threads pairs.
        baseline: Baseline configuration (CU,threads format).
        logger: Logger instance.

    Raises:
        RuntimeError: If analysis fails.
    """
    logger.info("Running compare-all-runs analysis...")
    logger.info(f"  Experiment directory: {experiment_dir}")
    logger.info(f"  Baseline: {baseline}")

    # Parse configurations
    pairs = parse_config_pairs(config_pairs)
    baseline_parts = baseline.split(",")
    if len(baseline_parts) != 2:
        raise RuntimeError(f"Invalid baseline format: {baseline}. Expected 'CU,threads'")

    baseline_cu, baseline_threads = baseline_parts
    baseline_dir_name = get_config_dir_name(baseline_cu, baseline_threads)
    baseline_path = f"{experiment_dir}/{baseline_dir_name}"

    # Build list of test directories (excluding baseline)
    test_dirs = []
    for cu, threads in pairs:
        if cu == baseline_cu and threads == baseline_threads:
            continue
        config_dir_name = get_config_dir_name(cu, threads)
        test_dirs.append(f"{experiment_dir}/{config_dir_name}")

    if not test_dirs:
        logger.warning("  No test directories found (excluding baseline)")
        return

    test_dirs_str = " ".join(test_dirs)
    output_dir = f"{experiment_dir}/compare_all_runs"

    logger.info(f"  Baseline: {baseline_path}")
    logger.info(f"  Test directories: {len(test_dirs)}")
    for td in test_dirs:
        logger.info(f"    - {td}")

    compare_all_script = f"""
        set -e
        export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH

        if [ ! -d "{baseline_path}" ]; then
            echo "Error: Baseline directory not found: {baseline_path}"
            exit 1
        fi

        mkdir -p "{output_dir}"

        echo "========================================"
        echo "Comparing all runs together"
        echo "  Baseline: {baseline_path}"
        echo "  Test directories: {test_dirs_str}"
        echo "========================================"

        python scripts/tracelens_single_config/run_full_analysis.py \\
            --baseline "{baseline_path}" \\
            --test {test_dirs_str} \\
            --output "{output_dir}" \\
            --skip-tracelens \\
            --compare-all-runs

        echo "Compare-all-runs analysis complete!"
    """

    try:
        docker_exec(
            container_name,
            compare_all_script,
            logger,
            workdir="/workspace/aorta",
            check=True,
        )
        logger.info("  ✓ Compare-all-runs analysis completed")
    except Exception as e:
        raise RuntimeError(f"Compare-all-runs analysis failed: {e}") from e


def _extract_date_label(experiment_dir: str) -> str:
    """Extract a date label from an experiment directory path.

    Handles two formats:
    - experiments/rccl_warp_speed_20260223_065820 -> "2026-02-23"
    - .aorta-report/2026-02-22/rccl-warp-speed -> "2026-02-22"

    Args:
        experiment_dir: Path to experiment directory.

    Returns:
        Date string in YYYY-MM-DD format, or directory name if extraction fails.
    """
    import re

    # Try to extract from rccl_warp_speed_YYYYMMDD_HHMMSS format
    match = re.search(r"rccl_warp_speed_(\d{4})(\d{2})(\d{2})_\d{6}", experiment_dir)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # Try to extract YYYY-MM-DD from path (aorta-report format)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", experiment_dir)
    if match:
        return match.group(1)

    # Fallback: use the last directory component
    return Path(experiment_dir).name


def stage_cross_timestamp_comparison(
    container_name: str,
    current_experiment_dir: str,
    baseline_experiment_dir: str,
    config_pairs: str,
    logger: logging.Logger,
    baseline_label: str = "",
    test_label: str = "",
) -> None:
    """Run cross-timestamp comparison between two experiment runs.

    Compares each configuration between the baseline (older) experiment
    and the current (newer) experiment to track performance changes over time.

    Args:
        container_name: Name of the Docker container.
        current_experiment_dir: Path to current experiment directory.
        baseline_experiment_dir: Path to baseline (older) experiment directory.
        config_pairs: Space-separated CU,threads pairs.
        logger: Logger instance.
        baseline_label: Optional label for baseline in reports (auto-generated if empty).
        test_label: Optional label for test in reports (auto-generated if empty).

    Raises:
        RuntimeError: If comparison fails.
    """
    logger.info("Running cross-timestamp comparison...")
    logger.info(f"  Current experiment: {current_experiment_dir}")
    logger.info(f"  Baseline experiment: {baseline_experiment_dir}")

    # Auto-generate labels from directory names if not provided
    if not baseline_label:
        baseline_label = _extract_date_label(baseline_experiment_dir)
        logger.info(f"  Baseline label (auto): {baseline_label}")
    else:
        logger.info(f"  Baseline label: {baseline_label}")

    if not test_label:
        test_label = _extract_date_label(current_experiment_dir)
        logger.info(f"  Test label (auto): {test_label}")
    else:
        logger.info(f"  Test label: {test_label}")

    # Parse configurations
    pairs = parse_config_pairs(config_pairs)

    output_dir = f"{current_experiment_dir}/cross_timestamp_comparison"

    # Build label arguments (always provided now, either explicit or auto-generated)
    label_args = f' --baseline-label "{baseline_label}" --test-label "{test_label}"'

    for cu, threads in pairs:
        config_dir_name = get_config_dir_name(cu, threads)
        baseline_test_dir = f"{baseline_experiment_dir}/{config_dir_name}"
        current_test_dir = f"{current_experiment_dir}/{config_dir_name}"
        comparison_output = f"{output_dir}/{config_dir_name}"

        logger.info(f"  Comparing timestamps for {config_dir_name}...")

        comparison_script = f"""
            set -e
            export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH

            if [ ! -d "{baseline_test_dir}" ]; then
                echo "Warning: Baseline directory not found: {baseline_test_dir}"
                exit 0
            fi

            if [ ! -d "{current_test_dir}" ]; then
                echo "Warning: Current directory not found: {current_test_dir}"
                exit 0
            fi

            mkdir -p "{comparison_output}"

            echo "========================================"
            echo "Cross-timestamp comparison for {config_dir_name}"
            echo "  Baseline (older): {baseline_test_dir}"
            echo "  Current (newer): {current_test_dir}"
            echo "========================================"

            aorta-report pipeline summary \\
                --baseline "{baseline_test_dir}" \\
                --test "{current_test_dir}" \\
                --skip-tracelens \\
                --output "{comparison_output}"{label_args}

            echo "Cross-timestamp comparison complete for {config_dir_name}!"
        """

        try:
            docker_exec(
                container_name,
                comparison_script,
                logger,
                workdir="/workspace/aorta",
                check=True,
            )
            logger.info(f"    ✓ Cross-timestamp comparison complete: {config_dir_name}")
        except Exception as e:
            logger.warning(f"    ⚠ Cross-timestamp comparison failed for {config_dir_name}: {e}")

    logger.info("  ✓ Cross-timestamp comparison completed")

