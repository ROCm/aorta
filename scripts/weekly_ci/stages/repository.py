"""
Repository management stages for Weekly CI Kickoff.

This module provides:
- aorta-report repository checkout/update
- Results pushing to aorta-report
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from ..utils import extract_date_from_experiment_dir, run_command

# Trace files larger than this (bytes) are gzipped before upload
TRACE_GZIP_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB

# File extensions treated as trace files (case-insensitive)
TRACE_EXTENSIONS = (".pt.trace.json", ".trace.json", ".traces.json", ".pt.traces.json", "trace.json", "pt.trace.json")


def stage_checkout_aorta_report(
    aorta_report_path: str,
    repo_root: Path,
    logger: logging.Logger,
    git_token: Optional[str] = None,
) -> Path:
    """Checkout or update the aorta-report repository.

    If the repository exists, performs a git fetch and pull.
    If it doesn't exist, clones it from GitHub.

    Args:
        aorta_report_path: Path to the aorta-report repository (relative or absolute).
        repo_root: Path to the aorta repository root.
        logger: Logger instance.
        git_token: Optional GitHub token for authentication.

    Returns:
        Path to the aorta-report repository.

    Raises:
        RuntimeError: If checkout/update fails.
    """
    logger.info("Checking out aorta-report repository...")

    # Resolve path
    if os.path.isabs(aorta_report_path):
        aorta_report_dir = Path(aorta_report_path)
    else:
        aorta_report_dir = repo_root / aorta_report_path

    aorta_report_dir = aorta_report_dir.resolve()
    logger.info(f"  Target path: {aorta_report_dir}")

    if aorta_report_dir.exists() and (aorta_report_dir / ".git").exists():
        # Repository exists, update it
        logger.info("  Repository exists, updating...")
        try:
            run_command(
                "git fetch origin",
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )
            run_command(
                "git pull --rebase origin main",
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )
            logger.info("  ✓ Repository updated successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to update aorta-report repository: {e}") from e
    else:
        # Clone repository
        logger.info("  Repository not found, cloning...")

        # Create parent directory if needed
        aorta_report_dir.parent.mkdir(parents=True, exist_ok=True)

        # Build clone URL
        if git_token:
            clone_url = f"https://{git_token}@github.com/ROCm/aorta-report.git"
        else:
            # Try environment variable
            env_token = os.environ.get("AORTA_REPORT_GITHUB_TOKEN", "")
            if env_token:
                clone_url = f"https://{env_token}@github.com/ROCm/aorta-report.git"
            else:
                # Use SSH (requires SSH key setup)
                clone_url = "git@github.com:ROCm/aorta-report.git"
                logger.info("  Note: Using SSH authentication (no token provided)")

        try:
            run_command(
                f"git clone {clone_url} {aorta_report_dir}",
                logger,
                capture_output=True,
                check=True,
            )
            logger.info("  ✓ Repository cloned successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clone aorta-report repository: {e}") from e

    return aorta_report_dir


def stage_push_results(
    aorta_report_dir: Path,
    experiment_dir: str,
    repo_root: Path,
    logger: logging.Logger,
    report_label: Optional[str] = None,
    git_user_name: str = "Weekly CI Bot",
    git_user_email: str = "weekly-ci@aorta.local",
) -> None:
    """Copy experiment results to aorta-report and push.

    Creates a date-based directory structure in aorta-report:
    aorta-report/YYYY-MM-DD/rccl-warp-speed/

    Args:
        aorta_report_dir: Path to the aorta-report repository.
        experiment_dir: Path to the experiment directory (relative to repo_root).
        repo_root: Path to the aorta repository root.
        logger: Logger instance.
        report_label: Optional override for directory name (default: date from experiment dir).
        git_user_name: Git user name for commit.
        git_user_email: Git user email for commit.

    Raises:
        RuntimeError: If push fails.
    """
    logger.info("Pushing results to aorta-report...")

    # Use report_label override or extract date from experiment directory
    if report_label and report_label.strip():
        date_str = report_label.strip()
    else:
        date_str = extract_date_from_experiment_dir(experiment_dir, logger)
    target_dir = aorta_report_dir / date_str / "rccl-warp-speed"

    logger.info(f"  Date directory: {date_str}")
    logger.info(f"  Target path: {target_dir}")

    # Source experiment directory
    source_dir = repo_root / experiment_dir
    if not source_dir.exists():
        raise RuntimeError(f"Experiment directory not found: {source_dir}")

    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy experiment results
        logger.info("  Copying experiment results...")
        _copy_experiment_results(source_dir, target_dir, logger)

        # Configure git user
        run_command(
            f'git config user.name "{git_user_name}"',
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )
        run_command(
            f'git config user.email "{git_user_email}"',
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )

        # Pull latest changes
        logger.info("  Pulling latest changes...")
        run_command(
            "git pull --rebase origin main",
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=False,  # May fail if no upstream
        )

        # Add and commit (include README.md for dashboard updates)
        logger.info("  Committing changes...")
        run_command(
            f"git add {date_str} README.md",
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )

        # Check if there are changes to commit
        result = run_command(
            "git status --porcelain",
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )

        if result.stdout.strip():
            run_command(
                f'git commit -m "Add RCCL warp speed results for {date_str} and update dashboard"',
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )

            # Push
            logger.info("  Pushing to remote...")
            run_command(
                "git push origin main",
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )
            logger.info("  ✓ Results pushed successfully")
        else:
            logger.info("  No changes to commit (results may already exist)")

    except Exception as e:
        raise RuntimeError(f"Failed to push results: {e}") from e


def _is_trace_file(path: Path) -> bool:
    """Check if path is a trace file (e.g. .pt.trace.json)."""
    name_lower = path.name.lower()
    return any(name_lower.endswith(ext) for ext in TRACE_EXTENSIONS)


def _copy_file_with_gzip(
    src: Path,
    dest: Path,
    logger: logging.Logger,
) -> bool:
    """Copy a file, gzipping if it's a large trace file (>100MB).

    Returns True if copied successfully.
    """
    try:
        if _is_trace_file(src) and src.stat().st_size > TRACE_GZIP_THRESHOLD_BYTES:
            dest_path = dest.with_suffix(dest.suffix + ".gz")
            with open(src, "rb") as f_in:
                with gzip.open(dest_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            size_mb = src.stat().st_size / (1024 * 1024)
            gz_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.info(f"    Gzipped trace {src.name}: {size_mb:.1f}MB -> {gz_mb:.1f}MB")
            return True
        else:
            shutil.copy2(src, dest)
            return True
    except Exception as e:
        logger.warning(f"    Failed to copy {src.name}: {e}")
        return False


def _copy_tree_with_gzip(
    source_dir: Path,
    target_dir: Path,
    exclude_patterns: list[str],
    logger: logging.Logger,
) -> int:
    """Recursively copy directory tree, gzipping large trace files.

    Returns count of items copied.
    """
    count = 0

    for item in source_dir.iterdir():
        if item.name.startswith("."):
            continue

        dest = target_dir / item.name

        # Check exclusions
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern.endswith("/"):
                if item.is_dir() and item.name == pattern[:-1]:
                    should_exclude = True
                    break
            elif item.match(pattern):
                should_exclude = True
                break

        if should_exclude:
            logger.debug(f"    Excluding: {item.name}")
            continue

        try:
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                count += _copy_tree_with_gzip(item, dest, exclude_patterns, logger)
            else:
                if _copy_file_with_gzip(item, dest, logger):
                    count += 1
        except Exception as e:
            logger.warning(f"    Failed to copy {item.name}: {e}")

    return count


def _copy_experiment_results(
    source_dir: Path,
    target_dir: Path,
    logger: logging.Logger,
) -> None:
    """Copy experiment results to target directory.

    Copies important result files. Trace files (.pt.trace.json, .trace.json)
    larger than 100MB are gzipped before copy to reduce repository size.

    Args:
        source_dir: Source experiment directory.
        target_dir: Target directory in aorta-report.
        logger: Logger instance.
    """
    # Patterns to exclude (large files we don't want in the report)
    exclude_patterns = [
        "*.rpd",  # ROCm profiler data
        "*.sqlite",
        "*.db",
        "traces/",  # Raw trace directories
    ]

    # Start fresh: remove existing content so we don't merge with old runs
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    items_copied = _copy_tree_with_gzip(source_dir, target_dir, exclude_patterns, logger)

    logger.info(f"  Copied {items_copied} items")

