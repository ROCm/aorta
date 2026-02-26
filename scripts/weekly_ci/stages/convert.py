"""
HTML to Markdown conversion stage for Weekly CI Kickoff.

This module provides:
- Recursive conversion of HTML files to Markdown in an experiment directory
- Standalone invocation for ad-hoc conversion
- Graceful handling when beautifulsoup4 is not available
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def _get_html_to_md_converter():
    """Import convert_file from html_to_md, with graceful handling if unavailable.

    Returns:
        The convert_file function, or None if beautifulsoup4 is not installed.
    """
    try:
        import bs4  # noqa: F401

        # beautifulsoup4 is available
    except ImportError:
        return None

    try:
        # Ensure repo root is on path for standalone invocation
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from scripts.utils.html_to_md import convert_file

        return convert_file
    except ImportError as e:
        if "bs4" in str(e).lower() or "beautifulsoup" in str(e).lower():
            return None
        raise


def convert_html_to_md_in_dir(
    exp_path: Path,
    logger: logging.Logger,
) -> int:
    """Convert all HTML files in a directory tree to Markdown.

    Creates .md alongside each .html. Original HTML files are never removed.

    Args:
        exp_path: Path to the experiment directory.
        logger: Logger instance.

    Returns:
        Number of files converted. Returns 0 if beautifulsoup4 is not available.
    """
    convert_file = _get_html_to_md_converter()
    if convert_file is None:
        logger.warning(
            "  beautifulsoup4 is required for HTML-to-Markdown conversion. "
            "Install with: pip install beautifulsoup4"
        )
        return 0

    if not exp_path.exists() or not exp_path.is_dir():
        logger.warning(f"  Experiment path does not exist or is not a directory: {exp_path}")
        return 0

    html_files = sorted(exp_path.rglob("*.html"))
    if not html_files:
        logger.info("  No HTML files found to convert")
        return 0

    count = 0
    for html_path in html_files:
        try:
            convert_file(html_path)
            count += 1
            logger.debug(f"  Converted: {html_path.relative_to(exp_path)}")
        except Exception as e:
            logger.warning(f"  Failed to convert {html_path}: {e}")

    return count


def stage_convert_html_to_md(
    experiment_dir: str,
    repo_root: Path,
    logger: logging.Logger,
) -> int:
    """Convert all HTML files in the experiment directory to Markdown.

    This stage runs after all comparisons and before copying to aorta-report.
    Creates .md files alongside each .html. Original HTML files are kept.

    Args:
        experiment_dir: Path to the experiment directory (relative to repo_root).
        repo_root: Path to the aorta repository root.
        logger: Logger instance.

    Returns:
        Number of files converted. Returns 0 if skipped or if beautifulsoup4 unavailable.
    """
    logger.info("Converting HTML files to Markdown...")

    exp_path = repo_root / experiment_dir
    count = convert_html_to_md_in_dir(exp_path, logger)

    if count > 0:
        logger.info(f"  ✓ Converted {count} HTML file(s) to Markdown")
    return count


def main() -> int:
    """Standalone entry point for converting HTML to MD in an experiment directory.

    Usage:
        python scripts/weekly_ci/stages/convert.py <experiment_dir>
        python -m scripts.weekly_ci.stages.convert <experiment_dir>

    Returns:
        0 on success, 1 on failure or missing dependency.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HTML files to Markdown in an experiment directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/weekly_ci/stages/convert.py experiments/rccl_warp_speed_20260224_065602
    python -m scripts.weekly_ci.stages.convert experiments/rccl_warp_speed_20260224_065602
        """,
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory (e.g., experiments/rccl_warp_speed_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    exp_path = args.experiment_dir.resolve()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    convert_file = _get_html_to_md_converter()
    if convert_file is None:
        logger.error(
            "beautifulsoup4 is required. Install with: pip install beautifulsoup4"
        )
        return 1

    if not exp_path.exists() or not exp_path.is_dir():
        logger.error(f"Directory not found or not a directory: {exp_path}")
        return 1

    count = convert_html_to_md_in_dir(exp_path, logger)
    if count > 0:
        logger.info(f"Converted {count} file(s)")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
