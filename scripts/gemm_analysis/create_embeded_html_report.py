#!/usr/bin/env python3
"""
Create a self-contained HTML report comparing two experiment sweeps.
Embeds all images as base64 for easy sharing.

Uses the report_generator framework with ComparisonReportBuilder.

TODO: Future enhancement - support multiple sweep comparisons using comma-separated
      input (e.g., --sweeps sweep1,sweep2,sweep3) for N-way comparisons.
      Current implementation focuses on pairwise comparison which covers the most
      common use case of A/B testing.
"""

import sys
import argparse
from pathlib import Path

# Add scripts directory to path for report_generator imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from report_generator.comparison_builder import ComparisonReportBuilder


def get_default_config_path() -> Path:
    """Return the default path to the config JSON file."""
    return Path(__file__).parent.parent / "utils" / "gemm_comparison_config.json"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create HTML comparison report for two experiment sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two sweeps
  python create_embeded_html_report.py \\
    --sweep1 experiments/sweep_20251121_155219 \\
    --sweep2 experiments/sweep_20251124_222204 \\
    --output sweep_comparison.html

  # With custom labels
  python create_embeded_html_report.py \\
    --sweep1 experiments/sweep_20251121_155219 \\
    --sweep2 experiments/sweep_20251124_222204 \\
    --label1 "Base ROCm" \\
    --label2 "ROCm 7.0" \\
    --output comparison_report.html
        """,
    )

    parser.add_argument("--sweep1", type=Path, required=True, help="Path to first sweep directory")

    parser.add_argument("--sweep2", type=Path, required=True, help="Path to second sweep directory")

    parser.add_argument(
        "--label1", type=str, default=None, help="Label for first sweep (default: directory name)"
    )

    parser.add_argument(
        "--label2", type=str, default=None, help="Label for second sweep (default: directory name)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file path (default: sweep_comparison_report.html in current directory)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file (default: utils/gemm_comparison_config.json)",
    )

    return parser.parse_args()


def create_comparison_report(
    sweep1_path: Path,
    sweep2_path: Path,
    output_path: Path,
    config_path: Path | None = None,
    label1: str | None = None,
    label2: str | None = None,
) -> Path:
    """Create HTML report comparing two sweeps using ComparisonReportBuilder."""
    if config_path is None:
        config_path = get_default_config_path()

    builder = ComparisonReportBuilder(
        sweep1_path=sweep1_path,
        sweep2_path=sweep2_path,
        output_path=output_path,
        config_path=config_path,
        label1=label1,
        label2=label2,
    )
    builder.save()
    return output_path


def main():
    """Main entry point"""
    args = parse_args()

    # Validate sweep directories exist
    if not args.sweep1.exists():
        print(f"Error: Sweep 1 directory not found: {args.sweep1}")
        return 1

    if not args.sweep2.exists():
        print(f"Error: Sweep 2 directory not found: {args.sweep2}")
        return 1

    # Set default output path if not specified
    if args.output is None:
        args.output = Path.cwd() / "sweep_comparison_report.html"

    print("=" * 70)
    print("GEMM Sweep Comparison HTML Report Generator")
    print("=" * 70)
    print(f"Sweep 1: {args.sweep1}")
    print(f"Sweep 2: {args.sweep2}")
    print(f"Output:  {args.output}")
    print()

    # Create the report
    create_comparison_report(
        sweep1_path=args.sweep1,
        sweep2_path=args.sweep2,
        output_path=args.output,
        config_path=args.config,
        label1=args.label1,
        label2=args.label2,
    )

    return 0


if __name__ == "__main__":
    exit(main())
