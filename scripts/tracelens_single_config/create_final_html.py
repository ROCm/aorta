"""Main entry point for creating HTML analysis reports."""

import sys
from pathlib import Path
import argparse

# Add scripts directory to path for report_generator imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from report_generator.report_builder import SingleReportBuilder


def get_default_config_path() -> Path:
    """Return the default path to the config JSON file."""
    return Path(__file__).parent.parent / "utils" / "html_report_config.json"


def create_final_html(
    plot_file_path: Path,
    output_path: Path,
    config_path: Path | None = None,
) -> None:
    """Factory function to create and save a report."""
    if config_path is None:
        config_path = get_default_config_path()
    builder = SingleReportBuilder(plot_file_path, output_path, config_path)
    builder.save()


def main():
    parser = argparse.ArgumentParser(
        description="Create a final HTML file for the analysis report."
    )
    parser.add_argument(
        "-p",
        "--plot-files-directory",
        type=Path,
        required=True,
        help="Path to the plot files directory.",
    )
    parser.add_argument(
        "-o", "--output-html", type=Path, default=None, help="Path to the output file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to the JSON config file (default: utils/html_report_config.json).",
    )
    args = parser.parse_args()
    output_path = (
        args.output_html
        if args.output_html
        else args.plot_files_directory.parent / "final_analysis_report.html"
    )
    create_final_html(args.plot_files_directory, output_path, args.config)


if __name__ == "__main__":
    main()
