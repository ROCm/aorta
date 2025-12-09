from pathlib import Path
import base64
import argparse

from html_report_config import (
    HTML_HEADER,
    HTML_FOOTER,
    OVERALL_GPU_CHARTS,
    CROSS_RANK_CHARTS,
    NCCL_CHARTS,
)


def get_image_base64(image_path):
    """Read an image file and return its base64-encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error getting image data from {image_path}: {e}")
        return None


def create_chart_html(plot_dir, chart_config):
    """Generate HTML for a single chart with title, image, and description."""
    image_data = get_image_base64(plot_dir / chart_config["file"])
    if image_data is None:
        return ""
    return f"""
    <h4> {chart_config['name']} </h4>
    <img src="data:image/png;base64,{image_data}" alt="{chart_config['alt']}" class="chart-image">
    {chart_config['description']}
    """


def create_section_html(title, plot_dir, charts):
    """Generate HTML for a complete section with multiple charts."""
    section_html = f"""
    <h3> {title} </h3>
    """
    for chart in charts:
        section_html += create_chart_html(plot_dir, chart)
    return section_html


def create_final_html(plot_file_path, output_path):
    html_body = """
<body>

<h1> Performance Analysis Report </h1>

<hr>

<h2> Executive Summary </h2>

Comparison of GPU performance metrics between baseline and Test
implementations across 8 ranks.
"""

    # Build all sections
    sections = [
        create_section_html(
            "1. Overall GPU Metrics Comparison", plot_file_path, OVERALL_GPU_CHARTS
        ),
        create_section_html(
            "2. Cross-Rank Performance Comparison", plot_file_path, CROSS_RANK_CHARTS
        ),
        create_section_html(
            "3. NCCL Collective Operations Analysis", plot_file_path, NCCL_CHARTS
        ),
    ]

    final_html = HTML_HEADER + html_body + "".join(sections) + HTML_FOOTER
    with open(output_path, "w") as f:
        f.write(final_html)
    print(f"Final HTML file created at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a final HTML file for the analysis report."
    )
    parser.add_argument(
        "-p",
        "--plot-files-directory",
        type=Path,
        required=True,
        help="Path to the plot files direcotry.",
    )
    parser.add_argument(
        "-o", "--output-html", type=None, default=None, help="Path to the output file."
    )
    args = parser.parse_args()
    output_path = (
        args.output_html
        if args.output_html
        else args.plot_files_directory.parent / "final_analysis_report.html"
    )
    create_final_html(args.plot_files_directory, output_path)


if __name__ == "__main__":
    main()
