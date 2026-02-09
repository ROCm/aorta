"""Generate performance analysis HTML report."""

from pathlib import Path
from typing import Dict

from ..templates.performance_report_template import (
    HTML_HEADER,
    HTML_FOOTER,
    OVERALL_GPU_CHARTS,
    CROSS_RANK_CHARTS,
    NCCL_CHARTS,
)


# Build expected plot files from chart configurations
PERFORMANCE_PLOT_FILES = {}

# Add files from each chart category
for chart in OVERALL_GPU_CHARTS:
    key = chart["file"].replace(".png", "").replace("_", "-")
    PERFORMANCE_PLOT_FILES[key] = chart["file"]

for chart in CROSS_RANK_CHARTS:
    key = chart["file"].replace(".png", "").replace("_", "-")
    PERFORMANCE_PLOT_FILES[key] = chart["file"]

for chart in NCCL_CHARTS:
    key = chart["file"].replace(".png", "").replace("_", "-")
    PERFORMANCE_PLOT_FILES[key] = chart["file"]


def create_chart_html(chart_config: dict, found: Dict[str, Path]) -> str:
    """Generate HTML for a single chart."""
    # Import here to avoid circular import
    from .html_generator import image_to_base64

    filename = chart_config["file"]
    key = filename.replace(".png", "").replace("_", "-")

    if key not in found:
        # Chart image not found, show placeholder
        return f"""
    <h4>{chart_config['name']}</h4>
    <div class="missing-chart">Image not available: {filename}</div>
    <p>{chart_config['description']}</p>
    """

    image_data = image_to_base64(found[key])
    if image_data is None:
        return f"""
    <h4>{chart_config['name']}</h4>
    <div class="missing-chart">Failed to encode: {filename}</div>
    <p>{chart_config['description']}</p>
    """

    return f"""
    <h4>{chart_config['name']}</h4>
    <img src="{image_data}" alt="{chart_config['alt']}">
    <p>{chart_config['description']}</p>
    """


def create_section_html(title: str, charts: list, found: Dict[str, Path]) -> str:
    """Generate HTML for a section with multiple charts."""
    section_html = f"<h3>{title}</h3>\n"
    for chart in charts:
        section_html += create_chart_html(chart, found)
    return section_html


def generate_performance_report(
    plots_dir: Path,
    output: Path,
    found: Dict[str, Path],
    verbose: bool = False,
) -> Path:
    """Generate performance analysis HTML report.

    Args:
        plots_dir: Directory containing pre-generated plots
        output: Output HTML file path
        found: Dict of found plots (key -> path)
        verbose: Enable verbose output

    Returns:
        Path to generated HTML file
    """
    print("Generating performance report...")

    html_body = """
<body>

<h1>Performance Analysis Report</h1>

<hr>

<h2>Executive Summary</h2>

<p>Comparison of GPU performance metrics between baseline and test
implementations across 8 ranks.</p>
"""

    # Build all sections
    sections = [
        create_section_html(
            "1. Overall GPU Metrics Comparison", OVERALL_GPU_CHARTS, found
        ),
        create_section_html(
            "2. Cross-Rank Performance Comparison", CROSS_RANK_CHARTS, found
        ),
        create_section_html(
            "3. NCCL Collective Operations Analysis", NCCL_CHARTS, found
        ),
    ]

    final_html = HTML_HEADER + html_body + "".join(sections) + HTML_FOOTER

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(final_html)

    file_size = output.stat().st_size / 1024 / 1024
    print(f"\n✓ Performance report created: {output}")
    print(f"  File size: {file_size:.2f} MB")

    return output
