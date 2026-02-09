"""Generate GEMM variance analysis HTML report for single analysis."""

from pathlib import Path
from typing import Dict, Optional

from ..templates.gemm_report_template import get_gemm_report_template


# Expected plot files for GEMM analysis (same as sweep)
GEMM_PLOT_FILES = {
    "threads": "variance_by_threads_boxplot.png",
    "channels": "variance_by_channels_boxplot.png",
    "ranks": "variance_by_ranks_boxplot.png",
    "violin": "variance_violin_combined.png",
    "interaction": "variance_thread_channel_interaction.png",
}


def generate_gemm_report(
    plots_dir: Path,
    output: Path,
    label: str,
    sweep_path: Path,
    found: Dict[str, Path],
    csv_path: Optional[Path] = None,
    verbose: bool = False,
) -> Path:
    """Generate GEMM variance analysis HTML report.

    Args:
        plots_dir: Directory containing GEMM variance plots
        output: Output HTML file path
        label: Label for this analysis
        sweep_path: Original sweep directory path
        found: Dict of found plots (key -> path)
        csv_path: Optional path to the CSV data file
        verbose: Enable verbose output

    Returns:
        Path to generated HTML file
    """
    # Import here to avoid circular import
    from .html_generator import image_to_base64

    # Convert found images to base64
    image_data = {}

    if verbose:
        print(f"Encoding images from {plots_dir}...")

    for key, path in found.items():
        b64 = image_to_base64(path)
        if b64:
            image_data[key] = b64
            if verbose:
                print(f"  [✓] {key}: {path.name}")
        else:
            image_data[key] = ""
            if verbose:
                print(f"  [✗] {key}: failed to encode")

    # Fill in empty strings for missing images
    for key in GEMM_PLOT_FILES.keys():
        if key not in image_data:
            image_data[key] = ""

    # Generate HTML using template
    html_content = get_gemm_report_template(
        label=label,
        sweep_path=str(sweep_path),
        image_data=image_data,
        csv_path=str(csv_path) if csv_path else None,
    )

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_content)

    if verbose:
        print(f"\nHTML report written to: {output}")

    return output
