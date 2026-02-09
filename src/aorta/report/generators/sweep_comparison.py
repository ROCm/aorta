"""Generate GEMM variance comparison HTML for two sweeps."""

from pathlib import Path
from typing import Dict

from ..templates.sweep_comparison_template import get_comparison_template


# Expected plot files for sweep comparison
SWEEP_PLOT_FILES = {
    "threads": "variance_by_threads_boxplot.png",
    "channels": "variance_by_channels_boxplot.png",
    "ranks": "variance_by_ranks_boxplot.png",
    "violin": "variance_violin_combined.png",
    "interaction": "variance_thread_channel_interaction.png",
}


def generate_sweep_comparison(
    plots_dir1: Path,
    plots_dir2: Path,
    label1: str,
    label2: str,
    sweep1_path: Path,
    sweep2_path: Path,
    output: Path,
    found1: Dict[str, Path],
    found2: Dict[str, Path],
    verbose: bool = False,
) -> Path:
    """Generate sweep comparison HTML report.

    Args:
        plots_dir1: Plots directory for sweep 1
        plots_dir2: Plots directory for sweep 2
        label1: Label for sweep 1
        label2: Label for sweep 2
        sweep1_path: Original sweep 1 directory path
        sweep2_path: Original sweep 2 directory path
        output: Output HTML file path
        found1: Dict of found plots for sweep 1 (key -> path)
        found2: Dict of found plots for sweep 2 (key -> path)
        verbose: Enable verbose output

    Returns:
        Path to generated HTML file
    """
    # Import here to avoid circular import
    from .html_generator import image_to_base64

    # Convert found images to base64
    image_data = {}

    print(f"Encoding images for {label1}...")
    for key, path in found1.items():
        b64 = image_to_base64(path)
        if b64:
            image_data[f"{key}_sweep1"] = b64
        else:
            image_data[f"{key}_sweep1"] = ""

    print(f"Encoding images for {label2}...")
    for key, path in found2.items():
        b64 = image_to_base64(path)
        if b64:
            image_data[f"{key}_sweep2"] = b64
        else:
            image_data[f"{key}_sweep2"] = ""

    # Fill in empty strings for missing images
    for key in SWEEP_PLOT_FILES.keys():
        if f"{key}_sweep1" not in image_data:
            image_data[f"{key}_sweep1"] = ""
        if f"{key}_sweep2" not in image_data:
            image_data[f"{key}_sweep2"] = ""

    # Generate HTML using template
    html_content = get_comparison_template(
        label1=label1,
        label2=label2,
        sweep1_path=sweep1_path,
        sweep2_path=sweep2_path,
        image_data=image_data,
    )

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_size = output.stat().st_size / 1024 / 1024
    print(f"\n✓ HTML report created: {output}")
    print(f"  File size: {file_size:.2f} MB")

    return output
