"""
Unified HTML report generator.

Routes to appropriate generator based on mode:
- sweep: GEMM variance comparison between two sweeps
- performance: GPU/NCCL performance analysis report
"""

import base64
from pathlib import Path
from typing import Optional, Dict, Tuple

from .sweep_comparison import generate_sweep_comparison, SWEEP_PLOT_FILES
from .performance_report import generate_performance_report, PERFORMANCE_PLOT_FILES
from .gemm_report import generate_gemm_report, GEMM_PLOT_FILES


# =============================================================================
# Shared Utilities
# =============================================================================


def image_to_base64(image_path: Path) -> Optional[str]:
    """Convert an image file to base64 data URI string."""
    try:
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            suffix = image_path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
            }
            mime = mime_types.get(suffix, "image/png")
            return f"data:{mime};base64,{b64_data}"
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: Failed to encode image {image_path}: {e}")
        return None


def validate_directory(path: Path, name: str) -> None:
    """Validate that a directory exists."""
    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")


def find_plots_directory(base_path: Path) -> Path:
    """Find the plots directory within a tracelens analysis structure."""
    candidates = [
        base_path / "tracelens_analysis" / "plots",
        base_path / "plots",
        base_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find plots directory in {base_path}")


def check_plots_status(
    plots_dir: Path,
    expected_files: Dict[str, str],
) -> Tuple[Dict[str, Path], Dict[str, str]]:
    """
    Check which expected plots exist and which are missing.

    Args:
        plots_dir: Directory to search for plots
        expected_files: Dict mapping key -> filename

    Returns:
        Tuple of (found_dict, missing_dict)
        - found_dict: key -> full path for found files
        - missing_dict: key -> filename for missing files
    """
    found = {}
    missing = {}

    for key, filename in expected_files.items():
        full_path = plots_dir / filename
        if full_path.exists():
            found[key] = full_path
        else:
            missing[key] = filename

    return found, missing


def print_plot_status(
    expected_files: Dict[str, str],
    found: Dict[str, Path],
    missing: Dict[str, str],
    plots_dir: Path,
    label: str = "",
) -> None:
    """Print clear status of expected plots with found/missing status."""

    header = f"Plot Status{f' ({label})' if label else ''}"
    print(f"\n{'=' * 60}")
    print(f"{header}")
    print(f"{'=' * 60}")
    print(f"Directory: {plots_dir}")
    print(f"\nExpected plots ({len(expected_files)}):")

    for key, filename in expected_files.items():
        if key in found:
            print(f"  [✓ FOUND]   {filename}")
        else:
            print(f"  [✗ MISSING] {filename}")

    # Summary line
    print(f"\nSummary: {len(found)} found, {len(missing)} missing")
    print(f"{'=' * 60}\n")


# =============================================================================
# Main Entry Point
# =============================================================================


def generate_html(
    mode: str,
    output: Path,
    # Sweep mode options
    sweep1: Optional[Path] = None,
    sweep2: Optional[Path] = None,
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    # Performance mode options
    plots_dir: Optional[Path] = None,
    # GEMM mode options
    sweep_dir: Optional[Path] = None,
    label: Optional[str] = None,
    csv_path: Optional[Path] = None,
    # Common options
    verbose: bool = False,
) -> Path:
    """
    Generate HTML report based on mode.

    Args:
        mode: 'sweep', 'performance', or 'gemm'
        output: Output HTML file path
        sweep1: [sweep mode] First sweep directory
        sweep2: [sweep mode] Second sweep directory
        label1: [sweep mode] Label for first sweep
        label2: [sweep mode] Label for second sweep
        plots_dir: [performance/gemm mode] Directory containing pre-generated plots
        sweep_dir: [gemm mode] Original sweep directory
        label: [gemm mode] Label for this analysis
        csv_path: [gemm mode] Optional path to CSV data file
        verbose: Enable verbose output

    Returns:
        Path to generated HTML file
    """
    output = Path(output)

    if mode == "sweep":
        return _generate_sweep_mode(
            sweep1=sweep1,
            sweep2=sweep2,
            label1=label1,
            label2=label2,
            output=output,
            verbose=verbose,
        )

    elif mode == "performance":
        return _generate_performance_mode(
            plots_dir=plots_dir,
            output=output,
            verbose=verbose,
        )

    elif mode == "gemm":
        return _generate_gemm_mode(
            plots_dir=plots_dir,
            sweep_dir=sweep_dir,
            label=label,
            csv_path=csv_path,
            output=output,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Mode Handlers
# =============================================================================


def _generate_sweep_mode(
    sweep1: Optional[Path],
    sweep2: Optional[Path],
    label1: Optional[str],
    label2: Optional[str],
    output: Path,
    verbose: bool,
) -> Path:
    """Handle sweep comparison mode."""
    if not sweep1 or not sweep2:
        raise ValueError("Sweep mode requires both --sweep1 and --sweep2")

    sweep1 = Path(sweep1)
    sweep2 = Path(sweep2)

    validate_directory(sweep1, "Sweep 1")
    validate_directory(sweep2, "Sweep 2")

    # Find plots directories
    plots1 = find_plots_directory(sweep1)
    plots2 = find_plots_directory(sweep2)

    # Use directory names as default labels
    if label1 is None:
        label1 = sweep1.name
    if label2 is None:
        label2 = sweep2.name

    # Check and report plot status for sweep 1
    found1, missing1 = check_plots_status(plots1, SWEEP_PLOT_FILES)
    print_plot_status(
        SWEEP_PLOT_FILES, found1, missing1, plots1, label=f"Sweep 1: {label1}"
    )

    # Check and report plot status for sweep 2
    found2, missing2 = check_plots_status(plots2, SWEEP_PLOT_FILES)
    print_plot_status(
        SWEEP_PLOT_FILES, found2, missing2, plots2, label=f"Sweep 2: {label2}"
    )

    # Delegate to sweep comparison generator
    return generate_sweep_comparison(
        plots_dir1=plots1,
        plots_dir2=plots2,
        label1=label1,
        label2=label2,
        sweep1_path=sweep1,
        sweep2_path=sweep2,
        output=output,
        found1=found1,
        found2=found2,
        verbose=verbose,
    )


def _generate_performance_mode(
    plots_dir: Optional[Path],
    output: Path,
    verbose: bool,
) -> Path:
    """Handle performance report mode."""
    if not plots_dir:
        raise ValueError(
            "Performance mode requires --plots-dir\n"
            "This should point to a directory containing plots generated by:\n"
            "  - 'aorta-report pipeline full', or\n"
            "  - 'aorta-report generate plots'"
        )

    plots_dir = Path(plots_dir)
    validate_directory(plots_dir, "Plots directory")

    # Check and report plot status
    found, missing = check_plots_status(plots_dir, PERFORMANCE_PLOT_FILES)
    print_plot_status(
        PERFORMANCE_PLOT_FILES, found, missing, plots_dir, label="Performance Report"
    )

    # Delegate to performance report generator
    return generate_performance_report(
        plots_dir=plots_dir,
        output=output,
        found=found,
        verbose=verbose,
    )


def _generate_gemm_mode(
    plots_dir: Optional[Path],
    sweep_dir: Optional[Path],
    label: Optional[str],
    csv_path: Optional[Path],
    output: Path,
    verbose: bool,
) -> Path:
    """Handle GEMM variance analysis report mode."""
    if not plots_dir:
        raise ValueError(
            "GEMM mode requires --plots-dir\n"
            "This should point to a directory containing GEMM variance plots generated by:\n"
            "  - 'aorta-report pipeline gemm', or\n"
            "  - 'aorta-report generate plots --type gemm'"
        )

    plots_dir = Path(plots_dir)
    validate_directory(plots_dir, "Plots directory")

    # Use plots_dir parent as sweep_dir if not specified
    if sweep_dir is None:
        sweep_dir = plots_dir.parent
    else:
        sweep_dir = Path(sweep_dir)

    # Use directory name as label if not specified
    if label is None:
        label = sweep_dir.name

    # Check and report plot status
    found, missing = check_plots_status(plots_dir, GEMM_PLOT_FILES)
    print_plot_status(
        GEMM_PLOT_FILES, found, missing, plots_dir, label=f"GEMM Analysis: {label}"
    )

    # Delegate to GEMM report generator
    return generate_gemm_report(
        plots_dir=plots_dir,
        output=output,
        label=label,
        sweep_path=sweep_dir,
        found=found,
        csv_path=csv_path,
        verbose=verbose,
    )

