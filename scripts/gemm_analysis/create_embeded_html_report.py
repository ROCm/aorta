#!/usr/bin/env python3
"""
Create a self-contained HTML report comparing two experiment sweeps.
Embeds all images as base64 for easy sharing.
"""

import base64
import argparse
from pathlib import Path
from html_template import get_comparison_template

def image_to_base64(image_path):
    """Convert an image file to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image not found: {image_path}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create HTML comparison report for two experiment sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two sweeps
  python create_embeded_html_report.py \\
    --sweep1 /home/oyazdanb/aorta/experiments/sweep_20251121_155219 \\
    --sweep2 /home/oyazdanb/aorta/experiments/sweep_20251124_222204 \\
    --output /home/oyazdanb/sweep_comparison.html

  # With custom labels
  python create_embeded_html_report.py \\
    --sweep1 experiments/sweep_20251121_155219 \\
    --sweep2 experiments/sweep_20251124_222204 \\
    --label1 "Base ROCm" \\
    --label2 "ROCm 7.0" \\
    --output comparison_report.html
        """
    )

    parser.add_argument(
        '--sweep1',
        type=Path,
        required=True,
        help='Path to first sweep directory'
    )

    parser.add_argument(
        '--sweep2',
        type=Path,
        required=True,
        help='Path to second sweep directory'
    )

    parser.add_argument(
        '--label1',
        type=str,
        default=None,
        help='Label for first sweep (default: directory name)'
    )

    parser.add_argument(
        '--label2',
        type=str,
        default=None,
        help='Label for second sweep (default: directory name)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output HTML file path (default: sweep_comparison_report.html in current directory)'
    )

    return parser.parse_args()

def get_plot_images(sweep_path):
    """Get paths to all plot images for a sweep"""
    plots_dir = sweep_path / "tracelens_analysis" / "plots"

    return {
        'threads': plots_dir / 'variance_by_threads_boxplot.png',
        'channels': plots_dir / 'variance_by_channels_boxplot.png',
        'ranks': plots_dir / 'variance_by_ranks_boxplot.png',
        'violin': plots_dir / 'variance_violin_combined.png',
        'interaction': plots_dir / 'variance_thread_channel_interaction.png',
    }

def create_html_report(sweep1_path, sweep2_path, label1, label2, output_path):
    """Create HTML report comparing two sweeps"""

    # Get sweep names from paths if labels not provided
    if label1 is None:
        label1 = sweep1_path.name
    if label2 is None:
        label2 = sweep2_path.name

    # Get image paths for both sweeps
    images_sweep1 = get_plot_images(sweep1_path)
    images_sweep2 = get_plot_images(sweep2_path)

    # Convert images to base64
    print("Converting images to base64...")
    print(f"\nSweep 1: {label1}")
    image_data = {}
    for key, path in images_sweep1.items():
        print(f"  Processing: {key}")
        b64 = image_to_base64(path)
        if b64:
            image_data[f'{key}_sweep1'] = f"data:image/png;base64,{b64}"
            print(f"    [OK]")
        else:
            image_data[f'{key}_sweep1'] = ""
            print(f"    [MISSING] {path}")

    print(f"\nSweep 2: {label2}")
    for key, path in images_sweep2.items():
        print(f"  Processing: {key}")
        b64 = image_to_base64(path)
        if b64:
            image_data[f'{key}_sweep2'] = f"data:image/png;base64,{b64}"
            print(f"    [OK]")
        else:
            image_data[f'{key}_sweep2'] = ""
            print(f"    [MISSING] {path}")

    # Create HTML with embedded images
    html_content = get_comparison_template(label1, label2, sweep1_path, sweep2_path, image_data)

    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[OK] HTML report created: {output_path}")
    print(f"     File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
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
    create_html_report(
        args.sweep1,
        args.sweep2,
        args.label1,
        args.label2,
        args.output
    )

    return 0


if __name__ == "__main__":
    exit(main())
