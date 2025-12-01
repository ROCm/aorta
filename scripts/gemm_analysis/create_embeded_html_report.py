#!/usr/bin/env python3
"""
Create a self-contained HTML report comparing two experiment sweeps.
Embeds all images as base64 for easy sharing.
"""

import base64
import os
import argparse
from pathlib import Path

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
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GEMM Kernel Variance - Sweep Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
            color: #2c3e50;
        }}
        h2 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 40px;
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
        }}
        .comparison-table {{ width: 100%; }}
        .comparison-table td {{ width: 50%; vertical-align: top; }}
        .comparison-table th {{
            background-color: #34495e;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .data-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .data-section h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        @media print {{
            body {{ margin: 0; background-color: white; }}
            .container {{ box-shadow: none; }}
            h2 {{ page-break-before: always; }}
            h2:first-of-type {{ page-break-before: auto; }}
        }}
    </style>
</head>
<body>
<div class="container">

<h1>GEMM Kernel Variance - Sweep Comparison</h1>

<div class="info-box">
<p><strong>Visual comparison of GEMM kernel performance variance between two training sweeps.</strong></p>
<p>This report compares kernel variance across different thread counts, channel configurations, and ranks.</p>
</div>

<hr>

<h2>Sweep Information</h2>

<table>
<tr>
<th>Sweep</th>
<th>Path</th>
</tr>
<tr>
<td><strong>Sweep 1</strong></td>
<td>{label1}</td>
</tr>
<tr>
<td><strong>Sweep 2</strong></td>
<td>{label2}</td>
</tr>
</table>

<hr>

<h2>Variance by Thread Count</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('threads_sweep1', '')}" alt="Threads Sweep 1">
</td>
<td>
<img src="{image_data.get('threads_sweep2', '')}" alt="Threads Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Variance by Channel Count</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('channels_sweep1', '')}" alt="Channels Sweep 1">
</td>
<td>
<img src="{image_data.get('channels_sweep2', '')}" alt="Channels Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Variance by Rank</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('ranks_sweep1', '')}" alt="Ranks Sweep 1">
</td>
<td>
<img src="{image_data.get('ranks_sweep2', '')}" alt="Ranks Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Variance Distribution (Violin Plots)</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('violin_sweep1', '')}" alt="Violin Sweep 1">
</td>
<td>
<img src="{image_data.get('violin_sweep2', '')}" alt="Violin Sweep 2">
</td>
</tr>
</table>

<hr>

<h2>Thread-Channel Interaction</h2>

<table class="comparison-table">
<tr>
<th>{label1}</th>
<th>{label2}</th>
</tr>
<tr>
<td>
<img src="{image_data.get('interaction_sweep1', '')}" alt="Interaction Sweep 1">
</td>
<td>
<img src="{image_data.get('interaction_sweep2', '')}" alt="Interaction Sweep 2">
</td>
</tr>
</table>

<hr>

<div class="data-section">
<h2>Data Files Information</h2>

<h3>Sweep 1: {label1}</h3>
<ul>
<li>Path: {sweep1_path}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

<h3>Sweep 2: {label2}</h3>
<ul>
<li>Path: {sweep2_path}</li>
<li>GEMM Variance CSV</li>
<li>TraceLens Reports</li>
<li>Plots</li>
</ul>

<p><em>Note: This is a self-contained HTML report with embedded images. All data is embedded for easy sharing.</em></p>
</div>

</div>
</body>
</html>
"""

    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ HTML report created: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
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
