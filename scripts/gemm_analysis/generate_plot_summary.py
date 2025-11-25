import argparse
from pathlib import Path
import csv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze tracelens comparison report for components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python analyze_gemm_reports.py
  # Specify custom base path
  python analyze_gemm_reports.py --base-path /path/to/tracelens_analysis
  # Specify custom configurations
  python analyze_gemm_reports.py --threads 256 512 --channels 28 42 56 70 --ranks 0 1 2 3 4 5 6 7
  # Extract top 10 kernels instead of top 5
  python analyze_gemm_reports.py --top-k 10
  # Custom output file
  python analyze_gemm_reports.py --output-plot-directory path/to/directory/to/save/plots
        """
    )

    parser.add_argument(
        '--input-plot-dir',
        type=Path,
        default=Path("/home/oyazdanb/aorta/experiments/sweep_20251121_155219/tracelens_analysis/plots"),
        help='Base path to tracelens_analysis directory (default: %(default)s)'
    )

    return parser.parse_args()

def main() :
    args = parse_args()
    input = args.input_plot_dir

    if not input.exists():
        print(f"Error: Input plot directory does not exist: {base_path}")
        return

    html_prembale = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GEMM Kernel Variance Plot </title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
        h2 { 
            border-bottom: 1px solid #ccc; 
            padding-bottom: 5px; 
            margin-top: 40px;
            page-break-before: auto;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
            word-wrap: break-word;
        }
        th { background-color: #f2f2f2; }
        img { 
            max-width: 100%; 
            height: auto; 
            display: block;
            margin: 10px auto;
        }
        .comparison-table { width: 100%; }
        .comparison-table td { width: 50%; vertical-align: top; }
        hr { 
            border: none; 
            border-top: 1px solid #ccc; 
            margin: 30px 0; 
        }
        @media print {
            body { margin: 0; }
            h2 { page-break-before: always; }
            h2:first-of-type { page-break-before: auto; }
        }
    </style>
</head>
<body>
"""

    thread_variance_string = f"""
<h1> Top GEMM Kernels with highest time variance </h1>

<h2> Variance by Thread Count </h2>
<img src="./variance_by_threads_boxplot.png" alt="Threads Base">

<h2> Variance by Channel Count <h2>
<img src="./variance_by_channels_boxplot.png" alt="Channel Base">

<h2> Variance by Rank </h2>
<img src="./variance_by_ranks_boxplot.png" alt="Rank Base">

<h2> Variance Distribution (Violin Plots) </h2>
<img src="./variance_violin_combined.png" alt="Violin">

<h2>Thread-Channel Interaction</h2>
<img src="./variance_thread_channel_interaction.png" alt="Violin">

<hr>

<h2> Kernel Distribution </h2>
"""    

    kernel_section = """
<h3> Kernel Information </h3>
<table>
<tr>
<th> Kernel Id </th>
<th> Kernel Name </th>
</tr>
"""
    kernel_info_file = input / "kernel_info.csv"

    with open(kernel_info_file) as f : 
        kernel_data = csv.DictReader(f) 
        row_ids = [] 
        for row in kernel_data : 
            row_ids.append(row['id'])
            kernel_section += f"<tr><td>{row['id']}</td><td>{row['kernel_name']}</td></tr>\n"
        kernel_section += "</table>\n"
        kernel_section += "<h3> Kernel Plots </h3>"
        for id in row_ids : 
            kernel_section += f"<img src=./Kernel_{id}.png>\n"
    




    output_file = input / "summary.html"
    with open(output_file, "w") as f : 
        f.write(html_prembale)
        f.write("\n")
        f.write(thread_variance_string)
        f.write("\n")
        f.write(kernel_section)
        f.write("\n</body></html>")


if __name__ == "__main__":
    main()