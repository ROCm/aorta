import os
from pathlib import Path
import base64
import argparse


def get_image_data(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error getting image data: {e}")
        return None


def create_final_html(plot_file_path, output_path):
    html_header = """ <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Analysis Report</title>
    <style>
        body {
            font-family: sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            padding: 20px;
            max-width: 800px;
        }
        h1, h2, h3 {
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
    </head>
    <body>

    <h1> Performance Analysis Report </h1>

    <hr>

    <h2> Executive Summary </h2>

    Comparison of GPU performance metrics
    implementations across 8 ranks.
    """

    summary_section = f"""
    <h3> 1. Overall GPU Metrics Comparison </h3>
    """

    summary_chart = get_image_data(plot_file_path / "improvement_chart.png")
    if summary_chart is not None:
        summary_section += f"""
        <h4> Percentage Change Overview </h4>
        <img src="data:image/png;base64,{summary_chart}" alt="Summary Chart" class="chart-image">
        Overall performance change across key GPU metrics. Positive values indicate improvement
        """
    absolute_time_chart = get_image_data(plot_file_path / "abs_time_comparison.png")
    if absolute_time_chart is not None:
        summary_section += f"""
        <h4> Absolute Time Comparison </h4>
        <img src="data:image/png;base64,{absolute_time_chart}" alt="Absolute Time Comparison" class="chart-image">
        Side-by-side comparison of absolute execution times for all GPU metrics.
        """

    cross_rank_comparison_section = f"""
    <h3> 2. Cross-Rank Performance Comparison </h3>
    """
    gpu_time_heatmap = get_image_data(plot_file_path / "gpu_time_heatmap.png")
    if gpu_time_heatmap is not None:
        cross_rank_comparison_section += f"""
        <h4> Performance Heatmap by Rank  </h4>
        <img src="data:image/png;base64,{gpu_time_heatmap}" alt="GPU Metric Percentage Change by Rank (HeatMap)" class="chart-image">
        Comprehensive heatmap showing percent change for all metrics across all ranks. Green indicates better performance (positive % change).
        """

    item_list = {
        "total_time": {
            "name": "Total Time",
            "description": "Total execution time comparison across all ranks, showing end-to-end performance characteristics.",
            "chart_path": plot_file_path / "total_time_by_rank.png",
        },
        "computation_time": {
            "name": "Computation Time",
            "description": "Pure computation time excluding communication overhead, analyzed per rank.",
            "chart_path": plot_file_path / "computation_time_by_rank.png",
        },
        "total_comm_time": {
            "name": "Communication Time",
            "description": "Total time spent in collective communication operations across ranks.",
            "chart_path": plot_file_path / "total_comm_time_by_rank.png",
        },
        "idle_time": {
            "name": "Idle Time",
            "description": "GPU idle time comparison showing resource utilization efficiency per rank.",
            "chart_path": plot_file_path / "idle_time_by_rank.png",
        },
        "gpu_time_change_percentage_summaryby_rank": {
            "name": "Detailed Percentage Change by Metric",
            "description": "Detailed breakdown of percent change for each metric type across all ranks.",
            "chart_path": plot_file_path
            / "gpu_time_change_percentage_summaryby_rank.png",
        },
    }
    for item in item_list.keys():
        cross_rank_comparison_chart = get_image_data(item_list[item]["chart_path"])
        if cross_rank_comparison_chart is not None:
            cross_rank_comparison_section += f"""
            <h4> {item_list[item]['name']}  </h4>
            <img src="data:image/png;base64,{cross_rank_comparison_chart}" alt="{item} by Rank" class="chart-image">
            {item_list[item]['description']}.
            """

    summary_section += cross_rank_comparison_section

    nccl_charst_section = f"""
    <h3> 3. NCCL Collective Operations Analysis </h3>
    """
    nccl_chart_item_list = {
        "NCCL Communication Latency": "Mean communication latency for NCCL allreduce operations across different message sizes",
        "NCCL Algorithm Bandwidth": "Algorithm bandwidth achieved for different message sizes in NCCL collective operations.",
        "NCCL Bus Bandwidth": "Bus bandwidth utilization across NCCL operations and message sizes.",
        "NCCL Performance Percentage Change": "Percent change in communication latency and bandwidth metrics for each message sizec configuration",
        "NCCL Total Communication Latency": "Aggregate communication latency summed across all operations for each message size.",
    }
    for item in nccl_chart_item_list.keys():
        nccl_image_data = get_image_data(
            plot_file_path / f'{item.replace(" ", "_")}_comparison.png'
        )
        if nccl_image_data is not None:
            nccl_charst_section += f"""
            <h4> {item} </h4>
            <img src="data:image/png;base64,{get_image_data(plot_file_path / f'{item.replace(" ", "_")}_comparison.png')}" alt="{item} Comparison" class="chart-image">
            {nccl_chart_item_list[item]}
            """

    summary_section += nccl_charst_section

    footer_section = f"""

    </body>
    </html>
    """
    summary_section += footer_section

    final_html = html_header + summary_section
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
