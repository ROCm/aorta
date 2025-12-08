"""Configuration constants for HTML report generation."""

HTML_HEADER = """<!DOCTYPE html>
<html lang="en"><head>
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
"""

HTML_FOOTER = """
</body>
</html>
"""

# Chart configuration for each section
OVERALL_GPU_CHARTS = [
    {
        "name": "Percentage Change Overview",
        "file": "improvement_chart.png",
        "alt": "Summary Chart",
        "description": "Overall performance change across key GPU metrics. Positive values indicate improvement (Saleelk is faster/better).",
    },
    {
        "name": "Absolute Time Comparison",
        "file": "abs_time_comparison.png",
        "alt": "Absolute Time Comparison",
        "description": "Side-by-side comparison of absolute execution times for all GPU metrics.",
    },
]

CROSS_RANK_CHARTS = [
    {
        "name": "Performance Heatmap by Rank",
        "file": "gpu_time_heatmap.png",
        "alt": "GPU Metric Percentage Change by Rank (HeatMap)",
        "description": "Comprehensive heatmap showing percent change for all metrics across all ranks. Green indicates better performance (positive % change).",
    },
    {
        "name": "Total Time",
        "file": "total_time_by_rank.png",
        "alt": "total_time by Rank",
        "description": "Total execution time comparison across all ranks, showing end-to-end performance characteristics.",
    },
    {
        "name": "Computation Time",
        "file": "computation_time_by_rank.png",
        "alt": "computation_time by Rank",
        "description": "Pure computation time excluding communication overhead, analyzed per rank.",
    },
    {
        "name": "Communication Time",
        "file": "total_comm_time_by_rank.png",
        "alt": "total_comm_time by Rank",
        "description": "Total time spent in collective communication operations across ranks.",
    },
    {
        "name": "Idle Time",
        "file": "idle_time_by_rank.png",
        "alt": "idle_time by Rank",
        "description": "GPU idle time comparison showing resource utilization efficiency per rank.",
    },
    {
        "name": "Detailed Percentage Change by Metric",
        "file": "gpu_time_change_percentage_summaryby_rank.png",
        "alt": "gpu_time_change_percentage_summaryby_rank by Rank",
        "description": "Detailed breakdown of percent change for each metric type across all ranks.",
    },
]

NCCL_CHARTS = [
    {
        "name": "NCCL Communication Latency",
        "file": "NCCL_Communication_Latency_comparison.png",
        "alt": "NCCL Communication Latency Comparison",
        "description": "Mean communication latency for NCCL allreduce operations across different message sizes",
    },
    {
        "name": "NCCL Algorithm Bandwidth",
        "file": "NCCL_Algorithm_Bandwidth_comparison.png",
        "alt": "NCCL Algorithm Bandwidth Comparison",
        "description": "Algorithm bandwidth achieved for different message sizes in NCCL collective operations.",
    },
    {
        "name": "NCCL Bus Bandwidth",
        "file": "NCCL_Bus_Bandwidth_comparison.png",
        "alt": "NCCL Bus Bandwidth Comparison",
        "description": "Bus bandwidth utilization across NCCL operations and message sizes.",
    },
    {
        "name": "NCCL Performance Percentage Change",
        "file": "NCCL_Performance_Percentage_Change_comparison.png",
        "alt": "NCCL Performance Percentage Change Comparison",
        "description": "Percent change in communication latency and bandwidth metrics for each message sizec configuration",
    },
    {
        "name": "NCCL Total Communication Latency",
        "file": "NCCL_Total_Communication_Latency_comparison.png",
        "alt": "NCCL Total Communication Latency Comparison",
        "description": "Aggregate communication latency summed across all operations for each message size.",
    },
]
