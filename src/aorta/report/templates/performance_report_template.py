"""Configuration constants for HTML performance report generation."""

HTML_HEADER = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Performance Analysis Report</title>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        line-height: 1.6;
        margin: 0 auto;
        padding: 20px;
        max-width: 900px;
        background-color: #f5f5f5;
    }
    .container {
        background-color: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    h1 {
        border-bottom: 3px solid #333;
        padding-bottom: 10px;
    }
    h2 {
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
        margin-top: 40px;
    }
    h3 {
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
        margin-top: 30px;
    }
    h4 {
        margin-top: 20px;
        color: #34495e;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 15px auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        background-color: white;
    }
    p {
        color: #555;
    }
    hr {
        border: none;
        border-top: 2px solid #ecf0f1;
        margin: 30px 0;
    }
    .missing-chart {
        background-color: #f8d7da;
        border: 2px dashed #dc3545;
        padding: 40px;
        text-align: center;
        color: #721c24;
        border-radius: 4px;
        margin: 15px 0;
    }
    @media print {
        body { margin: 0; background-color: white; }
        .container { box-shadow: none; }
    }
</style>
</head>
<div class="container">
"""

HTML_FOOTER = """
</div>
</body>
</html>
"""

# Chart configuration for each section
OVERALL_GPU_CHARTS = [
    {
        "name": "Percentage Change Overview",
        "file": "improvement_chart.png",
        "alt": "Summary Chart",
        "description": "Overall performance change across key GPU metrics. Positive values indicate improvement (Test is faster/better).",
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
        "file": "gpu_time_change_percentage_summary_by_rank.png",
        "alt": "gpu_time_change_percentage_summary_by_rank by Rank",
        "description": "Detailed breakdown of percent change for each metric type across all ranks.",
    },
]

NCCL_CHARTS = [
    {
        "name": "NCCL Communication Latency",
        "file": "NCCL_Communication_Latency_comparison.png",
        "alt": "NCCL Communication Latency Comparison",
        "description": "Mean communication latency for NCCL allreduce operations across different message sizes.",
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
        "description": "Percent change in communication latency and bandwidth metrics for each message size configuration.",
    },
    {
        "name": "NCCL Total Communication Latency",
        "file": "NCCL_Total_Communication_Latency_comparison.png",
        "alt": "NCCL Total Communication Latency Comparison",
        "description": "Aggregate communication latency summed across all operations for each message size.",
    },
]

