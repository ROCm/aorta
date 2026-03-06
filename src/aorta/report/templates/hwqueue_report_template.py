"""Configuration constants for HW Queue Eval HTML report generation."""

HTML_HEADER = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HW Queue Eval Analysis Report</title>
<style>
    :root {
        --color-positive: #2ecc71;
        --color-negative: #e74c3c;
        --color-neutral: #95a5a6;
        --color-primary: #3498db;
        --color-secondary: #e67e22;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        line-height: 1.6;
        margin: 0 auto;
        padding: 20px;
        max-width: 1100px;
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
        border-bottom: 2px solid var(--color-primary);
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
    .summary-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .summary-box h3 {
        margin-top: 0;
        border-bottom: none;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.85em;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-card .value {
        font-size: 1.5em;
        font-weight: bold;
        color: #333;
    }
    .metric-card .unit {
        font-size: 0.8em;
        color: #888;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th, td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #333;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
    .status-improved {
        color: var(--color-positive);
        font-weight: bold;
    }
    .status-regression {
        color: var(--color-negative);
        font-weight: bold;
    }
    .status-ok {
        color: var(--color-neutral);
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .badge-improved {
        background-color: #d4edda;
        color: #155724;
    }
    .badge-regression {
        background-color: #f8d7da;
        color: #721c24;
    }
    .badge-ok {
        background-color: #e2e3e5;
        color: #383d41;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 15px;
        margin: 15px 0;
    }
    .warning-box strong {
        color: #856404;
    }
    /* Verdict Box Styles */
    .verdict-box {
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid;
    }
    .verdict-box.verdict-good {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .verdict-box.verdict-warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .verdict-box.verdict-poor {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .verdict-box.verdict-improved {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .verdict-box.verdict-degraded {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .verdict-box.verdict-mixed {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .verdict-box.verdict-unchanged {
        background-color: #e2e3e5;
        border-left-color: #6c757d;
    }
    .verdict-box h2 {
        margin-top: 0;
        border-bottom: none;
        font-size: 1.3em;
    }
    .verdict-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 0 0;
        background-color: rgba(255,255,255,0.7);
        border-radius: 4px;
    }
    .verdict-table td {
        padding: 8px 12px;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }
    .verdict-table td:first-child {
        font-weight: 600;
        width: 40%;
        color: #333;
    }
    .verdict-table tr:last-child td {
        border-bottom: none;
    }
    .verdict-table .good {
        color: #155724;
    }
    .verdict-table .warning {
        color: #856404;
    }
    .verdict-table .poor {
        color: #721c24;
    }
    .verdict-headline {
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
        margin: 10px 0 20px 0;
    }
    .verdict-headline.improved { color: #155724; }
    .verdict-headline.degraded { color: #721c24; }
    .verdict-headline.mixed { color: #856404; }
    .verdict-headline.unchanged { color: #383d41; }
    .verdict-stats {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 15px 0;
        flex-wrap: wrap;
    }
    .verdict-stats .stat {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 500;
    }
    .verdict-stats .stat.improved {
        background-color: #d4edda;
        color: #155724;
    }
    .verdict-stats .stat.degraded {
        background-color: #f8d7da;
        color: #721c24;
    }
    .verdict-stats .stat.unchanged {
        background-color: #e2e3e5;
        color: #383d41;
    }
    .verdict-summary-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        background-color: rgba(255,255,255,0.9);
        border-radius: 6px;
        overflow: hidden;
    }
    .verdict-summary-table th {
        background-color: rgba(0,0,0,0.05);
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
    }
    .verdict-summary-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }
    .verdict-summary-table tr.improved td:first-child {
        color: #155724;
    }
    .verdict-summary-table tr.degraded td:first-child {
        color: #721c24;
    }
    .verdict-summary-table tr.total {
        font-weight: bold;
        background-color: rgba(0,0,0,0.03);
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
<hr>
<p style="text-align: center; color: #888; font-size: 0.9em;">
Generated by aorta-report hwqueue pipeline
</p>
</div>
</body>
</html>
"""

# Single Run plot configuration
SINGLE_RUN_PLOTS = [
    {
        "name": "Latency Distribution",
        "file": "latency_histogram_{workload}.png",
        "alt": "Latency Histogram",
        "description": "Distribution of iteration latencies with percentile markers (P50, P95, P99).",
    },
    {
        "name": "Latency Percentiles",
        "file": "latency_percentiles_{workload}.png",
        "alt": "Latency Percentiles Bar Chart",
        "description": "Bar chart showing mean, P50, P95, P99, min, and max latencies.",
    },
    {
        "name": "Per-Stream Times",
        "file": "per_stream_times_{workload}.png",
        "alt": "Per-Stream Execution Times",
        "description": "Execution time for each stream. Green bars are at or below mean, red bars are above.",
    },
]

# Sweep plot configuration
SWEEP_PLOTS = [
    {
        "name": "Throughput Scaling",
        "file": "throughput_scaling_{workload}.png",
        "alt": "Throughput Scaling Curve",
        "description": "Throughput vs stream count with ideal linear scaling reference. Star marks the best configuration.",
    },
    {
        "name": "Scaling Efficiency",
        "file": "scaling_efficiency_{workload}.png",
        "alt": "Scaling Efficiency",
        "description": "Scaling efficiency (actual/ideal) at each stream count. Green ≥80%, orange ≥60%, red <60%.",
    },
    {
        "name": "Latency vs Streams",
        "file": "latency_vs_streams_{workload}.png",
        "alt": "Latency vs Stream Count",
        "description": "Latency percentiles (P50, P95, P99) across different stream counts.",
    },
    {
        "name": "Latency Heatmap",
        "file": "latency_heatmap_{workload}.png",
        "alt": "Latency Heatmap",
        "description": "Heatmap showing latency metrics across all stream configurations.",
    },
]

# Comparison plot configuration
COMPARISON_PLOTS = [
    {
        "name": "Throughput Comparison",
        "file": "throughput_comparison.png",
        "alt": "Throughput Comparison Bar Chart",
        "description": "Side-by-side comparison of best throughput for each workload.",
    },
    {
        "name": "Delta Summary",
        "file": "delta_summary.png",
        "alt": "Delta Summary Chart",
        "description": "Throughput change (%) per workload, sorted by change. Green = improvement, red = regression.",
    },
    {
        "name": "Regression Heatmap",
        "file": "regression_heatmap.png",
        "alt": "Regression Heatmap",
        "description": "Throughput change heatmap across workloads and stream counts. Red indicates regression, green indicates improvement.",
    },
    {
        "name": "Latency Delta",
        "file": "latency_delta.png",
        "alt": "Latency Delta Chart",
        "description": "P99 latency change at best throughput configuration. Higher latency (positive change) is a regression.",
    },
]

