"""
HW Queue Eval HTML report generator.

Generates self-contained HTML reports with embedded plots (base64) for:
- Single run analysis (Mode A)
- Sweep analysis (Mode B)
- Multi-workload comparison (Mode C)
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime

from ..templates.hwqueue_report_template import (
    HTML_HEADER,
    HTML_FOOTER,
    SINGLE_RUN_PLOTS,
    SWEEP_PLOTS,
    COMPARISON_PLOTS,
)
from ..processing.hwqueue_loader import SingleRunData, SweepData
from .html_generator import image_to_base64

if TYPE_CHECKING:
    from ..pipelines.hwqueue_pipeline import SweepSummary, ComparisonSummary


# =============================================================================
# Helper Functions
# =============================================================================


def _create_chart_html(
    plot_config: dict,
    plots_dir: Path,
    workload_name: str = "",
) -> str:
    """Generate HTML for a single chart."""
    # Substitute workload name in filename
    filename = plot_config["file"].format(workload=workload_name)
    plot_path = plots_dir / filename

    if not plot_path.exists():
        return f"""
    <div class="missing-chart">Image not available: {filename}</div>
    """

    image_data = image_to_base64(plot_path)
    if image_data is None:
        return f"""
    <div class="missing-chart">Failed to encode: {filename}</div>
    """

    return f"""
    <h4>{plot_config['name']}</h4>
    <img src="{image_data}" alt="{plot_config['alt']}">
    <p>{plot_config['description']}</p>
    """


def _format_value(value: Any, precision: int = 2) -> str:
    """Format a numeric value for display."""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _get_status_badge(change_pct: float, threshold: float = 5.0) -> str:
    """Generate status badge HTML based on change percentage."""
    if change_pct < -threshold:
        return '<span class="badge badge-regression">⚠ REGRESSION</span>'
    elif change_pct > threshold:
        return '<span class="badge badge-improved">✓ IMPROVED</span>'
    else:
        return '<span class="badge badge-ok">OK</span>'


def _generate_sweep_verdict_html(summary: "SweepSummary") -> str:
    """Generate HTML for sweep verdict box."""
    status_class = summary.get_overall_status()

    # Determine CSS class for each verdict
    def get_css_class(verdict_str: str) -> str:
        if "✗" in verdict_str:
            return "poor"
        elif "⚠" in verdict_str:
            return "warning"
        else:
            return "good"

    scaling_class = get_css_class(summary.scaling_verdict[0])
    latency_class = get_css_class(summary.latency_verdict[0])

    return f"""
<div class="verdict-box verdict-{status_class}">
    <h2>📊 Analysis Summary</h2>
    <table class="verdict-table">
        <tr>
            <td>Peak Performance</td>
            <td class="good">✓ {summary.peak_throughput:.0f} {summary.throughput_unit} at {summary.peak_streams} streams</td>
        </tr>
        <tr>
            <td>Scaling Efficiency</td>
            <td class="{scaling_class}">{summary.scaling_verdict[0]} ({summary.peak_efficiency*100:.0f}% efficiency at peak)</td>
        </tr>
        <tr>
            <td>Latency Trend</td>
            <td class="{latency_class}">{summary.latency_verdict[0]} - {summary.latency_verdict[1]}</td>
        </tr>
        <tr>
            <td>Recommendation</td>
            <td>→ Use <strong>{summary.optimal_streams} streams</strong> for best throughput/efficiency balance</td>
        </tr>
    </table>
</div>
"""


def _generate_comparison_verdict_html(
    summary: "ComparisonSummary",
    baseline_label: str,
    test_label: str,
) -> str:
    """Generate HTML for comparison verdict box."""
    # Build stats section
    stats_html = f"""
    <div class="verdict-stats">
        <span class="stat improved">✓ {summary.num_improved} improved{f" (avg +{summary.avg_improved_change:.1f}%)" if summary.avg_improved_change else ""}</span>
        <span class="stat degraded">✗ {summary.num_degraded} degraded{f" (avg {summary.avg_degraded_change:.1f}%)" if summary.avg_degraded_change else ""}</span>
        <span class="stat unchanged">─ {summary.num_unchanged} unchanged</span>
    </div>
"""

    # Build summary table
    table_rows = ""
    if summary.top_improvement:
        table_rows += f"""
        <tr class="improved">
            <td>✓ Top Improvement</td>
            <td>{summary.top_improvement[0]}</td>
            <td>+{summary.top_improvement[1]:.1f}%</td>
        </tr>"""
    if summary.top_regression:
        table_rows += f"""
        <tr class="degraded">
            <td>✗ Top Regression</td>
            <td>{summary.top_regression[0]}</td>
            <td>{summary.top_regression[1]:.1f}%</td>
        </tr>"""

    summary_table = ""
    if table_rows:
        summary_table = f"""
    <table class="verdict-summary-table">
        <tr>
            <th>Category</th>
            <th>Workload</th>
            <th>Change</th>
        </tr>
        {table_rows}
        <tr class="total">
            <td>Overall Average</td>
            <td>{summary.total_workloads} workloads</td>
            <td>{summary.avg_change:+.1f}%</td>
        </tr>
    </table>
"""

    return f"""
<div class="verdict-box verdict-{summary.verdict_class}">
    <h2>📊 Comparison Summary: {baseline_label} → {test_label}</h2>
    <div class="verdict-headline {summary.verdict_class}">{summary.verdict}</div>
    {stats_html}
    {summary_table}
</div>
"""


# =============================================================================
# Single Run HTML (Mode A)
# =============================================================================


def generate_single_run_html(
    data: SingleRunData,
    plots_dir: Path,
    output_path: Path,
    verbose: bool = False,
) -> Path:
    """
    Generate HTML report for single run analysis.

    Args:
        data: SingleRunData object
        plots_dir: Directory containing generated plots
        output_path: Output HTML file path
        verbose: Print progress

    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Generating single run HTML: {output_path.name}")

    # Build HTML content
    body = f"""
<body>

<h1>HW Queue Eval - Single Run Analysis</h1>

<div class="summary-box">
    <h3>Run Summary</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">Workload</div>
            <div class="value">{data.workload_name}</div>
        </div>
        <div class="metric-card">
            <div class="label">Stream Count</div>
            <div class="value">{data.stream_count}</div>
        </div>
        <div class="metric-card">
            <div class="label">Throughput</div>
            <div class="value">{data.throughput:.2f}</div>
            <div class="unit">{data.throughput_unit}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Time</div>
            <div class="value">{data.total_time_ms:.2f}</div>
            <div class="unit">ms</div>
        </div>
    </div>
</div>

<hr>

<h2>1. Latency Metrics</h2>

<table>
    <tr><th>Metric</th><th>Value (ms)</th></tr>
    <tr><td>Mean</td><td>{data.latency.mean:.3f}</td></tr>
    <tr><td>P50 (Median)</td><td>{data.latency.p50:.3f}</td></tr>
    <tr><td>P95</td><td>{data.latency.p95:.3f}</td></tr>
    <tr><td>P99</td><td>{data.latency.p99:.3f}</td></tr>
    <tr><td>Min</td><td>{data.latency.min:.3f}</td></tr>
    <tr><td>Max</td><td>{data.latency.max:.3f}</td></tr>
    <tr><td>Std Dev</td><td>{data.latency.std:.3f}</td></tr>
</table>

<h2>2. Visualizations</h2>
"""

    # Add plots
    for plot_config in SINGLE_RUN_PLOTS:
        body += _create_chart_html(plot_config, plots_dir, data.workload_name)

    # Add switch latency section if available
    if data.switch_latency:
        body += f"""
<h2>3. Switch Latency Analysis</h2>

<table>
    <tr><th>Metric</th><th>Value (ms)</th></tr>
    <tr><td>Inter-Stream Gap</td><td>{data.switch_latency.inter_stream_gap_ms:.3f}</td></tr>
    <tr><td>Intra-Stream Gap</td><td>{data.switch_latency.intra_stream_gap_ms:.3f}</td></tr>
    <tr><td>Estimated Switch Overhead</td><td>{data.switch_latency.estimated_switch_overhead_ms:.3f}</td></tr>
</table>
"""

    final_html = HTML_HEADER + body + HTML_FOOTER

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    if verbose:
        file_size = output_path.stat().st_size / 1024
        print(f"    Created: {output_path.name} ({file_size:.1f} KB)")

    return output_path


# =============================================================================
# Sweep HTML (Mode B)
# =============================================================================


def generate_sweep_html(
    data: SweepData,
    plots_dir: Path,
    output_path: Path,
    summary: Optional["SweepSummary"] = None,
    verbose: bool = False,
) -> Path:
    """
    Generate HTML report for sweep analysis.

    Args:
        data: SweepData object
        plots_dir: Directory containing generated plots
        output_path: Output HTML file path
        summary: Optional SweepSummary for verdict box
        verbose: Print progress

    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Generating sweep HTML: {output_path.name}")

    best_streams, best_throughput = data.get_best_throughput()

    # Build verdict box if summary provided
    verdict_html = ""
    if summary:
        verdict_html = _generate_sweep_verdict_html(summary)

    # Build HTML content
    body = f"""
<body>

<h1>HW Queue Eval - Sweep Analysis</h1>

{verdict_html}

<div class="summary-box">
    <h3>Sweep Summary</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">Workload</div>
            <div class="value">{data.workload_name}</div>
        </div>
        <div class="metric-card">
            <div class="label">Configurations Tested</div>
            <div class="value">{len(data.results)}</div>
        </div>
        <div class="metric-card">
            <div class="label">Best Stream Count</div>
            <div class="value">{best_streams}</div>
        </div>
        <div class="metric-card">
            <div class="label">Peak Throughput</div>
            <div class="value">{best_throughput:.2f}</div>
            <div class="unit">{data.results[0].throughput_unit}</div>
        </div>
    </div>
</div>

<hr>

<h2>1. Scaling Analysis</h2>
"""

    # Add plots
    for plot_config in SWEEP_PLOTS:
        body += _create_chart_html(plot_config, plots_dir, data.workload_name)

    # Add scaling data table
    body += """
<h2>2. Detailed Scaling Data</h2>

<table>
    <tr>
        <th>Streams</th>
        <th>Throughput</th>
        <th>P50 (ms)</th>
        <th>P95 (ms)</th>
        <th>P99 (ms)</th>
        <th>Total Time (ms)</th>
    </tr>
"""

    for result in sorted(data.results, key=lambda r: r.stream_count):
        is_best = result.stream_count == best_streams
        row_style = ' style="background-color: #d4edda;"' if is_best else ""
        body += f"""
    <tr{row_style}>
        <td>{result.stream_count}{"★" if is_best else ""}</td>
        <td>{result.throughput:.2f}</td>
        <td>{result.latency.p50:.3f}</td>
        <td>{result.latency.p95:.3f}</td>
        <td>{result.latency.p99:.3f}</td>
        <td>{result.total_time_ms:.2f}</td>
    </tr>
"""

    body += "</table>"

    # Add environment info if available
    if data.environment.hostname:
        body += f"""
<h2>3. Environment</h2>

<table>
    <tr><th>Property</th><th>Value</th></tr>
    <tr><td>Hostname</td><td>{data.environment.hostname}</td></tr>
    <tr><td>GPU Count</td><td>{data.environment.gpu_count}</td></tr>
    <tr><td>HIP Version</td><td>{data.environment.hip_version or "N/A"}</td></tr>
    <tr><td>PyTorch Version</td><td>{data.environment.torch_version or "N/A"}</td></tr>
</table>
"""

    final_html = HTML_HEADER + body + HTML_FOOTER

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    if verbose:
        file_size = output_path.stat().st_size / 1024
        print(f"    Created: {output_path.name} ({file_size:.1f} KB)")

    return output_path


# =============================================================================
# Comparison HTML (Mode C)
# =============================================================================


def generate_comparison_html(
    baseline_data: Dict[str, SweepData],
    test_data: Dict[str, SweepData],
    common_workloads: List[str],
    baseline_only: List[str],
    test_only: List[str],
    regressions: List[Dict],
    improvements: List[Dict],
    plots_dir: Path,
    output_path: Path,
    baseline_label: str = "Baseline",
    test_label: str = "Test",
    threshold: float = 0.05,
    summary: Optional["ComparisonSummary"] = None,
    verbose: bool = False,
) -> Path:
    """
    Generate HTML report for multi-workload comparison.

    Args:
        baseline_data: Dict of workload_name -> SweepData for baseline
        test_data: Dict of workload_name -> SweepData for test
        common_workloads: List of workloads in both baseline and test
        baseline_only: List of workloads only in baseline
        test_only: List of workloads only in test
        regressions: List of regression dicts from Excel generator
        improvements: List of improvement dicts from Excel generator
        plots_dir: Directory containing generated plots
        output_path: Output HTML file path
        baseline_label: Label for baseline
        test_label: Label for test
        threshold: Regression threshold (fraction)
        summary: Optional ComparisonSummary for verdict box
        verbose: Print progress

    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Generating comparison HTML: {output_path.name}")

    threshold_pct = threshold * 100

    # Count regressions and improvements
    regression_count = len(regressions)
    improvement_count = len(improvements)

    # Build verdict box if summary provided
    verdict_html = ""
    if summary:
        verdict_html = _generate_comparison_verdict_html(summary, baseline_label, test_label)

    # Build HTML content
    body = f"""
<body>

<h1>HW Queue Eval - Comparison Report</h1>

<p style="color: #666;">
    Comparing <strong>{baseline_label}</strong> (baseline) vs <strong>{test_label}</strong> (test)
    <br>
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</p>

{verdict_html}

<div class="summary-box">
    <h3>Detailed Metrics</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="label">Common Workloads</div>
            <div class="value">{len(common_workloads)}</div>
        </div>
        <div class="metric-card">
            <div class="label">Regressions</div>
            <div class="value" style="color: var(--color-negative);">{regression_count}</div>
        </div>
        <div class="metric-card">
            <div class="label">Improvements</div>
            <div class="value" style="color: var(--color-positive);">{improvement_count}</div>
        </div>
        <div class="metric-card">
            <div class="label">Threshold</div>
            <div class="value">±{threshold_pct:.0f}%</div>
        </div>
    </div>
</div>
"""

    # Add warnings for missing workloads
    if baseline_only or test_only:
        body += '<div class="warning-box">'
        if baseline_only:
            body += f"<p><strong>⚠ Workloads in baseline but not test:</strong> {', '.join(baseline_only)}</p>"
        if test_only:
            body += f"<p><strong>⚠ Workloads in test but not baseline:</strong> {', '.join(test_only)}</p>"
        body += "</div>"

    body += """
<hr>

<h2>1. Workload Comparison Summary</h2>

<table>
    <tr>
        <th>Workload</th>
        <th>Best Streams (Base/Test)</th>
        <th>Throughput (Base)</th>
        <th>Throughput (Test)</th>
        <th>Change (%)</th>
        <th>Status</th>
    </tr>
"""

    for wl in common_workloads:
        b_best_s, b_best_t = baseline_data[wl].get_best_throughput()
        t_best_s, t_best_t = test_data[wl].get_best_throughput()

        if b_best_t > 0:
            change_pct = (t_best_t - b_best_t) / b_best_t * 100
        else:
            change_pct = 0

        status_badge = _get_status_badge(change_pct, threshold_pct)

        body += f"""
    <tr>
        <td>{wl}</td>
        <td>{b_best_s} / {t_best_s}</td>
        <td>{b_best_t:.2f}</td>
        <td>{t_best_t:.2f}</td>
        <td>{change_pct:+.1f}%</td>
        <td>{status_badge}</td>
    </tr>
"""

    body += "</table>"

    # Add comparison plots
    body += """
<h2>2. Comparison Visualizations</h2>
"""

    for plot_config in COMPARISON_PLOTS:
        body += _create_chart_html(plot_config, plots_dir, "")

    # Add regressions table
    if regressions:
        body += """
<h2>3. Regressions Detail</h2>

<table>
    <tr>
        <th>Workload</th>
        <th>Stream Count</th>
        <th>Metric</th>
        <th>Baseline</th>
        <th>Test</th>
        <th>Change (%)</th>
    </tr>
"""
        for reg in regressions[:20]:  # Limit to first 20
            body += f"""
    <tr>
        <td>{reg['Workload']}</td>
        <td>{reg['Stream_Count']}</td>
        <td>{reg['Metric']}</td>
        <td>{reg['Baseline']}</td>
        <td>{reg['Test']}</td>
        <td class="status-regression">{reg['Change_%']:+.1f}%</td>
    </tr>
"""
        body += "</table>"
        if len(regressions) > 20:
            body += f"<p><em>... and {len(regressions) - 20} more regressions (see Excel report for full list)</em></p>"

    # Add improvements table
    if improvements:
        body += """
<h2>4. Improvements Detail</h2>

<table>
    <tr>
        <th>Workload</th>
        <th>Stream Count</th>
        <th>Metric</th>
        <th>Baseline</th>
        <th>Test</th>
        <th>Change (%)</th>
    </tr>
"""
        for imp in improvements[:20]:  # Limit to first 20
            body += f"""
    <tr>
        <td>{imp['Workload']}</td>
        <td>{imp['Stream_Count']}</td>
        <td>{imp['Metric']}</td>
        <td>{imp['Baseline']}</td>
        <td>{imp['Test']}</td>
        <td class="status-improved">{imp['Change_%']:+.1f}%</td>
    </tr>
"""
        body += "</table>"
        if len(improvements) > 20:
            body += f"<p><em>... and {len(improvements) - 20} more improvements (see Excel report for full list)</em></p>"

    final_html = HTML_HEADER + body + HTML_FOOTER

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    if verbose:
        file_size = output_path.stat().st_size / 1024
        print(f"    Created: {output_path.name} ({file_size:.1f} KB)")

    return output_path


# =============================================================================
# Auto-dispatch Function
# =============================================================================


def generate_hwqueue_html(
    data: SingleRunData | SweepData,
    plots_dir: Path,
    output_path: Path,
    summary: Optional["SweepSummary"] = None,
    verbose: bool = False,
) -> Path:
    """
    Generate HTML report for single run or sweep data.

    Automatically detects data type and calls the appropriate generator.

    Args:
        data: SingleRunData or SweepData object
        plots_dir: Directory containing generated plots
        output_path: Output HTML file path
        summary: Optional SweepSummary for verdict box (sweep mode only)
        verbose: Print progress

    Returns:
        Path to generated HTML file
    """
    if isinstance(data, SweepData):
        return generate_sweep_html(data, plots_dir, output_path, summary=summary, verbose=verbose)
    elif isinstance(data, SingleRunData):
        return generate_single_run_html(data, plots_dir, output_path, verbose=verbose)
    else:
        raise ValueError(f"Unknown data type: {type(data)}")
