#!/usr/bin/env python3
"""
Enhanced plot generation matching the PDF report style.
Generates exactly 12 plots as specified.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
import warnings
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot1_percentage_change(summary_data, output_dir):
    """Plot 1: Percentage Change Overview."""
    print("\nGenerating Plot 1: Percentage Change Overview")

    columns = summary_data.columns.tolist()
    baseline_label = columns[1] if len(columns) > 1 else 'Baseline'
    test_label = columns[2] if len(columns) > 2 else 'Test'

    if 'Improvement (%)' not in summary_data.columns:
        print("  No improvement data found")
        return

    metrics = summary_data['Metric'].values
    values = summary_data['Improvement (%)'].values

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ax.text(x_pos + (0.5 if x_pos > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left' if x_pos > 0 else 'right', va='center', fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Percent Change (%)', fontsize=12)
    ax.set_title(f'GPU Metrics: Percent Change ({baseline_label} vs {test_label})\nPositive = Improvement ({test_label} Faster)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'plot1_percentage_change_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot1_percentage_change_overview.png")


def plot2_absolute_time_comparison(summary_data, output_dir):
    """Plot 2: Absolute Time Comparison."""
    print("\nGenerating Plot 2: Absolute Time Comparison")

    columns = summary_data.columns.tolist()
    baseline_label = columns[1] if len(columns) > 1 else 'Baseline'
    test_label = columns[2] if len(columns) > 2 else 'Test'

    metrics = summary_data['Metric'].values
    baseline_values = summary_data[baseline_label].values
    test_values = summary_data[test_label].values

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label=baseline_label, alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_values, width, label=test_label, alpha=0.8, color='darkorange')

    ax.set_xlabel('Metric Type', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('GPU Metrics: Absolute Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'plot2_absolute_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot2_absolute_time_comparison.png")


def plot3_performance_heatmap(byrank_data, output_dir):
    """Plot 3: Performance Heatmap by Rank."""
    print("\nGenerating Plot 3: Performance Heatmap by Rank")

    if byrank_data is None or byrank_data.empty:
        print("  No by-rank data available")
        return

    metrics = byrank_data['type'].unique() if 'type' in byrank_data.columns else []
    ranks = sorted(byrank_data['rank'].unique()) if 'rank' in byrank_data.columns else []

    time_cols = [col for col in byrank_data.columns if 'time' in col.lower() and 'diff' not in col.lower()]
    time_col = time_cols[-1] if len(time_cols) > 1 else time_cols[0] if time_cols else None

    if not time_col:
        print("  No time column found")
        return

    heatmap_data = byrank_data.pivot_table(index='type', columns='rank', values=time_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Time (ms)'}, ax=ax)

    ax.set_title('Performance Heatmap by Rank (Time in ms)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Metric Type', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot3_performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot3_performance_heatmap.png")


def plot4_total_execution_time(byrank_data, output_dir):
    """Plot 4: Total Execution Time by Rank (Line Plot)."""
    print("\nGenerating Plot 4: Total Execution Time by Rank")

    if byrank_data is None or byrank_data.empty:
        print("  No by-rank data available")
        return

    total_time_data = byrank_data[byrank_data['type'] == 'total_time']
    if total_time_data.empty:
        print("  No total_time data found")
        return

    ranks = sorted(total_time_data['rank'].unique())
    time_cols = [col for col in total_time_data.columns if 'time' in col.lower() and 'diff' not in col.lower()]

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in time_cols[:2]:
        times = [total_time_data[total_time_data['rank'] == r][col].values[0] if not total_time_data[total_time_data['rank'] == r].empty else 0 for r in ranks]
        label = col.replace('_time_ms', '').replace('_', ' ')
        ax.plot(ranks, times, marker='o', markersize=8, linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Total Execution Time (ms)', fontsize=12)
    ax.set_title('Total Execution Time by Rank', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot4_total_execution_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot4_total_execution_time.png")


def plot5_computation_time(byrank_data, output_dir):
    """Plot 5: Computation Time Across Ranks."""
    print("\nGenerating Plot 5: Computation Time Across Ranks")

    if byrank_data is None or byrank_data.empty:
        print("  No by-rank data available")
        return

    comp_data = byrank_data[byrank_data['type'] == 'computation_time']
    if comp_data.empty:
        print("  No computation_time data found")
        return

    ranks = sorted(comp_data['rank'].unique())
    time_cols = [col for col in comp_data.columns if 'time' in col.lower() and 'diff' not in col.lower()]

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in time_cols[:2]:
        times = [comp_data[comp_data['rank'] == r][col].values[0] if not comp_data[comp_data['rank'] == r].empty else 0 for r in ranks]
        label = col.replace('_time_ms', '').replace('_', ' ')
        ax.plot(ranks, times, marker='s', markersize=8, linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Computation Time Across Ranks', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot5_computation_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot5_computation_time.png")


def plot6_communication_time(byrank_data, output_dir):
    """Plot 6: Total Communication Time Across Ranks."""
    print("\nGenerating Plot 6: Total Communication Time Across Ranks")

    if byrank_data is None or byrank_data.empty:
        print("  No by-rank data available")
        return

    comm_data = byrank_data[byrank_data['type'] == 'total_comm_time']
    if comm_data.empty:
        print("  No total_comm_time data found")
        return

    ranks = sorted(comm_data['rank'].unique())
    time_cols = [col for col in comm_data.columns if 'time' in col.lower() and 'diff' not in col.lower()]

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in time_cols[:2]:
        times = [comm_data[comm_data['rank'] == r][col].values[0] if not comm_data[comm_data['rank'] == r].empty else 0 for r in ranks]
        label = col.replace('_time_ms', '').replace('_', ' ')
        ax.plot(ranks, times, marker='^', markersize=8, linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Communication Time (ms)', fontsize=12)
    ax.set_title('Total Communication Time Across Ranks', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot6_communication_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot6_communication_time.png")


def plot7_idle_time(byrank_data, output_dir):
    """Plot 7: Idle Time Across Ranks."""
    print("\nGenerating Plot 7: Idle Time Across Ranks")

    if byrank_data is None or byrank_data.empty:
        print("  No by-rank data available")
        return

    idle_data = byrank_data[byrank_data['type'] == 'idle_time']
    if idle_data.empty:
        print("  No idle_time data found")
        return

    ranks = sorted(idle_data['rank'].unique())
    time_cols = [col for col in idle_data.columns if 'time' in col.lower() and 'diff' not in col.lower()]

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in time_cols[:2]:
        times = [idle_data[idle_data['rank'] == r][col].values[0] if not idle_data[idle_data['rank'] == r].empty else 0 for r in ranks]
        label = col.replace('_time_ms', '').replace('_', ' ')
        ax.plot(ranks, times, marker='D', markersize=8, linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Idle Time (ms)', fontsize=12)
    ax.set_title('Idle Time Across Ranks', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot7_idle_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot7_idle_time.png")


def plot8_percentage_time_difference(byrank_data, output_dir):
    """Plot 8: Percentage Time Difference Across Ranks (8 subplots in 2x4 grid)."""
    print("\nGenerating Plot 8: Percentage Time Difference (8 subplots)")

    if byrank_data is None or byrank_data.empty:
        print("  No by-rank data available")
        return

    metrics = ['busy_time', 'computation_time', 'total_comm_time', 'exposed_comm_time',
               'idle_time', 'total_memcpy_time', 'exposed_memcpy_time', 'total_time']

    pct_cols = [col for col in byrank_data.columns if 'percent_change' in col.lower()]
    if not pct_cols:
        print("  No percent_change column found")
        return

    pct_col = pct_cols[0]
    ranks = sorted(byrank_data['rank'].unique()) if 'rank' in byrank_data.columns else []

    # Create 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = byrank_data[byrank_data['type'] == metric]

        if not metric_data.empty:
            values = [metric_data[metric_data['rank'] == r][pct_col].values[0] if not metric_data[metric_data['rank'] == r].empty else 0 for r in ranks]

            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
            ax.bar(ranks, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Rank', fontsize=10)
            ax.set_ylabel('Percent Change (%)', fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(ranks)

    plt.suptitle('Percentage Time Difference Across Ranks (All Metrics)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'plot8_percentage_difference_all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot8_percentage_difference_all_metrics.png")


def plot9_nccl_latency(nccl_data, output_dir):
    """Plot 9: Communication Latency Comparison per Message Size."""
    print("\nGenerating Plot 9: Communication Latency vs Message Size")

    if nccl_data is None or nccl_data.empty:
        print("  No NCCL data available")
        return

    if 'In msg nelems' not in nccl_data.columns:
        print("  Required columns not found")
        return

    latency_cols = [col for col in nccl_data.columns if 'comm_latency' in col.lower() or 'latency_mean' in col.lower()]
    if not latency_cols:
        print("  No latency columns found")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    nccl_sorted = nccl_data.sort_values('In msg nelems')
    msg_sizes = nccl_sorted['In msg nelems'].values

    x = np.arange(len(msg_sizes))
    width = 0.35

    if len(latency_cols) >= 2:
        baseline_values = nccl_sorted[latency_cols[0]].values
        test_values = nccl_sorted[latency_cols[1]].values

        baseline_label = latency_cols[0].replace('_comm_latency_mean', '').replace('_', ' ').title()
        test_label = latency_cols[1].replace('_comm_latency_mean', '').replace('_', ' ').title()

        ax.bar(x - width/2, baseline_values, width, label=baseline_label, alpha=0.8, color='steelblue')
        ax.bar(x + width/2, test_values, width, label=test_label, alpha=0.8, color='darkorange')
    else:
        ax.bar(x, nccl_sorted[latency_cols[0]].values, alpha=0.8, color='steelblue')

    ax.set_xlabel('Message Size (elements)', fontsize=12)
    ax.set_ylabel('Communication Latency (ms)', fontsize=12)
    ax.set_title('Communication Latency Comparison per Message Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(s):,}' for s in msg_sizes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'plot9_nccl_latency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot9_nccl_latency.png")


def plot10_algorithm_bandwidth(nccl_data, output_dir):
    """Plot 10: Algorithm Bandwidth."""
    print("\nGenerating Plot 10: Algorithm Bandwidth")

    if nccl_data is None or nccl_data.empty:
        print("  No NCCL data available")
        return

    algo_bw_cols = [col for col in nccl_data.columns if 'algo bw' in col.lower()]
    if not algo_bw_cols or 'In msg nelems' not in nccl_data.columns:
        print("  Required columns not found")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    nccl_sorted = nccl_data.sort_values('In msg nelems')
    msg_sizes = nccl_sorted['In msg nelems'].values

    x = np.arange(len(msg_sizes))
    width = 0.35

    if len(algo_bw_cols) >= 2:
        baseline_values = nccl_sorted[algo_bw_cols[0]].values
        test_values = nccl_sorted[algo_bw_cols[1]].values

        baseline_label = algo_bw_cols[0].replace('_algo bw (GB/s)_mean', '').replace('_', ' ').title()
        test_label = algo_bw_cols[1].replace('_algo bw (GB/s)_mean', '').replace('_', ' ').title()

        ax.bar(x - width/2, baseline_values, width, label=baseline_label, alpha=0.8, color='steelblue')
        ax.bar(x + width/2, test_values, width, label=test_label, alpha=0.8, color='darkorange')
    else:
        ax.bar(x, nccl_sorted[algo_bw_cols[0]].values, alpha=0.8, color='steelblue')

    ax.set_xlabel('Message Size (elements)', fontsize=12)
    ax.set_ylabel('Algorithm Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Algorithm Bandwidth Comparison per Message Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(s):,}' for s in msg_sizes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'plot10_algorithm_bandwidth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot10_algorithm_bandwidth.png")


def plot11_bus_bandwidth(nccl_data, output_dir):
    """Plot 11: Bus Bandwidth."""
    print("\nGenerating Plot 11: Bus Bandwidth")

    if nccl_data is None or nccl_data.empty:
        print("  No NCCL data available")
        return

    bus_bw_cols = [col for col in nccl_data.columns if 'bus bw' in col.lower()]
    if not bus_bw_cols or 'In msg nelems' not in nccl_data.columns:
        print("  Required columns not found")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    nccl_sorted = nccl_data.sort_values('In msg nelems')
    msg_sizes = nccl_sorted['In msg nelems'].values

    x = np.arange(len(msg_sizes))
    width = 0.35

    if len(bus_bw_cols) >= 2:
        baseline_values = nccl_sorted[bus_bw_cols[0]].values
        test_values = nccl_sorted[bus_bw_cols[1]].values

        baseline_label = bus_bw_cols[0].replace('_bus bw (GB/s)_mean', '').replace('_', ' ').title()
        test_label = bus_bw_cols[1].replace('_bus bw (GB/s)_mean', '').replace('_', ' ').title()

        ax.bar(x - width/2, baseline_values, width, label=baseline_label, alpha=0.8, color='steelblue')
        ax.bar(x + width/2, test_values, width, label=test_label, alpha=0.8, color='darkorange')
    else:
        ax.bar(x, nccl_sorted[bus_bw_cols[0]].values, alpha=0.8, color='steelblue')

    ax.set_xlabel('Message Size (elements)', fontsize=12)
    ax.set_ylabel('Bus Bandwidth (GB/s)', fontsize=12)
    ax.set_title('Bus Bandwidth Comparison per Message Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(s):,}' for s in msg_sizes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'plot11_bus_bandwidth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot11_bus_bandwidth.png")


def plot12_nccl_summary(nccl_data, output_dir):
    """Plot 12: NCCL Percentage Summary and Total Communication Latency."""
    print("\nGenerating Plot 12: NCCL Summary (Percentage & Total Latency)")

    if nccl_data is None or nccl_data.empty:
        print("  No NCCL data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Percentage change summary for key metrics
    pct_cols = [col for col in nccl_data.columns if 'percent_change' in col.lower()]
    if pct_cols and len(pct_cols) > 0:
        metrics = []
        values = []

        for col in pct_cols:
            metric_name = col.replace('percent_change_', '').replace('_', ' ').title()
            metrics.append(metric_name)
            avg_value = nccl_data[col].mean()
            values.append(avg_value)

        if metrics:
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
            bars = ax1.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

            for bar, val in zip(bars, values):
                x_pos = bar.get_width()
                ax1.text(x_pos + (1 if x_pos > 0 else -1), bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', ha='left' if x_pos > 0 else 'right', va='center', fontweight='bold')

            ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax1.set_xlabel('Percent Change (%)', fontsize=12)
            ax1.set_title('NCCL Metrics: Average Percent Change', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No percentage change data available',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    # Right: Total communication latency comparison
    total_latency_cols = [col for col in nccl_data.columns if ('Total comm latency' in col or 'total_latency' in col.lower()) and 'percent' not in col.lower()]

    if total_latency_cols and len(total_latency_cols) >= 1:
        labels = []
        totals = []

        for col in total_latency_cols[:2]:
            label = col.replace('_Total comm latency (ms)', '').replace('_total_latency', '').replace('_', ' ').strip().title()
            if not label:
                label = 'Total'
            total = nccl_data[col].sum()
            labels.append(label)
            totals.append(total)

        if totals:
            colors = ['steelblue', 'darkorange'] if len(totals) > 1 else ['steelblue']
            bars = ax2.bar(labels, totals, color=colors[:len(totals)], alpha=0.8, edgecolor='black', linewidth=1)

            for bar, val in zip(bars, totals):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

            if len(totals) == 2 and totals[0] > 0:
                improvement = (totals[0] - totals[1]) / totals[0] * 100
                y_pos = max(totals) * 0.6
                ax2.text(0.5, y_pos, f'Improvement: {improvement:.1f}%',
                        ha='center', fontsize=13, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6, edgecolor='black'))

            ax2.set_ylabel('Total Communication Latency (ms)', fontsize=12)
            ax2.set_title('Total Communication Latency Comparison', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No total latency data available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot12_nccl_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot12_nccl_summary.png")


def generate_html_report(input_path, output_dir, baseline_label='Baseline', test_label='Test'):
    """Generate HTML report with all plots embedded."""
    print("\nGenerating HTML Report...")

    plot_files = sorted(output_dir.glob('plot*.png'))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RCCL Performance Analysis: {baseline_label} vs {test_label}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 40px;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .plot-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        @media print {{
            .plot-container {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <h1>RCCL Performance Analysis Report</h1>
    <h2 style="text-align: center; color: #3498db;">Comparing: {baseline_label} vs {test_label}</h2>
    <p style="text-align: center;"><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>GPU Performance Metrics</h2>
"""

    plot_titles = {
        'plot1': 'Percentage Change Overview',
        'plot2': 'Absolute Time Comparison',
        'plot3': 'Performance Heatmap by Rank',
        'plot4': 'Total Execution Time by Rank',
        'plot5': 'Computation Time Across Ranks',
        'plot6': 'Communication Time Across Ranks',
        'plot7': 'Idle Time Across Ranks',
        'plot8': 'Percentage Time Difference (All Metrics)',
        'plot9': 'NCCL Communication Latency',
        'plot10': 'NCCL Algorithm Bandwidth',
        'plot11': 'NCCL Bus Bandwidth',
        'plot12': 'NCCL Summary'
    }

    # Add GPU plots first (plot1-plot8)
    for plot_file in plot_files:
        plot_num = plot_file.stem.split('_')[0]
        if plot_num not in ['plot1', 'plot2', 'plot3', 'plot4', 'plot5', 'plot6', 'plot7', 'plot8']:
            continue

        title = plot_titles.get(plot_num, plot_file.stem.replace('_', ' ').title())

        with open(plot_file, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()

        html_content += f"""
    <div class="plot-container">
        <div class="plot-title">{title}</div>
        <img src="data:image/png;base64,{img_data}" alt="{title}">
    </div>
"""

    # Add NCCL section
    html_content += "\n    <h2>NCCL/Collective Performance</h2>\n"

    # Add NCCL plots (plot9-plot12)
    for plot_file in plot_files:
        plot_num = plot_file.stem.split('_')[0]
        if plot_num not in ['plot9', 'plot10', 'plot11', 'plot12']:
            continue

        title = plot_titles.get(plot_num, plot_file.stem.replace('_', ' ').title())

        with open(plot_file, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()

        html_content += f"""
    <div class="plot-container">
        <div class="plot-title">{title}</div>
        <img src="data:image/png;base64,{img_data}" alt="{title}">
    </div>
"""

    html_content += """
    <p style="text-align: center; margin-top: 50px; color: #7f8c8d;">
        Generated by TraceLens Analysis Pipeline
    </p>
</body>
</html>
"""

    html_path = output_dir / 'performance_analysis_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  HTML report saved to: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description='Generate 12 analysis plots')
    parser.add_argument('--input', required=True, help='Path to final_analysis_report.xlsx')
    parser.add_argument('--output', default='plots', help='Output directory for plots')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    sheets = pd.read_excel(input_path, sheet_name=None)

    print(f"\nGenerating 12 plots from {input_path.name}...")

    # Extract baseline and test labels from Summary_Dashboard
    baseline_label = 'Baseline'
    test_label = 'Test'

    summary_sheet = sheets.get('Summary_Dashboard')
    if summary_sheet is not None:
        columns = summary_sheet.columns.tolist()
        if len(columns) >= 3:
            baseline_label = columns[1]
            test_label = columns[2]

        plot1_percentage_change(summary_sheet, output_dir)
        plot2_absolute_time_comparison(summary_sheet, output_dir)

    # GPU by-rank data
    byrank_sheet = None
    for name in ['GPU_ByRank_Cmp', 'GPU_ByRank_Comparison', 'Comparison_By_Rank']:
        if name in sheets:
            byrank_sheet = sheets[name]
            break

    if byrank_sheet is not None:
        plot3_performance_heatmap(byrank_sheet, output_dir)
        plot4_total_execution_time(byrank_sheet, output_dir)
        plot5_computation_time(byrank_sheet, output_dir)
        plot6_communication_time(byrank_sheet, output_dir)
        plot7_idle_time(byrank_sheet, output_dir)
        plot8_percentage_time_difference(byrank_sheet, output_dir)

    # NCCL data
    nccl_sheet = None
    for name in sheets:
        if 'nccl' in name.lower() and ('cmp' in name.lower() or 'comparison' in name.lower()):
            nccl_sheet = sheets[name]
            break

    # Try to get the actual NCCL data sheets (not just comparison)
    if not nccl_sheet or nccl_sheet.empty:
        for name in sheets:
            if 'nccl' in name.lower() and 'summary' in name.lower():
                nccl_sheet = sheets[name]
                break

    if nccl_sheet is not None and not nccl_sheet.empty:
        plot9_nccl_latency(nccl_sheet, output_dir)
        plot10_algorithm_bandwidth(nccl_sheet, output_dir)
        plot11_bus_bandwidth(nccl_sheet, output_dir)
        plot12_nccl_summary(nccl_sheet, output_dir)

    # Generate HTML report with configuration labels
    html_path = generate_html_report(input_path, output_dir, baseline_label, test_label)

    print(f"\n{'='*60}")
    print(f"All 12 plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"\nHTML Report: {html_path}")
    print("  - Open in browser to view all plots")
    print("  - Print to PDF: Ctrl+P or Cmd+P")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
