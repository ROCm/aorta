import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def plot_improvement_chart(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars based on positive/negative values
    colors = ["#2ecc71" if val > 0 else "#e74c3c" for val in df["Improvement (%)"]]

    bars = ax.barh(df["Metric"], df["Improvement (%)"], color=colors)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Customize the chart
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_xlabel("Change (%)", fontsize=12)
    ax.set_title(
        "GPU Metrics Percentage Change (Test vs Baseline)\n(Positive = Test is better)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path / "improvement_chart.png", dpi=150)
    plt.close()


def plot_abs_time_comparison(df, output_path):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up bar positions
    x = range(len(df))
    width = 0.35

    # Create bars for Baseline and Test
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        df["Baseline"],
        width,
        label="Baseline",
        color="#3498db",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x], df["Test"], width, label="Test", color="#e67e22"
    )

    # Add horizontal grid lines only
    ax.xaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)

    # Remove border/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Customize the chart
    ax.set_xlabel("Metric Type", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(
        "GPU Metrics Absolute Time Comparison ", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "abs_time_comparison.png", dpi=150)
    plt.close()


def create_summary_charts(excel_path, output_path):

    # Read the Summary_Dashboard sheet
    df = pd.read_excel(excel_path, sheet_name="Summary_Dashboard")

    plot_improvement_chart(df, output_path)
    plot_abs_time_comparison(df, output_path)
    # Create the horizontal bar chart


def plot_gpu_type_by_rank(total_time_df, output_path, title):
    # Create the line plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot baseline total_time by rank
    ax.plot(
        total_time_df["rank"],
        total_time_df["baseline_time_ms"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#3498db",
        label="Baseline",
    )

    # Plot Saleelk (test) total_time by rank
    ax.plot(
        total_time_df["rank"],
        total_time_df["saleelk_time_ms"],
        marker="s",
        linewidth=2,
        markersize=8,
        color="#e67e22",
        label="Test",
    )

    # Add horizontal grid lines only
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)

    # Customize the chart
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Total Time (ms)", fontsize=12)
    ax.set_title(f"{title} Comparison across all ranks", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_gpu_time_accross_all_ranks(excel_path, output_path):
    # Read the GPU_ByRank_Cmp sheet
    df = pd.read_excel(excel_path, sheet_name="GPU_ByRank_Cmp")

    # Filter for total_time rows only
    for type in ["total_time", "computation_time", "total_comm_time", "idle_time"]:
        total_time_df = df[df["type"] == type]
        plot_gpu_type_by_rank(total_time_df, output_path / f"{type}_by_rank.png", type)


def plot_gpu_time_change_percentage_summaryby_rank(df, ax):
    colors = ["#2ecc71" if val > 0 else "#e74c3c" for val in df["percent_change"]]
    bars = ax.bar(df["rank"].astype(str), df["percent_change"], color=colors)
    # Add horizontal line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add horizontal grid lines only
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Percent Change (%)")


def create_gpu_time_change_percentage_summaryby_rank(excel_path, output_path):
    # Read the GPU_ByRank_Cmp sheet
    df = pd.read_excel(excel_path, sheet_name="GPU_ByRank_Cmp")

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    row_types = [
        "busy_time",
        "computation_time",
        "exposed_comm_time",
        "exposed_memcpy_time",
        "idle_time",
        "total_comm_time",
        "total_memcpy_time",
        "total_time",
    ]
    # Filter for total_time rows only
    for i, type in enumerate(row_types):
        type_df = df[df["type"] == type]
        plot_gpu_time_change_percentage_summaryby_rank(type_df, ax[i // 4, i % 4])
        ax[i // 4, i % 4].set_title(f"{type}")
    plt.tight_layout()
    plt.savefig(output_path / "gpu_time_change_percentage_summary_by_rank.png", dpi=150)
    plt.close()


def create_nccl_charts(excel_path, output_path):
    # Read the NCCL_Charst sheet
    df = pd.read_excel(excel_path, sheet_name="NCCL_ImplSync_Cmp")
    df["label"] = df["Collective name"] + "\n" + df["In msg nelems"].astype(str)
    x = range(len(df))

    plot_item = {
        "NCCL Communication Latency": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Communication Latency (ms)",
            "y_col_names": ["baseline_comm_latency_mean", "saleelk_comm_latency_mean"],
        },
        "NCCL Algorithm Bandwidth": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Algorithm Bandwidth (GB/s)",
            "y_col_names": [
                "baseline_algo bw (GB/s)_mean",
                "saleelk_algo bw (GB/s)_mean",
            ],
        },
        "NCCL Bus Bandwidth": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Bus Bandwidth (GB/s)",
            "y_col_names": [
                "baseline_bus bw (GB/s)_mean",
                "saleelk_bus bw (GB/s)_mean",
            ],
        },
        "NCCL Total Communication Latency": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Total Communication Latency (ms)",
            "y_col_names": [
                "baseline_Total comm latency (ms)",
                "saleelk_Total comm latency (ms)",
            ],
        },
    }
    for item in plot_item.keys():
        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.35
        bars1 = ax.bar(
            [i - width / 2 for i in x],
            df[plot_item[item]["y_col_names"][0]],
            width,
            label="Baseline",
            color="#3498db",
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            df[plot_item[item]["y_col_names"][1]],
            width,
            label="Test",
            color="#e67e22",
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(plot_item[item]["x_label"], fontsize=12)
        ax.set_ylabel(plot_item[item]["y_label"], fontsize=12)
        ax.set_title(f"{item} Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / f'{item.replace(" ", "_")}_comparison.png', dpi=150)
        plt.close()

    percentage_chart_item = {
        "Comm Latency": "percent_change_comm_latency_mean",
        "Algo BW": "percent_change_algo bw (GB/s)_mean",
        "Bus BW": "percent_change_bus bw (GB/s)_mean",
    }
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    plot_item_index = 0
    for item in percentage_chart_item.keys():
        colors = [
            "#2ecc71" if val > 0 else "#e74c3c"
            for val in df[percentage_chart_item[item]]
        ]
        bars = ax[plot_item_index].barh(
            df["In msg nelems"].astype(str),
            df[percentage_chart_item[item]],
            color=colors,
        )
        ax[plot_item_index].yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax[plot_item_index].set_axisbelow(True)
        ax[plot_item_index].set_xlabel("Percent Change (%)")
        ax[plot_item_index].set_title(f"{item} \n Percent Change (Positive = better)")
        plot_item_index += 1
    fig.suptitle(
        "NCCL Performance Percentage Change By Message Size",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_path / f"NCCL_Performance_Percentage_Change_comparison.png", dpi=150
    )
    plt.close()


def create_gpu_time_heatmap(excel_path, output_path):
    # Read the GPU_ByRank_Cmp sheet
    df = pd.read_excel(excel_path, sheet_name="GPU_ByRank_Cmp")
    # Plot the GPU time heatmap
    pivot_df = df.pivot(index="type", columns="rank", values="percent_change")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot_df,
        annot=True,  # Show values in cells
        fmt=".1f",  # Format as 1 decimal
        cmap="RdYlGn",  # Red-Yellow-Green colormap (red=bad, green=good)
        center=0,  # Center colormap at 0
        linewidths=0.5,  # Add gridlines
        cbar_kws={"label": "Percent Change (%)"},
    )

    ax.set_title(
        "GPU Metric Percentage Change by Rank (HeatMap) \n (Positive = Better Test)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Metric Type", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / "gpu_time_heatmap.png", dpi=150)
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate improvement chart from generated reports"
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default="~/aorta/aorta_single_config/aorta/expt_compare/final_analysis_report.xlsx",
        help="Path to the input Excel file (should have Summary_Dashboard sheet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output directory to save PNG files",
    )

    args = parser.parse_args()
    output_path = args.output if args.output else args.report_path.parent / "plots"
    output_path.mkdir(exist_ok=True, parents=True)
    create_summary_charts(args.report_path, output_path)
    print(f"Summary charts saved to: {args.output}")
    create_gpu_time_heatmap(args.report_path, output_path)
    print(f"GPU time heatmap saved to: {output_path}")
    create_gpu_time_accross_all_ranks(args.report_path, output_path)
    print(f"GPU time across all runs saved to: {output_path}")
    create_gpu_time_change_percentage_summaryby_rank(args.report_path, output_path)
    print(f"GPU time change percentage summary by rank saved to: {output_path}")
    create_nccl_charts(args.report_path, output_path)
    print(f"NCCL communication charts saved to: {output_path}")


if __name__ == "__main__":
    main()
