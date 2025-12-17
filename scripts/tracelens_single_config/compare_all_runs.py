import argparse
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_command(cmd, description):
    """Execute a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {description} failed!")
        print(f"Stderr: {result.stderr}")
        return False

    print(result.stdout)
    return True


def plot_nccl_data_per_msg(df, labels, output_dir: Path):
    """
    Plot comm_latency_mean for each message size from NCCL data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get unique index values (Collective_MsgSize)
    indices = df["index"].values

    x = np.arange(len(indices))
    width = 0.8 / len(labels)
    # Vibrant color palette
    vibrant_colors = [
        "#E63946",
        "#2A9D8F",
        "#E9C46A",
        "#264653",
        "#F4A261",
        "#8338EC",
        "#06D6A0",
        "#FF006E",
    ]

    plot_items = {
        "NCCL Communication Latency": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Communication Latency (ms)",
            "y_col": "comm_latency_mean",
        },
        "NCCL Algorithm Bandwidth": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Algorithm Bandwidth (GB/s)",
            "y_col": "algo bw (GB/s)_mean",
        },
        "NCCL Bus Bandwidth": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Bus Bandwidth (GB/s)",
            "y_col": "bus bw (GB/s)_mean",
        },
        "NCCL Total Communication Latency": {
            "x_label": "Collective Operation (Message Size)",
            "y_label": "Total Communication Latency (ms)",
            "y_col": "Total comm latency (ms)",
        },
    }

    for plot_item in plot_items.keys():
        fig, ax = plt.subplots(figsize=(14, 7))
        for i, label in enumerate(labels):
            col_name = f"{plot_items[plot_item]['y_col']}_{label}"
            print(f"Plotting {col_name}")
            if col_name in df.columns:
                values = df[col_name].values
                color = vibrant_colors[i % len(vibrant_colors)]
                offset = (i - len(labels) / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    values,
                    width,
                    label=label,
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )
            else:
                print(f"Column {col_name} not found in dataframe")

        ax.set_xlabel(plot_items[plot_item]["x_label"], fontsize=12, fontweight="bold")
        ax.set_ylabel(plot_items[plot_item]["y_label"], fontsize=12, fontweight="bold")
        ax.set_title(f"{plot_item} per Message Size", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(indices, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_file = output_path / f'{plot_item.replace(" ", "_")}_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}")
    print("Completed plotting NCCL data per message size")


def plot_all_types_per_rank(df, labels, output_dir: Path):
    """
    Plot data for every rank, where every unique type is a different file.

    Parameters:
    -----------
    df : DataFrame
        Merged gpu_time_per_rank_df with columns like 'type', 'rank0_label1', 'rank0_label2', etc.
    labels : list
        List of configuration labels (e.g., ['32cu_512threads', '37cu_384threads'])
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unique_types = df["type"].unique()

    # Find rank columns (extract rank numbers from column names)
    # Columns are like: rank0_32cu_512threads, rank1_32cu_512threads, etc.
    sample_label = labels[0]
    rank_cols = [col for col in df.columns if col.endswith(f"_{sample_label}") and col != "type"]
    ranks = [col.replace(f"_{sample_label}", "") for col in rank_cols]

    print(f"Found ranks: {ranks}")
    print(f"Found types: {unique_types}")

    for metric_type in unique_types:
        type_data = df[df["type"] == metric_type]

        if type_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(ranks))
        # Vibrant color palette
        vibrant_colors = [
            "#E63946",
            "#2A9D8F",
            "#E9C46A",
            "#264653",
            "#F4A261",
            "#8338EC",
            "#06D6A0",
            "#FF006E",
        ]
        markers = ["o", "s", "^", "D", "v", "p", "h", "*"]

        for i, label in enumerate(labels):
            values = []
            for rank in ranks:
                col_name = f"{rank}_{label}"
                if col_name in type_data.columns:
                    val = type_data[col_name].values[0]
                    values.append(val if pd.notna(val) else 0)
                else:
                    values.append(0)

            color = vibrant_colors[i % len(vibrant_colors)]
            marker = markers[i % len(markers)]
            ax.plot(
                x,
                values,
                label=label,
                color=color,
                marker=marker,
                markersize=8,
                linewidth=2,
                alpha=0.85,
            )

        ax.set_xlabel("Rank", fontsize=12, fontweight="bold")
        ax.set_ylabel("Time (ms)", fontsize=12, fontweight="bold")
        ax.set_title(f"{metric_type} - Time per Rank", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ranks)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with sanitized filename
        safe_name = metric_type.replace("/", "_").replace(" ", "_").replace(":", "_")
        output_file = output_path / f"{safe_name}_by_rank.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}")


def plot_gpu_time_summary(df, labels, output_dir: Path):

    types = df["type"].values
    values = []

    for label in labels:
        values.append(df[f"time ms_{label}"].values)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(types))
    width = 0.15
    for i, value in enumerate(values):
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, value, width, label=labels[i])

    ax.set_xlabel("Type")
    ax.set_ylabel("Time (ms)")
    ax.set_title("GPU Time Summary by Rank")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "abs_time_comparison.png")
    plt.close()


"""
def plot_improvement_chart(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars based on positive/negative values
    colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in df['Improvement (%)']]

    bars = ax.barh(df['Metric'], df['Improvement (%)'], color=colors)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Customize the chart
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_xlabel('Change (%)', fontsize=12)
    ax.set_title(
        'GPU Metrics Percentage Change (Test vs Baseline)\n(Positive = Test is better)',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path / 'improvement_chart.png', dpi=150)
    plt.close()

"""


def plot_gpu_time_percentage_change(df, labels, output_dir: Path):
    """
    Create separate horizontal bar charts for each label comparing against baseline (labels[0]).
    """
    types = df["type"].values
    base_label = labels[0]

    # Vibrant color palette
    vibrant_colors = [
        "#E63946",
        "#2A9D8F",
        "#E9C46A",
        "#264653",
        "#F4A261",
        "#8338EC",
        "#06D6A0",
        "#FF006E",
    ]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, max(8, len(types) * 0.5)))
    for i, label in enumerate(labels[1:]):
        ax = axes[i]
        col_name = f"percentage_change_{label}"
        if col_name not in df.columns:
            print(f"Column {col_name} not found, skipping")
            continue

        values = df[col_name].values

        # Create 1x2 subplot figure

        # Color bars based on positive/negative values (green = improvement, red = regression)
        colors = ["#2ecc71" if val < 0 else "#e74c3c" for val in values]

        # Horizontal bar chart
        y = np.arange(len(types))
        bars = ax.barh(y, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

        # Add vertical line at 0
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels(types, fontsize=10)
        ax.set_xlabel("Percentage Change (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Type", fontsize=12, fontweight="bold")
        ax.set_title(
            f"GPU Time Percentage Change: {label} vs {base_label}\n(Negative = Improvement)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    output_file = output_dir / f"improvement_chart.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def calculate_gpu_timepercentage_change(df, labels):
    base_label = labels[0]
    for label in labels[1:]:
        df[f"percentage_change_{label}"] = (
            (df[f"time ms_{label}"] - df[f"time ms_{base_label}"])
            / df[f"time ms_{base_label}"]
            * 100
        )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing gpu_timeline_summary_mean.xlsx",
    )

    parser.add_argument("--output", type=Path, required=True, help="Output xls file name")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    labels = []
    summary_dfs = []
    gpu_time_per_rank_dfs = []
    nccl_dfs = []

    for directory in args.inputs:
        dir_path = Path(directory)
        label = dir_path.stem

        if not dir_path.exists():
            print(f"Directory not found: {dir_path}")
            continue

        input_excel_file = dir_path / "tracelens_analysis" / "gpu_timeline_summary_mean.xlsx"
        nccl_excel_file = (
            dir_path / "tracelens_analysis" / "collective_reports" / "collective_all_ranks.xlsx"
        )
        if not input_excel_file.exists() or not nccl_excel_file.exists():
            print(f"ERROR: Required files not found")
            print(
                f"  GPU file: {input_excel_file} - {'OK' if input_excel_file.exists() else 'MISSING'}"
            )
            print(
                f"  NCCL file: {nccl_excel_file} - {'OK' if nccl_excel_file.exists() else 'MISSING'}"
            )
            continue

        labels.append(label)

        # Read and rename columns with label suffix
        summary = pd.read_excel(input_excel_file, sheet_name="Summary")
        gpu_time = pd.read_excel(input_excel_file, sheet_name="Per_Rank_Time_ms")

        # Rename non-key columns with label suffix
        summary = summary.rename(
            columns={col: f"{col}_{label}" for col in summary.columns if col != "type"}
        )
        gpu_time = gpu_time.rename(
            columns={col: f"{col}_{label}" for col in gpu_time.columns if col != "type"}
        )

        summary_dfs.append(summary)
        gpu_time_per_rank_dfs.append(gpu_time)
        print(f"Loaded: {label}")

        # Process NCCL file (guaranteed to exist from earlier check)
        nccl_df = pd.read_excel(nccl_excel_file, sheet_name="nccl_summary_implicit_sync")

        # Create index column by appending "Collective name" and "In msg nelems"
        nccl_df["index"] = (
            nccl_df["Collective name"].astype(str) + "_" + nccl_df["In msg nelems"].astype(str)
        )

        # Rename non-key columns with label suffix (exclude 'index' as it's the merge key)
        nccl_df = nccl_df.rename(
            columns={col: f"{col}_{label}" for col in nccl_df.columns if col != "index"}
        )
        nccl_dfs.append(nccl_df)
        print(f"Loaded: {label} NCCL")

    # Merge all DataFrames on 'type'
    summary_df = summary_dfs[0]
    gpu_time_per_rank_df = gpu_time_per_rank_dfs[0]
    nccl_df = nccl_dfs[0]

    for i in range(1, len(summary_dfs)):
        summary_df = pd.merge(summary_df, summary_dfs[i], on="type", how="outer")
        gpu_time_per_rank_df = pd.merge(
            gpu_time_per_rank_df, gpu_time_per_rank_dfs[i], on="type", how="outer"
        )
        nccl_df = pd.merge(nccl_df, nccl_dfs[i], on="index", how="outer")

    summary_df = calculate_gpu_timepercentage_change(summary_df, labels)

    with pd.ExcelWriter(
        args.output / "final_analysis_report_for_all.xlsx", engine="openpyxl"
    ) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        gpu_time_per_rank_df.to_excel(writer, sheet_name="Per_Rank_Time_ms", index=False)
        nccl_df.to_excel(writer, sheet_name="NCCL_Summary", index=False)

    output_dir = Path(args.output) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_gpu_time_percentage_change(summary_df, labels, output_dir)
    plot_gpu_time_summary(summary_df, labels, output_dir)
    plot_all_types_per_rank(gpu_time_per_rank_df, labels, output_dir)
    plot_nccl_data_per_msg(nccl_df, labels, output_dir)

    html_script_path = Path(__file__).parent / "create_final_html.py"
    cmd = [
        "python3",
        str(html_script_path),
        "--plot-files-directory",
        str(output_dir),
        "--output-html",
        str(args.output / "final_analysis_report_for_all.html"),
    ]
    if run_command(cmd, "Creating final HTML"):
        print(
            f"Final HTML file created at: {args.output.parent / 'final_analysis_report_for_all.html'}"
        )
    else:
        print("Failed to create final HTML file")


if __name__ == "__main__":
    main()
