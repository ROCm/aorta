#!/usr/bin/env python3
"""Visualize compute-communication timeline from metrics JSONL files."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will use text-based visualization")


def load_metrics(jsonl_file: Path):
    """Load metrics from a JSONL file."""
    records = []
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    return records


def extract_timeline_data(records, start_step=5, num_steps=5):
    """Extract timeline data for visualization."""
    timeline_data = []

    for record in records:
        step = record.get("step", 0)
        if step < start_step or step >= start_step + num_steps:
            continue

        profile = record.get("profile", {})
        ranges = profile.get("ranges", [])

        for r in ranges:
            timeline_data.append({
                "step": step,
                "stream": r.get("stream", "unknown"),
                "tag": r.get("tag", "unknown"),
                "start_ms": r.get("start_ms", 0.0),
                "end_ms": r.get("end_ms", 0.0),
                "duration_ms": r.get("duration_ms", 0.0),
            })

    return timeline_data


def categorize_operation(tag, stream):
    """Categorize operation into compute/comm/other."""
    tag_lower = tag.lower()

    if "forward" in tag_lower or "backward" in tag_lower:
        return "compute"
    elif "all_gather" in tag_lower or "reduce_scatter" in tag_lower or "all_reduce" in tag_lower:
        return "comm"
    elif "optimizer" in tag_lower or "grad_clip" in tag_lower:
        return "optimizer"
    elif "prefetch" in tag_lower:
        return "prefetch"
    else:
        return "other"


def plot_timeline_matplotlib(timeline_data, output_file=None):
    """Create timeline visualization using matplotlib."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available!")
        return

    # Group by step and stream
    steps = sorted(set(d["step"] for d in timeline_data))
    streams = sorted(set(d["stream"] for d in timeline_data))

    # Create color mapping for operation types
    colors = {
        "compute": "#2E86AB",      # Blue
        "comm": "#A23B72",          # Purple
        "optimizer": "#F18F01",     # Orange
        "prefetch": "#C73E1D",      # Red
        "other": "#CCCCCC",         # Gray
    }

    # Create figure with subplots for each step
    fig, axes = plt.subplots(len(steps), 1, figsize=(14, 3 * len(steps)), squeeze=False)

    for step_idx, step in enumerate(steps):
        ax = axes[step_idx, 0]

        # Filter data for this step
        step_data = [d for d in timeline_data if d["step"] == step]

        if not step_data:
            continue

        # Normalize time to start at 0 for this step
        min_start = min(d["start_ms"] for d in step_data)

        # Plot each stream on a separate horizontal track
        stream_y_positions = {stream: i for i, stream in enumerate(streams)}

        for data in step_data:
            stream = data["stream"]
            y_pos = stream_y_positions[stream]

            start = data["start_ms"] - min_start
            duration = data["duration_ms"]

            category = categorize_operation(data["tag"], stream)
            color = colors.get(category, colors["other"])

            # Draw rectangle for this operation
            rect = mpatches.Rectangle(
                (start, y_pos - 0.4),
                duration,
                0.8,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Add text label for longer operations
            if duration > 1.0:  # Only label operations > 1ms
                label = data["tag"].replace(f"epoch0_step{step}_", "")
                ax.text(
                    start + duration / 2,
                    y_pos,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    rotation=0,
                )

        # Configure axes
        ax.set_ylim(-0.5, len(streams) - 0.5)
        ax.set_yticks(range(len(streams)))
        ax.set_yticklabels(streams)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Step {step} Timeline")
        ax.grid(True, axis="x", alpha=0.3)

        # Set x-axis limits
        max_end = max(d["end_ms"] - min_start for d in step_data)
        ax.set_xlim(0, max_end * 1.05)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=colors["compute"], label="Compute"),
        mpatches.Patch(facecolor=colors["comm"], label="Communication"),
        mpatches.Patch(facecolor=colors["optimizer"], label="Optimizer"),
        mpatches.Patch(facecolor=colors["prefetch"], label="Prefetch"),
        mpatches.Patch(facecolor=colors["other"], label="Other"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\nTimeline saved to: {output_file}")
    else:
        plt.show()


def plot_timeline_text(timeline_data, resolution_ms=0.5):
    """Create text-based timeline visualization."""
    steps = sorted(set(d["step"] for d in timeline_data))
    streams = sorted(set(d["stream"] for d in timeline_data))

    # Character mapping for operation types
    chars = {
        "compute": "█",
        "comm": "▓",
        "optimizer": "▒",
        "prefetch": "░",
        "other": "·",
    }

    print("\n" + "="*100)
    print("TEXT-BASED TIMELINE VISUALIZATION")
    print("="*100)
    print("\nLegend:")
    print(f"  {chars['compute']} = Compute (forward/backward)")
    print(f"  {chars['comm']} = Communication (all-gather, reduce-scatter)")
    print(f"  {chars['optimizer']} = Optimizer/Grad Clip")
    print(f"  {chars['prefetch']} = Prefetch")
    print(f"  {chars['other']} = Other")
    print(f"\nTime resolution: {resolution_ms} ms per character")

    for step in steps:
        print("\n" + "-"*100)
        print(f"STEP {step}")
        print("-"*100)

        # Filter data for this step
        step_data = [d for d in timeline_data if d["step"] == step]

        if not step_data:
            continue

        # Normalize time
        min_start = min(d["start_ms"] for d in step_data)
        max_end = max(d["end_ms"] for d in step_data)
        total_duration = max_end - min_start

        # Calculate timeline width
        timeline_width = int(total_duration / resolution_ms) + 1

        print(f"\nTotal duration: {total_duration:.2f} ms")
        print(f"Timeline: 0 ms {' ' * (timeline_width - 20)} {total_duration:.1f} ms")
        print(f"          |{'-' * (timeline_width - 2)}|")

        # Build timeline for each stream
        for stream in streams:
            # Initialize timeline
            timeline = [' '] * timeline_width

            # Fill in operations
            stream_ops = [d for d in step_data if d["stream"] == stream]
            for data in stream_ops:
                start_pos = int((data["start_ms"] - min_start) / resolution_ms)
                end_pos = int((data["end_ms"] - min_start) / resolution_ms)

                category = categorize_operation(data["tag"], stream)
                char = chars.get(category, chars["other"])

                for i in range(start_pos, min(end_pos + 1, timeline_width)):
                    timeline[i] = char

            # Print stream timeline
            stream_label = f"{stream:15s}"
            print(f"{stream_label} {''.join(timeline)}")

        # Print overlap analysis
        print(f"\nOverlap Analysis:")

        # Check for overlapping operations
        compute_ops = [d for d in step_data if categorize_operation(d["tag"], d["stream"]) == "compute"]
        comm_ops = [d for d in step_data if categorize_operation(d["tag"], d["stream"]) == "comm"]

        total_compute = sum(d["duration_ms"] for d in compute_ops)
        total_comm = sum(d["duration_ms"] for d in comm_ops)

        # Calculate overlap
        overlap_ms = 0.0
        for comp_op in compute_ops:
            comp_start = comp_op["start_ms"]
            comp_end = comp_op["end_ms"]

            for comm_op in comm_ops:
                comm_start = comm_op["start_ms"]
                comm_end = comm_op["end_ms"]

                # Calculate overlap
                overlap_start = max(comp_start, comm_start)
                overlap_end = min(comp_end, comm_end)

                if overlap_start < overlap_end:
                    overlap_ms += (overlap_end - overlap_start)

        print(f"  Compute time: {total_compute:.2f} ms")
        print(f"  Comm time: {total_comm:.2f} ms")
        print(f"  Overlap time: {overlap_ms:.2f} ms")
        if total_compute > 0:
            print(f"  Overlap ratio: {overlap_ms/total_compute:.4f} ({overlap_ms/total_compute*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize compute-communication timeline from metrics"
    )
    parser.add_argument(
        "metrics_dir",
        type=Path,
        help="Directory containing rank_*_metrics.jsonl files",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Which rank to analyze (default: 0)",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=5,
        help="First step to visualize (default: 5 to skip warmup)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of steps to visualize (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for matplotlib plot (e.g., timeline.png)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Force text-based visualization even if matplotlib is available",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Time resolution in ms for text visualization (default: 0.5)",
    )

    args = parser.parse_args()

    # Find metrics file
    metrics_file = args.metrics_dir / f"rank_{args.rank:02d}_metrics.jsonl"
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)

    print(f"Loading: {metrics_file}")
    records = load_metrics(metrics_file)
    print(f"Loaded {len(records)} iterations")

    # Extract timeline data
    timeline_data = extract_timeline_data(records, args.start_step, args.num_steps)
    print(f"Extracted {len(timeline_data)} timeline events for steps {args.start_step}-{args.start_step + args.num_steps - 1}")

    if not timeline_data:
        print("No timeline data found for the specified steps!")
        sys.exit(1)

    # Visualize
    if HAS_MATPLOTLIB and not args.text_only:
        plot_timeline_matplotlib(timeline_data, args.output)
    else:
        plot_timeline_text(timeline_data, args.resolution)

    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
