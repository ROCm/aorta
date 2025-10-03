#!/usr/bin/env python3
"""Analyze rank_*_metrics.jsonl files to understand overlap and stream usage."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import sys


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


def analyze_streams(records, skip_warmup=5):
    """Analyze stream usage and timing across iterations."""
    records = records[skip_warmup:] if len(records) > skip_warmup else records

    stream_stats = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "operations": defaultdict(int)})

    for record in records:
        profile = record.get("profile", {})
        ranges = profile.get("ranges", [])

        for r in ranges:
            stream = r.get("stream", "unknown")
            tag = r.get("tag", "unknown")
            duration = r.get("duration_ms", 0.0)

            stream_stats[stream]["count"] += 1
            stream_stats[stream]["total_ms"] += duration
            stream_stats[stream]["operations"][tag] += 1

    return stream_stats


def analyze_overlap(records, skip_warmup=5):
    """Analyze compute-communication overlap metrics."""
    records = records[skip_warmup:] if len(records) > skip_warmup else records

    overlap_data = []

    for record in records:
        step = record.get("step", 0)
        profile = record.get("profile", {})
        overlap = profile.get("overlap", {})

        overlap_data.append({
            "step": step,
            "overlap_ratio": overlap.get("overlap_ratio", {}).get("compute_comm", 0.0),
            "overlap_ms": overlap.get("overlap_ms", {}).get("compute_comm", 0.0),
            "compute_ms": overlap.get("per_stream_ms", {}).get("compute", 0.0),
            "comm_ms": overlap.get("per_stream_ms", {}).get("comm", 0.0),
            "total_ms": profile.get("total_ms", 0.0),
        })

    return overlap_data


def print_stream_summary(stream_stats):
    """Print summary of stream usage."""
    print("\n" + "="*80)
    print("STREAM USAGE SUMMARY")
    print("="*80)

    if not stream_stats:
        print("No stream data found!")
        return

    print(f"\n{'Stream':<15} {'Operations':<15} {'Total Time (ms)':<20} {'Avg per Op (ms)':<20}")
    print("-" * 80)

    for stream, stats in sorted(stream_stats.items()):
        count = stats["count"]
        total_ms = stats["total_ms"]
        avg_ms = total_ms / count if count > 0 else 0.0

        print(f"{stream:<15} {count:<15} {total_ms:<20.2f} {avg_ms:<20.4f}")

    # Show operation breakdown for each stream
    for stream, stats in sorted(stream_stats.items()):
        print(f"\n  {stream} operations:")
        for op, count in sorted(stats["operations"].items(), key=lambda x: -x[1])[:10]:
            print(f"    {op:<40} {count:>6} times")


def print_overlap_summary(overlap_data):
    """Print summary of overlap metrics."""
    print("\n" + "="*80)
    print("OVERLAP SUMMARY")
    print("="*80)

    if not overlap_data:
        print("No overlap data found!")
        return

    avg_overlap_ratio = sum(d["overlap_ratio"] for d in overlap_data) / len(overlap_data)
    avg_compute = sum(d["compute_ms"] for d in overlap_data) / len(overlap_data)
    avg_comm = sum(d["comm_ms"] for d in overlap_data) / len(overlap_data)
    avg_overlap_ms = sum(d["overlap_ms"] for d in overlap_data) / len(overlap_data)
    avg_total = sum(d["total_ms"] for d in overlap_data) / len(overlap_data)

    print(f"\nIterations analyzed: {len(overlap_data)}")
    print(f"Average overlap ratio: {avg_overlap_ratio:.4f} ({avg_overlap_ratio*100:.2f}%)")
    print(f"Average compute time: {avg_compute:.2f} ms")
    print(f"Average communication time: {avg_comm:.2f} ms")
    print(f"Average overlap time: {avg_overlap_ms:.2f} ms")
    print(f"Average total iteration time: {avg_total:.2f} ms")

    # Show per-iteration details
    print(f"\n{'Step':<8} {'Overlap Ratio':<15} {'Compute (ms)':<15} {'Comm (ms)':<15} {'Overlap (ms)':<15}")
    print("-" * 80)

    # Show first 10 and last 5 iterations
    for d in overlap_data[:10]:
        print(
            f"{d['step']:<8} "
            f"{d['overlap_ratio']:<15.4f} "
            f"{d['compute_ms']:<15.2f} "
            f"{d['comm_ms']:<15.2f} "
            f"{d['overlap_ms']:<15.2f}"
        )

    if len(overlap_data) > 15:
        print("  ...")
        for d in overlap_data[-5:]:
            print(
                f"{d['step']:<8} "
                f"{d['overlap_ratio']:<15.4f} "
                f"{d['compute_ms']:<15.2f} "
                f"{d['comm_ms']:<15.2f} "
                f"{d['overlap_ms']:<15.2f}"
            )


def verify_multi_stream_execution(stream_stats):
    """Check if multiple streams were actually used."""
    print("\n" + "="*80)
    print("MULTI-STREAM VERIFICATION")
    print("="*80)

    num_streams = len(stream_stats)
    print(f"\nNumber of active streams: {num_streams}")

    if num_streams > 1:
        print("✓ Multiple streams detected!")
        print("\nActive streams:")
        for stream in sorted(stream_stats.keys()):
            print(f"  - {stream}")
    else:
        print("✗ Only single stream detected")
        print("  This may indicate RCCL multi-stream is not enabled")

    # Check for SDMA-related operations
    sdma_ops = 0
    for stream, stats in stream_stats.items():
        for op in stats["operations"]:
            if "sdma" in op.lower() or "dma" in op.lower():
                sdma_ops += stats["operations"][op]

    if sdma_ops > 0:
        print(f"\n✓ SDMA operations detected: {sdma_ops} total")
    else:
        print("\n⚠ No explicit SDMA operations found in metrics")
        print("  (This doesn't mean SDMA isn't active - it may not be tagged)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rank_*_metrics.jsonl files from training runs"
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
        "--skip-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations to skip (default: 5)",
    )
    parser.add_argument(
        "--show-streams",
        action="store_true",
        help="Show detailed stream analysis",
    )
    parser.add_argument(
        "--show-overlap",
        action="store_true",
        help="Show detailed overlap analysis",
    )
    parser.add_argument(
        "--verify-streams",
        action="store_true",
        help="Verify multi-stream execution",
    )

    args = parser.parse_args()

    # Find metrics file
    metrics_file = args.metrics_dir / f"rank_{args.rank:02d}_metrics.jsonl"
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)

    print(f"Analyzing: {metrics_file}")

    # Load records
    records = load_metrics(metrics_file)
    print(f"Loaded {len(records)} iterations")

    if not records:
        print("No records found!")
        sys.exit(1)

    # Analyze
    stream_stats = analyze_streams(records, args.skip_warmup)
    overlap_data = analyze_overlap(records, args.skip_warmup)

    # Print results
    if args.show_streams or not any([args.show_overlap, args.verify_streams]):
        print_stream_summary(stream_stats)

    if args.show_overlap or not any([args.show_streams, args.verify_streams]):
        print_overlap_summary(overlap_data)

    if args.verify_streams or not any([args.show_streams, args.show_overlap]):
        verify_multi_stream_execution(stream_stats)

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
