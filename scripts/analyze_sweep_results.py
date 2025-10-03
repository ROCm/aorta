#!/usr/bin/env python
"""Analyze Optuna sweep results and generate summary report."""

import argparse
import json
from pathlib import Path
import sys

def load_trial_results(sweep_dir: Path):
    """Load results from all trials in a sweep directory."""
    trials = []

    trial_dirs = sorted(sweep_dir.glob("trial_*"))
    for trial_dir in trial_dirs:
        trial_num = int(trial_dir.name.split("_")[1])

        # Load trial metrics
        metrics_file = trial_dir / "trial_metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Load trial config
        config_file = trial_dir / "config.yaml"
        if not config_file.exists():
            continue

        trials.append({
            "trial": trial_num,
            "overlap_ratio": metrics.get("overlap_ratio", 0.0),
            "avg_compute_ms": metrics.get("avg_compute_ms", 0.0),
            "avg_overlap_ms": metrics.get("avg_overlap_ms", 0.0),
            "num_iterations": metrics.get("num_iterations", 0),
        })

    return trials


def print_summary(trials, top_n=10):
    """Print summary statistics and top trials."""
    if not trials:
        print("No trials found!")
        return

    # Sort by overlap ratio
    sorted_trials = sorted(trials, key=lambda t: t["overlap_ratio"], reverse=True)

    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    print(f"Total trials: {len(trials)}")
    print(f"Best overlap ratio: {sorted_trials[0]['overlap_ratio']:.6f} (trial {sorted_trials[0]['trial']})")
    print(f"Worst overlap ratio: {sorted_trials[-1]['overlap_ratio']:.6f} (trial {sorted_trials[-1]['trial']})")

    # Statistics
    overlap_ratios = [t["overlap_ratio"] for t in trials]
    avg_ratio = sum(overlap_ratios) / len(overlap_ratios)
    print(f"Average overlap ratio: {avg_ratio:.6f}")

    compute_times = [t["avg_compute_ms"] for t in trials]
    avg_compute = sum(compute_times) / len(compute_times)
    print(f"Average compute time: {avg_compute:.2f} ms")

    print(f"\nTop {min(top_n, len(trials))} trials by overlap ratio:")
    print("-" * 80)
    print(f"{'Trial':<8} {'Overlap Ratio':<15} {'Compute (ms)':<15} {'Overlap (ms)':<15}")
    print("-" * 80)

    for trial in sorted_trials[:top_n]:
        print(
            f"{trial['trial']:<8} "
            f"{trial['overlap_ratio']:<15.6f} "
            f"{trial['avg_compute_ms']:<15.2f} "
            f"{trial['avg_overlap_ms']:<15.2f}"
        )

    print(f"\nBottom {min(top_n, len(trials))} trials by overlap ratio:")
    print("-" * 80)
    print(f"{'Trial':<8} {'Overlap Ratio':<15} {'Compute (ms)':<15} {'Overlap (ms)':<15}")
    print("-" * 80)

    for trial in sorted_trials[-top_n:]:
        print(
            f"{trial['trial']:<8} "
            f"{trial['overlap_ratio']:<15.6f} "
            f"{trial['avg_compute_ms']:<15.2f} "
            f"{trial['avg_overlap_ms']:<15.2f}"
        )

    print("="*80 + "\n")


def compare_with_baseline(trials, baseline_ratio=0.007):
    """Compare trials with baseline overlap ratio."""
    better_trials = [t for t in trials if t["overlap_ratio"] > baseline_ratio]
    worse_trials = [t for t in trials if t["overlap_ratio"] < baseline_ratio]
    similar_trials = [t for t in trials if abs(t["overlap_ratio"] - baseline_ratio) <= 0.001]

    print(f"\nComparison with baseline (overlap_ratio={baseline_ratio:.6f}):")
    print(f"  Trials better than baseline: {len(better_trials)} ({len(better_trials)/len(trials)*100:.1f}%)")
    print(f"  Trials worse than baseline: {len(worse_trials)} ({len(worse_trials)/len(trials)*100:.1f}%)")
    print(f"  Trials similar to baseline: {len(similar_trials)} ({len(similar_trials)/len(trials)*100:.1f}%)")

    if better_trials:
        best_improvement = max(better_trials, key=lambda t: t["overlap_ratio"])
        improvement = (best_improvement["overlap_ratio"] - baseline_ratio) / baseline_ratio * 100
        print(f"\n  Best improvement: {improvement:+.1f}% (trial {best_improvement['trial']}, ratio={best_improvement['overlap_ratio']:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna sweep results")
    parser.add_argument("sweep_dir", type=Path, help="Sweep directory containing trial results")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top/bottom trials to show")
    parser.add_argument("--baseline", type=float, default=0.007, help="Baseline overlap ratio for comparison")

    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: Directory not found: {args.sweep_dir}")
        sys.exit(1)

    print(f"Loading results from: {args.sweep_dir}")
    trials = load_trial_results(args.sweep_dir)

    if not trials:
        print("No trial results found!")
        sys.exit(1)

    print_summary(trials, args.top_n)
    compare_with_baseline(trials, args.baseline)

    # Save summary
    summary_file = args.sweep_dir / "analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_trials": len(trials),
            "best_trial": max(trials, key=lambda t: t["overlap_ratio"]),
            "worst_trial": min(trials, key=lambda t: t["overlap_ratio"]),
            "avg_overlap_ratio": sum(t["overlap_ratio"] for t in trials) / len(trials),
            "all_trials": sorted(trials, key=lambda t: t["overlap_ratio"], reverse=True),
        }, f, indent=2)

    print(f"\nDetailed analysis saved to: {summary_file}")


if __name__ == "__main__":
    main()
