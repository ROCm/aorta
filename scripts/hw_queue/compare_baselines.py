#!/usr/bin/env python3
"""
Compare baseline and test results for regression detection.

Usage:
    python compare_baselines.py baseline.json test.json [--threshold 0.05]
    python compare_baselines.py results_dir/ --compare-latest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_results(filepath: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compare_single_result(
    baseline: Dict[str, Any],
    test: Dict[str, Any],
    threshold: float
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compare single baseline and test result.

    Returns:
        (has_regression, regressions_list, improvements_list)
    """
    regressions = []
    improvements = []

    # Compare throughput
    baseline_tp = baseline.get("throughput", 0)
    test_tp = test.get("throughput", 0)

    if isinstance(baseline_tp, dict):
        baseline_tp = baseline_tp.get("value", 0)
    if isinstance(test_tp, dict):
        test_tp = test_tp.get("value", 0)

    if baseline_tp > 0:
        change = (test_tp - baseline_tp) / baseline_tp
        if change < -threshold:
            regressions.append({
                "metric": "throughput",
                "baseline": baseline_tp,
                "test": test_tp,
                "change_pct": change * 100
            })
        elif change > threshold:
            improvements.append({
                "metric": "throughput",
                "baseline": baseline_tp,
                "test": test_tp,
                "change_pct": change * 100
            })

    # Compare latencies
    for metric in ["p50", "p95", "p99"]:
        baseline_lat = baseline.get("latency_ms", {}).get(metric,
                       baseline.get("latency", {}).get(f"{metric}_ms", 0))
        test_lat = test.get("latency_ms", {}).get(metric,
                   test.get("latency", {}).get(f"{metric}_ms", 0))

        if baseline_lat > 0:
            change = (test_lat - baseline_lat) / baseline_lat
            if change > threshold:  # Higher latency is regression
                regressions.append({
                    "metric": f"latency_{metric}",
                    "baseline": baseline_lat,
                    "test": test_lat,
                    "change_pct": change * 100
                })
            elif change < -threshold:  # Lower latency is improvement
                improvements.append({
                    "metric": f"latency_{metric}",
                    "baseline": baseline_lat,
                    "test": test_lat,
                    "change_pct": change * 100
                })

    return len(regressions) > 0, regressions, improvements


def compare_sweep_results(
    baseline_data: Dict[str, Any],
    test_data: Dict[str, Any],
    threshold: float
) -> Dict[str, Any]:
    """Compare sweep results (multiple stream counts)."""

    baseline_results = baseline_data.get("results", [baseline_data])
    test_results = test_data.get("results", [test_data])

    all_regressions = []
    all_improvements = []

    for baseline, test in zip(baseline_results, test_results):
        stream_count = baseline.get("stream_count", "unknown")
        has_reg, regs, imps = compare_single_result(baseline, test, threshold)

        for r in regs:
            r["stream_count"] = stream_count
            all_regressions.append(r)

        for i in imps:
            i["stream_count"] = stream_count
            all_improvements.append(i)

    return {
        "has_regressions": len(all_regressions) > 0,
        "regressions": all_regressions,
        "improvements": all_improvements,
        "baseline_file": str(baseline_data.get("_source_file", "unknown")),
        "test_file": str(test_data.get("_source_file", "unknown")),
    }


def find_latest_results(results_dir: Path, workload: str = None) -> List[Path]:
    """Find the latest result files in a directory."""
    pattern = f"{workload}_*.json" if workload else "*.json"
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print comparison results."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if comparison["regressions"]:
        print("\n🔴 REGRESSIONS DETECTED:")
        for reg in comparison["regressions"]:
            sc = reg.get("stream_count", "")
            sc_str = f" (streams={sc})" if sc else ""
            print(f"  - {reg['metric']}{sc_str}: {reg['baseline']:.3f} -> {reg['test']:.3f} ({reg['change_pct']:+.1f}%)")

    if comparison["improvements"]:
        print("\n🟢 Improvements:")
        for imp in comparison["improvements"]:
            sc = imp.get("stream_count", "")
            sc_str = f" (streams={sc})" if sc else ""
            print(f"  + {imp['metric']}{sc_str}: {imp['baseline']:.3f} -> {imp['test']:.3f} ({imp['change_pct']:+.1f}%)")

    if not comparison["regressions"] and not comparison["improvements"]:
        print("\n✓ No significant changes detected")

    print()

    if comparison["has_regressions"]:
        print("RESULT: ❌ REGRESSIONS FOUND")
    else:
        print("RESULT: ✓ No regressions")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and test results for regressions"
    )
    parser.add_argument("baseline", help="Baseline results JSON file or directory")
    parser.add_argument("test", nargs="?", help="Test results JSON file")
    parser.add_argument("--threshold", type=float, default=0.05,
                       help="Regression threshold (default: 0.05 = 5%%)")
    parser.add_argument("--compare-latest", action="store_true",
                       help="Compare two latest files in directory")
    parser.add_argument("--workload", default=None,
                       help="Filter by workload name")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")

    args = parser.parse_args()

    baseline_path = Path(args.baseline)

    if args.compare_latest or baseline_path.is_dir():
        # Find latest files in directory
        if not baseline_path.is_dir():
            print(f"Error: {baseline_path} is not a directory", file=sys.stderr)
            sys.exit(1)

        files = find_latest_results(baseline_path, args.workload)
        if len(files) < 2:
            print("Error: Need at least 2 result files to compare", file=sys.stderr)
            sys.exit(1)

        test_file = files[0]
        baseline_file = files[1]
        print(f"Comparing: {baseline_file.name} (baseline) vs {test_file.name} (test)")
    else:
        if not args.test:
            print("Error: Need both baseline and test files", file=sys.stderr)
            sys.exit(1)

        baseline_file = baseline_path
        test_file = Path(args.test)

    # Load and compare
    baseline_data = load_results(baseline_file)
    baseline_data["_source_file"] = str(baseline_file)

    test_data = load_results(test_file)
    test_data["_source_file"] = str(test_file)

    comparison = compare_sweep_results(baseline_data, test_data, args.threshold)

    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print_comparison(comparison)

    # Exit with error code if regressions found
    sys.exit(1 if comparison["has_regressions"] else 0)


if __name__ == "__main__":
    main()
