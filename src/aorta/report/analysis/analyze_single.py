"""
Single configuration analysis - analyze traces from one experiment.

Generates individual per-rank performance reports and multi-rank collective reports
using TraceLens with GEMM patches for ROCm Tensile kernel recognition.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .tracelens_wrapper import TraceLensWrapper


def geometric_mean(values: np.ndarray) -> float:
    """Calculate geometric mean, handling zeros."""
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return float(np.exp(np.mean(np.log(values))))


def detect_trace_directory(input_dir: Path) -> Tuple[Path, Path]:
    """
    Auto-detect directory structure for traces.

    Args:
        input_dir: Input directory path

    Returns:
        Tuple of (torch_profiler_dir, base_dir)

    Raises:
        ValueError: If directory structure cannot be determined
    """
    # Check if input_dir contains rank directories (i.e., it IS torch_profiler/)
    # Use is_dir() to filter out files matching rank* pattern (e.g., rank0.log)
    rank_dirs = [p for p in input_dir.glob("rank*") if p.is_dir()]
    if rank_dirs:
        return input_dir, input_dir.parent

    # Check if input_dir contains torch_profiler/ subdirectory
    torch_prof_dir = input_dir / "torch_profiler"
    if torch_prof_dir.exists():
        rank_dirs = [p for p in torch_prof_dir.glob("rank*") if p.is_dir()]
        if rank_dirs:
            return torch_prof_dir, input_dir

    raise ValueError(
        f"Cannot find rank directories in expected structure.\n"
        f"Expected one of:\n"
        f"  1. Directory with rank0/, rank1/, ... subdirectories (torch_profiler/)\n"
        f"  2. Parent directory containing torch_profiler/rank0/, rank1/, ...\n"
        f"Provided: {input_dir}"
    )


def find_trace_file(rank_dir: Path) -> Optional[Path]:
    """Find trace file in a rank directory.
    
    Searches for JSON trace files in the following order:
    1. Directly in rank_dir (e.g., rank0/*.json)
    2. In trace/ subdirectory (e.g., rank0/trace/pt.trace.json)
    3. Recursively in any subdirectory (e.g., rank0/**/*.json)
    """
    # First, look directly in the rank directory
    json_files = list(rank_dir.glob("*.json"))
    if json_files:
        return json_files[0]
    
    # Then check trace/ subdirectory (common after collective report prep)
    trace_subdir = rank_dir / "trace"
    if trace_subdir.exists():
        json_files = list(trace_subdir.glob("*.json"))
        if json_files:
            return json_files[0]
    
    # Finally, search recursively
    json_files = list(rank_dir.glob("**/*.json"))
    if json_files:
        return json_files[0]
    
    return None


def process_gpu_timeline(
    reports_dir: Path,
    use_geo_mean: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Create mean/geometric mean aggregated GPU timeline across all ranks.

    Args:
        reports_dir: Path to individual_reports directory
        use_geo_mean: If True, use geometric mean; otherwise use arithmetic mean
        verbose: Whether to print verbose output

    Returns:
        Path to output Excel file or None if no data processed
    """
    if not reports_dir.exists():
        raise FileNotFoundError(f"Directory not found: {reports_dir}")

    print(f"Processing GPU timeline from: {reports_dir}")
    print(f"Aggregation: {'Geometric Mean' if use_geo_mean else 'Arithmetic Mean'}")

    perf_files = sorted(reports_dir.glob("perf_rank*.xlsx"))

    if not perf_files:
        print("Error: No perf_rank*.xlsx files found")
        return None

    print(f"Found {len(perf_files)} rank files")

    rank_data = []
    for file_path in perf_files:
        rank_num = int(file_path.stem.replace("perf_rank", ""))
        try:
            df = pd.read_excel(file_path, sheet_name="gpu_timeline")
            df["rank"] = rank_num
            rank_data.append(df)
            if verbose:
                print(f"  Rank {rank_num}: OK")
        except Exception as e:
            print(f"  Rank {rank_num}: Error - {e}")

    if not rank_data:
        print("Error: No valid data loaded")
        return None

    combined = pd.concat(rank_data, ignore_index=True)

    agg_func = geometric_mean if use_geo_mean else "mean"
    aggregated = (
        combined.groupby("type")
        .agg({"time ms": agg_func, "percent": agg_func})
        .reset_index()
    )

    aggregated["num_ranks"] = len(perf_files)

    method_suffix = "geomean" if use_geo_mean else "mean"
    output_path = reports_dir.parent / f"gpu_timeline_summary_{method_suffix}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        aggregated.to_excel(writer, sheet_name="Summary", index=False)

        combined_sorted = combined.sort_values(["rank", "type"])
        combined_sorted.to_excel(writer, sheet_name="All_Ranks_Combined", index=False)

        per_rank = combined.pivot_table(
            values="time ms", index="type", columns="rank", aggfunc="first"
        )
        per_rank.to_excel(writer, sheet_name="Per_Rank_Time_ms")

        per_rank_pct = combined.pivot_table(
            values="percent", index="type", columns="rank", aggfunc="first"
        )
        per_rank_pct.to_excel(writer, sheet_name="Per_Rank_Percent")

    print(f"\nSaved: {output_path}")
    print("\nSummary:")
    print(aggregated.to_string(index=False))

    return output_path


def analyze_single_config(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    run_individual: bool = True,
    run_collective: bool = True,
    aggregate_timeline: bool = True,
    use_geo_mean: bool = False,
    short_kernel_threshold_us: int = 50,
    topk_ops: int = 100,
    verbose: bool = False,
    output_prefix: Optional[str] = None,
) -> dict:
    """
    Run TraceLens analysis on a single configuration trace directory.

    Args:
        input_dir: Path to trace directory (torch_profiler/ or its parent)
        output_dir: Output directory (default: input_dir/tracelens_analysis)
        run_individual: Generate individual per-rank reports
        run_collective: Generate multi-rank collective report
        aggregate_timeline: Aggregate GPU timeline across ranks
        use_geo_mean: Use geometric mean for aggregation
        short_kernel_threshold_us: Threshold for short kernel study
        topk_ops: Number of top operations to include
        verbose: Whether to print verbose output
        output_prefix: Custom prefix for output files (e.g., "28ch" -> perf_28ch_rank0.xlsx)

    Returns:
        Dictionary with paths to generated reports
    """
    input_path = Path(input_dir)

    # Detect directory structure
    torch_prof_dir, base_dir = detect_trace_directory(input_path)

    # Set output directory
    if output_dir is None:
        output_path = base_dir / "tracelens_analysis"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    individual_reports_dir = output_path / "individual_reports"
    collective_reports_dir = output_path / "collective_reports"

    if run_individual:
        individual_reports_dir.mkdir(parents=True, exist_ok=True)
    if run_collective:
        collective_reports_dir.mkdir(parents=True, exist_ok=True)

    # Detect ranks
    rank_dirs = sorted(torch_prof_dir.glob("rank*"))
    num_ranks = len(rank_dirs)

    if num_ranks == 0:
        raise ValueError(f"No rank directories found in {torch_prof_dir}")

    print("=" * 80)
    print("TraceLens Analysis - Single Configuration")
    print("=" * 80)
    print(f"\nInput directory: {input_path}")
    print(f"Torch profiler traces: {torch_prof_dir}")
    print(f"Detected {num_ranks} ranks")
    print(f"Output directory: {output_path}")

    results = {
        "output_dir": output_path,
        "individual_reports": [],
        "collective_report": None,
        "gpu_timeline_summary": None,
    }

    # Initialize TraceLens wrapper
    wrapper = TraceLensWrapper(verbose=verbose)

    # Step 1: Generate individual reports
    if run_individual:
        print("\n" + "=" * 80)
        print("Step 1: Generating Individual Performance Reports")
        print("=" * 80)

        for rank_dir in rank_dirs:
            rank_name = rank_dir.name
            # Extract rank number
            if rank_name.startswith("rank"):
                rank_num = rank_name[4:]  # Remove "rank" prefix
                try:
                    rank_num = int(rank_num.lstrip("_").lstrip("0") or "0")
                except ValueError:
                    rank_num = rank_name

            trace_file = find_trace_file(rank_dir)
            if trace_file is None:
                print(f"  Skip {rank_name} - no trace file found")
                continue

            # Use custom prefix if provided (for sweep mode), otherwise default naming
            if output_prefix:
                output_file = individual_reports_dir / f"perf_{output_prefix}_rank{rank_num}.xlsx"
            else:
                output_file = individual_reports_dir / f"perf_rank{rank_num}.xlsx"

            print(f"\nProcessing {rank_name}...")
            print(f"  Trace: {trace_file.name}")

            try:
                wrapper.generate_perf_report(
                    trace_path=trace_file,
                    output_path=output_file,
                    include_unlinked_kernels=True,
                    short_kernel_study=True,
                    short_kernel_threshold_us=short_kernel_threshold_us,
                    topk_ops=topk_ops,
                    topk_roofline_ops=topk_ops,
                )
                print(f"  Done: {output_file.name}")
                results["individual_reports"].append(output_file)
            except Exception as e:
                print(f"  Error processing {rank_name}: {e}")

    # Step 2: Generate collective report
    if run_collective:
        print("\n" + "=" * 80)
        print("Step 2: Generating Multi-Rank Collective Report")
        print("=" * 80)

        output_file = collective_reports_dir / "collective_all_ranks.xlsx"

        # Create trace.json symlinks for consistent pattern
        for rank_dir in rank_dirs:
            trace_file = find_trace_file(rank_dir)
            if trace_file:
                symlink_path = rank_dir / "trace.json"
                if not symlink_path.exists():
                    try:
                        # Use relative path from rank_dir to trace_file
                        # This handles cases where trace is in subdirectory (e.g., trace/pt.trace.json)
                        relative_path = trace_file.relative_to(rank_dir)
                        symlink_path.symlink_to(relative_path)
                    except (OSError, FileExistsError, ValueError):
                        pass  # Symlink already exists or cannot be created

        trace_pattern = str(torch_prof_dir / "rank*" / "trace.json")

        print(f"\nGenerating collective report for {num_ranks} ranks...")
        print(f"  Trace pattern: rank*/trace.json")

        try:
            wrapper.generate_collective_report(
                trace_pattern=trace_pattern,
                world_size=num_ranks,
                output_path=output_file,
                detailed_analysis=True,
                use_multiprocessing=True,
            )
            print(f"  Done: {output_file.name}")
            results["collective_report"] = output_file
        except Exception as e:
            print(f"  Error generating collective report: {e}")

    # Step 3: Aggregate GPU timeline
    if aggregate_timeline and run_individual:
        print("\n" + "=" * 80)
        print("Step 3: Aggregating GPU Timeline")
        print("=" * 80)

        try:
            summary_path = process_gpu_timeline(
                reports_dir=individual_reports_dir,
                use_geo_mean=use_geo_mean,
                verbose=verbose,
            )
            results["gpu_timeline_summary"] = summary_path
        except Exception as e:
            print(f"  Error aggregating GPU timeline: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {output_path}")
    print(f"\nGenerated reports:")
    print(f"  Individual reports: {len(results['individual_reports'])}")
    print(f"  Collective report: {'Yes' if results['collective_report'] else 'No'}")
    print(f"  GPU timeline summary: {'Yes' if results['gpu_timeline_summary'] else 'No'}")

    if results["individual_reports"]:
        print("\n📊 Individual Performance Reports:")
        for report in results["individual_reports"]:
            print(f"  {report.name}")

    if results["collective_report"]:
        print(f"\n📊 Collective Report:")
        print(f"  {results['collective_report'].name}")

    if results["gpu_timeline_summary"]:
        print(f"\n📊 GPU Timeline Summary:")
        print(f"  {results['gpu_timeline_summary'].name}")

    return results

