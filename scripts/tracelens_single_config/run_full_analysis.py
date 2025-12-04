#!/usr/bin/env python3
"""
Master script for complete TraceLens analysis pipeline.
Runs analysis on baseline and test traces, then performs all comparisons.
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path


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


def run_tracelens_analysis(trace_dir, output_name, individual_only=False, collective_only=False):
    """Run TraceLens analysis on a single trace directory."""
    print(f"\nAnalyzing: {trace_dir}")
    
    # Build command
    script_path = Path(__file__).parent / "run_tracelens_single_config.sh"
    cmd = ["bash", str(script_path), trace_dir]
    
    if individual_only:
        cmd.append("--individual-only")
    elif collective_only:
        cmd.append("--collective-only")
    
    return run_command(cmd, f"TraceLens analysis for {output_name}")


def process_gpu_timeline(reports_dir):
    """Process GPU timeline from individual reports."""
    script_path = Path(__file__).parent / "process_gpu_timeline.py"
    cmd = ["python3", str(script_path), "--reports-dir", reports_dir]
    
    return run_command(cmd, "Processing GPU timeline")


def combine_reports(baseline_file, test_file, output_file):
    """Combine baseline and test reports."""
    script_path = Path(__file__).parent / "combine_reports.py"
    cmd = ["python3", str(script_path), 
           "--baseline", baseline_file,
           "--test", test_file,
           "--output", output_file]
    
    return run_command(cmd, f"Combining reports to {output_file}")


def add_comparison_sheets(input_file, output_file):
    """Add comparison sheets for GPU timeline."""
    script_path = Path(__file__).parent / "add_comparison_sheets.py"
    cmd = ["python3", str(script_path),
           "--input", input_file,
           "--output", output_file]
    
    return run_command(cmd, "Adding GPU timeline comparison sheets")


def add_collective_comparison(input_file, output_file):
    """Add comparison sheets for collective operations."""
    script_path = Path(__file__).parent / "add_collective_comparison.py"
    cmd = ["python3", str(script_path),
           "--input", input_file,
           "--output", output_file]
    
    return run_command(cmd, "Adding collective comparison sheets")


def create_final_report(gpu_combined, gpu_comparison, coll_combined, coll_comparison, output_file):
    """Create comprehensive final report with all data."""
    script_path = Path(__file__).parent / "create_final_report.py"
    cmd = ["python3", str(script_path),
           "--gpu-combined", gpu_combined,
           "--gpu-comparison", gpu_comparison,
           "--coll-combined", coll_combined,
           "--coll-comparison", coll_comparison,
           "--output", output_file]
    
    return run_command(cmd, "Creating comprehensive final report")


def main():
    parser = argparse.ArgumentParser(
        description='Complete TraceLens analysis pipeline with comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with everything including final report
  python run_full_analysis.py \\
    --baseline /path/to/baseline/traces \\
    --test /path/to/test/traces \\
    --output /path/to/output \\
    --all
  
  # Only GPU timeline comparison
  python run_full_analysis.py \\
    --baseline /path/to/baseline \\
    --test /path/to/test \\
    --output /path/to/output \\
    --gpu-timeline
  
  # Create final report (skip TraceLens if already done)
  python run_full_analysis.py \\
    --baseline /path/to/baseline \\
    --test /path/to/test \\
    --output /path/to/output \\
    --gpu-timeline --collective --final-report \\
    --skip-tracelens
        """
    )
    
    # Required arguments
    parser.add_argument('--baseline', required=True,
                       help='Path to baseline trace directory')
    parser.add_argument('--test', required=True,
                       help='Path to test trace directory')
    parser.add_argument('--output', required=True,
                       help='Output directory for comparison results')
    
    # Analysis options
    parser.add_argument('--skip-tracelens', action='store_true',
                       help='Skip TraceLens report generation (if already done)')
    parser.add_argument('--individual-only', action='store_true',
                       help='Generate only individual reports')
    parser.add_argument('--collective-only', action='store_true',
                       help='Generate only collective reports')
    
    # Comparison options
    parser.add_argument('--gpu-timeline', action='store_true',
                       help='Perform GPU timeline comparison')
    parser.add_argument('--collective', action='store_true',
                       help='Perform collective/NCCL comparison')
    parser.add_argument('--final-report', action='store_true',
                       help='Create comprehensive final report with tables and hidden raw data')
    parser.add_argument('--all', action='store_true',
                       help='Perform all analyses and comparisons including final report')
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.gpu_timeline = True
        args.collective = True
        args.final_report = True
    
    # Validate inputs
    baseline_path = Path(args.baseline)
    test_path = Path(args.test)
    output_path = Path(args.output)
    
    if not baseline_path.exists():
        print(f"Error: Baseline path not found: {args.baseline}")
        return 1
    
    if not test_path.exists():
        print(f"Error: Test path not found: {args.test}")
        return 1
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("TRACELENS FULL ANALYSIS PIPELINE")
    print("="*80)
    print(f"Baseline: {args.baseline}")
    print(f"Test: {args.test}")
    print(f"Output: {args.output}")
    print(f"Options:")
    print(f"  Skip TraceLens: {args.skip_tracelens}")
    print(f"  GPU timeline: {args.gpu_timeline}")
    print(f"  Collective: {args.collective}")
    print(f"  Final report: {args.final_report}")
    
    # Step 1: Run TraceLens analysis on both directories
    if not args.skip_tracelens:
        print("\n" + "="*80)
        print("STEP 1: Running TraceLens Analysis")
        print("="*80)
        
        if not run_tracelens_analysis(args.baseline, "baseline", 
                                     args.individual_only, args.collective_only):
            return 1
        
        if not run_tracelens_analysis(args.test, "test",
                                     args.individual_only, args.collective_only):
            return 1
    else:
        print("\nSkipping TraceLens report generation (--skip-tracelens flag)")
    
    # Determine analysis directories
    baseline_analysis = baseline_path / "tracelens_analysis"
    test_analysis = test_path / "tracelens_analysis"
    
    if not baseline_analysis.exists():
        print(f"Error: Baseline analysis not found: {baseline_analysis}")
        print("Run without --skip-tracelens flag first")
        return 1
    
    if not test_analysis.exists():
        print(f"Error: Test analysis not found: {test_analysis}")
        print("Run without --skip-tracelens flag first")
        return 1
    
    # Step 2: GPU Timeline Comparison
    if args.gpu_timeline:
        print("\n" + "="*80)
        print("STEP 2: GPU Timeline Comparison")
        print("="*80)
        
        # Process GPU timelines
        baseline_reports = baseline_analysis / "individual_reports"
        test_reports = test_analysis / "individual_reports"
        
        if not baseline_reports.exists() or not test_reports.exists():
            print("Error: Individual reports not found. Run without --individual-only flag")
            return 1
        
        print("\nProcessing baseline GPU timeline...")
        if not process_gpu_timeline(str(baseline_reports)):
            return 1
        
        print("\nProcessing test GPU timeline...")
        if not process_gpu_timeline(str(test_reports)):
            return 1
        
        # Combine GPU timeline summaries
        baseline_gpu = baseline_analysis / "gpu_timeline_summary_mean.xlsx"
        test_gpu = test_analysis / "gpu_timeline_summary_mean.xlsx"
        combined_gpu = output_path / "gpu_timeline_combined.xlsx"
        
        if not combine_reports(str(baseline_gpu), str(test_gpu), str(combined_gpu)):
            return 1
        
        # Add comparison sheets
        gpu_comparison = output_path / "gpu_timeline_comparison.xlsx"
        if not add_comparison_sheets(str(combined_gpu), str(gpu_comparison)):
            return 1
        
        print(f"\nGPU timeline comparison saved to: {gpu_comparison}")
    
    # Step 3: Collective Comparison
    if args.collective:
        print("\n" + "="*80)
        print("STEP 3: Collective/NCCL Comparison")
        print("="*80)
        
        baseline_collective = baseline_analysis / "collective_reports" / "collective_all_ranks.xlsx"
        test_collective = test_analysis / "collective_reports" / "collective_all_ranks.xlsx"
        
        if not baseline_collective.exists() or not test_collective.exists():
            print("Error: Collective reports not found. Run without --collective-only flag")
            return 1
        
        # Combine collective reports
        combined_collective = output_path / "collective_combined.xlsx"
        if not combine_reports(str(baseline_collective), str(test_collective), 
                             str(combined_collective)):
            return 1
        
        # Add collective comparison
        collective_comparison = output_path / "collective_comparison.xlsx"
        if not add_collective_comparison(str(combined_collective), 
                                        str(collective_comparison)):
            return 1
        
        print(f"\nCollective comparison saved to: {collective_comparison}")
    
    # Step 4: Create final comprehensive report
    if args.final_report and args.gpu_timeline and args.collective:
        print("\n" + "="*80)
        print("STEP 4: Creating Final Comprehensive Report")
        print("="*80)
        
        gpu_combined = output_path / "gpu_timeline_combined.xlsx"
        gpu_comparison = output_path / "gpu_timeline_comparison.xlsx"
        collective_combined = output_path / "collective_combined.xlsx"
        collective_comparison = output_path / "collective_comparison.xlsx"
        final_report = output_path / "final_analysis_report.xlsx"
        
        if not create_final_report(str(gpu_combined), str(gpu_comparison),
                                  str(collective_combined), str(collective_comparison),
                                  str(final_report)):
            return 1
        
        print(f"\nFinal comprehensive report saved to: {final_report}")
        print("  - Summary Dashboard as first sheet")
        print("  - All comparison sheets visible")
        print("  - Raw data sheets hidden (can be unhidden in Excel)")
        print("  - All data formatted as Excel tables with filters")
        print("  - Color coding applied (green=better, red=worse)")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    
    files = list(output_path.glob("*.xlsx"))
    if files:
        print("\nGenerated files:")
        for f in sorted(files):
            print(f"  - {f.name}")
    
    print("\nAnalysis pipeline completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
