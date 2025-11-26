#!/bin/bash
# TraceLens Analysis for Single Configuration (No Sweep)
# Usage: ./run_tracelens_single_config.sh <directory_path>
# 
# The script accepts either:
#   - Path to parent directory containing torch_profiler/
#   - Path to torch_profiler/ directory directly
#
# Examples:
#   ./run_tracelens_single_config.sh /path/to/traces
#   ./run_tracelens_single_config.sh /path/to/traces/torch_profiler
#
# Note: Uses GEMM-patched TraceLens wrapper to recognize ROCm Tensile kernels

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use patched TraceLens wrapper for GEMM recognition
TRACELENS_WRAPPER="python $SCRIPT_DIR/../tracelens_with_gemm_patch.py"

# Check if directory provided
if [ -z "$1" ]; then
    echo "Error: Please provide trace directory"
    echo ""
    echo "Usage: $0 <directory_path>"
    echo ""
    echo "Examples:"
    echo "  # Parent directory containing torch_profiler/"
    echo "  $0 /home/oyazdanb/aorta_sonbol/saleel_data/saleelk_hip_runtime_test_traces"
    echo ""
    echo "  # torch_profiler/ directory directly"
    echo "  $0 /home/oyazdanb/aorta_sonbol/saleel_data/saleelk_hip_runtime_test_traces/torch_profiler"
    echo ""
    exit 1
fi

INPUT_DIR="$1"

# Verify directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory not found: $INPUT_DIR"
    exit 1
fi

# Auto-detect structure: is this torch_profiler/ or its parent?
TORCH_PROF_DIR=""
BASE_DIR=""

# Check if INPUT_DIR contains rank directories (i.e., it IS torch_profiler/)
if find "$INPUT_DIR" -maxdepth 1 -type d -name "rank*" | grep -q .; then
    TORCH_PROF_DIR="$INPUT_DIR"
    BASE_DIR=$(dirname "$INPUT_DIR")
    echo "Detected torch_profiler directory: $TORCH_PROF_DIR"
# Check if INPUT_DIR contains torch_profiler/ subdirectory
elif [ -d "$INPUT_DIR/torch_profiler" ]; then
    TORCH_PROF_DIR="$INPUT_DIR/torch_profiler"
    BASE_DIR="$INPUT_DIR"
    echo "Found torch_profiler subdirectory: $TORCH_PROF_DIR"
else
    echo "Error: Cannot find rank directories in expected structure"
    echo ""
    echo "Expected one of:"
    echo "  1. Directory with rank0/, rank1/, ... subdirectories (torch_profiler/)"
    echo "  2. Parent directory containing torch_profiler/rank0/, rank1/, ..."
    echo ""
    echo "Provided: $INPUT_DIR"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "           TraceLens Analysis - Single Configuration"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Input directory: $INPUT_DIR"
echo "Torch profiler traces: $TORCH_PROF_DIR"
echo ""

# Create output directory in the base directory
OUTPUT_DIR="${BASE_DIR}/tracelens_analysis"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/individual_reports"
mkdir -p "$OUTPUT_DIR/collective_reports"

# Detect number of ranks
NUM_RANKS=$(find "$TORCH_PROF_DIR" -maxdepth 1 -type d -name "rank*" | wc -l)

if [ $NUM_RANKS -eq 0 ]; then
    echo "Error: No rank directories found in $TORCH_PROF_DIR"
    exit 1
fi

echo "Detected $NUM_RANKS ranks"

# Show sample trace files
echo ""
echo "Sample trace files:"
for rank_dir in $(find "$TORCH_PROF_DIR" -maxdepth 1 -type d -name "rank*" | sort | head -3); do
    rank_name=$(basename "$rank_dir")
    trace_file=$(find "$rank_dir" -name "*.json" | head -1)
    if [ -n "$trace_file" ]; then
        echo "  $rank_name: $(basename "$trace_file")"
    fi
done
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "Step 1: Generating Individual Performance Reports"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Process each rank
for rank_idx in $(seq 0 $((NUM_RANKS - 1))); do
    # Try multiple directory naming patterns
    RANK_DIR=""
    if [ -d "$TORCH_PROF_DIR/rank${rank_idx}" ]; then
        RANK_DIR="$TORCH_PROF_DIR/rank${rank_idx}"
    elif [ -d "$TORCH_PROF_DIR/rank_${rank_idx}" ]; then
        RANK_DIR="$TORCH_PROF_DIR/rank_${rank_idx}"
    elif [ -d "$TORCH_PROF_DIR/rank_$(printf "%02d" $rank_idx)" ]; then
        RANK_DIR="$TORCH_PROF_DIR/rank_$(printf "%02d" $rank_idx)"
    fi

    if [ -z "$RANK_DIR" ] || [ ! -d "$RANK_DIR" ]; then
        echo "  Skip rank ${rank_idx} - directory not found"
        continue
    fi

    # Find trace file
    TRACE=$(find "$RANK_DIR" -name "*.json" -type f | head -1)

    if [ -z "$TRACE" ]; then
        echo "⚠️  Skip rank ${rank_idx} - no trace file found"
        continue
    fi

    OUTPUT="$OUTPUT_DIR/individual_reports/perf_rank${rank_idx}.xlsx"

    echo "Processing rank ${rank_idx}..."
    echo "  Trace: $(basename "$TRACE")"
    
    $TRACELENS_WRAPPER generate_perf_report \
        --profile_json_path "$TRACE" \
        --output_xlsx_path "$OUTPUT" \
        --include_unlinked_kernels \
        --short_kernel_study \
        --short_kernel_threshold_us 50 \
        --topk_ops 100 \
        --topk_roofline_ops 100

    echo "  Done: $OUTPUT"
    echo ""
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Generating Multi-Rank Collective Report"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Find a sample trace file to get the filename pattern
SAMPLE_TRACE=$(find "$TORCH_PROF_DIR/rank0" -name "*.json" -type f | head -1)
if [ -z "$SAMPLE_TRACE" ]; then
    # Try alternative rank naming
    SAMPLE_TRACE=$(find "$TORCH_PROF_DIR/rank_0" -name "*.json" -type f | head -1)
fi

if [ -z "$SAMPLE_TRACE" ]; then
    # Try rank_00
    SAMPLE_TRACE=$(find "$TORCH_PROF_DIR/rank_00" -name "*.json" -type f | head -1)
fi

if [ -n "$SAMPLE_TRACE" ]; then
    OUTPUT="$OUTPUT_DIR/collective_reports/collective_all_ranks.xlsx"

    echo "Generating collective report for all $NUM_RANKS ranks..."
    echo "  Trace pattern: rank*/*.json"
    
    $TRACELENS_WRAPPER generate_multi_rank_collective \
        --trace_pattern "$TORCH_PROF_DIR/rank*/*.json" \
        --world_size $NUM_RANKS \
        --output_xlsx_path "$OUTPUT" \
        --detailed_analysis \
        --use_multiprocessing

    echo "  Done: $OUTPUT"
else
    echo "⚠️  Could not generate collective report - no trace files found"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Analysis Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Results saved to:"
echo "   $OUTPUT_DIR/"
echo ""

# Count generated reports
INDIV_COUNT=$(find "$OUTPUT_DIR/individual_reports" -name "*.xlsx" 2>/dev/null | wc -l)
COLL_COUNT=$(find "$OUTPUT_DIR/collective_reports" -name "*.xlsx" 2>/dev/null | wc -l)

echo "Generated reports:"
echo "  Individual reports (per rank): $INDIV_COUNT"
echo "  Collective reports (all ranks): $COLL_COUNT"
echo ""

echo "📊 Report Files:"
echo ""
echo "Individual Performance Reports:"
if [ $INDIV_COUNT -gt 0 ]; then
    find "$OUTPUT_DIR/individual_reports" -name "*.xlsx" | sort | sed 's/^/  /'
else
    echo "  (none generated)"
fi
echo ""

echo "Collective Reports:"
if [ $COLL_COUNT -gt 0 ]; then
    find "$OUTPUT_DIR/collective_reports" -name "*.xlsx" | sed 's/^/  /'
else
    echo "  (none generated)"
fi

echo ""
echo "Done!"

