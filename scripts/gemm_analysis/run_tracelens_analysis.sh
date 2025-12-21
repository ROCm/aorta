#!/bin/bash
# TraceLens Analysis for Sweep Directory
# Usage: ./run_tracelens_analysis.sh <sweep_directory> [--rocprof]
# Example: ./run_tracelens_analysis.sh ~/aorta/experiments/sweep_20251120_212921
# Example (rocprof): ./run_tracelens_analysis.sh ~/aorta/experiments/sweep_20251217_103450 --rocprof

set -e

# Parse command-line arguments
USE_ROCPROF=false
SWEEP_DIR=""

for arg in "$@"; do
    case $arg in
        --rocprof)
            USE_ROCPROF=true
            shift
            ;;
        *)
            if [ -z "$SWEEP_DIR" ]; then
                SWEEP_DIR="$arg"
            fi
            ;;
    esac
done

# Check if directory provided
if [ -z "$SWEEP_DIR" ]; then
    echo "Error: Please provide sweep directory"
    echo ""
    echo "Usage: $0 <sweep_directory> [--rocprof]"
    echo ""
    echo "Examples:"
    echo "  PyTorch profiler (default): $0 ~/aorta/experiments/sweep_20251120_212921"
    echo "  ROCprof profiler:           $0 ~/aorta/experiments/sweep_20251217_103450 --rocprof"
    echo ""
    exit 1
fi

# Verify directory exists
if [ ! -d "$SWEEP_DIR" ]; then
    echo "Error: Directory not found: $SWEEP_DIR"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "           TraceLens Analysis Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Sweep directory: $SWEEP_DIR"
if [ "$USE_ROCPROF" = true ]; then
    echo "Mode: ROCprof profiler (*_results.json)"
else
    echo "Mode: PyTorch profiler (torch_profiler/*.json)"
fi
echo ""

# Check if sweep directory is writable
if [ ! -w "$SWEEP_DIR" ]; then
    echo "Error: No write permission for sweep directory: $SWEEP_DIR"
    echo ""
    echo "Please fix permissions by running:"
    echo "  sudo chown -R $(whoami):$(id -gn) $SWEEP_DIR"
    echo "  sudo chmod -R 775 $SWEEP_DIR"
    echo ""
    exit 1
fi

# Create output directory
OUTPUT_DIR="${SWEEP_DIR}/tracelens_analysis"
if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
    echo "Error: Cannot create output directory: $OUTPUT_DIR"
    echo ""
    echo "Please fix permissions by running:"
    echo "  sudo chown -R $(whoami):$(id -gn) $SWEEP_DIR"
    echo "  sudo chmod -R 775 $SWEEP_DIR"
    echo ""
    exit 1
fi

# Auto-discover configurations
echo "Discovering configurations..."
THREAD_CONFIGS=($(find "$SWEEP_DIR" -maxdepth 1 -type d -name "*thread" -exec basename {} \; | sort))
echo "Thread configs: ${THREAD_CONFIGS[@]}"

# Find channels for each thread config
declare -A CHANNELS
for thread in "${THREAD_CONFIGS[@]}"; do
    channels=$(find "$SWEEP_DIR/$thread" -maxdepth 1 -type d -name "nccl_*channels" -exec basename {} \; | sed 's/nccl_\|channels//g' | sort -n | tr '\n' ' ')
    CHANNELS[$thread]="$channels"
    echo "  $thread: $channels"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 1: Generating Individual Reports (All Ranks)"
echo "════════════════════════════════════════════════════════════════"
echo "NOTE: Model parallelism - analyzing all ranks separately"
echo ""

# Show sample trace files for debugging
if [ "$USE_ROCPROF" = true ]; then
    SAMPLE_DIR="${SWEEP_DIR}/${THREAD_CONFIGS[0]}/nccl_${CHANNELS[${THREAD_CONFIGS[0]}]%% *}channels/rocprof_traces/pass_1"
else
    SAMPLE_DIR="${SWEEP_DIR}/${THREAD_CONFIGS[0]}/nccl_${CHANNELS[${THREAD_CONFIGS[0]}]%% *}channels/torch_profiler"
fi
if [ -d "$SAMPLE_DIR" ]; then
    echo "Sample trace files found in first config:"
    if [ "$USE_ROCPROF" = true ]; then
        find "$SAMPLE_DIR" -name "*_results.json" 2>/dev/null | head -5 | sed 's|^|  |'
    else
        find "$SAMPLE_DIR" -name "*.json" 2>/dev/null | head -5 | sed 's|^|  |'
    fi
    echo ""
fi

for thread in "${THREAD_CONFIGS[@]}"; do
    if ! mkdir -p "$OUTPUT_DIR/$thread/individual_reports" 2>/dev/null; then
        echo "Error: Cannot create directory: $OUTPUT_DIR/$thread/individual_reports"
        echo ""
        echo "Please fix permissions by running:"
        echo "  sudo chown -R $(whoami):$(id -gn) $OUTPUT_DIR"
        echo "  sudo chmod -R 775 $OUTPUT_DIR"
        echo ""
        exit 1
    fi

    for ch in ${CHANNELS[$thread]}; do
        if [ "$USE_ROCPROF" = true ]; then
            TRACE_DIR="$SWEEP_DIR/$thread/nccl_${ch}channels/rocprof_traces/pass_1"
        else
            TRACE_DIR="$SWEEP_DIR/$thread/nccl_${ch}channels/torch_profiler"
        fi

        if [ ! -d "$TRACE_DIR" ]; then
            echo "[WARN] Skip $thread/${ch}ch - no traces"
            continue
        fi

        echo "Processing $thread/${ch}ch..."

        if [ "$USE_ROCPROF" = true ]; then
            # ROCprof mode: Find all *_results.json files and sort by PID to map to ranks
            ROCPROF_FILES=($(find "$TRACE_DIR" -name "*_results.json" -type f | sort))

            if [ ${#ROCPROF_FILES[@]} -eq 0 ]; then
                echo "  [WARN] No *_results.json files found"
                continue
            fi

            echo "  Found ${#ROCPROF_FILES[@]} rocprof result files"

            # Process each file, mapping to ranks 0-7
            for rank_idx in "${!ROCPROF_FILES[@]}"; do
                TRACE="${ROCPROF_FILES[$rank_idx]}"
                rank=$rank_idx

                if [ $rank -ge 8 ]; then
                    echo "  [WARN] Skip file ${rank} - only processing ranks 0-7"
                    break
                fi

                OUTPUT="$OUTPUT_DIR/$thread/individual_reports/perf_${ch}ch_rank${rank}.xlsx"

                echo "  Rank ${rank}... ($(basename "$TRACE"))"
                TraceLens_generate_perf_report_rocprof \
                    --profile_json_path "$TRACE" \
                    --output_xlsx_path "$OUTPUT" \
                    --kernel_details \
                    --short_kernel_study \
                    --short_kernel_threshold_us 50 \
                    --topk_kernels 100

                echo "    [OK] $OUTPUT"
            done
        else
            # PyTorch mode: Process ALL ranks (model parallelism = different compute per rank)
            for rank in 0 1 2 3 4 5 6 7; do
                # Try multiple trace file patterns
                TRACE=$(find "$TRACE_DIR" -type f \( \
                    -path "*/rank${rank}/*trace*.json" -o \
                    -path "*/rank_${rank}/*trace*.json" -o \
                    -path "*/rank${rank}*.json" -o \
                    -path "*/*_rank${rank}_*.json" -o \
                    -path "*/customer_trace*.json" \
                    \) | grep -E "rank${rank}|rank_${rank}" | head -1)

                # If still not found, try looking in rank subdirectory with any json
                if [ -z "$TRACE" ]; then
                    TRACE=$(find "$TRACE_DIR/rank${rank}" -name "*.json" 2>/dev/null | head -1)
                fi

                # Last resort: try rank_0X format
                if [ -z "$TRACE" ]; then
                    RANK_PADDED=$(printf "%02d" $rank)
                    TRACE=$(find "$TRACE_DIR" -path "*/rank_${RANK_PADDED}/*trace*.json" 2>/dev/null | head -1)
                fi

                if [ -z "$TRACE" ]; then
                    echo "  [WARN] Skip rank ${rank} - no trace file"
                    continue
                fi

                OUTPUT="$OUTPUT_DIR/$thread/individual_reports/perf_${ch}ch_rank${rank}.xlsx"

                echo "  Rank ${rank}..."
                TraceLens_generate_perf_report_pytorch \
                    --profile_json_path "$TRACE" \
                    --output_xlsx_path "$OUTPUT" \
                    --include_unlinked_kernels \
                    --short_kernel_study \
                    --short_kernel_threshold_us 50 \
                    --topk_ops 100 \
                    --enable_kernel_summary \
                    --topk_roofline_ops 100

                echo "    [OK] $OUTPUT"
            done
        fi
        echo ""
    done
    echo ""
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Generating Collective Reports (All Ranks)"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ "$USE_ROCPROF" = true ]; then
    echo "NOTE: Collective reports are not supported for ROCprof mode."
    echo "      Only individual per-rank reports are generated for ROCprof data."
    echo "      Skipping Step 2..."
    echo ""
else
    for thread in "${THREAD_CONFIGS[@]}"; do
        if ! mkdir -p "$OUTPUT_DIR/$thread/collective_reports" 2>/dev/null; then
            echo "Error: Cannot create directory: $OUTPUT_DIR/$thread/collective_reports"
            echo ""
            echo "Please fix permissions by running:"
            echo "  sudo chown -R $(whoami):$(id -gn) $OUTPUT_DIR"
            echo "  sudo chmod -R 775 $OUTPUT_DIR"
            echo ""
            exit 1
        fi

        for ch in ${CHANNELS[$thread]}; do
            TRACE_DIR="$SWEEP_DIR/$thread/nccl_${ch}channels/torch_profiler"

            if [ ! -d "$TRACE_DIR" ]; then
                echo "[WARN] Skip $thread/${ch}ch"
                continue
            fi

            OUTPUT="$OUTPUT_DIR/$thread/collective_reports/collective_${ch}ch.xlsx"

            echo "Processing $thread/${ch}ch (all 8 ranks)..."

            # Use trace_pattern instead of trace_dir for better subdirectory support
            # It is not guaranteed that trace files will have the exact same name in all the ranks.
            # To avoid file not found errors with `--trace_pattern` flag in TraceLens, we first
            # create a directory called `trace` in all rank folders and then mv the respective
            # trace file in the rank folder to the canonical `trace/pt.trace.json` path.
            # This will satisfy TraceLens's requirement of only one `*` being present in the trace pattern
            # while also avoiding FileNotFoundErrors due to different filenames.
            find $TRACE_DIR/rank* -name "*.json" -exec sh -c 'mkdir -p "$(dirname "$0")/trace" && mv "$0" "$(dirname "$0")/trace/pt.trace.json"' {} \;

            TraceLens_generate_multi_rank_collective_report_pytorch \
                --trace_pattern "$TRACE_DIR/rank*/trace/pt.trace.json" \
                --world_size 8 \
                --output_xlsx_path "$OUTPUT" \
                --detailed_analysis \
                --use_multiprocessing

            echo "  [OK] $OUTPUT"
        done
        echo ""
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 3: Comparing Channels Across Thread Configurations"
echo "════════════════════════════════════════════════════════════════"
echo "NOTE: Comparing per-rank across thread configs"
echo ""

if ! mkdir -p "$OUTPUT_DIR/comparisons" 2>/dev/null; then
    echo "Error: Cannot create directory: $OUTPUT_DIR/comparisons"
    echo ""
    echo "Please fix permissions by running:"
    echo "  sudo chown -R $(whoami):$(id -gn) $OUTPUT_DIR"
    echo "  sudo chmod -R 775 $OUTPUT_DIR"
    echo ""
    exit 1
fi

# Get all unique channel numbers across all thread configs
ALL_CHANNELS=($(for thread in "${THREAD_CONFIGS[@]}"; do echo ${CHANNELS[$thread]}; done | tr ' ' '\n' | sort -nu))

echo "Comparing channels: ${ALL_CHANNELS[@]}"
echo "Comparing ranks: 0-7"
echo ""

# For each channel and each rank, compare across thread configurations
for ch in "${ALL_CHANNELS[@]}"; do
    echo "Channel ${ch}:"

    for rank in 0 1 2 3 4 5 6 7; do
        reports=()
        names=()

        # Collect reports for this channel+rank from all thread configs
        for thread in "${THREAD_CONFIGS[@]}"; do
            REPORT="$OUTPUT_DIR/$thread/individual_reports/perf_${ch}ch_rank${rank}.xlsx"
            if [ -f "$REPORT" ]; then
                reports+=("$REPORT")
                names+=("$thread")
            fi
        done

        # Need at least 2 reports to compare
        if [ ${#reports[@]} -lt 2 ]; then
            echo "  [WARN] Skip rank ${rank} - only in ${#reports[@]} thread config(s)"
            continue
        fi

        OUTPUT="$OUTPUT_DIR/comparisons/compare_${ch}ch_rank${rank}_across_threads.xlsx"

        echo "  Rank ${rank}: comparing ${names[@]}..."

        # Use different sheets based on profiler mode
        if [ "$USE_ROCPROF" = true ]; then
            # ROCprof reports have kernel_summary instead of ops_summary
            TraceLens_compare_perf_reports_pytorch \
                "${reports[@]}" \
                --names "${names[@]}" \
                --sheets gpu_timeline kernel_summary \
                -o "$OUTPUT"
        else
            TraceLens_compare_perf_reports_pytorch \
                "${reports[@]}" \
                --names "${names[@]}" \
                --sheets gpu_timeline ops_summary \
                -o "$OUTPUT"
        fi

        echo "    [OK] $OUTPUT"
    done
    echo ""
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Analysis Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
if [ "$USE_ROCPROF" = true ]; then
    echo "Mode: ROCprof"
else
    echo "Mode: PyTorch Profiler"
fi
echo "Results location: $OUTPUT_DIR/"
echo ""
echo "Generated reports:"

for thread in "${THREAD_CONFIGS[@]}"; do
    indiv=$(find "$OUTPUT_DIR/$thread/individual_reports" -name "*.xlsx" 2>/dev/null | wc -l)
    channels_count=$(echo ${CHANNELS[$thread]} | wc -w)

    if [ "$USE_ROCPROF" = true ]; then
        echo "  $thread: $indiv individual reports"
    else
        coll=$(find "$OUTPUT_DIR/$thread/collective_reports" -name "*.xlsx" 2>/dev/null | wc -l)
        echo "  $thread: $indiv individual (${channels_count} channels × 8 ranks), $coll collective"
    fi
done

comp=$(find "$OUTPUT_DIR/comparisons" -name "*.xlsx" 2>/dev/null | wc -l)
echo "  Comparisons: $comp (per rank across thread configs)"

echo ""
echo "Report Structure (Model Parallelism):"
echo ""
echo "Individual reports (per thread/channel/rank):"
echo "  Format: perf_<channels>ch_rank<0-7>.xlsx"
for thread in "${THREAD_CONFIGS[@]}"; do
    count=$(find "$OUTPUT_DIR/$thread/individual_reports" -name "*.xlsx" 2>/dev/null | wc -l)
    echo "    $thread: $count reports"
done

if [ "$USE_ROCPROF" = false ]; then
    echo ""
    echo "Collective reports (all ranks together):"
    echo "  Format: collective_<channels>ch.xlsx"
    for thread in "${THREAD_CONFIGS[@]}"; do
        count=$(find "$OUTPUT_DIR/$thread/collective_reports" -name "*.xlsx" 2>/dev/null | wc -l)
        echo "    $thread: $count reports"
    done
fi

echo ""
echo "Comparisons (same rank/channel across thread configs):"
echo "  Format: compare_<channels>ch_rank<0-7>_across_threads.xlsx"
echo "  Total: $comp reports"

echo ""
if [ "$USE_ROCPROF" = true ]; then
    echo "Analysis Tips for ROCprof Mode:"
    echo "  - Each rank has different operations - check individual reports per rank"
    echo "  - ROCprof provides detailed kernel-level performance metrics"
    echo "  - Use kernel_details sheets to analyze GPU resource utilization"
    echo "  - Compare same rank across thread configs to see impact of RCCL settings"
else
    echo "Analysis Tips for Model Parallelism:"
    echo "  - Each rank has different operations - check individual reports per rank"
    echo "  - Look for load imbalance across ranks in collective reports"
    echo "  - Compare same rank across thread configs to see impact of RCCL settings"
fi
echo ""
echo "Done! Open .xlsx files in Excel to explore."
