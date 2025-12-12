#!/bin/bash

# Advanced NCCL channel and thread sweep script with logging and result aggregation

# Default NCCL channel values to test
DEFAULT_CHANNELS=(28 42 56 70 )

# Default RCCL threads per block values to test
DEFAULT_THREADS=(256 512)

# Default training parameters
DEFAULT_NPROC=8

# Parse command line arguments
CHANNELS_TO_RUN=()
THREADS_TO_RUN=()
SKIP_EXISTING=false
AGGREGATE_RESULTS=true
CONFIG_FILE="config/distributed.yaml"
NPROC_PER_NODE=$DEFAULT_NPROC
ENABLE_ROCPROF=false
ROCPROF_STATS=false
ROCPROF_INPUT=""
OUTPUT_DIR_ARG=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --channels CHANNELS     Comma-separated list of channels (default: 38,42,56,70)"
    echo "  -t, --threads THREADS       Comma-separated list of threads per block (default: 256,512)"
    echo "  -o, --output-dir DIR        Base output directory (default: experiments/sweep_<timestamp>)"
    echo "  -f, --config CONFIG         Config file path (default: config/distributed.yaml)"
    echo "  -p, --nproc NPROC           Number of processes per node (default: 8)"
    echo "  -r, --rocprof               Enable rocprofv3 tracing"
    echo "  -m, --stats                 Enable rocprof stats (CU utilization, occupancy)"
    echo "      --rocprof-input FILE    Use rocprofv3 input yaml/json (filtering supported)"
    echo "  -s, --skip-existing         Skip existing output directories"
    echo "  -n, --no-aggregate          No result aggregation at the end"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --channels 28,42,56 --threads 256,512 --skip-existing"
    echo "  $0 --output-dir experiments/my_sweep --channels 28,42"
    echo "  $0 --rocprof --channels 28,42,56                    # Trace all kernels"
    echo "  $0 --rocprof --stats --channels 28                  # Add CU stats"
    echo "  $0 --rocprof --rocprof-input path/to/rocprof.yaml   # Use yaml to filter kernels"
    echo ""
    exit 1
}

# Handle long options
for arg in "$@"; do
    shift
    case "$arg" in
        --channels)       set -- "$@" "-c" ;;
        --threads)        set -- "$@" "-t" ;;
        --output-dir)     set -- "$@" "-o" ;;
        --config)         set -- "$@" "-f" ;;
        --nproc)          set -- "$@" "-p" ;;
        --rocprof)        set -- "$@" "-r" ;;
        --stats)          set -- "$@" "-m" ;;
        --skip-existing)  set -- "$@" "-s" ;;
        --no-aggregate)   set -- "$@" "-n" ;;
        --rocprof-input)  set -- "$@" "--rocprof-input" ;;
        --help)           set -- "$@" "-h" ;;
        *)                set -- "$@" "$arg" ;;
    esac
done

while getopts "c:t:o:f:p:snrmh-:" opt; do
    case $opt in
        c)
            IFS=',' read -ra CHANNELS_TO_RUN <<< "$OPTARG"
            ;;
        t)
            IFS=',' read -ra THREADS_TO_RUN <<< "$OPTARG"
            ;;
        o)
            OUTPUT_DIR_ARG="$OPTARG"
            ;;
        f)
            CONFIG_FILE="$OPTARG"
            ;;
        p)
            NPROC_PER_NODE="$OPTARG"
            ;;
        s)
            SKIP_EXISTING=true
            ;;
        n)
            AGGREGATE_RESULTS=false
            ;;
        r)
            ENABLE_ROCPROF=true
            ;;
        m)
            ROCPROF_STATS=true
            ;;
        h)
            usage
            ;;
        -)
            case "$OPTARG" in
                rocprof-input)
                    ROCPROF_INPUT="${!OPTIND}"; OPTIND=$((OPTIND+1))
                    ;;
                *)
                    echo "Invalid long option: --$OPTARG" >&2
                    usage
                    ;;
            esac
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# Use default channels if none specified
if [ ${#CHANNELS_TO_RUN[@]} -eq 0 ]; then
    CHANNELS_TO_RUN=("${DEFAULT_CHANNELS[@]}")
fi

# Use default threads if none specified
if [ ${#THREADS_TO_RUN[@]} -eq 0 ]; then
    THREADS_TO_RUN=("${DEFAULT_THREADS[@]}")
fi

# Base configuration

BASE_CMD="torchrun --nproc_per_node ${NPROC_PER_NODE} train.py --config ${CONFIG_FILE}"
BASE_OVERRIDES="--override profiling.tensorboard=false"

# Base output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "${OUTPUT_DIR_ARG}" ]; then
    BASE_OUTPUT_DIR="${OUTPUT_DIR_ARG}"
else
    BASE_OUTPUT_DIR="experiments/sweep_${TIMESTAMP}"
fi

# Create base output directory first
mkdir -p "${BASE_OUTPUT_DIR}"

# Log file for this sweep - save it in the output directory
SWEEP_LOG="${BASE_OUTPUT_DIR}/nccl_thread_sweep_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" | tee -a "${SWEEP_LOG}"
}

# Cleanup function for Ctrl+C
cleanup() {
    echo ""
    echo -e "${RED}=== Caught interrupt signal (Ctrl+C) ===${NC}" | tee -a "${SWEEP_LOG}"
    log "Cleaning up all training processes..."

    # Kill all train.py and torchrun processes
    sudo pkill -9 -f "train.py" 2>/dev/null || true
    sudo pkill -9 -f "torchrun" 2>/dev/null || true

    log "[OK] Cleanup complete. Exiting."
    exit 130
}

# Set up trap to catch Ctrl+C (SIGINT) and other termination signals
trap cleanup SIGINT SIGTERM

# Start sweep
echo -e "${GREEN}=== NCCL Channel & Thread Sweep ===${NC}" | tee "${SWEEP_LOG}"
log "Config file: ${CONFIG_FILE}"
log "Processes per node: ${NPROC_PER_NODE}"
log "Testing threads per block: ${THREADS_TO_RUN[*]}"
log "Testing channels: ${CHANNELS_TO_RUN[*]}"
log "Skip existing: ${SKIP_EXISTING}"
log "Aggregate results: ${AGGREGATE_RESULTS}"
log "rocprofv3 enabled: ${ENABLE_ROCPROF}"
if [ "${ENABLE_ROCPROF}" = true ]; then
    log "  Stats enabled: ${ROCPROF_STATS}"
    if [ -n "${ROCPROF_INPUT}" ]; then
        log "  rocprof input: ${ROCPROF_INPUT} (filtering/stats set in yaml)"
    else
            log "  To filter kernels, use: --rocprof-input path/to/rocprof_input.yaml"
    fi
fi
log "Results directory: ${BASE_OUTPUT_DIR}"
echo ""

# Track results
declare -A RUN_STATUS
declare -A RUN_TIMES

# Loop through each RCCL_THREADS_PER_BLOCK value
for THREADS in "${THREADS_TO_RUN[@]}"; do
    echo -e "${GREEN}=== Testing with RCCL_THREADS_PER_BLOCK=${THREADS} ===${NC}" | tee -a "${SWEEP_LOG}"

    # Loop through each NCCL_MAX_NCHANNELS value
    for CHANNELS in "${CHANNELS_TO_RUN[@]}"; do
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${THREADS}thread/nccl_${CHANNELS}channels"
        RUN_KEY="${THREADS}_${CHANNELS}"

        # Check if should skip
        if [ -d "${OUTPUT_DIR}" ] && [ "${SKIP_EXISTING}" = true ]; then
            log "Skipping THREADS=${THREADS}, CHANNELS=${CHANNELS} (directory exists)"
            RUN_STATUS[${RUN_KEY}]="SKIPPED"
            continue
        fi

        echo -e "${YELLOW}========================================${NC}" | tee -a "${SWEEP_LOG}"
        log "Running with RCCL_THREADS_PER_BLOCK=${THREADS}, NCCL_MAX_NCHANNELS=${CHANNELS}"
        log "Output directory: ${OUTPUT_DIR}"
        echo -e "${YELLOW}========================================${NC}" | tee -a "${SWEEP_LOG}"

        # Create output directory if it doesn't exist
        mkdir -p "${OUTPUT_DIR}"

        # Record start time
        START_TIME=$(date +%s)

        # Set environment variables
        export RCCL_THREADS_PER_BLOCK=${THREADS}
        export NCCL_MAX_NCHANNELS=${CHANNELS}
        export HSA_ENABLE_SDMA=0
        export PYTORCH_ROCM_PROFILER_ENABLE_TRACING=1

        # Run with or without rocprofv3
        if [ "${ENABLE_ROCPROF}" = true ]; then
            ROCPROF_DIR="${OUTPUT_DIR}/rocprof_traces"
            mkdir -p "${ROCPROF_DIR}"

            if [ -n "${ROCPROF_INPUT}" ]; then
                log "[INFO] Using rocprofv3 input file: ${ROCPROF_INPUT}"
                log "[INFO] Kernel filtering/stats should be set inside the input file"
                rocprofv3 -i "${ROCPROF_INPUT}" -d "${ROCPROF_DIR}" -- \
                    ${BASE_CMD} ${BASE_OVERRIDES} \
                    --override training.output_dir=${OUTPUT_DIR} \
                    2>&1 | tee "${OUTPUT_DIR}/run_output.log"
            else
                # Current rocprofv3 build does not support --kernel-names; run unfiltered
                # to avoid argument errors. Expect larger traces.
                ROCPROF_ARGS_ARRAY=("--kernel-trace")
                if [ -n "${KERNEL_NAMES}" ]; then
                    log "[WARN] rocprofv3 on this system ignores kernel filter; tracing all kernels"
                fi
                if [ "${ROCPROF_STATS}" = true ]; then
                    ROCPROF_ARGS_ARRAY+=("--stats")
                fi

                rocprofv3 "${ROCPROF_ARGS_ARRAY[@]}" -d "${ROCPROF_DIR}" -- \
                    ${BASE_CMD} ${BASE_OVERRIDES} \
                    --override training.output_dir=${OUTPUT_DIR} \
                    2>&1 | tee "${OUTPUT_DIR}/run_output.log"
            fi
        else
            ${BASE_CMD} ${BASE_OVERRIDES} \
                --override training.output_dir=${OUTPUT_DIR} \
                2>&1 | tee "${OUTPUT_DIR}/run_output.log"
        fi

        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        RUN_TIMES[${RUN_KEY}]=${DURATION}

        if [ $EXIT_CODE -eq 0 ]; then
            log "[OK] Completed run with THREADS=${THREADS}, CHANNELS=${CHANNELS} (duration: ${DURATION}s)"
            RUN_STATUS[${RUN_KEY}]="SUCCESS"
        else
            log "[ERROR] Failed run with THREADS=${THREADS}, CHANNELS=${CHANNELS} (exit code: $EXIT_CODE, duration: ${DURATION}s)"
            RUN_STATUS[${RUN_KEY}]="FAILED"
        fi

        echo ""

        # Wait between runs
        log "Waiting 5 seconds before next run..."
        sleep 5
    done

    echo ""
done

# Generate summary report
if [ "${AGGREGATE_RESULTS}" = true ]; then
    echo -e "${BLUE}========================================${NC}" | tee -a "${SWEEP_LOG}"
    echo -e "${BLUE}SUMMARY REPORT${NC}" | tee -a "${SWEEP_LOG}"
    echo -e "${BLUE}========================================${NC}" | tee -a "${SWEEP_LOG}"

    # Create summary file in the output directory
    SUMMARY_FILE="${BASE_OUTPUT_DIR}/nccl_thread_sweep_summary_${TIMESTAMP}.txt"
    {
        echo "NCCL Channel & Thread Sweep Summary"
        echo "Generated: $(date)"
        echo ""
        printf "%-10s %-15s %-10s %-15s\n" "THREADS" "CHANNELS" "STATUS" "DURATION(s)"
        echo "----------------------------------------------------"

        for THREADS in "${THREADS_TO_RUN[@]}"; do
            for CHANNELS in "${CHANNELS_TO_RUN[@]}"; do
                RUN_KEY="${THREADS}_${CHANNELS}"
                STATUS="${RUN_STATUS[${RUN_KEY}]:-UNKNOWN}"
                DURATION="${RUN_TIMES[${RUN_KEY}]:-N/A}"
                printf "%-10s %-15s %-10s %-15s\n" "${THREADS}" "${CHANNELS}" "${STATUS}" "${DURATION}"
            done
            echo ""
        done

        echo ""
        echo "Output directories:"
        for THREADS in "${THREADS_TO_RUN[@]}"; do
            echo "  THREADS=${THREADS}:"
            for CHANNELS in "${CHANNELS_TO_RUN[@]}"; do
                echo "    - ${BASE_OUTPUT_DIR}/${THREADS}thread/nccl_${CHANNELS}channels/"
            done
        done
    } | tee "${SUMMARY_FILE}"

    log "Summary saved to: ${SUMMARY_FILE}"

    # Quick comparison script
    echo ""
    echo -e "${GREEN}To compare profiler traces across runs:${NC}"
    echo "# For each thread configuration:"
    for THREADS in "${THREADS_TO_RUN[@]}"; do
        echo "# THREADS=${THREADS}:"
        echo "python scripts/merge_gpu_traces.py \\"
        for CHANNELS in "${CHANNELS_TO_RUN[@]}"; do
            echo "  ${BASE_OUTPUT_DIR}/${THREADS}thread/nccl_${CHANNELS}channels/torch_profiler/rank0/trace_step*.json \\"
        done | head -n -1
        echo ""
    done
fi

log "All runs completed!"
