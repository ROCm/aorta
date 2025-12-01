#!/bin/bash

# Compare specific RCCL Warp Speed configurations
# Usage: ./run_rccl_warp_speed_comparison.sh [OPTIONS]
#   -c CONFIG_FILE    Config file (default: config/distributed.yaml)
#   -p PAIRS          CU,threads pairs (e.g., "56,256 37,384 32,512")
#   -h                Show help
#
# Examples:
#   # Use default 3 configurations
#   ./run_rccl_warp_speed_comparison.sh
#
#   # Custom configurations
#   ./run_rccl_warp_speed_comparison.sh -p "56,256 37,384 32,512"
#
#   # Different config file with custom pairs
#   ./run_rccl_warp_speed_comparison.sh -c myconfig.yaml -p "40,256 30,512"

CONFIG_FILE="config/distributed.yaml"
CUSTOM_PAIRS=""

# Parse command line arguments
while getopts "c:p:h" opt; do
    case $opt in
        c)
            CONFIG_FILE="$OPTARG"
            ;;
        p)
            CUSTOM_PAIRS="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [OPTIONS]"
            echo "  -c CONFIG_FILE    Config file (default: config/distributed.yaml)"
            echo "  -p PAIRS          CU,threads pairs (e.g., \"56,256 37,384 32,512\")"
            echo "  -h                Show help"
            echo ""
            echo "Examples:"
            echo "  # Use default 3 configurations"
            echo "  $0"
            echo ""
            echo "  # Custom configurations"
            echo "  $0 -p \"56,256 37,384 32,512\""
            echo ""
            echo "  # Different config file with custom pairs"
            echo "  $0 -c myconfig.yaml -p \"40,256 30,512\""
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done
BASE_CMD="torchrun --nproc_per_node 8 train.py --config ${CONFIG_FILE}"
BASE_OVERRIDES="--override training.max_steps=100 --override profiling.tensorboard=false"

# Base output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="experiments/rccl_warp_speed_${TIMESTAMP}"

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Log file
SWEEP_LOG="${BASE_OUTPUT_DIR}/rccl_warp_speed_comparison_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    sudo pkill -9 -f "train.py" 2>/dev/null || true
    sudo pkill -9 -f "torchrun" 2>/dev/null || true
    log "Cleanup complete. Exiting."
    exit 130
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN}=== RCCL Warp Speed Configuration Comparison ===${NC}" | tee "${SWEEP_LOG}"
log "Config file: ${CONFIG_FILE}"
log "Results directory: ${BASE_OUTPUT_DIR}"
echo ""

# Define configurations to test
# Format: "NAME|CU_COUNT|THREADS_PER_BLOCK"
if [ -n "$CUSTOM_PAIRS" ]; then
    # Parse custom pairs
    CONFIGS=()
    for pair in $CUSTOM_PAIRS; do
        IFS=',' read -r cu threads <<< "$pair"
        CONFIGS+=("${cu}cu_${threads}threads|${cu}|${threads}")
    done
    log "Using custom configurations: ${CUSTOM_PAIRS}"
else
    # Use default configurations
    CONFIGS=(
        "56cu_256threads|56|256"
        "37cu_384threads|37|384"
        "32cu_512threads|32|512"
    )
    log "Using default RCCL Warp Speed configurations"
fi

# Track results
declare -A RUN_STATUS
declare -A RUN_TIMES

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME CU_COUNT THREADS <<< "$config"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${NAME}"

    echo -e "${YELLOW}========================================${NC}" | tee -a "${SWEEP_LOG}"
    log "Running configuration: ${NAME}"
    log "  RCCL_WARP_SPEED_CU_COUNT=${CU_COUNT}"
    log "  RCCL_THREADS_PER_BLOCK=${THREADS}"
    log "  Output directory: ${OUTPUT_DIR}"
    echo -e "${YELLOW}========================================${NC}" | tee -a "${SWEEP_LOG}"

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"

    # Record start time
    START_TIME=$(date +%s)

    # Set environment variables and run
    RCCL_WARP_SPEED_ENABLE=1 \
    RCCL_UNROLL_FACTOR=1 \
    RCCL_WARP_SPEED_CU_COUNT=${CU_COUNT} \
    RCCL_THREADS_PER_BLOCK=${THREADS} \
    HSA_ENABLE_SDMA=0 \
    PYTORCH_ROCM_PROFILER_ENABLE_TRACING=1 \
    ${BASE_CMD} ${BASE_OVERRIDES} \
        --override training.output_dir=${OUTPUT_DIR} \
        2>&1 | tee "${OUTPUT_DIR}/run_output.log"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    RUN_TIMES[${NAME}]=${DURATION}

    if [ $EXIT_CODE -eq 0 ]; then
        log "[OK] Completed ${NAME} (duration: ${DURATION}s)"
        RUN_STATUS[${NAME}]="SUCCESS"
    else
        log "[ERROR] Failed ${NAME} (exit code: $EXIT_CODE, duration: ${DURATION}s)"
        RUN_STATUS[${NAME}]="FAILED"
    fi

    echo ""
    log "Waiting 5 seconds before next run..."
    sleep 5
done

# Generate summary report
echo -e "${BLUE}========================================${NC}" | tee -a "${SWEEP_LOG}"
echo -e "${BLUE}SUMMARY REPORT${NC}" | tee -a "${SWEEP_LOG}"
echo -e "${BLUE}========================================${NC}" | tee -a "${SWEEP_LOG}"

SUMMARY_FILE="${BASE_OUTPUT_DIR}/rccl_warp_speed_summary_${TIMESTAMP}.txt"
{
    echo "RCCL Warp Speed Configuration Comparison"
    echo "Generated: $(date)"
    echo ""
    printf "%-20s %-10s %-15s %-10s\n" "CONFIGURATION" "CU_COUNT" "THREADS" "STATUS"
    echo "----------------------------------------------------------------"

    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r NAME CU_COUNT THREADS <<< "$config"
        STATUS="${RUN_STATUS[${NAME}]:-UNKNOWN}"
        DURATION="${RUN_TIMES[${NAME}]:-N/A}"
        printf "%-20s %-10s %-15s %-10s (duration: %ss)\n" "${NAME}" "${CU_COUNT}" "${THREADS}" "${STATUS}" "${DURATION}"
    done

    echo ""
    echo "Output directories:"
    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r NAME CU_COUNT THREADS <<< "$config"
        echo "  ${NAME}: ${BASE_OUTPUT_DIR}/${NAME}/"
    done

    echo ""
    echo "Trace files for each configuration:"
    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r NAME CU_COUNT THREADS <<< "$config"
        echo "  ${NAME}: ${BASE_OUTPUT_DIR}/${NAME}/torch_profiler/"
    done
} | tee "${SUMMARY_FILE}"

log "Summary saved to: ${SUMMARY_FILE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Next Steps: Run TraceLens Analysis${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To analyze and compare these configurations:"
echo ""
echo "./scripts/tracelens_single_config/run_tracelens_analysis.sh ${BASE_OUTPUT_DIR}"
echo ""
echo "This will generate:"
echo "  - Individual reports for each rank (all 3 configs)"
echo "  - Collective reports (all 3 configs)"
echo "  - Comparison reports across the 3 configurations"
echo ""

log "All runs completed! Run TraceLens analysis next."
