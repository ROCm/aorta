#!/bin/bash
# Merge GPU traces for all thread and channel configurations
# Usage: ./merge_all_traces.sh <sweep_directory>

set -e

# Check if directory provided
if [ -z "$1" ]; then
    echo "Error: Please provide sweep directory"
    echo ""
    echo "Usage: $0 <sweep_directory>"
    echo ""
    echo "Example:"
    echo "  $0 experiments/sweep_20251124_222204"
    echo "  $0 /home/oyazdanb/aorta/experiments/sweep_20251124_222204"
    echo ""
    exit 1
fi

SWEEP_DIR="$1"

# Verify directory exists
if [ ! -d "$SWEEP_DIR" ]; then
    echo "Error: Directory not found: $SWEEP_DIR"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "           Multi-Rank Trace Merger"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Sweep directory: $SWEEP_DIR"
echo ""

# Create output directory
OUTPUT_DIR="${SWEEP_DIR}/merged_traces"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

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

# Count total configurations
total_configs=0
for thread in "${THREAD_CONFIGS[@]}"; do
    channel_count=$(echo ${CHANNELS[$thread]} | wc -w)
    total_configs=$((total_configs + channel_count))
done

echo "Total configurations to merge: $total_configs"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track results
success_count=0
skip_count=0
error_count=0

echo "════════════════════════════════════════════════════════════════"
echo "Starting Trace Merging"
echo "════════════════════════════════════════════════════════════════"
echo ""

config_num=0

for thread in "${THREAD_CONFIGS[@]}"; do
    # Extract thread number (e.g., "256" from "256thread")
    thread_num=$(echo "$thread" | sed 's/thread//')
    
    for ch in ${CHANNELS[$thread]}; do
        config_num=$((config_num + 1))
        
        TRACE_DIR="$SWEEP_DIR/$thread/nccl_${ch}channels/torch_profiler"
        OUTPUT_FILE="$OUTPUT_DIR/${thread_num}thread_${ch}ch_merged.json"
        
        echo "[$config_num/$total_configs] Processing $thread / ${ch} channels..."
        
        # Check if trace directory exists
        if [ ! -d "$TRACE_DIR" ]; then
            echo -e "  ${YELLOW}⚠️  Skip - trace directory not found${NC}"
            skip_count=$((skip_count + 1))
            echo ""
            continue
        fi
        
        # Check if at least some rank directories exist
        rank_count=$(find "$TRACE_DIR" -maxdepth 1 -type d -name "rank*" 2>/dev/null | wc -l)
        if [ $rank_count -eq 0 ]; then
            echo -e "  ${YELLOW}⚠️  Skip - no rank directories found${NC}"
            skip_count=$((skip_count + 1))
            echo ""
            continue
        fi
        
        echo "  Trace dir: $TRACE_DIR"
        echo "  Output: ${OUTPUT_FILE##*/}"
        echo "  Found $rank_count rank directories"
        
        # Run the merge script
        if python scripts/merge_gpu_trace_ranks.py \
            "$TRACE_DIR" \
            -o "$OUTPUT_FILE" \
            -n 8 \
            --trace-name "customer_trace.json" 2>&1 | grep -E "(Processing rank|Added|Successfully|Total events|Error)" | sed 's/^/  /'; then
            
            # Check if output file was created
            if [ -f "$OUTPUT_FILE" ]; then
                file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
                echo -e "  ${GREEN}✓ Success${NC} (${file_size})"
                success_count=$((success_count + 1))
            else
                echo -e "  ${RED}✗ Failed - output file not created${NC}"
                error_count=$((error_count + 1))
            fi
        else
            echo -e "  ${RED}✗ Failed - merge script error${NC}"
            error_count=$((error_count + 1))
        fi
        
        echo ""
    done
done

echo "════════════════════════════════════════════════════════════════"
echo "Merge Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Total configurations: $total_configs"
echo -e "${GREEN}✓ Successfully merged: $success_count${NC}"
echo -e "${YELLOW}⚠ Skipped: $skip_count${NC}"
echo -e "${RED}✗ Failed: $error_count${NC}"
echo ""

if [ $success_count -gt 0 ]; then
    echo "Merged trace files:"
    ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo "📊 View traces at: https://ui.perfetto.dev"
    echo ""
    echo "Usage:"
    echo "  1. Open https://ui.perfetto.dev in your browser"
    echo "  2. Click 'Open trace file'"
    echo "  3. Select one of the merged JSON files"
    echo "  4. View all 8 ranks on a single timeline"
    echo ""
fi

if [ $error_count -gt 0 ]; then
    echo "⚠️  Some merges failed. Check the error messages above."
    exit 1
fi

echo "✅ All traces merged successfully!"

