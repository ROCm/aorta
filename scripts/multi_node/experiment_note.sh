#!/bin/bash
# Quick script to add notes to the most recent experiment

usage() {
    echo "Usage: $0 [EXPERIMENT_DIR] \"Your notes here\""
    echo ""
    echo "If EXPERIMENT_DIR is not provided, uses the most recent experiment."
    echo ""
    echo "Examples:"
    echo "  $0 \"Commented out aux wait_stream line 602-603\""
    echo "  $0 experiments/multinode_28ch_256th_20251219_140715_test \"Changed optimizer\""
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -eq 0 ]]; then
    usage
fi

if [[ $# -eq 1 ]]; then
    NOTE="$1"
    EXPERIMENT_DIR=$(ls -td "$AORTA_ROOT"/experiments/multinode_* 2>/dev/null | head -1)
    if [[ -z "$EXPERIMENT_DIR" ]]; then
        echo "Error: No experiments found"
        exit 1
    fi
else
    EXPERIMENT_DIR="$1"
    NOTE="$2"
    if [[ ! "$EXPERIMENT_DIR" =~ ^/ ]]; then
        EXPERIMENT_DIR="$AORTA_ROOT/$EXPERIMENT_DIR"
    fi
fi

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

INFO_FILE="$EXPERIMENT_DIR/experiment_info.txt"
if [[ ! -f "$INFO_FILE" ]]; then
    echo "Error: experiment_info.txt not found in $EXPERIMENT_DIR"
    exit 1
fi

echo "" >> "$INFO_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NOTE" >> "$INFO_FILE"

echo "Note added to: $EXPERIMENT_DIR"
echo "Note: $NOTE"

