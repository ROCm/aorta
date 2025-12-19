#!/bin/bash
# List experiments with their metadata

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$AORTA_ROOT/experiments"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n NUM    Show last NUM experiments (default: 10)"
    echo "  -a        Show all experiments"
    echo "  -d        Show detailed info for each experiment"
    echo "  -h        Show this help"
    exit 1
}

NUM_EXPERIMENTS=10
SHOW_ALL=false
DETAILED=false

while getopts "n:adh" opt; do
    case $opt in
        n) NUM_EXPERIMENTS=$OPTARG ;;
        a) SHOW_ALL=true ;;
        d) DETAILED=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
    echo "No experiments directory found"
    exit 1
fi

EXPERIMENTS=$(ls -td "$EXPERIMENTS_DIR"/multinode_* 2>/dev/null)

if [[ -z "$EXPERIMENTS" ]]; then
    echo "No experiments found"
    exit 0
fi

if [[ "$SHOW_ALL" == false ]]; then
    EXPERIMENTS=$(echo "$EXPERIMENTS" | head -n "$NUM_EXPERIMENTS")
fi

echo "Recent Experiments:"
echo "===================="
echo ""

for exp in $EXPERIMENTS; do
    exp_name=$(basename "$exp")
    info_file="$exp/experiment_info.txt"
    
    if [[ -f "$info_file" ]]; then
        label=$(grep "^Experiment:" "$info_file" | cut -d: -f2- | xargs)
        timestamp=$(grep "^Timestamp:" "$info_file" | cut -d: -f2- | xargs)
        config=$(grep "^- Config file:" "$info_file" | cut -d: -f2- | xargs)
        git_branch=$(grep "^- Branch:" "$info_file" | cut -d: -f2- | xargs)
        git_commit=$(grep "^- Commit:" "$info_file" | cut -d: -f2- | xargs)
        
        echo "$exp_name"
        echo "  Label: $label"
        echo "  Config: $(basename "$config")"
        echo "  Branch: $git_branch ($git_commit)"
        
        if [[ "$DETAILED" == true ]]; then
            echo ""
            cat "$info_file"
            echo ""
            echo "---"
        fi
    else
        echo "$exp_name"
        echo "  [No metadata file]"
    fi
    echo ""
done

if [[ "$SHOW_ALL" == false ]]; then
    total=$(ls -d "$EXPERIMENTS_DIR"/multinode_* 2>/dev/null | wc -l)
    echo "Showing $NUM_EXPERIMENTS of $total experiments"
    echo "Use -a to show all, -n NUM to show different number, -d for details"
fi

