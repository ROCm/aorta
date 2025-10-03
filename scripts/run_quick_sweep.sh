#!/bin/bash
# Quick Optuna sweep for rapid testing and validation
#
# This script runs a small sweep to quickly test configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# Quick test parameters
NUM_TRIALS=${NUM_TRIALS:-10}
MAX_STEPS=${MAX_STEPS:-20}
NUM_GPUS=${NUM_GPUS:-2}
OUTPUT_DIR=${OUTPUT_DIR:-optuna_sweeps/quick_test_$(date +%Y%m%d_%H%M%S)}

echo "=========================================="
echo "Optuna Quick Test Sweep"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Steps per trial: $MAX_STEPS"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

python scripts/optuna_sweep.py \
  --num-trials "$NUM_TRIALS" \
  --max-steps "$MAX_STEPS" \
  --num-gpus "$NUM_GPUS" \
  --search-space full \
  --output-dir "$OUTPUT_DIR" \
  --gpu-check-retries 2 \
  --gpu-check-delay 30

echo ""
echo "=========================================="
echo "Quick sweep complete!"
echo "Results: $OUTPUT_DIR/optimization_results.json"
echo "Best config: $OUTPUT_DIR/best_config.yaml"
echo "=========================================="
