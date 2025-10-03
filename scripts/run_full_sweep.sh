#!/bin/bash
# Full Optuna sweep exploring all tuning knobs for MI350 overlap optimization
#
# This script runs a comprehensive hyperparameter sweep to reproduce and understand
# the compute-communication overlap issues reported in REPORT.md and README.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# Default parameters
NUM_TRIALS=${NUM_TRIALS:-100}
MAX_STEPS=${MAX_STEPS:-80}
NUM_GPUS=${NUM_GPUS:-2}
OUTPUT_DIR=${OUTPUT_DIR:-optuna_sweeps/full_sweep_$(date +%Y%m%d_%H%M%S)}
STUDY_NAME=${STUDY_NAME:-mi350_full_sweep}
STORAGE=${STORAGE:-sqlite:///optuna_mi350.db}

echo "=========================================="
echo "Optuna Full Hyperparameter Sweep"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Steps per trial: $MAX_STEPS"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "Study: $STUDY_NAME"
echo "Storage: $STORAGE"
echo "=========================================="
echo ""

python scripts/optuna_sweep.py \
  --num-trials "$NUM_TRIALS" \
  --max-steps "$MAX_STEPS" \
  --num-gpus "$NUM_GPUS" \
  --search-space full \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-check-retries 5 \
  --gpu-check-delay 120

echo ""
echo "=========================================="
echo "Sweep complete!"
echo "Results: $OUTPUT_DIR/optimization_results.json"
echo "Best config: $OUTPUT_DIR/best_config.yaml"
echo "=========================================="
