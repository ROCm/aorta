#!/bin/bash
# FSDP-focused Optuna sweep for overlap optimization
#
# This script sweeps only FSDP parameters to understand their impact on overlap

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

NUM_TRIALS=${NUM_TRIALS:-50}
MAX_STEPS=${MAX_STEPS:-80}
NUM_GPUS=${NUM_GPUS:-2}
OUTPUT_DIR=${OUTPUT_DIR:-optuna_sweeps/fsdp_sweep_$(date +%Y%m%d_%H%M%S)}
STUDY_NAME=${STUDY_NAME:-mi350_fsdp_sweep}
STORAGE=${STORAGE:-sqlite:///optuna_mi350.db}

echo "=========================================="
echo "Optuna FSDP Parameter Sweep"
echo "=========================================="
echo "Trials: $NUM_TRIALS"
echo "Steps per trial: $MAX_STEPS"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo "Study: $STUDY_NAME"
echo "=========================================="
echo ""

python scripts/optuna_sweep.py \
  --num-trials "$NUM_TRIALS" \
  --max-steps "$MAX_STEPS" \
  --num-gpus "$NUM_GPUS" \
  --search-space fsdp_only \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-check-retries 5 \
  --gpu-check-delay 120

echo ""
echo "=========================================="
echo "FSDP sweep complete!"
echo "Results: $OUTPUT_DIR/optimization_results.json"
echo "Best config: $OUTPUT_DIR/best_config.yaml"
echo "=========================================="
