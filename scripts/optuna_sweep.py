#!/usr/bin/env python
"""Optuna hyperparameter sweep for compute-communication overlap optimization.

This script sweeps over all tuning knobs mentioned in README.md "Tuning for Overlap"
section to reproduce and understand overlap issues on MI350 GPUs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import yaml

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

log = logging.getLogger(__name__)


class GPUAvailabilityChecker:
    """Check if GPUs are available for training, handling shared GPU scenarios."""

    def __init__(self, num_gpus: int = 2, max_retries: int = 3, retry_delay: int = 60):
        self.num_gpus = num_gpus
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def check_gpu_memory(self) -> bool:
        """Check if GPUs have sufficient free memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            for gpu_id in range(min(self.num_gpus, torch.cuda.device_count())):
                # Check if we can allocate a small tensor
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    # Try to get memory info
                    mem_free = torch.cuda.mem_get_info(gpu_id)[0]
                    mem_total = torch.cuda.mem_get_info(gpu_id)[1]
                    utilization = 1.0 - (mem_free / mem_total)

                    # If GPU is >90% utilized, likely someone else is using it
                    if utilization > 0.9:
                        log.warning(f"GPU {gpu_id} appears busy (utilization: {utilization:.1%})")
                        return False
                except Exception as e:
                    log.warning(f"Error checking GPU {gpu_id}: {e}")
                    return False

            return True
        except ImportError:
            log.warning("PyTorch not available for GPU checking")
            return True  # Assume available if we can't check

    def wait_for_gpus(self) -> bool:
        """Wait for GPUs to become available with retries."""
        for attempt in range(self.max_retries):
            if self.check_gpu_memory():
                return True

            if attempt < self.max_retries - 1:
                log.info(f"GPUs busy, waiting {self.retry_delay}s before retry {attempt + 2}/{self.max_retries}...")
                time.sleep(self.retry_delay)

        log.error(f"GPUs still busy after {self.max_retries} attempts")
        return False


def generate_trial_config(trial: optuna.Trial, base_config: Dict[str, Any],
                         search_space: str = "full") -> Dict[str, Any]:
    """Generate configuration for a single Optuna trial.

    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary
        search_space: One of "full", "fsdp_only", "env_only", "workload_only"

    Returns:
        Configuration dictionary with trial parameters
    """
    config = json.loads(json.dumps(base_config))  # Deep copy

    # FSDP scheduling knobs
    if search_space in ["full", "fsdp_only"]:
        config["fsdp"]["forward_prefetch"] = trial.suggest_categorical(
            "fsdp.forward_prefetch", [True, False]
        )
        config["fsdp"]["limit_all_gathers"] = trial.suggest_categorical(
            "fsdp.limit_all_gathers", [True, False]
        )
        config["fsdp"]["backward_prefetch"] = trial.suggest_categorical(
            "fsdp.backward_prefetch", ["BACKWARD_PRE", "BACKWARD_POST"]
        )
        config["fsdp"]["sync_module_states"] = trial.suggest_categorical(
            "fsdp.sync_module_states", [True, False]
        )
        config["fsdp"]["use_orig_params"] = trial.suggest_categorical(
            "fsdp.use_orig_params", [True, False]
        )

    # Workload intensity knobs
    if search_space in ["full", "workload_only"]:
        config["training"]["batch_size"] = trial.suggest_categorical(
            "training.batch_size", [4, 8, 16, 32]
        )
        config["training"]["gradient_accumulation"] = trial.suggest_categorical(
            "training.gradient_accumulation", [1, 2, 4, 8]
        )
        config["training"]["mixed_precision"] = trial.suggest_categorical(
            "training.mixed_precision", ["bf16", "fp16", "none"]
        )

    # Environment variables for RCCL (stored as trial params, applied at runtime)
    if search_space in ["full", "env_only"]:
        trial.suggest_categorical("RCCL_ENABLE_SDMA", ["0", "1"])
        trial.suggest_categorical("RCCL_NUM_CHANNELS", ["2", "4", "8"])
        trial.suggest_int("RCCL_BUFFER_SIZE", 256*1024, 4*1024*1024, step=256*1024)
        trial.suggest_categorical("RCCL_SDMA_WORKERS_PER_CHANNEL", ["0", "2", "4"])

    return config


def extract_overlap_metrics(output_dir: Path) -> Optional[Dict[str, float]]:
    """Extract overlap metrics from training run artifacts.

    Args:
        output_dir: Directory containing rank_*_metrics.jsonl files

    Returns:
        Dictionary with aggregate metrics or None if failed
    """
    try:
        metrics_files = list(output_dir.glob("rank_*_metrics.jsonl"))
        if not metrics_files:
            log.error(f"No metrics files found in {output_dir}")
            return None

        all_records = []
        for metrics_file in metrics_files:
            with open(metrics_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            all_records.append(record)
                        except json.JSONDecodeError:
                            continue

        if not all_records:
            log.error("No valid records found in metrics files")
            return None

        # Skip warmup iterations (first 5)
        records = all_records[5:] if len(all_records) > 5 else all_records

        # Calculate aggregate metrics
        total_compute = sum(
            r.get("profile", {}).get("overlap", {}).get("per_stream_ms", {}).get("compute", 0.0)
            for r in records
        )
        total_overlap = sum(
            r.get("profile", {}).get("overlap", {}).get("overlap_ms", {}).get("compute_comm", 0.0)
            for r in records
        )

        avg_compute = total_compute / len(records) if records else 0.0
        avg_overlap = total_overlap / len(records) if records else 0.0
        overlap_ratio = avg_overlap / avg_compute if avg_compute > 0 else 0.0

        # Extract iteration time
        avg_iter_time = sum(r.get("iteration_ms", 0.0) for r in records) / len(records) if records else 0.0

        return {
            "avg_compute_ms": avg_compute,
            "avg_overlap_ms": avg_overlap,
            "overlap_ratio": overlap_ratio,
            "avg_iteration_ms": avg_iter_time,
            "num_iterations": len(records),
        }

    except Exception as e:
        log.error(f"Error extracting metrics: {e}")
        return None


def run_training_trial(
    trial: optuna.Trial,
    config: Dict[str, Any],
    num_gpus: int,
    max_steps: int,
    gpu_checker: GPUAvailabilityChecker,
) -> Optional[float]:
    """Run a single training trial and return the objective value.

    Args:
        trial: Optuna trial
        config: Trial configuration
        num_gpus: Number of GPUs to use
        max_steps: Maximum training steps
        gpu_checker: GPU availability checker

    Returns:
        Overlap ratio (higher is better) or None if failed
    """
    # Wait for GPUs to be available
    if not gpu_checker.wait_for_gpus():
        log.error(f"Trial {trial.number}: GPUs not available, pruning trial")
        raise optuna.TrialPruned("GPUs busy")

    # Create trial-specific output directory
    trial_dir = Path(config["training"]["output_dir"]) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save trial config
    config_path = trial_dir / "config.yaml"
    config["training"]["output_dir"] = str(trial_dir)
    config["training"]["max_steps"] = max_steps
    config["profiling"]["enabled"] = False  # Disable torch profiler for speed

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Build environment variables
    env = os.environ.copy()

    # Apply RCCL environment variables from trial params
    if "RCCL_ENABLE_SDMA" in trial.params:
        env["RCCL_ENABLE_SDMA"] = str(trial.params["RCCL_ENABLE_SDMA"])
    if "RCCL_NUM_CHANNELS" in trial.params:
        env["RCCL_NUM_CHANNELS"] = str(trial.params["RCCL_NUM_CHANNELS"])
    if "RCCL_BUFFER_SIZE" in trial.params:
        env["RCCL_BUFFER_SIZE"] = str(trial.params["RCCL_BUFFER_SIZE"])
    if "RCCL_SDMA_WORKERS_PER_CHANNEL" in trial.params:
        env["RCCL_SDMA_WORKERS_PER_CHANNEL"] = str(trial.params["RCCL_SDMA_WORKERS_PER_CHANNEL"])

    # Enable RCCL debug logging to verify stream/channel usage
    env["RCCL_DEBUG"] = "INFO"  # Use "TRACE" for more verbose output
    env["RCCL_DEBUG_SUBSYS"] = "INIT,COLL"  # Show initialization and collective operations

    # Set PYTHONPATH
    env["PYTHONPATH"] = str(SRC_ROOT)

    # Build torchrun command
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node", str(num_gpus),
        str(REPO_ROOT / "train.py"),
        "--config", str(config_path),
    ]

    log.info(f"Trial {trial.number}: Starting training with config: {config_path}")
    log.info(f"Trial {trial.number}: RCCL env: {[(k, v) for k, v in env.items() if k.startswith('RCCL_')]}")

    # Run training
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            log.error(f"Trial {trial.number}: Training failed with return code {result.returncode}")
            log.error(f"STDERR: {result.stderr[-1000:]}")  # Last 1000 chars

            # Check if it's a GPU OOM or busy error
            if "out of memory" in result.stderr.lower() or "cuda error" in result.stderr.lower():
                log.warning(f"Trial {trial.number}: GPU error detected, pruning trial")
                raise optuna.TrialPruned("GPU error")

            return None

        # Extract metrics
        metrics = extract_overlap_metrics(trial_dir)
        if metrics is None:
            log.error(f"Trial {trial.number}: Failed to extract metrics")
            return None

        # Log trial results
        log.info(f"Trial {trial.number}: overlap_ratio={metrics['overlap_ratio']:.4f}, "
                f"compute={metrics['avg_compute_ms']:.2f}ms, "
                f"overlap={metrics['avg_overlap_ms']:.2f}ms")

        # Save metrics
        metrics_path = trial_dir / "trial_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics["overlap_ratio"]

    except subprocess.TimeoutExpired:
        log.error(f"Trial {trial.number}: Training timed out")
        raise optuna.TrialPruned("Timeout")
    except Exception as e:
        log.error(f"Trial {trial.number}: Unexpected error: {e}")
        return None


def objective(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    num_gpus: int,
    max_steps: int,
    search_space: str,
    gpu_checker: GPUAvailabilityChecker,
) -> float:
    """Optuna objective function to maximize overlap ratio."""
    config = generate_trial_config(trial, base_config, search_space)

    overlap_ratio = run_training_trial(trial, config, num_gpus, max_steps, gpu_checker)

    if overlap_ratio is None:
        raise optuna.TrialPruned("Training failed")

    return overlap_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for compute-communication overlap optimization"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "config" / "default.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs per trial",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum training steps per trial",
    )
    parser.add_argument(
        "--search-space",
        choices=["full", "fsdp_only", "env_only", "workload_only"],
        default="full",
        help="Which parameter space to search",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="overlap_optimization",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db). If None, uses in-memory storage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "optuna_sweeps",
        help="Output directory for sweep results",
    )
    parser.add_argument(
        "--gpu-check-retries",
        type=int,
        default=3,
        help="Number of retries when GPUs are busy",
    )
    parser.add_argument(
        "--gpu-check-delay",
        type=int,
        default=60,
        help="Delay in seconds between GPU availability checks",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load base config
    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config["training"]["output_dir"] = str(args.output_dir)

    # Initialize GPU checker
    gpu_checker = GPUAvailabilityChecker(
        num_gpus=args.num_gpus,
        max_retries=args.gpu_check_retries,
        retry_delay=args.gpu_check_delay,
    )

    # Create or load study
    storage = args.storage if args.storage else None
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        # direction="maximize",  # Maximize overlap ratio
        direction="minimize",  # Minimize overlap ratio
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    log.info(f"Starting Optuna sweep: {args.num_trials} trials")
    log.info(f"Base config: {args.base_config}")
    log.info(f"Search space: {args.search_space}")
    log.info(f"Output dir: {args.output_dir}")

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, base_config, args.num_gpus, args.max_steps, args.search_space, gpu_checker
        ),
        n_trials=args.num_trials,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best overlap ratio: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_file = args.output_dir / "optimization_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "num_trials": len(study.trials),
        }, f, indent=2)

    log.info(f"Results saved to {results_file}")

    # Generate best config
    best_config = generate_trial_config(study.best_trial, base_config, args.search_space)
    best_config_path = args.output_dir / "best_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f)

    log.info(f"Best configuration saved to {best_config_path}")


if __name__ == "__main__":
    main()
