#!/usr/bin/env python
"""Run the SDMA overlap prototype benchmark."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from aorta.experiments import BenchmarkConfig, run_sdma_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDMA overlap prototype")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix dimension for GEMM")
    parser.add_argument("--copy-mb", type=int, default=64, help="Megabytes to copy via SDMA per iteration")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measured iterations")
    parser.add_argument("--cold-iters", type=int, default=3, help="Warm-up iterations to skip from timing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        device=args.device,
        matrix_size=args.matrix_size,
        copy_megabytes=args.copy_mb,
        iterations=args.iterations,
        cold_iters=args.cold_iters,
    )

    metrics = run_sdma_benchmark(cfg)
    print("SDMA overlap benchmark")
    print(f"  Device: cuda:{cfg.device}")
    print(f"  Matrix size: {cfg.matrix_size} x {cfg.matrix_size}")
    print(f"  Copy volume: {cfg.copy_megabytes} MB")
    print(f"  Iterations: {cfg.iterations} (warm-up {cfg.cold_iters})")
    print(f"  Sequential avg time (ms): {metrics['sequential_ms']:.3f}")
    print(f"  Overlapped avg time (ms): {metrics['overlapped_ms']:.3f}")
    print(f"  Estimated savings (%): {metrics['savings_percent']:.2f}")


if __name__ == "__main__":
    main()
