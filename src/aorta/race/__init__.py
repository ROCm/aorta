"""
RCCL Race Condition Reproducer - Standalone test module.

This module provides a minimal, standalone test for detecting runtime-level bugs
in RCCL/HIP that manifest in multi-node distributed training with overlapping streams.

Usage:
    # Default mode (TorchRec-like)
    torchrun --nproc_per_node=8 -m aorta.race --warmup 100 --verify 10000

    # DDP mode (gradient all_reduce)
    torchrun --nproc_per_node=8 -m aorta.race --mode ddp --warmup 100 --verify 10000
"""

# Canonical config location
from aorta.race.config import RaceConfig, ReproducerConfig, ReproducerResult

# Modular mode factory
from aorta.race.modes import create_reproducer

__all__ = [
    # Config
    "RaceConfig",
    "ReproducerConfig",
    "ReproducerResult",
    # Modular system
    "create_reproducer",
]
