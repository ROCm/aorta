"""AORTA: PyTorch compute-communication overlap debugging toolkit.

This package provides:
- FSDP2 training benchmarks for overlap debugging
- GPU hardware queue evaluation framework (hw_queue_eval subpackage)
"""

__version__ = "0.2.0"

from importlib import import_module
from typing import Any


def load_training_entrypoint() -> Any:
    """Lazily import and return the default training entry point."""
    module = import_module("aorta.training.fsdp_trainer")
    return getattr(module, "main")


def load_hw_queue_eval():
    """Lazily import the hw_queue_eval subpackage."""
    return import_module("aorta.hw_queue_eval")


__all__ = ["load_training_entrypoint", "load_hw_queue_eval", "__version__"]
