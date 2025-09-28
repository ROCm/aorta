"""AORTA: PyTorch compute-communication overlap debugging toolkit."""

from importlib import import_module
from typing import Any


def load_training_entrypoint() -> Any:
    """Lazily import and return the default training entry point."""

    module = import_module("aorta.training.fsdp_trainer")
    return getattr(module, "main")


__all__ = ["load_training_entrypoint"]
