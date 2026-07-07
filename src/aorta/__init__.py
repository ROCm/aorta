"""AORTA: PyTorch compute-communication overlap debugging toolkit.

This package provides:
- FSDP2 training benchmarks for overlap debugging
- GPU hardware queue evaluation framework (hw_queue_eval subpackage)
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

# Single source of truth for the version is the distribution metadata, which is
# generated from ``pyproject.toml`` at build/install time. This keeps
# ``aorta.__version__`` in lockstep with whatever was actually installed (wheel,
# ``pip install .``, editable, or ``pip install git+...``) instead of a hard-coded
# literal that silently drifts from the released version. The fallback only
# triggers when running against an uninstalled source tree with no metadata.
try:
    __version__ = version("amd-aorta")
except PackageNotFoundError:  # pragma: no cover - source tree without dist-info
    __version__ = "0.0.0+unknown"


def load_training_entrypoint() -> Any:
    """Lazily import and return the default training entry point."""
    module = import_module("aorta.training.fsdp_trainer")
    return module.main


def load_hw_queue_eval():
    """Lazily import the hw_queue_eval subpackage."""
    return import_module("aorta.hw_queue_eval")


__all__ = ["load_training_entrypoint", "load_hw_queue_eval", "__version__"]
