"""AORTA: PyTorch compute-communication overlap debugging toolkit.

This package provides:
- FSDP2 training benchmarks for overlap debugging
- GPU hardware queue evaluation framework (hw_queue_eval subpackage)
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    """Resolve ``aorta.__version__`` on first access (PEP 562).

    Single source of truth for the version is the distribution metadata, which
    is generated from ``pyproject.toml`` at build/install time. This keeps
    ``aorta.__version__`` in lockstep with whatever was actually installed
    (wheel, ``pip install .``, editable, or ``pip install git+...``) instead of
    a hard-coded literal that silently drifts from the released version.

    Reading that metadata pulls in ``importlib.metadata`` and walks ``sys.path``
    for dist-info, which dominated the cost of ``import aorta`` -- roughly
    three quarters of it, for an attribute almost no caller reads (issue #417).
    Stated as a share rather than a duration because the absolute number moves
    with the interpreter and how many distributions are installed. Resolving on
    demand and caching the result in the module globals keeps the guarantee
    without charging every import for it.

    The fallback is best-effort and defensive: besides the expected
    missing-metadata case (uninstalled source tree), any unexpected metadata
    error (e.g. unreadable/corrupted dist-info) must not break ``import aorta``.
    """
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        resolved = version("amd-aorta")
    except PackageNotFoundError:  # source tree without dist-info
        resolved = "0.0.0+unknown"
    except Exception:  # pragma: no cover - defensive: never break on metadata errors
        resolved = "0.0.0+unknown"

    globals()["__version__"] = resolved
    return resolved


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def load_training_entrypoint() -> Any:
    """Lazily import and return the default training entry point."""
    module = import_module("aorta.training.fsdp_trainer")
    return module.main


def load_hw_queue_eval():
    """Lazily import the hw_queue_eval subpackage."""
    return import_module("aorta.hw_queue_eval")


__all__ = ["load_training_entrypoint", "load_hw_queue_eval", "__version__"]
