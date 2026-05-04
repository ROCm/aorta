"""Workload discovery via importlib.metadata entry-points.

Workloads are discovered from the 'aorta.workloads' entry-point group.
Both public workloads (in aorta.workloads.*) and private workloads
(in separate packages like aorta-internal) register against this group.
"""

import importlib.metadata
import logging

from aorta.workloads import Workload

logger = logging.getLogger(__name__)


def discover_workloads() -> dict[str, type[Workload]]:
    """Discover all workloads registered under aorta.workloads entry-point group.

    Returns:
        Dict mapping workload names to their classes.

    Note:
        Failed imports and entries that don't resolve to a ``Workload``
        subclass are logged via the ``aorta.run.discovery`` logger but
        do not crash discovery -- other workloads remain available.
    """
    workloads: dict[str, type[Workload]] = {}
    # The project requires Python >= 3.10 (see pyproject.toml), so the
    # ``EntryPoints.select`` API is always available; the older 3.9
    # ``entry_points().get(...)`` form is intentionally not supported.
    group = importlib.metadata.entry_points().select(group="aorta.workloads")

    for ep in group:
        try:
            cls = ep.load()
        except Exception:
            # Log but don't crash - allow other workloads to load.  Use
            # a logger (not print) so library callers can control
            # verbosity and filter/redirect normally.  ``exc_info=True``
            # keeps the full traceback on the warning record so plugin
            # load failures (most often ImportError chains) are
            # actually diagnosable.
            logger.warning("Failed to load workload '%s'", ep.name, exc_info=True)
            continue

        # Validate that the entry point actually points at a Workload
        # subclass.  Mis-registered plugins (returning a function, an
        # instance, or an unrelated class) would otherwise be returned
        # here and fail much later with a confusing AttributeError /
        # TypeError inside the dispatcher.
        if not isinstance(cls, type) or not issubclass(cls, Workload):
            logger.warning(
                "Entry point '%s' = %r is not a Workload subclass; skipping.",
                ep.name,
                cls,
            )
            continue

        workloads[ep.name] = cls

    return workloads


def get_workload_class(name: str) -> type[Workload]:
    """Get workload class by name.

    Args:
        name: Registered name of the workload.

    Returns:
        The workload class.

    Raises:
        ValueError: If workload is not found, with list of available workloads.
    """
    workloads = discover_workloads()
    if name not in workloads:
        available = sorted(workloads.keys())
        raise ValueError(f"Workload '{name}' not found. Available: {available}")
    return workloads[name]


__all__ = ["discover_workloads", "get_workload_class"]
