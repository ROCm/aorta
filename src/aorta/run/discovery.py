"""Workload discovery via importlib.metadata entry-points.

Workloads are discovered from the 'aorta.workloads' entry-point group.
Both public workloads (in aorta.workloads.*) and private workloads
(in separate packages like aorta-internal) register against this group.
"""

import importlib.metadata
from typing import Type

from aorta.workloads import Workload


def discover_workloads() -> dict[str, Type[Workload]]:
    """Discover all workloads registered under aorta.workloads entry-point group.

    Returns:
        Dict mapping workload names to their classes.

    Note:
        Failed imports are logged but don't crash discovery, allowing
        other workloads to still be available.
    """
    workloads: dict[str, Type[Workload]] = {}
    eps = importlib.metadata.entry_points()

    # Handle both Python 3.10+ (select) and 3.9 (get) APIs
    if hasattr(eps, "select"):
        group = eps.select(group="aorta.workloads")
    else:
        group = eps.get("aorta.workloads", [])

    for ep in group:
        try:
            cls = ep.load()
            workloads[ep.name] = cls
        except Exception as e:
            # Log but don't crash - allow other workloads to load
            print(f"Warning: Failed to load workload '{ep.name}': {e}")

    return workloads


def get_workload_class(name: str) -> Type[Workload]:
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
