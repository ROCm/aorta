"""Workload discovery via importlib.metadata entry-points.

Workloads are discovered from the 'aorta.workloads' entry-point group.
Both public workloads (in aorta.workloads.*) and private workloads
(in separate downstream packages) register against this group.
"""

import importlib.metadata
import logging

from aorta.run.validation import (
    IN_PROCESS_ONLY_POLICY,
    IN_PROCESS_REQUIRED_POLICY,
    PROCESS_OPTIONAL_POLICY,
    PROCESS_REQUIRED_POLICY,
    WorkloadIsolationPolicy,
)
from aorta.workloads import Workload

logger = logging.getLogger(__name__)

_WORKLOAD_GROUP = "aorta.workloads"
_WORKLOAD_POLICY_GROUP = "aorta.workload_policies"
_POLICY_TARGETS: dict[str, WorkloadIsolationPolicy] = {
    "aorta.run.validation:IN_PROCESS_ONLY_POLICY": IN_PROCESS_ONLY_POLICY,
    "aorta.run.validation:IN_PROCESS_REQUIRED_POLICY": IN_PROCESS_REQUIRED_POLICY,
    "aorta.run.validation:PROCESS_OPTIONAL_POLICY": PROCESS_OPTIONAL_POLICY,
    "aorta.run.validation:PROCESS_REQUIRED_POLICY": PROCESS_REQUIRED_POLICY,
}
# Editable installs created before the policy entry-point group was added still
# expose the workload targets. Keep built-ins correct without requiring a
# reinstall; packaged releases use the explicit policy entries below.
_BUILTIN_POLICY_BY_TARGET: dict[str, WorkloadIsolationPolicy] = {
    "aorta.workloads._subprocess:SubprocessWorkload": IN_PROCESS_REQUIRED_POLICY,
    "aorta.workloads.llm_determinism:LlmDeterminismWorkload": PROCESS_OPTIONAL_POLICY,
    "aorta.workloads.race:RaceWorkload": PROCESS_REQUIRED_POLICY,
}


class UnknownWorkloadError(ValueError):
    """Raised when no installed workload entry point matches a name."""


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
    group = importlib.metadata.entry_points().select(group=_WORKLOAD_GROUP)

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
        raise UnknownWorkloadError(
            f"Workload '{name}' not found. Available: {available}"
        )
    return workloads[name]


def get_workload_policy(name: str) -> WorkloadIsolationPolicy:
    """Return isolation metadata without importing workload implementation code."""
    catalog = importlib.metadata.entry_points()
    workload_entries = list(catalog.select(group=_WORKLOAD_GROUP))
    matches = [entry for entry in workload_entries if entry.name == name]
    if not matches:
        available = sorted({entry.name for entry in workload_entries})
        raise UnknownWorkloadError(
            f"Workload '{name}' not found. Available: {available}"
        )

    policy_entries = [
        entry
        for entry in catalog.select(group=_WORKLOAD_POLICY_GROUP)
        # Test doubles written before this group existed often return their
        # workload entries for every select() call. Real EntryPoint objects
        # carry ``group``; filter explicitly so those doubles safely fall back.
        if getattr(entry, "group", None) == _WORKLOAD_POLICY_GROUP
        and entry.name == name
    ]
    if len(policy_entries) > 1:
        raise ValueError(
            f"Workload '{name}' has multiple {_WORKLOAD_POLICY_GROUP!r} entries"
        )
    if policy_entries:
        target = policy_entries[0].value
        try:
            return _POLICY_TARGETS[target]
        except KeyError as exc:
            raise ValueError(
                f"Workload '{name}' has unsupported isolation policy target "
                f"{target!r}; expected one of {sorted(_POLICY_TARGETS)}"
            ) from exc

    workload_target = getattr(matches[-1], "value", None)
    if isinstance(workload_target, str):
        built_in = _BUILTIN_POLICY_BY_TARGET.get(workload_target)
        if built_in is not None:
            return built_in
    return IN_PROCESS_ONLY_POLICY


__all__ = [
    "UnknownWorkloadError",
    "discover_workloads",
    "get_workload_class",
    "get_workload_policy",
]
