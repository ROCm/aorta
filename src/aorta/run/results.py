"""Per-trial result dataclass.

The TrialResult wraps WorkloadResult with additional metadata about
the execution environment, configuration, and timing.
"""

import copy
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TrialResult:
    """Per-trial result wrapper around WorkloadResult.

    Schema version 0.1 (unstable until external consumers pin it).

    The dataclass is ``frozen=True`` to prevent attribute reassignment,
    but ``execution_env`` / ``config`` / ``env`` / ``result`` are dicts
    -- ``frozen`` does not stop callers from mutating those nested
    structures.  ``__post_init__`` and ``from_dict`` therefore store
    deep copies, so a ``TrialResult`` is effectively immutable from
    construction time and an in-memory result can never silently drift
    from its persisted JSON form.

    Attributes:
        schema_version: Version of the result schema (for future migration).
        trial_id: Unique identifier for this trial.  Encodes the cell
            coordinates so artifacts from different cells in a triage
            matrix don't collide: ``<workload>_d<dataset_index>_m<mitigation_index>_t<trial_index>``
            (e.g. ``"fsdp_d0_m0_t0"``).  ``aorta run`` is one cell so
            ``d`` and ``m`` are always ``0``; ``aorta triage`` (B2)
            varies them across the matrix.
        workload: Name of the workload that was executed.
        execution_env: Environment descriptor as dict.  Mirrors the
            :class:`aorta.registry.Environment` shape:
            ``{"name": str, "docker": str | None, "venv": str | None,
            "source_package": str}``.  ROCm version, runtime kind, and
            container image digest are NOT part of this block -- they
            live inside ``env`` (A1's ``EnvSnapshot``: ``rocm``,
            ``runtime_context.type``, ``docker.digest``) so that the
            descriptor stays a static recipe and the snapshot stays a
            runtime observation.
        mitigations_applied: Tuple of mitigation names that were applied.
        config: Configuration dict passed to the workload.
        env: Environment snapshot as dict (from A1's
            ``collect_env`` -- includes ``rocm``, ``hip``,
            ``runtime_context``, ``docker``, ``env_vars``,
            ``partial`` / ``partial_reasons``, etc.).
        result: WorkloadResult serialized to dict.
        wall_clock_sec: Total wall clock time for the trial.
        exit_status: Outcome of the trial execution.  ``"timeout"`` is
            deliberately NOT in the literal: B1 ships no ``--timeout``
            flag and no watchdog, so no code path can produce it.
            Re-add the value in the same commit that adds a producer
            (e.g. when a ``--timeout`` watchdog lands).
    """

    trial_id: str
    workload: str
    execution_env: dict[str, Any]
    mitigations_applied: tuple[str, ...]
    config: dict[str, Any]
    env: dict[str, Any]
    result: dict[str, Any]
    wall_clock_sec: float
    exit_status: Literal["ok", "workload_failed", "infrastructure_failed"]
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        # Defensively deep-copy the mutable dict fields so the caller
        # cannot mutate them out from under us after construction.
        # ``frozen=True`` blocks attribute reassignment, so we use
        # ``object.__setattr__`` to install the copies.
        for field_name in ("execution_env", "config", "env", "result"):
            object.__setattr__(self, field_name, copy.deepcopy(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Returns deep copies of the mutable dict fields so callers cannot
        mutate the result's internal state by editing the serialized
        view.
        """
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "workload": self.workload,
            "execution_env": copy.deepcopy(self.execution_env),
            "mitigations_applied": list(self.mitigations_applied),
            "config": copy.deepcopy(self.config),
            "env": copy.deepcopy(self.env),
            "result": copy.deepcopy(self.result),
            "wall_clock_sec": self.wall_clock_sec,
            "exit_status": self.exit_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResult":
        """Deserialize from dict.

        ``__post_init__`` deep-copies the mutable fields, so subsequent
        mutation of ``data`` cannot affect the constructed instance.
        """
        return cls(
            schema_version=data.get("schema_version", "0.1"),
            trial_id=data["trial_id"],
            workload=data["workload"],
            execution_env=data["execution_env"],
            mitigations_applied=tuple(data["mitigations_applied"]),
            config=data["config"],
            env=data["env"],
            result=data["result"],
            wall_clock_sec=data["wall_clock_sec"],
            exit_status=data["exit_status"],
        )


__all__ = ["TrialResult"]
