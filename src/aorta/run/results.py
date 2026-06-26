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
        exit_status: Outcome of the trial execution.  Values:

            * ``"ok"`` -- workload ran and reported ``passed=True``.
            * ``"workload_failed"`` -- workload ran and reported
              ``passed=False`` from ``run()`` (e.g. NaN detected,
              assertion fired mid-loop).
            * ``"workload_setup_failed"`` -- ``workload.setup()`` raised
              before the workload's main work could begin (e.g. missing
              dependency at import time, broken env probe). Distinct
              from ``infrastructure_failed`` so a setup-time crash
              can't masquerade as a 100 % failure rate of the
              measurement under test -- a row of all-setup-failures
              means the workload never got off the ground, not that
              the bug reproduces every trial.
            * ``"infrastructure_failed"`` -- the dispatcher caught an
              exception that wasn't attributable to ``setup()``: the
              workload class itself failed to construct
              (``workload_cls(config)`` raised), or ``run()`` raised
              after ``setup()`` returned cleanly.

            ``"timeout"`` is deliberately NOT in the literal: B1 ships
            no ``--timeout`` flag and no watchdog, so no code path can
            produce it.  Re-add the value in the same commit that adds
            a producer (e.g. when a ``--timeout`` watchdog lands).
    """

    trial_id: str
    workload: str
    execution_env: dict[str, Any]
    mitigations_applied: tuple[str, ...]
    config: dict[str, Any]
    env: dict[str, Any]
    result: dict[str, Any]
    wall_clock_sec: float
    exit_status: Literal[
        "ok", "workload_failed", "workload_setup_failed", "infrastructure_failed"
    ]
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


def trial_verdict(trial: Any) -> str:
    """Three-way verdict (``"pass"`` / ``"fail"`` / ``"error"``) for a trial.

    This is the single shared predicate (issue #230) used by the matrix
    aggregator (pass / fail / error counts) and the ``stop_after`` event
    counter (:mod:`aorta.run.dispatcher`) so the two can never disagree
    about whether a trial reproduced the bug, failed for an infra reason,
    or passed.

    Accepts any object exposing ``workload`` / ``exit_status`` and a
    ``result`` dict (duck-typed -- callers pass :class:`TrialResult` or
    lightweight stand-ins in tests). Resolution order:

    1. **Probe trials** carry the classifier's three-way verdict in
       ``result["metrics"]["verdict"]``; it is authoritative. (A probe
       ``error`` trial reports ``passed=False`` and therefore
       ``exit_status == "workload_failed"``, so the metric is the only
       place the error/fail distinction survives.)  This is trusted
       **only** for the probe producer (``trial.workload ==
       _subprocess``).  ``metrics`` is otherwise a free-form,
       workload-owned field, so a non-probe workload could legitimately
       stash its own ``metrics["verdict"]`` -- trusting it for every
       workload would let a failed trial be miscounted as a pass in the
       matrix and the ``stop_after`` rule.  Gating on the producer keeps
       the metric authoritative exactly where ``_subprocess`` writes it.
    2. **Other trials** (triage workloads with no probe verdict) derive
       it from ``exit_status``: an ``infrastructure_failed`` /
       ``workload_setup_failed`` trial never validly ran the measurement
       -> ``error``; any other non-``ok`` status, or a ``WorkloadResult``
       reporting ``passed is False`` -> ``fail``; otherwise ``pass``.
    """
    result = getattr(trial, "result", None)
    workload = getattr(trial, "workload", None)
    # Only the probe producer's metric verdict is authoritative (see
    # docstring). The producer name and verdict vocabulary are imported
    # locally from their canonical sources so neither can drift from this
    # predicate, and so aorta.run keeps no module-load dependency on
    # aorta.probe (no cycle today; the local imports keep it that way).
    if isinstance(result, dict) and workload is not None:
        from aorta.probe.recipe_builder import SUBPROCESS_WORKLOAD_NAME

        if workload == SUBPROCESS_WORKLOAD_NAME:
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                v = metrics.get("verdict")
                from aorta.probe.classifier.verdict import VALID_VERDICTS

                # The isinstance guard ensures a non-string metric value can
                # never match (and never reaches frozenset membership with an
                # unhashable type).
                if isinstance(v, str) and v in VALID_VERDICTS:
                    return v

    status = getattr(trial, "exit_status", None)
    if status in ("infrastructure_failed", "workload_setup_failed"):
        return "error"
    if status is not None and status != "ok":
        return "fail"
    if isinstance(result, dict) and result.get("passed") is False:
        return "fail"
    return "pass"


__all__ = ["TrialResult", "trial_verdict"]
