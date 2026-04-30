"""Per-trial result dataclass.

The TrialResult wraps WorkloadResult with additional metadata about
the execution environment, configuration, and timing.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TrialResult:
    """Per-trial result wrapper around WorkloadResult.

    Schema version 0.1 (unstable until external consumers pin it).

    Attributes:
        schema_version: Version of the result schema (for future migration).
        trial_id: Unique identifier for this trial (e.g., "fsdp_t0").
        workload: Name of the workload that was executed.
        execution_env: Environment descriptor as dict (kind, name, image, etc.).
        mitigations_applied: Tuple of mitigation names that were applied.
        config: Configuration dict passed to the workload.
        env: Environment snapshot as dict (from A1's collect_env).
        result: WorkloadResult serialized to dict.
        wall_clock_sec: Total wall clock time for the trial.
        exit_status: Outcome of the trial execution.
    """

    trial_id: str
    workload: str
    execution_env: dict[str, Any]
    mitigations_applied: tuple[str, ...]
    config: dict[str, Any]
    env: dict[str, Any]
    result: dict[str, Any]
    wall_clock_sec: float
    exit_status: Literal["ok", "workload_failed", "infrastructure_failed", "timeout"]
    schema_version: str = "0.1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "workload": self.workload,
            "execution_env": self.execution_env,
            "mitigations_applied": list(self.mitigations_applied),
            "config": self.config,
            "env": self.env,
            "result": self.result,
            "wall_clock_sec": self.wall_clock_sec,
            "exit_status": self.exit_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResult":
        """Deserialize from dict."""
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
