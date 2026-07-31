"""Launch-mode validation for workloads.

Validates that the runtime environment (WORLD_SIZE) matches the workload's
declared launch_mode before allowing execution. This catches common
misconfiguration errors early with clear error messages.
"""

import os
from dataclasses import dataclass
from typing import Literal

from aorta.workloads import Workload

TrialIsolation = Literal["in_process", "process"]
TrialIsolationRequest = Literal["auto", "in_process", "process"]
TRIAL_ISOLATION_REQUESTS = frozenset({"auto", "in_process", "process"})
TRIAL_ISOLATION_MODES = frozenset({"in_process", "process"})


@dataclass(frozen=True)
class WorkloadIsolationPolicy:
    """Side-effect-free trial-isolation metadata for a workload entry point."""

    default: TrialIsolation = "in_process"
    required: bool = False
    supported: frozenset[str] = frozenset({"in_process"})


IN_PROCESS_ONLY_POLICY = WorkloadIsolationPolicy()
IN_PROCESS_REQUIRED_POLICY = WorkloadIsolationPolicy(required=True)
PROCESS_OPTIONAL_POLICY = WorkloadIsolationPolicy(
    supported=frozenset({"in_process", "process"})
)
PROCESS_REQUIRED_POLICY = WorkloadIsolationPolicy(
    default="process",
    required=True,
    supported=frozenset({"process"}),
)


def race_startup_env(config: dict[str, object]) -> dict[str, str]:
    """Side-effect-free pre-import defaults for the Race workload."""
    value = config.get("gpu_max_hw_queues", 4)
    if value is None:
        return {}
    return {"GPU_MAX_HW_QUEUES": str(value)}


def validate_launch_mode(workload_cls: type[Workload]) -> None:
    """Validate WORLD_SIZE matches workload's launch_mode declaration.

    This validation runs before setup() to catch misconfiguration early.

    Args:
        workload_cls: The workload class to validate.

    Raises:
        RuntimeError: On launch mode mismatch with clear remediation guidance.

    Examples:
        # single_process workload incorrectly launched under torchrun:
        RuntimeError: Workload 'MyWorkload' is single_process;
            do not wrap with torchrun (WORLD_SIZE=4)

        # distributed workload without torchrun:
        RuntimeError: Workload 'FsdpWorkload' requires WORLD_SIZE >= 2
            (got 1); launch with: torchrun --nproc_per_node=2 -m aorta run ...
    """
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as e:
        raise RuntimeError(
            f"Invalid WORLD_SIZE={raw_world_size!r}: expected an integer "
            "(launchers should set WORLD_SIZE to the rank count)."
        ) from e

    # WORLD_SIZE is the rank count -- zero or negative is structurally
    # invalid for both launch modes, regardless of what the workload
    # declares.  Reject it up-front with a clear message instead of
    # silently treating ``WORLD_SIZE=0`` like ``WORLD_SIZE=1`` (the
    # default branch of the ``> 1`` / ``< min`` checks below).
    if world_size < 1:
        raise RuntimeError(
            f"Invalid WORLD_SIZE={world_size}: must be >= 1 "
            "(launchers set this to the rank count, which is always "
            "at least 1)."
        )

    launch_mode = workload_cls.launch_mode
    min_world_size = workload_cls.min_world_size

    if launch_mode == "single_process" and world_size > 1:
        raise RuntimeError(
            f"Workload '{workload_cls.__name__}' is single_process; "
            f"do not wrap with torchrun (WORLD_SIZE={world_size})"
        )

    if launch_mode == "distributed" and world_size < min_world_size:
        raise RuntimeError(
            f"Workload '{workload_cls.__name__}' requires WORLD_SIZE >= {min_world_size} "
            f"(got {world_size}); launch with: "
            f"torchrun --standalone --nproc_per_node={min_world_size} "
            f"$(which aorta) run --workload ... "
            f"(use the 'aorta' console script; '-m aorta' is not a runnable module)"
        )


def _resolve_trial_isolation_metadata(
    *,
    workload_label: str,
    default: object,
    required: object,
    supported: object,
    requested: str,
) -> TrialIsolation:
    if not isinstance(requested, str) or requested not in TRIAL_ISOLATION_REQUESTS:
        raise ValueError(
            f"trial_isolation must be one of {sorted(TRIAL_ISOLATION_REQUESTS)}, "
            f"got {requested!r}"
        )

    if default not in TRIAL_ISOLATION_MODES:
        raise ValueError(
            f"{workload_label} declares invalid "
            f"trial_isolation_default={default!r}"
        )
    if not isinstance(required, bool):
        raise ValueError(
            f"{workload_label} declares non-boolean "
            f"trial_isolation_required={required!r}"
        )
    if not isinstance(supported, frozenset) or not supported or not supported <= TRIAL_ISOLATION_MODES:
        raise ValueError(
            f"{workload_label} declares invalid "
            f"trial_isolation_supported={supported!r}"
        )
    if default not in supported:
        raise ValueError(
            f"{workload_label} default isolation {default!r} "
            f"is not in its supported modes {sorted(supported)}"
        )

    effective = default if requested == "auto" else requested
    if required and effective != default:
        raise ValueError(
            f"{workload_label} requires "
            f"trial_isolation={default!r}; recipe/request asked for {effective!r}"
        )
    if effective not in supported:
        raise ValueError(
            f"{workload_label} does not support "
            f"trial_isolation={effective!r}; supported: {sorted(supported)}"
        )
    return "process" if effective == "process" else "in_process"


def resolve_trial_isolation(
    workload_cls: type[Workload],
    requested: str,
) -> TrialIsolation:
    """Resolve a recipe/request isolation policy against loaded class metadata."""
    result_scope = getattr(workload_cls, "distributed_result_scope", "global")
    if result_scope not in {"global", "rank_local"}:
        raise ValueError(
            f"Workload '{workload_cls.__name__}' declares invalid "
            f"distributed_result_scope={result_scope!r}"
        )
    return _resolve_trial_isolation_metadata(
        workload_label=f"Workload '{workload_cls.__name__}'",
        default=getattr(workload_cls, "trial_isolation_default", "in_process"),
        required=getattr(workload_cls, "trial_isolation_required", False),
        supported=getattr(
            workload_cls,
            "trial_isolation_supported",
            frozenset({"in_process"}),
        ),
        requested=requested,
    )


def resolve_trial_isolation_policy(
    workload_name: str,
    policy: WorkloadIsolationPolicy,
    requested: str,
) -> TrialIsolation:
    """Resolve isolation without importing the workload implementation."""
    if not isinstance(policy, WorkloadIsolationPolicy):
        raise ValueError(
            f"Workload {workload_name!r} has invalid isolation policy "
            f"{policy!r}"
        )
    return _resolve_trial_isolation_metadata(
        workload_label=f"Workload {workload_name!r}",
        default=policy.default,
        required=policy.required,
        supported=policy.supported,
        requested=requested,
    )


__all__ = [
    "IN_PROCESS_ONLY_POLICY",
    "IN_PROCESS_REQUIRED_POLICY",
    "PROCESS_OPTIONAL_POLICY",
    "PROCESS_REQUIRED_POLICY",
    "TRIAL_ISOLATION_MODES",
    "TRIAL_ISOLATION_REQUESTS",
    "TrialIsolation",
    "TrialIsolationRequest",
    "WorkloadIsolationPolicy",
    "resolve_trial_isolation",
    "resolve_trial_isolation_policy",
    "race_startup_env",
    "validate_launch_mode",
]
