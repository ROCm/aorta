"""Launch-mode validation for workloads.

Validates that the runtime environment (WORLD_SIZE) matches the workload's
declared launch_mode before allowing execution. This catches common
misconfiguration errors early with clear error messages.
"""

import os

from aorta.workloads import Workload


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
            f"torchrun --nproc_per_node={min_world_size} -m aorta run ..."
        )


__all__ = ["validate_launch_mode"]
