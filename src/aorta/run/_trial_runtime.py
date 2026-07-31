"""Shared trial lifecycle entry used by direct and isolated execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aorta.registry import Environment
from aorta.run.results import TrialResult
from aorta.workloads import Workload, WorkloadResult

if TYPE_CHECKING:
    from aorta.run.dispatcher import RunRequest


def execute_trial(
    *,
    trial_idx: int,
    workload_cls: type[Workload],
    request: RunRequest,
    env_descriptor: Environment,
    mitigation_env: dict[str, str],
    results_dir: Path,
    should_write: bool,
    persist_result: bool,
    result_transform: (Callable[[WorkloadResult, str], tuple[WorkloadResult, str]] | None) = None,
    skip_cleanup_on_error: bool = False,
) -> TrialResult:
    # Local import avoids a dispatcher -> runtime -> dispatcher import cycle.
    from aorta.run.dispatcher import _run_single_trial

    return _run_single_trial(
        trial_idx=trial_idx,
        workload_cls=workload_cls,
        request=request,
        env_descriptor=env_descriptor,
        mitigation_env=mitigation_env,
        results_dir=results_dir,
        should_write=should_write,
        persist_result=persist_result,
        result_transform=result_transform,
        skip_cleanup_on_error=skip_cleanup_on_error,
    )


__all__ = ["execute_trial"]
