"""aorta.run - Universal workload runner.

This module provides the core infrastructure for running workloads across
trials, environments, and mitigations. It supports both single-process and
distributed (torchrun) launch modes.

Public API:
    run_trials(request: RunRequest) -> list[TrialResult]
    RunRequest: Configuration for a run
    TrialResult: Per-trial result wrapper

Example:
    from aorta.run import run_trials, RunRequest

    request = RunRequest(
        workload="fsdp",
        trials=3,
        environment="local",
        mitigations=("tf32_off",),
        steps=100,
    )
    results = run_trials(request)
"""

from aorta.run.dispatcher import RunRequest, run_trials
from aorta.run.results import TrialResult

__all__ = ["RunRequest", "TrialResult", "run_trials"]
