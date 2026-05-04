"""Run dispatcher - orchestrates workload execution across trials.

The dispatcher is the core of `aorta run`. It:
1. Discovers and instantiates workloads
2. Validates launch mode before execution
3. Applies environment and mitigation configuration
4. Runs trials and collects results
5. Persists results as JSON (rank 0 only for distributed)
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aorta.workloads import Workload, WorkloadResult
from aorta.run.collectors import KNOWN_RECIPES
from aorta.run.discovery import get_workload_class
from aorta.run.validation import validate_launch_mode
from aorta.run.results import TrialResult

# Import from stubs (replace when A1/B3 land)
from aorta.run._stubs import (
    collect_env,
    get_environment,
    get_mitigation,
    Environment,
)


@dataclass(frozen=True)
class RunRequest:
    """Configuration for a run_trials() invocation.

    Attributes:
        workload: Name of the workload to run (from entry-point group).
        trials: Number of trials to execute.
        environment: Environment name (default: local).
        mitigations: Tuple of mitigation names to apply.
        extra_env: Additional environment variables (override mitigations).
        steps: Number of steps per trial (workload-specific).
        config_overrides: Additional workload configuration.
        results_dir: Directory to write per-trial JSON files.
        collect: Collector recipe names (MVP: validated but no-op).
    """

    workload: str
    trials: int
    environment: str = "local"
    mitigations: tuple[str, ...] = ("none",)
    extra_env: dict[str, str] = field(default_factory=dict)
    steps: int | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    results_dir: Path = field(default_factory=lambda: Path("results"))
    collect: tuple[str, ...] = field(default_factory=tuple)


def run_trials(request: RunRequest) -> list[TrialResult]:
    """Run N trials for a single (workload, environment, mitigation-set) combination.

    This is the main entry point for the workload runner. It handles:
    - Workload discovery and instantiation
    - Launch mode validation
    - Environment and mitigation configuration
    - Trial execution with error handling
    - JSON result persistence (rank 0 only)

    Args:
        request: Configuration for the run.

    Returns:
        List of TrialResult objects, one per trial.

    Raises:
        ValueError: If workload or environment/mitigation not found.
        RuntimeError: If launch mode validation fails.
    """
    # 1. Validate collector recipe names.  The CLI also validates this
    #    against KNOWN_RECIPES, but ``run_trials`` is a public library
    #    API consumed by B2 (triage matrix runner) -- programmatic
    #    callers deserve the same protection.
    invalid_collectors = set(request.collect) - KNOWN_RECIPES
    if invalid_collectors:
        raise ValueError(
            f"Unknown collector recipes: {sorted(invalid_collectors)}. "
            f"Valid: {sorted(KNOWN_RECIPES)}"
        )

    # 2. Discover workload
    workload_cls = get_workload_class(request.workload)

    # 3. Validate launch mode BEFORE setup()
    validate_launch_mode(workload_cls)

    # 4. Resolve environment
    env_descriptor = get_environment(request.environment)

    # 5. Resolve and union mitigations
    mitigation_env: dict[str, str] = {}
    for name in request.mitigations:
        mitigation = get_mitigation(name)
        mitigation_env.update(mitigation.env_vars)

    # 6. Determine if we should write (rank 0 only for distributed).
    #    Only rank 0 needs the output directory; creating it on every
    #    rank causes shared-FS contention and weakens the rank-0-only
    #    write guarantee.
    rank = int(os.environ.get("RANK", "0"))
    should_write = rank == 0
    results_dir = request.results_dir / request.workload
    if should_write:
        results_dir.mkdir(parents=True, exist_ok=True)

    # 7. Run trials
    results: list[TrialResult] = []
    for trial_idx in range(request.trials):
        result = _run_single_trial(
            trial_idx=trial_idx,
            workload_cls=workload_cls,
            request=request,
            env_descriptor=env_descriptor,
            mitigation_env=mitigation_env,
            results_dir=results_dir,
            should_write=should_write,
        )
        results.append(result)

    return results


def _run_single_trial(
    trial_idx: int,
    workload_cls: type[Workload],
    request: RunRequest,
    env_descriptor: Environment,
    mitigation_env: dict[str, str],
    results_dir: Path,
    should_write: bool,
) -> TrialResult:
    """Execute a single trial.

    Args:
        trial_idx: Index of the current trial (0-based).
        workload_cls: The workload class to instantiate.
        request: The run request configuration.
        env_descriptor: Resolved environment descriptor.
        mitigation_env: Environment variables from mitigations.
        results_dir: Directory for JSON output.
        should_write: Whether to write JSON (rank 0 only).

    Returns:
        TrialResult with execution outcome.
    """
    trial_id = f"{request.workload}_t{trial_idx}"
    # ``perf_counter`` is monotonic; ``time.time()`` can jump backward
    # or forward when the system clock is adjusted (NTP, suspend/resume),
    # which would corrupt ``wall_clock_sec``.
    start_time = time.perf_counter()

    # Capture environment snapshot (using stub, replace with A1 when available)
    env_snapshot = collect_env()

    # Build config
    config: dict[str, Any] = {**request.config_overrides}
    if request.steps is not None:
        config["steps"] = request.steps

    # Save original environment for restoration
    original_env = dict(os.environ)

    # Apply mitigation env + extra_env
    os.environ.update(mitigation_env)
    os.environ.update(request.extra_env)

    # Instantiate and run workload
    exit_status: str = "ok"
    workload_result: WorkloadResult
    workload: Workload | None = None

    try:
        # Construct positionally to match the documented Workload(config)
        # contract -- third-party plugins are free to name their first
        # parameter something other than ``config``.
        workload = workload_cls(config)
        workload.setup()
        workload_result = workload.run()

        if not workload_result.passed:
            exit_status = "workload_failed"

    except Exception as e:
        exit_status = "infrastructure_failed"
        # Create error WorkloadResult
        workload_result = WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[{"error": str(e), "type": type(e).__name__}],
        )

    finally:
        # Always attempt cleanup if the workload was constructed, even
        # when setup()/run() raised -- otherwise we leak GPU memory,
        # process groups, file handles, etc.  Cleanup failures are not
        # allowed to mask the original exception/exit_status.
        if workload is not None:
            try:
                workload.cleanup()
            except Exception:
                pass
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    wall_clock = time.perf_counter() - start_time

    # Build execution_env block
    execution_env = {
        "kind": env_descriptor.kind,
        "name": env_descriptor.name,
        "image": env_descriptor.docker,
        "digest": None,  # Best-effort, would need docker inspect
        "venv": env_descriptor.venv,
        "rocm": env_descriptor.rocm,
        "source_package": env_descriptor.source_package,
    }

    # Build TrialResult
    trial_result = TrialResult(
        trial_id=trial_id,
        workload=request.workload,
        execution_env=execution_env,
        mitigations_applied=request.mitigations,
        config=config,
        env=env_snapshot.to_dict(),
        result=asdict(workload_result),
        wall_clock_sec=wall_clock,
        exit_status=exit_status,  # type: ignore[arg-type]
    )

    # Write JSON (rank 0 only)
    if should_write:
        output_path = results_dir / f"trial_{trial_idx}.json"
        with open(output_path, "w") as f:
            json.dump(trial_result.to_dict(), f, indent=2)

    return trial_result


__all__ = ["RunRequest", "run_trials"]
