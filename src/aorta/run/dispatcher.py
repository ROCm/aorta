"""Run dispatcher - orchestrates workload execution across trials.

The dispatcher is the core of `aorta run`. It:
1. Discovers and instantiates workloads
2. Validates launch mode before execution
3. Applies environment and mitigation configuration
4. Runs trials and collects results
5. Persists results as JSON (rank 0 only for distributed)
"""

import copy
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aorta.instrumentation.environment import collect_env
from aorta.registry import Environment, get_environment, get_mitigation
from aorta.run.collectors import KNOWN_RECIPES
from aorta.run.discovery import get_workload_class
from aorta.run.results import TrialResult
from aorta.run.validation import validate_launch_mode
from aorta.workloads import Workload, WorkloadResult

logger = logging.getLogger(__name__)

# Conservative POSIX env-var name shape: must start with a letter or
# underscore and contain only [A-Za-z0-9_].  The CLI also enforces this
# at parse time; library callers pass ``extra_env`` directly so we
# re-validate here for parity.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class RunRequest:
    """Configuration for a run_trials() invocation.

    The dataclass is ``frozen=True`` to prevent attribute reassignment,
    but ``extra_env`` and ``config_overrides`` are dicts -- ``frozen``
    does not stop callers from mutating those nested structures.
    ``__post_init__`` therefore stores deep copies, so an in-flight
    request can never be mutated out from under the dispatcher.  This
    mirrors the same defensive pattern used by :class:`TrialResult`.

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
        sidecar_files: JSON sidecar files describing ad-hoc mitigations
            and/or environments (B3.1).  Forwarded to
            ``aorta.registry.get_mitigation`` /
            ``aorta.registry.get_environment`` so that names declared in
            the sidecar resolve in the same call as built-ins and
            entry-point plugins.
        dataset_index: Cell coordinate on the dataset / environment axis,
            used in ``trial_id`` and the per-trial JSON filename.
            ``aorta run`` is one cell, so the default ``0`` is correct
            for direct CLI use; ``aorta triage`` (B2) calls
            ``run_trials`` once per cell and varies this index across
            its environment axis so cells in the matrix don't collide
            on disk.
        mitigation_index: Cell coordinate on the mitigation axis (same
            rationale as ``dataset_index``).  ``aorta triage`` varies
            this across its mitigation axis; ``aorta run`` always emits
            ``m0``.
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
    sidecar_files: tuple[Path, ...] = field(default_factory=tuple)
    dataset_index: int = 0
    mitigation_index: int = 0

    def __post_init__(self) -> None:
        # Defensively deep-copy mutable dict fields.  ``frozen=True``
        # blocks attribute reassignment, so we use
        # ``object.__setattr__`` to install the copies.
        for field_name in ("extra_env", "config_overrides"):
            object.__setattr__(self, field_name, copy.deepcopy(getattr(self, field_name)))


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
        ValueError: If ``trials`` is not positive, an unknown collector
            recipe is requested, or the workload is not found.
        UnknownEnvironmentError / UnknownMitigationError: If the
            requested environment or mitigation is not in the registry
            (both subclass ``KeyError`` -- callers can also catch
            ``LookupError`` to handle either).
        RuntimeError: If launch-mode validation fails.
    """
    # 1. Validate trial count.  ``trials <= 0`` would silently no-op,
    #    which is almost never what either the CLI or a library caller
    #    intended.
    if request.trials < 1:
        raise ValueError(f"trials must be >= 1 (got {request.trials})")

    # 2. Validate collector recipe names.  The CLI also validates this
    #    against KNOWN_RECIPES, but ``run_trials`` is a public library
    #    API consumed by B2 (triage matrix runner) -- programmatic
    #    callers deserve the same protection.
    invalid_collectors = set(request.collect) - KNOWN_RECIPES
    if invalid_collectors:
        raise ValueError(
            f"Unknown collector recipes: {sorted(invalid_collectors)}. "
            f"Valid: {sorted(KNOWN_RECIPES)}"
        )

    # 3. Validate ``extra_env`` keys.  The CLI validates this at parse
    #    time, but library callers (B2, future programmatic users) pass
    #    ``extra_env`` directly -- without parity here, a bad key would
    #    only fail mid-trial inside ``os.environ.update`` with the much
    #    less friendly ``ValueError: illegal environment variable name``.
    bad_keys = [k for k in request.extra_env if not _ENV_KEY_RE.match(k)]
    if bad_keys:
        raise ValueError(
            f"Invalid extra_env keys {bad_keys}: each key must match "
            "[A-Za-z_][A-Za-z0-9_]* (POSIX env-var name shape)."
        )

    # 4. Discover workload
    workload_cls = get_workload_class(request.workload)

    # 5. Validate launch mode BEFORE setup()
    validate_launch_mode(workload_cls)

    # 6. Resolve environment.  Forward ``sidecar_files`` so any
    #    operator-supplied JSON sidecars (B3.1) are merged with
    #    built-ins and entry-point plugins.
    sidecar_files = list(request.sidecar_files) or None
    env_descriptor = get_environment(request.environment, extra_files=sidecar_files)

    # 7. Resolve and union mitigations.  ``aorta.registry.get_mitigation``
    #    returns a defensive ``dict[str, str]`` per-call, so later
    #    mitigations naturally win over earlier ones in the union.
    mitigation_env: dict[str, str] = {}
    for name in request.mitigations:
        mitigation_env.update(get_mitigation(name, extra_files=sidecar_files))

    # 8. Determine if we should write (rank 0 only for distributed).
    #    Only rank 0 needs the output directory; creating it on every
    #    rank causes shared-FS contention and weakens the rank-0-only
    #    write guarantee.  Parse RANK defensively -- a misconfigured
    #    launcher passing a non-integer should not crash the run.
    raw_rank = os.environ.get("RANK", "0")
    try:
        rank = int(raw_rank)
    except ValueError:
        logger.warning(
            "Ignoring non-integer RANK=%r; treating this process as rank 0.",
            raw_rank,
        )
        rank = 0
    should_write = rank == 0
    results_dir = request.results_dir / request.workload
    if should_write:
        results_dir.mkdir(parents=True, exist_ok=True)

    # 9. Run trials
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
    # Spec format: ``<workload>_d<dataset>_m<mitigation>_t<trial>`` so
    # ``aorta triage`` (B2) can fan out across the dataset/mitigation
    # axes without per-cell trial files colliding.  ``aorta run`` is
    # one cell, so ``d``/``m`` default to 0 here; B2 sets them per
    # cell when it calls ``run_trials`` directly.
    trial_id = (
        f"{request.workload}_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}"
    )
    # ``perf_counter`` is monotonic; ``time.time()`` can jump backward
    # or forward when the system clock is adjusted (NTP, suspend/resume),
    # which would corrupt ``wall_clock_sec``.
    start_time = time.perf_counter()

    # Build config
    config: dict[str, Any] = {**request.config_overrides}
    if request.steps is not None:
        config["steps"] = request.steps

    # Snapshot the env BEFORE applying mitigation / extra_env so the
    # ``finally`` block can restore both the dispatcher's overlay and
    # any workload-side mutations introduced by ``setup()`` / ``run()``.
    pre_trial_env = dict(os.environ)

    # Apply mitigation env + extra_env BEFORE the env snapshot.  The
    # snapshot is supposed to describe the actual environment the
    # workload ran under -- including operator overrides like
    # ``HSA_XNACK=1`` from a mitigation or one-off ``DISABLE_TF32=1``
    # from ``--extra-env``.  Capturing pre-override loses that signal
    # for reproducibility / debugging.
    os.environ.update(mitigation_env)
    os.environ.update(request.extra_env)

    # Capture environment snapshot AFTER env-var application.
    # ``collect_env`` is fail-soft and never raises (see A1 docs).
    env_snapshot = collect_env()

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
            except Exception as cleanup_exc:
                # Log -- silently swallowing makes leaked GPU memory /
                # process groups invisible to the operator.  Use
                # ``exc_info=True`` so the original traceback survives.
                logger.warning(
                    "workload.cleanup() raised %s during trial '%s'; "
                    "continuing so the original outcome is preserved.",
                    type(cleanup_exc).__name__,
                    trial_id,
                    exc_info=True,
                )
        # Restore environment by diff against the pre-trial snapshot.
        # We deliberately do NOT use ``os.environ.clear() +
        # os.environ.update(snapshot)`` -- ``run_trials`` is a public
        # library API and ``clear()`` would, for an instant, blank the
        # entire environment for every other thread in the process.
        # The diff approach has no such window: each key transitions
        # at most once, directly to its target value.
        current_keys = set(os.environ)
        saved_keys = set(pre_trial_env)
        for key in current_keys - saved_keys:
            # Added during the trial (mitigation / extra_env / workload
            # setup) -- remove.
            del os.environ[key]
        for key, value in pre_trial_env.items():
            # Restore both the keys we overwrote and any workload-side
            # mutations to pre-existing keys.  ``os.environ.get`` is
            # cheap; this skip avoids a redundant write when the value
            # is already correct.
            if os.environ.get(key) != value:
                os.environ[key] = value

    wall_clock = time.perf_counter() - start_time

    # Build execution_env block.  Mirrors the public
    # ``aorta.registry.Environment`` shape (no ``kind`` / ``rocm`` --
    # those were stub-isms; ROCm version now lives inside
    # ``env_snapshot.rocm`` and the runtime kind in
    # ``env_snapshot.runtime_context.type``).
    execution_env = {
        "name": env_descriptor.name,
        "docker": env_descriptor.docker,
        "venv": env_descriptor.venv,
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

    # Write JSON (rank 0 only).  Filename mirrors ``trial_id`` so the
    # cell coordinates (``d`` / ``m`` / ``t``) are visible on disk
    # without parsing the JSON -- B2's matrix collator can slice by
    # axis from the filename alone.
    if should_write:
        output_path = results_dir / (
            f"trial_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}.json"
        )
        with open(output_path, "w") as f:
            json.dump(trial_result.to_dict(), f, indent=2)

    return trial_result


__all__ = ["RunRequest", "run_trials"]
