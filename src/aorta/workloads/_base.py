"""Workload plugin contract.

The `Workload` ABC is the shape every plugin under the `aorta.workloads`
entry-point group must implement. `aorta run` discovers workloads via that
group, instantiates the class with a config dict, and drives the
setup -> run -> cleanup lifecycle once per trial.

`WorkloadResult` is the per-trial return value. Generic enough to cover
both correctness-flavored workloads (NaN, corruption, divergence) and
perf-flavored workloads (timing, throughput). Workload-specific data goes
in the `metrics` dict.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


@dataclass
class WorkloadResult:
    """Per-trial result returned by `Workload.run()`.

    Attributes:
        passed: True if the trial met the workload's success criterion.
        failure_count: Number of failures observed (NaN events, corruption
            counts, divergence steps, etc. - workload-defined).
        first_failure_iteration: Iteration index of the first observed
            failure, or None if the trial passed.
        failure_details: Workload-specific structured failure records.
            Each entry is a dict the workload populates (e.g., the iteration,
            rank, expected vs actual values, layer name).
        total_iterations: Total iterations the workload executed.
        step_times_ms: Per-iteration step times in milliseconds. Consumed by
            `aorta triage --mode matrix` for speed-confound detection.
        elapsed_sec: Total wall-clock elapsed time for the trial.
        metrics: Extension point for workload-specific scalars (overlap
            ratio, throughput, NaN rate per phase, etc.).
        main_work_started: True once the workload's primary,
            measurement-relevant code path begins (training loop entered;
            first kernel dispatched; build invoked). Generic across workload
            types — no "training" / "loop" / iteration assumption baked in.
            False means the trial died during import/setup before doing any
            measurable work; the triage matrix uses this to refuse step-time
            and confound classification for the cell. None means the workload
            doesn't track this, which opts the safety net out gracefully.
        executed_iterations: Iterations actually completed by the workload
            (workload defines what counts as an iteration). None when the
            workload doesn't track this. Pairs with configured_iterations to
            populate the matrix.md "Iters" column.
        configured_iterations: Iterations the workload was asked to run.
            None when the workload doesn't track this. Used as the
            denominator of the "Iters" column.
    """

    passed: bool
    failure_count: int = 0
    first_failure_iteration: int | None = None
    failure_details: list[dict[str, Any]] = field(default_factory=list)
    total_iterations: int = 0
    step_times_ms: list[float] = field(default_factory=list)
    elapsed_sec: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    main_work_started: bool | None = None
    executed_iterations: int | None = None
    configured_iterations: int | None = None


class Workload(ABC):
    """Base class for workloads registered under `aorta.workloads`.

    Concrete workloads live in either the public `aorta.workloads.*`
    namespace (e.g., `aorta.workloads.fsdp`) or in private plugin packages
    (e.g., `<your_pkg>.workloads.<name>`). Both register against the same
    entry-point group so `aorta run --workload <name>` discovers them
    uniformly.

    Lifecycle (driven by `aorta run`, once per trial):

        wl = WorkloadClass(config)
        wl.setup()
        result = wl.run()
        wl.cleanup()

    Subclasses MUST implement `setup()` and `run()`. `cleanup()` is
    optional and defaults to a no-op.

    Launch-mode declaration:
        Workloads default to `launch_mode = "single_process"` — `aorta run`
        is invoked once and runs without a torchrun wrapper. Workloads that
        require torch.distributed (FSDP, DDP, etc.) override
        `launch_mode = "distributed"` and set `min_world_size` to the
        minimum rank count they need. `aorta run` validates the active
        `WORLD_SIZE` env var against these declarations before calling
        `setup()` and raises a clear error on mismatch (single_process
        invoked under torchrun, or distributed invoked without).
    """

    launch_mode: ClassVar[Literal["single_process", "distributed"]] = "single_process"
    min_world_size: ClassVar[int] = 1

    def __init__(self, config: dict[str, Any]) -> None:
        """Store the per-trial config dict.

        Args:
            config: Configuration for this trial, assembled by `aorta run`
                from CLI flags, workload defaults, and any active mitigations.
                Workload reads its own keys; unknown keys are ignored.
        """
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Allocate buffers, init distributed, prepare model.

        Called once before `run()`. Workloads that need torch.distributed
        init their own process group here.
        """

    @abstractmethod
    def run(self) -> WorkloadResult:
        """Execute the workload and return a result.

        Implementations should fill in as many `WorkloadResult` fields as
        applicable. At minimum, `passed` must be set.
        """

    def cleanup(self) -> None:
        """Release resources held by `setup()`. Default no-op."""


__all__ = ["Workload", "WorkloadResult"]
