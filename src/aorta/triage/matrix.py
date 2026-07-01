"""Contingency-table data structure and per-cell aggregation for triage matrices.

Consumes B1's :class:`aorta.run.TrialResult` list for a single cell and emits
a :class:`CellStats` record with the fields the matrix.md table, confound
detection, and matrix.json all need.

Step-time source order (per cell, per trial):

1. ``trial.result["step_times_ms"]`` -- B1 surfaces the workload's
   ``WorkloadResult.step_times_ms`` list here. Preferred signal for
   confound detection since it comes from the workload's own clocks.
   Tagged ``"per_step"``.
2. ``trial.result["elapsed_sec"] / trial.result["total_iterations"] * 1000``
   -- fallback when the workload didn't provide per-iteration step times.
   Tagged ``"elapsed_per_iter"``.
3. ``trial.wall_clock_sec / <steps>`` when both of the above are absent
   (happens for workloads that fail in ``setup()`` before they compute any
   timing at all); attributed to the cell's resolved step count. Folds
   setup / teardown into the per-step number, so the figure is honest about
   total time but **not** a clean iteration-rate signal. Tagged
   ``"wall_clock_total"``.

The chosen branch is recorded on :class:`CellStats` as ``step_time_source``
and persisted in matrix.json so confound classification can refuse to
compare cells whose timing came from different fallback levels (mixing
"per_step" against "wall_clock_total" is comparing different kinds of
numbers; see :func:`aorta.triage.confound.classify`).

Each trial is sorted into one of three verdicts via the shared
:func:`aorta.run.results.trial_verdict` predicate (issue #230):

* ``pass`` -- ran clean.
* ``fail`` -- the event of interest manifested (``passed=False`` from a
  workload that actually ran, or a probe classifier ``fail``).
* ``error`` -- the trial never validly ran: an infra crash
  (``infrastructure_failed`` / ``workload_setup_failed``) or a probe
  classifier ``error`` (launch failure, timeout with no recognised
  hang). Counted in ``error_count``.

The cell's ``failure_rate`` (event rate) is ``failed / (passed + failed)``
-- **error trials are excluded from the denominator** so an infra flake
doesn't make the bug look rarer or more reproducible than it is.
``error_rate`` (``error / trials``) is surfaced separately.

The aggregator does NOT inspect *why* a fail happened (NaN vs corruption
vs divergence); that's :class:`WorkloadResult` / ``exit_status`` territory
and is preserved separately via the ``exit_status_counts`` histogram so
callers can distinguish e.g. ``workload_failed`` (run() returned
passed=False) from ``workload_setup_failed`` (setup() raised, never
reached the measurement) from ``infrastructure_failed`` (construction or
run() itself raised) when triaging.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from aorta.run.results import trial_verdict

StepTimeSource = Literal["per_step", "elapsed_per_iter", "wall_clock_total", "missing"]

# Issue #230: a cell whose error_rate (infra-error trials / total) meets
# this threshold is flagged "unreliable" in matrix.md -- its event-rate
# is computed over too few valid trials to trust. Same spirit as the
# speed-confound flag: a loud advisory, not a hard gate. Module-level
# (not recipe-configurable) for now; revisit if operators need to tune it.
UNRELIABLE_ERROR_RATE_THRESHOLD = 0.10

# Per-trial outcome enum. Snake-case so the JSON key, the Confound tag, and
# the matrix.md / terminal display string are all the same string -- avoids
# a translation layer that drifts between renderers.
OUTCOME_DID_NOT_RUN = "did_not_run"
OUTCOME_CRASHED_AFTER_ITERATIONS = "crashed_after_iterations"
OUTCOME_COMPLETED = "completed"
OUTCOME_UNKNOWN = "unknown"

# Lower index == higher fidelity. The cell-level ``step_time_source`` is the
# WORST source any trial in the cell actually contributed samples from -- if
# even one trial fell back to ``wall_clock_total``, the cell's mean folds
# setup/teardown for at least one trial and we can't honestly claim per-step
# fidelity for the aggregate. ``missing`` is reported only when no trial
# produced any timing data at all.
_SOURCE_RANK: dict[str, int] = {
    "per_step": 0,
    "elapsed_per_iter": 1,
    "wall_clock_total": 2,
    "missing": 3,
}


@dataclass(frozen=True)
class CellStats:
    """Aggregated statistics for one matrix cell.

    ``error`` is non-None when the whole cell failed (docker pull failure,
    environment resolve error, etc.) -- the matrix row is preserved so the
    matrix is complete but all numeric fields are zero / NaN.

    ``step_times_ms`` is the concatenation of every trial's step-time
    samples. Kept on the dataclass so matrix.json can embed the raw series
    for downstream analysis; matrix.md shows only the mean. The min / max /
    p50 / p90 / p99 fields summarise this series for the matrix.json
    consumer without forcing them to recompute.

    ``exit_status_counts`` is a histogram keyed by ``TrialResult.exit_status``
    (``"ok"``, ``"workload_failed"``, ``"workload_setup_failed"``,
    ``"infrastructure_failed"``, ...). It lets matrix.json consumers
    distinguish failure modes that ``failure_rate`` alone collapses
    into a single number. ``workload_setup_failed`` specifically calls
    out trials where ``workload.setup()`` raised, so a row of all-setup-
    failures can't be misread as "the bug reproduces 100 %" -- the
    workload never got far enough to test the bug.

    ``step_time_source`` records which branch of the fallback ladder
    populated ``step_times_ms`` -- ``"per_step"``, ``"elapsed_per_iter"``,
    ``"wall_clock_total"``, or ``"missing"``. When trials in the same cell
    used different branches the cell-level value is the worst (lowest
    fidelity) source any trial fell back to. Confound classification reads
    this field to refuse cell-vs-baseline ratios whose numerators and
    denominators were derived from incomparable signals (issue #160 /
    Sonbol's review on PR #160).

    ``failure_hints`` is the deduplicated list of one-line ``hint`` strings
    pulled from each trial's ``result["failure_details"][*]["hint"]``, paired
    with how many trials in the cell emitted that exact hint. The aggregator
    treats the hint as opaque text -- the workload owns the contract.
    Trials with no ``failure_details`` or no ``hint`` key contribute nothing.
    Order of appearance is preserved (first hint seen comes first) so the
    matrix.md renderer is stable across runs with the same trial order.

    ``outcome_counts`` is a histogram across the cell's trials over the
    platform-level outcome enum (``did_not_run``,
    ``crashed_after_iterations``, ``completed``, ``unknown``). Always
    populated (one entry per trial) for non-error cells: legacy
    success-path trials land in ``unknown``; trials that the platform-
    side inference (``_looks_like_did_not_run``) flags land in
    ``did_not_run`` even when the workload doesn't speak the new
    ``main_work_started`` contract; new-contract trials hit the explicit
    branches. Error cells (``error is not None``) report ``{}`` because
    no trials were aggregated.

    ``executed_iter_min`` / ``executed_iter_max`` summarise how far the
    cell's trials actually got; ``configured_iters`` records the
    workload-reported denominator. ``iters_display`` is the pre-rendered
    cell value for matrix.md's "Iters" column (e.g. ``"0/50"``,
    ``"199..200/200"``, ``"?/?"`` if trials disagreed on the configured
    count, or ``"—"`` when the workload didn't track iterations).

    ``workload_config`` is the per-cell ``Request.config_overrides`` dict
    (the merge of ``recipe.workload_config`` and ``cell.workload_config``,
    cell wins on collision). Persisted in matrix.json so the report
    renderer can surface workload-knob differences (e.g.
    ``shampoo_api=old``) in the matrix.md ``Config`` column. Empty dict
    when the recipe omits the field for every cell -- behaviourally
    identical to the pre-B2.2 schema.
    """

    name: str
    mitigations: tuple[str, ...]
    environment: str
    extra_env: dict[str, str]
    resolved_env_vars: dict[str, str]
    trials: int
    passed_count: int
    failed_count: int
    mean_step_time_ms: float
    std_step_time_ms: float
    min_step_time_ms: float
    max_step_time_ms: float
    p50_step_time_ms: float
    p90_step_time_ms: float
    p99_step_time_ms: float
    mean_wall_clock_sec: float
    exit_status_counts: dict[str, int] = field(default_factory=dict)
    step_times_ms: list[float] = field(default_factory=list)
    trial_paths: list[str] = field(default_factory=list)
    error: str | None = None
    step_time_source: StepTimeSource = "missing"
    failure_hints: list[tuple[str, int]] = field(default_factory=list)
    outcome_counts: dict[str, int] = field(default_factory=dict)
    executed_iter_min: int | None = None
    executed_iter_max: int | None = None
    configured_iters: int | None = None
    iters_display: str = "—"
    workload_config: dict[str, Any] = field(default_factory=dict)
    # Phase 2 (issue #188): the highest-frequency Tier 4 / custom
    # detector IDs across the cell's trials. ``None`` when no trial
    # in the cell fired anything (or when ``result.json`` doesn't
    # have the Phase-2 keys, e.g. triage-mode cells). The matrix.md
    # renderer hides the ``Top failure`` / ``Top warn`` columns when
    # NO cell carries data — so triage-mode runs stay byte-equivalent.
    top_failure_detector_id: str | None = None
    top_warn_detector_id: str | None = None
    # Issue #232: human-readable stop_after outcome for the cell, e.g.
    # "stopped early: 3 fail event(s) in 24 trial(s)" vs
    # "cap reached: 0 fail event(s) in 160 trial(s)".
    # ``None`` for cells with no stop_after rule, and also for cells that
    # errored before a note could be computed. The matrix renderer shows
    # the "Stop after" column whenever the recipe carries a stop_after
    # rule (config-gated, not note-gated, so an all-errored run still
    # surfaces it); cells without a note render "—". Legacy runs with no
    # stop_after rule stay byte-equivalent.
    stop_after_note: str | None = None
    # Issue #230: trials that never validly ran (infra crash / probe
    # ``error`` verdict). Excluded from the ``failure_rate`` denominator;
    # surfaced via ``error_rate`` and the matrix.md "Errors" column.
    error_count: int = 0

    @property
    def failure_rate(self) -> float:
        """Event rate: ``failed / (passed + failed)``. 0.0 when no valid trials.

        Issue #230: ``error`` trials (infra crash, probe ``error`` verdict)
        are **excluded from the denominator** -- a bug-rate of "2/128" is
        meaningless if 40 of those 128 were infra errors that never reached
        the code path under test. For a cell with no error trials this is
        identical to ``failed / trials`` (the pre-#230 definition), so
        legacy matrices are unchanged.

        This is *not* a NaN-specific rate -- :class:`WorkloadResult.failure_count`
        is generic (NaN / corruption / divergence). Callers that need to
        distinguish failure modes should read ``exit_status_counts``;
        callers that need the error share should read :attr:`error_rate`.
        """
        valid = self.passed_count + self.failed_count
        if valid == 0:
            return 0.0
        return self.failed_count / valid

    @property
    def error_rate(self) -> float:
        """Fraction of ALL trials that errored (infra, no valid observation).

        ``error_count / trials``; 0.0 for an empty cell. Denominator is the
        *total* trial count (not valid trials) -- this is the share of the
        cell's runs that were wasted on infrastructure breakage.
        """
        if self.trials == 0:
            return 0.0
        return self.error_count / self.trials

    @property
    def is_unreliable(self) -> bool:
        """True when the cell's error rate makes its event rate untrustworthy.

        Fires when ``error_rate >= UNRELIABLE_ERROR_RATE_THRESHOLD`` (and at
        least one trial errored). Whole-cell error rows (``error is not
        None``) are not flagged here -- they already render ``n/a``.
        """
        return self.error is None and self.error_count > 0 and (
            self.error_rate >= UNRELIABLE_ERROR_RATE_THRESHOLD
        )


def _step_times_from_trial(trial: Any, effective_steps: int) -> tuple[list[float], StepTimeSource]:
    """Pull per-step times from a trial result, applying the fallback ladder.

    Returns ``(times, source)`` where ``source`` names which branch fired:
    ``"per_step"``, ``"elapsed_per_iter"``, ``"wall_clock_total"``, or
    ``"missing"`` (the trial produced no usable timing). The aggregator
    uses ``source`` to compute the cell-level ``step_time_source`` so
    downstream confound detection can reject incomparable comparisons.

    When the workload explicitly reports ``main_work_started=False``
    OR the platform-observable did-not-run pattern fires
    (``_looks_like_did_not_run``), the wall-clock fallback is
    suppressed: dividing setup-only wall clock by configured steps
    would produce a number that looks like iteration timing but is
    actually import-time variance (issue #173). The per-step /
    elapsed-per-iter branches are still consulted first because a
    workload that *did* surface those signals before deciding it
    didn't really run is telling us something we should not throw
    away.
    """
    result = getattr(trial, "result", None)
    if isinstance(result, dict):
        times = result.get("step_times_ms")
        if isinstance(times, list) and times:
            cleaned = [float(t) for t in times if isinstance(t, (int, float))]
            if cleaned:
                return cleaned, "per_step"
        iters = result.get("total_iterations")
        elapsed = result.get("elapsed_sec")
        if (
            isinstance(iters, int)
            and iters > 0
            and isinstance(elapsed, (int, float))
            and elapsed > 0
        ):
            return [float(elapsed) / iters * 1000.0], "elapsed_per_iter"
        if result.get("main_work_started") is False or _looks_like_did_not_run(trial, result):
            return [], "missing"
    wall = getattr(trial, "wall_clock_sec", 0.0) or 0.0
    if wall > 0 and effective_steps > 0:
        return [float(wall) / effective_steps * 1000.0], "wall_clock_total"
    return [], "missing"


def _trial_outcome(trial: Any) -> str:
    """Classify a trial against the platform-level outcome enum.

    Two signal sources, in priority order:

    1. **Explicit contract** -- the workload populated
       ``main_work_started`` / ``executed_iterations`` /
       ``configured_iterations`` on its ``WorkloadResult``. The
       workload's report always wins.
    2. **Platform inference** -- when ``main_work_started`` is unset,
       inspect observable signals: a trial that exited with a
       non-``ok`` ``exit_status``, produced no per-step times, and
       reports ``elapsed_sec == 0`` is one that crashed before
       measuring any iteration -- "workload didn't run" in the
       sense the matrix.md reader cares about. Issue #173's problem
       statement: "the platform detects 'workload didn't run' even
       when the workload is silent about why." Demo case:
       ``recom_repro`` ImportError trials carry exactly this
       signature (passed=false, step_times_ms=[], elapsed_sec=0.0,
       exit_status="workload_failed").

    Returns ``OUTCOME_UNKNOWN`` for legacy success-path trials (no
    explicit contract, no inference triggered) -- they continue to
    render exactly as today.
    """
    result = getattr(trial, "result", None)
    if not isinstance(result, dict):
        return OUTCOME_UNKNOWN
    started = result.get("main_work_started")
    if started is False:
        return OUTCOME_DID_NOT_RUN
    if started is True:
        executed = result.get("executed_iterations")
        configured = result.get("configured_iterations")
        if not isinstance(executed, int) or not isinstance(configured, int):
            return OUTCOME_UNKNOWN
        if executed >= configured:
            return OUTCOME_COMPLETED
        if result.get("passed") is False:
            return OUTCOME_CRASHED_AFTER_ITERATIONS
        return OUTCOME_UNKNOWN
    # main_work_started is None -- platform inference branch.
    if _looks_like_did_not_run(trial, result):
        return OUTCOME_DID_NOT_RUN
    return OUTCOME_UNKNOWN


def _looks_like_did_not_run(trial: Any, result: dict) -> bool:
    """Platform-observable did-not-run detector for legacy workloads.

    Fires when ALL three signals agree:

    * ``exit_status`` is set and is not ``"ok"`` (the trial actually
      failed at the system level -- not just NaN-flagged).
    * ``step_times_ms`` is missing or empty (the workload never
      measured an iteration).
    * ``elapsed_sec`` is missing or 0 (no measured runtime either).

    The conjunction is deliberately strict so legacy perf workloads
    that fail mid-run (and thus report partial step_times_ms or
    non-zero elapsed_sec) keep their existing ``unknown`` outcome and
    flow through the wall_clock_total fallback as today. Only the
    "exited without producing any iteration data" pattern triggers
    inferred did_not_run.

    ``total_iterations`` is intentionally NOT consulted: workloads in
    the wild (e.g. recom_repro) populate it with the *configured*
    count even on setup-time crash, so trusting it would suppress the
    very signal we want to surface.
    """
    exit_status = getattr(trial, "exit_status", None)
    if exit_status is None or exit_status == "ok":
        return False
    step_times = result.get("step_times_ms")
    if isinstance(step_times, list) and any(
        isinstance(t, (int, float)) and t > 0 for t in step_times
    ):
        return False
    elapsed = result.get("elapsed_sec")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        return False
    return True


def _aggregate_iter_counts(
    trials: list[Any],
) -> tuple[int | None, int | None, int | None, str]:
    """Reduce per-trial iteration counts into the cell-level summary.

    Returns ``(executed_min, executed_max, configured, display)``.

    ``display`` rendering rules (matches issue #173 §"CellStats aggregation"):

    * Workload doesn't populate iteration fields (no trial has
      ``configured_iterations``): all four return values are ``None`` /
      ``"—"``. The matrix.md renderer hides the column entirely if
      *every* cell ends up here.
    * Trials disagree on ``configured_iterations``: ``"?/?"``. Defensive
      -- shouldn't happen in practice (recipe pins steps per cell), but
      surfaces the contradiction instead of silently picking one value.
    * Configured is known but at least one trial has
      ``executed_iterations is None``: ``"—/<configured>"``. The budget
      is real and worth surfacing; the missing executed count is the
      honest part. Renders as a visible row instead of hiding the cell.
    * All trials executed the same count: ``"<N>/<configured>"``.
    * Trials executed different counts: ``"<min>..<max>/<configured>"``.
    """
    configured_values: list[int] = []
    executed_values: list[int | None] = []
    for trial in trials:
        result = getattr(trial, "result", None)
        if not isinstance(result, dict):
            executed_values.append(None)
            continue
        cfg = result.get("configured_iterations")
        if isinstance(cfg, int):
            configured_values.append(cfg)
        executed = result.get("executed_iterations")
        executed_values.append(executed if isinstance(executed, int) else None)

    if not configured_values:
        return None, None, None, "—"

    distinct_configured = set(configured_values)
    configured = configured_values[0] if len(distinct_configured) == 1 else None
    if configured is None:
        return None, None, None, "?/?"

    populated_executed = [e for e in executed_values if e is not None]
    if len(populated_executed) != len(executed_values) or not populated_executed:
        return None, None, configured, f"—/{configured}"

    exec_min = min(populated_executed)
    exec_max = max(populated_executed)
    if exec_min == exec_max:
        display = f"{exec_min}/{configured}"
    else:
        display = f"{exec_min}..{exec_max}/{configured}"
    return exec_min, exec_max, configured, display


def _reduce_step_time_sources(sources: list[StepTimeSource]) -> StepTimeSource:
    """Pick the worst (lowest-fidelity) source any contributing trial used.

    Trials that produced no samples (``"missing"``) are filtered out so a
    single setup-crashing trial in an otherwise-healthy cell does not poison
    the cell's source label. If every trial was ``"missing"`` (or ``sources``
    is empty), the cell is reported as ``"missing"``.
    """
    contributing = [s for s in sources if s != "missing"]
    if not contributing:
        return "missing"
    return max(contributing, key=lambda s: _SOURCE_RANK.get(s, 99))


def _collect_trial_hints(trial: Any) -> list[str]:
    """Return non-empty ``hint`` strings from a trial's ``failure_details``.

    Only the explicit ``hint`` field is considered. Status / returncode /
    stderr tails are deliberately ignored -- the renderer must not
    paraphrase failure causes from logs, only surface the workload's own
    one-line hint when it chose to provide one.
    """
    result = getattr(trial, "result", None)
    if not isinstance(result, dict):
        return []
    details = result.get("failure_details")
    if not isinstance(details, list):
        return []
    hints: list[str] = []
    for entry in details:
        if not isinstance(entry, dict):
            continue
        hint = entry.get("hint")
        if isinstance(hint, str) and hint:
            hints.append(hint)
    return hints


def _aggregate_failure_hints(trials: list[Any]) -> list[tuple[str, int]]:
    """Deduplicate hints across a cell's trials, counting *trials that emitted*.

    Each trial may produce zero or more hint strings (one per
    ``failure_details`` entry that carries a ``hint``). The count is
    the number of distinct *trials* that emitted a given hint, NOT the
    number of total occurrences -- if a single trial emits the same
    hint twice it counts as 1, not 2. This matches the matrix.md
    rendering ``({count}/{cell.trials} trials)`` and keeps the count
    bounded by ``len(trials)`` so the fraction is never impossible.

    Order is the order each distinct hint string was first encountered
    while walking trials in input order.
    """
    counts: Counter[str] = Counter()
    order: list[str] = []
    for trial in trials:
        # Dedupe within the trial before counting -- one trial can
        # legitimately have multiple ``failure_details`` entries (e.g.
        # multi-phase workloads) and they may share a hint.
        seen_in_trial: set[str] = set()
        for hint in _collect_trial_hints(trial):
            if hint in seen_in_trial:
                continue
            seen_in_trial.add(hint)
            if hint not in counts:
                order.append(hint)
            counts[hint] += 1
    return [(hint, counts[hint]) for hint in order]


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    data = sorted(samples)
    k = (len(data) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] + (data[c] - data[f]) * (k - f)


def aggregate_cell(
    name: str,
    mitigations: tuple[str, ...],
    environment: str,
    extra_env: dict[str, str],
    resolved_env_vars: dict[str, str],
    trials: list[Any],
    effective_steps: int,
    trial_paths: list[str] | None = None,
    error: str | None = None,
    workload_config: dict[str, Any] | None = None,
    stop_after_note: str | None = None,
) -> CellStats:
    """Aggregate a list of TrialResult-shaped objects into a :class:`CellStats`.

    Any object with ``exit_status``, ``wall_clock_sec``, and ``result`` (dict)
    is accepted -- the aggregator does not import :class:`aorta.run.TrialResult`
    so tests can pass plain dataclasses or ``SimpleNamespace`` stand-ins.

    Args:
        name: Cell name from the recipe.
        mitigations: Tuple of mitigation names applied.
        environment: Resolved environment name (possibly ``_inline_<hash>``).
        extra_env: Ad-hoc overrides from the cell (for audit, recorded as-is).
        resolved_env_vars: Final env-var set applied to the trials (union of
            mitigation bundles + ``extra_env``).
        trials: List of trial results. Empty list is allowed for error cells.
        effective_steps: The per-trial step count the cell was configured
            with; used for the step-time fallback when the workload did not
            surface per-step times.
        trial_paths: Optional list of filesystem paths to per-trial JSON
            files; recorded in matrix.json.
        error: Cell-level error message; when set, the cell is marked as an
            error row and all numeric aggregates are forced to zero / 0.0.
        workload_config: Per-cell merged ``workload_config`` dict (recipe
            scope union cell scope, cell wins on collision). Persisted on
            the returned :class:`CellStats` so matrix.md and matrix.json
            can surface workload-knob differences across cells. ``None``
            and ``{}`` both collapse to an empty dict on the result.

    Returns:
        :class:`CellStats` populated from the trials.
    """
    workload_config = dict(workload_config or {})
    if error is not None:
        return CellStats(
            name=name,
            mitigations=mitigations,
            environment=environment,
            extra_env=dict(extra_env),
            resolved_env_vars=dict(resolved_env_vars),
            trials=len(trials),
            passed_count=0,
            failed_count=len(trials),
            mean_step_time_ms=0.0,
            std_step_time_ms=0.0,
            min_step_time_ms=0.0,
            max_step_time_ms=0.0,
            p50_step_time_ms=0.0,
            p90_step_time_ms=0.0,
            p99_step_time_ms=0.0,
            mean_wall_clock_sec=0.0,
            exit_status_counts={},
            step_times_ms=[],
            trial_paths=list(trial_paths or []),
            error=error,
            step_time_source="missing",
            failure_hints=_aggregate_failure_hints(trials),
            outcome_counts={},
            executed_iter_min=None,
            executed_iter_max=None,
            configured_iters=None,
            iters_display="—",
            workload_config=workload_config,
            top_failure_detector_id=None,
            top_warn_detector_id=None,
            stop_after_note=stop_after_note,
            error_count=0,
        )

    trial_count = len(trials)
    # Three-way split (issue #230): pass / fail / error. ``failed`` is
    # genuine failures only; infra errors land in ``errored`` and are
    # excluded from the event-rate denominator (see CellStats.failure_rate).
    verdicts = [trial_verdict(t) for t in trials]
    passed = sum(1 for v in verdicts if v == "pass")
    errored = sum(1 for v in verdicts if v == "error")
    failed = trial_count - passed - errored

    all_step_times: list[float] = []
    wall_clocks: list[float] = []
    status_counter: Counter[str] = Counter()
    trial_sources: list[StepTimeSource] = []
    # ``outcome_counts`` is always populated (one entry per trial). Legacy
    # success-path trials classify as ``OUTCOME_UNKNOWN``; trials that
    # platform-inference flags fall under ``OUTCOME_DID_NOT_RUN``; new-
    # contract trials hit the explicit branches. The downstream
    # "did_not_run legend" gate in output.py is on the rendered confound
    # tag (not on this dict's emptiness), and ``is_did_not_run_cell``
    # checks both ``set(counts) == {DID_NOT_RUN}`` AND the count covers
    # every trial -- so an all-unknown legacy cell never false-flags as
    # did_not_run.
    outcome_counter: Counter[str] = Counter()
    for trial in trials:
        times, source = _step_times_from_trial(trial, effective_steps)
        all_step_times.extend(times)
        trial_sources.append(source)
        wall = getattr(trial, "wall_clock_sec", 0.0) or 0.0
        wall_clocks.append(float(wall))
        # Histogram by raw exit_status so callers can distinguish e.g.
        # "workload_failed" (the workload's run() returned a failed
        # WorkloadResult), "workload_setup_failed" (setup() raised --
        # the workload never reached the measurement), and
        # "infrastructure_failed" (B1's dispatcher caught an exception
        # around construction or run() and synthesised a passed=False
        # WorkloadResult so the cell still has a row).
        # ``aorta.run.dispatcher`` populates a WorkloadResult in every
        # case; the distinction is purely the exit_status value.
        # Falling back to "unknown" keeps the histogram total == trial_count
        # even for stand-in trial objects that omit the attribute.
        status = getattr(trial, "exit_status", None) or "unknown"
        status_counter[str(status)] += 1
        outcome_counter[_trial_outcome(trial)] += 1

    cell_source = _reduce_step_time_sources(trial_sources)
    exec_min, exec_max, configured_iters, iters_display = _aggregate_iter_counts(trials)
    top_failure_id, top_warn_id = _aggregate_top_detector_ids(trials, trial_paths or [])

    if all_step_times:
        mean_step = float(statistics.fmean(all_step_times))
        std_step = float(statistics.pstdev(all_step_times)) if len(all_step_times) > 1 else 0.0
        min_step = float(min(all_step_times))
        max_step = float(max(all_step_times))
        p50 = _percentile(all_step_times, 0.50)
        p90 = _percentile(all_step_times, 0.90)
        p99 = _percentile(all_step_times, 0.99)
    else:
        mean_step = std_step = min_step = max_step = p50 = p90 = p99 = 0.0

    mean_wall = float(statistics.fmean(wall_clocks)) if wall_clocks else 0.0

    return CellStats(
        name=name,
        mitigations=mitigations,
        environment=environment,
        extra_env=dict(extra_env),
        resolved_env_vars=dict(resolved_env_vars),
        trials=trial_count,
        passed_count=passed,
        failed_count=failed,
        mean_step_time_ms=mean_step,
        std_step_time_ms=std_step,
        min_step_time_ms=min_step,
        max_step_time_ms=max_step,
        p50_step_time_ms=p50,
        p90_step_time_ms=p90,
        p99_step_time_ms=p99,
        mean_wall_clock_sec=mean_wall,
        exit_status_counts=dict(status_counter),
        step_times_ms=all_step_times,
        trial_paths=list(trial_paths or []),
        error=None,
        step_time_source=cell_source,
        failure_hints=_aggregate_failure_hints(trials),
        outcome_counts=dict(outcome_counter),
        executed_iter_min=exec_min,
        executed_iter_max=exec_max,
        configured_iters=configured_iters,
        iters_display=iters_display,
        workload_config=workload_config,
        top_failure_detector_id=top_failure_id,
        top_warn_detector_id=top_warn_id,
        stop_after_note=stop_after_note,
        error_count=errored,
    )


def _aggregate_top_detector_ids(
    trials: list[Any],
    trial_paths: list[str],
) -> tuple[str | None, str | None]:
    """Pick the highest-frequency detector ID from a cell's probe trials.

    Reads ``failure_detectors_fired`` / ``warn_detectors_fired`` from
    each :class:`~aorta.run.TrialResult`'s **in-memory** ``trial.result``
    payload, NOT from disk. Two source shapes are accepted (both live
    on ``trial.result``):

    1. ``result["metrics"]["failure_detectors_fired"]`` — the channel
       :class:`~aorta.workloads._subprocess.SubprocessWorkload` writes
       into when it returns a :class:`WorkloadResult`. The dispatcher
       serialises ``WorkloadResult`` into ``trial.result``, so this is
       the live-run path.
    2. ``result["failure_detectors_fired"]`` — a flat fallback some
       test trials populate directly. Accepting both keeps the
       presentation layer test-friendly.

    Triage-mode trials never set these keys; the function returns
    ``(None, None)`` and the matrix.md renderer hides the columns.

    **No on-disk fallback.** ``trial_paths`` is accepted for symmetry
    with the matrix-builder call sites but explicitly discarded
    here — see :func:`_collect_trial_paths` / resume hydration in
    ``runner.py`` for the disk-walk path; both already load the
    per-trial JSON back into ``trial.result`` so the in-memory shape
    above remains correct on resumed runs. Reading ``result.json``
    again from this presentation helper would duplicate that work and
    couple the matrix renderer to the probe artifact layout.

    "Highest frequency" is fire-count across the cell's trials, NOT
    earliest-fired. Ties resolve by encounter order so the result is
    stable across runs.
    """
    del trial_paths  # see docstring; on-disk fallback deferred
    fail_counter: Counter[str] = Counter()
    warn_counter: Counter[str] = Counter()
    fail_order: list[str] = []
    warn_order: list[str] = []
    for trial in trials:
        result = getattr(trial, "result", None)
        if not isinstance(result, dict):
            continue
        # Two source shapes:
        #   - SubprocessWorkload.run() routes the detector lists
        #     through WorkloadResult.metrics. The dispatcher then
        #     serialises WorkloadResult-as-dict into trial.result,
        #     so result["metrics"]["failure_detectors_fired"] is
        #     the in-memory channel.
        #   - Some test trials populate ``result`` directly. Accept
        #     either shape.
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        # Dedup within a SINGLE trial across the two source shapes
        # above. Production ``SubprocessWorkload`` writes to
        # ``metrics`` only, but the docstring above explicitly
        # invites the flat ``result`` shape from test fixtures —
        # any trial that populates both would double-count the
        # detector and skew ``top_failure_detector_id`` /
        # ``top_warn_detector_id``. The seen-sets are reset per
        # trial so cross-trial fire counts (the docstring's
        # "fire-count across the cell's trials") are unchanged.
        # Per Sonbol's PR #197 review.
        seen_fail_this_trial: set[str] = set()
        seen_warn_this_trial: set[str] = set()
        for source in (metrics, result):
            for entry in source.get("failure_detectors_fired", []) or []:
                if not isinstance(entry, str) or entry in seen_fail_this_trial:
                    continue
                seen_fail_this_trial.add(entry)
                if entry not in fail_counter:
                    fail_order.append(entry)
                fail_counter[entry] += 1
            for entry in source.get("warn_detectors_fired", []) or []:
                if not isinstance(entry, str) or entry in seen_warn_this_trial:
                    continue
                seen_warn_this_trial.add(entry)
                if entry not in warn_counter:
                    warn_order.append(entry)
                warn_counter[entry] += 1

    top_fail = _pick_top(fail_counter, fail_order)
    top_warn = _pick_top(warn_counter, warn_order)
    return top_fail, top_warn


def _pick_top(counter: Counter[str], order: list[str]) -> str | None:
    """Return the most-fired detector ID, ties broken by encounter order."""
    if not counter:
        return None
    best: str | None = None
    best_count = -1
    for detector_id in order:
        count = counter[detector_id]
        if count > best_count:
            best = detector_id
            best_count = count
    return best


__all__ = [
    "CellStats",
    "OUTCOME_COMPLETED",
    "OUTCOME_CRASHED_AFTER_ITERATIONS",
    "OUTCOME_DID_NOT_RUN",
    "OUTCOME_UNKNOWN",
    "StepTimeSource",
    "aggregate_cell",
]
