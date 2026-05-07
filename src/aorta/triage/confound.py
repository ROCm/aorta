"""Speed-confound detection for the triage matrix.

A mitigation that eliminates a numeric failure is only "real" if it does so
without paying a measurable iteration-time cost. The Confound column in
matrix.md captures exactly that distinction:

* ``(baseline)`` -- the cell step-time ratios are computed against.
* ``-`` (em-dash) -- mitigation appears to work without a speed cost. Trust
  the cell.
* ``speed (+N%)`` -- mitigation may be suppressing failure via slower
  iteration rather than a real fix. Verify with a profiler before drawing
  causal conclusions.
* ``no effect`` -- neither the failure rate nor the iteration time moved.
  The mitigation likely doesn't apply to this workload.
* ``n/a`` -- the baseline cell errored or produced no usable timing data,
  so no step-time ratio could be computed for *any* non-baseline cell.
  Distinct from ``-`` (which advertises a trustworthy cell): re-using the
  neutral tag here would mislead readers of matrix.md into trusting cells
  that were never compared against anything.
* ``error`` -- the whole cell failed. Row preserved so the matrix is
  complete; classification is skipped.

See §"Confound rules" of issue #151 for the exact semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from aorta.triage.matrix import CellStats
from aorta.triage.recipe import Cell, RecipeCellError

# The em-dash is what matrix.md renders; keep it here as a single constant so
# downstream consumers comparing classifications don't have to know the
# rendering. Two-char dash sequences (``--``) would break the markdown table
# alignment the issue's target format relies on.
CONFOUND_BASELINE = "(baseline)"
CONFOUND_NEUTRAL = "-"
CONFOUND_NO_EFFECT = "no effect"
CONFOUND_ERROR = "error"
# Distinct from CONFOUND_NEUTRAL: the cell could not be classified at all
# because the baseline didn't produce a usable step-time ratio. Sharing the
# neutral tag would let "ratio could not be computed" cells render as
# "mitigation works without a speed cost" in matrix.md.
CONFOUND_NA = "n/a"

ConfoundTag = Literal["(baseline)", "-", "no effect", "error", "n/a"] | str


def resolve_baseline(
    cells: Iterable[Cell],
    explicit_name: str | None,
) -> Cell:
    """Resolve the baseline cell per the rules in the recipe schema.

    Order (from issue #151 §"Schema rules"):

    1. Explicit ``confound.baseline_cell`` (validated to exist at recipe
       load time).
    2. First cell whose name starts with ``baseline-``.
    3. First cell whose mitigations == ``["none"]``.
    4. If the recipe has exactly one cell, use it.
    5. Otherwise raise :class:`RecipeCellError`.

    Raises:
        RecipeCellError: None of the rules resolves and the recipe has more
            than one cell (so confound detection can't anchor).
    """
    cells_list = list(cells)
    if not cells_list:
        raise RecipeCellError("cannot resolve baseline: recipe has no cells")

    if explicit_name is not None:
        for c in cells_list:
            if c.name == explicit_name:
                return c
        # recipe loader validates this; defensive second-chance error for
        # callers that construct a Recipe without going through the loader.
        raise RecipeCellError(
            f"confound.baseline_cell {explicit_name!r} does not match any "
            f"cell name; cells: {sorted(c.name for c in cells_list)}"
        )

    for c in cells_list:
        if c.name.startswith("baseline-"):
            return c

    for c in cells_list:
        if tuple(c.mitigations) == ("none",):
            return c

    if len(cells_list) == 1:
        return cells_list[0]

    raise RecipeCellError(
        "cannot resolve baseline cell: no cell named 'baseline-*' and no "
        "cell with mitigations == ['none']. Add one, or set "
        "confound.baseline_cell explicitly in the recipe."
    )


def classify(
    cell: CellStats,
    baseline: CellStats,
    threshold: float,
) -> tuple[ConfoundTag, float | None]:
    """Classify a cell's Confound column and return the step-time ratio.

    Args:
        cell: The cell under classification.
        baseline: The baseline cell (result of :func:`resolve_baseline`
            fed through :func:`aorta.triage.matrix.aggregate_cell`).
        threshold: ``cell.mean_step_time_ms / baseline.mean_step_time_ms``
            above this triggers a ``speed (+N%)`` flag. Pulled from
            ``ConfoundCfg.threshold`` (default 1.15).

    Returns:
        ``(tag, ratio)`` where ``tag`` is the matrix.md cell text and
        ``ratio`` is ``None`` for the baseline row or whenever a meaningful
        ratio could not be computed (baseline errored, baseline has no
        usable timing, the cell itself has no usable timing, or the cell
        and baseline derived their step-time from different fallback
        branches -- comparing per-step workload clocks against
        wall-clock-divided-by-step-count would be apples to oranges).
    """
    if cell.error is not None:
        return CONFOUND_ERROR, None

    if cell.name == baseline.name:
        return CONFOUND_BASELINE, None

    if baseline.error is not None or baseline.mean_step_time_ms <= 0:
        # Baseline unusable: we can't compute a ratio against anything, so
        # the cell is unclassifiable -- NOT trustworthy. Use CONFOUND_NA so
        # matrix.md renders "n/a" in the Confound column instead of the
        # neutral "-" tag (which advertises "mitigation works without speed
        # cost"). The runner additionally writes a top-of-file warning so
        # readers see why every non-baseline row collapsed to n/a.
        return CONFOUND_NA, None

    if cell.mean_step_time_ms <= 0 or cell.step_time_source == "missing":
        # Cell has no usable timing of its own -- the ratio would either be
        # zero (meaningless) or undefined. CONFOUND_NA preserves the row
        # without claiming the mitigation is trustworthy.
        return CONFOUND_NA, None

    if cell.step_time_source != baseline.step_time_source:
        # Mixing fidelity tiers (e.g. baseline ran a workload that emits
        # per-step times, this cell ran one that only exposes wall-clock)
        # produces a ratio between fundamentally different signals: the
        # baseline's number is iteration time only, the cell's number folds
        # setup / teardown. Refuse to label that comparison; matrix.json
        # keeps both cells' ``step_time_source`` so reviewers can see why
        # the row landed on n/a (issue #160 / Sonbol's review on PR #160).
        return CONFOUND_NA, None

    ratio = cell.mean_step_time_ms / baseline.mean_step_time_ms

    if ratio > threshold:
        # Round toward the nearest whole-percent for the UI; the raw float
        # is preserved in matrix.json for anyone who wants more precision.
        pct = round((ratio - 1.0) * 100)
        return f"speed (+{pct}%)", ratio

    # At this point the cell is NOT speed-confounded. Decide between
    # "-" (faster or equal-speed, with a measurable failure-rate drop) and
    # "no effect" (failure rate didn't drop AND no speed cost).
    if cell.failure_rate >= baseline.failure_rate:
        return CONFOUND_NO_EFFECT, ratio
    return CONFOUND_NEUTRAL, ratio


def classify_all(
    cells: list[CellStats],
    baseline_name: str,
    threshold: float,
) -> dict[str, tuple[ConfoundTag, float | None]]:
    """Apply :func:`classify` to every cell, keyed by cell name.

    Keyed by ``CellStats.name`` so the output renderer doesn't need to
    reason about ordering. The runner builds a parallel ``list[CellStats]``
    in matrix.md row order; the dict is purely for lookup convenience.
    """
    baselines = [c for c in cells if c.name == baseline_name]
    if not baselines:
        raise RecipeCellError(
            f"baseline cell {baseline_name!r} not found in aggregated cells; "
            f"this indicates a runner bug (baseline was resolved at recipe "
            f"load time but did not survive aggregation)."
        )
    baseline = baselines[0]
    return {c.name: classify(c, baseline, threshold) for c in cells}


__all__ = [
    "CONFOUND_BASELINE",
    "CONFOUND_ERROR",
    "CONFOUND_NA",
    "CONFOUND_NEUTRAL",
    "CONFOUND_NO_EFFECT",
    "ConfoundTag",
    "classify",
    "classify_all",
    "resolve_baseline",
]
