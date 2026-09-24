#!/usr/bin/env python3
"""A discrete-event reward: points for things that happened, not a blend.

Why this exists
===============
A reward that is one continuous scalar awarded at the end of an episode gives
GRPO no way to tell *which* decision earned it. An episode that stumbled onto
the resolver on its third guess and one that named it first time receive the
same number, so the gradient carries no information about the difference.

This makes the return a **sum of events that individually happened or did
not**, each attached to the step it happened on. Two episodes that reach the
same place by different routes now score differently, and the difference points
at the decisions rather than only at the outcome.

The hard constraint, and what it rules out
==========================================
**Every event must be decidable from the reply, the agent log and the archived
probe matrix, with no judgement call.** If scoring an event needs an opinion
about whether a hypothesis was "reasonable", it is not an event. Without that
rule a discrete-event reward is an unfalsifiable rubric with more knobs than the
thing it replaced.

That rule excluded several events that would otherwise be obvious:

* **hypothesis quality.** Not decidable: the log records the text and nothing
  that adjudicates it.
* **calibration** (``confidence`` against outcome). Decidable only after
  choosing a threshold, and a calibration term rewards a humble constant.
* **category correctness.** An archived matrix says which mitigation resolved a
  failure, not which autopsy category it belongs to, so a category term would
  score membership in the vocabulary rather than correctness.
* **stopping once the answer is visible.** Decidable in principle but
  structurally unreachable: ``run_agent_loop`` detects a winning cell and breaks
  *before* calling the proposer, so a policy never gets the chance to continue
  after a fix is on screen. An event that can never fire is a comment.

The point values were fixed before any policy was scored against them and are
not tuned to results; :data:`POINTS` is pinned byte for byte by a test so that
a change to one is a visible decision rather than a drift.

The event list
==============
Per step -- decided by the reply plus the ground truth
-----------------------------------------------------
``malformed_reply`` (-6.0)
    A reply the loop cannot use forfeits the whole proposal cycle and returns
    no information; priced below every well-formed reply *that spends the same
    cells*, which is an invariant a test pins rather than a number chosen by
    feel.
``resolver_named`` (+4.0)
    Naming a mitigation the archived matrix shows actually resolves the failure
    is the task, so it is the largest single award and no accumulation of
    process events can substitute for it.
``resolver_named_on_step_1`` (+2.0), ``..._on_step_2`` (+1.0)
    Each step costs a whole probe matrix, so the same answer reached sooner is
    worth more; the bonus decays to zero rather than turning negative, so a late
    correct answer is never worse than no answer at all.
A reply that sets ``stop: true`` proposes nothing: the loop stops before it
runs any name in it, so those names fire no name event and no resolver award.

``name_refuted`` (-0.5, per name)
    A named mitigation is an asserted causal hypothesis, and the matrix ran it
    and refuted it.
``name_unmeasured`` (0.0, per name)
    The matrix ran no cell for this name, so there is no fact about it to score
    and inventing one would be the judgement call this design rules out.
``name_not_offered`` (-1.0, per name)
    The consumer silently drops it, so the policy spent a proposal slot on
    something that cannot run; priced above a refuted name because it is a
    failure to read the prompt rather than a wrong guess about the failure.
``name_already_tried`` (-1.0, per name)
    Its cell already exists and its verdict is on screen, so the proposal
    re-asks a question the evidence in front of it already answered.
``verdict_correct`` / ``verdict_wrong`` (+1.0 / -1.0)
    Reading the failure correctly is a separate and independently decidable
    claim from fixing it; symmetric so that declining to read is not free.
``detector_attribution`` (graded, -1.0 to +1.0)
    The same read at the resolution it actually has: a verdict is one label but
    an attribution is a *set*. Graded by ``triage_reward.set_f1`` and mapped to
    ``2*f1 - 1``; see :func:`attribution_points` for why grading does not breach
    the decidability rule.

**In an episode the read is scored on the first step only.** Both events were
first paid on every step. A refuted step costs ``name_refuted`` plus one
``cell_spent``, less than a correct copied read, and the earliness bonus stops
at step 2, so every step of delay past the second raised the return: a policy
that reads nothing about the fix and walks through knobs that fix nothing
before the cover outscored every trained checkpoint. The claim is about the
baseline failure, and every cell shown during an episode carries the
baseline's verdict (a passing cell ends the episode first), so a later read
adds no fact; paying it again pays for one fact *k* times, which is why
``resolver_named`` is paid once. Step 1 is the single-step task byte for byte,
so an episode's read and a single completion's are the same quantity.
``score_completion`` and every one-step episode are unchanged, and no value in
``POINTS`` moved. Rejected: raising the per-step cost (a constant fitted to
beat another constant, and it taxes the genuine long search on the
unresolvable scenario), decaying the read per step (still collectable at small
*k*), averaging it (dilutes a wrong first read), and paying it on the last step
(which can be a stop the loop manufactured).

Per episode -- decided by the terminal state, at most one fires
---------------------------------------------------------------
``terminal_converged`` (+2.0)
    The only event certified by execution rather than by proposal text; smaller
    than naming the resolver because the loop, not the policy, decides when a
    cell has passed.
``terminal_unresolvable_correct`` (+4.0)
    Equal to naming a resolver, because on a scenario nothing resolves, saying
    so is the right answer. It fires only when the claim was **earned** -- see
    :func:`_classify_unresolvable_claim` -- and only when the stop was the
    **policy's** (see ``terminal_proposal_unresolved``).
``terminal_unresolvable_unearned`` (0.0)
    A correct unresolvability claim that eliminated nothing. Zero rather than
    negative because the claim is right and punishing it would make honest
    abstention unprofitable; zero rather than positive because an assertion
    that established nothing has established nothing.
``terminal_gave_up`` (-3.0)
    Abandoning a problem the matrix shows is solvable.
``terminal_proposal_unresolved`` (-3.0)
    The loop stopped because every mitigation the policy named was dropped by
    the candidate filter (unregistered, already tried, or off the allowlist) --
    the ``proposal_unresolved`` stop reason, aorta#449. Not a decision, so it is
    not adjudicated as one. Equal to ``terminal_gave_up`` deliberately: see the
    note beside the value in :data:`POINTS` for why it must not be cheaper.

Per cell -- the running cost
----------------------------
``cell_spent`` (-0.2, per probe cell)
    The GPU actually spent, priced so that a full ten-cell budget (-2.0) cannot
    outweigh naming the resolver (+4.0): cost is a tiebreaker between correct
    policies, not the objective.

Scope: the mitigation axis only
===============================
``run_agent_loop`` proposes and grows the **mitigation** axis; the diagnostic
axis is fixed by the recipe and is not an action the policy takes. So there is
no event for choosing a diagnostic and no terminal for "no axis could be
widened" -- the loop has no such stall condition. A budget stop is a
``policy_stop`` and is withheld as ``other``: it is a statement about the budget,
not about the search.

What this does NOT fix, stated because it is the next thing wrong
=================================================================
``name_refuted`` charges -0.5 for every candidate the matrix ran and refuted.
On a scenario with a resolver that is right. **On a scenario nothing resolves,
every candidate is refuted by construction**, so the only productive action --
eliminating the menu to earn ``terminal_unresolvable_correct`` -- is charged as
a string of mistakes plus its cells. The correct behaviour is no longer
dominated by a free assertion (the earned/unearned split fixed that), but the
earned path is not yet *attractive*. Pricing it properly changes an event's
meaning globally rather than a precondition, so it is left for a separate,
separately measured change.

Deliberate overlaps
===================
``resolver_named`` and ``terminal_converged`` co-fire on a converging episode,
by construction. They are not one fact counted twice: the first is decidable
from the *proposal text* against the archived matrix and is reachable by a
single reply with no episode at all; the second only from an episode that ran.

``name_refuted`` and ``cell_spent`` overlap while the diagnostic axis is one
wide, where one name is one cell. They answer different questions -- how many
claims were wrong, and how much GPU was burned -- and come apart on any recipe
with a wider diagnostic axis.

There is no floor
=================
A clamp at 0.0 would let several genuinely different bad policies tie, and GRPO
sees no advantage between tied samples. A malformed reply has its own explicit
negative event instead, so the ordering a floor would protect is produced by
the point values.

Usage
-----

    python examples/rl/event_reward.py --points
    python examples/rl/event_reward.py --matrix <archived-probe-run> \\
        --episode <agent-run-dir> --offered name_a,name_b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.agent.llm import PROBE_CATEGORIES, AgentStep
from aorta.agent.state import read_trial_results, winning_mitigation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proposal_reward import REQUIRED_KEYS  # noqa: E402
from triage_reward import Label, find_probe_cells, label_trials, set_f1  # noqa: E402

# ---------------------------------------------------------------------------
# The point table. See the module docstring for each value's justification.
# Do not fit these to results.
# ---------------------------------------------------------------------------

POINTS: dict[str, float] = {
    # per step
    "malformed_reply": -6.0,
    "resolver_named": 4.0,
    "resolver_named_on_step_1": 2.0,
    "resolver_named_on_step_2": 1.0,
    "name_refuted": -0.5,
    "name_unmeasured": 0.0,
    "name_not_offered": -1.0,
    "name_already_tried": -1.0,
    "verdict_correct": 1.0,
    "verdict_wrong": -1.0,
    # Graded, and the only graded event. See `attribution_points`.
    "detector_attribution": 1.0,
    # per episode, at most one
    "terminal_converged": 2.0,
    "terminal_unresolvable_correct": 4.0,
    "terminal_unresolvable_unearned": 0.0,
    "terminal_gave_up": -3.0,
    # The ending that is not the policy's: the candidate filter dropped every
    # name it proposed, so the loop had nothing to run and stopped. Valued at
    # `terminal_gave_up`, and the equality is the point -- what this event buys
    # is ATTRIBUTION, not a new number. The episode ended with the failure
    # unresolved and nothing established, which is what giving up is worth;
    # naming it separately stops that ending being read as a conclusion.
    #
    # It must not be CHEAPER than giving up, or the reward invents an exploit:
    # the cheapest way out of an episode becomes one already-tried name (-1.0)
    # instead of an honest empty proposal (-3.0). At equality the stall costs
    # -1.0 more than the quit, so nothing is bought by stalling. A test pins it.
    "terminal_proposal_unresolved": -3.0,
    # per probe cell
    "cell_spent": -0.2,
}


def attribution_points(
    cited: Iterable[str],
    actual: Iterable[str],
    points: Mapping[str, float] = POINTS,
) -> float:
    """The graded attribution award: ``points * (2 * set_f1 - 1)``.

    The one graded event, and the exception needs its justification. The rule
    is that an event must be decidable **with no judgement call** -- what it
    rules out is an *opinion*, such as a threshold on a number. Set overlap
    between the detectors a reply cited and the detectors that fired is
    neither: both sets are recorded and ``set_f1`` has no free parameter.

    ``set_f1`` is imported from ``triage_reward``, not reimplemented, so the
    triage reward and this one cannot disagree about the empty cases.

    Mapped to ``[-1, +1]`` rather than ``[0, 1]`` to match the
    ``verdict_correct`` / ``verdict_wrong`` pair it grades: on ``[0, 1]`` a
    reply that cites nothing scores 0.0, the same as not being scored, and
    "declining to read the evidence is free" is the defect the symmetry exists
    to remove.

    One inherited hole, adopted rather than re-decided: ``set_f1`` returns 1.0
    when both sets are empty, so on a cell where nothing fired, citing nothing
    earns the full +1.0. That is ``set_f1``'s documented convention and the
    binary pair has the same hole (a correct ``pass`` costs no reading either);
    forking the convention here would be worse than the hole.
    """
    return points["detector_attribution"] * (2 * set_f1(cited, actual) - 1)


#: Terminal classifications that score nothing and say why. ``unobserved`` is
#: the single-reply case -- one reply does not observe how a run ended -- and
#: ``other`` covers baseline_pass, a budget or walltime stop, approval, a
#: registry error and a crash, none of which is a statement about the search.
WITHHELD_TERMINALS = ("unobserved", "other")

#: The baseline name on both probe axes, as the harness spells it.
BASELINE = "none"
BASELINE_CELL = f"{BASELINE}-{BASELINE}"

#: The earliness bonus, by 1-based step index. An absent index has no bonus.
#: A table rather than a formula so "decays to zero and never goes negative"
#: is visible rather than asserted.
EARLINESS: dict[int, str] = {
    1: "resolver_named_on_step_1",
    2: "resolver_named_on_step_2",
}


# ---------------------------------------------------------------------------
# What an event is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One thing that demonstrably happened, and what it is worth.

    ``why`` is the decidable fact that fired it, in the words of the artifact it
    was read from, so a score can be audited back to the log line or cell that
    produced it without re-running anything.
    """

    name: str
    points: float
    why: str
    step: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "points": self.points, "why": self.why, "step": self.step}


@dataclass
class EventScore:
    """The events one episode fired, and their sum."""

    events: list[Event] = field(default_factory=list)
    terminal: str = "unobserved"
    cells: int = 0
    withheld: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(event.points for event in self.events)

    def count(self, name: str) -> int:
        return sum(1 for event in self.events if event.name == name)

    def fired(self, name: str) -> bool:
        return self.count(name) > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "terminal": self.terminal,
            "cells": self.cells,
            "events": [event.as_dict() for event in self.events],
            "withheld": list(self.withheld),
        }


# ---------------------------------------------------------------------------
# Ground truth: what the archived matrix knows, and what it does not
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSignature:
    """Everything about one cell that the *policy* is shown.

    What ``loop._read_cell_summaries`` puts in front of the proposer, field for
    field: the aggregated verdict, the exit code of the evidence trial, the
    union of failure and warn detector IDs across trials, and the *keys* of the
    capture dict. The policy sees this and nothing else -- no stdout, no
    stderr, no ``error_detectors_fired``.

    Every component is categorical, deliberately: no quantity of bytes can move
    any field, and capture *keys* rather than values because a bigger captured
    string is not a new fact.
    """

    verdict: str | None
    exit_code: int | None
    detectors: frozenset[str]
    capture_keys: frozenset[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "detectors": sorted(self.detectors),
            "capture_keys": sorted(self.capture_keys),
        }


def signature_from_trials(docs: list[dict[str, Any]], verdict: str | None) -> CellSignature:
    """``loop._read_cell_summaries`` for one cell, reduced the same way.

    A replication rather than a call because the loop's version walks a run
    directory and builds summaries for a whole grid, while this scores one
    archived cell. The reductions are the loop's -- detectors unioned across
    trials, exit code from the first non-passing trial and otherwise the first
    -- so "this is what the policy saw" is checkable line against line.
    """
    detectors: set[str] = set()
    for doc in docs:
        detectors.update(str(d) for d in doc.get("failure_detectors_fired") or [])
        detectors.update(str(d) for d in doc.get("warn_detectors_fired") or [])
    evidence = next(
        (d for d in docs if d.get("verdict") not in (None, "pass")),
        docs[0] if docs else {},
    )
    capture = evidence.get("capture") or {}
    return CellSignature(
        verdict=verdict,
        exit_code=evidence.get("exit_code"),
        detectors=frozenset(detectors),
        capture_keys=frozenset(str(k) for k in capture),
    )


@dataclass(frozen=True)
class Resolution:
    """Which mitigations an archived matrix shows resolving the failure.

    ``resolvers`` is a set: a matrix with three passing ``{m}-none`` cells has
    three right answers, and a policy that names the second is not wrong.
    """

    scenario_id: str
    resolvers: frozenset[str]
    baseline_failed: bool
    cells: int
    source: str


@dataclass(frozen=True)
class MatrixGrid:
    """Cell verdicts from an archived probe run, and the names it never ran.

    ``resolution`` answers "which mitigations resolved it". The grid answers
    the question that has to come first for a per-name event: **which
    mitigations does the matrix have an opinion about at all.** An archived
    matrix has as many cells as someone chose to run, and a menu is usually
    larger, so some proposed names have no cell. Scoring those as wrong would
    assert a measurement nobody made -- hence ``name_unmeasured``.
    """

    verdicts: dict[str, str]
    resolution: Resolution
    #: Per cell, what the policy is shown.
    signatures: dict[str, CellSignature] = field(default_factory=dict)

    @property
    def mitigation_axis(self) -> list[str]:
        """Mitigations the grid has cells for, baseline included."""
        return sorted({cell.rsplit("-", 1)[0] for cell in self.verdicts if "-" in cell})

    @property
    def measured_mitigations(self) -> frozenset[str]:
        """Mitigations with a ``{m}-none`` cell, so the matrix can speak to them.

        Keyed on the baseline-diagnostic cell specifically, because it is the
        only cell whose verdict is attributable to the mitigation alone -- the
        rule ``state.winning_mitigation`` applies when it decides what counts
        as a win.
        """
        names = set()
        for cell in self.verdicts:
            if "-" not in cell:
                continue
            mitigation, diagnostic = cell.rsplit("-", 1)
            if diagnostic == BASELINE and mitigation != BASELINE:
                names.add(mitigation)
        return frozenset(names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": len(self.verdicts),
            "resolvers": sorted(self.resolution.resolvers),
            "baseline_failed": self.resolution.baseline_failed,
            "measured_mitigations": sorted(self.measured_mitigations),
        }


def grid_from_matrix(root: Path, scenario_id: str | None = None) -> MatrixGrid:
    """Read an archived probe run into verdicts, signatures and resolvers.

    Verdicts are recomputed through ``triage_reward.label_trials`` -- aorta's
    own resolver -- rather than read off the artifact, so this and the triage
    reward cannot disagree about what a cell said.

    A resolver is decided by ``state.winning_mitigation``, the function the
    loop itself uses, which credits a pass only on ``{m}-none``: a pass on a
    cell with a non-baseline diagnostic is not attributable to the mitigation
    alone. ``baseline_failed`` is False when the ``none-none`` cell is absent
    or unreadable -- a matrix that cannot show a failure has nothing to resolve,
    and treating the missing cell as a failure would manufacture a scenario.
    """
    verdicts: dict[str, str] = {}
    signatures: dict[str, CellSignature] = {}
    resolvers: set[str] = set()
    baseline_failed = False
    cells = find_probe_cells(root)
    for cell in cells:
        docs = read_trial_results(cell)
        if not docs:
            continue
        verdict = label_trials(docs, source=str(cell)).verdict
        verdicts[cell.name] = verdict
        signatures[cell.name] = signature_from_trials(docs, verdict)
        if cell.name == BASELINE_CELL:
            baseline_failed = verdict == "fail"
        winner = winning_mitigation(cell.name, verdict)
        if winner is not None:
            resolvers.add(winner)
    return MatrixGrid(
        verdicts=verdicts,
        resolution=Resolution(
            scenario_id=scenario_id or root.name,
            resolvers=frozenset(resolvers),
            baseline_failed=baseline_failed,
            cells=len(cells),
            source=str(root),
        ),
        signatures=signatures,
    )


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------


def admissible(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Whether the reply is a reply at all. Returns ``(parsed, reason)``.

    Checks exactly three things, each one the shipped path already enforces:
    the reply parses as a JSON object (``llm._step_from_content``), it carries
    ``proposal_reward.REQUIRED_KEYS`` at the right types, and its category is
    one ``policy.validate_step`` accepts -- ``PROBE_CATEGORIES``, not the full
    autopsy vocabulary, because the loop refuses an evidence-only category from
    a probe step and a reply the loop refuses is a reply that did nothing.

    Name-level problems are deliberately **not** admissibility failures. "You
    wrote prose", "you invented a mitigation" and "you named one that is not on
    the menu" are three different mistakes with three different fixes;
    collapsing them into one malformed verdict is the loss of granularity this
    module exists to undo, so the name-level ones are events below.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"not JSON: {exc.msg}"
    if not isinstance(obj, dict):
        return None, f"parsed as {type(obj).__name__}, not an object"
    for key, expected in REQUIRED_KEYS.items():
        if key not in obj:
            return None, f"missing required key {key!r}"
        value = obj[key]
        # bool subclasses int, so `"confidence": true` would pass an isinstance
        # check against (int, float) and has to be excluded by hand.
        if expected is not bool and isinstance(value, bool):
            return None, f"{key!r} is a bool, expected {expected}"
        if not isinstance(value, expected):
            return None, f"{key!r} is {type(value).__name__}, expected {expected}"
    if obj["category"] not in PROBE_CATEGORIES:
        return None, f"category {obj['category']!r} is not one the probe agent may claim"
    return obj, ""


# ---------------------------------------------------------------------------
# One step's events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepContext:
    """What a step's events are decided against: the menu and what was bought.

    Frozen and explicit rather than pulled off a loop object, because every
    field is something a reader has to be able to check by hand against the
    log.
    """

    offered_mitigations: frozenset[str]
    tried_mitigations: frozenset[str] = frozenset()


def _classify_name(
    name: str,
    offered: frozenset[str],
    tried: frozenset[str],
    grid: MatrixGrid | None,
) -> tuple[str, str] | None:
    """The single event one proposed name fires, or None if it fires nothing.

    Exactly one, in a fixed precedence: already tried beats not offered beats
    the matrix's opinion. A name that is both on the axis and absent from the
    current menu is one mistake -- it was bought last round, which is *why* it
    left the menu -- and charging it twice would price one error as two.
    """
    if name == BASELINE:
        # On every axis by construction; `policy.validate_step` drops it.
        return None
    if name in tried:
        return "name_already_tried", f"mitigation {name!r} already on the axis"
    if name not in offered:
        return "name_not_offered", f"mitigation {name!r} was not offered"
    if grid is None:
        return None
    if name in grid.resolution.resolvers:
        # Scored by `resolver_named` at the step level, not per name: paying
        # per name would make repeating the right answer profitable.
        return None
    if name in grid.measured_mitigations:
        return "name_refuted", f"mitigation {name!r} has a cell and it failed"
    return "name_unmeasured", f"mitigation {name!r} has no cell in the matrix"


def step_events(
    raw: str,
    context: StepContext,
    grid: MatrixGrid | None,
    *,
    step: int,
    label: Label | None = None,
    resolver_already_named: bool = False,
    points: Mapping[str, float] = POINTS,
) -> tuple[list[Event], AgentStep | None]:
    """Every event one reply fires, and the parsed step (None if inadmissible).

    ``resolver_already_named`` is how the episode scorer stops a policy banking
    ``resolver_named`` on every step by repeating the winner: the award is for
    the *first* step that names it, and the earliness bonus keys off that index.
    """
    events: list[Event] = []
    obj, reason = admissible(raw)
    if obj is None:
        events.append(Event("malformed_reply", points["malformed_reply"], reason, step))
        return events, None

    parsed = AgentStep.from_dict(obj)
    # A reply that stops proposes nothing to run. The loop takes the stop
    # before it grows the axis, so no name in a stopping reply ever becomes a
    # cell: scoring those names -- charging a refuted one or paying a resolver
    # -- would score text the environment discards, the same reason an
    # un-offered name cannot resolve anything. So a stop naming the resolver
    # earns nothing for it: the policy had the answer and declined to try it,
    # and the episode ends unresolved. `stop` is read through `from_dict`, so
    # only a genuine JSON `true` counts, exactly as in the loop.
    proposed = [] if parsed.stop else parsed.next_mitigations
    # De-duplicated: a reply naming one mitigation twice made one proposal, and
    # the loop would buy one cell for it.
    for name in dict.fromkeys(proposed):
        fired = _classify_name(
            name, context.offered_mitigations, context.tried_mitigations, grid
        )
        if fired is not None:
            name_event, why = fired
            events.append(Event(name_event, points[name_event], why, step))

    # Only names the consumer would actually run can resolve anything. A name
    # outside the menu never becomes a cell, so it cannot have fixed the
    # failure however right it looks; it was charged `name_not_offered` above,
    # and paying it here would reward an action the loop discards. The same
    # holds for a name already on the axis: the filter drops it too.
    runnable = [
        m for m in proposed
        if m in context.offered_mitigations and m not in context.tried_mitigations
    ]
    if grid is not None and not resolver_already_named:
        named = sorted(set(runnable) & set(grid.resolution.resolvers))
        if named:
            events.append(
                Event(
                    "resolver_named",
                    points["resolver_named"],
                    f"named {named[0]!r}, which the matrix shows resolves it",
                    step,
                )
            )
            bonus = EARLINESS.get(step)
            if bonus is not None:
                events.append(Event(bonus, points[bonus], f"on step {step}", step))

    if label is not None:
        # An absent verdict scores as wrong rather than being withheld:
        # defaulting it to the label's would hand every policy that declines to
        # read the evidence a free point.
        claimed = obj.get("verdict")
        correct = isinstance(claimed, str) and claimed == label.verdict
        key = "verdict_correct" if correct else "verdict_wrong"
        events.append(
            Event(key, points[key], f"claimed {claimed!r}, truth {label.verdict!r}", step)
        )
        # The graded half of the same read, alongside the pair rather than
        # replacing it: the verdict is a single label, the attribution a set.
        # An absent `detectors` key cites nothing, for the same reason.
        cited = obj.get("detectors")
        cited_list = [str(d) for d in cited] if isinstance(cited, list) else []
        events.append(
            Event(
                "detector_attribution",
                attribution_points(cited_list, label.cited_detectors, points),
                f"cited {sorted(set(cited_list))}, fired {sorted(label.cited_detectors)}",
                step,
            )
        )
    return events, parsed


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoggedStep:
    """One decision: the reply, and the axis in force when it was made."""

    n: int
    raw: str
    mitigation_axis_before: list[str]
    stop: bool


@dataclass(frozen=True)
class LoggedEpisode:
    """An episode reduced to the facts the events are decided from."""

    run_dir: Path
    steps: list[LoggedStep]
    terminal: str
    terminal_why: str
    cells: int


def _step_to_raw(event: Mapping[str, Any]) -> str:
    """Reconstruct **what the model asked for** from an ``llm_step`` record.

    The log stores the *parsed* fields rather than the wire bytes, so this
    rebuilds the object the scorer reads. ``unresolved_mitigations`` is folded
    back into ``next_mitigations``, and that is the point of reading it: the
    proposer's filter drops un-offered and already-tried names *before* the
    loop logs the step, so the logged list alone is a proposal the model did not
    make, and scoring it would credit the policy for restraint it did not show.
    The key is absent, not empty, when nothing was dropped, so a log written
    before it existed rebuilds exactly as it did then.

    It cannot recreate a malformed reply and does not need to: the loop never
    writes an ``llm_step`` for one (``_step_from_content`` turns it into a safe
    stop, and a refused category never reaches the log). So ``malformed_reply``
    is structurally zero on log-derived episodes -- a property of the log, not
    evidence about the model. Score the wire text to see it.
    """
    proposed = [str(m) for m in event.get("next_mitigations") or []]
    for name in event.get("unresolved_mitigations") or []:
        if str(name) not in proposed:
            proposed.append(str(name))
    return json.dumps(
        {
            "category": event.get("category") or "unknown",
            "hypothesis": event.get("hypothesis") or "",
            "next_mitigations": proposed,
            "confidence": float(event.get("confidence") or 0.0),
            "stop": bool(event.get("stop")),
        }
    )


def read_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "agent_log.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _classify_unresolvable_claim(
    offered: Iterable[str], tried: Iterable[str]
) -> tuple[str, str]:
    """Split a correct "nothing resolves this" claim into earned and unearned.

    The claim is right on a scenario nothing resolves; what decides the award
    is whether the *evidence* licensing it exists. Without this split the award
    paid the same as naming a resolver and cost **zero cells**, so a first-turn
    ``{"stop": true, "next_mitigations": []}`` was the maximally rewarded reply
    on every unresolvable scenario -- the award paid for an assertion, not for
    an elimination. So the claim earns ``terminal_unresolvable_correct`` only
    when no offered candidate remains untried, and scores
    ``terminal_unresolvable_unearned`` (0.0) otherwise.

    Rejected alternatives:

    * **Reduce the award.** Leaves the structure intact -- a first-turn
      assertion is still the cheapest route to whatever the award becomes --
      and any value is a constant chosen to make a number look acceptable.
    * **Withhold the terminal when unearned.** Numerically identical to 0.0,
      but it hides the case in a ``withheld`` list; an explicit event worth
      zero says "adjudicated, earned nothing", a withheld one says "not looked
      at".
    * **Pay the unearned case something small.** Reinstates a free award for an
      assertion that established nothing, differing from the defect only in
      magnitude.

    Fail-closed on a missing menu: with no menu recorded, "nothing was offered"
    and "the caller forgot to pass it" are the same value, and the award must
    not be what that silence buys.
    """
    offered_set = {str(n) for n in offered}
    tried_set = {str(n) for n in tried} - {BASELINE}
    remaining = offered_set - tried_set
    if not offered_set:
        return (
            "unresolvable_unearned",
            "no menu recorded: whether the claim was earned cannot be decided",
        )
    if not tried_set:
        return (
            "unresolvable_unearned",
            "claimed nothing resolves it without trying a single candidate",
        )
    if remaining:
        return (
            "unresolvable_unearned",
            f"claimed nothing resolves it with {len(remaining)} offered "
            "candidates never tried: the claim is right and nothing established it",
        )
    return (
        "unresolvable_correct",
        f"claimed nothing resolves it having tried and refuted all "
        f"{len(tried_set)} offered candidates",
    )


def classify_stop(
    *,
    stop_reason: str | None,
    proposed_nothing: bool,
    unresolvable: bool,
    resolver_named: bool,
    offered: Iterable[str],
    tried: Iterable[str],
    unresolved: Iterable[str] = (),
) -> tuple[str, str]:
    """The terminal for a ``search_stopped`` ending, shared by both classifiers.

    One function because two callers decide it -- ``episode_env`` in flight and
    :func:`episode_from_log` afterwards from the log -- and two copies of a
    precedence rule are how a replay and a run come to disagree.

    ``proposal_unresolved`` is checked FIRST, and it is read off the loop's own
    stop reason rather than re-derived: it is the one ending that is not the
    policy's, and every branch below attributes a decision. The empty list at
    that stop is the filter's, so it must not be adjudicated as the claim
    "nothing resolves this". An explicit ``stop: true`` carries
    ``agent_requested`` and never reaches that branch, which is correct: the
    policy said so, even if the filter also dropped a name from the same reply.
    """
    if stop_reason == "proposal_unresolved":
        return (
            "proposal_unresolved",
            "the loop ran out of resolvable names: every mitigation the policy "
            f"named was dropped by the filter ({sorted(set(unresolved))}), so the "
            "stop is not a decision the policy made",
        )
    if unresolvable and proposed_nothing:
        return _classify_unresolvable_claim(offered, tried)
    if not resolver_named:
        return "gave_up", "stopped without ever naming a mitigation that resolves it"
    return "other", "stopped after naming a resolver"


def episode_from_log(
    run_dir: Path,
    grid: MatrixGrid | None = None,
    offered_mitigations: Iterable[str] = (),
) -> LoggedEpisode:
    """Replay an ``agent_log.jsonl`` into steps and a terminal classification.

    The mitigation axis carried on each step is the one in force **when the
    policy chose**, rebuilt from the preceding ``mitigation_tried`` records, so
    ``name_already_tried`` is decided against what was on screen at the time
    rather than against the final grid.

    Terminal classification, each read off an event the loop writes:

    ``converged``
        a ``converged`` record exists.
    a ``search_stopped`` record
        :func:`classify_stop` -- ``proposal_unresolved``, an unresolvability
        claim (earned or not), ``gave_up``, or ``other``.
    ``other``
        ``baseline_pass``, ``policy_stop`` (a budget), ``approval_required``,
        ``registry_error`` and ``error``: none is a statement about the search.
    ``unobserved``
        the log records no terminal event at all.

    Cells are one per mitigation on the final axis: ``run_agent_loop`` executes
    the whole axis every iteration and ends either before growing it or right
    after executing it. A run directory's own cells, where present, outrank
    that count -- a cell directory holding trial results exists if and only if
    GPU was spent there, which is also how a recipe with a wider diagnostic
    axis is priced correctly.
    """
    offered = list(offered_mitigations)
    mitigation_axis = [BASELINE]
    steps: list[LoggedStep] = []
    terminal, why = "unobserved", "the log records no terminal event"
    resolver_named = False
    resolvers = grid.resolution.resolvers if grid is not None else frozenset()
    unresolvable = (
        grid is not None
        and grid.resolution.baseline_failed
        and not grid.resolution.resolvers
    )

    for event in read_log(run_dir):
        etype = event.get("type")
        if etype == "llm_step":
            raw = _step_to_raw(event)
            steps.append(
                LoggedStep(
                    n=len(steps) + 1,
                    raw=raw,
                    mitigation_axis_before=list(mitigation_axis),
                    stop=bool(event.get("stop")),
                )
            )
            # The names the loop could actually run: what survived the filter,
            # and nothing from a step that stopped (see `step_events`).
            if not event.get("stop") and set(event.get("next_mitigations") or []) & set(
                resolvers
            ):
                resolver_named = True
        elif etype == "mitigation_tried":
            name = str(event.get("mitigation"))
            if name not in mitigation_axis:
                mitigation_axis.append(name)
        elif etype == "converged":
            terminal = "converged"
            why = f"cell {event.get('winning_mitigation')!r}-none passed"
        elif etype == "baseline_pass":
            terminal, why = "other", "the baseline cell passed: there was nothing to search for"
        elif etype == "policy_stop":
            terminal, why = "other", f"policy_stop: {event.get('reason')}"
        elif etype in ("registry_error", "approval_required", "error"):
            terminal, why = "other", f"the run ended on {etype}"
        elif etype == "search_stopped":
            stopping = steps[-1] if steps else None
            proposed_nothing = stopping is not None and not json.loads(
                stopping.raw
            )["next_mitigations"]
            terminal, why = classify_stop(
                stop_reason=event.get("stop_reason"),
                proposed_nothing=proposed_nothing,
                unresolvable=unresolvable,
                resolver_named=resolver_named,
                offered=offered,
                tried=[n for n in mitigation_axis if n != BASELINE],
                unresolved=event.get("unresolved_mitigations") or [],
            )

    on_disk = len([c for c in find_probe_cells(run_dir) if read_trial_results(c)])
    return LoggedEpisode(
        run_dir=run_dir,
        steps=steps,
        terminal=terminal,
        terminal_why=why,
        cells=max(len(mitigation_axis), on_disk),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _terminal_events(
    terminal: str, why: str, points: Mapping[str, float]
) -> tuple[list[Event], list[str]]:
    if terminal in WITHHELD_TERMINALS:
        return [], [f"terminal ({terminal}): {why}"]
    key = f"terminal_{terminal}"
    if key not in points:
        return [], [f"terminal ({terminal}): unrecognised, scored nothing"]
    return [Event(key, points[key], why)], []


def _cell_events(cells: int, points: Mapping[str, float]) -> list[Event]:
    if not cells:
        return []
    return [Event("cell_spent", points["cell_spent"] * cells, f"{cells} probe cells executed")]


def score_episode(
    episode: LoggedEpisode,
    grid: MatrixGrid | None,
    context: StepContext,
    *,
    labels: Mapping[int, Label] | None = None,
    points: Mapping[str, float] = POINTS,
) -> EventScore:
    """Score a whole episode: per-step events, one terminal, and the cells.

    The tried set is advanced as the episode runs -- names bought on step *n*
    are "already tried" from step *n+1* on -- by reading each step's own
    ``mitigation_axis_before`` rather than accumulating, so a resumed run that
    inherited a populated axis is scored against what was really on it.

    ``labels`` is read for the first step only: the read of the evidence is
    paid once per episode (module docstring). Labels for later steps are
    ignored rather than refused, so the rule lives here and in no caller.
    """
    score = EventScore(terminal=episode.terminal, cells=episode.cells)
    resolver_named = False
    for index, logged in enumerate(episode.steps):
        here = StepContext(
            offered_mitigations=context.offered_mitigations,
            tried_mitigations=frozenset(
                set(context.tried_mitigations)
                | {n for n in logged.mitigation_axis_before if n != BASELINE}
            ),
        )
        fired, _ = step_events(
            logged.raw,
            here,
            grid,
            step=logged.n,
            label=(labels or {}).get(logged.n) if index == 0 else None,
            resolver_already_named=resolver_named,
            points=points,
        )
        score.events.extend(fired)
        if any(event.name == "resolver_named" for event in fired):
            resolver_named = True

    terminal_events, withheld = _terminal_events(episode.terminal, episode.terminal_why, points)
    score.events.extend(terminal_events)
    score.withheld.extend(withheld)
    score.events.extend(_cell_events(episode.cells, points))
    if grid is None:
        score.withheld.append("no archived matrix: every ground-truth event was withheld")
    return score


def score_completion(
    raw: str,
    grid: MatrixGrid | None,
    context: StepContext,
    *,
    cells_already_spent: int = 0,
    cells_added: int = 0,
    label: Label | None = None,
    points: Mapping[str, float] = POINTS,
) -> EventScore:
    """Score a single reply as a one-step episode.

    One reply has step events and a cell cost and does **not** have an
    observed terminal, so the terminal is classified from the reply's own
    claim rather than from a run that never happened: a reply that stops while
    proposing nothing is making the terminal claim itself, and the matrix can
    adjudicate it. Any other reply leaves the terminal ``unobserved``, withheld
    and flagged rather than scored zero.

    ``cells_added`` is the caller's, so this module holds no second opinion
    about what an action costs.
    """
    score = EventScore()
    events, parsed = step_events(raw, context, grid, step=1, label=label, points=points)
    score.events.extend(events)

    terminal, why = "unobserved", "a single reply does not observe how a run ended"
    if parsed is not None and parsed.stop and not parsed.next_mitigations:
        if grid is None:
            terminal, why = "other", "no matrix: the stop claim cannot be adjudicated"
        elif grid.resolution.baseline_failed and not grid.resolution.resolvers:
            terminal, why = _classify_unresolvable_claim(
                context.offered_mitigations, context.tried_mitigations
            )
        else:
            terminal, why = "gave_up", "stopped with no mitigations while a resolver exists"
    score.terminal, score.cells = terminal, cells_already_spent + cells_added

    terminal_events, withheld = _terminal_events(terminal, why, points)
    score.events.extend(terminal_events)
    score.withheld.extend(withheld)
    score.events.extend(_cell_events(score.cells, points))
    if grid is None:
        score.withheld.append("no archived matrix: every ground-truth event was withheld")
    return score


# ---------------------------------------------------------------------------
# CLI: print the point table, or score one archived episode
# ---------------------------------------------------------------------------


def _print_points() -> None:
    width = max(len(name) for name in POINTS)
    print("Point table (see the module docstring for each value's justification):")
    for name, value in POINTS.items():
        print(f"  {name:<{width}}  {value:+.1f}")


def _print_episode(episode: LoggedEpisode, grid: MatrixGrid, score: EventScore) -> None:
    print(f"episode   {episode.run_dir}")
    print(f"matrix    {grid.resolution.source}")
    print(f"resolvers {sorted(grid.resolution.resolvers) or '(none)'}")
    print(f"steps     {len(episode.steps)}   cells {episode.cells}   "
          f"terminal {episode.terminal} -- {episode.terminal_why}")
    print()
    for event in score.events:
        where = f"step {event.step}" if event.step else "episode"
        print(f"  {event.points:+6.1f}  {event.name:<30} {where:<9} {event.why}")
    for note in score.withheld:
        print(f"  {'--':>6}  {note}")
    print()
    print(f"  {score.total:+6.1f}  TOTAL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", action="store_true",
                        help="print the point table and exit")
    parser.add_argument("--matrix", type=Path, default=None,
                        help="an archived probe run directory (the cell dirs' parent)")
    parser.add_argument("--episode", type=Path, default=None,
                        help="a run directory holding agent_log.jsonl")
    parser.add_argument("--offered", default="",
                        help="comma-separated mitigations that were on the menu. "
                             "Required with --episode: without it no name can be "
                             "judged offered and no unresolvability claim earned")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.points or not (args.matrix and args.episode):
        _print_points()
        return 0 if args.points else 2
    offered = frozenset(n.strip() for n in args.offered.split(",") if n.strip())
    if not offered:
        # Every name would score `name_not_offered` and the unresolvable claim
        # could never be earned, so the total would be a number about the
        # missing flag rather than about the episode.
        print("--offered is required with --episode", file=sys.stderr)
        return 2

    grid = grid_from_matrix(args.matrix)
    episode = episode_from_log(args.episode, grid, offered)
    score = score_episode(episode, grid, StepContext(offered_mitigations=offered))
    if args.json:
        print(json.dumps({"grid": grid.as_dict(), "score": score.as_dict()}, indent=2))
    else:
        _print_episode(episode, grid, score)
    return 0


if __name__ == "__main__":
    sys.exit(main())
