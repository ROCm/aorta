#!/usr/bin/env python3
"""The contract's fix half: did the proposed mitigation actually resolve the repro?

`proposal_reward.py` scores the *shape* of a proposal and `triage_reward.py`
scores the *read* of the evidence. Both are inspection. Neither can tell a
mitigation that works from one that merely exists, and the first end-to-end run
showed what that costs: a two-line constant that declines to classify and names
one mitigation scores 0.9000 while Qwen3-8B scores 0.8689. Every form term is
gameable by something that reads nothing, because every form term is a function
of the proposal's shape and a constant has a very good shape.

This is the term that is not. It asks one question of the archived matrix --
did any mitigation the policy named make the failing cell pass -- and the only
way to score on it is to name the right thing.

GPU is paid once, not per sample
--------------------------------
The obvious objection is cost: "checked by execution" sounds like a GPU run per
rollout sample, which would make GRPO unaffordable. It is not, and the reason is
that the execution has already happened. A probe run archives one cell per
``{mitigation}-{diagnostic}`` pair, and ``aorta.agent.state.winning_mitigation``
reads the winner back out of the cell *directory name*. So the ground truth for
a scenario is recovered by listing a directory, and the reward term is an
offline lookup over a set of strings. One probe run, then arbitrarily many
rollouts against it.

That is also why :func:`resolution_from_matrix` is separate from
:func:`fix_credit`: the first touches the filesystem once per scenario, the
second is pure and runs per sample.

Any winner, not the historical one
----------------------------------
``docs/tokenspeed-rl-post-training.md`` section 4.3 is explicit that the reward
must accept *any* mitigation that made the cell pass, not only the one the
historical agent search happened to try first. A matrix with three passing
mitigation cells has three right answers, and a policy that names the second is
not wrong. :attr:`Resolution.resolvers` is therefore a set, and
:func:`fix_credit` is membership in it.

F1 over the resolvers, for consistency with the attribution term
----------------------------------------------------------------
``fix_credit`` is the F1 of the proposed mitigations against the ones that
actually resolved the failure. It **was** containment -- 1.0 if the list held at
least one resolver -- and that formulation was measured and reported first (see
``docs/rl-fix-half-reward-findings.md``); it is kept as
:func:`fix_credit_contained` so the comparison stays reproducible.

The reason for the change is a correctness argument, not a number. Containment
is *monotone in list length*: naming the whole candidate set names every
possible resolver by construction, so a policy that reads nothing scores a
perfect 1.0. That is precisely the hole ``triage_reward.py`` already closed one
layer up, and it closed it this way -- ``0.4 * attribution-F1`` over the cited
detector IDs, F1 rather than "cited at least one", so that citing everything
cannot win. Scoring attribution by F1 while scoring the fix by containment is an
internal inconsistency in one reward, and removing it is right whichever
direction the resulting number moves.

The consistency is structural rather than asserted: this calls
``triage_reward.set_f1``, the same function the attribution term calls, so the
two cannot drift and cannot disagree about the empty cases. That function was
extracted from ``score_answer`` for this purpose and its arithmetic is
unchanged, so no recorded triage number moves. An empty proposal earns 0.0 --
it resolved nothing -- and the empty-empty case that scores 1.0 for attribution
is unreachable here, because a scenario with no resolvers is withheld before
fix credit is ever called.

Breadth is now priced twice, and that is the intended reading. Precision falls
as ``1/k`` here, and ``proposal_reward.precision_credit`` scales the tier 4-5
block by ``min(1, 2/k)`` on the form side. Both are expressions of one fact --
the loop spends one probe cell per proposed name -- and the earlier decision to
price it only once rested on containment being the fix formulation. It is not a
new free parameter: F1 introduces no coefficient.

The one weight in this module remains :data:`FIX_WEIGHT`, set to the neutral 0.5
("the contract has two halves") rather than fitted.

When the term must be withheld rather than scored
-------------------------------------------------
A scenario carries fix-half signal only if its baseline failed (there was
something to resolve) *and* at least one mitigation resolved it (a right answer
exists). Otherwise every policy scores identically and the term is a constant
added to every reward -- no gradient, and worse, a constant that looks like a
measurement. :attr:`Resolution.scoreable` says which case a scenario is in, and
callers are expected to withhold the term rather than score a zero.

Usage
-----

    python examples/rl/fix_reward.py --matrix path/to/run/TICKET
    python examples/rl/fix_reward.py --matrix path/to/run/TICKET --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.agent.state import read_trial_results, winning_mitigation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage_reward import find_probe_cells, label_trials, set_f1  # noqa: E402

# What the fix half is worth against the form half.
#
# 0.5 is the neutral point -- the contract has two halves and neither is
# subordinate -- chosen rather than fitted. It is the only free parameter in
# this module and it is exposed on the CLI so a reader can see what any other
# value would do instead of having to trust this one.
FIX_WEIGHT = 0.5

# The cell name a probe run gives the no-op configuration. `winning_mitigation`
# already hard-codes the same convention; named here so the baseline-failed
# check reads the same way.
BASELINE_CELL = "none-none"


@dataclass(frozen=True)
class Resolution:
    """What an archived probe matrix says about one scenario.

    The ground truth the fix half scores against, recovered by listing cell
    directories -- no GPU, no workload, no model.
    """

    scenario_id: str
    resolvers: frozenset[str]
    baseline_failed: bool
    cells: int
    source: str

    @property
    def scoreable(self) -> bool:
        """Whether the fix half carries signal for this scenario.

        False when there was no failure to resolve, or when nothing in the
        matrix resolved it. Both cases make the term a constant; see the module
        docstring for why that is worse than having no term.
        """
        return self.baseline_failed and bool(self.resolvers)

    @property
    def withheld_because(self) -> str:
        if not self.baseline_failed:
            return "the baseline cell did not fail: there is nothing to resolve"
        if not self.resolvers:
            return "no mitigation in the matrix resolved the failure"
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "resolvers": sorted(self.resolvers),
            "baseline_failed": self.baseline_failed,
            "cells": self.cells,
            "scoreable": self.scoreable,
            "withheld_because": self.withheld_because,
            "source": self.source,
        }


def resolution_from_matrix(root: Path, scenario_id: str | None = None) -> Resolution:
    """Recover the resolving mitigations from an archived probe matrix.

    Walks the cell directories, relabels each one through
    :func:`triage_reward.label_trials` -- so the verdict is recomputed by
    aorta's own resolver rather than read off the artifact, the same seam the
    triage reward uses -- and asks
    :func:`aorta.agent.state.winning_mitigation` which of them count as wins.

    That second call is what keeps this honest about attribution: a pass on a
    diagnostic-only cell, or on a mitigation *plus* diagnostic cell, is not
    attributable to the mitigation alone, and `winning_mitigation` returns None
    for both. Only a `{mitigation}-none` cell that passed is a resolver.
    """
    cells = find_probe_cells(root)
    resolvers: set[str] = set()
    baseline_failed = False
    for cell in cells:
        docs = read_trial_results(cell)
        if not docs:
            continue
        verdict = label_trials(docs, source=str(cell)).verdict
        if cell.name == BASELINE_CELL:
            baseline_failed = verdict == "fail"
        winner = winning_mitigation(cell.name, verdict)
        if winner is not None:
            resolvers.add(winner)
    return Resolution(
        scenario_id=scenario_id or root.name,
        resolvers=frozenset(resolvers),
        baseline_failed=baseline_failed,
        cells=len(cells),
        source=str(root),
    )


def fix_credit(names: Iterable[str], resolution: Resolution) -> float:
    """F1 of the proposed mitigations against the ones that resolved the failure.

    Pure and cheap: the archived matrix was read once, into
    ``resolution.resolvers``, and this is set arithmetic. Called once per
    rollout sample.

    Recall alone would be containment with extra steps -- it is maximised by
    naming everything. Precision alone would reward a single lucky name over a
    pair containing it. F1 is the combination the attribution term in
    ``triage_reward.py`` already uses over cited detector IDs, and the point of
    matching it is that one reward should not hold two different opinions about
    how to score a set against a set.
    """
    return set_f1(names, resolution.resolvers)


def fix_credit_contained(names: Iterable[str], resolution: Resolution) -> float:
    """The superseded formulation: 1.0 if the list held any resolver at all.

    Retained, and not as a knob. It is the formulation the first measurement was
    taken under, so keeping it is what lets the before/after comparison be one
    command rather than a checkout; and the property that condemned it --
    monotone in list length, so a shotgun scores 1.0 for free -- is pinned by a
    test against *this* function, where it is a documented fact rather than a
    regression waiting to be reintroduced.
    """
    return 1.0 if resolution.resolvers.intersection(str(n) for n in names) else 0.0


# The two formulations, by name, for the comparison harness. Not a search space:
# `f1` is the formulation, `containment` is the superseded one kept so the
# earlier measurement stays reproducible.
FORMULATIONS = {
    "f1": fix_credit,
    "containment": fix_credit_contained,
}


def composite_reward(form: float, fix: float, fix_weight: float = FIX_WEIGHT) -> float:
    """Blend the form half with the fix half.

    Kept as a named function rather than inlined at each call site so the one
    weight in the design has one home, and so a caller that withholds the fix
    half (see :attr:`Resolution.scoreable`) is visibly not calling this.
    """
    if not 0.0 <= fix_weight <= 1.0:
        raise ValueError(f"fix_weight {fix_weight} outside [0, 1]")
    return (1.0 - fix_weight) * form + fix_weight * fix


def _demo(root: Path, fix_weight: float, as_json: bool) -> int:
    resolution = resolution_from_matrix(root)
    if as_json:
        print(json.dumps(resolution.as_dict(), indent=2))
        return 0 if resolution.scoreable else 1

    print(f"Archived matrix: {resolution.source}")
    print(f"  cells inspected       {resolution.cells}")
    print(f"  baseline failed       {resolution.baseline_failed}")
    print(f"  resolving mitigations {sorted(resolution.resolvers) or '(none)'}")
    print()
    if not resolution.scoreable:
        print(f"The fix half is WITHHELD here: {resolution.withheld_because}.")
        print("Scoring it anyway would add the same constant to every policy,")
        print("which is a degenerate term wearing the clothes of a measurement.")
        return 1

    # The rows that matter are the ones where the two formulations disagree:
    # padding a correct answer with wrong ones is free under containment and
    # costly under F1, which is the whole reason for the change.
    resolver = sorted(resolution.resolvers)[:1]
    padded = resolver + [n for n in ("none", "xnack", "hsa_no_sdma") if n not in resolver]
    print(f"  what the term pays, for a form score of 1.00, weight {fix_weight:g}:")
    print(f"    {'proposal':<34} {'F1':>6} {'containment':>12} {'composite (F1)':>16}")
    for label, names in (
        ("exactly one resolver", resolver),
        (f"one resolver plus {len(padded) - 1} wrong", padded),
        ("one non-resolver", ["none"]),
        ("nothing", []),
    ):
        f1 = fix_credit(names, resolution)
        contained = fix_credit_contained(names, resolution)
        print(
            f"    {label:<34} {f1:>6.3f} {contained:>12.3f} "
            f"{composite_reward(1.0, f1, fix_weight):>16.4f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, required=True,
                        help="an archived probe run directory (the one holding "
                             "the {mitigation}-{diagnostic} cell dirs)")
    parser.add_argument("--fix-weight", type=float, default=FIX_WEIGHT,
                        help=f"weight on the fix half (default {FIX_WEIGHT})")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    return _demo(args.matrix, args.fix_weight, args.json)


if __name__ == "__main__":
    sys.exit(main())
