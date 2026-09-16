#!/usr/bin/env python3
"""Minimum-cost root-cause localisation: the objective, beside the old one.

The task changed. "Sanitizer selection" -- given a failure, pick which
sanitizer to run -- is a classification problem with a handful of labels and
does not need a trained model. The task this scores is: given a failing run,
choose a *sequence* of experiments across both probe axes (a diagnostic buys
information and costs GPU; a mitigation tests a causal hypothesis) and stop
when you can name the cause and a fix that demonstrably works. Scored on
correctness minus the probe cells spent.

Why a new module instead of a patch to the old ones
---------------------------------------------------
`proposal_reward.py` (form), `triage_reward.py` (read of the evidence) and
`fix_reward.py` (did the mitigation work) stay exactly as they are, and this
imports all three rather than reimplementing any of them. Both objectives have
to be runnable on one set of rollouts in one command, because the single most
useful thing to show is the *same* policies ranked by the two rewards -- the
constant that currently wins visibly losing. A patch would have destroyed the
comparison.

PRE-REGISTERED WEIGHTS
======================
Fixed before any number was computed, and not moved afterwards::

    W_TRIAGE = 0.3      verdict correctness + attribution F1 over detector IDs
    W_FIX    = 0.5      F1 of proposed mitigations against the true resolvers
    W_COST   = 0.2      probe cells spent, normalised against a budget

This project has twice found weight regions that flattered a result -- the
containment formulation of the fix half failed criterion 1 *by arithmetic* at
`FIX_WEIGHT = 0.5`, and raising the weight only paid more for the model's
habits. So the rule here is that a weight change is reported as a finding and
is not applied silently. If these three numbers are wrong, the way to find out
is to say so, not to search.

Four rules, and one special case
--------------------------------
1. **Form is a gate, not a score.** Today's five-tier form ladder is worth up
   to 1.0 on its own, and that is the whole reason a two-line constant that
   reads nothing scores 0.9000 against Qwen3-8B's 0.7569: the ladder measures
   the *shape* of a reply, and a constant has an excellent shape. Here, valid
   JSON with the required fields, a category in the closed autopsy set, and
   every proposed name both registered and currently offered are *entry
   conditions*. Fail any and the total is 0.0 with nothing else computed. A
   well-formed reply earns no points for being well-formed.

2. **Triage term** -- `triage_reward.score_answer`, unchanged and not
   reimplemented: `0.6 * verdict-correct + 0.4 * attribution-F1`.

3. **Fix term** -- `fix_reward.fix_credit`, the F1 of the proposed mitigations
   against the resolver set recovered from the archived matrix. Also
   unchanged. GPU is paid once per scenario, not per rollout sample: the
   matrix is read by listing cell directories.

4. **Cost term** -- the probe cells the episode spent, over a budget. The cells
   are counted by `aorta.agent.loop.plan_axis_growth`, which is *the same
   function the loop charges*, so the reward cannot price an action differently
   from the code that runs it. With two axes that matters more than it sounds:
   the cell grid is a cross product, so one diagnostic re-runs every
   mitigation already on the axis and a joint proposal of `a` mitigations and
   `b` diagnostics costs `a*D + b*M + a*b` cells, not `a + b`.

The special case (rule 4 of the brief): **an empty resolver set.** Some real
failures are resolved by nothing in the registry -- the archived fp16-overflow
scenario is one, measured against 21 mitigations, all of which fail. Without a
special case the fix term would be 0.0 for every policy there, which teaches
"always guess, a guess is free". So when the true resolver set is empty and
the baseline did fail, correctly reporting that nothing resolves it earns
**full** fix credit. "Correctly reporting" is expressible in the existing
schema and needs no new field: propose no mitigations *and* set `stop` true.
An empty list without `stop` is a policy with nothing to say, which is a
different claim.

Two things about the scale, stated rather than smoothed over
-----------------------------------------------------------
The maximum attainable total is `W_TRIAGE + W_FIX = 0.8`, reached only by a
policy that spends zero cells -- which is impossible, since the baseline cell
always runs. Today's objective tops out at 1.0. **The two columns are not on
the same scale and must not be compared value against value; what transfers is
the ranking**, which is the only thing GRPO reads anyway.

And the total is floored at 0.0. Without the floor, a gate failure (0.0 by
rule 1) would outscore a well-formed answer that earned little and spent a
lot, which would teach the model to emit garbage rather than try. The floor
buys that at the price of a flat spot: a gate-passing answer that earns
nothing and spends its whole budget ties garbage at 0.0. That is a known
defect of this formulation, recorded here rather than fixed by moving a
weight.

Usage
-----

    python examples/rl/cost_aware_reward.py \
        --results /apps/vikhande/probe-gpu/rl-nan/results \
        --matrix  /apps/vikhande/probe-gpu/nan_matrix/PROBE-NAN-WS-MATRIX
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.agent.llm import AUTOPSY_CATEGORIES, AgentStep
from aorta.agent.loop import plan_axis_growth
from aorta.agent.policy import AgentPolicy, PolicyViolation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_reward import Resolution, fix_credit, resolution_from_matrix  # noqa: E402
from proposal_reward import REQUIRED_KEYS  # noqa: E402
from triage_reward import Answer, Label, score_answer  # noqa: E402

# ---------------------------------------------------------------------------
# The pre-registered weights. See the module docstring. Do not fit these.
# ---------------------------------------------------------------------------
W_TRIAGE = 0.3
W_FIX = 0.5
W_COST = 0.2

#: Probe cells one episode is allowed. Not a tuned coefficient -- it is the
#: denominator that turns a cell count into a [0, 1] penalty, and it is the
#: same number the loop is given as ``--max-cells``. The reward and the
#: runtime budget being one number is the point: a reward that normalises
#: against a budget the loop does not enforce is scoring a fiction.
DEFAULT_BUDGET_CELLS = 10

#: The baseline names on the two axes, as the probe harness spells them.
BASELINE = "none"


# ---------------------------------------------------------------------------
# Rule 1: form as a gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """Whether a reply is admissible at all, and why not if not."""

    passed: bool
    reason: str = ""
    obj: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason}


def form_gate(raw: str, offered: Iterable[str], policy: AgentPolicy | None = None) -> Gate:
    """Admissibility, checked in the order the real consumer checks it.

    Every condition here is one the shipped path already enforces somewhere --
    ``_step_from_content``'s parse, ``REQUIRED_KEYS`` from the system prompt,
    ``AgentPolicy.validate_step``'s category and registry checks, and the
    loop's candidate filter. Nothing new is demanded of the model. What
    changes is that meeting them is worth zero rather than up to 1.0.

    The offered-set check is deliberately part of the *gate* rather than a
    deduction. A registered name that is not currently offered is a name the
    loop would silently drop, so a reply containing one is a reply whose cost
    and effect cannot be predicted from its text -- and the fix half already
    scores only what would run. Admitting it at a discount would mean scoring
    a proposal the consumer never sees.
    """
    offered_set = {str(n) for n in offered}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return Gate(False, f"not JSON: {exc.msg}")
    if not isinstance(obj, dict):
        return Gate(False, f"parsed as {type(obj).__name__}, not an object")

    for key, expected in REQUIRED_KEYS.items():
        if key not in obj:
            return Gate(False, f"missing required key {key!r}")
        # bool is a subclass of int, so a "confidence": true would satisfy an
        # isinstance check against (int, float) and has to be excluded by hand.
        value = obj[key]
        if expected is not bool and isinstance(value, bool):
            return Gate(False, f"{key!r} is a bool, expected {expected}")
        if not isinstance(value, expected):
            return Gate(False, f"{key!r} is {type(value).__name__}, expected {expected}")

    if obj["category"] not in AUTOPSY_CATEGORIES:
        return Gate(False, f"category {obj['category']!r} outside the closed set")

    step = AgentStep.from_dict(obj)
    try:
        (policy or AgentPolicy()).validate_step(step)
    except PolicyViolation as exc:
        return Gate(False, f"policy: {exc}")

    # Both axes, same rule. `none` is dropped by validate_step because it is
    # the baseline and already on every axis, so it is not an offence here.
    for axis_label, names in (
        ("mitigation", step.next_mitigations),
        ("diagnostic", step.next_diagnostics),
    ):
        for name in names:
            if name != BASELINE and name not in offered_set:
                return Gate(
                    False,
                    f"{axis_label} {name!r} is registered but was not offered",
                )
    return Gate(True, obj=obj)


# ---------------------------------------------------------------------------
# Rule 4: cost, priced by the function the loop charges
# ---------------------------------------------------------------------------


def cells_for_proposal(
    mitigations: list[str],
    diagnostics: list[str],
    mitigation_axis: list[str],
    diagnostic_axis: list[str],
) -> int:
    """Probe cells this proposal would add, per the loop's own accounting.

    Delegates to :func:`aorta.agent.loop.plan_axis_growth` rather than
    recomputing the cross product, so the reward and the runtime cannot
    disagree about what an action costs. They disagreeing is the defect
    already characterised in this repo from the other direction: the loop
    appends every proposed name to an axis and charges the iteration budget
    once, so a wide-candidate run can spend 160 cells against a budget of 8.
    """
    return plan_axis_growth(
        mitigation_axis, diagnostic_axis, mitigations, diagnostics
    ).cells_added


def cost_penalty(cells_spent: int, budget_cells: int = DEFAULT_BUDGET_CELLS) -> float:
    """Cells spent over the budget, clipped into [0, 1].

    Clipped rather than allowed to run away: a policy that has already blown
    the budget by 2x and one that blew it by 20x are both "spent everything",
    and letting the term grow without bound would let the cost dominate a
    reward whose other two terms are bounded -- which is a weight change by the
    back door.
    """
    if budget_cells < 1:
        raise ValueError(f"budget_cells must be >= 1, got {budget_cells}")
    return min(1.0, max(0, cells_spent) / budget_cells)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


@dataclass
class CostAwareScore:
    """One proposal's score under the cost-aware objective."""

    total: float = 0.0
    gate_passed: bool = False
    gate_reason: str = ""
    triage: float = 0.0
    fix: float = 0.0
    fix_withheld: bool = False
    cells_added: int = 0
    cells_spent: int = 0
    cost_penalty: float = 0.0
    names_runnable: list[str] = field(default_factory=list)
    diagnostics_runnable: list[str] = field(default_factory=list)
    reported_unresolvable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "gate_passed": self.gate_passed,
            "gate_reason": self.gate_reason,
            "triage": round(self.triage, 4),
            "fix": round(self.fix, 4),
            "fix_withheld": self.fix_withheld,
            "cells_added": self.cells_added,
            "cells_spent": self.cells_spent,
            "cost_penalty": round(self.cost_penalty, 4),
            "names_runnable": list(self.names_runnable),
            "diagnostics_runnable": list(self.diagnostics_runnable),
            "reported_unresolvable": self.reported_unresolvable,
        }


def reported_unresolvable(step: AgentStep) -> bool:
    """Whether the policy claimed that nothing in the registry fixes this.

    Expressible in the shipped schema, which is why no field was added:
    propose no mitigations *and* stop. Requiring the stop is what separates
    the claim from a step that merely had nothing to add -- ``run_agent_loop``
    treats those two identically today, but they are different assertions and
    only one of them deserves credit on an unresolvable scenario.
    """
    return not step.next_mitigations and step.stop


def score_cost_aware(
    raw: str,
    *,
    offered: list[str],
    label: Label | None,
    resolution: Resolution | None,
    mitigation_axis: list[str],
    diagnostic_axis: list[str],
    budget_cells: int = DEFAULT_BUDGET_CELLS,
    cells_already_spent: int = 0,
    w_triage: float = W_TRIAGE,
    w_fix: float = W_FIX,
    w_cost: float = W_COST,
    policy: AgentPolicy | None = None,
) -> CostAwareScore:
    """Score one proposal. Pure: no filesystem, no GPU, no model.

    ``label`` is the triage ground truth for the scenario and ``resolution``
    the fix-half ground truth recovered from the archived matrix. Either may be
    None, in which case that term is withheld (contributes 0.0) -- withheld
    rather than scored zero, and the flag says which happened.
    """
    score = CostAwareScore()
    gate = form_gate(raw, offered, policy)
    score.gate_passed = gate.passed
    score.gate_reason = gate.reason
    if not gate.passed:
        # Rule 1: fail the gate, total 0, stop. Nothing else is *scored* --
        # not as an optimisation but because the remaining terms would be
        # reading fields whose types were just rejected.
        #
        # The episode's sunk cells are still recorded, though. They were
        # genuinely spent, and reporting 0 would make a policy that emits
        # prose look like the cheapest one on a dashboard. It changes no
        # score: the total is 0.0 either way.
        score.cells_spent = cells_already_spent
        score.cost_penalty = cost_penalty(cells_already_spent, budget_cells)
        return score

    assert gate.obj is not None
    step = AgentStep.from_dict(gate.obj)
    score.names_runnable = [m for m in step.next_mitigations if m in offered]
    score.diagnostics_runnable = [d for d in step.next_diagnostics if d in offered]
    score.reported_unresolvable = reported_unresolvable(step)

    if label is not None:
        # An absent verdict is scored as a wrong verdict, NOT defaulted to the
        # label's. Defaulting would hand every policy that declines to read the
        # evidence a free 0.6, which is the exact shape of defect this
        # objective exists to remove.
        score.triage = score_answer(
            Answer(
                verdict=str(gate.obj.get("verdict") or ""),
                detectors=[str(d) for d in (gate.obj.get("detectors") or [])],
            ),
            label,
        ).reward

    if resolution is None or not resolution.baseline_failed:
        # No failure to resolve, or no ground truth at all. Scoring 0.0 here
        # would add a constant to every policy and look like a measurement.
        score.fix_withheld = True
    elif not resolution.resolvers:
        # The special case. A real scenario nothing resolves is not a scenario
        # with no right answer -- the right answer is "nothing in the registry
        # fixes this", and it has to pay, or the policy learns that guessing
        # is free.
        score.fix = 1.0 if score.reported_unresolvable else 0.0
    else:
        score.fix = fix_credit(score.names_runnable, resolution)

    score.cells_added = cells_for_proposal(
        score.names_runnable,
        score.diagnostics_runnable,
        mitigation_axis,
        diagnostic_axis,
    )
    score.cells_spent = cells_already_spent + score.cells_added
    score.cost_penalty = cost_penalty(score.cells_spent, budget_cells)

    earned = w_triage * score.triage + w_fix * score.fix
    # Floored at 0.0 so garbage (0.0 by rule 1) can never outscore a
    # well-formed attempt. See the module docstring for the flat spot this
    # buys and why it is recorded rather than tuned away.
    score.total = max(0.0, earned - w_cost * score.cost_penalty)
    return score


def max_attainable(w_triage: float = W_TRIAGE, w_fix: float = W_FIX) -> float:
    """The ceiling, for a reader comparing columns. Not 1.0 -- see the docstring."""
    return w_triage + w_fix


# ---------------------------------------------------------------------------
# CLI: score the reference policies on one archived matrix
# ---------------------------------------------------------------------------


def _demo(matrix: Path, budget: int, as_json: bool) -> int:
    from rescore_e2e import reference_policies

    resolution = resolution_from_matrix(matrix)
    offered = sorted(
        n
        for n in (
            "tf32_off",
            "hsa_no_sdma",
            "hip_launch_blocking",
            "xnack",
            "pytorch_alloc_expandable_segments",
            "pytorch_no_cuda_memory_caching",
            "gpu_max_hw_queues_2",
        )
    )
    rows = []
    for name, raw in reference_policies(offered, resolution).items():
        score = score_cost_aware(
            raw,
            offered=offered,
            label=None,
            resolution=resolution,
            mitigation_axis=[BASELINE],
            diagnostic_axis=[BASELINE],
            budget_cells=budget,
            cells_already_spent=1,
        )
        rows.append({"policy": name, **score.as_dict()})

    if as_json:
        print(json.dumps({"resolution": resolution.as_dict(), "policies": rows}, indent=2))
        return 0

    print(f"matrix     {resolution.source}")
    print(f"resolvers  {sorted(resolution.resolvers) or '(none)'}")
    print(f"weights    triage={W_TRIAGE} fix={W_FIX} cost={W_COST} "
          f"budget={budget} cells; ceiling {max_attainable():.2f}")
    print()
    print(f"  {'policy':<26} {'gate':>5} {'fix':>6} {'cells':>6} {'cost':>6} {'total':>7}")
    for row in sorted(rows, key=lambda r: -r["total"]):
        print(
            f"  {row['policy']:<26} {str(row['gate_passed']):>5} {row['fix']:>6.3f} "
            f"{row['cells_spent']:>6} {row['cost_penalty']:>6.3f} {row['total']:>7.4f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, required=True,
                        help="an archived probe run directory (the cell dirs' parent)")
    parser.add_argument("--budget-cells", type=int, default=DEFAULT_BUDGET_CELLS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return _demo(args.matrix, args.budget_cells, args.json)


if __name__ == "__main__":
    sys.exit(main())
