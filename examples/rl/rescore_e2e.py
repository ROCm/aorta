#!/usr/bin/env python3
"""Re-score the recorded end-to-end proposals, and check the saturation broke.

`run_e2e.py` needs a served model and a GPU. This does not: it reads the raw
completions `run_e2e.py` already wrote and re-scores them through whatever
`proposal_reward.py` currently says, so a reward change can be evaluated against
real model output in milliseconds.

That separation is the point. The recorded files carry the *raw wire text* of
every completion alongside the reward it earned at the time, so the "before" is
in the data rather than reconstructed, and the "after" is the current grader
applied to the same bytes.

What it checks
--------------
The four criteria the reward has to satisfy to be trainable at all:

1. the two abstaining constant templates score strictly below the model;
2. the contract-perfect reference scores at or near the top;
3. per-scenario groups show non-zero within-group spread -- the one that
   matters, because a group with zero spread has zero GRPO advantage;
4. the model lands between the constants and the reference.

Each is evaluated in code and reported pass/fail with its numbers, rather than
being read off a table by eye.

A warning about criterion 3
---------------------------
Within-group spread is a property of the *recorded rollouts*, not only of the
reward. `LiteLLMProposer.propose` sends no `temperature` and `run_e2e.py`
injected none, so the engine decoded at its default and every group's five
completions came back byte-identical. A reward is a function of the completion,
so identical completions earn identical rewards and the within-group spread is
exactly zero for any reward whatsoever. `distinct_completions` in the output is
what makes that visible; `--check-determinism` fails the run when it holds.

`spread_across_on_contract_policies` is reported alongside as the honest
substitute: the range the reward achieves on one scenario across the policies
that clear the format gate. It shows the reward has regained discriminative
power, and it is *not* the same measurement as within-group spread -- it needs
several policies, whereas GRPO needs one policy sampled several times. Do not
report it as if it satisfied criterion 3.

Scoring the fix half against rollouts that have no matrix
---------------------------------------------------------
`fix_reward.py` needs an archived probe matrix to know which mitigation
resolved a scenario. The recorded rollouts here do not have one: all nine
scenarios are `mode: sanitizer` recipes, which archive a `sanitizer_report.json`
and no `{mitigation}-{diagnostic}` cells at all. So the fix half cannot simply
be switched on over this data, and pointing `--matrix` at an unrelated run
would be fabricating a ground truth.

`--resolver-sweep` is the honest instrument instead. It enumerates *every*
hypothesis about which single offered mitigation resolves the failure -- plus
the hypothesis that none does -- and reports the criteria under each. Nothing
is chosen, so nothing can be chosen favourably; what comes out is the range of
outcomes the term could produce, and whether any hypothesis at all flips
criterion 1. A sweep in which no hypothesis flips it is a stronger result than
a single hypothesis that does, and in the opposite direction.

`--resolvers` scores one named hypothesis, and says HYPOTHETICAL in its output
for the same reason. `--matrix` is the real path, for a corpus built from probe
runs.

Usage
-----

    python examples/rl/rescore_e2e.py results/faithful.json
    python examples/rl/rescore_e2e.py results/*.json --json
    python examples/rl/rescore_e2e.py results/*.json --resolver-sweep
    python examples/rl/rescore_e2e.py results/probe.json --matrix runs/TICKET
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_reward import (  # noqa: E402
    FIX_WEIGHT,
    Resolution,
    composite_reward,
    fix_credit,
    resolution_from_matrix,
)
from proposal_reward import (  # noqa: E402
    ABSTENTION_CATEGORY,
    Proposal,
    Score,
    score_proposal,
)

# The category the contract-perfect reference commits to.
#
# Deliberately arbitrary, and the arbitrariness is the finding. This reward
# scores category *membership*, not correctness, so no choice here is better
# than another as far as the score can tell. `checkpoint_race` is the name the
# real model picked for `consan-racy` -- the nearest available label for a
# kernel-level data race, and still the wrong one, because the closed set has no
# category for that failure. So the "perfect" reference is a confidently wrong
# label on 8 of 9 scenarios, which is the labelling blocker showing through:
# until it is fixed, the top of this ladder is not a correct answer.
REFERENCE_CATEGORY = "checkpoint_race"

# "at or near the top" for criterion 2.
NEAR_TOP = 0.99


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def runnable_names(raw: str, offered: list[str]) -> list[str]:
    """The names the loop would actually turn into probe cells.

    The fix half asks whether a *run* mitigation resolved the failure, so it
    has to score what the consumer would run, not what the model wrote. Both
    filters the real path applies are replayed: an unparseable reply yields
    nothing, and `LiteLLMProposer.propose` drops every name outside `offered`
    before the loop sees it. A hallucinated name therefore earns no fix credit
    by construction rather than by a rule stated here -- it never becomes a
    cell, so it cannot resolve anything.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    proposed = obj.get("next_mitigations")
    if not isinstance(proposed, list):
        return []
    return [str(m) for m in proposed if str(m) in offered]


def _body(
    category: str, mitigations: list[str], hypothesis: str, confidence: float
) -> str:
    return json.dumps(
        {
            "category": category,
            "hypothesis": hypothesis,
            "next_mitigations": mitigations,
            "confidence": confidence,
            "stop": False,
        }
    )


def reference_policies(
    offered: list[str], resolution: Resolution | None = None
) -> dict[str, str]:
    """The fixed policies the model is read against, on its own loop state.

    None of the *constants* reads the input: each emits the same object for
    every scenario. That is what makes them the right comparison -- where a
    constant ties or beats the model, the reward has not measured the model.

    The oracle is the exception, and it has to move when the fix half is on.
    Under a form-only reward "contract-perfect" was the best attainable score,
    so naming an arbitrary available mitigation was the ceiling. Once the
    reward checks whether the mitigation *worked*, an arbitrary name is no
    longer a ceiling -- it is a coin flip -- and criterion 2 would be measuring
    whether `offered[0]` happens to be the right answer. So when a resolution
    is supplied the oracle names a resolver, which is what "oracle" meant all
    along.
    """
    oracle_names = offered[:1]
    if resolution is not None and resolution.scoreable:
        in_reach = sorted(n for n in resolution.resolvers if n in offered)
        if in_reach:
            oracle_names = in_reach[:1]
    return {
        # The ceiling the reward admits: commit to a category, name one
        # available mitigation, say something. Under the form-only reward this
        # is not a *correct* answer (see REFERENCE_CATEGORY), just the
        # best-scoring one; with the fix half on, the mitigation is correct and
        # the category still is not.
        "oracle_contract_perfect": _body(
            REFERENCE_CATEGORY,
            oracle_names,
            "Single most likely cause given the evidence; one mitigation to test it.",
            0.8,
        ),
        # The best answer available to a policy that is honest about this
        # corpus, where 8 of 9 scenarios have no correct category in the closed
        # set. The gap between this and the row above is the price the reward
        # currently puts on declining, and it is worth looking at directly.
        "honest_abstainer": _body(
            ABSTENTION_CATEGORY,
            offered[:1],
            "No category in the closed set covers this failure; one mitigation to probe it.",
            0.4,
        ),
        # The two two-line constants from the first run, verbatim.
        "abstain_and_shotgun": _body(ABSTENTION_CATEGORY, offered, "", 0.5),
        "abstain_and_pick_first": _body(ABSTENTION_CATEGORY, offered[:1], "", 0.5),
        "always_prose": "not JSON, just a sentence.",
    }


def rescore_recorded(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-score every recorded proposal from its raw completion text.

    Reads `raw`, never the stored `reward`: the stored value is the "before" and
    has to stay untouched for the comparison to mean anything.
    """
    meta = doc["meta"]
    candidates = list(meta["candidates"])
    tried = list(meta["tried"])

    out: list[dict[str, Any]] = []
    for row in doc["proposals"]:
        score = score_proposal(
            Proposal(
                name=f"{row['scenario_id']}:sample{row['sample']}",
                raw=row["raw"],
                candidates=candidates,
                tried=tried,
            )
        )
        out.append(
            {
                "scenario_id": row["scenario_id"],
                "sample": row["sample"],
                "raw": row["raw"],
                "reward_before": row["reward"],
                "tier_before": row["tier"],
                "reward_after": round(score.reward, 4),
                **{k: v for k, v in score.as_dict().items() if k != "reward"},
            }
        )
    return out


def score_reference(raw: str, candidates: list[str], tried: list[str]) -> Score:
    return score_proposal(Proposal("reference", raw, candidates, tried))


def group_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """One group per scenario -- the unit a GRPO advantage is computed over."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["scenario_id"], []).append(row)
    return dict(sorted(groups.items()))


def hypothetical_resolution(resolvers: list[str], scenario_id: str = "hypothetical") -> Resolution:
    """A `Resolution` asserted rather than read off an archived matrix.

    Carries `source="hypothetical"` so every consumer can say so in its own
    output. A hypothesis is only worth printing next to a sweep of the other
    hypotheses, which is what `--resolver-sweep` is for.
    """
    return Resolution(
        scenario_id=scenario_id,
        resolvers=frozenset(resolvers),
        baseline_failed=True,
        cells=0,
        source="hypothetical",
    )


@dataclass
class FormPass:
    """The half of the analysis that does not depend on the fix-half ground truth.

    Split out because it is the expensive half -- every proposal walks the tier
    ladder and every mitigation name hits the registry -- and because
    :func:`resolver_sweep` scores the same rollouts against N+2 different
    hypotheses. Recomputing the ladder each time cost minutes and could not
    change any of its answers: the form score is a function of the completion
    alone.
    """

    meta: dict[str, Any]
    candidates: list[str]
    tried: list[str]
    offered: list[str]
    rows: list[dict[str, Any]]
    _reference_cache: dict[str, Score] = field(default_factory=dict)

    def reference_score(self, raw: str) -> Score:
        """Score one reference policy's output, memoised on its raw text.

        The oracle's text changes with the fix-half hypothesis while the four
        constants' does not, so a sweep re-scores one policy and reuses four.
        """
        if raw not in self._reference_cache:
            self._reference_cache[raw] = score_reference(raw, self.candidates, self.tried)
        return self._reference_cache[raw]


def form_pass(doc: dict[str, Any]) -> FormPass:
    """Score every recorded completion, the half that no hypothesis can change."""
    meta = doc["meta"]
    candidates = list(meta["candidates"])
    tried = list(meta["tried"])
    return FormPass(
        meta=meta,
        candidates=candidates,
        tried=tried,
        offered=[c for c in candidates if c not in tried and c != "none"],
        rows=rescore_recorded(doc),
    )


def analyse(
    doc: dict[str, Any],
    resolution: Resolution | None = None,
    fix_weight: float = FIX_WEIGHT,
    form: FormPass | None = None,
) -> dict[str, Any]:
    """Score the recorded rollouts, optionally with the execution-checked half.

    With `resolution` omitted this is the form-only reward and every number is
    what it was before the fix half existed. With one supplied and scoreable,
    every reward becomes `composite_reward(form, fix)` -- including the
    reference policies' -- and the four criteria are evaluated on that.

    `form` is an optional precomputed :func:`form_pass`, so a caller scoring
    one set of rollouts against several hypotheses pays for the tier ladder
    once.
    """
    form = form or form_pass(doc)
    meta, offered = form.meta, form.offered

    # Copied because the fix half stamps per-row keys onto them and a shared
    # FormPass is scored repeatedly; without this, hypothesis N+1 would read
    # hypothesis N's credit.
    rows = [dict(row) for row in form.rows]
    fix_active = resolution is not None and resolution.scoreable
    if fix_active:
        assert resolution is not None
        for row in rows:
            row["names_runnable"] = runnable_names(row["raw"], offered)
            row["fix_credit"] = fix_credit(row["names_runnable"], resolution)
            row["composite"] = round(
                composite_reward(row["reward_after"], row["fix_credit"], fix_weight), 4
            )
    # The single column every downstream reader means by "the reward". It is
    # the form score until the fix half is switched on, and the blend after,
    # so the criteria code below is written once rather than twice.
    scored_key = "composite" if fix_active else "reward_after"
    for row in rows:
        row["scored"] = row[scored_key]
    groups = group_by_scenario(rows)

    reference_raw = reference_policies(offered, resolution if fix_active else None)
    references = {name: form.reference_score(raw) for name, raw in reference_raw.items()}
    reference_means = {name: score.reward for name, score in references.items()}
    reference_fix: dict[str, float] = {}
    if fix_active:
        assert resolution is not None
        for name, raw in reference_raw.items():
            credit = fix_credit(runnable_names(raw, offered), resolution)
            reference_fix[name] = credit
            reference_means[name] = round(
                composite_reward(reference_means[name], credit, fix_weight), 4
            )
    # Policies that clear the format gate. `always_prose` is excluded from the
    # spread below because its 0.0 is a *parse* failure: leaving it in makes
    # every scenario's range 1.0 and hides whether the reward separates the
    # policies that are actually on contract, which is the only question here.
    on_contract = {
        name: reward
        for name, reward in reference_means.items()
        if references[name].tier >= 3
    }

    model_mean = _mean([r["scored"] for r in rows])
    model_mean_before = _mean([r["reward_before"] for r in rows])

    # Per-scenario: the within-group spread GRPO needs, the number of distinct
    # completions that spread could possibly come from, and the range the reward
    # achieves across the policy set on the same scenario.
    per_scenario: dict[str, Any] = {}
    for scenario, members in groups.items():
        rewards = [m["scored"] for m in members]
        across = list(on_contract.values()) + [_mean(rewards)]
        per_scenario[scenario] = {
            "n": len(members),
            "distinct_completions": len({m["raw"] for m in members}),
            "mean": round(_mean(rewards), 4),
            "min": round(min(rewards), 4),
            "max": round(max(rewards), 4),
            "spread_within_group": round(max(rewards) - min(rewards), 4),
            "spread_within_group_before": round(
                max(m["reward_before"] for m in members)
                - min(m["reward_before"] for m in members),
                4,
            ),
            "spread_across_on_contract_policies": round(max(across) - min(across), 4),
        }

    constants = ["abstain_and_shotgun", "abstain_and_pick_first"]
    top = reference_means["oracle_contract_perfect"]
    best_constant = max(reference_means[c] for c in constants)

    criteria = {
        "1_constants_below_model": {
            "holds": all(reference_means[c] < model_mean for c in constants),
            "detail": {c: round(reference_means[c], 4) for c in constants}
            | {"model": round(model_mean, 4)},
        },
        "2_reference_at_top": {
            "holds": top >= NEAR_TOP,
            "detail": {"oracle_contract_perfect": round(top, 4), "threshold": NEAR_TOP},
        },
        "3_within_group_spread_nonzero": {
            "holds": all(
                v["spread_within_group"] > 0.0 for v in per_scenario.values()
            ),
            "detail": {
                "groups": len(per_scenario),
                "groups_with_spread": sum(
                    1 for v in per_scenario.values() if v["spread_within_group"] > 0.0
                ),
                "groups_with_one_distinct_completion": sum(
                    1 for v in per_scenario.values() if v["distinct_completions"] == 1
                ),
            },
        },
        "4_model_between": {
            "holds": best_constant < model_mean <= top,
            "detail": {
                "best_constant": round(best_constant, 4),
                "model": round(model_mean, 4),
                "oracle_contract_perfect": round(top, 4),
            },
        },
    }

    return {
        "condition": meta.get("condition"),
        "model": meta.get("model"),
        "offered_count": len(offered),
        "n": len(rows),
        "model_mean_before": round(model_mean_before, 4),
        "model_mean_after": round(model_mean, 4),
        "references": {
            name: {
                "reward": round(reference_means[name], 4),
                "form_reward": round(score.reward, 4),
                "fix_credit": reference_fix.get(name),
                "tier": score.tier,
                "category_credit": round(score.category_credit, 4),
                "precision": round(score.precision, 4),
                "n_mitigations": score.n_mitigations,
                "consumer_outcome": score.consumer_outcome,
            }
            for name, score in references.items()
        },
        "abstained": sum(1 for r in rows if r["category_credit"] < 1.0),
        "mean_mitigations": round(_mean([float(r["n_mitigations"]) for r in rows]), 4),
        "fix_half": {
            "active": fix_active,
            "weight": fix_weight if fix_active else None,
            "resolution": resolution.as_dict() if resolution is not None else None,
            "model_form_mean": round(_mean([r["reward_after"] for r in rows]), 4),
            "model_fix_rate": (
                round(_mean([r["fix_credit"] for r in rows]), 4) if fix_active else None
            ),
        },
        "per_scenario": per_scenario,
        "criteria": criteria,
    }


def print_analysis(result: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"Re-score: condition {result['condition']!r}, "
          f"{result['offered_count']} mitigations offered")
    print("=" * 78)
    print(f"  recorded proposals   {result['n']}")
    print(f"  mean reward before   {result['model_mean_before']:.4f}")
    print(f"  mean reward after    {result['model_mean_after']:.4f}")
    print(f"  abstained (unknown)  {result['abstained']}/{result['n']}")
    print(f"  mean names proposed  {result['mean_mitigations']:.2f}")
    fix = result["fix_half"]
    if fix["active"]:
        resolution = fix["resolution"] or {}
        origin = (
            "HYPOTHETICAL, not measured"
            if resolution.get("source") == "hypothetical"
            else f"archived matrix {resolution.get('source')}"
        )
        print()
        print(f"  fix half ACTIVE, weight {fix['weight']:g}")
        print(f"    resolving mitigations {resolution.get('resolvers')}  ({origin})")
        print(f"    model form mean       {fix['model_form_mean']:.4f}")
        print(f"    model fix rate        {fix['model_fix_rate']:.4f}"
              "   (share of samples naming a resolver)")
    print()
    print("  policy ladder (constants read no input at all)")
    ladder = sorted(
        result["references"].items(), key=lambda kv: -kv[1]["reward"]
    )
    printed_model = False
    for name, row in ladder:
        if not printed_model and row["reward"] < result["model_mean_after"]:
            print(f"    {'*** the model ***':<26} {result['model_mean_after']:.4f}")
            printed_model = True
        print(f"    {name:<26} {row['reward']:.4f}   tier {row['tier']}  "
              f"{row['n_mitigations']} name(s)")
    if not printed_model:
        print(f"    {'*** the model ***':<26} {result['model_mean_after']:.4f}")
    print()
    print("  per scenario")
    print(f"    {'scenario':<24} {'mean':>7} {'within':>7} {'was':>6} "
          f"{'across':>7} {'distinct':>9}")
    for scenario, row in result["per_scenario"].items():
        print(f"    {scenario:<24} {row['mean']:>7.4f} "
              f"{row['spread_within_group']:>7.4f} "
              f"{row['spread_within_group_before']:>6.3f} "
              f"{row['spread_across_on_contract_policies']:>7.4f} "
              f"{row['distinct_completions']:>4}/{row['n']:<4}")
    print()
    print("  acceptance criteria")
    for key, row in result["criteria"].items():
        mark = "PASS" if row["holds"] else "FAIL"
        print(f"    [{mark}] {key}")
        print(f"           {row['detail']}")


def resolver_sweep(doc: dict[str, Any], fix_weight: float = FIX_WEIGHT) -> dict[str, Any]:
    """Every hypothesis about which offered mitigation resolves the failure.

    The recorded rollouts have no archived matrix, so the fix half has no
    ground truth to read. Rather than assert one, this enumerates all of them:
    each offered mitigation as the sole resolver, plus the null hypothesis that
    none of them resolves anything -- which is the live possibility here, since
    the offered names on this corpus are diagnostic and serialisation toggles
    that cannot repair a source-level LDS race.

    Enumerating is what makes the result honest. Nothing is selected, so a
    favourable hypothesis cannot be selected, and the reported answer is the
    *range* of outcomes -- specifically, whether **any** assignment of the
    ground truth flips criterion 1. "None does" is a much stronger finding than
    "one of them does", and it is the finding this sweep is most likely to
    produce, because a constant that shotguns the offered set names every
    possible resolver by construction.
    """
    form = form_pass(doc)
    meta, offered = form.meta, form.offered

    rows: list[dict[str, Any]] = []
    baseline = analyse(doc, form=form)
    rows.append(
        {
            "hypothesis": "(none: form half only)",
            "model": baseline["criteria"]["1_constants_below_model"]["detail"]["model"],
            "best_constant": baseline["criteria"]["4_model_between"]["detail"][
                "best_constant"
            ],
            "criteria_passing": sum(
                1 for c in baseline["criteria"].values() if c["holds"]
            ),
            "criterion_1": baseline["criteria"]["1_constants_below_model"]["holds"],
        }
    )
    # The null hypothesis is scored by hand rather than through `analyse`,
    # because `Resolution.scoreable` correctly refuses it: with no resolver,
    # the term is a constant 0 on every policy. The row is still printed --
    # "the term is withheld" is an outcome the reader has to see.
    rows.append(
        {
            "hypothesis": "nothing resolves it",
            "model": None,
            "best_constant": None,
            "criteria_passing": None,
            "criterion_1": False,
            "note": "term withheld: no right answer exists, so it is a constant",
        }
    )
    for name in offered:
        result = analyse(doc, hypothetical_resolution([name]), fix_weight, form=form)
        criteria = result["criteria"]
        rows.append(
            {
                "hypothesis": f"{name} resolves it",
                "model": criteria["1_constants_below_model"]["detail"]["model"],
                "best_constant": criteria["4_model_between"]["detail"]["best_constant"],
                "model_fix_rate": result["fix_half"]["model_fix_rate"],
                "criteria_passing": sum(1 for c in criteria.values() if c["holds"]),
                "criterion_1": criteria["1_constants_below_model"]["holds"],
            }
        )
    return {
        "condition": meta.get("condition"),
        "offered": offered,
        "fix_weight": fix_weight,
        "any_hypothesis_flips_criterion_1": any(r["criterion_1"] for r in rows[1:]),
        "hypotheses": rows,
    }


def print_sweep(sweep: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"Resolver sweep: condition {sweep['condition']!r}, "
          f"{len(sweep['offered'])} mitigations offered, "
          f"fix weight {sweep['fix_weight']:g}")
    print("=" * 78)
    print("  No ground truth exists for these scenarios, so every hypothesis about")
    print("  which mitigation resolves them is scored and none is chosen.")
    print()
    print(f"    {'hypothesis':<44} {'model':>7} {'best const':>11} {'fix':>6} "
          f"{'crit 1':>7} {'of 4':>5}")
    for row in sweep["hypotheses"]:
        model = "  --   " if row["model"] is None else f"{row['model']:>7.4f}"
        const = "     --    " if row["best_constant"] is None else f"{row['best_constant']:>11.4f}"
        rate = row.get("model_fix_rate")
        rate_s = "   -- " if rate is None else f"{rate:>6.2f}"
        passing = row["criteria_passing"]
        passing_s = "   --" if passing is None else f"{passing:>5}"
        mark = "PASS" if row["criterion_1"] else "fail"
        print(f"    {row['hypothesis']:<44} {model} {const} {rate_s} {mark:>7} {passing_s}")
        if row.get("note"):
            print(f"      {row['note']}")
    print()
    verdict = (
        "at least one hypothesis flips criterion 1"
        if sweep["any_hypothesis_flips_criterion_1"]
        else "NO hypothesis flips criterion 1 -- the fix half does not rescue this set"
    )
    print(f"  Result: {verdict}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="+",
                        help="results JSON written by run_e2e.py")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="exit non-zero if any group's completions are byte-identical, which "
             "makes within-group spread unreachable for any reward",
    )
    parser.add_argument(
        "--matrix", type=Path, default=None,
        help="an archived probe run directory; switches on the execution-checked "
             "fix half with ground truth read from the matrix",
    )
    parser.add_argument(
        "--resolvers", default=None,
        help="comma-separated mitigation names asserted to resolve the failure. "
             "A hypothesis, not a measurement, and labelled as one in the output",
    )
    parser.add_argument(
        "--resolver-sweep", action="store_true",
        help="score every hypothesis about which offered mitigation resolves the "
             "failure, and report whether any of them flips criterion 1",
    )
    parser.add_argument(
        "--fix-weight", type=float, default=FIX_WEIGHT,
        help=f"weight on the fix half (default {FIX_WEIGHT})",
    )
    args = parser.parse_args(argv)

    if args.matrix and args.resolvers:
        parser.error("--matrix and --resolvers are two sources for one fact; pick one")

    resolution: Resolution | None = None
    if args.matrix:
        resolution = resolution_from_matrix(args.matrix)
        if not resolution.scoreable:
            print(f"{args.matrix}: fix half withheld -- {resolution.withheld_because}",
                  file=sys.stderr)
    elif args.resolvers:
        resolution = hypothetical_resolution(
            [n.strip() for n in args.resolvers.split(",") if n.strip()]
        )

    if args.resolver_sweep:
        sweeps = []
        for path in args.results:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not doc.get("proposals"):
                print(f"no proposals in {path}", file=sys.stderr)
                continue
            sweep = resolver_sweep(doc, args.fix_weight)
            sweep["source"] = str(path)
            sweeps.append(sweep)
        if not sweeps:
            return 2
        if args.json:
            print(json.dumps(sweeps, indent=2))
        else:
            for sweep in sweeps:
                print_sweep(sweep)
        return 0

    results = []
    for path in args.results:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not doc.get("proposals"):
            print(f"no proposals in {path}", file=sys.stderr)
            continue
        result = analyse(doc, resolution, args.fix_weight)
        result["source"] = str(path)
        results.append(result)

    if not results:
        return 2

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print_analysis(result)

    if args.check_determinism:
        collapsed = [
            (r["source"], scenario)
            for r in results
            for scenario, row in r["per_scenario"].items()
            if row["distinct_completions"] == 1
        ]
        if collapsed:
            print(
                f"\n{len(collapsed)} group(s) have a single distinct completion; "
                "within-group spread is unreachable for any reward. Set a "
                "temperature on the rollout.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
