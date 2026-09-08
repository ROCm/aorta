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

Usage
-----

    python examples/rl/rescore_e2e.py /apps/vikhande/rl-e2e/results/faithful.json
    python examples/rl/rescore_e2e.py results/*.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def reference_policies(offered: list[str]) -> dict[str, str]:
    """The fixed policies the model is read against, on its own loop state.

    None of them reads the input: every one emits the same object for every
    scenario. That is what makes them the right comparison -- where a constant
    ties or beats the model, the reward has not measured the model.
    """
    return {
        # The ceiling the contract admits: commit to a category, name one
        # available mitigation, say something. Not a *correct* answer (see
        # REFERENCE_CATEGORY) -- the best-scoring one.
        "oracle_contract_perfect": _body(
            REFERENCE_CATEGORY,
            offered[:1],
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


def analyse(doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc["meta"]
    candidates = list(meta["candidates"])
    tried = list(meta["tried"])
    offered = [c for c in candidates if c not in tried and c != "none"]

    rows = rescore_recorded(doc)
    groups = group_by_scenario(rows)

    references = {
        name: score_reference(raw, candidates, tried)
        for name, raw in reference_policies(offered).items()
    }
    reference_means = {name: score.reward for name, score in references.items()}
    # Policies that clear the format gate. `always_prose` is excluded from the
    # spread below because its 0.0 is a *parse* failure: leaving it in makes
    # every scenario's range 1.0 and hides whether the reward separates the
    # policies that are actually on contract, which is the only question here.
    on_contract = {
        name: reward
        for name, reward in reference_means.items()
        if references[name].tier >= 3
    }

    model_mean = _mean([r["reward_after"] for r in rows])
    model_mean_before = _mean([r["reward_before"] for r in rows])

    # Per-scenario: the within-group spread GRPO needs, the number of distinct
    # completions that spread could possibly come from, and the range the reward
    # achieves across the policy set on the same scenario.
    per_scenario: dict[str, Any] = {}
    for scenario, members in groups.items():
        rewards = [m["reward_after"] for m in members]
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
                "reward": round(score.reward, 4),
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
    args = parser.parse_args(argv)

    results = []
    for path in args.results:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not doc.get("proposals"):
            print(f"no proposals in {path}", file=sys.stderr)
            continue
        result = analyse(doc)
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
