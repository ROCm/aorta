#!/usr/bin/env python3
"""Seam demonstration: a reward for the shape of an `aorta agent` proposal.

The outer layer of the debugging-vertical reward, and the cheapest one: does a
proposal satisfy the contract `aorta agent` actually demands of an LLM? Strict
JSON, a `category` from the closed autopsy set, and mitigation names that
resolve in the registry and sit inside the candidate set the loop offered.

Zero GPU, microseconds per sample, and no labelling: the contract is code.

Why this is scored at all, when the consumer already validates
--------------------------------------------------------------
Because the consumer validates *quietly*. `LiteLLMProposer.propose` drops
unrecognised mitigation names before the policy ever sees them::

    filtered = [m for m in step.next_mitigations if m in remaining]

A proposal naming only mitigations that do not exist therefore arrives at the
loop as a well-formed step with an empty `next_mitigations`, and
`run_agent_loop` reads an empty list as a decision to stop searching. The
outcome is `agent_stop`, carrying the model's own hypothesis as the operator's
recommended action. Nothing raises, nothing logs a rejection, and the search
ends early on a fabricated name that looked plausible.

`AgentStep.from_dict` coerces on the same principle: a non-bool `stop` becomes
`False`, a non-list `next_mitigations` becomes `[]`, a null `category` becomes
`"unknown"`, and an unparseable numeric `confidence` becomes `0.0`. Those
defences are right for a *serving* path -- the audit trail must survive a bad
provider -- but they mean the consumer cannot be used as an oracle for training.
It reports success on input it silently repaired. A reward has to measure the
contract as stated, which is what this does.

The one thing the consumer does reject loudly is a category outside the closed
set: `AgentPolicy.validate_step` raises `PolicyViolation`, the loop catches it,
and the outcome is `policy_stop`. So a bad category costs the whole search too,
just more visibly.

Scored through aorta's own code
-------------------------------
The tiers call `AgentStep.from_dict`, `AgentPolicy.validate_step` and
`aorta.registry.get_mitigation` rather than reimplementing them, for the same
reason `triage_reward.py` recomputes labels through the verdict resolver: a
reward that restates the contract drifts from it, and a drifted reward still
trains. If the autopsy set gains a category or the registry gains a mitigation,
this reward changes in that commit.

The ladder
----------
Graded, not pass/fail, so a policy that is nearly right gets a gradient::

    1  parses as a JSON object                                        0.2
    2  the five demanded keys are present with the demanded types     0.4
    3  `category` is in the closed autopsy set                        0.6
    4  a non-empty mitigation list, every name in the registry        0.8
    5  every name inside the offered candidates, confidence in [0,1]  1.0

Tier 4 is the one that separates a useful proposal from a plausible one. Tier 5
is the difference between a name that exists and a name that is *available*:
proposing an already-tried or non-allowlisted mitigation is silently filtered,
so it costs a wasted iteration in exactly the way tier 4 costs a wasted search.

`consumer_outcome` records what `run_agent_loop` would do with each proposal --
`accepted`, `silent_stop`, or `policy_stop` -- so the reward can be read against
its real consequence rather than as an abstract score.

What this deliberately does not score
-------------------------------------
Whether the category is *correct* for the failure (that is `triage_reward.py`)
and whether the mitigation actually fixes the repro (that needs a probe cell and
a GPU). This is the format half only. A policy can score 1.0 here while being
diagnostically useless, which is precisely why it is the outer layer and not the
reward.

Usage
-----

    python examples/rl/proposal_reward.py           # fixtures + baselines
    python examples/rl/proposal_reward.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from aorta.agent.llm import AUTOPSY_CATEGORIES, AgentStep
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.registry import get_mitigation
from aorta.registry.errors import UnknownMitigationError

MAX_TIER = 5

# The keys the system prompt demands, and the type each must have. `stop_reason`
# is optional and is not part of the demanded set.
REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "category": str,
    "hypothesis": str,
    "next_mitigations": list,
    "confidence": (int, float),
    "stop": bool,
}


@dataclass
class Proposal:
    """One model output, plus the loop state it was produced against."""

    name: str
    raw: str
    candidates: list[str] = field(default_factory=list)
    tried: list[str] = field(default_factory=list)

    @property
    def offered(self) -> list[str]:
        """What `propose` would have put in front of the model.

        Mirrors the proposer: the candidate set minus what has been tried,
        minus the no-op baseline.
        """
        return [c for c in self.candidates if c not in self.tried and c != "none"]


@dataclass
class Score:
    tier: int = 0
    reward: float = 0.0
    stopped_at: str = ""
    detail: str = ""
    consumer_outcome: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "reward": round(self.reward, 4),
            "stopped_at": self.stopped_at,
            "detail": self.detail,
            "consumer_outcome": self.consumer_outcome,
        }


def _consumer_outcome(raw_obj: dict[str, Any], offered: list[str]) -> str:
    """What `run_agent_loop` would do with this proposal.

    Replays the two filters the real path applies, in order: the proposer's
    silent name filter, then the policy's category/registry check.
    """
    step = AgentStep.from_dict(raw_obj)
    filtered = [m for m in step.next_mitigations if m in offered]
    step = AgentStep(
        category=step.category,
        hypothesis=step.hypothesis,
        next_mitigations=filtered,
        confidence=step.confidence,
        stop=step.stop,
        stop_reason=step.stop_reason,
    )
    try:
        AgentPolicy().validate_step(step)
    except PolicyViolation:
        return "policy_stop"
    if step.stop or not step.next_mitigations:
        return "silent_stop"
    return "accepted"


def score_proposal(proposal: Proposal) -> Score:
    """Walk the ladder, stopping at the first tier that fails."""
    score = Score()

    # Tier 1 -- strict JSON object. `response_format=json_object` asks the
    # provider for this, but providers return partial and non-object JSON, and
    # the consumer's own except-clause exists because of it.
    try:
        raw_obj = json.loads(proposal.raw)
    except json.JSONDecodeError as exc:
        score.stopped_at = "tier1_json"
        score.detail = f"does not parse: {exc.msg}"
        score.consumer_outcome = "silent_stop"
        return score
    if not isinstance(raw_obj, dict):
        score.stopped_at = "tier1_json"
        score.detail = f"parsed as {type(raw_obj).__name__}, not an object"
        score.consumer_outcome = "silent_stop"
        return score
    score.tier = 1
    score.consumer_outcome = _consumer_outcome(raw_obj, proposal.offered)

    # Tier 2 -- the demanded keys, with the demanded types. Checked against the
    # raw object rather than the coerced AgentStep: from_dict would have
    # repaired a wrong type into a plausible default, which is the behaviour
    # this tier exists to catch.
    missing = [k for k in REQUIRED_KEYS if k not in raw_obj]
    if missing:
        score.stopped_at = "tier2_schema"
        score.detail = f"missing key(s): {sorted(missing)}"
        return _finish(score)
    mistyped = [
        f"{k}={type(raw_obj[k]).__name__}"
        for k, want in REQUIRED_KEYS.items()
        # bool is a subclass of int; a bare `True` confidence is not a number.
        if not isinstance(raw_obj[k], want) or (k == "confidence" and isinstance(raw_obj[k], bool))
    ]
    if mistyped:
        score.stopped_at = "tier2_schema"
        score.detail = f"wrong type(s): {sorted(mistyped)}"
        return _finish(score)
    score.tier = 2

    # Tier 3 -- the closed category set. The only thing the consumer rejects
    # loudly, via PolicyViolation.
    category = raw_obj["category"]
    if category not in AUTOPSY_CATEGORIES:
        score.stopped_at = "tier3_category"
        score.detail = f"category {category!r} not in the autopsy set"
        return _finish(score)
    score.tier = 3

    # Tier 4 -- names that exist. An empty list is a decision to stop, so it
    # fails here too: a proposal that names nothing is not a proposal.
    names = [str(m) for m in raw_obj["next_mitigations"]]
    if not names:
        score.stopped_at = "tier4_registry"
        score.detail = "no mitigation proposed; the loop reads this as a stop"
        return _finish(score)
    unknown: list[str] = []
    for name in names:
        try:
            get_mitigation(name)
        except UnknownMitigationError:
            unknown.append(name)
    if unknown:
        score.stopped_at = "tier4_registry"
        score.detail = (
            f"unregistered mitigation(s) {sorted(unknown)}; "
            "silently dropped by the proposer"
        )
        return _finish(score)
    score.tier = 4

    # Tier 5 -- names that are available, and a usable confidence. Registered
    # but not offered is still silently dropped.
    unavailable = [n for n in names if n not in proposal.offered]
    if unavailable:
        score.stopped_at = "tier5_available"
        score.detail = (
            f"mitigation(s) {sorted(unavailable)} are registered but were not "
            "offered (already tried, or outside the allowlist)"
        )
        return _finish(score)
    confidence = float(raw_obj["confidence"])
    if not 0.0 <= confidence <= 1.0:
        score.stopped_at = "tier5_available"
        score.detail = f"confidence {confidence} outside [0, 1]"
        return _finish(score)
    score.tier = MAX_TIER
    score.detail = "on contract"
    return _finish(score)


def _finish(score: Score) -> Score:
    score.reward = score.tier / MAX_TIER
    return score


# --------------------------------------------------------------------------- #
# Fixtures
#
# Every failure mode below is one an LLM actually produces on this prompt, and
# each is drawn from the debugging vertical rather than being generically
# malformed. The candidate set is the real registry's, narrowed the way an
# operator narrows it with --mitigation.
# --------------------------------------------------------------------------- #

_CANDIDATES = [
    "nccl_launch_order_implicit",
    "hsa_no_sdma",
    "gpu_max_hw_queues_2",
    "hip_launch_blocking",
    "pytorch_alloc_expandable_segments",
    "tf32_off",
]


def _ok(**over: Any) -> str:
    body: dict[str, Any] = {
        "category": "rccl_hang",
        "hypothesis": "tier4:collective_timeout on all ranks; suspect launch ordering.",
        "next_mitigations": ["nccl_launch_order_implicit"],
        "confidence": 0.7,
        "stop": False,
    }
    body.update(over)
    return json.dumps(body)


def _without(key: str) -> dict[str, Any]:
    body = json.loads(_ok())
    body.pop(key)
    return body


FIXTURES: tuple[Proposal, ...] = (
    Proposal("on-contract rccl hang proposal", _ok(), _CANDIDATES),
    Proposal(
        "on-contract oom proposal",
        _ok(
            category="oom_fragment",
            hypothesis="exit 137 with vram growth; fragmentation, not a true OOM.",
            next_mitigations=["pytorch_alloc_expandable_segments"],
        ),
        _CANDIDATES,
    ),
    # Prose around the object is the commonest real failure on models that do
    # not honour response_format.
    Proposal(
        "JSON wrapped in prose",
        "Here is my analysis:\n" + _ok(),
        _CANDIDATES,
    ),
    Proposal("truncated JSON", _ok()[:-3], _CANDIDATES),
    Proposal("a JSON list, not an object", '[{"category": "rccl_hang"}]', _CANDIDATES),
    Proposal(
        "confidence as a string",
        _ok(confidence="high"),
        _CANDIDATES,
    ),
    Proposal(
        "stop as a string",
        _ok(stop="false"),
        _CANDIDATES,
    ),
    Proposal("hypothesis omitted", json.dumps(_without("hypothesis")), _CANDIDATES),
    # A category that reads like a real one but is not in the closed set.
    Proposal(
        "invented category",
        _ok(category="rccl_timeout"),
        _CANDIDATES,
    ),
    Proposal(
        "free-text category",
        _ok(category="RCCL hang on rank 3"),
        _CANDIDATES,
    ),
    # The dangerous one: a plausible env-var name that is not registered.
    Proposal(
        "hallucinated mitigation",
        _ok(next_mitigations=["rccl_p2p_disable"]),
        _CANDIDATES,
    ),
    Proposal(
        "shell command as a mitigation",
        _ok(next_mitigations=["export NCCL_P2P_DISABLE=1"]),
        _CANDIDATES,
    ),
    Proposal(
        "one real name, one invented",
        _ok(next_mitigations=["nccl_launch_order_implicit", "rccl_disable_p2p"]),
        _CANDIDATES,
    ),
    Proposal("empty mitigation list", _ok(next_mitigations=[]), _CANDIDATES),
    # Registered, but already tried: silently dropped, so the iteration is spent
    # re-proposing something the loop has already ruled out.
    Proposal(
        "re-proposes an already-tried mitigation",
        _ok(next_mitigations=["hsa_no_sdma"]),
        _CANDIDATES,
        tried=["hsa_no_sdma"],
    ),
    # Registered, but outside what the operator allowed.
    Proposal(
        "proposes outside the allowlist",
        _ok(next_mitigations=["xnack"]),
        _CANDIDATES,
    ),
    Proposal("confidence out of range", _ok(confidence=42.0), _CANDIDATES),
)


def _fixture_expectations() -> dict[str, int]:
    """The tier each fixture should reach, asserted by the test suite."""
    return {
        "on-contract rccl hang proposal": 5,
        "on-contract oom proposal": 5,
        "JSON wrapped in prose": 0,
        "truncated JSON": 0,
        "a JSON list, not an object": 0,
        "confidence as a string": 1,
        "stop as a string": 1,
        "hypothesis omitted": 1,
        "invented category": 2,
        "free-text category": 2,
        "hallucinated mitigation": 3,
        "shell command as a mitigation": 3,
        "one real name, one invented": 3,
        "empty mitigation list": 3,
        "re-proposes an already-tried mitigation": 4,
        "proposes outside the allowlist": 4,
        "confidence out of range": 4,
    }


def baselines() -> list[dict[str, Any]]:
    """Degenerate policies, so a real score is read against something.

    The first is the one that matters: a policy that always returns the same
    on-contract proposal scores 1.0 here, because this reward measures form and
    nothing else. That is the ceiling a format reward can give you, and the
    reason it cannot be the only term.
    """
    rows = []
    for name, raw in (
        ("always the same valid proposal", _ok()),
        ("always an empty object", "{}"),
        ("always prose", "The RCCL collective timed out on rank 3."),
    ):
        scores = [
            score_proposal(Proposal(name, raw, f.candidates, f.tried))
            for f in FIXTURES
        ]
        n = len(scores) or 1
        rows.append(
            {
                "policy": name,
                "proposals": len(scores),
                "mean_reward": round(sum(s.reward for s in scores) / n, 4),
                "accepted_rate": round(
                    sum(s.consumer_outcome == "accepted" for s in scores) / n, 4
                ),
            }
        )
    return rows


def run_demo(as_json: bool) -> int:
    scored = [(f, score_proposal(f)) for f in FIXTURES]
    if as_json:
        print(
            json.dumps(
                {
                    "proposals": [
                        {"name": f.name, **s.as_dict()} for f, s in scored
                    ],
                    "baselines": baselines(),
                },
                indent=2,
            )
        )
        return 0

    print("=" * 72)
    print("The proposal contract: what `aorta agent` demands of a model")
    print("=" * 72)
    print(
        "reward = tier/5. Tiers 1-2 are form, 3 is the closed category set,\n"
        "4-5 are registry membership and availability.\n"
    )
    for f, s in scored:
        print(f"tier {s.tier}/{MAX_TIER}  reward {s.reward:.2f}  {f.name}")
        print(f"       consumer would: {s.consumer_outcome}")
        if s.stopped_at:
            print(f"       stopped at {s.stopped_at}: {s.detail}")
        elif s.detail:
            print(f"       {s.detail}")
        print()

    print("=" * 72)
    print("Degenerate policies")
    print("=" * 72)
    for row in baselines():
        print(
            f"  {row['policy']:<34} mean reward {row['mean_reward']:.2f}  "
            f"accepted {row['accepted_rate']:.2f}"
        )
    print(
        "\nThe first baseline is the point: a fixed on-contract proposal scores\n"
        "1.00 without diagnosing anything. Form is a gate, not a signal --\n"
        "pair it with triage_reward.py, which scores whether the read is right."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    return run_demo(args.json)


if __name__ == "__main__":
    sys.exit(main())
