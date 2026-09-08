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

Why the tier is no longer the whole reward
------------------------------------------
The tiers above are *membership* tests, and the first end-to-end run against a
real model showed that membership alone saturates: Qwen3-8B, a known-perfect
answer, and two two-line constants that read no input all scored 1.0 on all 45
recorded proposals, so no GRPO advantage existed to train on. See
``docs/tokenspeed-rl-e2e-sanitizer-routing.md``. Two of the tiers are therefore
graded *within* the tier rather than being pass/fail:

* **tier 3 is graded by what the category commits to.** ``unknown`` is a member
  of ``AUTOPSY_CATEGORIES`` and ``AgentPolicy`` accepts it, so declining to
  classify used to clear the tier that exists to test classification. It now
  earns ``ABSTENTION_CREDIT`` of the step instead of all of it.
* **tiers 4-5 are graded by the length of the mitigation list.** ``run_agent_loop``
  appends *every* proposed name to the mitigation axis and runs a probe cell for
  each, while charging the whole proposal a single unit of the iteration budget
  (``loop.py``, ``check_iteration_budget`` then the ``for mitigation in
  step.next_mitigations`` append). So a k-name proposal costs k GPU cells and the
  budget the policy enforces does not restrain it at all. ``precision_credit``
  prices that.

Two consequences worth stating plainly, because both are deliberate:

* The reward is no longer a pure function of the tier. A wide enough sweep at
  tier 5 can score below a precise proposal at tier 4, or below a tier-3 miss.
  That is the intended reading -- twenty cells is a real cost -- and it means
  the tier and the reward have to be reported separately, which ``Score`` does.
* Neither graded term can tell a *right* category or name from a wrong one,
  because nothing here has labels (that is the category-labelling blocker, and
  the mitigation half needs the probe cell this module explicitly does not run).
  Both terms therefore score form, and each is gameable in its own direction:
  see ``ABSTENTION_CREDIT`` and ``FREE_MITIGATIONS`` for which direction, and
  what was rejected.

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
from pathlib import Path
from typing import Any

from aorta.agent.llm import AUTOPSY_CATEGORIES, AgentStep
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.registry import get_mitigation
from aorta.registry.errors import UnknownMitigationError

MAX_TIER = 5
TIER_STEP = 1.0 / MAX_TIER

# The legal way to decline, and what declining is worth.
#
# `unknown` is in AUTOPSY_CATEGORIES and `AgentPolicy.validate_step` accepts it,
# so it has to stay legal here -- a reward that rejects what the consumer
# accepts is a reward that has drifted from the contract, which is the failure
# this module's docstring is built around. It just stops being worth full marks.
#
# The value is half a step, chosen as the neutral point rather than fitted:
# declining is worth half of committing. What matters is the ordering, not 0.5.
#
# Rejected, and why:
#
# * **Excluding `unknown` from tier 3's accepted set** (the first option in the
#   report's §5). It drops an abstention to 0.4 while any in-set category, right
#   or wrong, still earns 1.0 -- a 0.6 gradient pointing straight at "invent a
#   confident label". On this corpus that is not hypothetical: 8 of the 9
#   scenarios have no correct category available in the closed set, so the
#   policy it trains is one that emits a wrong label instead of an honest
#   `unknown`. Strictly worse for an operator, and worse for the agent loop,
#   which routes on the category.
# * **Gating the credit on "the evidence genuinely does not support a category"**
#   (the second option in §5). Not implementable: that predicate needs the
#   per-scenario category labels which do not exist. It is the labelling blocker
#   wearing a different hat, so §5 offered it as an alternative to fix 1 when it
#   is really a restatement of fix 3.
# * **Coupling the credit to `confidence`**, docking an abstention that also
#   claims certainty. Genuinely attractive -- incoherence is checkable without
#   labels, and it does not push the policy towards a confident wrong label,
#   because committing and abstaining stay equally available. Rejected on the
#   data: the recorded model abstains at confidence 0.6-0.95 while the two
#   constant templates abstain at exactly 0.5, so the term ranks a humble
#   two-line constant *above* the model on 8 of 9 scenarios. That inverts the
#   one comparison the whole exercise exists to make. Worth revisiting once a
#   correctness signal exists to anchor it.
#
# What survives the rejections is still not clean, and the honest statement of
# the residue is: partial credit keeps a wrong-but-specific label worth more
# than an honest abstention (a full step against half a step). It shrinks that
# perverse gradient rather than removing it, and it is the formulation that
# degrades gracefully -- when the category set is widened to cover kernel-level
# races, the same term becomes correctness-sensitive with no rewrite.
ABSTENTION_CATEGORY = "unknown"
ABSTENTION_CREDIT = 0.5

# How many mitigations a proposal may name before hedging starts costing.
#
# The cost model is the loop's, not a preference: every name becomes its own
# probe cell, so a k-name proposal is k GPU runs. A *pair* is the smallest hedge
# that survives one wrong guess without spending another proposal round, so it
# is priced free; past that the proposal is a sweep of the candidate set and is
# priced by its cell count.
#
# Two is also what neutralises the specific perversity a brevity term invites.
# With no correctness signal, "one wrong name" and "the right name" are
# indistinguishable, so any brevity term makes a 1-name proposal beat a 2-name
# proposal that contains the right answer. At FREE_MITIGATIONS = 2 that margin
# is exactly zero: the two tie. `1/len(next_mitigations)` -- the form the report
# suggested -- puts a 0.2 reward cliff there instead, which is the largest
# single step the term can produce and points the wrong way.
#
# It does not remove the perversity, it relocates it: at three names and up, a
# single confident wrong name still outscores a list containing the right one.
# That cannot be fixed by any function of the list's *shape*; it needs the
# contract's fix half, which is a probe cell and a GPU.
FREE_MITIGATIONS = 2

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


def category_credit(category: str) -> float:
    """What tier 3's step is worth for `category`.

    Full credit for committing to a category, `ABSTENTION_CREDIT` for declining.
    Blind to *which* category was named, deliberately: there is nothing here
    that could tell a right one from a wrong one.
    """
    return ABSTENTION_CREDIT if category == ABSTENTION_CATEGORY else 1.0


def precision_credit(n_mitigations: int) -> float:
    """What the tier 4-5 block is worth for a list of `n_mitigations` names.

    Flat up to `FREE_MITIGATIONS`, then the reciprocal of the cell count the
    loop would spend. Blind to *which* names were chosen, for the same reason
    `category_credit` is blind to which category.
    """
    if n_mitigations <= 0:
        return 0.0
    return min(1.0, FREE_MITIGATIONS / n_mitigations)


@dataclass
class Score:
    tier: int = 0
    reward: float = 0.0
    stopped_at: str = ""
    detail: str = ""
    consumer_outcome: str = ""
    # The two graded terms, reported separately so a score can be read back
    # apart from the tier it was reached at -- the tier no longer determines it.
    category_credit: float = 1.0
    precision: float = 1.0
    n_mitigations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "reward": round(self.reward, 4),
            "stopped_at": self.stopped_at,
            "detail": self.detail,
            "consumer_outcome": self.consumer_outcome,
            "category_credit": round(self.category_credit, 4),
            "precision": round(self.precision, 4),
            "n_mitigations": self.n_mitigations,
        }

    @property
    def abstained(self) -> bool:
        return self.tier >= 3 and self.category_credit < 1.0


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
    # `unknown` clears the tier -- the consumer accepts it -- but does not earn
    # all of it. Declining to classify is the task's legal escape hatch, not a
    # performance of it.
    score.category_credit = category_credit(category)

    # Tier 4 -- names that exist. An empty list is a decision to stop, so it
    # fails here too: a proposal that names nothing is not a proposal.
    names = [str(m) for m in raw_obj["next_mitigations"]]
    score.n_mitigations = len(names)
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
    # Every name is a probe cell the loop will run, so the block that rewards
    # naming things is scaled by how many cells the proposal spends.
    score.precision = precision_credit(len(names))

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
    """Turn a reached tier into a reward, grading the two tiers that saturate.

    Tiers 1-2 are pure form and stay pass/fail: there is no partial way to
    parse. Tier 3's step is scaled by what the category commits to, and the
    tier 4-5 block by the cost of the mitigation list. Every tier still has to
    be *reached* first, so the gate ordering is unchanged.
    """
    reward = TIER_STEP * min(score.tier, 2)
    if score.tier >= 3:
        reward += TIER_STEP * score.category_credit
    if score.tier >= 4:
        reward += TIER_STEP * (score.tier - 3) * score.precision
    score.reward = reward
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
    # The two saturation routes the first real run found. Both are on contract
    # -- they reach tier 5 and the consumer accepts them -- and both used to
    # score exactly what a diagnosis scores.
    Proposal(
        "declines to classify",
        _ok(category="unknown", hypothesis="Cannot attribute from this evidence."),
        _CANDIDATES,
    ),
    Proposal(
        "declines and shotguns the candidate set",
        _ok(
            category="unknown",
            hypothesis="",
            next_mitigations=list(_CANDIDATES),
        ),
        _CANDIDATES,
    ),
    Proposal(
        "commits, then shotguns the candidate set",
        _ok(next_mitigations=list(_CANDIDATES)),
        _CANDIDATES,
    ),
    # A primary and one fallback: the hedge the loop can absorb without a
    # second proposal round, so it is priced the same as naming one.
    Proposal(
        "names a primary and one fallback",
        _ok(next_mitigations=["nccl_launch_order_implicit", "hsa_no_sdma"]),
        _CANDIDATES,
    ),
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
        "declines to classify": 5,
        "declines and shotguns the candidate set": 5,
        "commits, then shotguns the candidate set": 5,
        "names a primary and one fallback": 5,
    }


def _reward_expectations() -> dict[str, float]:
    """The reward each fixture should earn, asserted by the test suite.

    Separate from `_fixture_expectations` because the tier no longer fixes the
    reward: the four fixtures that reach tier 5 span 0.63 to 1.00, which is the
    range fixes 1 and 2 exist to create.
    """
    step = TIER_STEP
    offered = len([c for c in _CANDIDATES if c != "none"])
    return {
        # Form tiers: ungraded, so still tier/MAX_TIER.
        "JSON wrapped in prose": 0.0,
        "truncated JSON": 0.0,
        "a JSON list, not an object": 0.0,
        "confidence as a string": step,
        "stop as a string": step,
        "hypothesis omitted": step,
        "invented category": 2 * step,
        "free-text category": 2 * step,
        # Committed category, so tier 3's step is whole.
        "hallucinated mitigation": 3 * step,
        "shell command as a mitigation": 3 * step,
        "one real name, one invented": 3 * step,
        "empty mitigation list": 3 * step,
        "re-proposes an already-tried mitigation": 4 * step,
        "proposes outside the allowlist": 4 * step,
        "confidence out of range": 4 * step,
        "on-contract rccl hang proposal": 1.0,
        "on-contract oom proposal": 1.0,
        "names a primary and one fallback": 1.0,
        # Fix 1: declining costs half of tier 3's step.
        "declines to classify": 1.0 - step * (1.0 - ABSTENTION_CREDIT),
        # Fix 2: the tier 4-5 block is scaled by the cell count.
        "commits, then shotguns the candidate set": (
            3 * step + 2 * step * precision_credit(offered)
        ),
        # Both at once, which is what the recorded model and the constant
        # templates both did.
        "declines and shotguns the candidate set": (
            2 * step
            + step * ABSTENTION_CREDIT
            + 2 * step * precision_credit(offered)
        ),
    }


def baselines() -> list[dict[str, Any]]:
    """Degenerate policies, so a real score is read against something.

    The first is the one that matters: a policy that always returns the same
    on-contract proposal scores 1.0 here, because this reward measures form and
    nothing else. That is the ceiling a format reward can give you, and the
    reason it cannot be the only term.

    The two abstaining rows are the constants the first end-to-end run found
    tying a real model at 1.0. They no longer tie it, but note what they are
    still worth: `always abstain, one mitigation` reaches 0.9 while reading
    nothing at all, because a single-name honest abstention is a *cheap* answer
    and cheapness is most of what this reward can see.
    """
    rows = []
    for name, raw in (
        ("always the same valid proposal", _ok()),
        (
            "always abstain, one mitigation",
            _ok(
                category="unknown",
                hypothesis="",
                next_mitigations=["nccl_launch_order_implicit"],
            ),
        ),
        (
            "always abstain, shotgun everything",
            _ok(category="unknown", hypothesis="", next_mitigations=list(_CANDIDATES)),
        ),
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


def load_corpus(path: Path) -> list[tuple[Proposal, str]]:
    """Load a `build_corpus.py` proposal JSONL, with each row's workload family.

    The corpus stores the raw model output verbatim, so scoring a corpus row is
    the same code path as scoring a fixture: nothing about the ladder is
    corpus-specific, which is what makes the two comparable.
    """
    out: list[tuple[Proposal, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != "proposal":
            continue
        spec = row["proposal"]
        out.append((
            Proposal(
                name=spec["name"],
                raw=spec["raw"],
                candidates=list(spec.get("candidates") or []),
                tried=list(spec.get("tried") or []),
            ),
            row.get("workload_family", "unknown"),
        ))
    return out


def run_demo(as_json: bool, corpus: Path | None = None) -> int:
    families: dict[str, int] = {}
    if corpus is not None:
        rows = load_corpus(corpus)
        if not rows:
            print(f"no proposal examples in {corpus}", file=sys.stderr)
            return 2
        proposals = [p for p, _ in rows]
        for _, family in rows:
            families[family] = families.get(family, 0) + 1
    else:
        proposals = list(FIXTURES)

    scored = [(f, score_proposal(f)) for f in proposals]
    if as_json:
        print(
            json.dumps(
                {
                    "proposals": [
                        {"name": f.name, **s.as_dict()} for f, s in scored
                    ],
                    "workload_families": families,
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
        "Tiers 1-2 are form, 3 is the closed category set, 4-5 are registry\n"
        f"membership and availability. Each tier is worth {TIER_STEP:.1f}, but "
        "tier 3's step is\n"
        f"scaled to {ABSTENTION_CREDIT:g} for `category: unknown` and the tier 4-5 block "
        "is scaled by\n"
        f"min(1, {FREE_MITIGATIONS}/names) -- so the tier no longer fixes the reward.\n"
    )
    if families:
        print(f"corpus workload families: {families}\n")
    for f, s in scored:
        print(f"tier {s.tier}/{MAX_TIER}  reward {s.reward:.2f}  {f.name}")
        if s.tier >= 3:
            print(
                f"       category credit {s.category_credit:.2f}"
                f"   precision {s.precision:.2f} over {s.n_mitigations} name(s)"
            )
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
        "\nThe first baseline is still the point: a fixed on-contract proposal\n"
        "scores 1.00 without diagnosing anything. Form is a gate, not a signal --\n"
        "pair it with triage_reward.py, which scores whether the read is right.\n"
        "The abstaining rows are what fixes 1 and 2 moved: they used to tie a\n"
        "real model at 1.00, and the one-name abstention is still worth 0.90 for\n"
        "reading nothing, because a cheap answer is most of what form can see."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=None,
                        help="proposal.jsonl written by build_corpus.py")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    return run_demo(args.json, args.corpus)


if __name__ == "__main__":
    sys.exit(main())
