"""The offline evaluator for a later plan mode (design Section 6.3, "Logging for evaluating a
later plan mode").

v1 runs every leg whatever the plan says, and logs per candidate the plan, every leg's full result and
cost, and the audit coin it would have drawn (the plan log of `log_only.SelectorHook`). From that log
alone this module computes:

- per leg: the skip rate, the miss rate with its Wilson 95% interval, the share of the leg's defects
  the plan would have missed, and the worker-seconds the skips would have saved;
- the reward each sample would have received under options A, A + C, B and B + C, and the change in
  the group advantages under each;
- the option table of Section 6.3 with measured columns, and when the safety switch would have armed
  or fired.

Definitions. A plan counts only when its status is "ok"; a sample whose plan was unavailable, or never
requested (decided before S2, or a reward-cache hit), runs every leg and scores as logged under every
option. Over the samples with a plan, for each planned leg:

    skip rate          skipped / plans
    miss               the plan skips the leg and the leg's full result is a defect (finding or fail)
    miss rate          misses / samples where the leg settled (clean, finding or fail), Wilson 95%
    defects missed     misses / defects, Wilson 95%
    unknown            the leg did not settle (unsettled, or not run): neither a miss nor a non-miss
    saved worker-s     the run time of the skipped leg where it ran (a cache hit cost nothing)

The miss rate is per sample: the share of samples that would carry a defect in a leg the plan
skipped, which is what the safety switch bounds. Options, for a sample with a plan:

    A     the skipped legs reach the reward function as `skipped_by_plan` (an unsettled leg to
          the reward's rule 5)
    B     the skipped legs are taken as clean and a clean reward is capped: min(reward, cap)
    +C    an audited sample (its audit coin) runs every leg and scores as logged
    +S    the safety switch, replayed in log order: a leg is skipped only while the audited samples
          so far allow it (`base.SafetySwitch`)

Each counterfactual reward is the reward function's own `decide` over the record plan mode would have
had: the logged results of the legs that ran, the skipped legs as above, and a leg that requires a
skipped one as not run. A leg that early exit kept from running in v1 is left out, so `decide` answers
only if no result of it could change the answer; when it cannot answer, the sample is reported as
undetermined, never guessed.

    python -m aorlta.selector.plan_eval PLAN_LOG.jsonl [--cap 0.5] [--audit-fractions 0.1,0.5] [--out DIR]
    python -m aorlta.selector.plan_eval --fake FAKE_LOG.jsonl       write the fake log with known answers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_bytes, atomic_write_json, read_text
from ..ids import canonical_json, candidate_id, policy_version
from ..reward.profiles import Profile, load_profile
from ..reward.rewardfn.base import Decided, NeedMore, Unscored
from ..reward.rewardfn.for_profile import standin_rewardfn
from ..reward.rewardfn.standin_v11 import StandinV11
from ..schemas import SETTLED_STATES, Candidate, LegResult, RewardResult, SchemaError, TaskRef, from_dict
from .base import Z95, SafetySwitch, audit_coin, audit_draw, is_defect, wilson_interval
from .log_only import PLAN_LOG_SCHEMA, PLAN_RECORD_SCHEMA, CandidatePlan, plan_log_line, plan_record

EVAL_SCHEMA = "aorlta.selector.plan_eval/1"
EARLY_EXIT_REASONS = frozenset({"decided_before_run", "decided_before_result"})
# Unscored for a reason of the service, not of the reward function: no record can reproduce these, and
# no option changes them.
SERVICE_REASONS = frozenset({"service_closed", "drain_failed", "reward_fn_error", "reward_invariant_violated"})
OPTIONS = ("full", "A", "A+C", "A+C+S", "B", "B+C", "B+C+S")
DEFAULT_CAP = 0.5
UNDETERMINED = "undetermined"

OPTION_TEXT = {  # Section 6.3's option table: what a skipped leg means, and what the option needs
    "full": ("every leg runs (v1, the logged reward)", "nothing"),
    "A": ("skipped legs reach the reward function as unsettled; rule 5 leaves the sample unscored unless a finding is "
          "confirmed", "nothing new"),
    "A+C": ("A on unaudited samples; audited samples run every leg and score normally", "nothing new"),
    "A+C+S": ("A + C, skipping only the legs the safety switch allows at that point of the log", "nothing new"),
    "B": ("all run legs clean: reward capped at the cap; a confirmed finding is penalised as usual",
          "a new reward version"),
    "B+C": ("B on unaudited samples; audited samples run every leg and score normally",
            "a new reward version, f >= 0.2"),
    "B+C+S": ("B + C with the safety switch: D6's recommended v2", "a new reward version, f >= 0.2"),
}


class PlanEvalError(ValueError):
    """The plan log cannot be evaluated: malformed, or it does not reproduce its own rewards."""


@dataclass(frozen=True)
class Sample:
    line: int  # 1-based line in the plan log
    cand: Candidate  # rebuilt for the reward function; response_text is empty (it is never logged)
    source_sha256: str
    audit: bool
    audit_u: float
    audit_fraction: float
    result: RewardResult

    @property
    def plan(self) -> dict:
        return self.result.plan or {}

    @property
    def plan_ok(self) -> bool:
        return self.plan.get("status") == "ok"

    @property
    def skip(self) -> frozenset[str]:
        return frozenset(self.plan.get("skip") or ()) if self.plan_ok else frozenset()

    @property
    def group(self) -> tuple[int, str]:
        return (self.cand.iteration, self.cand.prompt_id)


# --- reading the log -------------------------------------------------------------------------------

_CAND_KEYS = {"candidate_id", "run_id", "iteration", "prompt_id", "k", "policy_version", "check_profile",
              "source_sha256", "response_tokens", "max_new_tokens", "task"}
_AUDIT_KEYS = {"coin", "u", "fraction", "drawn_at"}


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PlanEvalError(message)


def parse_plan_log(lines: Iterable[Any]) -> list[Sample]:
    out: list[Sample] = []
    seen: set[str] = set()
    for n, line in enumerate(lines, 1):
        try:
            _need(isinstance(line, Mapping) and set(line) == {"schema", "candidate", "audit", "result"}
                  and line["schema"] == PLAN_LOG_SCHEMA, f"not a {PLAN_LOG_SCHEMA} line")
            c, a = line["candidate"], line["audit"]
            _need(isinstance(c, Mapping) and set(c) == _CAND_KEYS, "candidate block has the wrong keys")
            _need(isinstance(a, Mapping) and set(a) == _AUDIT_KEYS, "audit block has the wrong keys")
            for name in ("iteration", "k", "response_tokens", "max_new_tokens"):
                _need(type(c[name]) is int, f"candidate.{name} is not an int")
            for name in ("candidate_id", "run_id", "prompt_id", "policy_version", "check_profile", "source_sha256"):
                _need(isinstance(c[name], str) and c[name], f"candidate.{name} is not a non-empty str")
            cand = Candidate(c["candidate_id"], c["run_id"], c["iteration"], c["prompt_id"], c["k"],
                             c["policy_version"], "", c["response_tokens"], c["max_new_tokens"],
                             from_dict(TaskRef, c["task"]), c["check_profile"], None)
            _need(type(a["coin"]) is bool and type(a["u"]) is float and 0.0 <= a["u"] < 1.0
                  and type(a["fraction"]) is float and 0.0 <= a["fraction"] <= 1.0, "malformed audit block")
            _need(a["coin"] == audit_coin(a["u"], a["fraction"]), "the audit coin does not match its draw")
            result = from_dict(RewardResult, line["result"])
            _need(result.candidate_id == cand.candidate_id, "the result is for another candidate")
            _need(result.audit == a["coin"], "the result's audit flag is not the logged coin")
            plan = result.plan
            _need(isinstance(plan, dict) and plan.get("schema") == PLAN_RECORD_SCHEMA, "the result carries no plan record")
            _need(plan.get("status") in ("ok", "plan_unavailable", "not_requested"), "unknown plan status")
            if plan["status"] == "ok":
                _need(isinstance(plan.get("legs"), dict) and isinstance(plan.get("skip"), list)
                      and set(plan["skip"]) <= set(plan["legs"]), "a usable plan without its decisions")
            _need(cand.candidate_id not in seen, "the candidate appears twice")
            seen.add(cand.candidate_id)
            out.append(Sample(n, cand, c["source_sha256"], a["coin"], a["u"], a["fraction"], result))
        except (KeyError, TypeError, SchemaError, ValueError) as exc:
            raise PlanEvalError(f"plan log line {n}: {exc}") from None
    if not out:
        raise PlanEvalError("the plan log is empty")
    return out


def read_plan_log(path: str | Path) -> list[Sample]:
    try:
        lines = [json.loads(t) for t in read_text(path).splitlines() if t.strip()]
    except ValueError as exc:
        raise PlanEvalError(f"{path}: not JSON lines ({exc})") from None
    return parse_plan_log(lines)


# --- counterfactual rewards ------------------------------------------------------------------------


def _clean(leg: str, repeats: int, t: float) -> LegResult:
    return LegResult(leg, "clean", None, [], repeats, 1, None, None, False, t, t, t)


def _stand_in(leg: str, state: str, reason: str | None, t: float) -> LegResult:
    return LegResult(leg, state, reason, [], 0, 0, None, None, False, t, None, None)


def leg_cost(r: LegResult) -> float:
    """Worker-seconds the leg's final run took for this sample: 0 for a cache hit or a leg not run."""
    if r.cache_hit or r.state in ("not_run", "skipped_by_plan") or r.t_start is None or r.t_end is None:
        return 0.0
    return max(0.0, r.t_end - r.t_start)


class Evaluator:
    def __init__(self, profile: Profile, rewardfn: Any, *, cap: float = DEFAULT_CAP):
        if not (type(cap) in (int, float) and math.isfinite(cap) and -1.55 <= cap <= 1.0):
            raise PlanEvalError(f"cap must be a reward in [-1.55, 1.0], got {cap!r}")
        self.profile = profile
        self.rf = rewardfn
        self.cap = float(cap)
        self.planned = tuple(leg for leg in profile.legs if not leg.startswith("gate."))

    def _record(self, s: Sample, skip: frozenset[str], fill: str) -> dict[str, LegResult]:
        """The record plan mode would have had (see the module docstring)."""
        dead: set[str] = set()
        for leg in self.profile.legs:
            if leg not in skip and any(need in skip or need in dead for need in self.profile.requires.get(leg, ())):
                dead.add(leg)
        out: dict[str, LegResult] = {}
        for leg, r in s.result.legs.items():
            if leg in skip:
                out[leg] = (_clean(leg, self.profile.clean_repeats, 0.0) if fill == "clean"
                            else _stand_in(leg, "skipped_by_plan", None, 0.0))
            elif leg in dead:
                out[leg] = _stand_in(leg, "not_run", "requirement_skipped", 0.0)
            elif r.state == "not_run" and r.reason in EARLY_EXIT_REASONS:
                continue  # it would run in plan mode if the decision needed it; its result is unknown
            else:
                out[leg] = r
        return out

    def decide(self, s: Sample, skip: frozenset[str], fill: str) -> tuple[str, float | None, Any]:
        """('scored', reward, parts) | ('unscored', None, reason) | ('undetermined', None, legs needed)."""
        d = self.rf.decide(self._record(s, skip, fill), s.cand)
        if isinstance(d, Decided):
            reward = d.reward
            if fill == "clean" and skip and self.rf.claims_clean(d.reward, d.parts):
                reward = min(d.reward, self.cap)
            return "scored", reward, d.parts
        if isinstance(d, Unscored):
            return "unscored", None, d.reason
        if isinstance(d, NeedMore):
            return UNDETERMINED, None, tuple(d.legs)
        raise PlanEvalError(f"the reward function answered {d!r}")

    def check_full(self, s: Sample) -> str:
        """'ok' if the logged reward is what the reward function gives on the logged record, 'service'
        if the sample was unscored by the service itself; anything else is refused."""
        r = s.result
        if r.reward is None and r.reason in SERVICE_REASONS:
            return "service"
        kind, reward, extra = self.decide(s, frozenset(), "skipped")
        if kind == "scored" and r.reward == reward and r.parts == extra:
            return "ok"
        if kind == "unscored" and r.reward is None and r.reason == extra:
            return "ok"
        raise PlanEvalError(f"plan log line {s.line} ({s.cand.candidate_id}): the logged reward {r.reward!r} "
                            f"({r.reason}) is not what {self.rf.name} gives on the logged record ({kind}, {reward!r}, "
                            f"{extra!r})")

    def reward_with(self, s: Sample, skip: frozenset[str], fill: str) -> Any:
        """The sample's reward when plan mode skips `skip`: a float, None (unscored) or UNDETERMINED."""
        if not skip:
            return s.result.reward
        if s.result.reward is None and s.result.reason in SERVICE_REASONS:
            return None
        kind, reward, _ = self.decide(s, skip, fill)
        return UNDETERMINED if kind == UNDETERMINED else reward

    def option_reward(self, s: Sample, option: str, audited: bool) -> Any:
        """The sample's reward under full, A, A+C, B or B+C."""
        skip = applied_skips([s], option, {s.cand.candidate_id: audited})[s.cand.candidate_id]
        return self.reward_with(s, skip, "clean" if option.startswith("B") else "skipped")


# --- advantages ------------------------------------------------------------------------------------


def group_advantages(rewards: Mapping[str, Any], groups: Mapping[str, tuple], min_scored: int) -> tuple[dict, dict]:
    """Advantages without std normalisation (Dr. GRPO): reward minus the mean of the group's
    scored rewards; none for a group with fewer than `min_scored` scored members, and none for a group
    with an undetermined member. Returns (advantage by candidate, group status counts)."""
    members: dict[tuple, list[str]] = {}
    for cid, g in groups.items():
        members.setdefault(g, []).append(cid)
    adv: dict[str, float] = {}
    status = Counter()
    for g, cids in members.items():
        values = [rewards[c] for c in cids]
        if any(isinstance(v, str) for v in values):  # UNDETERMINED
            status["undetermined"] += 1
            continue
        scored = [(c, v) for c, v in zip(cids, values) if v is not None]
        if len(scored) < min_scored:
            status["too_few_scored"] += 1
            continue
        status["trainable"] += 1
        mean = sum(v for _, v in scored) / len(scored)
        for c, v in scored:
            adv[c] = v - mean
    return adv, dict(status)


# --- the evaluation --------------------------------------------------------------------------------


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": None if n == 0 else k / n, "ci95": [lo, hi]}


def _leg_table(samples: Sequence[Sample], legs: Sequence[str]) -> dict:
    out = {}
    for leg in legs:
        plans = [s for s in samples if s.plan_ok]
        skipped = [s for s in plans if leg in s.skip]
        settled = [s for s in plans if s.result.legs[leg].state in SETTLED_STATES]
        defects = [s for s in settled if is_defect(s.result.legs[leg])]
        missed = [s for s in defects if leg in s.skip]
        unknown = [s for s in plans if s.result.legs[leg].state not in SETTLED_STATES]
        out[leg] = {
            "plans": len(plans),
            "skipped": len(skipped),
            "skip_rate": None if not plans else len(skipped) / len(plans),
            "settled": len(settled),
            "defects": len(defects),
            "missed": len(missed),
            "miss_rate": _rate(len(missed), len(settled)),
            "defects_missed": _rate(len(missed), len(defects)),
            "unknown": len(unknown),
            "unknown_skipped": sum(leg in s.skip for s in unknown),
            "run_s": sum(leg_cost(s.result.legs[leg]) for s in samples),
            "saved_s": sum(leg_cost(s.result.legs[leg]) for s in skipped),
        }
    return out


def _resources(profile: Profile) -> dict[str, str]:
    return {leg: profile.pools[stage.pool].resource for stage in profile.stages for leg in stage.legs}


def applied_skips(samples: Sequence[Sample], option: str, coins: Mapping[str, bool],
                  switched: Mapping[str, frozenset] | None = None) -> dict[str, frozenset]:
    """The legs plan mode would skip on each sample under an option: none under full; the plan's skips
    under A and B; on unaudited samples only under +C; and under +C+S only those the safety switch
    allowed at that point of the log (`switched`, from `replay_switch`)."""
    out = {}
    for s in samples:
        cid = s.cand.candidate_id
        if option == "full":
            out[cid] = frozenset()
        elif option.endswith("+S"):
            out[cid] = switched[cid]
        elif option.endswith("+C") and coins[cid]:
            out[cid] = frozenset()
        else:
            out[cid] = s.skip
    return out


def _option_summary(ev: Evaluator, samples: Sequence[Sample], option: str, skips: Mapping[str, frozenset],
                    full: Mapping[str, Any], min_scored: int, full_adv: Mapping[str, float]) -> dict:
    """One row of the option table, for the skips the option applies (`applied_skips`). `defects in
    skipped legs` are the samples whose plan skips a leg with a defect, whatever the option; they are
    `hidden` where the option applies that skip."""
    fill = "clean" if option.startswith("B") else "skipped"
    rewards = {s.cand.candidate_id: ev.reward_with(s, skips[s.cand.candidate_id], fill) for s in samples}
    groups = {s.cand.candidate_id: s.group for s in samples}
    adv, gstatus = group_advantages(rewards, groups, min_scored)
    resources = _resources(ev.profile)
    applied = [s for s in samples if skips[s.cand.candidate_id]]
    in_skipped = {s.cand.candidate_id for s in samples if any(is_defect(s.result.legs[leg]) for leg in s.skip)}
    hidden = {s.cand.candidate_id for s in applied
              if any(is_defect(s.result.legs[leg]) for leg in skips[s.cand.candidate_id])}
    clean_full = {s.cand.candidate_id for s in samples
                  if s.result.reward is not None and ev.rf.claims_clean(s.result.reward, s.result.parts)}

    def mean(ids):
        vals = [rewards[c] for c in ids if isinstance(rewards[c], float)]
        return None if not vals else sum(vals) / len(vals)

    saved, total = Counter(), Counter()
    for s in applied:
        for leg in skips[s.cand.candidate_id]:
            saved[resources[leg]] += leg_cost(s.result.legs[leg])
    for s in samples:
        for leg in ev.planned:
            total[resources[leg]] += leg_cost(s.result.legs[leg])
    both = [c for c in adv if c in full_adv]
    deltas = [adv[c] - full_adv[c] for c in both]
    determined = [c for c, v in rewards.items() if not isinstance(v, str)]
    as_clean = 0
    for s in applied:
        if s.cand.candidate_id in hidden:
            kind, reward, parts = ev.decide(s, skips[s.cand.candidate_id], fill)
            as_clean += kind == "scored" and ev.rf.claims_clean(reward, parts)
    gap_clean, gap_defect = mean(clean_full), mean(in_skipped)
    return {
        "samples": len(samples),
        "scored": sum(isinstance(v, float) for v in rewards.values()),
        "unscored": sum(v is None for v in rewards.values()),
        "undetermined": len(samples) - len(determined),
        "changed": sum(1 for c in determined if rewards[c] != full[c]),
        "lost": sum(1 for c in determined if full[c] is not None and rewards[c] is None),
        "skips_applied": len(applied),
        "defects_in_skipped_legs": len(in_skipped),
        "hidden_defects": len(hidden),
        "hidden_defects_scored_as_clean": as_clean,
        "clean_mean_reward": gap_clean,
        "defects_in_skipped_legs_mean_reward": gap_defect,
        "clean_minus_defect_gap": None if gap_clean is None or gap_defect is None else gap_clean - gap_defect,
        "groups": gstatus,
        "advantages": len(adv),
        "mean_abs_delta_advantage": None if not deltas else sum(abs(d) for d in deltas) / len(deltas),
        "max_abs_delta_advantage": None if not deltas else max(abs(d) for d in deltas),
        "advantage_sign_flips": sum(1 for c in both if adv[c] * full_adv[c] < 0),
        "hidden_defects_advantage_up": sum(1 for c in both if c in hidden and adv[c] > full_adv[c]),
        "saved_s": sum(saved.values()),
        "saved_share": None if not sum(total.values()) else sum(saved.values()) / sum(total.values()),
        "saved_s_by_resource": {k: saved[k] for k in sorted(total)},
        "run_s_by_resource": {k: total[k] for k in sorted(total)},
        "rewards": rewards,
    }


def replay_switch(samples: Sequence[Sample], legs: Sequence[str], max_miss_rate: float,
                  coins: Mapping[str, bool]) -> tuple[dict, dict[str, frozenset]]:
    """Plan mode's safety switch over the log, in log order: an unaudited sample with a plan skips the
    planned legs the switch allows at that point, and an audited sample with a plan is one observation
    per settled leg. Returns when each leg would have armed (skips allowed) and fired (latched off), and
    the skips each sample would have had."""
    switch = SafetySwitch(legs, max_miss_rate)
    armed_at: dict[str, str | None] = {leg: None for leg in legs}
    tripped_at: dict[str, str | None] = {leg: None for leg in legs}
    skips: dict[str, frozenset] = {}
    audited = 0
    for s in samples:
        cid = s.cand.candidate_id
        coin = coins[cid]
        skips[cid] = frozenset(leg for leg in s.skip if not coin and switch.allows_skip(leg))
        if not (coin and s.plan_ok):
            continue
        audited += 1
        for leg in legs:
            r = s.result.legs[leg]
            if r.state not in SETTLED_STATES:
                continue
            before = switch.state(leg)
            after = switch.observe(leg, would_skip=leg in s.skip, defect=is_defect(r))
            if before != after and after == "armed":
                armed_at[leg] = cid
            if before != after and after == "tripped":
                tripped_at[leg] = cid
    snap = switch.snapshot()
    replay = {"max_miss_rate": max_miss_rate, "audited_with_plan": audited,
              "legs": {leg: dict(snap[leg], armed_with=armed_at[leg], tripped_with=tripped_at[leg]) for leg in legs}}
    return replay, skips


def evaluate(samples: Sequence[Sample], *, rewardfn: Any = None, cap: float = DEFAULT_CAP, min_scored_per_group: int = 2,
             audit_fractions: Sequence[float] = (), max_miss_rate: float = 0.02) -> dict:
    """The whole evaluation as JSON values (see the module docstring)."""
    names = {s.cand.check_profile for s in samples}
    _need(len(names) == 1, f"one check profile per log, got {sorted(names)}")
    profile = load_profile(names.pop())
    if rewardfn is None:
        _need(profile.reward_fn == "standin.v11", f"profile {profile.name!r} is scored by "
              f"{profile.reward_fn!r}; pass its reward function")
        rewardfn = standin_rewardfn(profile)
    identity = {"name": rewardfn.name, "version": rewardfn.version, "code_sha256": rewardfn.code_sha256}
    for s in samples:
        _need(s.result.reward_fn == identity, f"plan log line {s.line}: scored by {s.result.reward_fn}, not by the "
              f"evaluator's {identity}")
        _need(set(s.result.legs) >= set(profile.legs), f"plan log line {s.line}: the record lacks profile legs")
    ev = Evaluator(profile, rewardfn, cap=cap)
    checks = Counter(ev.check_full(s) for s in samples)
    coins = {s.cand.candidate_id: s.audit for s in samples}
    full = {s.cand.candidate_id: s.result.reward for s in samples}
    groups = {s.cand.candidate_id: s.group for s in samples}
    full_adv, _ = group_advantages(full, groups, min_scored_per_group)
    fractions = sorted({s.audit_fraction for s in samples})
    status = Counter(s.plan.get("status") + ("" if not s.plan.get("reason") else f":{s.plan['reason']}")
                     for s in samples)

    def rows(coin_map: Mapping[str, bool], names: Sequence[str]) -> tuple[dict, dict]:
        replay, switched = replay_switch(samples, ev.planned, max_miss_rate, coin_map)
        out = {}
        for opt in names:
            summary = _option_summary(ev, samples, opt, applied_skips(samples, opt, coin_map, switched), full,
                                      min_scored_per_group, full_adv)
            summary.pop("rewards")
            out[opt] = summary
        return out, replay

    options, switch = rows(coins, OPTIONS)
    sweep = []
    for f in audit_fractions:
        fc = {s.cand.candidate_id: audit_coin(s.audit_u, f) for s in samples}
        summaries, replay = rows(fc, ("A+C", "B+C", "B+C+S"))
        sweep.append({"audit_fraction": f, "audited": sum(fc.values()), **summaries, "switch": replay})
    return {
        "schema": EVAL_SCHEMA,
        "log": {"samples": len(samples), "groups": len(set(groups.values())),
                "iterations": sorted({s.cand.iteration for s in samples}), "profile": profile.name,
                "reward_fn": identity, "plan_status": dict(sorted(status.items())),
                "selector_versions": dict(Counter(s.plan.get("selector_version") for s in samples if s.plan_ok)),
                "audit_fraction": fractions[0] if len(fractions) == 1 else fractions,
                "audited": sum(coins.values()), "reproduced": checks.get("ok", 0),
                "unscored_by_service": checks.get("service", 0)},
        "settings": {"cap": ev.cap, "min_scored_per_group": min_scored_per_group, "z": Z95,
                     "max_miss_rate": max_miss_rate},
        "legs": _leg_table(samples, ev.planned),
        "audited_legs": _leg_table([s for s in samples if s.audit], ev.planned),
        "options": options,
        "sweep": sweep,
        "switch": switch,
    }


# --- the report ------------------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(rate: dict) -> str:
    if rate["n"] == 0:
        return "- (no data)"
    lo, hi = rate["ci95"]
    return f"{rate['k']}/{rate['n']} = {_pct(rate['rate'], 2)} [{_pct(lo, 2)}, {_pct(hi, 2)}]"


def markdown(report: dict) -> str:
    log = report["log"]
    lines = [
        f"Plan log: {log['samples']} samples in {log['groups']} groups, iterations {log['iterations']}, profile "
        f"`{log['profile']}`, scored by `{log['reward_fn']['name']}` (code {log['reward_fn']['code_sha256'][:12]}). "
        f"Every logged reward reproduced: {log['reproduced']} of {log['samples']} "
        f"({log['unscored_by_service']} unscored by the service itself).",
        f"Plans: {', '.join(f'{k} {v}' for k, v in log['plan_status'].items())}. Audit fraction {log['audit_fraction']}: "
        f"{log['audited']} samples audited. Cap for B: {report['settings']['cap']}.",
        "",
        "### Per leg (samples with a plan)",
        "",
        "| leg | plans | skip rate | defects | missed | miss rate per sample [95% CI] | share of defects missed "
        "[95% CI] | unknown (skipped) | worker-s run | worker-s saved |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for leg, t in report["legs"].items():
        lines.append(f"| `{leg}` | {t['plans']} | {t['skipped']} ({_pct(t['skip_rate'])}) | {t['defects']} | {t['missed']} | "
                     f"{_ci(t['miss_rate'])} | {_ci(t['defects_missed'])} | {t['unknown']} ({t['unknown_skipped']}) | "
                     f"{t['run_s']:.1f} | {t['saved_s']:.1f} |")
    lines += [
        "",
        "### Options (Section 6.3)",
        "",
        "A defect in a skipped leg is hidden where the option applies the skip. The gap is the mean reward of "
        "the samples clean under full checks minus that of the samples whose plan skips a leg with a defect "
        "(scored samples only): the training signal against hiding a defect.",
        "",
        "| option | what a skipped leg means | scored | changed vs full | undetermined | hidden defects | hidden "
        "defects scored as clean | clean − defect gap | groups trainable | mean abs Δadvantage | "
        "hidden defects with advantage up | worker-s saved (share) | needs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for opt, o in report["options"].items():
        text, needs = OPTION_TEXT[opt]
        lines.append(
            f"| {opt} | {text} | {o['scored']} | {o['changed']} | {o['undetermined']} | {o['hidden_defects']} | "
            f"{o['hidden_defects_scored_as_clean']} | {_num(o['clean_minus_defect_gap'])} | "
            f"{o['groups'].get('trainable', 0)} | {_num(o['mean_abs_delta_advantage'])} | "
            f"{o['hidden_defects_advantage_up']} | {o['saved_s']:.1f} ({_pct(o['saved_share'])}) | {needs} |")
    by_res = report["options"]["B"]
    lines += ["", "Worker-seconds by resource, all planned legs in the log / saved under A or B: " + ", ".join(
        f"{k} {by_res['run_s_by_resource'][k]:.1f} / {by_res['saved_s_by_resource'][k]:.1f}"
        for k in by_res["run_s_by_resource"]) + "."]
    if report["sweep"]:
        lines += ["", "### Audit fraction", "", "| f | audited | B+C worker-s saved (share) | B+C hidden defects "
                  "| B+C clean − defect gap | B+C+S worker-s saved (share) | B+C+S hidden defects | A+C scored | "
                  "switch: legs armed at the end |", "|---|---|---|---|---|---|---|---|---|"]
        for row in report["sweep"]:
            b, bs, a = row["B+C"], row["B+C+S"], row["A+C"]
            armed = [leg for leg, s in row["switch"]["legs"].items() if s["state"] == "armed"]
            lines.append(f"| {row['audit_fraction']} | {row['audited']} | {b['saved_s']:.1f} ({_pct(b['saved_share'])}) | "
                         f"{b['hidden_defects']} | {_num(b['clean_minus_defect_gap'])} | {bs['saved_s']:.1f} "
                         f"({_pct(bs['saved_share'])}) | {bs['hidden_defects']} | {a['scored']} | "
                         f"{', '.join(f'`{x}`' for x in armed) or 'none'} |")
    sw = report["switch"]
    lines += ["", f"### Safety switch, replayed over the {sw['audited_with_plan']} audited samples with a plan "
              f"(max miss rate {sw['max_miss_rate']}, Wilson 95% upper bound)", "",
              "| leg | observations | misses | upper bound | state at the end | armed with | fired with |",
              "|---|---|---|---|---|---|---|"]
    for leg, s in sw["legs"].items():
        lines.append(f"| `{leg}` | {s['observations']} | {s['misses']} | {_pct(s['upper'], 2)} | {s['state']} | "
                     f"{s['armed_with'] or '-'} | {s['tripped_with'] or '-'} |")
    return "\n".join(lines) + "\n"


# --- the fake log with known answers ---------------------------------------------------------------

FAKE_RUN = "wp6-fake"
FAKE_SEED = 20261102
FAKE_T0 = 1_800_000_000
FAKE_COST = {"ref.native": 11.0, "ref.edge": 11.0, "san.consan": 11.0, "san.waitcheck": 11.0, "san.asan": 8.5}
FAKE_TIMEOUT_COST = 120.0
FAKE_PLANNED = ("ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan")
GATES = ("gate.parse", "gate.contract", "gate.static", "gate.reference", "gate.compile", "gate.no_artifact",
         "gate.spec_invalid", "gate.static_reject", "gate.no_change_on_failing_parent")


def fake_outcome(i: int) -> dict:
    """The fake log's ground truth for sample i, from the residues a = i mod 8 (= k), b = i mod 3,
    c = i mod 5 and d = i mod 7. Over any 840 consecutive samples every (a, b, c, d) appears exactly
    once, so every count below is an exact product of residue shares.

        compile error                 d = 6                 (no plan is requested)
        plan unavailable (timeout)    d = 5                 (every leg runs)
        plan ok                       d in 0..4
        ref.native fail(wrong_output) c = 4 and b = 2       (decides -0.7; ASan is then not run)
        san.consan finding            c = 0
        san.waitcheck unsettled       a = 7 and c = 1       (timeout: 120 worker-s)
        san.waitcheck finding         a = 1
        san.asan finding              a = 3
        the plan runs ref.* always; skips san.consan iff b = 0 (blind to ConSan's defects); skips
        san.waitcheck iff a in {0, 2, 4, 6} (never one with a defect); skips san.asan iff
        (a != 3 and b != 0) or (a = 3 and d = 0) (misses 1 in 5 ASan defects)
    """
    a, b, c, d = i % 8, i % 3, i % 5, i % 7
    if d == 6:
        return {"compile_error": True, "plan": "not_requested"}
    native = "fail" if (c == 4 and b == 2) else "clean"
    waitcheck = "unsettled" if (a == 7 and c == 1) else ("finding" if a == 1 else "clean")
    asan = "not_run" if native == "fail" else ("finding" if a == 3 else "clean")
    out = {"compile_error": False, "plan": "ok" if d <= 4 else "timeout",
           "legs": {"ref.native": native, "ref.edge": "clean", "san.consan": "finding" if c == 0 else "clean",
                    "san.waitcheck": waitcheck, "san.asan": asan}}
    if d <= 4:
        out["skip"] = sorted(leg for leg, s in (("san.consan", b == 0), ("san.waitcheck", a in (0, 2, 4, 6)),
                                                ("san.asan", (a != 3 and b != 0) or (a == 3 and d == 0))) if s)
    return out


def fake_log(*, iterations: int = 9, prompts: int = 35, K: int = 8, seed: int = FAKE_SEED,
             audit_fraction: float = 0.2) -> list[dict]:
    """Plan-log lines for iterations x prompts x K samples (2,520 by default: three periods of 840), in
    the hook's format, with the ground truth of `fake_outcome`. Rewards are standin.v11 on the full
    record, as the service would log them."""
    rf = StandinV11()
    identity = {"name": rf.name, "version": rf.version, "code_sha256": rf.code_sha256}
    lines = []
    for i in range(iterations * prompts * K):
        it, p, k = i // (prompts * K) + 1, (i % (prompts * K)) // K, i % K
        pid = f"standin:fake{p:02d}"
        cid = candidate_id(FAKE_RUN, it, pid, k)
        truth = fake_outcome(i)
        cand = Candidate(cid, FAKE_RUN, it, pid, k, policy_version(FAKE_RUN, it - 1), "", 256 + (97 * i) % 3841, 4096,
                         TaskRef("contract-fake", None, None, "frozen", "0" * 64), "fake", None)
        t0 = float(FAKE_T0 + 1000 * i)
        bracket = {"before": f"control:{i}:a", "after": f"control:{i}:b", "clean": True}
        legs: dict[str, LegResult] = {}
        for g in GATES:
            if g == "gate.compile" and truth["compile_error"]:
                legs[g] = LegResult(g, "fail", "compile_error", [], 1, 1, "compile/1", None, False, t0, t0 + 1, t0 + 2)
            else:
                legs[g] = LegResult(g, "clean", None, [], 1, 1, "gates/1", None, False, t0, t0 + 1, t0 + 2)
        for n, leg in enumerate(FAKE_PLANNED):
            start = t0 + 10.0 + 200.0 * n
            if truth["compile_error"]:
                legs[leg] = LegResult(leg, "not_run", "requirement_not_clean", [], 0, 0, None, None, False, t0, None, None)
                continue
            state = truth["legs"][leg]
            if state == "not_run":
                legs[leg] = LegResult(leg, "not_run", "decided_before_run", [], 0, 0, None, None, False, t0, None, None)
            elif state == "unsettled":
                legs[leg] = LegResult(leg, "unsettled", "timeout", [], 0, 1, "sim/1", None, False, t0, start,
                                      start + FAKE_TIMEOUT_COST)
            else:
                reason = "wrong_output" if state == "fail" else None
                findings = [{"kind": "fake_finding", "leg": leg}] if state == "finding" else []
                legs[leg] = LegResult(leg, state, reason, findings, 1, 1, "sim/1" if leg != "san.asan" else "asan/1",
                                      bracket, False, t0, start, start + FAKE_COST[leg])
        reward, raw = rf.compute(legs, cand)
        u = audit_draw(seed, cid)
        cp = CandidatePlan(cid, f"{i:064x}", u, audit_coin(u, audit_fraction), t0)
        if truth["plan"] != "not_requested":
            cp.available_legs, cp.leg_cost_s = FAKE_PLANNED, dict(FAKE_COST)
            cp.requested_at, cp.t_requested, cp.deadline = t0 + 3.0, t0 + 3.0, t0 + 8.0
            if truth["plan"] == "ok":
                skip = truth["skip"]
                cp.status, cp.answered_after_s = "ok", 0.25
                cp.plan = {"selector_version": "fake.selector/1", "plan_id": f"fake:{i}",
                           "legs": {leg: {"decision": "skip" if leg in skip else "run", "p_finding": None, "score": None}
                                    for leg in FAKE_PLANNED},
                           "skip": list(skip), "context_sha256": None, "context_chars": None}
            else:
                cp.status, cp.reason = "plan_unavailable", "timeout"
        result = RewardResult(cid, reward, "scored" if reward is not None else "unscored",
                              None if reward is not None else raw["reason"], raw if reward is not None else None,
                              identity, legs, True, plan_record(cp, "log_only"), cp.audit, f"{i:064x}",
                              {"total_s": 0.0, "stages": {}})
        lines.append(plan_log_line(cp, cand, result, audit_fraction))
    return lines


def write_jsonl(path: str | Path, lines: Sequence[dict]) -> Path:
    return atomic_write_bytes(path, b"".join(canonical_json(line) + b"\n" for line in lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aorlta.selector.plan_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("log", nargs="?", type=Path, help="a plan log (JSON lines)")
    ap.add_argument("--fake", type=Path, help="write the fake log with known answers here, then evaluate it")
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP, help="option B's cap on a clean reward")
    ap.add_argument("--audit-fractions", default="0.1,0.2,0.3,0.5", help="extra audit fractions to sweep, comma separated")
    ap.add_argument("--max-miss-rate", type=float, default=0.02)
    ap.add_argument("--min-scored-per-group", type=int, default=2)
    ap.add_argument("--out", type=Path, help="write plan-eval.json and plan-eval.md here")
    args = ap.parse_args(argv)
    if (args.log is None) == (args.fake is None):
        ap.error("give a plan log, or --fake PATH")
    path = args.log
    if args.fake is not None:
        path = write_jsonl(args.fake, fake_log())
    fractions = [float(x) for x in args.audit_fractions.split(",") if x.strip()]
    try:
        report = evaluate(read_plan_log(path), cap=args.cap, audit_fractions=fractions,
                          max_miss_rate=args.max_miss_rate, min_scored_per_group=args.min_scored_per_group)
    except PlanEvalError as exc:
        print(f"plan_eval: {exc}", file=sys.stderr)
        return 2
    md = markdown(report)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out / "plan-eval.json", report)
        atomic_write_bytes(args.out / "plan-eval.md", md.encode("utf-8"))
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
