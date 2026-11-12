"""The chain record `aorlta.episode/1` and the iteration summary (design Section 6.8).

One record per chain, with the doc's fields, plus `source` (Section 6.10 item 8: every stand-in record says so)
and, per turn, the remaining TurnRecord fields (`ideal_cached_tokens`, `t_submit`, `t_done`).

Records carry ids, hashes, counts and timings only (Section 6.10 item 3). A turn's compile result is reduced to
`{ok, rc, errors (a count), seconds, raw_stderr_sha256}`: the structured errors hold the model's own tokens and
clang's messages, which are text, so they stay in the rollout's memory and the sealed evidence store. The check
plan is reduced to `{selector_version, skip}`: its free-text `context` is never written.

`validate_episode` is strict: exact keys, types, ranges, and the consistency rules between fields (ids agree with
the chain id, status agrees with the reward, a trained sample is scored and has an advantage, and so on).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from ..config import REWARD_MAX, REWARD_MIN
from ..ids import INT32_MAX, IdError, parse_chain_id, parse_policy_version, sha256_text
from ..schemas import Candidate, ChainRecord, CompileResult, RewardResult, TurnRecord

EPISODE_SCHEMA = "aorlta.episode/1"
SUMMARY_SCHEMA = "aorlta.iteration_summary/1"
SOURCES = ("standin",)
STATUSES = ("scored", "unscored", "quarantined")
END_REASONS = ("compiled", "no_change", "no_code_block", "turns_exhausted", "context_limit", "quarantined",
               "engine_error", "logprob_error")
LEG_STATES = ("clean", "finding", "fail", "unsettled", "skipped_by_plan", "not_run")
LEAK_CHECKS = ("pass", "fail", "not_run")
EXCLUDED_REASONS = ("quarantined", "engine_error", "unscored", "logprob_error", "no_sample", "group_dropped",
                    "gap_gate", "no_step", "score_only")
SCORE_ONLY = "score_only"  # a score run's episode (orchestrator.mode: score): never trained, sampled from v0
EPISODE_KEYS = frozenset({
    "schema", "source", "run_id", "iteration", "chain_id", "prompt_id", "k", "adapter_version", "engine_id", "turns",
    "final_turn", "end_reason", "candidate_sha256", "reward", "status", "unscored_reason", "reward_version",
    "reward_code_sha256", "legs", "plan", "audit", "advantage", "trained", "excluded_reason",
})
TURN_KEYS = frozenset({
    "turn", "seed", "prompt_ids_sha256", "input_len", "response_ids", "response_logprobs", "finish_reason",
    "cached_tokens", "ideal_cached_tokens", "t_submit", "t_done", "compile", "feedback_sha256", "leak_check",
})
COMPILE_KEYS = frozenset({"ok", "rc", "errors", "seconds", "raw_stderr_sha256"})
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LEG_RE = re.compile(r"(gate|ref|san)\.[a-z0-9_]+")
REASON_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")  # reasons are identifiers, never free text
MAX_LOGPROB = 1e-3  # WP2's alignment rule: a log-prob above this counts as misaligned


class EpisodeError(ValueError):
    """A record breaks the aorlta.episode/1 schema or the iteration summary schema."""


# --- building ---------------------------------------------------------------------------------------------


def compile_summary(c: CompileResult | None) -> dict | None:
    if c is None:
        return None
    return {"ok": c.ok, "rc": c.rc, "errors": len(c.errors), "seconds": float(c.seconds),
            "raw_stderr_sha256": c.raw_stderr_sha256}


def turn_record(tr: TurnRecord) -> dict:
    return {
        "turn": tr.turn, "seed": tr.seed, "prompt_ids_sha256": tr.input_ids_sha256, "input_len": tr.input_len,
        "response_ids": [int(x) for x in tr.response_ids], "response_logprobs": [float(x) for x in tr.response_logprobs],
        "finish_reason": dict(tr.finish_reason), "cached_tokens": tr.cached_tokens,
        "ideal_cached_tokens": tr.ideal_cached_tokens, "t_submit": float(tr.t_submit), "t_done": float(tr.t_done),
        "compile": compile_summary(tr.compile), "feedback_sha256": tr.feedback_sha256, "leak_check": tr.leak_check,
    }


def plan_summary(plan: Mapping | None) -> dict | None:
    """C3's plan as `{selector_version, skip}`: which legs it would have skipped. The plan's `context` is free text
    and is never written."""
    if plan is None:
        return None
    legs = plan.get("legs") if isinstance(plan.get("legs"), Mapping) else {}
    skip = sorted(leg for leg, d in legs.items() if isinstance(d, Mapping) and d.get("decision") == "skip")
    version = plan.get("selector_version")
    return {"selector_version": version if isinstance(version, str) else None, "skip": skip}


def build_episode(*, source: str, run_id: str, iteration: int, record: ChainRecord, candidate: Candidate | None,
                  result: RewardResult | None, advantage: float | None, trained: bool,
                  excluded_reason: str | None) -> dict:
    if result is None:
        # No candidate was handed over (engine_error, quarantined), or none came back: never a default reward.
        status = "quarantined" if record.end_reason == "quarantined" else "unscored"
        reward, version, code = None, None, None
        unscored_reason = record.end_reason if record.end_reason in ("quarantined", "engine_error") else "no_result"
        legs: dict[str, str] = {}
        plan, audit = None, False
    else:
        status, reward = result.status, result.reward
        unscored_reason = None if status == "scored" else (result.reason or "reason_missing")
        version = result.reward_fn.get("version") if isinstance(result.reward_fn, Mapping) else None
        code = result.reward_fn.get("code_sha256") if isinstance(result.reward_fn, Mapping) else None
        legs = {leg: lr.state for leg, lr in sorted(result.legs.items())}
        plan, audit = plan_summary(result.plan), bool(result.audit)
    return {
        "schema": EPISODE_SCHEMA, "source": source, "run_id": run_id, "iteration": iteration,
        "chain_id": record.chain_id, "prompt_id": record.prompt_id, "k": record.k,
        "adapter_version": record.policy_version, "engine_id": record.engine_id,
        "turns": [turn_record(t) for t in record.turns], "final_turn": record.final_turn,
        "end_reason": record.end_reason,
        "candidate_sha256": sha256_text(candidate.response_text) if candidate is not None else None,
        "reward": float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None,
        "status": status, "unscored_reason": unscored_reason,
        "reward_version": version if isinstance(version, str) else None,
        "reward_code_sha256": code if isinstance(code, str) else None,
        "legs": legs, "plan": plan, "audit": audit,
        "advantage": float(advantage) if advantage is not None else None,
        "trained": bool(trained), "excluded_reason": excluded_reason,
    }


# --- validation -------------------------------------------------------------------------------------------


def _fail(where: str, message: str) -> None:
    raise EpisodeError(f"{where}: {message}")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _keys(where: str, obj: Any, keys: frozenset) -> None:
    if not isinstance(obj, Mapping):
        _fail(where, f"expected an object, got {type(obj).__name__}")
    got = set(obj)
    if got != keys:
        _fail(where, f"keys differ from the schema (missing {sorted(keys - got)}, unknown {sorted(got - keys)})")


def _sha(where: str, value: Any, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not (isinstance(value, str) and SHA256_RE.fullmatch(value)):
        _fail(where, f"expected a sha256 hex digest, got {value!r:.80}")


def _reason(where: str, value: Any) -> None:
    if not (isinstance(value, str) and REASON_RE.fullmatch(value)):
        _fail(where, f"a reason is an identifier, got {value!r:.80}")


def _validate_turn(where: str, t: Any, index: int, end: str, final: bool) -> None:
    quarantined = end == "quarantined"
    _keys(where, t, TURN_KEYS)
    if t["turn"] != index or not _is_int(t["turn"]):
        _fail(where, f"turn is {t['turn']!r}, expected {index}")
    if not (_is_int(t["seed"]) and 0 <= t["seed"] <= INT32_MAX):
        _fail(where, f"seed must be a 31-bit int, got {t['seed']!r}")
    _sha(f"{where}.prompt_ids_sha256", t["prompt_ids_sha256"])
    if not (_is_int(t["input_len"]) and t["input_len"] >= 1):
        _fail(where, f"input_len must be an int >= 1, got {t['input_len']!r}")
    ids, lps = t["response_ids"], t["response_logprobs"]
    if not (isinstance(ids, list) and all(_is_int(x) and 0 <= x <= INT32_MAX for x in ids)):
        _fail(where, "response_ids must be a list of token ids")
    if not (isinstance(lps, list) and all(_is_num(x) and x <= MAX_LOGPROB for x in lps)):
        _fail(where, "response_logprobs must be a list of finite log-probabilities")
    if quarantined:
        if ids or lps:
            _fail(where, "a quarantined chain keeps only hashes: its turns carry no ids or log-probs")
    else:
        if not ids:
            _fail(where, "an answered turn has at least one response id")
        if not lps and not (final and end == "logprob_error"):
            _fail(where, "only the final turn of a logprob_error chain may lack log-probs")
        if lps and len(lps) != len(ids):
            _fail(where, f"{len(lps)} log-probs for {len(ids)} ids")
    fr = t["finish_reason"]
    if not (isinstance(fr, Mapping) and fr.get("type") in ("stop", "length")):
        _fail(where, f"finish_reason must be a stop or length reason, got {fr!r:.80}")
    for name in ("cached_tokens", "ideal_cached_tokens"):
        if not (_is_int(t[name]) and t[name] >= -1):
            _fail(where, f"{name} must be an int >= -1, got {t[name]!r}")
    if not (_is_num(t["t_submit"]) and _is_num(t["t_done"]) and t["t_done"] >= t["t_submit"]):
        _fail(where, "t_submit and t_done must be finite with t_done >= t_submit")
    c = t["compile"]
    if c is not None:
        _keys(f"{where}.compile", c, COMPILE_KEYS)
        if not (isinstance(c["ok"], bool) and _is_int(c["rc"]) and _is_int(c["errors"]) and c["errors"] >= 0
                and _is_num(c["seconds"]) and c["seconds"] >= 0):
            _fail(f"{where}.compile", "ok is a bool, rc an int, errors a count, seconds >= 0")
        _sha(f"{where}.compile.raw_stderr_sha256", c["raw_stderr_sha256"])
    _sha(f"{where}.feedback_sha256", t["feedback_sha256"], optional=True)
    if t["leak_check"] not in LEAK_CHECKS:
        _fail(where, f"leak_check must be one of {LEAK_CHECKS}, got {t['leak_check']!r}")


def validate_episode(rec: Any) -> None:
    where = "episode"
    _keys(where, rec, EPISODE_KEYS)
    where = f"episode {rec.get('chain_id')!r:.120}"
    if rec["schema"] != EPISODE_SCHEMA:
        _fail(where, f"schema {rec['schema']!r}")
    if rec["source"] not in SOURCES:
        _fail(where, f"source must be one of {SOURCES}, got {rec['source']!r}")
    try:
        key = parse_chain_id(rec["chain_id"])
        run_id, version = parse_policy_version(rec["adapter_version"])
    except IdError as exc:
        _fail(where, str(exc))
    if (key.run_id, key.iteration, key.prompt_id, key.k) != (rec["run_id"], rec["iteration"], rec["prompt_id"], rec["k"]):
        _fail(where, "run_id, iteration, prompt_id and k must be the chain id's")
    sampled_from = 0 if rec["excluded_reason"] == SCORE_ONLY else rec["iteration"] - 1
    if run_id != rec["run_id"] or version != sampled_from:
        _fail(where, f"adapter_version {rec['adapter_version']!r}: iteration n samples from v(n-1) of the same run, "
                     f"and every batch of a score run from v0")
    if not (_is_int(rec["engine_id"]) and rec["engine_id"] >= 0):
        _fail(where, f"engine_id must be an int >= 0, got {rec['engine_id']!r}")
    end = rec["end_reason"]
    if end not in END_REASONS:
        _fail(where, f"end_reason must be one of {END_REASONS}, got {end!r}")
    turns = rec["turns"]
    if not isinstance(turns, list):
        _fail(where, "turns must be a list")
    if not turns and end != "engine_error":
        _fail(where, f"only an engine_error chain can end before its first reply, not {end!r}")
    for i, t in enumerate(turns):
        _validate_turn(f"{where}.turns[{i}]", t, i, end, i == len(turns) - 1)
    if rec["final_turn"] != len(turns) - 1:
        _fail(where, f"final_turn {rec['final_turn']!r} but {len(turns)} turns")
    status, reward = rec["status"], rec["reward"]
    if status not in STATUSES:
        _fail(where, f"status must be one of {STATUSES}, got {status!r}")
    if (status == "quarantined") != (end == "quarantined"):
        _fail(where, "status quarantined exactly when the chain ended quarantined")
    if status == "scored":
        if not (_is_num(reward) and REWARD_MIN <= reward <= REWARD_MAX):
            _fail(where, f"a scored sample's reward is a number in [{REWARD_MIN}, {REWARD_MAX}], got {reward!r}")
        if rec["unscored_reason"] is not None:
            _fail(where, "a scored sample has no unscored_reason")
    else:
        if reward is not None:
            _fail(where, f"an {status} sample has reward None, got {reward!r}")
        _reason(f"{where}.unscored_reason", rec["unscored_reason"])
    has_candidate = rec["candidate_sha256"] is not None
    _sha(f"{where}.candidate_sha256", rec["candidate_sha256"], optional=True)
    if has_candidate == (end in ("engine_error", "quarantined")):
        _fail(where, "exactly the chains that did not end engine_error or quarantined hand over a candidate")
    if status == "scored" and not has_candidate:
        _fail(where, "a scored sample has a candidate")
    for name in ("reward_version", "reward_code_sha256"):
        if rec[name] is not None and not isinstance(rec[name], str):
            _fail(where, f"{name} must be a string or null")
    if status == "scored" and (rec["reward_version"] is None or rec["reward_code_sha256"] is None):
        _fail(where, "a scored sample records its reward function's version and code hash")
    legs = rec["legs"]
    if not (isinstance(legs, Mapping) and all(isinstance(k, str) and LEG_RE.fullmatch(k) and v in LEG_STATES
                                               for k, v in legs.items())):
        _fail(where, "legs maps leg names to leg states")
    plan = rec["plan"]
    if plan is not None:
        _keys(f"{where}.plan", plan, frozenset({"selector_version", "skip"}))
        if not (isinstance(plan["skip"], list) and all(isinstance(x, str) and LEG_RE.fullmatch(x) for x in plan["skip"])):
            _fail(where, "plan.skip lists leg names")
        if plan["selector_version"] is not None and not isinstance(plan["selector_version"], str):
            _fail(where, "plan.selector_version is a string or null")
    if not isinstance(rec["audit"], bool):
        _fail(where, "audit must be a bool")
    adv, trained, excl = rec["advantage"], rec["trained"], rec["excluded_reason"]
    if adv is not None and not _is_num(adv):
        _fail(where, f"advantage must be a finite number or null, got {adv!r}")
    if adv is not None and status != "scored":
        _fail(where, "only a scored sample has an advantage")
    if not isinstance(trained, bool):
        _fail(where, "trained must be a bool")
    if trained:
        if status != "scored" or adv is None or excl is not None:
            _fail(where, "a trained sample is scored, has an advantage and no excluded_reason")
        if end in ("logprob_error", "engine_error", "quarantined"):
            _fail(where, f"a {end} chain is never trained")
    else:
        if excl not in EXCLUDED_REASONS:
            _fail(where, f"an untrained sample says why, one of {EXCLUDED_REASONS}; got {excl!r}")


# --- the iteration summary ----------------------------------------------------------------------------------

SUMMARY_KEYS = frozenset({
    "schema", "source", "run_id", "iteration", "policy_version_sampled", "policy_version_trained", "samples",
    "scored_fraction", "status_counts", "unscored_by_reason", "end_reasons", "reward", "gates", "compile",
    "truncation", "length", "cache", "reward_service", "groups", "trainer", "sync", "policy", "deadlines", "timing",
    "spans", "noop_streak", "warnings", "eval", "rollout",
})


def validate_summary(s: Any) -> None:
    _keys("summary", s, SUMMARY_KEYS)
    where = f"summary {s.get('iteration')!r}"
    if s["schema"] != SUMMARY_SCHEMA or s["source"] not in SOURCES:
        _fail(where, "schema and source")
    if not (_is_int(s["iteration"]) and s["iteration"] >= 1):
        _fail(where, "iteration must be an int >= 1")
    for name in ("policy_version_sampled", "policy_version_trained"):
        try:
            run_id, _ = parse_policy_version(s[name])
        except IdError as exc:
            _fail(where, f"{name}: {exc}")
        if run_id != s["run_id"]:
            _fail(where, f"{name} names another run")
    f = s["scored_fraction"]
    if not (_is_num(f) and 0 <= f <= 1):
        _fail(where, f"scored_fraction must be in [0, 1], got {f!r}")
    for name in ("samples", "status_counts", "unscored_by_reason", "end_reasons"):
        d = s[name]
        if not (isinstance(d, Mapping) and all(isinstance(k, str) and _is_int(v) and v >= 0 for k, v in d.items())):
            _fail(where, f"{name} maps names to counts")
    for k in s["unscored_by_reason"]:
        _reason(f"{where}.unscored_by_reason", k)
    if not (isinstance(s["warnings"], list) and all(isinstance(w, str) and REASON_RE.fullmatch(w) for w in s["warnings"])):
        _fail(where, "warnings are identifiers")
    if not (_is_int(s["noop_streak"]) and s["noop_streak"] >= 0):
        _fail(where, "noop_streak must be a count")
