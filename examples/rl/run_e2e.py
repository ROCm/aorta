#!/usr/bin/env python3
"""End-to-end readiness check for the sanitizer-selection-and-routing slice.

Serves a real model, drives `aorta agent`'s own proposer against the committed
sanitizer corpus, and scores every output with the two existing graders. The
question is not what the model scores -- it is whether the scoring can *train*
anything, which is a different measurement and needs the group structure below.

What makes this the real path
-----------------------------
The proposals come from `aorta.agent.llm.LiteLLMProposer`, not from a prompt
written here. `propose()` owns the system prompt, the user payload, the
`response_format` and the post-filter, so what is measured is the contract the
policy will actually have to satisfy. Pointing it at a self-hosted engine needs
no code change: `OPENAI_API_BASE` and `OPENAI_API_KEY`, and a `model` carrying
litellm's `openai/` routing prefix -- which is stripped on the wire, so the
remainder has to be exactly what the engine advertises.

`propose()` returns an `AgentStep`, already coerced by `from_dict` and already
filtered against the offered set. That is the wrong input for
`proposal_reward.py`, whose first two tiers measure the *raw* wire text: by the
time an `AgentStep` exists, a truncated object has become a safe stop and a
string `stop` has become `False`. So `litellm.completion` is wrapped rather than
replaced -- the proposer calls it exactly as it always does, and the wrapper
records the request kwargs and the raw response content on the way past. The
`AgentStep` the proposer built is kept too, so the grader's replay of the
consumer's behaviour can be checked against the consumer itself.

Sampling is per-scenario groups, not one shot
---------------------------------------------
GRPO computes an advantage within a group of completions for the same prompt.
If every sample in a group scores the same, the advantage is zero and the batch
teaches nothing -- so a mean reward, however healthy, does not establish that a
reward is trainable. `--samples` draws a real group per scenario and the
aggregates report within-group spread, which is the quantity that decides it.

Serving parameters versus the contract
--------------------------------------
`--temperature` and `--no-think` are injected by the wrapper, and both are
serving parameters rather than changes to what the proposer asks for. The
distinction matters for `--no-think`: a Qwen3-class model emits a reasoning
trace ahead of its answer unless the chat template is told not to, and that
trace lands in `message.content` in front of the JSON. Whether the format gate
survives that is a property of the serving configuration, not of the model's
grasp of the schema, so it is worth being able to measure both ways with one
prompt held fixed.

The triage prompt is ours, and the proposal prompt is not
--------------------------------------------------------
aorta ships no triage prompt -- `triage_reward.py` is a scorer over fixtures --
so the one used here is written in this file and marked as such. It hands the
model the evidence the sanitizer report carries and asks for a verdict plus the
detector IDs that justify it. Both halves are derivable from the evidence
supplied, so this measures whether the model applies the resolver's precedence
rule and cites what actually fired; it does not measure diagnosis from raw logs.

Usage
-----

    python examples/rl/run_e2e.py \
        --corpus examples/rl/corpus/triage.jsonl \
        --base-url http://127.0.0.1:8000/v1 \
        --model openai/Qwen/Qwen3-8B \
        --samples 5 \
        --out results/as-is.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proposal_reward import MAX_TIER, Proposal, score_proposal  # noqa: E402
from triage_reward import (  # noqa: E402
    Answer,
    Label,
    load_corpus as load_triage_corpus,
    score_answer,
)

from aorta.agent.llm import AUTOPSY_CATEGORIES, LiteLLMProposer  # noqa: E402
from aorta.registry import load_mitigations  # noqa: E402

# The candidate set and tried list `build_corpus.py` uses for every scenario.
# Reused verbatim so a real model's tier-4 and tier-5 outcomes are comparable
# with the synthetic corpus rows rather than measured against a different
# registry slice.
CANDIDATES = ["hsa_no_sdma", "hip_launch_blocking", "amd_log_level_4", "none"]
TRIED = ["hsa_no_sdma"]


def full_candidates() -> list[str]:
    """Every registered mitigation, as the widest candidate set the loop allows.

    The corpus slice leaves exactly two names available, so tiers 4 and 5 are
    nearly free: a policy that copies both offered names scores 1.0 without
    choosing anything. Widening to the registry is what makes "names a
    registered mitigation" and "names an offered one" separable measurements
    again, and it is a legitimate loop state -- `--mitigation` narrowing is an
    operator's choice, not a property of the task.
    """
    return sorted(load_mitigations().keys())

# The verdict vocabulary a sanitizer report can carry. Offered to the model
# explicitly, so a wrong verdict is a wrong choice from a stated set rather than
# a guess at what the set is.
TRIAGE_VERDICTS = ["pass", "warn", "fail", "error", "not_checked"]


@dataclass
class Recorded:
    """One `litellm.completion` round trip, as it went over the wire."""

    request: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    reasoning: str = ""
    error: str = ""
    latency_sec: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


def sample_seed(base: int, scenario: str, index: int) -> int:
    """A per-sample seed that is distinct but reproducible.

    Rollouts want both: GRPO needs the samples in a group to *differ*, and a
    result nobody can re-derive is not evidence. Hashing the scenario id with
    the sample index gives a value that varies across a group, does not repeat
    across groups, and is a pure function of `--seed` -- so the whole run
    replays from one integer.

    `hashlib` rather than `hash()`, which is salted per process.
    """
    import hashlib

    digest = hashlib.sha256(f"{base}:{scenario}:{index}".encode()).digest()
    # Positive and inside int32, which is the range engines accept.
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class RecordingLiteLLM:
    """Wraps `litellm.completion` to record it, and to inject serving params.

    Injection is deliberately narrow: a temperature, a per-request seed, and
    the chat-template switch that suppresses a reasoning trace. None of them
    touches the messages, the model or the `response_format`, so the proposer's
    contract goes out unaltered and the recorded request proves it.

    Why the temperature lives here and not in `LiteLLMProposer`
    -----------------------------------------------------------
    `LiteLLMProposer` is the *production* path: it is what `aorta agent` calls
    to diagnose a real failure. For a diagnostic tool, reproducibility is a
    feature -- the same evidence should yield the same recommendation, an
    operator should be able to re-run a triage and get the same answer, and the
    nightly matrix carries an `llm_determinism` entry, so this project already
    treats determinism as something to measure. Making the shipped agent
    stochastic to serve a training harness would be a real behaviour change to
    the consumer, made invisibly and for the wrong reason.

    Sampling diversity is a property of how *training data is collected*, not of
    the consumer contract. So it is threaded through the rollout driver as an
    explicit, off-by-default parameter, and the proposer is left exactly as it
    ships. `--temperature` omitted means the engine default, which is what the
    first end-to-end run measured.
    """

    def __init__(
        self,
        *,
        temperature: float | None = None,
        no_think: bool = False,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> None:
        import litellm

        self._litellm = litellm
        self._real = litellm.completion
        self._temperature = temperature
        self._no_think = no_think
        self._max_tokens = max_tokens
        self._retries = retries
        self.calls: list[Recorded] = []
        # Set by the driver immediately before each call, so one recorder can
        # give every sample in a group its own seed. `None` sends none.
        self.seed: int | None = None
        # How the seed goes over the wire. `top_level` is the OpenAI-standard
        # `seed=`; `sampling_seed` is the name TokenSpeed's SGLang-compat layer
        # shims onto its own `seed`, reached through `extra_body`. Which one the
        # engine actually honours is measured by `probe_seed.py`, not assumed.
        self.seed_mode: str = "top_level"

    def install(self) -> None:
        self._litellm.completion = self._call  # type: ignore[assignment]

    def restore(self) -> None:
        self._litellm.completion = self._real  # type: ignore[assignment]

    def _call(self, **kwargs: Any) -> Any:
        if self._temperature is not None:
            kwargs.setdefault("temperature", self._temperature)
        if self._max_tokens is not None:
            kwargs.setdefault("max_tokens", self._max_tokens)
        if self.seed is not None:
            if self.seed_mode == "top_level":
                kwargs.setdefault("seed", self.seed)
            else:
                extra = dict(kwargs.get("extra_body") or {})
                extra.setdefault(self.seed_mode, self.seed)
                kwargs["extra_body"] = extra
        if self._no_think:
            extra = dict(kwargs.get("extra_body") or {})
            extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
            kwargs["extra_body"] = extra

        record = Recorded(request=_summarise_request(kwargs))
        started = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._real(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                last_exc = exc
                if attempt < self._retries:
                    time.sleep(2.0 * (attempt + 1))
        else:
            record.latency_sec = time.monotonic() - started
            record.error = f"{type(last_exc).__name__}: {last_exc}"
            self.calls.append(record)
            raise last_exc  # type: ignore[misc]

        record.latency_sec = time.monotonic() - started
        message = response.choices[0].message
        record.content = message.content or ""
        # Some OpenAI-compatible gateways split a reasoning trace out of
        # `content` into its own field. Recorded either way: if the trace is
        # split off, the format gate never sees it, and that is the difference
        # between the two conditions this script can run.
        record.reasoning = str(getattr(message, "reasoning_content", "") or "")
        usage = getattr(response, "usage", None)
        if usage is not None:
            record.usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        self.calls.append(record)
        return response


def _summarise_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request, minus the payload bulk, for the record."""
    out = {k: v for k, v in kwargs.items() if k != "messages"}
    messages = kwargs.get("messages") or []
    out["message_roles"] = [m.get("role") for m in messages]
    out["system_prompt_sha"] = None
    for message in messages:
        if message.get("role") == "system":
            import hashlib

            out["system_prompt_sha"] = hashlib.sha256(
                (message.get("content") or "").encode()
            ).hexdigest()[:12]
    return out


# --------------------------------------------------------------------------- #
# Loop state: what the real agent loop would have in front of the model
# --------------------------------------------------------------------------- #

def loop_state(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """A scenario's symptom line and cell summaries, from its corpus row.

    Shaped like what `run_agent_loop` accumulates: one summary per executed
    cell, carrying the verdict and the detectors that fired. The sanitizer
    evidence is the cell's, so the model sees the same thing the loop would.
    """
    scenario = row["scenario_id"]
    label = row["label"]
    checks = row.get("checks") or []
    check_line = ", ".join(
        f"{c.get('sanitizer')}={c.get('verdict')} ({c.get('findings')} findings)"
        for c in checks
    )
    symptom = (
        f"Sanitizer run {scenario!r} on gfx950 returned overall verdict "
        f"{label['verdict']!r}. Per-sanitizer: {check_line or 'none'}. "
        f"Workload family: {row.get('workload_family')}."
    )

    kernels = sorted(
        {k for c in checks for k in (c.get("kernel_names") or [])}
    )
    summaries = [
        {
            "cell_name": "none-none",
            "verdict": label["verdict"],
            "failure_detectors_fired": list(label.get("failure_detectors") or []),
            "error_detectors_fired": list(label.get("error_detectors") or []),
            "kernel_names": kernels,
            "distinct_evidence": [
                {
                    "sanitizer": e.get("sanitizer"),
                    "code": e.get("code"),
                    "severity": e.get("severity"),
                    "kernel_names": e.get("kernel_names_from_identity"),
                    "metadata": e.get("metadata"),
                }
                for e in (row.get("distinct_evidence") or [])
            ],
            "finding_counts": row.get("finding_counts"),
        }
    ]
    return symptom, summaries


def _claimed(raw: str, key: str) -> Any:
    """One field as the model wrote it, or None if the output did not parse."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj.get(key) if isinstance(obj, dict) else None


def failure_kind(stopped_at: str, detail: str, tier: int) -> str:
    """The failure breakdown key for one scored proposal."""
    if tier == MAX_TIER:
        return "on_contract"
    if stopped_at == "tier1_json":
        return "malformed_json"
    if stopped_at == "tier2_schema":
        return "missing_key" if "missing key" in detail else "wrong_type"
    if stopped_at == "tier3_category":
        return "category_outside_set"
    if stopped_at == "tier4_registry":
        if "no mitigation proposed" in detail:
            return "empty_mitigations"
        return "hallucinated_mitigation"
    if stopped_at == "tier5_available":
        if "confidence" in detail:
            return "confidence_out_of_range"
        return "registered_but_not_offered"
    return "unknown"


# --------------------------------------------------------------------------- #
# The two drives
# --------------------------------------------------------------------------- #

def drive_proposals(
    rows: list[dict[str, Any]],
    *,
    model: str,
    samples: int,
    recorder: RecordingLiteLLM,
    candidates: list[str] | None = None,
    tried: list[str] | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Sample a group of real proposals per scenario, through the real proposer."""
    proposer = LiteLLMProposer(model=model)
    candidates = list(candidates or CANDIDATES)
    tried = list(tried or TRIED)
    offered = [c for c in candidates if c not in tried and c != "none"]
    out: list[dict[str, Any]] = []

    for row in rows:
        scenario = row["scenario_id"]
        symptom, summaries = loop_state(row)
        for index in range(samples):
            # Per sample, not per run: a single seed for the whole group would
            # make the group's completions identical again, which is the defect
            # this parameter exists to remove.
            recorder.seed = (
                None if seed is None else sample_seed(seed, scenario, index)
            )
            before = len(recorder.calls)
            step: Any = None
            error = ""
            try:
                step = proposer.propose(
                    symptom=symptom,
                    cell_summaries=summaries,
                    candidates=candidates,
                    tried=tried,
                )
            except Exception as exc:  # noqa: BLE001 - a transport failure is a result
                error = f"{type(exc).__name__}: {exc}"

            record = recorder.calls[before] if len(recorder.calls) > before else Recorded()
            proposal = Proposal(
                name=f"{scenario}:sample{index}",
                raw=record.content,
                candidates=candidates,
                tried=tried,
            )
            score = score_proposal(proposal)
            out.append(
                {
                    "scenario_id": scenario,
                    "workload_family": row.get("workload_family"),
                    "sample": index,
                    "raw": record.content,
                    "reasoning_split_off": bool(record.reasoning),
                    "reasoning_chars": len(record.reasoning),
                    "raw_chars": len(record.content),
                    "latency_sec": round(record.latency_sec, 2),
                    "usage": record.usage,
                    "transport_error": error or record.error,
                    "tier": score.tier,
                    "reward": round(score.reward, 4),
                    "stopped_at": score.stopped_at,
                    "detail": score.detail,
                    "consumer_outcome": score.consumer_outcome,
                    "failure_kind": failure_kind(
                        score.stopped_at, score.detail, score.tier
                    ),
                    # Recorded separately from the tier, because `unknown` is a
                    # member of the closed set: a proposal that declines to
                    # classify passes tier 3 and can reach 1.0. Whether it did
                    # so by diagnosing or by abstaining is not visible in the
                    # reward, and this is the field that makes it visible.
                    "category_claimed": _claimed(record.content, "category"),
                    "mitigations_claimed": _claimed(
                        record.content, "next_mitigations"
                    ),
                    # What the consumer itself built, so the grader's replay of
                    # the consumer can be checked against the consumer.
                    "proposer_step": None
                    if step is None
                    else {
                        "category": step.category,
                        "next_mitigations": list(step.next_mitigations),
                        "confidence": step.confidence,
                        "stop": step.stop,
                        "stop_reason": step.stop_reason,
                    },
                    "offered": offered,
                    # Recorded so a single completion can be replayed on its
                    # own, and so a group's seeds can be checked for being
                    # distinct rather than assumed to be.
                    "seed": recorder.seed,
                }
            )
            if verbose:
                print(
                    f"  proposal {scenario}:{index} tier {score.tier}/{MAX_TIER} "
                    f"reward {score.reward:.2f} {score.consumer_outcome} "
                    f"({record.latency_sec:.1f}s)",
                    flush=True,
                )
    return out


TRIAGE_SYSTEM = (
    "You are an AORTA triage classifier. You are given one sanitizer run's "
    "evidence. Decide what happened and cite the evidence that justifies it. "
    "Return strict JSON with keys: verdict (one of "
    f"{TRIAGE_VERDICTS}), detectors (list of strings), reasoning (string). "
    "Each detector must be a finding you were shown, written as "
    "'<sanitizer>:<code>' -- for example 'consan:1' or 'waitcheck:wait_hazard'. "
    "Cite every distinct finding code that fired and nothing else; a clean run "
    "cites an empty list."
)


def drive_triage(
    rows: list[dict[str, Any]],
    labels: dict[str, Label],
    *,
    model: str,
    recorder: RecordingLiteLLM,
    blind: bool = False,
    hide_status: bool = False,
    seed: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Ask the model to triage each scenario, and score verdict + attribution.

    Two independent ablations, because they remove different things and
    conflating them mis-attributes the result:

    `blind`
        withholds each check's own verdict. With those supplied the overall
        verdict is just the max-ranked one, so the answer is a copy rather than
        a judgement -- faithful to what the report contains, but it caps what
        the measurement can show. Blind asks the model to apply the resolver's
        precedence rule itself.
    `hide_status`
        withholds `execution_status`. This is the only field that separates
        `error` from `pass` on a run with zero findings, so removing it makes
        the fail-vs-error split -- the one the reward weights most -- decidable
        from nothing that remains.
    """
    import litellm

    out: list[dict[str, Any]] = []
    for row in rows:
        scenario = row["scenario_id"]
        label = labels[f"triage:{scenario}"]
        payload: dict[str, Any] = {
            "scenario": scenario,
            "workload_family": row.get("workload_family"),
            "checks": [
                {
                    "sanitizer": c.get("sanitizer"),
                    **({} if blind else {"verdict": c.get("verdict")}),
                    "findings": c.get("findings"),
                    "kernel_names": c.get("kernel_names"),
                }
                for c in (row.get("checks") or [])
            ],
            "distinct_findings": [
                {
                    "sanitizer": e.get("sanitizer"),
                    "code": e.get("code"),
                    "severity": e.get("severity"),
                    "kernel_names": e.get("kernel_names_from_identity"),
                }
                for e in (row.get("distinct_evidence") or [])
            ],
            "raw_finding_count": (row.get("finding_counts") or {}).get("raw"),
        }
        if not hide_status:
            payload["execution_status"] = (row.get("ground_truth") or {}).get(
                "observed_execution_status"
            )

        # One triage answer per scenario, so there is no group to diversify --
        # the seed is here only to make the answer reproducible.
        recorder.seed = None if seed is None else sample_seed(seed, scenario, 0)
        before = len(recorder.calls)
        error = ""
        try:
            litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": TRIAGE_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        record = recorder.calls[before] if len(recorder.calls) > before else Recorded()
        answer, parse_error = parse_triage(record.content)
        score = score_answer(answer, label)
        out.append(
            {
                "scenario_id": scenario,
                "workload_family": row.get("workload_family"),
                "raw": record.content,
                "reasoning_split_off": bool(record.reasoning),
                "raw_chars": len(record.content),
                "latency_sec": round(record.latency_sec, 2),
                "transport_error": error or record.error,
                "parse_error": parse_error,
                "answer": {"verdict": answer.verdict, "detectors": answer.detectors},
                "label": {
                    "verdict": label.verdict,
                    "cited_detectors": sorted(label.cited_detectors),
                },
                "verdict_correct": score.verdict_correct,
                "attribution_f1": round(score.attribution_f1, 4),
                "reward": round(score.reward, 4),
            }
        )
        if verbose:
            print(
                f"  triage {scenario}: said {answer.verdict!r} "
                f"(truth {label.verdict!r}) f1 {score.attribution_f1:.2f} "
                f"reward {score.reward:.2f}",
                flush=True,
            )
    return out


def parse_triage(raw: str) -> tuple[Answer, str]:
    """Read a triage answer out of raw model text.

    Unlike the proposal path, a parse failure here is not the thing being
    measured -- `triage_reward.py` scores a verdict and a citation, not JSON
    discipline, and the proposal ladder already measures format. So an
    unparseable answer becomes an empty answer, which scores whatever an empty
    answer deserves, and the parse failure is recorded alongside.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return Answer(verdict="", detectors=[]), f"does not parse: {exc.msg}"
    if not isinstance(obj, dict):
        return Answer(verdict="", detectors=[]), f"parsed as {type(obj).__name__}"
    verdict = obj.get("verdict")
    verdict = verdict if isinstance(verdict, str) else ""
    detectors_raw = obj.get("detectors")
    detectors = (
        [str(d) for d in detectors_raw] if isinstance(detectors_raw, list) else []
    )
    return Answer(verdict=verdict, detectors=detectors), ""


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def template_baselines(
    candidates: list[str], tried: list[str], scenarios: int
) -> list[dict[str, Any]]:
    """Score fixed templates on the same loop state the model was given.

    The point of comparison the mean reward needs. `proposal_reward.baselines()`
    already shows that a fixed on-contract proposal scores 1.0 on the shipped
    fixtures; these are the same question asked against *this* run's candidate
    set, so the results table can put the model and a template that reads
    nothing side by side under one grader. Where they tie, the reward has not
    measured the model.

    `abstain_and_shotgun` is the specific template the observed outputs
    resemble: decline to classify, then name everything on offer.
    """
    offered = [c for c in candidates if c not in tried and c != "none"]
    templates = {
        "abstain_and_shotgun": {
            "category": "unknown",
            "hypothesis": "",
            "next_mitigations": offered,
            "confidence": 0.5,
            "stop": False,
        },
        "abstain_and_pick_first": {
            "category": "unknown",
            "hypothesis": "",
            "next_mitigations": offered[:1],
            "confidence": 0.5,
            "stop": False,
        },
        "always_prose": None,
    }
    rows: list[dict[str, Any]] = []
    for name, body in templates.items():
        raw = "not JSON, just a sentence." if body is None else json.dumps(body)
        scores = [
            score_proposal(Proposal(name, raw, candidates, tried))
            for _ in range(scenarios)
        ]
        rows.append(
            {
                "template": name,
                "n": len(scores),
                "mean_reward": round(_mean([s.reward for s in scores]), 4),
                "tier": scores[0].tier if scores else None,
                "consumer_outcome": scores[0].consumer_outcome if scores else None,
            }
        )
    return rows


def aggregate(
    proposals: list[dict[str, Any]], triage: list[dict[str, Any]]
) -> dict[str, Any]:
    rewards = [p["reward"] for p in proposals]
    tiers = [p["tier"] for p in proposals]

    tier_hist = {f"tier_{t}": tiers.count(t) for t in range(MAX_TIER + 1)}
    kinds: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    categories: dict[str, int] = {}
    for p in proposals:
        kinds[p["failure_kind"]] = kinds.get(p["failure_kind"], 0) + 1
        outcomes[p["consumer_outcome"]] = outcomes.get(p["consumer_outcome"], 0) + 1
        claimed = p.get("category_claimed")
        key = claimed if isinstance(claimed, str) else "<unparseable>"
        categories[key] = categories.get(key, 0) + 1

    # How a top score was reached, not just that it was. `unknown` is a legal
    # category and naming every offered mitigation is a legal list, so both are
    # routes to 1.0 that involve no diagnosis and no choice. A reward saturated
    # by these two moves is saturated whatever the mean says.
    abstained = sum(1 for p in proposals if p.get("category_claimed") == "unknown")
    # Set inclusion, not list length. A proposal that repeats one name enough
    # times is long but has named one mitigation, and counting it here would
    # report the opposite of what the field means. Each proposal is checked
    # against its own `offered`, so a mixed-condition results file -- one drive
    # with `--full-candidates`, one without -- stays countable.
    shotgun = sum(
        1
        for p in proposals
        if isinstance(p.get("mitigations_claimed"), list)
        and len(p.get("offered") or []) > 1
        and set(p["offered"]).issubset(
            {m for m in p["mitigations_claimed"] if isinstance(m, str)}
        )
    )

    # The format gate, as the plan states it: parses, carries the keys, and
    # names a category in the closed set. That is exactly tier >= 3.
    gate_pass = sum(1 for t in tiers if t >= 3)

    # Within-group spread is what decides trainability under GRPO: a group whose
    # samples all score the same contributes a zero advantage.
    #
    # `distinct_completions` is reported next to it because the two failure modes
    # are different and only one is the reward's. A reward is a function of the
    # completion and the loop state is constant inside a group, so a group whose
    # completions are byte-identical has zero spread under *any* reward -- that
    # is a sampling defect, and the first end-to-end run hit it on all 9 groups
    # by decoding greedily. Zero spread with several distinct completions is the
    # reward saturating. Without this field the two are indistinguishable.
    groups: dict[str, list[float]] = {}
    raws: dict[str, list[str]] = {}
    for p in proposals:
        groups.setdefault(p["scenario_id"], []).append(p["reward"])
        raws.setdefault(p["scenario_id"], []).append(p["raw"])
    per_scenario = {
        scenario: {
            "n": len(vals),
            "distinct_completions": len(set(raws[scenario])),
            "mean": round(_mean(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "spread": round(max(vals) - min(vals), 4),
            "degenerate": max(vals) == min(vals),
        }
        for scenario, vals in sorted(groups.items())
    }
    degenerate_groups = sum(1 for v in per_scenario.values() if v["degenerate"])
    collapsed_groups = sum(
        1 for v in per_scenario.values() if v["distinct_completions"] == 1
    )

    triage_rewards = [t["reward"] for t in triage]
    return {
        "proposal": {
            "n": len(proposals),
            "mean_reward": round(_mean(rewards), 4),
            "min_reward": round(min(rewards), 4) if rewards else 0.0,
            "max_reward": round(max(rewards), 4) if rewards else 0.0,
            "tier_distribution": tier_hist,
            "tier_fractions": {
                k: round(v / len(tiers), 4) if tiers else 0.0
                for k, v in tier_hist.items()
            },
            "format_gate_pass": gate_pass,
            "format_gate_rate": round(gate_pass / len(tiers), 4) if tiers else 0.0,
            "parse_rate": round(
                sum(1 for t in tiers if t >= 1) / len(tiers), 4
            ) if tiers else 0.0,
            "schema_rate": round(
                sum(1 for t in tiers if t >= 2) / len(tiers), 4
            ) if tiers else 0.0,
            "failure_kinds": dict(sorted(kinds.items())),
            "consumer_outcomes": dict(sorted(outcomes.items())),
            "categories_claimed": dict(
                sorted(categories.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "abstained_unknown": abstained,
            "abstained_rate": round(abstained / len(proposals), 4)
            if proposals
            else 0.0,
            "shotgun_all_offered": shotgun,
            "shotgun_rate": round(shotgun / len(proposals), 4) if proposals else 0.0,
            "offered_count": len(proposals[0]["offered"]) if proposals else 0,
            "per_scenario": per_scenario,
            "degenerate_groups": degenerate_groups,
            "collapsed_groups": collapsed_groups,
            "distinct_completions": len({p["raw"] for p in proposals}),
            # What the samples are actually worth. 45 draws over 9 prompts that
            # collapse to 9 completions is an effective n of 9, and quoting 45
            # would be quoting the same observation five times.
            "effective_n": len({(p["scenario_id"], p["raw"]) for p in proposals}),
            "groups": len(per_scenario),
            "transport_errors": sum(1 for p in proposals if p["transport_error"]),
        },
        "triage": {
            "n": len(triage),
            "mean_reward": round(_mean(triage_rewards), 4),
            "verdict_accuracy": round(
                _mean([1.0 if t["verdict_correct"] else 0.0 for t in triage]), 4
            ),
            "mean_attribution_f1": round(
                _mean([t["attribution_f1"] for t in triage]), 4
            ),
            "parse_failures": sum(1 for t in triage if t["parse_error"]),
            "transport_errors": sum(1 for t in triage if t["transport_error"]),
            "per_scenario": {
                t["scenario_id"]: {
                    "said": t["answer"]["verdict"],
                    "truth": t["label"]["verdict"],
                    "verdict_correct": t["verdict_correct"],
                    "attribution_f1": t["attribution_f1"],
                    "reward": t["reward"],
                }
                for t in triage
            },
        },
    }


def print_report(agg: dict[str, Any], meta: dict[str, Any]) -> None:
    prop, tri = agg["proposal"], agg["triage"]
    print()
    print("=" * 72)
    print("Proposal contract (real model, real proposer)")
    print("=" * 72)
    print(f"  samples            {prop['n']} over {prop['groups']} scenarios")
    print(f"  mean reward        {prop['mean_reward']:.4f}")
    print(f"  format gate        {prop['format_gate_rate']:.2%} "
          f"({prop['format_gate_pass']}/{prop['n']} reach tier 3)")
    print(f"  parses as JSON     {prop['parse_rate']:.2%}")
    print(f"  degenerate groups  {prop['degenerate_groups']}/{prop['groups']} "
          "(zero within-group spread -> zero GRPO advantage)")
    print(f"  collapsed groups   {prop.get('collapsed_groups')}/{prop['groups']} "
          "(one distinct completion -> no reward can create spread)")
    print(f"  distinct outputs   {prop.get('distinct_completions')}/{prop['n']}  "
          f"effective n {prop.get('effective_n')}")
    print("  tier distribution")
    for key, count in prop["tier_distribution"].items():
        share = prop["tier_fractions"][key]
        print(f"    {key}  {count:>4}  {share:>7.2%}")
    print(f"  abstained (category=unknown)  {prop['abstained_rate']:.2%} "
          f"({prop['abstained_unknown']}/{prop['n']})")
    print(f"  named all {prop['offered_count']} offered mitigations  "
          f"{prop['shotgun_rate']:.2%} ({prop['shotgun_all_offered']}/{prop['n']})")
    print("  categories claimed")
    for key, count in prop["categories_claimed"].items():
        print(f"    {key:<32} {count}")
    print("  failure kinds")
    for key, count in prop["failure_kinds"].items():
        print(f"    {key:<32} {count}")
    print("  consumer outcomes")
    for key, count in prop["consumer_outcomes"].items():
        print(f"    {key:<32} {count}")
    print("  fixed templates on the same loop state (read nothing)")
    for row in prop.get("template_baselines", []):
        print(f"    {row['template']:<32} mean {row['mean_reward']:.4f}  "
              f"tier {row['tier']}  {row['consumer_outcome']}")
    print()
    print("=" * 72)
    print("Triage (verdict vs attribution)")
    print("=" * 72)
    print(f"  mean reward         {tri['mean_reward']:.4f}")
    print(f"  verdict accuracy    {tri['verdict_accuracy']:.2%}")
    print(f"  attribution F1      {tri['mean_attribution_f1']:.4f}")
    print(f"  parse failures      {tri['parse_failures']}/{tri['n']}")
    print()
    # 0.533, not the 0.629 the rollout plan quotes. Recomputed here from the
    # nine committed labels: always-`pass` is right on 4 of 9, and cites
    # correctly on the 4 passes plus the 2 zero-finding errors, so the reward is
    # 0.6*(4/9) + 0.4*(6/9). `triage_reward.py --corpus` prints the same number,
    # and no data source in the tree produces 0.629.
    print("  reference points: oracle 1.0, always-pass floor 0.533 "
          "(recomputed; the plan's 0.629 does not reproduce)")
    print(f"  run: {meta.get('model')} on {meta.get('node')}, "
          f"condition {meta.get('condition')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True,
                        help="triage.jsonl written by build_corpus.py")
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True,
                        help="litellm model id; the openai/ prefix is stripped "
                             "on the wire, so the rest must match what the "
                             "engine advertises")
    parser.add_argument("--samples", type=int, default=5,
                        help="completions per scenario (the GRPO group)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="injected serving parameter; omit to use the "
                             "engine default and leave the proposer's call "
                             "exactly as it ships. Off by default on purpose: "
                             "`LiteLLMProposer` is the production path and a "
                             "diagnostic tool should be reproducible, so "
                             "sampling belongs to the rollout, not the agent")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed; each sample gets sha256(seed:scenario:"
                             "index), recorded per sample. NOT reproducibility: "
                             "TokenSpeed accepted and ignored every seed form "
                             "tested (see --seed-mode and probe_seed.py), so "
                             "this makes the request replayable, not the "
                             "completion. Do not read a matching seed as "
                             "evidence that two runs sampled alike")
    parser.add_argument("--seed-mode", default="top_level",
                        choices=["top_level", "sampling_seed", "seed"],
                        help="how the seed reaches the engine: the OpenAI "
                             "standard `seed=`, or an extra_body key -- "
                             "`sampling_seed` is what TokenSpeed's SGLang-compat "
                             "layer shims onto its own seed. Measure with "
                             "probe_seed.py rather than guessing")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-think", action="store_true",
                        help="suppress the reasoning trace via "
                             "chat_template_kwargs.enable_thinking=false")
    parser.add_argument("--skip-triage", action="store_true")
    parser.add_argument("--include-disagreements", action="store_true",
                        help="also sample scenarios whose observed verdict "
                             "contradicts the committed baseline. Off by "
                             "default: those rows score a policy against a "
                             "verdict already known to be wrong. On is the "
                             "right setting when the question is how the tool "
                             "behaves rather than what the answer is")
    parser.add_argument("--full-candidates", action="store_true",
                        help="offer the whole registry instead of the corpus's "
                             "four-name slice, so tiers 4 and 5 involve a choice")
    parser.add_argument("--triage-blind", action="store_true",
                        help="withhold each check's own verdict, so the overall "
                             "verdict has to be resolved rather than copied")
    parser.add_argument("--triage-hide-status", action="store_true",
                        help="withhold execution_status, the only field that "
                             "separates error from pass on a zero-finding run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key

    rows = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("kind") == "triage"]
    labels = {
        eid: label
        for eid, label, _ in load_triage_corpus(
            args.corpus, include_disagreements=args.include_disagreements
        )
    }
    # The scorer is what decides which rows are scoreable -- by default it drops
    # the ones whose observed verdict contradicts the committed baseline -- so
    # the rollout has to be driven from its answer rather than from the file.
    # Read the other way round this samples GPU rollouts for prompts that cannot
    # be graded, and then dies indexing `labels` after they have been paid for.
    # Both drives are filtered, not just triage: a contradicted report is the
    # same report the proposal prompt is built from.
    dropped = sorted(
        r["scenario_id"] for r in rows if f"triage:{r['scenario_id']}" not in labels
    )
    if dropped:
        print(
            f"skipping {len(dropped)} scenario(s) the scorer excludes as "
            f"baseline disagreements ({', '.join(dropped)}); "
            f"pass --include-disagreements to sample them anyway",
            file=sys.stderr,
        )
    rows = [r for r in rows if f"triage:{r['scenario_id']}" in labels]
    if not rows:
        print(f"no scoreable triage rows in {args.corpus}", file=sys.stderr)
        return 2

    candidates = full_candidates() if args.full_candidates else CANDIDATES
    condition = "-".join(
        filter(
            None,
            [
                "no-think" if args.no_think else "as-is",
                None if args.temperature is None else f"t{args.temperature:g}",
                "full-candidates" if args.full_candidates else None,
                "triage-blind" if args.triage_blind else None,
                "triage-hide-status" if args.triage_hide_status else None,
            ],
        )
    )
    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "served_name_on_wire": args.model.split("/", 1)[1]
        if args.model.startswith("openai/")
        else args.model,
        "samples_per_scenario": args.samples,
        "scenarios": len(rows),
        "temperature_injected": args.temperature,
        "seed_base": args.seed,
        "seed_mode": args.seed_mode if args.seed is not None else None,
        "max_tokens_injected": args.max_tokens,
        "condition": condition,
        "node": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "corpus": str(args.corpus),
        "include_disagreements": args.include_disagreements,
        "scenarios_excluded_as_disagreements": dropped,
        "autopsy_categories": sorted(AUTOPSY_CATEGORIES),
        "candidates": candidates,
        "tried": TRIED,
        "offered": [c for c in candidates if c not in TRIED and c != "none"],
        "triage_blind": args.triage_blind,
        "triage_hide_status": args.triage_hide_status,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"driving {len(rows)} scenarios x {args.samples} samples, "
          f"condition {condition}", flush=True)

    recorder = RecordingLiteLLM(
        temperature=args.temperature,
        no_think=args.no_think,
        max_tokens=args.max_tokens,
    )
    recorder.seed_mode = args.seed_mode
    recorder.install()
    try:
        proposals = drive_proposals(
            rows,
            model=args.model,
            samples=args.samples,
            recorder=recorder,
            candidates=candidates,
            seed=args.seed,
        )
        triage = (
            []
            if args.skip_triage
            else drive_triage(
                rows,
                labels,
                model=args.model,
                recorder=recorder,
                blind=args.triage_blind,
                hide_status=args.triage_hide_status,
                seed=args.seed,
            )
        )
    finally:
        recorder.restore()

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    agg = aggregate(proposals, triage)
    agg["proposal"]["template_baselines"] = template_baselines(
        candidates, TRIED, len(rows)
    )
    doc = {
        "schema": "aorta.rl_e2e/0.1",
        "meta": meta,
        "aggregates": agg,
        "proposals": proposals,
        "triage": triage,
        "requests": [c.request for c in recorder.calls[:2]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print_report(agg, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
