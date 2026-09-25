"""LLM proposers for the probe agent loop.

``FakeLLMProposer`` round-robins registered mitigations (offline tests).
``LiteLLMProposer`` calls LiteLLM when ``amd-aorta[agent]`` is installed.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from aorta.agent.prompt_profiles import DEFAULT_PROMPT_PROFILE, PromptProfile, get_prompt_profile

log = logging.getLogger(__name__)

# Why the proposer set ``stop=True`` (drives CLI/report outcome labels).
# ``proposal_unresolved`` is the one the proposer never sets itself: it means
# the loop stopped because every name the model asked for was dropped by the
# candidate filter below, so there was nothing left to run. It exists so that
# stop is separable from a model that genuinely concluded the search.
StopReason = Literal[
    "baseline_pass",
    "exhausted_candidates",
    "agent_requested",
    "proposal_unresolved",
]

#: Reachable only from instrument evidence: a sanitizer that watched two waves
#: collide, a debugger that read a stopped wave, or a tool that could not run at
#: all. Nothing the probe agent does can establish one of these -- it reaches a
#: category by trying mitigations and seeing what changes, which cannot observe
#: a race or distinguish "the sanitizer found nothing" from "the sanitizer never
#: ran".
EVIDENCE_ONLY_CATEGORIES: frozenset[str] = frozenset(
    {
        "gpu_race",
        "numeric_silent",
        "tooling_gap",
    }
)

# The closed autopsy label set, each member carrying the one-line gloss the
# reader is told to route on. Set and glosses are one object deliberately:
# several members are near neighbours (``checkpoint_race`` vs ``gpu_race``,
# ``gpu_race`` vs ``illegal_mem``, ``tooling_gap`` vs ``unknown``) and a bare
# list of names is not enough to choose between them, so a member added without
# a gloss would be a member nobody has been told how to use.
#
# Labels exist to route toward a *diagnostic*, so the granularity follows the
# diagnostic rather than the symptom: a race inside a kernel is localised and
# confirmed with ConSan or waitcheck, where an allocator failure is established
# by re-running under a different allocator.
#
# Every member of ``EVIDENCE_ONLY_CATEGORIES`` has to carry a gloss here too.
# The shared set is derived from this mapping, so a name missing here would
# drop out of the vocabulary entirely rather than merely lose its gloss;
# ``tests/cia/test_probe_vocabulary.py`` pins the subset relation from the other
# side and ``tests/agent/test_autopsy_categories.py`` pins the gloss.
AUTOPSY_CATEGORY_GUIDANCE: Mapping[str, str] = MappingProxyType(
    {
        "rccl_hang": (
            "A collective stopped making progress -- ranks stuck in RCCL/NCCL, "
            "watchdog timeout, no forward progress."
        ),
        "thermal_throttle": (
            "Sustained clock or throughput loss attributable to thermal or power "
            "limiting rather than to the workload itself."
        ),
        "illegal_mem": (
            "Illegal or out-of-bounds device memory access -- HIP/CUDA fault, VM "
            "protection fault, fault in device code. A fault that stops the "
            "workload, as against a race that silently corrupts it."
        ),
        "oom_fragment": (
            "Out of memory or allocator fragmentation, including an OOM kill."
        ),
        "checkpoint_race": (
            "A concurrency defect around checkpoint I/O: a save or load racing "
            "with training, with another rank, or with a filesystem barrier. This "
            "label is about checkpointing, never about code inside a GPU kernel -- "
            "for that use gpu_race."
        ),
        "gpu_race": (
            "An unsynchronised access inside a single GPU kernel: an LDS or global "
            "data race, or a missing wait-count hazard. Localised to one kernel and "
            "named site by site by a sanitizer (ConSan, waitcheck). Evidence-only, "
            "and for the reason the set is split: the race is what an instrument "
            "watched happen, not what a mitigation sweep inferred from a verdict "
            "moving."
        ),
        "launch_error": (
            "The workload failed at or before launch -- bad argv, missing "
            "dependency, early non-zero exit before real work started."
        ),
        "perf_regression": (
            "The workload produces correct results but is slower than its reference."
        ),
        "numeric_silent": (
            "Arithmetic that came out wrong without the workload saying so: a loss "
            "that goes NaN or Inf (``tier4:nan_signature``), values that overflow, "
            "or results drifting past tolerance through precision or accumulation "
            "order. Evidence-only: it takes a repro cell and a clean mitigation "
            "column, or a debugger reading the stopped wave, to tell this from a "
            "workload that merely failed."
        ),
        "tooling_gap": (
            "The instrument could not answer. The sanitizer was rejected before it "
            "instrumented anything, failed to run, or produced no records -- so the "
            "absence of a finding says nothing about the workload. Distinct from "
            "``unknown``, which is evidence that fits no label rather than evidence "
            "never collected."
        ),
        "unknown": (
            "The evidence does not support any label above. The honest answer when "
            "nothing fits -- not a placeholder for a guess."
        ),
    }
)

#: Every category either front door may return. The probe agent reaches a
#: category by trying mitigations; :mod:`aorta.cia` reaches one by reading
#: instrument evidence. They answer different questions and share this
#: vocabulary, so a verdict means the same thing whichever produced it -- and
#: anything reading a report validates against this.
#:
#: Derived from the guidance rather than listed again beside it, so a name and
#: the gloss that distinguishes it from its neighbours cannot drift apart.
AUTOPSY_CATEGORIES: frozenset[str] = frozenset(AUTOPSY_CATEGORY_GUIDANCE)

#: What the probe agent may propose: the shared vocabulary less what only an
#: instrument can establish.
#:
#: Derived rather than written out a second time. Listing it by hand is how the
#: two drift, and the drift is silent -- offering the probe model a category it
#: has no way to reach teaches it to guess one, and the guess validates.
PROBE_CATEGORIES: frozenset[str] = AUTOPSY_CATEGORIES - EVIDENCE_ONLY_CATEGORIES


def format_category_guidance(categories: Iterable[str] | None = None) -> str:
    """Render labels as one ``- name: gloss`` line each.

    Takes the names to render, because the two front doors offer different
    ones: the probe agent is shown only what it can reach, and showing it a
    label it cannot establish is what teaches it to guess. Defaults to the
    whole vocabulary for callers that document the set rather than prompt on it.
    """
    names = AUTOPSY_CATEGORY_GUIDANCE if categories is None else categories
    return "\n".join(
        f"- {name}: {AUTOPSY_CATEGORY_GUIDANCE[name]}" for name in sorted(names)
    )

_BASELINE_CELL = "none-none"


@dataclass(frozen=True)
class AgentStep:
    """Structured output from one agent decision step."""

    category: str
    hypothesis: str
    next_mitigations: list[str]
    confidence: float
    stop: bool
    stop_reason: StopReason | None = None
    #: Names the model proposed that the candidate filter dropped -- not in the
    #: registry, already tried, outside the operator's allowlist, or the
    #: ``none`` baseline, which the filter never offers. Set by the
    #: proposer, never by the model (see from_dict). Without this the loop
    #: cannot say which name failed to resolve, so an affected run can be
    #: detected but not repaired; with it, ``next_mitigations`` plus this list
    #: reconstruct what the model actually asked for.
    unresolved_mitigations: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentStep:
        # Accept only a genuine JSON boolean: bool("false") is True, so a
        # malformed/untrusted "stop": "false" must not prematurely stop the
        # loop. Anything that isn't a real bool defaults to not-stopping.
        stop_raw = raw.get("stop", False)
        stop = stop_raw if isinstance(stop_raw, bool) else False
        reason_raw = raw.get("stop_reason")
        stop_reason: StopReason | None = None
        # The model-claimable reasons only. "proposal_unresolved" is
        # deliberately absent: it is a statement about what the agent did with
        # the model's names, so a model that claimed it would be reporting on
        # machinery it cannot see -- the same reason a claimed "baseline_pass"
        # is downgraded in loop._resolve_stop_outcome unless the probe
        # verdicts agree. unresolved_mitigations is likewise never read from
        # raw: the proposer computes it.
        if stop and isinstance(reason_raw, str) and reason_raw in (
            "baseline_pass",
            "exhausted_candidates",
            "agent_requested",
        ):
            stop_reason = reason_raw  # type: ignore[assignment]
        # Defensive coercion: a real (or buggy) LLM can send a bare string,
        # null, or object for these fields. Only accept a genuine list for
        # next_mitigations -- never list("tf32_off"), which explodes into
        # single characters -- and fall back to a safe confidence instead of
        # raising on a non-numeric value. PolicyValidation re-checks names.
        raw_mitigations = raw.get("next_mitigations")
        next_mitigations = (
            [str(m) for m in raw_mitigations] if isinstance(raw_mitigations, list) else []
        )
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        # Type-aware, not str(): a null/non-string category or hypothesis from
        # the LLM must NOT become the literal "None"/"null" (which fails policy
        # validation for category and pollutes the report for hypothesis).
        category_raw = raw.get("category")
        category = (
            category_raw
            if isinstance(category_raw, str) and category_raw.strip()
            else "unknown"
        )
        hypothesis_raw = raw.get("hypothesis")
        hypothesis = hypothesis_raw if isinstance(hypothesis_raw, str) else ""
        return cls(
            category=category,
            hypothesis=hypothesis,
            next_mitigations=next_mitigations,
            confidence=confidence,
            stop=stop,
            stop_reason=stop_reason,
        )


class LLMProposer(Protocol):
    """Protocol for agent step proposers."""

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
    ) -> AgentStep: ...


# Tokens that say *which tool* found something, and tokens that say the finding
# is inside one kernel. `barrier` and a bare sanitizer name are each ambiguous
# on their own -- see `_is_kernel_race_id`.
_SANITIZER_TOKENS = frozenset({"consan", "waitcheck", "rocjitsu"})
_INTRA_KERNEL_TOKENS = frozenset({"race", "waitcnt", "barrier", "hazard", "lds"})


def _id_word_sequence(detector: str) -> list[str]:
    """One detector ID split into words, in the order it writes them.

    IDs separate words with ":" and "_" (`tier4:python_traceback`,
    `custom:consan_data_race`), so splitting on non-alphanumerics is what makes
    a whole-word test possible. Splitting on the *class* rather than on a list
    of separators is the point: `custom:<raw_id>` is free-form, and
    `tier5_custom.py` requires only a non-empty string, so `custom:nan:signature`
    and `custom:numerics/mismatch` are as legal as the underscored spellings.
    Any run of non-alphanumerics is a separator here, so there is no allowlist
    to leave a spelling out of.
    """
    return [word for word in re.split(r"[^a-z0-9]+", detector.lower()) if word]


def _id_words(detector: str) -> set[str]:
    """One detector ID's words, unordered, for the tests that ask "is it in here"."""
    return set(_id_word_sequence(detector))


def _any_id_writes(detectors: list[str], first: str, second: str) -> bool:
    """Whether *one* detector ID writes `first` immediately before `second`.

    Two properties, and the legs that call this need both.

    Per ID rather than over the joined detector list, for the reason
    `_is_kernel_race_id` is: a signature assembled from two IDs is a finding
    neither of them reports. Today the join cannot in fact produce one, because
    every ID carries a `tierN:` / `custom:` / `meta:` prefix and so can never
    *begin* with the second half of a pair -- but that is an invariant of the
    classifier holding this function up, and deciding per ID does not need it.

    Whole words rather than a substring, which is what the separator question
    exposed. `hip[-_ ]error` also matched `custom:chip_error`,
    `custom:whip_error` and `custom:gpu_chip_error`, labelling a chip error an
    illegal access; generalising the separator class without bounding the words
    keeps every one of those. It is the collision this file has now fixed three
    times -- `race` inside `traceback`, `lds` inside `fields`, `nan` inside
    `canonical` -- so the word split is the mechanism already agreed here rather
    than a fourth one.

    Not quite strictly widening, and the one spelling it gives up is worth
    naming: an id that glues a signature word to something else, `custom:isnan_signature`,
    used to match and now does not. That is the same trade as `\\bnans?\\b` on
    the symptom path, taken for the same reason -- "isnan" is not "nan", the
    same way "chip" is not "hip".
    """
    for detector in detectors:
        words = _id_word_sequence(detector)
        pairs = zip(words, words[1:], strict=False)
        if any(a == first and b == second for a, b in pairs):
            return True
    return False


def _is_kernel_race_id(detector: str) -> bool:
    """Whether *one* detector ID is evidence of an intra-kernel race.

    Judged per ID rather than over the joined string, which matters: two IDs
    that each carry half the evidence -- say `custom:distributed_barrier_timeout`
    beside `custom:consan_tool_failure` -- would otherwise combine into a
    finding neither of them reports.

    Two things are required, on the same ID: a sanitizer named the finding, and
    the evidence is intra-kernel. Neither half is sufficient alone, and
    `custom:*` IDs are why -- they are free-form
    (`probe/classifier/tier5_custom.py` builds `custom:<raw_id>` from whatever
    the recipe named), so every one of these tokens turns up in IDs that are
    not intra-kernel races:

    * `custom:distributed_barrier_timeout` -- a collective that did not arrive.
    * `custom:consan_tool_failure` -- the sanitizer itself falling over.
    * `custom:host_data_race` -- a race, and not one inside a kernel.

    `race` was briefly allowed to stand alone, on the reasoning that it is
    specific enough. The third example is why it is not: "race" says there was
    a race, not where, and `gpu_race` is a claim about where. Requiring the
    sanitizer token costs nothing on the reachable spellings, since a ConSan or
    Waitcheck finding names its tool -- `custom:consan_data_race`,
    `custom:waitcheck_missing_waitcnt`.
    """
    words = _id_words(detector)
    return bool(words & _SANITIZER_TOKENS) and bool(words & _INTRA_KERNEL_TOKENS)


# Where a symptom says the hazard is. `kernel` and `lds` are the location words
# a person writes; the sanitizer names come from the detector path, so the two
# branches share one vocabulary rather than two that drift.
_SYMPTOM_LOCATION_TOKENS = frozenset({"kernel", "lds"}) | _SANITIZER_TOKENS
# What it says the hazard *is*. `barrier` is included here, unlike on the
# detector path, because a symptom naming a barrier has already had to name a
# location to get this far.
_SYMPTOM_HAZARD_TOKENS = frozenset({"race", "hazard", "barrier"})


def _symptom_is_a_kernel_race(low: str) -> bool:
    """Whether free text says a race happened *inside a kernel*.

    Both halves required, as on the detector path: something saying where, and
    something saying what. `gpu_race` is a claim about location, so "race
    condition between the two writer threads" must not reach it.

    Word-bounded, which the first version of this guard was not, and the
    counter-example is one this file should have anticipated: `"lds" in
    "fields"` holds, so "data race between fields" satisfied a location test by
    accident -- the same substring collision as `race` inside `traceback` and
    `nan` inside `canonical`, reintroduced in the fix for the first of them.

    Matching tokens rather than the phrases "data race" / "race condition" also
    widens it correctly. Those two spellings missed "LDS race", "kernel race"
    and a waitcheck hazard report, all of which are what `gpu_race` names.

    `waitcnt` is matched with a trailing boundary only, because the spelling in
    the wild is `s_waitcnt` and `_` is a word character. No English word
    contains the sequence, so the looser side costs nothing here.
    """
    words = set(re.split(r"[^a-z0-9]+", low))
    located = bool(words & _SYMPTOM_LOCATION_TOKENS)
    hazard = bool(words & _SYMPTOM_HAZARD_TOKENS) or re.search(r"waitcnt\b", low)
    return located and bool(hazard)


def _infer_category_from_detectors(detectors: list[str]) -> str:
    joined = " ".join(detectors).lower()
    # Detector IDs separate words with ":" and "_" (`custom:consan_data_race`,
    # `tier4:python_traceback`), so a substring test for a short word like
    # "race" also fires inside "traceback". Every leg naming a word that can be
    # glued inside another is therefore decided per ID off the word split --
    # the kernel-race conjunction by `_is_kernel_race_id`, the three two-word
    # signatures by `_any_id_writes`. What is left on `joined` is the broad
    # legs, whose words are long enough not to collide, and `tier1:exit`, which
    # is a built-in id the classifier spells one way.
    #
    # The legs are ordered most specific test first, and that ordering is
    # load-bearing rather than cosmetic. A leg keyed on a generic word decides
    # every ID that merely contains it, including IDs a later leg would have
    # identified exactly, so an accidental order silently downgrades the
    # function's best answers to its vaguest ones. The order here is: exact
    # multi-word signatures, then the per-ID conjunction, then the broad
    # single-word legs. Adding a leg means placing it by how much evidence it
    # demands, not appending it.
    #
    # `numerics_mismatch` alongside the built-in signature, because a shipping
    # recipe already emits it: `recipes/tokenspeed/tokenspeed-kernel-gemm-smoke.yaml`
    # declares a tier-5 detector `ts_kernel_numerics_mismatch` on
    # `TS_KERNEL_FAIL: numerics_mismatch`, which arrives as
    # `custom:ts_kernel_numerics_mismatch`. An out-of-tolerance kernel result is
    # exactly what `numeric_silent` names, and matching only the built-in
    # left the one detector in the tree that means it falling through to
    # `unknown` -- the gap this PR exists to close, in the category it adds.
    #
    # Three legs here name a two-word signature, and all three go through
    # `_any_id_writes` rather than a regex over `joined`. `[-_ ]` was an
    # allowlist of separators, which is the wrong shape for a field that permits
    # any non-empty string: `custom:nan:signature` and `custom:numerics/mismatch`
    # are legal ids meaning exactly what the underscored spellings mean, and they
    # matched no leg at all. These three are the only literals in the file with a
    # morpheme boundary for a separator to fall on -- every other token here
    # (`checkpoint`, `hang`, `rccl`, `oom`, `illegal`, `memory`, `launch`,
    # `tier2`, `137`, `tier1:exit`) is a single word, an acronym, or a built-in
    # id with a spelling the classifier fixes.
    if _any_id_writes(detectors, "nan", "signature") or _any_id_writes(
        detectors, "numerics", "mismatch"
    ):
        return "numeric_silent"
    if "checkpoint" in joined:
        return "checkpoint_race"
    # "barrier" used to land here, but in this codebase a barrier is a GPU-side
    # object -- ConSan barrier sites, barrier patching -- not a checkpoint
    # barrier, so the old branch routed intra-kernel evidence to a checkpoint-I/O
    # label. Checked after "checkpoint" so a detector naming both still wins for
    # checkpoint_race.
    if any(_is_kernel_race_id(detector) for detector in detectors):
        return "gpu_race"
    if "tier2" in joined or "hang" in joined or "rccl" in joined:
        return "rccl_hang"
    if "oom" in joined or "137" in joined:
        return "oom_fragment"
    # Last of the categories, and the reason is that "memory" is the most
    # generic word any leg keys on: an intra-kernel race is a race on memory, a
    # numerics mismatch is read out of memory, a checkpoint fault touches
    # memory. Every ID this leg should claim -- `tier4:hip_error`,
    # `custom:illegal_address` -- says so in a word no other leg wants, so
    # deciding it last costs nothing and stops it from answering for the three
    # legs above. `custom:consan_global_memory_race` is the case that showed
    # this: a ConSan intra-kernel race report, named exactly as one, which this
    # leg used to label an illegal access because it ran first.
    if (
        _any_id_writes(detectors, "hip", "error")
        or "illegal" in joined
        or "memory" in joined
    ):
        return "illegal_mem"
    if "tier1:exit" in joined or "launch" in joined:
        return "launch_error"
    return "unknown"


def _infer_category_from_symptom(low: str) -> str:
    """The symptom-text dispatcher, mirroring `_infer_category_from_detectors`.

    A separate function because the two really are parallel dispatchers over
    one taxonomy -- the comments below have said so since the precedence fix
    -- and because what each *identifies* is a different question from what
    the probe is allowed to *assert*. `FakeLLMProposer.propose` applies the
    second; these two apply the first, and both can be tested for it.
    """
    # Ordered most specific test first, the same way and for the same
    # reason as `_infer_category_from_detectors`. This path had the
    # identical defect: `memory`, `hang` and `oom` decided any symptom
    # that merely contained them, ahead of every leg that demands more,
    # so "NaN in device memory" was an illegal access and "global memory
    # race in the kernel" never reached the conjunction below. Twenty-four
    # broad/narrow pairs, against twenty-eight on the detector path.
    #
    # The two chains are parallel dispatchers over the same taxonomy, so
    # a fix to one belongs in both -- which is the lesson of this pair
    # rather than an aside: the detector path was reordered first and
    # this one was left, and the mirror had to be reported before it was
    # looked at. There are exactly two such chains; no third dispatcher
    # exists.
    #
    # Whole word, for the same reason the `race` legs want one: "nan" is
    # a substring of ordinary words a GPU symptom is likely to contain
    # -- "canonical", "nanoseconds", "maintenance" -- and this branch
    # only runs once the detectors have already fallen through to
    # `unknown`, which is exactly when a stray match decides the label.
    # `nans?` because the plural is how people write it ("NaNs in the
    # gradients") and a bare `\bnan\b` would miss it.
    if re.search(r"\bnans?\b", low):
        return "numeric_silent"
    # Checkpoint first, and before the race legs. "checkpoint save race
    # condition" is a checkpoint race by name, and with no checkpoint
    # leg here at all it was landing on `gpu_race` -- the same
    # mislabel the detector branch was fixed for, on the other path.
    elif "checkpoint" in low:
        return "checkpoint_race"
    # `gpu_race` is a claim about *where*, so the phrase alone is not
    # enough: the symptom also has to name a kernel or the tool that
    # found it. This mirrors `_is_kernel_race_id`, where an evidence
    # token needs a sanitizer token beside it -- the two paths now
    # demand the same thing, which is why "nondeterministic data race
    # in the reduction kernel" can be ordered ahead of the
    # nondeterminism leg rather than being shadowed by it.
    #
    # A bare host or checkpoint race with no kernel named stays
    # `unknown`, which understates rather than asserting a hazard
    # nothing observed.
    elif _symptom_is_a_kernel_race(low):
        return "gpu_race"
    # A `nondeterminism` leg sat here and is held out pending the
    # vocabulary question on this PR: the name is not in
    # `AUTOPSY_CATEGORY_GUIDANCE`, so emitting it would fail the closed
    # set. The spelling work that leg carried is not lost -- the three
    # id-path signatures keep it, and the argument for the name is that
    # a probe *can* establish nondeterminism by repeating a cell and
    # getting different verdicts, which would make it the one member of
    # this group belonging in `PROBE_CATEGORIES` rather than beside the
    # evidence-only three.
    elif "hang" in low or "nccl" in low or "rccl" in low:
        return "rccl_hang"
    elif "oom" in low:
        return "oom_fragment"
    # Last, for the reason `illegal_mem` is last on the detector path:
    # "memory" is the most generic word either chain keys on, and a NaN,
    # a checkpoint fault and an intra-kernel race are all describable as
    # being about memory.
    elif "memory" in low or "illegal" in low:
        return "illegal_mem"
    return "unknown"


class FakeLLMProposer:
    """Deterministic proposer: heuristic category + round-robin mitigations."""

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
    ) -> AgentStep:
        last = cell_summaries[-1] if cell_summaries else {}
        detectors = list(last.get("failure_detectors_fired") or [])
        category = _infer_category_from_detectors(detectors)
        if symptom and category == "unknown":
            category = _infer_category_from_symptom(symptom.lower())
        # The two chains above label *evidence*, and evidence can say
        # `gpu_race` or `numeric_silent`. A probe step may not: `AgentPolicy`
        # validates against `PROBE_CATEGORIES`, so returning one of those here
        # would raise `PolicyViolation` in `run_agent_loop` rather than
        # mislabel anything. Downgrading to `unknown` is the honest move and
        # the one the shared vocabulary already prescribes -- "evidence that
        # fits no label I am allowed to assert" is exactly `unknown`, and it is
        # what `cia.autopsy.router.coerce_category` does with a name it cannot
        # place.
        #
        # This loses no accuracy that the probe was entitled to: before the
        # taxonomy widened, a ConSan race report reached `checkpoint_race` or
        # `illegal_mem` through the broad legs, so the change here is a wrong
        # label becoming an honest one, not a right one becoming vague. The
        # corpus keeps the precise label, because
        # `examples/rl/corpus/scenario_labels.json` is ground truth read from
        # instrument evidence and validates against the whole vocabulary.
        if category in EVIDENCE_ONLY_CATEGORIES:
            category = "unknown"

        # Baseline pass wins even when the allowlist has no further mitigations.
        for summary in cell_summaries:
            if summary.get("cell_name") == _BASELINE_CELL and summary.get("verdict") == "pass":
                return AgentStep(
                    category="unknown",
                    hypothesis="Baseline cell passed; no mitigation search needed.",
                    next_mitigations=[],
                    confidence=1.0,
                    stop=True,
                    stop_reason="baseline_pass",
                )

        remaining = [c for c in candidates if c not in tried and c != "none"]
        if not remaining:
            return AgentStep(
                category=category,
                hypothesis="No remaining registered mitigations to try.",
                next_mitigations=[],
                confidence=0.9,
                stop=True,
                stop_reason="exhausted_candidates",
            )

        next_m = remaining[0]
        return AgentStep(
            category=category,
            hypothesis=(
                f"Try mitigation {next_m!r} based on detectors {detectors!r}."
                + (f" Symptom: {symptom}" if symptom else "")
            ),
            next_mitigations=[next_m],
            confidence=0.5,
            stop=False,
        )


def _remaining_candidates(candidates: list[str], tried: list[str]) -> list[str]:
    return [c for c in candidates if c not in tried and c != "none"]


def _exhausted_step() -> AgentStep:
    """Stop without spending tokens, and without needing a backend installed."""
    return AgentStep(
        category="unknown",
        hypothesis="No remaining registered mitigations to try.",
        next_mitigations=[],
        confidence=0.9,
        stop=True,
        stop_reason="exhausted_candidates",
    )


def _safe_stop(hypothesis: str) -> AgentStep:
    """Turn an unusable model response into a stop the loop can still report on."""
    return AgentStep(
        category="unknown",
        hypothesis=hypothesis,
        next_mitigations=[],
        confidence=0.0,
        stop=True,
        stop_reason="agent_requested",
    )


def _build_prompt(
    symptom: str | None,
    cell_summaries: list[dict[str, Any]],
    remaining: list[str],
    tried: list[str],
) -> tuple[str, str]:
    """The system and user messages, shared by every real proposer.

    One definition so the two backends cannot drift into asking for different
    JSON, which is the failure a shared provider layer is supposed to prevent.
    """
    system = (
        "You are an AORTA probe agent. Propose ONLY registered mitigation "
        "names from the candidate list. Never propose shell commands or argv. "
        "Return strict JSON with keys: category, hypothesis, next_mitigations "
        "(list of strings), confidence (0-1), stop (bool). "
        "category must be exactly one of the labels below. Several are near "
        "neighbours, so the gloss -- not the name -- is what tells them apart; "
        "read it before choosing.\n"
        f"{format_category_guidance(PROBE_CATEGORIES)}"
    )
    user = json.dumps(
        {
            "symptom": symptom,
            "cell_summaries": cell_summaries,
            "candidates": remaining,
            "already_tried": tried,
        },
        indent=2,
    )
    return system, user


def _profile_prompt(
    profile: PromptProfile,
    symptom: str | None,
    cell_summaries: list[dict[str, Any]],
    remaining: list[str],
    tried: list[str],
) -> tuple[str, str]:
    """The system and user messages ``profile`` sends for this loop state."""
    if profile.build is None:
        return _build_prompt(symptom, cell_summaries, remaining, tried)
    return profile.build(cell_summaries, remaining)


def _strip_code_fence(content: str) -> str:
    """Unwrap a ```json fenced block.

    Needed on the chat-provider path, which has no ``response_format`` knob to
    ask for a bare object; a fenced reply is otherwise a parse failure and a
    wasted iteration.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


#: Qwen3's reasoning delimiters; DeepSeek-R1 uses the same pair.
_REASONING_OPEN = "<think>"
_REASONING_CLOSE = "</think>"


def _strip_reasoning(content: str) -> str:
    """Drop the reasoning block a thinking model writes before its answer.

    A Qwen3-family model served without a server-side reasoning parser replies
    ``<think>...</think>`` and then the JSON. Qwen3.8's chat template opens the
    block in the *prompt*, so its replies carry only the closing tag. Left in,
    every such reply fails to parse and the loop stops after its first step --
    measured on every Qwen3 model tried, on vLLM and TokenSpeed alike, with no
    scenario converging.

    Requiring the server flag (``--reasoning-parser qwen3``) instead was
    rejected: neither engine enables it by default, so every default deployment
    would stay broken, and nothing on this side would say why. It remains the
    recommended setup (``docs/chat/providers.md``); this covers its absence.
    With the flag on, the block arrives in ``reasoning_content``, which neither
    proposer reads, so a generation cut off mid-reasoning reaches
    :func:`_step_from_content` as empty content and stops there. Asking the
    server not to think (``chat_template_kwargs``) also yields parseable
    replies, but the proposals get markedly worse, and the parameter is not
    portable across providers.

    Narrow on purpose, because each widening reads an answer out of text that
    is not one:

    * Only a *terminated* block. Qwen3 drafts the object mid-thought, so an
      unterminated ``<think>`` -- a generation that ended before its answer --
      often contains JSON; searching it for the first ``{`` would promote a
      discarded draft to a decision. That raises instead.
    * Only a *leading* block, or the headless form. A reply that begins as an
      answer (``{`` or a fence) is left alone, so a ``</think>`` quoted in a
      hypothesis is data. Removing ``<think>...</think>`` wherever it occurs
      would edit string values.
    * Split at the *first* closing tag, so an answer that quotes one survives.
      Reasoning that spelled the tag out as text would leave prose in front of
      the answer, which fails to parse -- the safe direction.

    A headless reply cut off before its closing tag is indistinguishable from
    prose and fails to parse exactly as prose does. A terminated block with
    nothing after it raises with a message saying so, rather than surfacing as
    a JSON error at column 1 that reads like malformed output.

    Raises:
        ValueError: The reply is reasoning with no answer after it.
    """
    text = content.strip()
    if text.startswith(_REASONING_OPEN):
        end = text.find(_REASONING_CLOSE, len(_REASONING_OPEN))
        if end == -1:
            raise ValueError("reasoning block never closed; the generation ended before an answer")
        answer = text[end + len(_REASONING_CLOSE) :]
    elif _REASONING_CLOSE in text and not text.startswith(("{", "```")):
        answer = text.split(_REASONING_CLOSE, 1)[1]
    else:
        return text
    answer = answer.strip()
    if not answer:
        raise ValueError("reply was reasoning with no answer after it")
    return answer


def _answer_text(content: str) -> str:
    """The part of a reply that should be JSON: reasoning dropped, fence unwrapped.

    Anything that claims to know what the proposer would make of a reply
    should import this rather than re-derive it, or the claim drifts from the
    proposer. ``examples/rl/proposal_reward.py`` still calls
    :func:`_strip_code_fence`, so its consumer outcome for a
    reasoning-prefixed reply is ``silent_stop`` where the proposer accepts
    it; moving it here changes reward scoring and is left to its own change.

    Raises:
        ValueError: The reply is reasoning with no answer after it.
    """
    return _strip_code_fence(_strip_reasoning(content))


def _step_from_content(content: str | None, remaining: list[str]) -> AgentStep:
    """Parse a model reply into an :class:`AgentStep`, failing safe.

    Providers return malformed or partial JSON, a non-object, reasoning with no
    answer, or nothing at all even when asked for strict JSON. Every one of
    those becomes a stop rather than an exception, so the loop still writes a
    report.
    """
    if not content or not content.strip():
        return _safe_stop("Empty LLM response")
    try:
        raw = json.loads(_answer_text(content))
        if not isinstance(raw, dict):
            raise TypeError(f"expected a JSON object, got {type(raw).__name__}")
        step = AgentStep.from_dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _safe_stop(f"LLM returned unparseable response: {exc}")

    # Never let the model widen its own allowlist: PolicyValidation re-checks,
    # but a name outside `remaining` is a mitigation already tried or never
    # registered, and running it is not the agent's call.
    #
    # Keep what was dropped. This filter runs BEFORE
    # AgentPolicy.validate_step, so it empties the very list validation would
    # have rejected, and the loop then reads the empty list as a decision to
    # stop (aorta#449). Recording the names is what separates "the model
    # concluded" from "the agent could not resolve what the model asked for",
    # and it is the only record of the dropped half when some names survive
    # and the loop carries on. Not de-duplicated: this is the audit trail of
    # what the model actually emitted, so it stays faithful to the reply.
    filtered = [m for m in step.next_mitigations if m in remaining]
    unresolved = [m for m in step.next_mitigations if m not in remaining]
    if unresolved:
        log.warning(
            "proposer named %d mitigation(s) that do not resolve against the "
            "remaining candidates and were dropped: %s (remaining: %s)",
            len(unresolved),
            sorted(set(unresolved)),
            sorted(remaining),
        )
    stop_reason = step.stop_reason
    if step.stop and stop_reason is None:
        stop_reason = "agent_requested"
    return AgentStep(
        category=step.category,
        hypothesis=step.hypothesis,
        next_mitigations=filtered,
        confidence=step.confidence,
        stop=step.stop,
        stop_reason=stop_reason,
        unresolved_mitigations=unresolved,
    )


class LiteLLMProposer:
    """LiteLLM-backed proposer (requires ``pip install 'amd-aorta[agent]'``).

    Retained after Phase 5b as the path that works on an ``[agent]``-only
    install, where the chat provider layer is not present. ``--llm-backend
    litellm`` has shipped and must keep working without the chat extra, so
    :func:`make_proposer` prefers the shared layer and falls back to this.
    """

    def __init__(
        self, *, model: str = "gpt-4o-mini", prompt_profile: str = DEFAULT_PROMPT_PROFILE
    ) -> None:
        self._model = model
        self._profile = get_prompt_profile(prompt_profile)

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
        if not remaining:
            return _exhausted_step()

        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLM is required for --llm-backend=litellm. "
                "Install it with either:\n"
                "  pip install litellm\n"
                "  pip install -e '.[agent]'   # from the aorta repo root (editable + extra)\n"
                "If pip says the 'agent' extra does not exist, your installed amd-aorta "
                "distribution is stale — reinstall from this repo with -e '.[agent]'."
            ) from exc

        system, user = _profile_prompt(self._profile, symptom, cell_summaries, remaining, tried)
        request: dict[str, Any] = {}
        if self._profile.json_mode:
            request["response_format"] = {"type": "json_object"}
        extra_body = self._profile.extra_body()
        if extra_body:
            request["extra_body"] = extra_body
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **request,
        )
        return _step_from_content(response.choices[0].message.content, remaining)


#: Backends resolved through the shared chat provider layer (Decision 7a). The
#: names are the chat factory's own, so ``--llm-backend`` and
#: ``--llm-provider`` mean the same thing on both front doors.
CHAT_PROVIDER_BACKENDS: frozenset[str] = frozenset({"litellm", "openai", "vllm"})

_CHAT_EXTRA_HINT = (
    "--llm-backend={backend} is configured through the shared chat provider "
    "layer, which needs the chat-cli extra.\n"
    "Install it with:  pip install 'amd-aorta[chat-cli]'\n"
    "The provider itself (endpoint, model, API key, gateway headers) is then "
    "configured once, in ~/.config/aorta/chat.toml or AORTA_CHAT_* -- the same "
    "settings 'aorta chat' uses.\n"
    "(missing: {missing})"
)


class ChatProviderProposer:
    """Proposer on the shared chat provider layer (locked Decision 7a).

    The point is single configuration: ``vllm`` / ``openai`` / ``litellm`` are
    set up once, in the chat profile, and both front doors read it. Before this,
    ``aorta agent`` reached litellm directly and knew nothing about the endpoint,
    gateway headers or auth scheme that ``aorta chat`` had already been told.

    Unlike :class:`LiteLLMProposer` there is no ``response_format`` to lean on
    -- the layer returns a LangChain chat model, not a raw completion call -- so
    the reply is fence- and reasoning-tolerant and every parse failure still
    fails safe.
    """

    def __init__(
        self,
        provider: str,
        *,
        model: str | None = None,
        prompt_profile: str = DEFAULT_PROMPT_PROFILE,
    ) -> None:
        self._provider = provider
        self._model = model
        self._profile = get_prompt_profile(prompt_profile)

    def _chat_model(self) -> Any:
        """Resolve the configured chat model, or explain which extra is missing.

        Imported inside the call, not at module scope: this is the one place
        core reaches into ``aorta.chat``, and deferring it keeps ``import
        aorta.agent`` (and so ``aorta --help``) free of langchain.
        """
        try:
            from aorta.chat.config import apply_cli_overrides
            from aorta.chat.inference.providers.factory import get_backend
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            if missing.startswith("aorta.chat"):
                # A broken aorta.chat submodule is a real bug, not a missing
                # extra; advising an install would bury it. Same rule as
                # cli/chat.py's _load and cli/bench.py.
                raise
            raise ImportError(
                _CHAT_EXTRA_HINT.format(backend=self._provider, missing=missing)
            ) from exc

        # Reuse the chat layer's own precedence, which also knows whether a
        # model name belongs to the local or the remote field.
        apply_cli_overrides(provider=self._provider, model=self._model)
        return get_backend(self._provider).get_chat_model(temperature=0.0, streaming=False)

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
        # Checked before the import, so an exhausted loop neither spends tokens
        # nor requires the extra to be installed. Mirrors both siblings.
        if not remaining:
            return _exhausted_step()

        system, user = _profile_prompt(self._profile, symptom, cell_summaries, remaining, tried)
        # Role tuples rather than langchain message classes: one fewer import on
        # a path that only needs to say who said what.
        messages = [("system", system), ("human", user)]
        extra_body = self._profile.extra_body()
        chat_model = self._chat_model()
        if extra_body:
            response = chat_model.invoke(messages, extra_body=extra_body)
        else:
            response = chat_model.invoke(messages)
        return _step_from_content(getattr(response, "content", None), remaining)


def _chat_layer_available() -> bool:
    """Whether the chat provider layer can be imported at all.

    ``find_spec`` rather than an import: this is asked on the ``litellm`` path
    to choose between two working implementations, and it must not pull in
    langchain for an install that is going to use the direct path anyway.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("langchain_core") is not None
    except (ImportError, ValueError):
        return False


def make_proposer(
    backend: str,
    *,
    model: str | None = None,
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
) -> LLMProposer:
    """Build the proposer for ``--llm-backend``.

    Phase 5b (locked Decision 7a) put ``vllm`` / ``openai`` / ``litellm`` onto
    the shared chat provider layer, so a provider is configured once and both
    front doors read that configuration.

    ``litellm`` is the one backend with a shipped contract to keep: it has
    worked on an ``[agent]``-only install since before ``aorta.chat`` existed.
    So it prefers the shared layer and falls back to the direct
    :class:`LiteLLMProposer` when the chat extra is absent, rather than
    breaking an install that used to work. ``vllm`` and ``openai`` are new, have
    no such history, and say plainly which extra they need.

    ``fake`` stays the default and stays fully offline -- it imports nothing and
    reaches nothing, which is what makes the test suite and ``--dry-run``
    hermetic.

    ``prompt_profile`` (see :mod:`aorta.agent.prompt_profiles`) is checked
    before the backend, so a misspelt profile fails whichever backend was
    asked for. ``fake`` sends no prompt, so it refuses any profile but
    ``default`` rather than accept one and ignore it.
    """
    get_prompt_profile(prompt_profile)
    if backend == "fake":
        if prompt_profile != DEFAULT_PROMPT_PROFILE:
            raise ValueError(
                f"prompt profile {prompt_profile!r} needs a real model: "
                "--llm-backend=fake builds no prompt, so it would be ignored"
            )
        return FakeLLMProposer()
    if backend == "litellm" and not _chat_layer_available():
        return LiteLLMProposer(model=model or "gpt-4o-mini", prompt_profile=prompt_profile)
    if backend in CHAT_PROVIDER_BACKENDS:
        return ChatProviderProposer(backend, model=model, prompt_profile=prompt_profile)
    raise ValueError(
        f"unknown agent LLM backend: {backend!r} "
        f"(expected one of {', '.join(sorted({'fake', *CHAT_PROVIDER_BACKENDS}))})"
    )


__all__ = [
    "AUTOPSY_CATEGORIES",
    "AUTOPSY_CATEGORY_GUIDANCE",
    "EVIDENCE_ONLY_CATEGORIES",
    "PROBE_CATEGORIES",
    "CHAT_PROVIDER_BACKENDS",
    "AgentStep",
    "ChatProviderProposer",
    "FakeLLMProposer",
    "LLMProposer",
    "LiteLLMProposer",
    "StopReason",
    "format_category_guidance",
    "make_proposer",
]
