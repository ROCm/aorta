"""The Laya seam: typed questions in, calibrated probabilities out.

Six decisions in this codebase share one shape. An LLM emits prose, the prose is
string-parsed into a verdict, and a self-reported ``confidence`` float is
compared against a hand-picked threshold. That float is not a probability --
nothing trained it to be one -- so the constant sitting beside it was fitted
against a number that does not mean what its name says. Autopsy's prompt is the
clearest case: ``src/aorta/cia/autopsy/router.py`` asks the model in English for
``~0.62`` and ``~0.55``, and ``autopsy/probe.py`` escalates below ``0.85``.

Laya answers typed questions instead. A ``noul`` is a yes/no whose answer is
p(yes); a ``choice`` is one of N named options with a probability each. Both are
trained against strictly proper scoring rules, both come out of one forward pass,
and neither generates a token -- so there is nothing to parse and no path by
which an answer can name an option that was not offered. That last property is
the security argument for the log finder as much as the cost argument.

This module is the seam and not the integration. It exists so Watch's
clean-gate, the probe agent's proposer and the log finder can each be written
against one contract and tested against a fake, before any weights exist on the
machine that runs the tests.

**The shape is deliberately** :mod:`aorta.agent.llm`'s. That module declares
``LLMProposer`` as a Protocol, ships ``FakeLLMProposer`` as the offline
reference implementation that the default backend and every test rely on, and
resolves a backend name in one factory. A ``LayaProposer`` is going to arrive as
a third implementation of ``LLMProposer`` built on top of what is here, so two
seams with two different shapes would be one shape too many to hold in a
reader's head.

**``score`` is deliberately absent.** Laya has a third primitive that returns a
level on an ordinal scale, and it is the weakest of the three on the model
card's own numbers. Modelling failure severity with it would put the least
trustworthy of the three answers in the most consequential place.

**The library surface below was read out of the published 0.3.5 wheel**, not out
of the README or the model card: ``laya.load(repo, device=, token=, subfolder=)``,
``Agent.predict`` as an alias of ``Agent.system_one``, and the answer payload
:func:`_as_answer` parses. :data:`VERIFIED_LAYA_VERSION` records which release
that was, and ``pyproject.toml`` floors the extra at the same one. 0.3.3, 0.3.4
and 0.3.5 shipped on three consecutive days and moved both the Python floor and
the transformers floor inside that window, so a project moving at that rate does
not get the benefit of the doubt about its call signatures: the floor and the
verification are kept the same number on purpose.

**Nothing here imports torch or ``laya`` at module scope.** ``[chat-cli]`` is
CI-gated against resolving torch (``.github/workflows/nightly.yml`` and
``release.yml`` fail the build), and ``_HEAVY_PREFIXES`` in
``tests/cli/test_chat_boundaries.py`` asserts that ``import aorta.cli`` pulls in
neither torch nor onnxruntime, while ``_CHAT_ONLY_PREFIXES`` asserts the same of
the agents. The loader therefore lives inside
:meth:`LayaAgentPredictor._agent`, and importing this module costs an import of
``dataclasses``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Union

#: The option-count buckets the library fits one temperature per, in the order it
#: tests them. Mirrors ``laya.common.temp_bucket``, whose keys look like
#: ``choice:11+`` and ``noul:2``.
#:
#: Duplicated rather than imported, and this is the one place in this module where
#: that is the right call: ``laya.common`` imports torch at its own module scope,
#: so reading the function would put torch on the import path of every caller that
#: only wants to know which bucket a question falls in -- which is every caller
#: that wants the answer *before* loading a checkpoint, and offline is exactly
#: where the answer is most useful. :func:`bucket_for` is a pure function of a
#: question, and ``tests/cia/test_laya_predictor.py`` pins it against the boundary
#: values so a change upstream shows up as a failure rather than as a caveat that
#: silently stops firing.
_BUCKET_CEILINGS: tuple[tuple[int, str], ...] = ((2, "2"), (5, "3-5"), (10, "6-10"))

#: What a wider option set than every ceiling above is called.
_WIDEST_BUCKET = "11+"

#: The Laya release this module's translation was verified against, by reading
#: the published wheel. Kept equal to the floor in ``pyproject.toml``'s ``[laya]``
#: extra, and asserted so by ``tests/cia/test_laya_predictor.py``.
#:
#: Not a runtime check. Refusing to run against a newer release would be worse
#: than useless -- this is a floor, not a pin, and the point of a floor is that
#: later releases are expected to work. What the constant buys is that the next
#: person to widen the floor has somewhere to record that they re-read the wheel.
VERIFIED_LAYA_VERSION = "0.3.5"

#: Checkpoint name -> (repository, subfolder). ``laya.load`` takes the two
#: separately because one repository bundles several checkpoints.
#:
#: Both multilingual checkpoints are deliberately absent. The logs are English
#: and the assembly is GCN, so script detection buys nothing, and pinning a
#: single checkpoint avoids the reload hazard that comes with the library's
#: default of keeping one model resident: alternating checkpoints rebuilds one
#: on every request, which the card measures at a 7.4 s median on CPU.
#:
#: These are names, not a pinned identity. Decision 22 owns pinning the weights
#: by digest; until it is written, :meth:`LayaAgentPredictor.model_id` reports
#: whatever was actually loaded so that a report cannot silently attribute a
#: verdict to the wrong weights.
CHECKPOINTS: Mapping[str, tuple[str, str | None]] = {
    "laya": ("convaiinnovations/laya", None),
    "laya-typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}

#: The name :func:`make_predictor` resolves when nothing is asked for. ``fake``
#: rather than a checkpoint, for the reason ``aorta.agent.llm`` defaults to
#: ``fake``: the default has to work on a base install with no network.
DEFAULT_BACKEND = "fake"


class LayaUnavailable(RuntimeError):
    """A real predictor was asked for and the machine cannot provide one.

    Raised rather than degraded. A predictor that silently answers 0.5 to
    everything would let a clean-gate pass its own tests, ship, and then gate
    on noise -- and the failure would look like a model that is merely bad
    rather than a model that is not there.
    """


# ── questions ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is p(yes), not a yes/no.

    *when_true* and *when_false* describe the two sides when the question alone
    is ambiguous. They are optional because most of ours are not: "is this log
    delta healthy" needs no gloss, where "is this file worth watching" benefits
    from one.
    """

    question: str
    when_true: str = ""
    when_false: str = ""

    @property
    def options(self) -> tuple[str, str]:
        """The names an answer is a distribution over, in the model's own order.

        A property rather than a field, so it cannot be overridden per instance:
        the underlying model lays a noul out as [false, true] and reads p(yes)
        off the second position, so a caller reordering these would invert every
        answer. The scoring code reads ``options`` off either question type, and
        this is what lets it.
        """
        return ("false", "true")


@dataclass(frozen=True)
class Choice:
    """One of *options*, with a probability each.

    The option set travels with the question rather than being configured once,
    because two of the three planned callers vary it per request: the proposer
    offers the candidate mitigations that are actually left to try, and the log
    finder offers the files that are actually in the directory. An option set
    fixed at construction would put back the thing this replaces -- a model
    free to name something that is not on offer.

    *criteria* glosses individual options, in the order given; an option with no
    entry is offered by name alone.

    **A wide option set costs accuracy as well as tokens.** Every option shares
    one ``head_max_len`` budget, so a long list leaves few tokens per label, and
    the shipped calibration is at its worst in the eleven-or-more bucket -- which
    is where the probe agent's 21 registered mitigations land. Laya 0.3.5 added an
    opt-in ``shortlist`` module for exactly this: embed the state and the options,
    keep the top k, ask about those. Not used here, because whether to narrow the
    candidate list before asking is Track B's decision and not this seam's, but it
    is the upstream answer if the measurement says the wide choice is the problem.
    """

    question: str
    options: tuple[str, ...]
    criteria: tuple[tuple[str, str], ...] = ()


Question = Union[Noul, Choice]


# ── answers ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoulAnswer:
    """p(yes) for one :class:`Noul`.

    There is no ``value`` property, and that omission is the point. Watch's
    existing ``confidence_threshold: 0.70`` and the ``clean_threshold`` a Laya
    tier needs gate opposite directions, so a default boolean read here would be
    an inversion bug waiting for whichever caller forgot which way round it
    went. :meth:`at` makes the caller name the threshold it means.
    """

    probability: float

    def at(self, threshold: float) -> bool:
        """Whether p(yes) reaches *threshold*.

        ``>=`` rather than ``>``, matching ``should_alert`` in
        ``aorta/cia/watch/poll.py`` so that "at the threshold" means the same
        thing on both sides of the poll loop.
        """
        return self.probability >= threshold


@dataclass(frozen=True)
class ChoiceAnswer:
    """A distribution over one :class:`Choice`'s options, in the order offered.

    The whole distribution rather than the argmax, because two of the planned
    callers need the runners-up: the log finder ranks a directory listing by
    probability and takes ``max_files``, and a shadow-mode comparison is only
    readable if it can say how close the second option was.

    Laya's own answer payload also carries a ``confidence`` field, and this
    class deliberately does not surface it. That field is normalised Shannon
    entropy over the distribution, which measures how peaked the answer is
    rather than how likely the top option is to be right -- a uniform-ish
    two-option answer and a uniform-ish ten-option answer score differently
    while being equally uninformative. :attr:`probability` is the number a
    threshold and a Brier score both want.
    """

    probabilities: tuple[tuple[str, float], ...]

    @property
    def option(self) -> str:
        """The most probable option. Ties break toward the order offered."""
        if not self.probabilities:
            raise ValueError("a choice answer with no options has no argmax")
        return max(self.probabilities, key=lambda pair: pair[1])[0]

    @property
    def probability(self) -> float:
        """p(:attr:`option`) -- the calibrated confidence in the answer given."""
        if not self.probabilities:
            return 0.0
        return max(probability for _, probability in self.probabilities)

    def probability_of(self, option: str) -> float:
        """p(*option*), or 0.0 for an option that was never offered."""
        for name, probability in self.probabilities:
            if name == option:
                return probability
        return 0.0

    def ranked(self) -> list[tuple[str, float]]:
        """Every option, best first. What a ranker consumes."""
        return sorted(self.probabilities, key=lambda pair: -pair[1])


Answer = Union[NoulAnswer, ChoiceAnswer]


# ── which temperature answered the question ────────────────────────────────


def bucket_for(question: Question) -> str:
    """The library's calibration bucket for *question*: e.g. ``choice:11+``.

    One definition, because three callers need it and two of them fall on
    opposite sides of the interesting boundary. The probe agent's candidate
    choice is 21 options wide, which is ``choice:11+``; Autopsy's category choice
    is over ``AUTOPSY_CATEGORIES``, which currently holds eleven members and is
    therefore *also* ``choice:11+``; and the probe agent's category choice is
    eight, which is not. Each of them re-deriving "am I in the bad bucket" from an
    option count is how one of them ends up deriving it differently, or stops
    deriving it after the set it reads widens by three.

    Pure, and deliberately usable before any weights exist. Which bucket a
    question falls in is a property of the question; whether that bucket's
    shipped temperature was trustworthy is a property of the checkpoint, and that
    is :meth:`LayaPredictor.calibration`'s job. Conflating the two is what makes a
    disclosure depend on a model being loaded, and a disclosure that needs the
    model is one the report read afterwards will not carry.

    Not the same thing as :func:`aorta.laya.eval.calibration_key`, which keys on
    the *exact* option count because our own refit has no reason to group a
    six-way with a ten-way. This one has to reproduce the library's coarser
    grouping, because it answers which shipped number was applied.
    """
    kind = "noul" if isinstance(question, Noul) else "choice"
    width = len(question.options)
    for ceiling, name in _BUCKET_CEILINGS:
        if width <= ceiling:
            return f"{kind}:{name}"
    return f"{kind}:{_WIDEST_BUCKET}"


@dataclass(frozen=True)
class ClampedBucket:
    """One bucket whose shipped temperature the library refused to apply as given."""

    bucket: str
    #: What the checkpoint's own table asked for.
    shipped: float
    #: What was applied instead.
    applied: float


@dataclass(frozen=True)
class Calibration:
    """Which temperatures a loaded checkpoint actually applied, and which it could not.

    Decision 22 asks that a report record the checkpoint *and* the temperature fit
    that produced a probability, because a probability is a function of both. The
    checkpoint half was already reportable through ``model_id()``; this is the
    other half, and until it existed there was nothing real to record for it.

    A clamped bucket is the case that matters. Laya 0.3.5 refuses a fitted
    temperature outside [0.5, 5.0] and substitutes the nearest bound, because the
    published table's ``choice:11+`` entry is 0.1006 -- below 1, so it *sharpens*
    logits rather than softening them, publishing a 0.24 top probability as 0.99.
    The clamp stops the sharpening; it does not make the bucket calibrated. So a
    caller thresholding a probability from a clamped bucket needs to know, and
    the library's own channel for saying so is a ``RuntimeWarning`` raised once at
    load: it is emitted from inside whichever call first touched the weights, it
    reaches no artifact anyone reads afterwards, and ``PYTHONWARNINGS=ignore`` --
    which is what a CI wrapper or a job launcher sets -- deletes it outright. This
    object is the channel; the warning is the backstop.

    *unknown_because* is non-empty when the clamp set could not be established at
    all, and carries the reason. Three cases reach it: a predictor with no weights
    behind it, a checkpoint that has not been loaded yet, and a ``laya`` older than
    0.3.5, which applies the shipped table verbatim and does not keep a copy to
    compare against. All three are reported as unknown rather than as "nothing was
    clamped", because on the third one the answer is not merely unknown -- it is
    that the sharpening this class exists to disclose was silently applied.
    """

    model_id: str
    clamped: tuple[ClampedBucket, ...] = ()
    #: Every bucket the checkpoint has an entry for, as applied. Recorded whole
    #: rather than only where it differs, because Decision 22 asks for the fit and
    #: "the buckets that survived" is most of it.
    applied: tuple[tuple[str, float], ...] = ()
    unknown_because: str = ""

    @property
    def known(self) -> bool:
        return not self.unknown_because

    def is_clamped(self, bucket: str) -> bool | None:
        """Whether *bucket* was clamped, or None when that could not be established.

        Three-valued on purpose. A caller that collapses None to False is claiming
        a checkpoint was fine on the strength of not having asked it, which is the
        specific mistake :attr:`unknown_because` exists to prevent. Most callers
        want :meth:`caveat`, which handles all three.
        """
        if not self.known:
            return None
        return any(entry.bucket == bucket for entry in self.clamped)

    def caveat(self, question: Question) -> str:
        """What has to be said about a probability answering *question*, or "".

        Returns a fragment meant to be appended to a line a human will read --
        a hypothesis, a rationale, an event's assessment -- because that is where
        the disclosure has to end up to survive being copied into a ticket.
        """
        bucket = bucket_for(question)
        if not self.known:
            return f"; CALIBRATION UNKNOWN: {self.unknown_because}"
        for entry in self.clamped:
            if entry.bucket == bucket:
                return (
                    f"; NOT CALIBRATED: {self.model_id} ships temperature "
                    f"{entry.shipped:.4g} for {bucket}, which laya clamped to "
                    f"{entry.applied:.4g}"
                )
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "known": self.known,
            "unknown_because": self.unknown_because,
            "clamped": [
                {
                    "bucket": entry.bucket,
                    "shipped": entry.shipped,
                    "applied": entry.applied,
                }
                for entry in self.clamped
            ],
            "applied": {bucket: value for bucket, value in self.applied},
        }


#: What a predictor with no weights behind it reports.
_NO_WEIGHTS = (
    "this predictor has no checkpoint behind it, so there is no temperature table "
    "to report"
)


# ── the seam ───────────────────────────────────────────────────────────────


class LayaPredictor(Protocol):
    """Protocol for a typed-decision predictor.

    The batching is in the contract rather than bolted on beside it. One call
    carries N *states* and M *questions*, and the underlying model answers every
    question about one state in a single forward pass -- so M questions about one
    state costs one pass, and N states costs N passes however few questions each
    carries. That asymmetry is the thing a caller has to know: it is why a
    question per tool, or per candidate mitigation, is free and why reranking
    thirty retrieved chunks cannot be.

    Three methods, having started at two. :meth:`calibration` was added because
    two callers on opposite sides of the ``choice:11+`` boundary had each begun
    re-deriving "was this probability trustworthy" from their own option counts.
    That works and it duplicates a fact the loaded checkpoint already knows, and a
    third caller would have derived it differently or not at all. Decision 22 also
    asks a report to record the temperature fit, and before this there was nothing
    real to record.
    """

    def ask(
        self, states: Sequence[str], questions: Sequence[Question]
    ) -> list[list[Answer]]:
        """One list of answers per state, aligned with *questions*."""
        ...

    def model_id(self) -> str:
        """What produced these answers, for a report to record.

        Named after ``EmbeddingProvider.model_id()`` on the retrieval side,
        which exists for the same reason: a before/after comparison that does
        not name the thing that changed is not a comparison. A weights swap
        changes verdicts silently.
        """
        ...

    def calibration(self) -> Calibration:
        """Which temperature buckets this predictor applied, and which were clamped.

        The other half of what a probability is a function of. Cheap and
        side-effect-free: an implementation that has not loaded its weights yet
        reports that as :attr:`Calibration.unknown_because` rather than loading
        them, so this never turns an accessor into a seven-second checkpoint build.
        """
        ...


def ask_one(
    predictor: LayaPredictor, state: str, questions: Sequence[Question]
) -> list[Answer]:
    """Ask several questions about one state. One forward pass."""
    return predictor.ask([state], questions)[0]


def ask_noul(predictor: LayaPredictor, state: str, question: Noul) -> NoulAnswer:
    """p(yes) for one question about one state."""
    answer = ask_one(predictor, state, [question])[0]
    if not isinstance(answer, NoulAnswer):
        raise TypeError(f"{type(predictor).__name__} answered a noul with {type(answer).__name__}")
    return answer


def ask_choice(predictor: LayaPredictor, state: str, question: Choice) -> ChoiceAnswer:
    """The distribution over one choice's options for one state."""
    answer = ask_one(predictor, state, [question])[0]
    if not isinstance(answer, ChoiceAnswer):
        raise TypeError(
            f"{type(predictor).__name__} answered a choice with {type(answer).__name__}"
        )
    return answer


# ── implementations ────────────────────────────────────────────────────────


def _deterministic_unit(*parts: str) -> float:
    """A stable pseudo-probability in [0, 1) from *parts*.

    blake2b rather than ``hash()``: the built-in is salted per process for
    strings, so a fake built on it would answer differently on every run and a
    test that pinned an answer would pass locally and fail in CI.
    """
    digest = hashlib.blake2b("\x00".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


class FakeLayaPredictor:
    """Deterministic, offline, weightless: the reference implementation.

    The same role ``FakeLLMProposer`` plays for the probe agent. It imports
    nothing and reaches nothing, which is what lets a threshold test, a ranking
    test and a scoring test run on a base install with no checkpoint anywhere
    near the machine.

    Unpinned answers are hashed from the state and the question. That makes them
    stable and arbitrary, in that order. **They are not predictions**: nothing
    about a hash knows anything about a ROCm log, and no number this class
    returns may be reported as a measurement of anything. *pinned* is how a test
    says what it actually wants to exercise, keyed by question text.
    """

    def __init__(self, *, pinned: Mapping[str, Answer] | None = None) -> None:
        self._pinned = dict(pinned or {})

    def model_id(self) -> str:
        return "fake"

    def calibration(self) -> Calibration:
        """Unknown, not clean.

        A fake has no temperature table, so the honest answer is that there is
        nothing to report -- and a caller reading this gets a "calibration
        unknown" caveat rather than silence. That is the right way round: a test
        double whose calibration reads as fine would be one more way for a
        hash-derived number to pass for a measured one.
        """
        return Calibration(model_id=self.model_id(), unknown_because=_NO_WEIGHTS)

    def ask(
        self, states: Sequence[str], questions: Sequence[Question]
    ) -> list[list[Answer]]:
        return [[self._answer(state, question) for question in questions] for state in states]

    def _answer(self, state: str, question: Question) -> Answer:
        pinned = self._pinned.get(question.question)
        if pinned is not None:
            return pinned
        if isinstance(question, Noul):
            return NoulAnswer(probability=_deterministic_unit("noul", question.question, state))
        weights = [
            # The option is in the key so two options of one question differ,
            # and the question is in it so one option's weight is not shared
            # across every question that happens to offer it.
            _deterministic_unit("choice", question.question, option, state) + 1e-9
            for option in question.options
        ]
        total = sum(weights) or 1.0
        return ChoiceAnswer(
            probabilities=tuple(
                (option, weight / total)
                for option, weight in zip(question.options, weights)
            )
        )


class LayaAgentPredictor:
    """A pinned Laya checkpoint, loaded on first use.

    The translation is thin on purpose: this class owns the mapping between our
    :class:`Noul` / :class:`Choice` and the library's question dicts, and owns
    reading a probability out of the answer payload. Everything else -- which
    threshold, which option set, what to do when the model defers -- belongs to
    the caller, because those are the decisions each of the three integration
    tracks makes differently.

    Loading is deferred to the first :meth:`ask` rather than done in
    ``__init__`` for the reason ``LogWatcher`` defers building its ReAct module:
    one instance serves a whole poll loop, a cold checkpoint build costs
    seconds, and a headless run that never reaches an ambiguous log should not
    pay for weights it does not use.

    *load* is the injection point. It defaults to the library's own
    ``laya.load``, and a caller that has the weights staged somewhere this does
    not know about -- an air-gapped node, a digest-pinned local directory once
    Decision 22 exists -- passes its own.

    A *checkpoint* naming a local directory is how the Phase 1 gate scores a
    fine-tune, which exists on disk before it exists in any repository. The
    directory has to look like a checkpoint: 0.3.5 fetches exactly
    ``rl_agent_config.json``, ``model.safetensors``, ``tokenizer/`` and
    ``encoder/`` from a hub repository, and the loader fails on a directory with
    no ``rl_agent_config.json`` beside the weights. A training run's raw output
    directory is usually not that shape, so the failure to expect from
    ``--checkpoint`` is a missing config rather than a bad path.
    """

    def __init__(
        self,
        checkpoint: str = "laya-typed-decisions",
        *,
        device: str | None = None,
        load: Any = None,
    ) -> None:
        if checkpoint not in CHECKPOINTS and load is None and not _is_local(checkpoint):
            raise LayaUnavailable(
                f"unknown Laya checkpoint {checkpoint!r} "
                f"(expected one of {', '.join(sorted(CHECKPOINTS))}, or a local "
                "directory holding a fine-tune). "
                "Pass load= to reach a checkpoint this package does not name."
            )
        self._checkpoint = checkpoint
        self._device = device
        self._load = load
        self._loaded: Any = None

    def model_id(self) -> str:
        """The checkpoint name, plus the device once one has been chosen.

        Reported before loading as well as after, so a run that fails to load
        still says which weights it was reaching for.
        """
        suffix = f"@{self._device}" if self._device else ""
        return f"{self._checkpoint}{suffix}"

    def _agent(self) -> Any:
        """The loaded model. Imports the library here, not at module scope.

        ``import laya`` inside the method is what keeps torch off the import
        path of ``aorta.cli`` and of the agents; see this module's docstring for
        the two CI gates that assert it.

        ``USE_TF`` is set before the import because transformers probes for
        TensorFlow when it is imported, and abseil can deadlock during model
        construction if it finds one. On the CIA side that stalls a poll loop;
        on the chat side it would hang the Chainlit server, which is a large
        cost for a framework nothing here uses.

        A ``RuntimeWarning`` from the load is deliberately not suppressed. 0.3.5
        emits one when a checkpoint ships a fitted temperature it has had to clamp
        into [0.5, 5.0], and it names the bucket -- which is the single most
        useful thing anyone gating on a probability from this model could be told,
        because a clamped bucket's confidence is uncalibrated whatever the number
        looks like. Swallowing it here would leave a Watch clean-gate thresholding
        on a figure the library had already said not to trust.
        """
        if self._loaded is not None:
            return self._loaded

        loader = self._load
        if loader is None:
            import os

            os.environ.setdefault("USE_TF", "0")
            try:
                import laya
            except ImportError as exc:
                raise LayaUnavailable(
                    "Laya is required for a real typed-decision predictor. "
                    "Install it with:\n"
                    "  pip install 'amd-aorta[laya]'\n"
                    "It brings torch, which is why it is its own extra and why "
                    "nothing reachable from the chat-cli extra may import it "
                    "(Decision 19a is enforced in nightly.yml and release.yml).\n"
                    f"(missing: {exc.name or 'laya'})"
                ) from exc

            # A local directory addresses itself. The Phase 1 gate has to score a
            # fine-tune against the two published checkpoints, and a fine-tune
            # exists on disk before it exists in a repository -- so the name is
            # either one of ours or a path, and both reach the same loader.
            repository, subfolder = CHECKPOINTS.get(
                self._checkpoint, (self._checkpoint, None)
            )

            def loader(_checkpoint: str) -> Any:
                return laya.load(repository, device=self._device, subfolder=subfolder)

        try:
            self._loaded = loader(self._checkpoint)
        except LayaUnavailable:
            raise
        except Exception as exc:
            # The weights are downloaded on first use, so this is the ordinary
            # no-egress failure as much as it is a bad checkpoint name. Naming
            # what was being loaded matters: the library's own error says which
            # file it could not fetch and not which of our tiers asked for it.
            raise LayaUnavailable(
                f"could not load the Laya checkpoint {self.model_id()}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return self._loaded

    def calibration(self) -> Calibration:
        """What this checkpoint's temperature table asked for, and what was applied.

        Read by comparing the two tables 0.3.5 keeps on the loaded agent --
        ``temperature_by_options_raw`` against ``temperature_by_options``, and the
        per-question-type fallback list beside them. Comparing the tables rather
        than capturing the ``RuntimeWarning`` is deliberate: the warning is raised
        once, during ``__init__``, so anything that wanted to catch it would have
        to have installed a filter before the load it does not control the timing
        of. The tables are still there afterwards.

        Does not load. An unloaded checkpoint reports itself unknown, because a
        caller asking a question about calibration should not be the thing that
        spends seven seconds building a model.
        """
        if self._loaded is None:
            return Calibration(
                model_id=self.model_id(),
                unknown_because=(
                    "the checkpoint has not been loaded yet, so its temperature "
                    "table has not been read; ask a question first"
                ),
            )
        agent = self._loaded
        shipped = getattr(agent, "temperature_by_options_raw", None)
        applied = getattr(agent, "temperature_by_options", None)
        if not isinstance(shipped, Mapping) or not isinstance(applied, Mapping):
            # Laya below 0.3.5 applies the shipped table verbatim and keeps no
            # copy to compare against. Reported as unknown rather than as clean,
            # because on that version the answer is not that nothing was clamped
            # -- it is that the sharpening was applied and nobody was told.
            # ``pyproject.toml`` floors the extra at 0.3.5 for this reason.
            return Calibration(
                model_id=self.model_id(),
                unknown_because=(
                    "this laya build does not report the temperature table it was "
                    "given, which 0.3.5 added; a build that cannot say whether it "
                    "clamped is one that did not"
                ),
            )

        clamped = [
            ClampedBucket(bucket=str(bucket), shipped=float(value), applied=float(applied[bucket]))
            for bucket, value in shipped.items()
            if bucket in applied and float(applied[bucket]) != float(value)
        ]
        clamped.extend(self._clamped_fallbacks(agent))
        return Calibration(
            model_id=self.model_id(),
            clamped=tuple(sorted(clamped, key=lambda entry: entry.bucket)),
            applied=tuple(sorted((str(k), float(v)) for k, v in applied.items())),
        )

    @staticmethod
    def _clamped_fallbacks(agent: Any) -> list[ClampedBucket]:
        """Clamps in the per-question-type fallback list, named by question type.

        The fallback applies to every bucket the explicit table has no entry for,
        so a clamp here is wider in effect than a clamp in one bucket and would be
        the more serious of the two to miss. The names come from the library's own
        ``QTYPE_NAMES`` where it can be read -- the weights are loaded by the time
        this runs, so torch is already imported and the import is free -- and fall
        back to the position when it cannot, because a bucket reported as
        ``fallback[0]`` is still better than one not reported.
        """
        shipped = getattr(agent, "temperature_raw", None)
        applied = getattr(agent, "temperature", None)
        if not isinstance(shipped, Sequence) or not isinstance(applied, Sequence):
            return []
        try:
            from laya.common import QTYPE_NAMES

            names = {int(index): str(name) for index, name in QTYPE_NAMES.items()}
        except (ImportError, AttributeError, TypeError, ValueError):
            names = {}
        found: list[ClampedBucket] = []
        for index, value in enumerate(shipped):
            if index >= len(applied):
                continue
            try:
                was, now = float(value), float(applied[index])
            except (TypeError, ValueError):
                continue
            if was != now:
                label = names.get(index, f"fallback[{index}]")
                found.append(ClampedBucket(bucket=f"{label}:*", shipped=was, applied=now))
        return found

    def token_count(self, text: str) -> int:
        """How many tokens *text* costs this checkpoint's own tokenizer.

        Exposed because Phase 1 has to measure what fraction of Watch's log
        deltas exceed the 512- and 1024-token context windows, and there is only
        one honest way to count: the tokenizer the model will actually use. A
        characters-per-token rule of thumb is calibrated on prose, and a ROCm log
        is timestamps, hex addresses and kernel names -- the case where such a
        rule is furthest wrong.

        Deliberately absent from :class:`FakeLayaPredictor`, so a census cannot
        be produced without weights. A fabricated token distribution would decide
        whether chunking is required, which is the finding that makes or breaks
        the single-forward-pass claim.
        """
        tokenizer = getattr(self._agent(), "tok", None)
        if tokenizer is None:
            raise LayaUnavailable(
                f"{self.model_id()} exposes no tokenizer, so token lengths cannot be counted"
            )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def context_length(self) -> int:
        """The checkpoint's own maximum sequence length.

        Read off the loaded config rather than assumed, because the three
        checkpoints differ: the English encoder is 512 and typed-decisions is
        1024, and a census against the wrong one would answer the chunking
        question backwards.
        """
        config = getattr(self._agent(), "cfg", None) or {}
        return int(config.get("max_len", 0))

    def ask(
        self, states: Sequence[str], questions: Sequence[Question]
    ) -> list[list[Answer]]:
        """Answer every question about each state.

        One library call per state, because that is what one forward pass is:
        the questions share the state's encoding, and a second state cannot.
        Nothing here pretends otherwise by batching across states -- the
        latency of N states is N passes, and a caller deciding whether it can
        afford thirty of them needs that to be the honest shape.
        """
        if not questions:
            return [[] for _ in states]
        agent = self._agent()
        payload = {f"q{index}": _as_library_question(q) for index, q in enumerate(questions)}
        results: list[list[Answer]] = []
        for state in states:
            answers = agent.predict(state, payload).get("answers") or {}
            results.append(
                [
                    _as_answer(questions[index], answers.get(f"q{index}"))
                    for index in range(len(questions))
                ]
            )
        return results


def _as_library_question(question: Question) -> dict[str, Any]:
    """One of ours in the shape the library reads.

    ``criteria`` is omitted rather than passed empty for a bare noul: the
    library supplies "yes, the statement holds" / "no, the statement does not
    hold" itself, and an empty dict would replace two sentences the model was
    trained against with nothing.
    """
    if isinstance(question, Noul):
        payload: dict[str, Any] = {"type": "noul", "instructions": question.question}
        if question.when_true or question.when_false:
            payload["criteria"] = {
                "true": question.when_true,
                "false": question.when_false,
            }
        return payload
    glossed = dict(question.criteria)
    return {
        "type": "choice",
        "instructions": question.question,
        # Every option appears as a key whether or not it is glossed, because
        # the keys are the answer space: an option missing from this dict is an
        # option the model is not offered and cannot return.
        "criteria": {option: glossed.get(option, "") for option in question.options},
    }


def _as_answer(question: Question, payload: Any) -> Answer:
    """One library answer as one of ours, or a loud failure.

    A missing or malformed payload raises rather than defaulting. This is the
    opposite of how the LLM proposers treat an unparseable reply, and
    deliberately: there, a failed parse is an expected condition that has to
    still produce a report, whereas here it means the library's answer shape has
    moved under us, and a 0.5 invented at this seam would travel into a
    threshold comparison as though it had been measured.
    """
    if not isinstance(payload, Mapping):
        raise LayaUnavailable(
            f"Laya returned no answer for {question.question!r} "
            f"(got {type(payload).__name__})"
        )
    if isinstance(question, Noul):
        value = payload.get("noul")
        if not isinstance(value, (int, float)):
            raise LayaUnavailable(
                f"Laya answered the noul {question.question!r} with no 'noul' "
                f"probability (keys: {sorted(payload)})"
            )
        return NoulAnswer(probability=float(value))

    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise LayaUnavailable(
            f"Laya answered the choice {question.question!r} with no "
            f"'probabilities' map (keys: {sorted(payload)})"
        )
    missing = [option for option in question.options if option not in probabilities]
    if missing:
        raise LayaUnavailable(
            f"Laya answered the choice {question.question!r} without the "
            f"offered options {missing}"
        )
    # Re-read in the order offered rather than in the order returned, so a
    # caller zipping an answer against its own option list cannot be misaligned.
    return ChoiceAnswer(
        probabilities=tuple(
            (option, float(probabilities[option])) for option in question.options
        )
    )


def _is_local(checkpoint: str) -> bool:
    """Whether *checkpoint* names a directory on this machine.

    A published checkpoint is ``org/name``, which also looks like a relative
    path, so the test is whether the directory is actually there rather than
    whether the string has a slash in it.
    """
    from pathlib import Path

    try:
        return Path(checkpoint).expanduser().is_dir()
    except OSError:
        return False


def make_predictor(
    backend: str = DEFAULT_BACKEND,
    *,
    device: str | None = None,
    checkpoint: str | None = None,
) -> LayaPredictor:
    """Build the predictor named by *backend*.

    ``fake`` stays the default and stays fully offline, for the reason
    ``make_proposer`` keeps ``fake`` as its default: it imports nothing and
    reaches nothing, which is what makes a test suite hermetic.

    *checkpoint* overrides *backend* with a local directory, which is how the
    Phase 1 gate scores a fine-tune: it exists on disk before it exists in any
    repository, and the gate is a comparison between it and the two published
    ones rather than a measurement of either alone.
    """
    if checkpoint:
        return LayaAgentPredictor(checkpoint, device=device)
    if backend == "fake":
        return FakeLayaPredictor()
    if backend in CHECKPOINTS:
        return LayaAgentPredictor(backend, device=device)
    raise ValueError(
        f"unknown Laya backend: {backend!r} "
        f"(expected one of {', '.join(sorted({'fake', *CHECKPOINTS}))}, or a "
        "local checkpoint directory via checkpoint=)"
    )


__all__ = [
    "CHECKPOINTS",
    "DEFAULT_BACKEND",
    "VERIFIED_LAYA_VERSION",
    "Answer",
    "Calibration",
    "Choice",
    "ChoiceAnswer",
    "ClampedBucket",
    "FakeLayaPredictor",
    "LayaAgentPredictor",
    "LayaPredictor",
    "LayaUnavailable",
    "Noul",
    "NoulAnswer",
    "Question",
    "ask_choice",
    "ask_noul",
    "ask_one",
    "bucket_for",
    "make_predictor",
]
