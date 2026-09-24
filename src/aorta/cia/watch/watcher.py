from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy

from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text
from aorta.cia.llm import ensure_configured

if TYPE_CHECKING:  # pragma: no cover - an annotation must not cost an import
    from aorta.local_classifier.predictor import Choice, DecisionPredictor, Noul

_SANITIZER_RESULT = re.compile(
    r"^\[sanitizer\] (?P<name>[A-Za-z0-9_-]+): "
    r"verdict=(?P<verdict>[a-z_]+) state=(?P<state>[a-z_]+) "
    r"findings=(?P<findings>[0-9]+)$",
    re.MULTILINE,
)


def sanitizer_assessment(new_content: str) -> dspy.Prediction | None:
    """Return a deterministic verdict for a sanitizer's own summary line.

    Machine-readable evidence does not need an LLM to decide whether the
    sanitizer ran and failed. Keeping this in Watch means both the production
    batch summary and the hardware acceptance smoke travel through the same
    alert path; malformed or unrelated logs still fall through to ReAct.
    """
    matches = list(_SANITIZER_RESULT.finditer(new_content))
    if not matches:
        return None

    failures = [
        match
        for match in matches
        if match.group("state") == "ran"
        and match.group("verdict") in {"warn", "fail", "error"}
    ]
    if failures:
        match = failures[0]
        line = match.group(0)
        return dspy.Prediction(
            healthy=False,
            signal="WATCH_UNKNOWN_ERROR",
            confidence=1.0,
            evidence=line,
            assessment=(
                f"{match.group('name')} ran and reported "
                f"{match.group('verdict')} with "
                f"{match.group('findings')} finding(s)."
            ),
        )

    if all(
        match.group("state") == "ran" and match.group("verdict") == "pass"
        for match in matches
    ):
        return dspy.Prediction(
            healthy=True,
            signal="WATCH_CLEAN",
            confidence=1.0,
            evidence="none",
            assessment="The requested sanitizers ran and reported pass.",
        )
    return None


# ---------------------------------------------------------------------------
# Tools available to the ReAct loop
# ---------------------------------------------------------------------------

#: What the tools below may read, for the assessment currently running.
#:
#: Their ``path`` argument is chosen by the model, so without this they read any
#: file the agent can: /etc/passwd, an SSH config, or a path the model noticed in
#: a log. Redaction does not help here -- it rewrites paths and addresses in the
#: text, not the contents of whatever was opened -- and what comes back goes
#: into the trajectory and on to the provider.
#:
#: Empty by default, and empty means refuse. A caller that forgets to bind gets
#: nothing rather than everything.
_ALLOWED_ROOTS: ContextVar[tuple[Path, ...]] = ContextVar("_ALLOWED_ROOTS", default=())


@contextmanager
def reading_within(*roots: Path | str) -> Iterator[None]:
    """Let the tools read under *roots*, and nowhere else, inside this block."""
    resolved = tuple(Path(r).resolve() for r in roots if r)
    token = _ALLOWED_ROOTS.set(resolved)
    try:
        yield
    finally:
        _ALLOWED_ROOTS.reset(token)


def _allowed(path: str) -> Path | None:
    """*path* resolved, if it is inside one of the roots for this assessment."""
    roots = _ALLOWED_ROOTS.get()
    if not roots or not path:
        return None
    try:
        candidate = Path(path).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    for root in roots:
        if candidate == root or candidate.is_relative_to(root):
            return candidate
    return None


def read_file_tail(path: str, lines: int = 80) -> str:
    """Read the last N lines of a file for extra context. Does not advance cursor."""
    try:
        p = _allowed(path)
        if p is None:
            return f"[refused: {path} is outside this job's directory]"
        if not p.is_file():
            return f"[file not found: {path}]"
        text = p.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except Exception as e:
        return f"[error reading {path}: {e}]"


def list_job_files(job_dir: str) -> str:
    """List files in job dir with sizes and mtimes — spot new log files."""
    import subprocess
    directory = _allowed(job_dir)
    if directory is None:
        return f"[refused: {job_dir} is outside this job's directory]"
    try:
        r = subprocess.run(
            ["find", str(directory), "-maxdepth", "4", "-type", "f",
             "-printf", "%T@ %s %p\n"],
            capture_output=True, text=True, timeout=10,
        )
        lines = sorted(r.stdout.strip().splitlines(), reverse=True)[:30]
        out = []
        for line in lines:
            parts = line.split(" ", 2)
            if len(parts) == 3:
                out.append(f"{parts[2]}  ({int(float(parts[1])):,} bytes)")
        return "\n".join(out) or "[empty directory]"
    except Exception as e:
        return f"[error: {e}]"


def count_repeated_lines(text: str) -> dict:
    """Count line frequencies — high repeat count signals a hang (same step printed over and over)."""
    from collections import Counter
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = Counter(lines)
    top = counts.most_common(5)
    max_repeats = top[0][1] if top else 0
    return {
        "max_repeats": max_repeats,
        "likely_hang": max_repeats >= 5,
        "top_repeated": [{"line": l, "count": c} for l, c in top[:3]],
    }


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------

class WatchAssessment(dspy.Signature):
    """
    You are monitoring a live GPU training job on an AMD GPU cluster.
    You receive the NEW log content since the last check — not the full history.
    The operator has stated what healthy training should look like.

    Your job:
    1. Read the new content carefully.
    2. Use tools to get more context if the new content is ambiguous or very short.
    3. Judge whether each expectation is being met.
    4. Alert only when an expectation is CLEARLY violated — not on startup noise,
       expected warnings, or one-off messages.

    Signal slugs to use:
    - WATCH_CLEAN          : everything looks healthy
    - WATCH_NUMERIC_NAN    : NaN, non-finite, or loss divergence detected
    - WATCH_HANG           : training appears stalled (no step progress)
    - WATCH_LOSS_STALL     : loss has plateaued and is not decreasing as expected
    - WATCH_THROUGHPUT_LOW : throughput/speed dropped significantly below expectation
    - WATCH_OOM            : out-of-memory error
    - WATCH_UNKNOWN_ERROR  : clear error but doesn't fit above categories
    """
    new_content: str = dspy.InputField(
        desc="New log lines since last poll, labelled by filename. May be empty if no new output.")
    job_context: str = dspy.InputField(
        desc="job_id, node, recipe, elapsed_time_sec, total_bytes_seen so far")
    expectations: str = dspy.InputField(
        desc="Operator-stated expectations for healthy training (plain English)")

    healthy: bool = dspy.OutputField(desc="True if no expectation is clearly violated")
    signal: str = dspy.OutputField(desc="One signal slug from the list above")
    confidence: float = dspy.OutputField(desc="0.0-1.0 — how certain you are of this assessment")
    evidence: str = dspy.OutputField(
        desc="Specific log lines (with filename) that show the violation, or 'none' if healthy")
    assessment: str = dspy.OutputField(
        desc="One paragraph: what training looks like right now based on the new content")


# ---------------------------------------------------------------------------
# The local-classifier tier: the clean-gate, between the sanitizer regex and ReAct
# ---------------------------------------------------------------------------

#: The checkpoint the tier resolves when the config names none.
DEFAULT_CLASSIFIER_BACKEND = "laya-typed-decisions"

#: How sure the local classifier has to be that a delta is healthy before Watch declines to
#: spend an LLM call on it.
#:
#: Deliberately high, and deliberately not a fitted number. Nothing in this
#: repository has yet scored a classifier checkpoint against a ROCm log, and the
#: per-(question type, option count) temperature refit that makes one of its
#: probabilities mean what it says has not been run either -- so 0.90 is a
#: policy rather than a measurement: decline far more often than fire until
#: Phase 1 of docs/plans/laya-system1-integration.md has something to say.
DEFAULT_CLEAN_THRESHOLD = 0.90

# The wording of a typed question is part of the question, and so is its option
# set. A temperature is fitted per (question type, option count) against a
# corpus of question/answer pairs, so a question reworded or an option added at
# one of the two call sites is a different question from the one the fit was
# made against -- and the probability that comes back is then calibrated
# against nothing while still looking like a probability. Nothing raises. The
# gate simply thresholds a number that no longer means what its name says.
#
# So there is one definition, here, and ``aorta.local_classifier.corpus.watch`` imports it
# to label against. This module is the canonical side for the same reason
# ``aorta/agent/llm.py`` is for the proposer's questions and Autopsy's category:
# the question is about *Watch's decision*, not about the encoder.
# ``aorta.local_classifier.predictor`` serves every track, and if each track's phrasings
# lived there the seam would become a string registry.
#
# Bare strings rather than assembled ``Noul`` / ``Choice`` objects, matching
# ``CLASSIFIER_CATEGORY_QUESTION`` and its neighbours, because the seam's types are
# what a caller assembles and the text is what has to be identical. The
# assembly is :func:`healthy_question` and :func:`signal_question` below, which
# is what a caller wanting the whole question should import -- re-assembling
# from the parts is where an option order or a missing gloss creeps in.

#: Watch's answer space for the signal question, minus ``WATCH_CLEAN``.
#:
#: The slugs are the ``WatchAssessment`` signature's own list, which is in this
#: module, which is the other half of why the definition belongs here.
#: ``WATCH_CLEAN`` is deliberately absent: it is the answer to the healthy noul,
#: and offering it here as well would let one forward pass disagree with itself.
WATCH_SIGNALS: tuple[str, ...] = (
    "WATCH_NUMERIC_NAN",
    "WATCH_HANG",
    "WATCH_LOSS_STALL",
    "WATCH_THROUGHPUT_LOW",
    "WATCH_OOM",
    "WATCH_UNKNOWN_ERROR",
)

#: The gate's question. p(yes) is what ``clean_threshold`` is compared against.
CLASSIFIER_HEALTHY_QUESTION = (
    "Is this new training log output healthy -- no expectation clearly "
    "violated, and nothing here that needs a closer look?"
)

#: The two sides of it. A bare noul gets the library's own generic pair, which
#: says nothing about training: "healthy" has to be told from "quiet", and the
#: false side has to name the three things Watch is actually looking for.
CLASSIFIER_HEALTHY_WHEN_TRUE = (
    "training is progressing normally, or this is ordinary startup noise"
)
CLASSIFIER_HEALTHY_WHEN_FALSE = (
    "an expectation is clearly violated: a bad number, a stall, an error"
)

#: The slug question. Never a verdict -- see :func:`signal_question`.
CLASSIFIER_SIGNAL_QUESTION = (
    "Which signal best describes what went wrong in this training log output?"
)

#: One gloss per slug, in the order offered. Part of the question too: the
#: library sends ``criteria`` as the answer space of a choice, so a slug missing
#: from here is a slug the model is not offered.
CLASSIFIER_SIGNAL_CRITERIA: tuple[tuple[str, str], ...] = (
    ("WATCH_NUMERIC_NAN", "a NaN, a non-finite value, or a diverging loss"),
    ("WATCH_HANG", "training is stalled: no step progress"),
    ("WATCH_LOSS_STALL", "loss has plateaued and is not decreasing as expected"),
    ("WATCH_THROUGHPUT_LOW", "throughput or speed dropped well below expectation"),
    ("WATCH_OOM", "an out-of-memory error"),
    ("WATCH_UNKNOWN_ERROR", "a clear error that fits none of the above"),
)


def healthy_question() -> Noul:
    """The gate's question, assembled. One definition, two callers.

    The runtime tier below and the corpus builder in
    ``aorta.local_classifier.corpus.watch`` both call this, so the thing a temperature is
    fitted against and the thing a threshold is applied to cannot drift apart.

    Imported inside the function rather than at module scope. The seam itself
    is pure dataclasses, but the module that holds it is also where the loader
    lives, and Decision 22 in ``docs/local-classifier-packaging.md`` asks for one rule
    rather than a per-symbol judgement about which imports are cheap today.
    """
    from aorta.local_classifier.predictor import Noul

    return Noul(
        question=CLASSIFIER_HEALTHY_QUESTION,
        when_true=CLASSIFIER_HEALTHY_WHEN_TRUE,
        when_false=CLASSIFIER_HEALTHY_WHEN_FALSE,
    )


def signal_question() -> Choice:
    """The slug question, asked only so the shadow comparison has something to compare.

    It costs nothing: ``ask`` answers every question about one state in a
    single forward pass, and only a second *state* costs a second pass. It also
    buys the measurement that the plan calls the real bar -- whether an encoder
    agrees with the eventual Autopsy category more often than the current ReAct
    assessment does -- which the healthy noul alone cannot answer.

    Nothing reads this answer as a verdict. The gate fires on the noul, and a
    slug is only ever written to a ``watchdog_shadow`` event.
    """
    from aorta.local_classifier.predictor import Choice

    return Choice(
        question=CLASSIFIER_SIGNAL_QUESTION,
        options=WATCH_SIGNALS,
        criteria=CLASSIFIER_SIGNAL_CRITERIA,
    )


@dataclass(frozen=True)
class LocalClassifierObservation:
    """What the local-classifier tier saw about one delta, and whether it acted on it.

    Carried on the returned ``dspy.Prediction`` as ``local_classifier`` rather than written
    from here, because this module has no idea where a job's events file is and
    acquiring one would give the assessment a side effect. ``poll.py`` already
    owns every write to ``events.jsonl``; this is the payload it writes.

    ``gated`` is the whole point of shadow mode being separable from the gate:
    in shadow it is always false, so the same observation is recorded whether or
    not anything was skipped, and the comparison that decides whether to turn
    the gate on is made from traffic rather than from a split.
    """

    model_id: str
    clean_probability: float
    clean_threshold: float
    gated: bool
    vetoed: bool
    signal: str
    signal_probability: float
    #: What has to be said about each probability, or "" when there is nothing.
    #:
    #: Two fields rather than one, because the two questions fall in two
    #: calibration buckets -- the healthy noul is ``noul:2`` and the slug choice
    #: is ``choice:6-10`` -- and a checkpoint can clamp one and not the other.
    #: One combined caveat would have to say which answer it was about, which is
    #: what having two fields says for free.
    clean_caveat: str = ""
    signal_caveat: str = ""

    def as_event_fields(self) -> dict[str, Any]:
        """The local-classifier half of a ``watchdog_shadow`` event.

        ``model_id`` travels inline rather than beside the run, which is rule 2
        of Decision 22: a report read on its own -- copied into a ticket, or
        produced on a node with no run area -- still has to be able to answer
        which weights said this. A checkpoint swap changes verdicts with no
        error and no diff.

        The caveats travel the same way and for the stronger version of the same
        reason. The whole point of a shadow event is that somebody scores it
        later, after this process is gone, and a probability out of a clamped
        bucket scored as though it were calibrated is exactly the error this
        integration exists to remove. They are emitted even when empty, so that
        "this verdict had nothing to disclose" and "this verdict predates the
        disclosure" are different rows rather than the same absent key.
        """
        return {
            "model_id": self.model_id,
            "clean_probability": round(self.clean_probability, 4),
            "clean_threshold": self.clean_threshold,
            "gated": self.gated,
            "vetoed": self.vetoed,
            "signal_probability": round(self.signal_probability, 4),
            "clean_caveat": self.clean_caveat,
            "signal_caveat": self.signal_caveat,
        }


class _LocalClassifierTier:
    """One forward pass, two typed answers, and no prose.

    The tier only ever short-circuits toward *clean*, and that asymmetry is
    forced rather than chosen. ``write_bundle`` persists ``evidence`` -- the log
    lines Autopsy reads -- and the events file carries ``assessment``, the
    paragraph an operator reads. The classifier generates no text, so it cannot produce
    either, and an unhealthy verdict from it would alert with an empty bundle.
    A clean verdict needs neither: ``sanitizer_assessment`` already establishes
    that the clean branch says ``evidence="none"`` and a single fixed sentence.

    So the tier decides whether to spend an LLM call, not what the answer is.
    Below the threshold it returns and ReAct runs exactly as before.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        predictor: DecisionPredictor | None = None,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.shadow = bool(cfg.get("shadow", False))
        self.clean_threshold = float(cfg.get("clean_threshold", DEFAULT_CLEAN_THRESHOLD))
        self.backend = str(cfg.get("backend", DEFAULT_CLASSIFIER_BACKEND) or "")
        # The injection point, the same one ``LayaAgentPredictor`` offers
        # through ``load=``: a test drives the tier with ``FakeDecisionPredictor``
        # and no checkpoint goes anywhere near CI.
        self._predictor = predictor
        # One LogWatcher serves the whole poll loop, so a checkpoint that is
        # not staged on this node has to be discovered once rather than on
        # every delta for as long as Watch runs.
        self._unavailable = False

    @property
    def wanted(self) -> bool:
        """Whether anything should ask the local classifier at all.

        Shadow and the gate are separate flags because they are separate
        decisions: shadow is what produces the comparison, and the gate is what
        the comparison justifies. Either one alone is a reason to load a
        checkpoint; neither is the common case, and it costs nothing.
        """
        return self.enabled or self.shadow

    def observe(self, new_content: str) -> LocalClassifierObservation | None:
        """Score one delta, or None when the tier cannot run.

        The state is the delta and nothing else -- not the job context, not the
        expectations -- because the state is half of a calibrated question. The
        corpus in ``aorta.local_classifier.corpus.watch`` is built from
        ``bundle/logs/watch.stderr.log``, which is the delta alone, so anything
        else here would fit a temperature against one input distribution and
        apply it to another. The cost is real and is recorded rather than
        hidden: ``elapsed_sec`` is exactly what tells "stalled" from "still
        starting", the ReAct tier gets it, and this tier does not.
        """
        if not self.wanted:
            return None
        predictor = self._resolve()
        if predictor is None:
            return None
        try:
            from aorta.local_classifier.predictor import ChoiceAnswer, NoulAnswer, ask_one

            healthy, slug = healthy_question(), signal_question()
            clean, signal = ask_one(predictor, new_content, [healthy, slug])
            if not isinstance(clean, NoulAnswer) or not isinstance(signal, ChoiceAnswer):
                raise TypeError(
                    f"{type(predictor).__name__} answered the clean-gate with "
                    f"{type(clean).__name__} and {type(signal).__name__}"
                )
            # Read after the answers rather than before them. ``calibration()``
            # never triggers a load, which is what makes it cheap and also what
            # would make it useless here if asked first: it would truthfully
            # report "nothing loaded yet" and every verdict from a perfectly
            # calibrated checkpoint would carry an unknown caveat. ``run_gate``
            # reads it in this order for the same reason.
            #
            # Inside the same try as the question, so a predictor that cannot
            # say costs the tier rather than producing a verdict with the
            # disclosure quietly missing. An undisclosed probability is the
            # failure being guarded against, not a degraded form of it.
            calibration = predictor.calibration()
            clean_caveat = calibration.caveat(healthy)
            signal_caveat = calibration.caveat(slug)
        except Exception as exc:  # noqa: BLE001 - a poll must survive a bad tier
            # Said out loud, once, and then not retried. An operator who turned
            # this on and silently got the old behaviour has no other way to
            # find out, and a node with no staged weights must not pay a load
            # attempt per delta for the life of the watcher.
            self._unavailable = True
            print(
                f"[watch] the local-classifier tier failed and will not be retried this run; "
                f"assessment continues without it: {type(exc).__name__}: {exc}"
            )
            return None

        # The veto, and the reason it exists. The plan describes a false clean
        # as costing one poll interval, and against this poll loop that is
        # optimistic: a healthy verdict commits the cursor, so the gated bytes
        # are never read again, and a NaN is usually printed once. A hang keeps
        # printing and would be caught on the next round; a single non-finite
        # loss line would not be. ``scan_stderr_text`` is the deterministic
        # scanner Watch already trusts for exactly that signature, so a delta it
        # flags is never gated however sure the classifier is. Nothing else about the
        # tier changes -- in particular this does not alert on its own, because
        # alerting is ReAct's to do with evidence attached.
        vetoed = scan_stderr_text(new_content).alert
        return LocalClassifierObservation(
            model_id=predictor.model_id(),
            clean_probability=clean.probability,
            clean_threshold=self.clean_threshold,
            # ``at`` rather than a comparison spelled out here, because
            # ``NoulAnswer`` deliberately has no default boolean read: this
            # threshold and ``confidence_threshold`` in poll.py gate opposite
            # directions, and naming which one is meant is the point.
            gated=self.enabled and not vetoed and clean.at(self.clean_threshold),
            vetoed=vetoed,
            signal=signal.option,
            signal_probability=signal.probability,
            clean_caveat=clean_caveat,
            signal_caveat=signal_caveat,
        )

    def _resolve(self) -> DecisionPredictor | None:
        """The predictor, built once, or None when this tier cannot run.

        ``fake`` is refused rather than resolved, which is the one place this
        differs from every other backend selector in the tree.
        ``FakeDecisionPredictor`` answers from a blake2b hash of the question and
        the state: stable, arbitrary, and indistinguishable from a model with an
        opinion. Behind the gate it would skip an LLM call on a coin flip;
        behind shadow it would write hash noise into the events file as a
        verdict, and the comparison shadow mode exists to produce would be run
        against it later by someone who was not here. Tests inject a fake
        directly, which is an explicit act by a caller who knows.
        """
        if self._unavailable:
            return None
        if self._predictor is not None:
            return self._predictor
        if not self.backend or self.backend == "fake":
            self._unavailable = True
            print(
                "[watch] watch.local_classifier is on but names no real checkpoint "
                f"(backend={self.backend!r}); the local-classifier tier is off. A hashed "
                "fake verdict is not a measurement and must not be recorded as one."
            )
            return None
        try:
            from aorta.local_classifier.predictor import make_predictor

            self._predictor = make_predictor(self.backend)
        except Exception as exc:  # noqa: BLE001 - see observe()
            self._unavailable = True
            print(
                f"[watch] could not build the local-classifier predictor {self.backend!r}; "
                f"the local-classifier tier is off: {type(exc).__name__}: {exc}"
            )
            return None
        return self._predictor


def gated_prediction(observation: LocalClassifierObservation) -> dspy.Prediction:
    """The verdict a gated delta gets, in the shape every caller already reads.

    The same five fields ``sanitizer_assessment``'s clean branch returns, for
    the same reason: ``poll.py`` reads them off the prediction with ``getattr``
    and does not care which tier produced it.

    ``assessment`` describes the gate and says so. The honest thing a tier that
    generates no text can write there is what it did and on what number --
    inventing a paragraph about a log nothing read would be worse than the
    sentence being dull, because that paragraph goes into the events file where
    an operator reads it as an observation.

    It also carries the calibration caveat when there is one, which is why the
    caveat is a sentence fragment rather than a structured field: the assessment
    is what gets truncated into an event, copied into a ticket and read by a
    person, and a disclosure that travels anywhere else is a disclosure that
    does not reach them. Only the *clean* caveat, because the gate fired on the
    healthy noul alone -- the ``WATCH_CLEAN`` here is this function's, not the
    slug question's, so attaching that question's caveat would disclose
    something about a number this prediction does not report.
    """
    return dspy.Prediction(
        healthy=True,
        signal="WATCH_CLEAN",
        # The calibrated number, not 1.0. ``sanitizer_assessment`` may claim
        # certainty because a parsed ``verdict=pass`` line is certainty; this is
        # a probability, and reporting it as one is what makes the shadow
        # comparison and the events file worth reading.
        confidence=observation.clean_probability,
        evidence="none",
        assessment=(
            f"The local classifier ({observation.model_id}) scored this delta "
            f"p(healthy)={observation.clean_probability:.2f} against a clean "
            f"threshold of {observation.clean_threshold:.2f}, so no LLM assessment "
            f"was made. This describes the gate, not the log{observation.clean_caveat}."
        ),
        local_classifier=observation,
    )


class LogWatcher(dspy.Module):
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        predictor: DecisionPredictor | None = None,
    ):
        # Machine-readable sanitizer summaries need no model. Build ReAct only
        # when an unstructured log actually reaches it, so a headless hardware
        # gate can verify sanitizer -> Watch -> Autopsy with no provider.
        self.react = None
        self.local_classifier_tier = _LocalClassifierTier(
            (config or {}).get("local_classifier", {}) or {}, predictor
        )

    def forward(
        self,
        new_content: str,
        job_context: str,
        expectations: str,
        allowed_roots: Sequence[Path | str] = (),
    ) -> dspy.Prediction:
        """Assess *new_content*, reading only under *allowed_roots*.

        Three tiers, cheapest first: the sanitizer regex, then the local classifier, then ReAct.
        Every one of them returns the same ``dspy.Prediction`` shape, which is
        why ``poll.py`` needs no branch for which of them answered.

        One instance serves every job in the poll loop, so the roots are bound
        per assessment rather than at construction. Passing none leaves the file
        tools refusing, which is the right default for a caller that has not
        said which job it is asking about.
        """
        machine_verdict = sanitizer_assessment(new_content)
        if machine_verdict is not None:
            # No local classifier, deliberately, and not only to save a pass. It sits
            # *after* this tier, so it will never be asked about a delta the
            # regex answers -- and a shadow comparison gathered on traffic the
            # gate cannot see would measure it on the wrong population.
            return machine_verdict

        # ``getattr`` because ``LogWatcher`` is a dspy.Module, which gets built
        # by paths other than its own ``__init__`` -- the tool-scope test builds
        # one through ``__new__``, and dspy itself copies modules around. A tier
        # that cannot say whether it is on without a constructor having run
        # would fail, and this way the missing answer is "off", which is the
        # safe direction.
        tier = getattr(self, "local_classifier_tier", None)
        observation = tier.observe(new_content) if tier is not None else None
        if observation is not None and observation.gated:
            return gated_prediction(observation)

        if self.react is None:
            ensure_configured()
            self.react = dspy.ReAct(
                WatchAssessment,
                tools=[read_file_tail, list_job_files, count_repeated_lines],
                max_iters=4,
            )
        with reading_within(*allowed_roots):
            prediction = self.react.forward(
                new_content=new_content,
                job_context=job_context,
                expectations=expectations,
            )
        if observation is not None:
            # Shadow mode in one line: the verdict travels beside the ReAct
            # answer and changes nothing about it. Everything downstream still
            # reads ``healthy``, ``signal`` and ``confidence`` off the same
            # object, which is what "no control-flow change" has to mean if it
            # is to be checkable rather than asserted.
            prediction.local_classifier = observation
        return prediction
