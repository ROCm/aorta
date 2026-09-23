"""A :class:`~aorta.laya.predictor.LayaPredictor` on onnxruntime, with no torch.

This is the chat path's implementation of the seam every other track reaches
through :class:`~aorta.laya.predictor.LayaAgentPredictor`. Same Protocol, same
:class:`~aorta.laya.predictor.Noul` and :class:`~aorta.laya.predictor.Choice`
in, same :class:`~aorta.laya.predictor.NoulAnswer` and
:class:`~aorta.laya.predictor.ChoiceAnswer` out -- so ``router_node`` and
``selector_node`` are written against the contract rather than against this
class, and their tests run on ``FakeLayaPredictor`` with no artifact anywhere
near the machine.

**Why a second implementation exists at all.** Decision 22, Rule 1: nothing
reachable from ``[chat-cli]`` may resolve torch, and the CI gate that enforces
it is a hard failure in ``nightly.yml`` and ``release.yml`` rather than a
convention. ``laya.load()`` is available -- the plan text saying otherwise is
wrong, and ``LayaAgentPredictor`` calls it -- but it is a torch loader, so it is
unavailable *to chat* for a packaging reason rather than a library one. The
runtime here is the onnxruntime ``fastembed`` already installs for
``BAAI/bge-small-en-v1.5``, so this costs nothing at install time.

**The batching asymmetry is the whole design.** One state's tokens are encoded
once, and every question's options are markers inside that same sequence, so M
questions about one state is one session run and N states is N runs. That is
what the Protocol's docstring promises and what lets ``selector_node`` ask one
question per registered tool for the price of one forward pass. Nothing here
batches across states, because pretending N states were free is how a caller
ends up putting thirty of them on an interactive path.

**Everything that can fail, fails loudly.** A missing artifact, a marker that
does not tokenise to one token, a state long enough to truncate an option out
of the window, a graph that returns the wrong number of scores: each raises
:class:`~aorta.chat.laya.artifact.ArtifactUnavailable` rather than returning a
number. The seam's own ``_as_answer`` takes the same line for the same reason --
a 0.5 invented at this level travels into a threshold comparison looking exactly
like a measurement.

**What has never been run.** There is no fine-tuned checkpoint and no Laya
weights on the machine this was written on, so the loaders, the graph signature
and the agreement between this path and the torch path are unexercised. What is
exercised is everything above the session: the rendering, the marker
arithmetic, the temperature application, the answer construction and every
failure message. See ``tests/chat/test_laya_onnx_predictor.py``.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aorta.chat.laya.artifact import (
    GRAPH_OUTPUT,
    Artifact,
    ArtifactUnavailable,
    Manifest,
    RenderTemplate,
    load_artifact,
)
from aorta.laya.predictor import (
    Answer,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
)

logger = logging.getLogger(__name__)

#: What the library puts either side of a bare noul when the caller glosses
#: neither. Quoted from :mod:`aorta.laya.predictor`'s ``_as_library_question``,
#: which records it as read out of the published 0.3.5 wheel.
#:
#: It is reproduced here rather than left out because the ONNX path renders the
#: prompt itself: the library is not in the process to supply its own default,
#: so omitting the gloss would replace two sentences the model was trained
#: against with nothing -- which is the same train/serve skew as re-phrasing
#: them, arriving by deletion instead of by invention.
DEFAULT_NOUL_CRITERIA = {
    "true": "yes, the statement holds",
    "false": "no, the statement does not hold",
}


def option_glosses(question: Question) -> tuple[tuple[str, str], ...]:
    """``(option, criteria)`` in the order the model is offered them.

    In the order offered, and that ordering is load-bearing twice over: a noul
    is laid out ``[false, true]`` and p(yes) is read off the second position, so
    a caller reordering these would invert every answer, and the scores come
    back from the graph positionally with nothing in them to say which option
    each belongs to.
    """
    if isinstance(question, Noul):
        glossed = {"true": question.when_true, "false": question.when_false}
        return tuple(
            (option, glossed[option] or DEFAULT_NOUL_CRITERIA[option])
            for option in question.options
        )
    criteria = dict(question.criteria)
    return tuple((option, criteria.get(option, "")) for option in question.options)


def render(template: RenderTemplate, state: str, questions: Sequence[Question]) -> str:
    """The one string the encoder reads for *state* and all of *questions*.

    One string for every question, not one per question, because that is what
    makes M questions one forward pass. The options of every question are marked
    with the same token; which marker belongs to which question is recovered by
    counting, in :func:`marker_spans`.
    """
    parts = [template.state_template.format(state=state)]
    for question in questions:
        parts.append(template.question_template.format(question=question.question))
        for option, criteria in option_glosses(question):
            parts.append(
                template.option_template.format(
                    marker=template.marker, option=option, criteria=criteria
                )
            )
    return template.separator.join(parts)


def marker_spans(questions: Sequence[Question]) -> list[tuple[int, int]]:
    """Where each question's options sit in the flat run of markers.

    Half-open ``[start, stop)`` per question, in the order rendered. Returned
    rather than recomputed at the point of use so that the rendering and the
    slicing share one definition of the layout -- they went out of step once
    already in review, and the symptom was every answer being one question's
    options late, which reads as a bad model rather than as an off-by-one.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for question in questions:
        width = len(question.options)
        spans.append((cursor, cursor + width))
        cursor += width
    return spans


def softmax(logits: Sequence[float]) -> list[float]:
    """A distribution over *logits*. Pure.

    Shifted by the maximum before exponentiating, which is the standard guard
    against overflow and matters here because nothing bounds what an exported
    scorer head returns.
    """
    if not logits:
        return []
    highest = max(logits)
    weights = [math.exp(value - highest) for value in logits]
    total = sum(weights)
    if total <= 0.0:
        # Unreachable through exp() of finite inputs, and handled anyway: a NaN
        # or an inf out of the graph would otherwise divide to NaN and be
        # published as a probability.
        return [1.0 / len(logits)] * len(logits)
    return [weight / total for weight in weights]


def answers_from_scores(
    questions: Sequence[Question],
    scores: Sequence[float],
    manifest: Manifest,
) -> list[Answer]:
    """One answer per question, from the flat run of marker scores. Pure.

    The temperature comes off the manifest per question bucket and is applied
    through :func:`aorta.laya.eval.apply_temperature`, rather than by dividing
    the logits here. The two are the same operation -- that function takes the
    log of each probability and divides -- and using the eval module's is what
    keeps a fit produced by the Phase 1 harness meaning at inference time
    exactly what it meant when it was fitted. A second implementation of
    temperature scaling, however correct, would be a second thing to keep in
    step with a number that decides whether a threshold holds.
    """
    from aorta.laya.eval import apply_temperature

    spans = marker_spans(questions)
    expected = spans[-1][1] if spans else 0
    if len(scores) != expected:
        raise ArtifactUnavailable(
            f"the exported graph returned {len(scores)} marker scores for "
            f"{expected} option marker(s) across {len(questions)} question(s). "
            "The graph and this build disagree about the signature; re-export."
        )

    answers: list[Answer] = []
    for question, (start, stop) in zip(questions, spans, strict=True):
        distribution = tuple(
            zip(question.options, softmax(list(scores[start:stop])), strict=True)
        )
        scaled = dict(apply_temperature(distribution, manifest.temperature_for(question)))
        if isinstance(question, Noul):
            answers.append(NoulAnswer(probability=scaled["true"]))
        else:
            answers.append(
                ChoiceAnswer(
                    probabilities=tuple(
                        (option, scaled[option]) for option in question.options
                    )
                )
            )
    return answers


class OnnxLayaPredictor:
    """An exported Laya artifact on onnxruntime. Loaded on first use.

    Loading is deferred to the first :meth:`ask` for the reason
    :class:`~aorta.laya.predictor.LayaAgentPredictor` defers its own: one
    instance serves a whole chat session, building an ONNX session costs real
    time, and a conversation that never routes anywhere should not pay for
    weights it does not use.

    *session_factory* and *tokenizer_factory* are the injection points, the same
    role ``load=`` plays on the torch side. Tests pass fakes; nothing on the
    chat path reaches these arguments, so a session cannot end up reporting a
    stub's output as a verdict.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        verify_digest: bool = False,
        session_factory: Any = None,
        tokenizer_factory: Any = None,
    ) -> None:
        self._directory = Path(directory).expanduser()
        self._verify_digest = verify_digest
        self._session_factory = session_factory
        self._tokenizer_factory = tokenizer_factory
        self._artifact: Artifact | None = None
        self._session: Any = None
        self._tokenizer: Any = None
        self._marker_id: int | None = None
        # Locked for the reason FastembedBgeEmbeddings locks its model: the
        # graph's async default runs a node through an executor, so two
        # Chainlit sessions whose first routed turns overlap arrive here on
        # different threads, both see None, and both build an ONNX session --
        # one of which is thrown away having spent the memory and the seconds.
        self._lock = threading.Lock()

    # ── identity ───────────────────────────────────────────────────────────

    def model_id(self) -> str:
        """The checkpoint, the graph digest and the temperature fit.

        Readable before the session is built, because a run that fails to load
        still has to be able to say which artifact it was reaching for. Falls
        back to the directory when even the manifest could not be read -- an
        unhelpful answer is still better than an exception thrown from the line
        that was assembling an error message.
        """
        try:
            return self._load().manifest.identity()
        except ArtifactUnavailable:
            return f"laya-onnx:{self._directory}(unloadable)"

    # ── loading ────────────────────────────────────────────────────────────

    def _load(self) -> Artifact:
        if self._artifact is None:
            self._artifact = load_artifact(self._directory, verify_digest=self._verify_digest)
        return self._artifact

    def _tokenizer_for(self, artifact: Artifact) -> Any:
        """The checkpoint's own tokenizer, over ``tokenizers`` rather than transformers.

        ``tokenizers`` is the Rust library ``fastembed`` already depends on, so
        it costs nothing extra -- and, unlike ``transformers``, it does not probe
        for TensorFlow at import. Decision 22 counts that as a benefit of the
        ONNX path rather than a mitigation of it: the abseil deadlock the model
        card warns about would be a Chainlit server that simply never answers,
        with no error and no log line, and ``USE_TF=0`` only prevents it on
        hosts where somebody remembered to set it.
        """
        if self._tokenizer_factory is not None:
            return self._tokenizer_factory(artifact.tokenizer_path)
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ArtifactUnavailable(
                "the Laya ONNX path needs the 'tokenizers' package, which arrives "
                "with fastembed in the chat-cli extra.\n"
                "  pip install 'amd-aorta[chat-cli]'\n"
                f"(missing: {exc.name or 'tokenizers'})"
            ) from exc
        try:
            return Tokenizer.from_file(str(artifact.tokenizer_path))
        except Exception as exc:
            raise ArtifactUnavailable(
                f"could not read the tokenizer at {artifact.tokenizer_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _session_for(self, artifact: Artifact) -> Any:
        if self._session_factory is not None:
            return self._session_factory(artifact.graph_path)
        try:
            import onnxruntime
        except ImportError as exc:
            raise ArtifactUnavailable(
                "the Laya ONNX path needs onnxruntime, which arrives with fastembed "
                "in the chat-cli extra.\n"
                "  pip install 'amd-aorta[chat-cli]'\n"
                f"(missing: {exc.name or 'onnxruntime'})"
            ) from exc
        try:
            return onnxruntime.InferenceSession(
                str(artifact.graph_path),
                # CPU only, said rather than defaulted. This runs beside the
                # embedding model on whatever the operator's laptop or login
                # node is, and a CUDA provider silently selected on a machine
                # that has one is not something an interactive path should
                # acquire by accident.
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ArtifactUnavailable(
                f"could not open the exported graph at {artifact.graph_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _ready(self) -> tuple[Artifact, Any, Any, int]:
        """The artifact, session, tokenizer and marker token id, built once."""
        if self._session is None or self._tokenizer is None:
            with self._lock:
                if self._session is None or self._tokenizer is None:
                    artifact = self._load()
                    tokenizer = self._tokenizer_for(artifact)
                    marker = artifact.manifest.render.marker
                    marker_id = tokenizer.token_to_id(marker)
                    if marker_id is None:
                        raise ArtifactUnavailable(
                            f"the option marker {marker!r} is not a single token in "
                            f"{artifact.tokenizer_path}. A marker that tokenises into "
                            "several pieces puts several positions where the scorer "
                            "expects one, which misaligns every answer after the first "
                            "without failing."
                        )
                    self._session = self._session_for(artifact)
                    self._tokenizer = tokenizer
                    self._marker_id = int(marker_id)
        assert self._marker_id is not None  # set together with the tokenizer
        return self._load(), self._session, self._tokenizer, self._marker_id

    # ── the seam ───────────────────────────────────────────────────────────

    def ask(
        self, states: Sequence[str], questions: Sequence[Question]
    ) -> list[list[Answer]]:
        """Answer every question about each state. One session run per state."""
        if not questions:
            return [[] for _ in states]
        artifact, session, tokenizer, marker_id = self._ready()
        return [
            self._ask_one(artifact.manifest, session, tokenizer, marker_id, state, questions)
            for state in states
        ]

    def _ask_one(
        self,
        manifest: Manifest,
        session: Any,
        tokenizer: Any,
        marker_id: int,
        state: str,
        questions: Sequence[Question],
    ) -> list[Answer]:
        text = render(manifest.render, state, questions)
        encoded = tokenizer.encode(text)
        ids = list(encoded.ids)
        expected = sum(len(question.options) for question in questions)

        if len(ids) > manifest.max_len:
            # Refused rather than truncated, and this is the failure most likely
            # to happen in practice: the questions are rendered after the state,
            # so a long chat turn does not degrade the answer, it silently eats
            # the option markers off the end and leaves the last questions with
            # no positions to score. Truncating here would turn "the state was
            # too long" into "the model preferred the first few tools", which is
            # indistinguishable from a verdict.
            raise ArtifactUnavailable(
                f"the rendered state and its {len(questions)} question(s) tokenise to "
                f"{len(ids)} tokens, over the {manifest.max_len} this checkpoint was "
                "exported for. Shorten the state; truncating it would drop option "
                "markers off the end and misreport the answer."
            )

        positions = [index for index, token in enumerate(ids) if token == marker_id]
        if len(positions) != expected:
            raise ArtifactUnavailable(
                f"rendering {len(questions)} question(s) put {len(positions)} option "
                f"marker(s) in the sequence where {expected} were offered. The render "
                "template and the tokenizer in this artifact disagree about the "
                "marker token."
            )

        scores = self._run(session, ids, positions)
        return answers_from_scores(questions, scores, manifest)

    def _run(self, session: Any, ids: list[int], positions: list[int]) -> list[float]:
        """One forward pass, as a flat list of per-marker scores.

        numpy is imported here rather than at module scope. It arrives with
        onnxruntime, so it is present wherever this class can actually run, and
        keeping it out of the module's import graph is what lets the pure
        functions above be unit-tested on an install that has neither.
        """
        import numpy

        feed = {
            "input_ids": numpy.asarray([ids], dtype=numpy.int64),
            "attention_mask": numpy.ones((1, len(ids)), dtype=numpy.int64),
            "marker_positions": numpy.asarray(positions, dtype=numpy.int64),
        }
        try:
            outputs = session.run([GRAPH_OUTPUT], feed)
        except Exception as exc:
            raise ArtifactUnavailable(
                f"the exported Laya graph failed to run: {type(exc).__name__}: {exc}"
            ) from exc
        if not outputs:
            raise ArtifactUnavailable(
                f"the exported Laya graph returned nothing for {GRAPH_OUTPUT!r}"
            )
        # reshape(-1) rather than indexing a known rank: the export writes a
        # 1-D score per marker, and a graph that came back [1, n] instead is a
        # shape difference rather than a wrong answer.
        return [float(value) for value in numpy.asarray(outputs[0]).reshape(-1)]


__all__ = [
    "DEFAULT_NOUL_CRITERIA",
    "OnnxLayaPredictor",
    "answers_from_scores",
    "marker_spans",
    "option_glosses",
    "render",
    "softmax",
]
