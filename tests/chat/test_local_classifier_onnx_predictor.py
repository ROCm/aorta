"""The ONNX predictor, everywhere above the session.

There is no classifier checkpoint and no exported artifact on the machine this was
written on, so what can be exercised is everything between the seam's typed
questions and the graph's inputs, and everything between the graph's outputs
and the seam's typed answers: the rendering, the marker arithmetic, the
temperature application, the answer construction, and every refusal.

The session and the tokenizer are injected, which is the same seam
``LayaAgentPredictor`` opens with ``load=`` and for the same reason -- a test
that needed weights would not be a test anyone could run.

What these do *not* cover, and cannot until weights exist: that the exported
graph computes what the torch path computes, and that the render template in a
real artifact is the format the checkpoint was fine-tuned against.
"""

from __future__ import annotations

import math

import pytest

from aorta.chat.local_classifier.artifact import (
    ARTIFACT_VERSION,
    GRAPH_NAME,
    MANIFEST_NAME,
    TOKENIZER_NAME,
    ArtifactUnavailableError,
    Manifest,
    RenderTemplate,
)
from aorta.chat.local_classifier.onnx_predictor import (
    DEFAULT_NOUL_CRITERIA,
    OnnxDecisionPredictor,
    answers_from_scores,
    marker_spans,
    option_glosses,
    render,
    softmax,
)
from aorta.local_classifier.predictor import (
    Choice,
    ChoiceAnswer,
    DecisionPredictor,
    Noul,
    NoulAnswer,
)

_MARKER = "[OPT]"
_MARKER_ID = 7

_TEMPLATE = RenderTemplate(
    state_template="STATE: {state}",
    question_template="Q: {question}",
    option_template="{marker} {option} : {criteria}",
    separator="\n",
    marker=_MARKER,
)

_CLEAN = Noul(question="is-this-clean")
_GLOSSED = Noul(question="is-it-worth-watching", when_true="it is", when_false="it is not")
_SIGNAL = Choice(question="which-signal", options=("oom", "nan", "hang"))


def _manifest(*, max_len: int = 512, temperatures: dict | None = None) -> Manifest:
    return Manifest.from_dict(
        {
            "artifact_version": ARTIFACT_VERSION,
            "source_checkpoint": "fine-tune",
            "source_revision": "sha256:deadbeef",
            "onnx_sha256": "a" * 64,
            "laya_version": "0.3.5",
            "max_len": max_len,
            "temperatures": temperatures if temperatures is not None else {},
            "render": _TEMPLATE.to_dict(),
        }
    )


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Whitespace tokenizer. Every word is id 1; the marker is :data:`_MARKER_ID`.

    *marker_known* and *encodes_marker* come apart on purpose: an artifact whose
    render template and tokenizer disagree is a real failure mode, and it is one
    the predictor has to catch rather than answer through.
    """

    def __init__(self, *, marker_known: bool = True, encodes_marker: bool = True) -> None:
        self._marker_known = marker_known
        self._encodes_marker = encodes_marker

    def token_to_id(self, token: str):
        if token == _MARKER and self._marker_known:
            return _MARKER_ID
        return None

    def encode(self, text: str) -> _Encoded:
        return _Encoded(
            [
                _MARKER_ID if (word == _MARKER and self._encodes_marker) else 1
                for word in text.split()
            ]
        )


class FakeSession:
    """Returns fixed marker scores and records every call, so one run is provable."""

    def __init__(self, scores: list[float] | None = None, *, raises: Exception | None = None):
        self.scores = scores or []
        self.raises = raises
        self.calls: list[dict] = []

    def run(self, names, feed):
        self.calls.append({"names": list(names), "feed": feed})
        if self.raises is not None:
            raise self.raises
        return [self.scores]


def _predictor(tmp_path, scores, *, manifest: Manifest | None = None, **tokenizer_kwargs):
    """A predictor over a staged directory, with the session and tokenizer faked."""
    manifest = manifest or _manifest()
    directory = tmp_path / "artifact"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / GRAPH_NAME).write_bytes(b"graph")
    (directory / TOKENIZER_NAME).write_text("{}", encoding="utf-8")
    import json

    (directory / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    session = FakeSession(scores) if not isinstance(scores, FakeSession) else scores
    predictor = OnnxDecisionPredictor(
        directory,
        session_factory=lambda _path: session,
        tokenizer_factory=lambda _path: FakeTokenizer(**tokenizer_kwargs),
    )
    return predictor, session


# ── rendering ─────────────────────────────────────────────────────────────


class TestTheOptionsOffered:
    def test_a_noul_is_laid_out_false_then_true(self):
        """The ordering p(yes) is read off. Reversing it inverts every answer."""
        assert [option for option, _ in option_glosses(_CLEAN)] == ["false", "true"]

    def test_a_bare_noul_carries_the_librarys_own_two_sentences(self):
        """Omitting them is the same skew as re-phrasing them, arriving by deletion.

        The torch path lets the library supply these; the ONNX path renders the
        prompt itself, so there is nothing in the process to supply them and
        leaving the gloss empty would train-and-ask differently.
        """
        assert dict(option_glosses(_CLEAN)) == DEFAULT_NOUL_CRITERIA

    def test_a_glossed_noul_keeps_its_own_wording(self):
        assert dict(option_glosses(_GLOSSED)) == {"true": "it is", "false": "it is not"}

    def test_a_choice_offers_every_option_whether_glossed_or_not(self):
        glossed = Choice(question="q", options=("a", "b"), criteria=(("a", "the first"),))
        assert dict(option_glosses(glossed)) == {"a": "the first", "b": ""}


class TestRendering:
    def test_the_state_comes_first_and_every_option_is_marked(self):
        text = render(_TEMPLATE, "a log delta", [_CLEAN, _SIGNAL])
        assert text.startswith("STATE: a log delta")
        assert text.count(_MARKER) == 2 + 3

    def test_every_question_is_in_the_one_string(self):
        """One string, because that is what makes M questions one forward pass."""
        text = render(_TEMPLATE, "state", [_CLEAN, _GLOSSED, _SIGNAL])
        for question in (_CLEAN, _GLOSSED, _SIGNAL):
            assert question.question in text


class TestMarkerSpans:
    def test_each_question_owns_a_contiguous_run(self):
        assert marker_spans([_CLEAN, _SIGNAL, _GLOSSED]) == [(0, 2), (2, 5), (5, 7)]

    def test_no_questions_is_no_markers(self):
        assert marker_spans([]) == []


# ── scoring ───────────────────────────────────────────────────────────────


class TestSoftmax:
    def test_it_is_a_distribution(self):
        assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_a_large_logit_does_not_overflow(self):
        """Nothing bounds what an exported scorer head returns."""
        assert sum(softmax([1000.0, 999.0])) == pytest.approx(1.0)

    def test_nothing_in_is_nothing_out(self):
        assert softmax([]) == []


class TestAnswersFromScores:
    def test_a_noul_reports_p_of_true(self):
        answers = answers_from_scores([_CLEAN], [0.0, 2.0], _manifest())
        assert isinstance(answers[0], NoulAnswer)
        assert answers[0].probability == pytest.approx(1 / (1 + math.exp(-2.0)))

    def test_a_choice_keeps_the_order_offered(self):
        answer = answers_from_scores([_SIGNAL], [3.0, 1.0, 2.0], _manifest())[0]
        assert isinstance(answer, ChoiceAnswer)
        assert [name for name, _ in answer.probabilities] == list(_SIGNAL.options)
        assert answer.option == "oom"

    def test_several_questions_are_sliced_apart_by_their_spans(self):
        answers = answers_from_scores([_CLEAN, _SIGNAL], [0.0, 2.0, 3.0, 1.0, 2.0], _manifest())
        assert answers[0].probability == pytest.approx(1 / (1 + math.exp(-2.0)))
        assert answers[1].option == "oom"

    def test_the_fitted_temperature_is_applied(self):
        """A fit is only a fit if it reaches inference under the same key."""
        warm = _manifest(temperatures={"noul:2": 5.0})
        answer = answers_from_scores([_CLEAN], [0.0, 2.0], warm)[0]
        assert answer.probability == pytest.approx(1 / (1 + math.exp(-0.4)), rel=1e-6)

    def test_an_unfitted_bucket_is_left_alone(self):
        fitted_elsewhere = _manifest(temperatures={"choice:3": 5.0})
        answer = answers_from_scores([_CLEAN], [0.0, 2.0], fitted_elsewhere)[0]
        assert answer.probability == pytest.approx(1 / (1 + math.exp(-2.0)))

    def test_a_graph_returning_the_wrong_number_of_scores_is_refused(self):
        with pytest.raises(ArtifactUnavailableError) as exc:
            answers_from_scores([_CLEAN, _SIGNAL], [1.0, 2.0], _manifest())
        assert "re-export" in str(exc.value).lower()


# ── the predictor ─────────────────────────────────────────────────────────


class TestThePredictor:
    def test_it_satisfies_the_protocol(self, tmp_path):
        predictor, _ = _predictor(tmp_path, [0.0, 1.0])
        checked: DecisionPredictor = predictor
        assert callable(checked.ask) and callable(checked.model_id)

    def test_many_questions_about_one_state_are_one_session_run(self, tmp_path):
        """The batching claim, asserted rather than described.

        This is what lets the selector ask one question per registered tool.
        """
        predictor, session = _predictor(tmp_path, [0.0, 1.0] * 3)
        predictor.ask(["one state"], [_CLEAN, _GLOSSED, Noul(question="third")])
        assert len(session.calls) == 1

    def test_many_states_are_many_runs(self, tmp_path):
        """The other half, and the reason reranking thirty chunks is not free."""
        predictor, session = _predictor(tmp_path, [0.0, 1.0])
        predictor.ask(["a", "b", "c"], [_CLEAN])
        assert len(session.calls) == 3

    def test_no_questions_builds_no_session_at_all(self, tmp_path):
        predictor, session = _predictor(tmp_path, [])
        assert predictor.ask(["a"], []) == [[]]
        assert session.calls == []

    def test_the_marker_positions_it_passes_are_where_the_markers_are(self, tmp_path):
        predictor, session = _predictor(tmp_path, [0.0, 1.0])
        predictor.ask(["a state"], [_CLEAN])
        feed = session.calls[0]["feed"]
        ids = list(feed["input_ids"][0])
        positions = list(feed["marker_positions"])
        assert positions == [index for index, token in enumerate(ids) if token == _MARKER_ID]
        assert len(positions) == 2

    def test_the_attention_mask_covers_the_whole_sequence(self, tmp_path):
        predictor, session = _predictor(tmp_path, [0.0, 1.0])
        predictor.ask(["a state"], [_CLEAN])
        feed = session.calls[0]["feed"]
        assert feed["attention_mask"].shape == feed["input_ids"].shape
        assert set(feed["attention_mask"].reshape(-1).tolist()) == {1}

    def test_the_session_is_built_once(self, tmp_path):
        built = []
        predictor, session = _predictor(tmp_path, [0.0, 1.0])
        predictor._session_factory = lambda path: (built.append(path), session)[1]
        predictor.ask(["a"], [_CLEAN])
        predictor.ask(["b"], [_CLEAN])
        assert len(built) == 1

    def test_loading_is_deferred_until_the_first_question(self, tmp_path):
        """A conversation that never routes must not pay for the weights."""
        opened = []
        directory = tmp_path / "artifact"
        predictor, session = _predictor(tmp_path, [0.0, 1.0])
        predictor._session_factory = lambda path: (opened.append(path), session)[1]
        assert opened == []
        predictor.ask(["a"], [_CLEAN])
        assert opened == [directory / GRAPH_NAME]


class TestTheModelId:
    def test_it_names_the_checkpoint_the_graph_and_the_fit(self, tmp_path):
        predictor, _ = _predictor(tmp_path, [0.0, 1.0])
        assert predictor.model_id().startswith("onnx:fine-tune@")

    def test_an_unloadable_artifact_still_answers(self, tmp_path):
        """A run that fails to load still has to say what it was reaching for.

        Raising from the line that is assembling an error message is how a
        missing artifact turns into a traceback about a missing artifact
        handler.
        """
        predictor = OnnxDecisionPredictor(tmp_path / "nothing")
        assert "unloadable" in predictor.model_id()


class TestItFailsLoudly:
    def test_a_marker_that_is_not_one_token_is_refused(self, tmp_path):
        """Several positions where the scorer expects one misaligns every answer."""
        predictor, _ = _predictor(tmp_path, [0.0, 1.0], marker_known=False)
        with pytest.raises(ArtifactUnavailableError) as exc:
            predictor.ask(["a"], [_CLEAN])
        assert "single token" in str(exc.value)

    def test_a_template_the_tokenizer_disagrees_with_is_refused(self, tmp_path):
        predictor, _ = _predictor(tmp_path, [0.0, 1.0], encodes_marker=False)
        with pytest.raises(ArtifactUnavailableError) as exc:
            predictor.ask(["a"], [_CLEAN])
        assert "disagree about the marker token" in str(exc.value)

    def test_a_state_past_the_window_is_refused_rather_than_truncated(self, tmp_path):
        """The failure most likely to happen in practice.

        The questions render after the state, so truncation does not degrade
        the answer -- it eats the option markers off the end and leaves the last
        questions with nothing to score, which reads as a preference rather than
        as an overflow.
        """
        predictor, _ = _predictor(tmp_path, [0.0, 1.0], manifest=_manifest(max_len=8))
        with pytest.raises(ArtifactUnavailableError) as exc:
            predictor.ask([" ".join(["word"] * 200)], [_CLEAN])
        message = str(exc.value)
        assert "Shorten the state" in message
        assert "misreport" in message

    def test_a_graph_that_raises_names_the_graph(self, tmp_path):
        session = FakeSession(raises=RuntimeError("bad input shape"))
        predictor, _ = _predictor(tmp_path, session)
        with pytest.raises(ArtifactUnavailableError) as exc:
            predictor.ask(["a"], [_CLEAN])
        assert "failed to run" in str(exc.value)

    def test_an_absent_artifact_is_refused_at_the_first_question(self, tmp_path):
        predictor = OnnxDecisionPredictor(tmp_path / "nothing")
        with pytest.raises(ArtifactUnavailableError):
            predictor.ask(["a"], [_CLEAN])
