"""Watch's third tier decides whether to spend an LLM call, not what to say.

Laya generates no text. ``write_bundle`` persists ``evidence`` -- the log lines
Autopsy reads -- and the events file carries ``assessment``, the paragraph an
operator reads, so a tier that cannot write either cannot produce an alert. It
can produce the other verdict: a clean one quotes nothing, and
``sanitizer_assessment`` already established what that looks like
(``evidence="none"`` and one fixed sentence). So the tier is a gate in one
direction only, and every test here is about that asymmetry holding.

The second thing being pinned is that shadow mode is *shadow*. The comparison
that justifies turning the gate on has to be gathered from real traffic, and it
is only worth gathering if running it changed nothing -- which is a property
about control flow, so it is asserted against control flow rather than
described in a comment.

Nothing here loads weights. ``FakeLayaPredictor`` answers from a hash, which is
exactly what a threshold test wants and exactly what production must refuse; the
tier refuses it by name, and that refusal is tested too.
"""

from __future__ import annotations

import pytest
import yaml

from aorta.cia.watch.poll import should_alert
from aorta.cia.watch.watcher import (
    DEFAULT_CLEAN_THRESHOLD,
    LAYA_HEALTHY_QUESTION,
    LAYA_HEALTHY_WHEN_FALSE,
    LAYA_HEALTHY_WHEN_TRUE,
    LAYA_SIGNAL_QUESTION,
    WATCH_SIGNALS,
    LayaObservation,
    LogWatcher,
    gated_prediction,
    healthy_question,
    signal_question,
)
from aorta.laya.predictor import (
    ChoiceAnswer,
    FakeLayaPredictor,
    NoulAnswer,
)

_CONFIG = "src/aorta/cia/watch/watch_config.yaml"

#: A delta with nothing in it a deterministic scanner would object to. Several
#: tests need the veto *not* to fire so that the threshold is the only thing
#: deciding, and "step 41 loss 0.31" is what that looks like.
_ORDINARY = "=== train.log ===\nstep 41 loss 0.31 tok/s 8123\nstep 42 loss 0.30 tok/s 8140\n"


class _Pinned(FakeLayaPredictor):
    """The fake with p(healthy) pinned, which is the whole of a threshold test."""

    def __init__(self, clean: float, signal: str = "WATCH_HANG") -> None:
        super().__init__(
            pinned={
                healthy_question().question: NoulAnswer(probability=clean),
                signal_question().question: ChoiceAnswer(
                    probabilities=tuple(
                        (option, 1.0 if option == signal else 0.0)
                        for option in WATCH_SIGNALS
                    )
                ),
            }
        )


class _FakeReAct:
    """Stands in for the tier the gate exists to skip, and counts being reached."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        import dspy

        self.calls.append(kwargs)
        return dspy.Prediction(
            healthy=False,
            signal="WATCH_HANG",
            confidence=0.81,
            evidence="step 41 repeated 9 times",
            assessment="The job has printed the same step for four minutes.",
        )


def _watcher(predictor=None, **laya) -> tuple[LogWatcher, _FakeReAct]:
    """A watcher whose ReAct tier is a counter rather than a model.

    ``react`` is assigned rather than left None because building the real one
    calls ``ensure_configured()``, and a test that needs a provider to prove a
    provider was not needed would be proving the opposite of the point.
    """
    watcher = LogWatcher({"laya": laya}, predictor=predictor)
    react = _FakeReAct()
    watcher.react = react
    return watcher, react


def _assess(watcher: LogWatcher, content: str = _ORDINARY):
    return watcher.forward(
        new_content=content,
        job_context="job_id=cia-aaa node=node1 elapsed_sec=240",
        expectations="- loss should be decreasing",
    )


class TestTheGateOnlyEverSaysClean:
    def test_a_confident_clean_skips_the_llm_call(self):
        watcher, react = _watcher(_Pinned(0.97), enabled=True)
        pred = _assess(watcher)

        assert react.calls == []
        assert pred.healthy is True
        assert pred.signal == "WATCH_CLEAN"

    def test_at_the_threshold_counts_as_reaching_it(self):
        """``NoulAnswer.at`` is ``>=``, matching ``should_alert`` on the other side.

        Both sides of the poll loop have to mean the same thing by "at the
        threshold", or an operator who sets both to the same number gets two
        different behaviours out of one written value.
        """
        watcher, react = _watcher(_Pinned(0.90), enabled=True, clean_threshold=0.90)
        _assess(watcher)
        assert react.calls == []

    def test_just_below_the_threshold_falls_through(self):
        watcher, react = _watcher(_Pinned(0.8999), enabled=True, clean_threshold=0.90)
        _assess(watcher)
        assert len(react.calls) == 1

    @pytest.mark.parametrize("clean", [0.0, 0.01, 0.4, 0.75, 0.899])
    def test_an_unconfident_or_unhealthy_delta_always_reaches_react(self, clean):
        """The tier has no second branch, and that is structural, not a gap.

        A low p(healthy) is Laya saying the log looks wrong, and acting on that
        would mean alerting with no evidence to bundle and no assessment to
        read. So the only thing a low score buys is the assessment that would
        have happened anyway.
        """
        watcher, react = _watcher(_Pinned(clean), enabled=True, clean_threshold=0.90)
        pred = _assess(watcher)

        assert len(react.calls) == 1
        assert pred.signal == "WATCH_HANG", "the ReAct verdict must survive untouched"

    def test_a_gated_verdict_quotes_nothing_and_says_which_model(self):
        """``evidence`` is the field ``write_bundle`` persists, so it must be honest.

        'none' is the same answer ``sanitizer_assessment`` gives on its clean
        branch, and the assessment names the checkpoint because a verdict read
        on its own -- copied into a ticket -- otherwise cannot say what produced
        it (Decision 22, rule 2).
        """
        watcher, _ = _watcher(_Pinned(0.99), enabled=True)
        pred = _assess(watcher)

        assert pred.evidence == "none"
        assert "fake" in pred.assessment
        assert "describes the gate, not the log" in pred.assessment

    def test_the_reported_confidence_is_the_probability_not_certainty(self):
        """1.0 would be a lie, and the events file is where it would be read."""
        watcher, _ = _watcher(_Pinned(0.93), enabled=True)
        assert _assess(watcher).confidence == pytest.approx(0.93)

    @pytest.mark.parametrize("clean", [0.90, 0.95, 0.99, 1.0])
    def test_a_gated_verdict_can_never_escalate(self, clean):
        """The two thresholds gate opposite directions; this is where they meet.

        ``should_alert`` needs ``not healthy``, and the gate only ever emits
        healthy, so no value of ``clean_threshold`` can make the gate alert. If
        that ever stops holding, a tier with no evidence is writing bundles.
        """
        watcher, _ = _watcher(_Pinned(clean), enabled=True)
        pred = _assess(watcher)
        assert not should_alert(pred.healthy, pred.confidence, 0.70)


class TestShadowModeChangesNothing:
    def test_react_still_runs_and_its_verdict_is_returned_unchanged(self):
        watcher, react = _watcher(_Pinned(0.999), shadow=True)
        pred = _assess(watcher)

        assert len(react.calls) == 1
        assert (pred.healthy, pred.signal, pred.confidence) == (False, "WATCH_HANG", 0.81)
        assert pred.evidence == "step 41 repeated 9 times"

    def test_the_observation_travels_beside_the_verdict(self):
        """How ``poll.py`` gets something to write without this module knowing a path."""
        watcher, _ = _watcher(_Pinned(0.999, signal="WATCH_OOM"), shadow=True)
        observation = _assess(watcher).laya

        assert isinstance(observation, LayaObservation)
        assert observation.gated is False
        assert observation.clean_probability == pytest.approx(0.999)
        assert observation.signal == "WATCH_OOM"

    def test_shadow_alone_never_gates_however_sure_it_is(self):
        """The flag that produces the measurement is not the flag that acts on it."""
        watcher, react = _watcher(_Pinned(1.0), shadow=True, enabled=False)
        _assess(watcher)
        assert len(react.calls) == 1

    def test_a_gated_delta_still_records_what_was_observed(self):
        """Otherwise the gate's own traffic would be invisible to the comparison."""
        watcher, _ = _watcher(_Pinned(0.99), enabled=True)
        assert _assess(watcher).laya.gated is True

    def test_the_sanitizer_tier_still_wins_and_is_not_shadowed(self):
        """Laya sits after the regex, so it must not be measured on the regex's traffic.

        A shadow comparison gathered on deltas the gate will never see would
        describe a population the gate does not serve.
        """
        watcher, react = _watcher(_Pinned(0.01), shadow=True)
        pred = _assess(watcher, "[sanitizer] consan: verdict=pass state=ran findings=0\n")

        assert react.calls == []
        assert pred.signal == "WATCH_CLEAN"
        assert getattr(pred, "laya", None) is None


class TestTheNanVeto:
    """A false clean is worse here than the plan's "one poll interval" suggests.

    A healthy verdict commits the file cursor, so the gated bytes are never read
    again. A hang keeps printing and is caught next round; a single
    ``loss nan`` line printed once is gone. ``scan_stderr_text`` is the
    deterministic scanner Watch already trusts for that exact signature, so it
    gets a veto over the gate -- not over the verdict, which is still ReAct's to
    reach with evidence attached.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "step 12 loss nan",
            "step 12 loss=-inf",
            "NaN detected in gradients",
        ],
    )
    def test_a_non_finite_delta_is_never_gated(self, line):
        watcher, react = _watcher(_Pinned(1.0), enabled=True)
        pred = _assess(watcher, f"=== train.log ===\nstep 11 loss 0.4\n{line}\n")

        assert len(react.calls) == 1, "a confident clean skipped a printed NaN"
        assert pred.laya.vetoed is True
        assert pred.laya.gated is False

    def test_the_veto_is_recorded_rather_than_silent(self):
        """Counting these is how the shadow period finds out whether it matters."""
        watcher, _ = _watcher(_Pinned(0.99), shadow=True)
        pred = _assess(watcher, "=== train.log ===\nloss = nan\n")
        assert pred.laya.vetoed is True

    def test_a_log_reporting_no_nan_is_not_vetoed(self):
        """The scanner's own negation rule, inherited rather than re-implemented.

        "no non-finite values found" is a clean run saying so, and a veto that
        fired on it would make the gate unreachable for any job that reports its
        own health.
        """
        watcher, react = _watcher(_Pinned(0.99), enabled=True)
        pred = _assess(watcher, "=== train.log ===\nno nan detected this epoch\n")

        assert react.calls == []
        assert pred.laya.vetoed is False


class TestTheTierRefusesToGuess:
    def test_the_fake_backend_is_refused_by_name(self, capsys):
        """Everywhere else in the tree ``fake`` is the safe default. Not here.

        ``FakeLayaPredictor`` answers from a blake2b hash of the question and
        the log: stable, arbitrary, and indistinguishable from a model with an
        opinion. Behind the gate that is a coin flip deciding whether a log is
        read; behind shadow it is hash noise written to the events file for
        somebody to score later.
        """
        watcher, react = _watcher(None, enabled=True, backend="fake")
        pred = _assess(watcher)

        assert len(react.calls) == 1
        assert getattr(pred, "laya", None) is None
        assert "not a measurement" in capsys.readouterr().out

    def test_a_config_naming_no_checkpoint_is_refused_too(self, capsys):
        watcher, react = _watcher(None, shadow=True, backend="")
        _assess(watcher)

        assert len(react.calls) == 1
        assert "names no real checkpoint" in capsys.readouterr().out

    def test_an_unloadable_checkpoint_costs_the_tier_and_not_the_poll(self, capsys):
        """A node with no staged weights must still watch its jobs.

        The name resolves -- ``LayaAgentPredictor`` is built without touching a
        file, by design, so that a poll loop which never reaches an ambiguous
        log never pays for weights -- and the absence surfaces on the first
        question instead.
        """
        watcher, react = _watcher(None, enabled=True, backend="laya-typed-decisions")
        pred = _assess(watcher)

        assert len(react.calls) == 1
        assert pred.signal == "WATCH_HANG"
        assert "assessment continues without it" in capsys.readouterr().out

    def test_the_failure_is_not_retried_on_every_delta(self, capsys):
        """One LogWatcher serves the whole poll loop, so a retry here is per delta."""
        watcher, _ = _watcher(None, enabled=True, backend="laya-typed-decisions")
        for _ in range(4):
            _assess(watcher)

        assert capsys.readouterr().out.count("will not be retried this run") == 1

    def test_a_predictor_that_raises_mid_answer_is_survivable(self, capsys):
        class Exploding(FakeLayaPredictor):
            def ask(self, states, questions):
                raise RuntimeError("the checkpoint was unloaded under us")

        watcher, react = _watcher(Exploding(), enabled=True)
        pred = _assess(watcher)

        assert len(react.calls) == 1
        assert pred.signal == "WATCH_HANG"
        assert "assessment continues without it" in capsys.readouterr().out

    def test_a_predictor_answering_the_wrong_shape_is_not_read_as_a_number(self):
        """A choice where a noul was asked must not be coerced into a threshold."""

        class Wrong(FakeLayaPredictor):
            def ask(self, states, questions):
                return [
                    [ChoiceAnswer(probabilities=(("true", 1.0),)) for _ in questions]
                    for _ in states
                ]

        watcher, react = _watcher(Wrong(), enabled=True)
        _assess(watcher)
        assert len(react.calls) == 1


class TestItIsOffUntilSomethingHasBeenMeasured:
    def test_a_watcher_with_no_config_asks_nothing(self):
        """The default install must not reach for weights it was never given."""

        class Loud(FakeLayaPredictor):
            def ask(self, states, questions):
                raise AssertionError("the tier ran with no configuration")

        watcher, react = _watcher(Loud())
        _assess(watcher)
        assert len(react.calls) == 1

    def test_the_shipped_config_ships_the_gate_off(self, repo_root):
        config = yaml.safe_load((repo_root / _CONFIG).read_text(encoding="utf-8"))
        laya = config["watch"]["laya"]

        assert laya["enabled"] is False
        assert laya["shadow"] is False
        assert laya["shadow_archive_bytes"] == 0

    def test_the_shipped_threshold_is_the_documented_one(self, repo_root):
        """A default nobody can find is a default nobody can trust."""
        config = yaml.safe_load((repo_root / _CONFIG).read_text(encoding="utf-8"))
        assert config["watch"]["laya"]["clean_threshold"] == DEFAULT_CLEAN_THRESHOLD

    def test_the_two_thresholds_are_two_keys(self, repo_root):
        """They gate opposite directions, so one number serving both inverts one.

        ``confidence_threshold`` escalates when the assessment is unhealthy and
        at least that sure; ``clean_threshold`` stays quiet when the delta is
        healthy and at least that sure.
        """
        config = yaml.safe_load((repo_root / _CONFIG).read_text(encoding="utf-8"))

        assert "clean_threshold" in config["watch"]["laya"]
        assert config["watch"]["confidence_threshold"] == 0.70
        assert config["watch"]["laya"]["clean_threshold"] != 0.70

    def test_the_config_says_what_would_justify_turning_it_on(self, repo_root):
        """The flag is the easy half. The criteria are what stop it drifting on."""
        text = (repo_root / _CONFIG).read_text(encoding="utf-8")
        assert "false-clean rate" in text
        assert "temperature fit" in text

    def test_no_accuracy_is_claimed_anywhere_in_the_tier(self, repo_root):
        """Nothing here has been measured, so nothing here may report a number.

        The plan's Phase 1 has not run: there are no weights on this machine, no
        corpus with a healthy class, and no temperature fit. A docstring or a
        comment carrying an accuracy, an ECE or a latency would be read as one.
        """
        import inspect

        from aorta.cia.watch import watcher as module

        source = inspect.getsource(module).lower()
        for claim in ("accuracy of", "ece of", "% accurate", "ms per call"):
            assert claim not in source, f"the tier claims {claim!r}"


class TestTheCorpusLabelsTheQuestionThisTierAsks:
    """The train/serve check, made against what the builder actually emits.

    A temperature is fitted per question, so a question reworded on either side
    is calibrated against nothing while still returning a number. The wording
    now has one home -- this module, as it does in ``aorta/agent/llm.py`` for
    the proposer's questions and Autopsy's category -- and the corpus builder
    imports it.

    Asserted through ``build_watch_corpus`` rather than against the builder's
    module-level constants. Two reasons. A corpus is only as good as the
    question attached to each labelled example, and that is what these read.
    And the assertion survives the builder assembling its questions however it
    likes, so it does not have to be rewritten the next time that file moves
    something around -- which is the failure mode of the equality pin this
    replaces, a test that had to be kept in step with the thing it was
    guarding.
    """

    @staticmethod
    def _corpus(tmp_path):
        """The smallest job directory both halves of the builder will read.

        A bundle plus an Autopsy report gives the unhealthy pair, and a
        ``watchdog_ok`` event with an excerpt over the builder's forty-character
        floor gives the healthy one.
        """
        from aorta.laya.corpus.watch import build_watch_corpus

        job_dir = tmp_path / "cia-aaa"
        (job_dir / "bundle" / "logs").mkdir(parents=True)
        (job_dir / "job.json").write_text("{}", encoding="utf-8")
        (job_dir / "bundle" / "logs" / "watch.stderr.log").write_text(
            "=== train.log ===\nstep 12 loss nan\nstep 13 loss nan\n", encoding="utf-8"
        )
        (job_dir / "bundle" / "report.json").write_text(
            '{"category": "numeric_silent"}', encoding="utf-8"
        )
        (job_dir / "events.jsonl").write_text(
            '{"event_type": "watchdog_ok", "event_id": "e1", "signal": "WATCH_CLEAN", '
            '"excerpt": "step 41 loss 0.31 tok/s 8123 and nothing else of note here"}\n',
            encoding="utf-8",
        )
        return build_watch_corpus(tmp_path)

    def test_the_healthy_examples_carry_this_module_s_noul(self, tmp_path):
        examples = [
            e for e in self._corpus(tmp_path).examples if e.decision == "watch_healthy"
        ]

        assert examples, "the builder emitted no healthy examples to check"
        assert all(e.question == healthy_question() for e in examples)

    def test_the_signal_examples_carry_this_module_s_choice(self, tmp_path):
        examples = [
            e for e in self._corpus(tmp_path).examples if e.decision == "watch_signal"
        ]

        assert examples, "the builder emitted no signal examples to check"
        assert all(e.question == signal_question() for e in examples)

    def test_both_classes_are_labelled_against_it(self, tmp_path):
        """A drift check that only ever saw one class would miss half the corpus."""
        labels = {
            e.label for e in self._corpus(tmp_path).examples if e.decision == "watch_healthy"
        }
        assert labels == {"true", "false"}

    def test_the_answer_space_is_the_one_the_corpus_scores(self, tmp_path):
        """An option set is part of a question, and part of which bucket it lands in."""
        signal = [
            e for e in self._corpus(tmp_path).examples if e.decision == "watch_signal"
        ][0]
        assert signal.question.options == WATCH_SIGNALS

    def test_watch_clean_is_not_among_the_slugs(self):
        """It is the answer to the noul, and offering it twice lets one pass disagree."""
        assert "WATCH_CLEAN" not in WATCH_SIGNALS

    def test_no_second_copy_of_the_wording_grows_inside_cia(self):
        """A copy is how drift starts: two spellings stay identical until one improves.

        Scoped to ``src/aorta/cia/`` because that is this track's package. The
        same property across the tree -- in particular that
        ``aorta.laya.corpus.watch`` imports these rather than restating them --
        belongs to the CIA-wide question-uniqueness check being added alongside
        the log finder's questions, and asserting it from here would be a second
        copy of that test rather than a second guard.
        """
        import ast
        from pathlib import Path

        cia = Path(__file__).resolve().parents[2] / "src" / "aorta" / "cia"
        wanted = {
            LAYA_HEALTHY_QUESTION,
            LAYA_HEALTHY_WHEN_TRUE,
            LAYA_HEALTHY_WHEN_FALSE,
            LAYA_SIGNAL_QUESTION,
        }
        spellings: dict[str, list[str]] = {text: [] for text in wanted}
        for path in sorted(cia.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and node.value in wanted:
                    spellings[node.value].append(path.relative_to(cia).as_posix())

        assert all(
            places == ["watch/watcher.py"] for places in spellings.values()
        ), f"each question must be spelled in exactly one module: {spellings}"


class TestWhatTheModelIsShown:
    class _Recording(FakeLayaPredictor):
        def __init__(self):
            super().__init__()
            self.asked: list[tuple[tuple[str, ...], int]] = []

        def ask(self, states, questions):
            self.asked.append((tuple(states), len(questions)))
            return super().ask(states, questions)

    def test_the_state_is_the_delta_and_nothing_else(self):
        """Because the corpus's state is the delta and nothing else.

        ``aorta.laya.corpus.watch`` builds from ``bundle/logs/watch.stderr.log``,
        which is the delta alone. Feeding the runtime tier a richer state would
        fit a temperature against one input distribution and apply it to
        another. The cost is real: ``elapsed_sec`` is exactly what tells
        "stalled" from "still starting", and this tier does not get it.
        """
        predictor = self._Recording()
        watcher, _ = _watcher(predictor, shadow=True)
        _assess(watcher)

        assert predictor.asked[0][0] == (_ORDINARY,)

    def test_both_questions_ride_one_forward_pass(self):
        """The contract's own asymmetry: M questions over one state is one pass.

        N states would be N passes, which is why the tier scores the delta once
        rather than scoring it per watched file.
        """
        predictor = self._Recording()
        watcher, _ = _watcher(predictor, shadow=True)
        _assess(watcher)

        assert len(predictor.asked) == 1
        assert predictor.asked[0] == ((_ORDINARY,), 2)


class TestTheCalibrationCaveatTravelsWithTheNumber:
    """A probability is a function of the checkpoint *and* the temperature fit.

    Decision 22 asks a report to record both. ``model_id`` was always the first
    half; ``calibration()`` is the second, and the reason it has to ride along
    rather than be looked up later is that the looking-up happens somewhere
    else, months on, by someone scoring a shadow corpus. Laya 0.3.5 ships a
    ``choice:11+`` temperature below 1 -- it sharpens rather than softens, which
    publishes a 0.24 top probability as 0.99 -- and the library's own channel
    for saying the clamp fired is a ``RuntimeWarning`` at load, which no
    artifact keeps and ``PYTHONWARNINGS=ignore`` deletes.

    Neither of Watch's two questions is in that bucket today: the healthy noul
    is ``noul:2`` and the slug choice is ``choice:6-10``. That is a fact about
    laya 0.3.5's published table and not about the fine-tune Phase 1 exists to
    produce, which ships its own, so the disclosure is read off the loaded
    checkpoint every time rather than argued from the option count once.
    """

    def test_both_questions_sit_outside_the_bucket_laya_clamps(self):
        """Pinned, because the argument for the current wording partly rests on it.

        Widening the slug question past ten options would move it into
        ``choice:11+`` -- and the natural way that happens is somebody adding a
        slug, which looks like a vocabulary change rather than a calibration
        change.
        """
        from aorta.laya.predictor import bucket_for

        assert bucket_for(healthy_question()) == "noul:2"
        assert bucket_for(signal_question()) == "choice:6-10"

    def test_an_unknown_calibration_is_disclosed_rather_than_read_as_clean(self):
        """``FakeLayaPredictor`` reports unknown on purpose, and unknown is not fine."""
        watcher, _ = _watcher(_Pinned(0.99), shadow=True)
        observation = _assess(watcher).laya

        assert "CALIBRATION UNKNOWN" in observation.clean_caveat
        assert "CALIBRATION UNKNOWN" in observation.signal_caveat

    def test_a_clamped_bucket_names_the_temperature_that_was_refused(self):
        from aorta.laya.predictor import Calibration, ClampedBucket

        class Clamped(_Pinned):
            def calibration(self):
                return Calibration(
                    model_id="a-fine-tune",
                    clamped=(
                        ClampedBucket(bucket="noul:2", shipped=0.1006, applied=0.5),
                    ),
                    applied=(("noul:2", 0.5), ("choice:6-10", 1.2)),
                )

        watcher, _ = _watcher(Clamped(0.99), shadow=True)
        observation = _assess(watcher).laya

        assert "NOT CALIBRATED" in observation.clean_caveat
        assert "noul:2" in observation.clean_caveat
        assert observation.signal_caveat == "", "the other bucket was fine"

    def test_a_calibrated_checkpoint_says_nothing(self):
        """The argument against disclosing at all, answered by the API itself.

        ``caveat()`` returns "" where there is nothing to say, so the common
        path carries no noise and the decision costs a field that is usually
        empty.
        """
        from aorta.laya.predictor import Calibration

        class Calibrated(_Pinned):
            def calibration(self):
                return Calibration(model_id="a-fine-tune", applied=(("noul:2", 1.1),))

        watcher, _ = _watcher(Calibrated(0.99), shadow=True)
        observation = _assess(watcher).laya

        assert observation.clean_caveat == ""
        assert observation.signal_caveat == ""

    def test_a_gated_assessment_carries_the_caveat_where_a_person_reads_it(self):
        """Into the events file and out to a ticket, which is where it has to land."""
        watcher, _ = _watcher(_Pinned(0.99), enabled=True)
        assessment = _assess(watcher).assessment

        assert "CALIBRATION UNKNOWN" in assessment

    def test_a_gated_assessment_carries_only_the_question_it_acted_on(self):
        """The gate fired on the noul; the ``WATCH_CLEAN`` slug is not Laya's answer."""
        from aorta.laya.predictor import Calibration, ClampedBucket

        class SlugClamped(_Pinned):
            def calibration(self):
                return Calibration(
                    model_id="a-fine-tune",
                    clamped=(
                        ClampedBucket(bucket="choice:6-10", shipped=0.1, applied=0.5),
                    ),
                    applied=(("noul:2", 1.0),),
                )

        watcher, _ = _watcher(SlugClamped(0.99), enabled=True)
        pred = _assess(watcher)

        assert "NOT CALIBRATED" not in pred.assessment
        assert "NOT CALIBRATED" in pred.laya.signal_caveat

    def test_a_predictor_that_cannot_say_costs_the_tier(self):
        """Not a verdict with the disclosure quietly missing, which is the failure itself."""

        class Mute(_Pinned):
            def calibration(self):
                raise RuntimeError("the table could not be read")

        watcher, react = _watcher(Mute(0.99), enabled=True)
        pred = _assess(watcher)

        assert len(react.calls) == 1, "a verdict was gated with no calibration known"
        assert getattr(pred, "laya", None) is None

    def test_the_caveats_reach_the_event_payload(self):
        observation = LayaObservation(
            model_id="m",
            clean_probability=0.9,
            clean_threshold=0.9,
            gated=True,
            vetoed=False,
            signal="WATCH_HANG",
            signal_probability=0.4,
            clean_caveat="; NOT CALIBRATED: ...",
        )
        fields = observation.as_event_fields()

        assert fields["clean_caveat"] == "; NOT CALIBRATED: ..."
        # Emitted even when empty, so "nothing to disclose" and "predates the
        # disclosure" are different rows rather than the same absent key.
        assert fields["signal_caveat"] == ""


class TestTheObservationCarriesItsProvenance:
    def test_the_event_fields_name_the_checkpoint(self):
        """Rule 2 of Decision 22: a verdict that cannot say which weights produced
        it is a verdict nobody can compare against the next one."""
        observation = LayaObservation(
            model_id="laya-typed-decisions@cpu",
            clean_probability=0.9312345,
            clean_threshold=0.9,
            gated=True,
            vetoed=False,
            signal="WATCH_HANG",
            signal_probability=0.4,
        )
        fields = observation.as_event_fields()

        assert fields["model_id"] == "laya-typed-decisions@cpu"
        assert fields["clean_probability"] == 0.9312
        assert fields["gated"] is True

    def test_a_gated_prediction_carries_the_observation_for_the_events_file(self):
        observation = LayaObservation(
            model_id="m",
            clean_probability=0.95,
            clean_threshold=0.9,
            gated=True,
            vetoed=False,
            signal="WATCH_HANG",
            signal_probability=0.4,
        )
        assert gated_prediction(observation).laya is observation


def test_importing_watch_pulls_in_no_model_machinery():
    """The property the lazy imports exist to protect, asserted directly.

    ``aorta.cia.watch.poll`` is imported by every Watch run and reaches the
    predictor seam through this tier. A module-scope ``import laya`` anywhere on
    that path would put torch behind ``aorta watchdog poll`` on nodes that never
    turn the tier on, which is what Decision 22 in ``docs/laya-packaging.md``
    asks it not to do.

    Measured in a subprocess, because ``sys.modules`` is already dirty by the
    time pytest runs this.
    """
    import json
    import subprocess
    import sys

    heavy = ("torch", "transformers", "laya", "onnxruntime", "safetensors")
    probe = (
        "import sys, json, aorta.cia.watch.poll, aorta.cia.watch.watcher;"
        f"heavy={heavy!r};"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in heavy)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == [], (
        f"importing Watch pulled in {out.stdout.strip()}; the Laya imports must "
        "stay inside the tier that loads a checkpoint."
    )
