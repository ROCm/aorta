"""Autopsy's confidence has always had more than one source. Now it says which.

The escalation cutoff in ``run_autopsy`` -- ``confidence < 0.85`` -- was chosen
against a scale somebody authored: ``merge_watchdog_matrix`` hard-codes 0.62 for
a bare watchdog NaN, and the router's prompt asks a model in English for the
same ``~0.62``. Both ends of that comparison were written together, which is why
a bare number in an ``if`` was good enough.

Phase 4 puts a calibrated encoder behind the same field, and a Laya probability
is a different distribution with the same name and the same range. Nothing in
this repository has measured it: there are no weights on this machine, Phase 1
has not run, and no temperature fit exists. So the tests here are not about
whether the encoder is right. They are about the one thing that can be checked
without a measurement -- that a cutoff derived against one source is never
quietly applied to another, and that a report says which source it had.

Nothing here loads weights. ``FakeLayaPredictor`` answers from a hash, which is
what a wiring test wants and what production must refuse; the tier refuses it by
name, and that refusal is tested too.
"""

from __future__ import annotations

import json

import pytest

from aorta.agent.llm import AUTOPSY_CATEGORIES, LAYA_CATEGORY_QUESTION
from aorta.cia.autopsy import escalation, router
from aorta.cia.autopsy.escalation import (
    ADAPTER_RULES,
    ESCALATION_THRESHOLD,
    LAYA,
    LLM_SELF_REPORT,
    SWEEP_PROBE,
    Confidence,
    UncalibratedThreshold,
)
from aorta.laya.predictor import ChoiceAnswer, FakeLayaPredictor, NoulAnswer


def _laya(value: float, detail: str = "laya-typed-decisions") -> Confidence:
    return Confidence(value, LAYA, detail)


def _adapters(value: float) -> Confidence:
    return Confidence(value, ADAPTER_RULES)


def _decide(reported: Confidence, rule_based: Confidence, **kwargs):
    return escalation.decide(SWEEP_PROBE, reported, rule_based=rule_based, **kwargs)


class TestTheCutoffStillMeansWhatItMeant:
    """The pre-Laya behaviour, pinned before anything is allowed to change it."""

    def test_the_cutoff_is_unchanged(self):
        assert ESCALATION_THRESHOLD == 0.85

    def test_it_is_registered_for_the_two_sources_it_was_chosen_against(self):
        """Both are the same authored scale; a third source is not on it."""
        assert escalation.THRESHOLD_SOURCES == {ADAPTER_RULES, LLM_SELF_REPORT}
        assert LAYA not in escalation.THRESHOLD_SOURCES

    @pytest.mark.parametrize("source", [ADAPTER_RULES, LLM_SELF_REPORT])
    @pytest.mark.parametrize(
        "value,escalates", [(0.0, True), (0.62, True), (0.8499, True), (0.85, False), (0.95, False)]
    )
    def test_below_the_cutoff_escalates_and_at_it_does_not(self, source, value, escalates):
        """``<``, not ``<=``: at the threshold is confident enough, as before."""
        reported = Confidence(value, source)
        assert _decide(reported, _adapters(value)).escalate is escalates

    def test_no_recommended_sweep_means_no_escalation_however_low(self):
        """The router's own recommendation is still the necessary condition."""
        decision = escalation.decide(
            "none", Confidence(0.0, LLM_SELF_REPORT), rule_based=_adapters(0.0)
        )
        assert decision.escalate is False
        assert "nothing to escalate" in decision.reason


class TestALayaProbabilityDoesNotInheritTheCutoff:
    """The single edit that could make this system worse, refused structurally.

    A calibrated probability and an LLM's self-report are both floats in [0, 1]
    and both read as plausible in a report. If 0.85 were carried across, the only
    symptom would be four-hour sweeps that stop happening, or four-hour sweeps
    that should not -- against a job that was already failing, so nobody would
    attribute either to a threshold.
    """

    def test_a_confident_laya_answer_cannot_suppress_a_sweep(self):
        """0.99 clears 0.85 on the old scale. It is not on the old scale."""
        decision = _decide(_laya(0.99), _adapters(0.62))

        assert decision.escalate is True
        assert decision.gated_on.source == ADAPTER_RULES
        assert decision.gated_on.value == 0.62

    def test_an_unconfident_laya_answer_cannot_cause_one_either(self):
        """The error is symmetric, so refusing the cutoff has to be too."""
        decision = _decide(_laya(0.10), _adapters(0.95))

        assert decision.escalate is False
        assert decision.gated_on.source == ADAPTER_RULES

    def test_the_control_flow_is_the_one_autopsy_already_had(self):
        """Turning the tier on changes what the report says, not what runs.

        The category is what Phase 4 set out to improve; the sweep is what a
        wrong threshold would break. Those are separable, and separating them is
        what makes the tier safe to ship before the measurement exists.
        """
        for value in (0.0, 0.5, 1.0):
            with_laya = _decide(_laya(value), _adapters(0.62))
            without = _decide(_adapters(0.62), _adapters(0.62))
            assert with_laya.escalate == without.escalate

    def test_the_report_can_tell_that_the_two_numbers_differ(self):
        """A reader told only the outcome cannot audit this; one told only the
        reported confidence would audit it wrongly."""
        fields = _decide(_laya(0.99), _adapters(0.62)).as_report_fields()

        assert fields["gated_on"] == {"source": ADAPTER_RULES, "value": 0.62}
        assert fields["threshold"] == ESCALATION_THRESHOLD
        assert "no cutoff has been derived" in fields["reason"]


class TestADerivedCutoffIsAccepted:
    """Phase 1's output has somewhere to land, rather than arriving as a patch."""

    @pytest.mark.parametrize("value,escalates", [(0.40, True), (0.55, False), (0.90, False)])
    def test_an_operator_supplied_threshold_gates_the_laya_number(self, value, escalates):
        decision = _decide(_laya(value), _adapters(0.95), laya_threshold=0.55)

        assert decision.escalate is escalates
        assert decision.gated_on.source == LAYA
        assert decision.threshold == 0.55

    def test_the_checkpoint_that_produced_it_is_named_in_the_reason(self):
        """Rule 2 of Decision 22: a threshold is derived against *a* checkpoint."""
        decision = _decide(_laya(0.4, "fine-tune-2026-09"), _adapters(0.95), laya_threshold=0.55)

        assert "fine-tune-2026-09" in decision.reason


class TestAnUnknownSourceIsNotGuessedAt:
    def test_a_source_with_no_registered_cutoff_raises(self):
        """Picking the nearest cutoff and carrying on is the failure being prevented."""
        with pytest.raises(UncalibratedThreshold, match="no escalation threshold"):
            _decide(Confidence(0.5, "some_future_head"), _adapters(0.62))

    def test_a_fallback_that_is_not_on_the_cutoff_s_scale_raises_too(self):
        """The fallback is only safe because of what it is, not because it is second."""
        with pytest.raises(UncalibratedThreshold, match="nothing left to gate on"):
            _decide(_laya(0.9), Confidence(0.62, "some_future_head"))


class TestNothingClaimsAMeasurement:
    """Phase 1 has not run. Anything numeric here would be read as its result."""

    @pytest.mark.parametrize("module", [escalation, router])
    def test_no_accuracy_or_calibration_figure_is_stated(self, module):
        import inspect

        source = inspect.getsource(module).lower()
        for claim in ("accuracy of", "ece of", "% accurate", "ms per call", "brier of"):
            assert claim not in source, f"{module.__name__} claims {claim!r}"


# ── the tier ───────────────────────────────────────────────────────────────


def _pinned(category: str, probability: float = 0.77) -> FakeLayaPredictor:
    """The fake with the category answer pinned, which is the whole of a wiring test."""
    others = sorted(AUTOPSY_CATEGORIES - {category})
    spread = (1.0 - probability) / len(others)
    return FakeLayaPredictor(
        pinned={
            LAYA_CATEGORY_QUESTION: ChoiceAnswer(
                probabilities=((category, probability),) + tuple((o, spread) for o in others)
            )
        }
    )


class _Clamped(FakeLayaPredictor):
    """A checkpoint that clamped ``choice:11+``, which is the shipped default's state.

    The numbers are the library's published ones for that bucket rather than
    anything measured here; they are what a caveat quotes, not a claim about how
    well the model performs.
    """

    def __init__(self, category: str, probability: float = 0.91) -> None:
        super().__init__(pinned=_pinned(category, probability)._pinned)

    def calibration(self):
        from aorta.laya.predictor import Calibration, ClampedBucket

        return Calibration(
            model_id=self.model_id(),
            clamped=(ClampedBucket(bucket="choice:11+", shipped=0.1006, applied=0.5),),
            applied=(("choice:11+", 0.5), ("noul:2", 1.4)),
        )


class _Calibrated(FakeLayaPredictor):
    """A checkpoint whose fit for this bucket survived, so there is nothing to say."""

    def __init__(self, category: str, probability: float = 0.91) -> None:
        super().__init__(pinned=_pinned(category, probability)._pinned)

    def calibration(self):
        from aorta.laya.predictor import Calibration

        return Calibration(
            model_id=self.model_id(), applied=(("choice:11+", 1.7), ("noul:2", 1.4))
        )


class _FakeReAct:
    """Stands in for the model, and records which signature it was built with."""

    built: list = []

    def __init__(self, signature, tools=None, max_iters=0):
        self.signature = signature
        self.calls: list[dict] = []
        _FakeReAct.built.append(self)

    def set_lm(self, lm):
        self.lm = lm

    def __call__(self, **kwargs):
        import dspy

        self.calls.append(kwargs)
        fields = {
            "rationale": "SAN_CONSAN_RACE in aorta/sanitizer_report.json",
            "next_probe": "none",
            "next_probe_reason": "",
        }
        if "category" in self.signature.output_fields:
            fields["category"] = "illegal_mem"
            fields["confidence"] = 0.41
        return dspy.Prediction(**fields)


@pytest.fixture
def fake_react(monkeypatch):
    """Every ReAct the router builds, in construction order."""
    _FakeReAct.built = []
    monkeypatch.setattr(router.dspy, "ReAct", _FakeReAct)
    return _FakeReAct.built


_EVIDENCE = [
    {"uri": "aorta/sanitizer_report.json", "signal": "SAN_CONSAN_RACE", "adapter": "sanitizer"},
]


def _route(fake_react, predictor=None, **laya):
    built = router.TriageRouter("/tmp", laya=laya, predictor=predictor)
    return built(evidence=_EVIDENCE, job_context="job_id=cia-aaa")


class TestTheTierIsOffUntilSomethingHasBeenMeasured:
    def test_a_router_with_no_configuration_asks_nothing(self, fake_react):
        class Loud(FakeLayaPredictor):
            def ask(self, states, questions):
                raise AssertionError("the tier ran with no configuration")

        prediction = _route(fake_react, Loud())

        assert prediction.category == "illegal_mem"
        assert getattr(prediction, "laya", None) is None

    def test_the_environment_ships_it_off(self, monkeypatch):
        """Default off: the measurement that would justify enabling it has not happened."""
        for name in (
            router.LAYA_ENABLED_ENV,
            router.LAYA_BACKEND_ENV,
            router.LAYA_ESCALATION_THRESHOLD_ENV,
        ):
            monkeypatch.delenv(name, raising=False)

        assert router.laya_config_from_env()["enabled"] is False

    def test_no_escalation_threshold_is_defaulted(self, monkeypatch):
        """The absence is the design. A number here would be one invented to fill a slot."""
        monkeypatch.delenv(router.LAYA_ESCALATION_THRESHOLD_ENV, raising=False)

        assert "escalation_threshold" not in router.laya_config_from_env()
        assert router._LayaCategoryTier({}).escalation_threshold is None

    def test_an_unparseable_threshold_is_refused_out_loud(self, monkeypatch, capsys):
        """Silently ignoring it would let an operator read the report as though
        their cutoff had applied."""
        monkeypatch.setenv(router.LAYA_ESCALATION_THRESHOLD_ENV, "point five")

        assert "escalation_threshold" not in router.laya_config_from_env()
        assert "is not a number" in capsys.readouterr().out


class TestTheEncoderAnswersTheClassification:
    def test_the_category_and_confidence_come_from_the_tier(self, fake_react):
        prediction = _route(fake_react, _pinned("gpu_race", 0.88), enabled=True)

        assert prediction.category == "gpu_race"
        assert prediction.confidence == pytest.approx(0.88)

    def test_the_rationale_and_next_probe_stay_on_the_model(self, fake_react):
        """Laya emits no tokens, so there is nothing to ask it for."""
        prediction = _route(fake_react, _pinned("gpu_race"), enabled=True)

        assert prediction.rationale.startswith("SAN_CONSAN_RACE")
        assert prediction.next_probe == "none"

    def test_the_model_is_never_asked_for_a_category_or_a_confidence(self, fake_react):
        """The only way the hand-fitted curve genuinely leaves the prompt.

        Dropping the four constants while keeping the field would leave a model
        asked for a number with no rule for producing one, which is worse than
        the rule it lost.
        """
        _route(fake_react, _pinned("gpu_race"), enabled=True)
        used = fake_react[-1].signature

        assert "category" not in used.output_fields
        assert "confidence" not in used.output_fields
        assert set(used.output_fields) == {"rationale", "next_probe", "next_probe_reason"}

    def test_the_prompt_the_model_sees_carries_no_calibration_constants(self):
        """The four numbers were a calibration curve written in English.

        Checked as "no decimal anywhere" rather than as the four literals,
        because the next one to be added would not be one of the four.
        """
        import re

        instructions = router.TriageDecisionWithAssignedCategory.instructions

        assert not re.findall(r"\d+\.\d+", instructions)
        assert "not asked for a category or for a confidence" in instructions

    def test_the_llm_signature_still_carries_them(self):
        """The default path is untouched, so the constants must still be there.

        Stripping them from the prompt the model is still asked to put a number
        into would degrade the default install for a feature it has not enabled.
        """
        instructions = router.TriageDecision.instructions

        assert "0.62" in instructions
        assert "confidence" in router.TriageDecision.output_fields

    def test_the_assigned_category_reaches_the_prompt(self, fake_react):
        """Otherwise the rationale argues for one category while the report carries another."""
        _route(fake_react, _pinned("gpu_race"), enabled=True)

        assert fake_react[-1].calls[0]["assigned_category"] == "gpu_race"

    def test_the_model_is_told_it_may_disagree(self):
        """A rationale that must justify the verdict is a rationale that will."""
        assert "say so" in router.TriageDecisionWithAssignedCategory.instructions

    def test_the_observation_travels_for_the_report_to_record(self, fake_react):
        prediction = _route(fake_react, _pinned("gpu_race", 0.88), enabled=True)
        fields = prediction.laya.as_report_fields()

        assert fields["model_id"] == "fake"
        assert fields["category"] == "gpu_race"
        assert fields["probability"] == pytest.approx(0.88)
        assert fields["question"] == LAYA_CATEGORY_QUESTION
        assert fields["options"] == len(AUTOPSY_CATEGORIES)
        assert fields["escalation_threshold"] is None


class TestTheWorstBucketSaysSo:
    """This surface's question is the one over the line, so this is the common case.

    ``AUTOPSY_CATEGORIES`` has eleven members, which puts the category choice in
    ``choice:11+`` -- the bucket whose shipped temperature is below 1 and
    therefore sharpens logits rather than softening them, publishing a low top
    probability as a high one. Laya 0.3.5 clamps it and warns; the clamp stops
    the sharpening and does not make the bucket calibrated. Watch's slug choice
    is seven options and the probe agent's category choice is eight, so this is
    the one place in the integration where the disclosure is expected rather
    than exceptional.
    """

    def test_the_question_is_in_the_bucket_the_library_distrusts(self):
        """Asserted rather than assumed, because the set can widen under us.

        The plan said nine. It is eleven, and either number is one side of a
        boundary that decides whether anything is disclosed at all.
        """
        from aorta.laya.predictor import bucket_for

        assert bucket_for(router.category_question()) == "choice:11+"

    def test_the_bucket_is_named_even_though_no_checkpoint_loaded(self, fake_react):
        """``bucket_for`` is pure, which is why half the disclosure survives a
        predictor that never reached any weights."""
        observation = _route(fake_react, _pinned("gpu_race"), enabled=True).laya

        assert observation.bucket == "choice:11+"

    def test_an_unreadable_calibration_is_unknown_and_not_clean(self, fake_react):
        """``FakeLayaPredictor`` has no temperature table, so there is nothing
        to report -- and nothing is a third state, not a pass."""
        observation = _route(fake_react, _pinned("gpu_race"), enabled=True).laya

        assert observation.clamped is None
        assert "CALIBRATION UNKNOWN" in observation.caveat

    def test_a_clamped_bucket_names_both_temperatures(self, fake_react):
        observation = _route(fake_react, _Clamped("gpu_race"), enabled=True).laya

        assert observation.clamped is True
        assert "NOT CALIBRATED" in observation.caveat
        assert "choice:11+" in observation.caveat

    def test_a_clean_bucket_says_nothing(self, fake_react):
        """``caveat()`` returns "" where there is nothing to say, so a fitted
        checkpoint does not paste a reassurance into every report."""
        observation = _route(fake_react, _Calibrated("gpu_race"), enabled=True).laya

        assert observation.clamped is False
        assert observation.caveat == ""

    def test_the_caveat_is_asked_of_the_question_not_of_an_option_count(self):
        """Track B derived one caveat from a candidate count and it could say
        nothing about its stop noul. There is one question here today, so the
        two agree -- which is exactly when the shortcut looks safe.
        """
        import inspect

        code = [
            line
            for line in inspect.getsource(router._LayaCategoryTier).splitlines()
            if not line.strip().startswith("#")
        ]
        source = "\n".join(code)

        assert "calibration.caveat(question)" in source
        assert "AUTOPSY_CATEGORIES" not in source

    def test_the_calibration_is_read_after_the_forward_pass(self):
        """``calibration()`` does not trigger a load, so asking it beside
        ``model_id()`` would report "not loaded yet" and stamp CALIBRATION
        UNKNOWN onto a report whose calibration was perfectly knowable.
        """
        import inspect

        source = inspect.getsource(router._LayaCategoryTier.observe)

        assert source.index("ask_one(") < source.index("predictor.calibration()")

    def test_a_predictor_that_cannot_report_its_calibration_costs_the_tier(
        self, fake_react, capsys
    ):
        """The direction matters. Losing a good classification to a broken
        accessor is the smaller harm; the other way round writes a probability
        into report.json with its calibration unexamined.
        """

        class Mute(FakeLayaPredictor):
            def calibration(self):
                raise RuntimeError("the temperature table could not be read")

        prediction = _route(fake_react, Mute(), enabled=True)

        assert prediction.category == "illegal_mem"
        assert getattr(prediction, "laya", None) is None
        assert "the router classifies as before" in capsys.readouterr().out


class TestWhatTheModelAndTheEncoderAreShown:
    class _Recording(FakeLayaPredictor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.asked: list[tuple[tuple[str, ...], int]] = []

        def ask(self, states, questions):
            self.asked.append((tuple(states), len(questions)))
            return super().ask(states, questions)

    def test_the_state_is_the_same_string_the_model_is_handed(self, fake_react):
        """They differ in model, not in input, which is what makes a later
        comparison between them a comparison rather than two measurements."""
        predictor = self._Recording()
        _route(fake_react, predictor, enabled=True)

        assert predictor.asked[0][0] == (json.dumps(_EVIDENCE),)
        assert fake_react[-1].calls[0]["evidence_json"] == json.dumps(_EVIDENCE)

    def test_one_question_and_one_forward_pass(self, fake_react):
        predictor = self._Recording()
        _route(fake_react, predictor, enabled=True)

        assert predictor.asked == [((json.dumps(_EVIDENCE),), 1)]


class TestTheQuestionHasOneDefinition:
    """A temperature is fitted per question, so the wording is part of the fit.

    Track B found the corpus builder and the inference path had independently
    phrased the same question. That does not fail loudly -- it answers slightly
    worse, for a reason nobody would look for. No corpus labels this decision
    yet, so there is nothing to pin against from the other side; importing the
    one public definition is how the builder that eventually does find it.
    """

    def test_the_wording_is_the_shared_constant(self):
        assert router.category_question().question == LAYA_CATEGORY_QUESTION

    def test_this_module_does_not_spell_the_question_a_second_time(self):
        import inspect

        source = inspect.getsource(router)
        assert source.count(f'"{LAYA_CATEGORY_QUESTION}"') == 0

    def test_the_answer_space_is_the_whole_shared_vocabulary(self):
        """Autopsy reaches every category; the probe agent reaches a subset."""
        assert set(router.category_question().options) == AUTOPSY_CATEGORIES

    def test_the_options_are_ordered_so_two_runs_ask_the_same_question(self):
        assert router.category_question().options == tuple(sorted(AUTOPSY_CATEGORIES))

    def test_no_option_is_glossed(self):
        """A gloss here would make this a different question from the proposer's
        while looking like the same one."""
        assert router.category_question().criteria == ()


class TestTheTierRefusesToGuess:
    def test_the_fake_backend_is_refused_by_name(self, fake_react, capsys):
        """A hashed category would reach report.json, and from there the corpus
        builders that read report.json as ground truth."""
        prediction = _route(fake_react, None, enabled=True, backend="fake")

        assert getattr(prediction, "laya", None) is None
        assert prediction.category == "illegal_mem"
        assert "not a measurement" in capsys.readouterr().out

    def test_a_config_naming_no_checkpoint_is_refused_too(self, fake_react, capsys):
        _route(fake_react, None, enabled=True, backend="")

        assert "no real checkpoint is named" in capsys.readouterr().out

    def test_an_unstaged_checkpoint_costs_the_tier_and_not_the_autopsy(self, fake_react, capsys):
        """A node with no weights must still produce a report.

        The name resolves -- ``LayaAgentPredictor`` is built without touching a
        file, by design -- and the absence surfaces on the first question.
        """
        prediction = _route(fake_react, None, enabled=True, backend="laya-typed-decisions")

        assert prediction.category == "illegal_mem"
        assert prediction.confidence == pytest.approx(0.41)
        assert "the router classifies as before" in capsys.readouterr().out

    def test_a_predictor_answering_the_wrong_shape_is_not_read_as_a_category(
        self, fake_react, capsys
    ):
        class Wrong(FakeLayaPredictor):
            def ask(self, states, questions):
                return [[NoulAnswer(probability=1.0) for _ in questions] for _ in states]

        prediction = _route(fake_react, Wrong(), enabled=True)

        assert prediction.category == "illegal_mem"
        assert "the router classifies as before" in capsys.readouterr().out

    def test_the_fallback_is_the_prompt_that_still_asks_for_a_category(self, fake_react, capsys):
        """Degrading has to mean the behaviour from before the flag, not a gap."""
        _route(fake_react, None, enabled=True, backend="laya-typed-decisions")
        capsys.readouterr()

        assert [r.signature for r in fake_react] == [router.TriageDecision]

    def test_the_guard_on_the_vocabulary_survives(self, fake_react):
        """``coerce_category`` stays, and is no longer the only thing standing
        between the report and an unhandled label -- an option-marker head can
        only return a label it was offered. What it still catches is the option
        set drifting away from AUTOPSY_CATEGORIES.
        """

        class OffVocabulary(FakeLayaPredictor):
            def ask(self, states, questions):
                return [[ChoiceAnswer(probabilities=(("probably_a_race", 1.0),))] for _ in states]

        assert _route(fake_react, OffVocabulary(), enabled=True).category == "unknown"


# ── what the report carries ────────────────────────────────────────────────


class _StubRouter:
    """A router that answers from a Laya tier, without dspy or a model."""

    prediction = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, **kwargs):
        return _StubRouter.prediction


@pytest.fixture
def evidenced_bundle(make_bundle):
    return make_bundle(stderr="step 5 loss=nan\nnon-finite loss, aborting run\n")


def _report(bundle, monkeypatch, prediction):
    from aorta.cia.autopsy import orchestrator

    _StubRouter.prediction = prediction
    monkeypatch.setattr("aorta.cia.autopsy.router.TriageRouter", _StubRouter)
    return orchestrator.run_autopsy(bundle, kb_version="test")


def _prediction(**fields):
    import dspy

    base = {
        "category": "numeric_silent",
        "confidence": 0.62,
        "rationale": "WATCH_NUMERIC_NAN in logs/watch.stderr.log",
        "next_probe": "aorta sweep run",
        "next_probe_reason": "need a production matrix",
    }
    base.update(fields)
    return dspy.Prediction(**base)


class TestTheReportSaysWhereItsConfidenceCameFrom:
    def test_an_llm_verdict_is_recorded_as_a_self_report(self, evidenced_bundle, monkeypatch):
        """Nothing trained that float to be a probability, and the report now says so."""
        report = _report(evidenced_bundle, monkeypatch, _prediction())

        assert report["confidence_source"] == {"source": LLM_SELF_REPORT, "value": 0.62}
        assert "laya" not in report

    def test_a_degraded_verdict_is_recorded_as_the_adapters(self, evidenced_bundle, monkeypatch):
        """Autopsy has always had two sources here. Only one of them was visible."""
        from aorta.cia.autopsy import orchestrator

        class _Unreachable:
            def __init__(self, *args, **kwargs):
                raise ConnectionError("connection refused")

        monkeypatch.setattr("aorta.cia.autopsy.router.TriageRouter", _Unreachable)
        report = orchestrator.run_autopsy(evidenced_bundle, kb_version="test")

        assert report["confidence_source"]["source"] == ADAPTER_RULES

    def test_a_laya_verdict_names_the_checkpoint_inline(self, evidenced_bundle, monkeypatch):
        """Rule 2 of Decision 22: a report is copied into a ticket and read alone."""
        observation = router.LayaCategory(
            model_id="laya-typed-decisions@cpu",
            category="numeric_silent",
            probability=0.71,
            options=len(AUTOPSY_CATEGORIES),
            escalation_threshold=None,
        )
        report = _report(
            evidenced_bundle, monkeypatch, _prediction(confidence=0.71, laya=observation)
        )

        assert report["confidence_source"] == {
            "source": LAYA,
            "value": 0.71,
            "detail": "laya-typed-decisions@cpu",
        }
        assert report["laya"]["model_id"] == "laya-typed-decisions@cpu"
        assert report["laya"]["question"] == LAYA_CATEGORY_QUESTION

    def test_the_escalation_block_says_what_it_gated_on(self, evidenced_bundle, monkeypatch):
        """The reported confidence and the gating one are allowed to differ, and
        a report that does not say so cannot be audited."""
        observation = router.LayaCategory(
            model_id="laya-typed-decisions",
            category="numeric_silent",
            probability=0.99,
            options=len(AUTOPSY_CATEGORIES),
            escalation_threshold=None,
        )
        report = _report(
            evidenced_bundle, monkeypatch, _prediction(confidence=0.99, laya=observation)
        )

        assert report["confidence"] == 0.99
        assert report["escalation"]["gated_on"]["source"] == ADAPTER_RULES
        assert report["escalation"]["threshold"] == ESCALATION_THRESHOLD

    def test_no_sweep_recommended_is_recorded_rather_than_left_blank(
        self, evidenced_bundle, monkeypatch
    ):
        report = _report(evidenced_bundle, monkeypatch, _prediction(next_probe="none"))

        assert report["escalation"]["escalated"] is False
        assert "nothing to escalate" in report["escalation"]["reason"]


def _observed(**fields) -> router.LayaCategory:
    base = {
        "model_id": "laya-typed-decisions",
        "category": "numeric_silent",
        "probability": 0.91,
        "options": len(AUTOPSY_CATEGORIES),
        "escalation_threshold": None,
        "bucket": "choice:11+",
        "clamped": True,
        "caveat": (
            "; NOT CALIBRATED: laya-typed-decisions ships temperature 0.1006 for "
            "choice:11+, which laya clamped to 0.5"
        ),
    }
    base.update(fields)
    return router.LayaCategory(**base)


class TestTheCaveatReachesTheArtifactAndNotOnlyTheTerminal:
    """The whole point of disclosing it in the report rather than in a warning.

    The library's own channel for this is a ``RuntimeWarning`` raised once at
    load. It is deleted outright by ``PYTHONWARNINGS=ignore``, which is what a
    CI wrapper or a job launcher sets, and it reaches no file. ``report.json``
    is the artifact that gets copied into a ticket and read a week later, so a
    disclosure that is not in it is a disclosure nobody gets.
    """

    def _laya_report(self, bundle, monkeypatch, observation):
        return _report(
            bundle,
            monkeypatch,
            _prediction(confidence=observation.probability, laya=observation),
        )

    def test_the_rationale_a_person_reads_carries_it(self, evidenced_bundle, monkeypatch):
        report = self._laya_report(evidenced_bundle, monkeypatch, _observed())

        assert "NOT CALIBRATED" in report["rationale"]
        assert "choice:11+" in report["rationale"]

    def test_a_program_gets_a_field_rather_than_a_substring(
        self, evidenced_bundle, monkeypatch
    ):
        """A consumer grepping prose for "NOT CALIBRATED" is one rewording away
        from disagreeing with the thing that wrote it."""
        report = self._laya_report(evidenced_bundle, monkeypatch, _observed())

        assert report["laya"]["bucket"] == "choice:11+"
        assert report["laya"]["clamped"] is True
        assert report["confidence_source"]["caveat"].startswith("; NOT CALIBRATED")

    def test_unknown_survives_the_round_trip_as_null_rather_than_false(
        self, evidenced_bundle, monkeypatch
    ):
        """``None`` means nobody asked the checkpoint, and a reader collapsing
        it to "not clamped" reads the worst case as the best one."""
        observation = _observed(clamped=None, caveat="; CALIBRATION UNKNOWN: no weights")
        report = self._laya_report(evidenced_bundle, monkeypatch, observation)

        assert json.loads(json.dumps(report))["laya"]["clamped"] is None
        assert "CALIBRATION UNKNOWN" in report["rationale"]

    def test_a_calibrated_bucket_leaves_the_rationale_alone(
        self, evidenced_bundle, monkeypatch
    ):
        """The disclosure has to mean something when it is absent."""
        observation = _observed(clamped=False, caveat="")
        report = self._laya_report(evidenced_bundle, monkeypatch, observation)

        assert "CALIBRAT" not in report["rationale"].upper()
        assert report["laya"]["clamped"] is False
        assert "caveat" not in report["confidence_source"]

    def test_an_llm_confidence_has_nothing_to_disclose(self, evidenced_bundle, monkeypatch):
        report = _report(evidenced_bundle, monkeypatch, _prediction())

        assert "CALIBRAT" not in report["rationale"].upper()
        assert "caveat" not in report["confidence_source"]

    def test_the_escalation_reason_discloses_only_the_number_it_gated_on(
        self, evidenced_bundle, monkeypatch
    ):
        """The caveat is about the Laya probability, and by default the sweep is
        not decided on it.

        With no derived cutoff, ``decide`` gates on the adapters' figure -- which
        is not from a clamped bucket, so stamping "NOT CALIBRATED" on that line
        would attribute the Laya number's problem to a decision that avoided it.
        The reason says instead why the Laya number was set aside, and the
        caveat stays where that number actually appears.
        """
        report = self._laya_report(evidenced_bundle, monkeypatch, _observed())
        reason = report["escalation"]["reason"]

        assert "NOT CALIBRATED" not in reason
        assert "no cutoff has been derived for laya" in reason
        assert report["escalation"]["gated_on"]["source"] == ADAPTER_RULES
        assert "NOT CALIBRATED" in report["confidence_source"]["caveat"]
        assert "NOT CALIBRATED" in report["rationale"]

    def test_the_caveat_is_not_what_the_rationale_cap_deletes(self):
        """The cap exists because a model can write at length, and a disclosure
        concatenated upstream would be the tail the slice removes -- on exactly
        the verbose reports most likely to be read carefully."""
        from aorta.cia.autopsy.reporter import build_report

        report = build_report(
            session_id="s",
            generated_at="t",
            bundle_job_id="j",
            bundle_root="/tmp",
            kb_version=None,
            category="numeric_silent",
            confidence=0.91,
            rationale="x" * 5000,
            evidence=[],
            next_probes=[],
            tooling_gaps=[],
            rationale_caveat="; NOT CALIBRATED: choice:11+",
        )

        assert report["rationale"].endswith("; NOT CALIBRATED: choice:11+")
        assert report["rationale"].count("x") == 2000


class TestDisclosingDoesNotChangeWhatRuns:
    """Refusing the threshold and disclosing the fit are separate facts.

    An operator who derived a cutoff against a clamped checkpoint derived it
    against the behaviour they will actually get, so acting on the caveat here
    would override a measurement with a rule of thumb.
    """

    def test_a_caveat_does_not_move_the_escalation(self):
        loud = Confidence(0.4, LAYA, "ckpt", caveat="; NOT CALIBRATED: choice:11+")
        quiet = Confidence(0.4, LAYA, "ckpt")

        assert (
            _decide(loud, _adapters(0.95), laya_threshold=0.55).escalate
            is _decide(quiet, _adapters(0.95), laya_threshold=0.55).escalate
        )

    def test_it_is_still_recorded_on_the_decision_that_ignored_it(self):
        decision = _decide(
            Confidence(0.4, LAYA, "ckpt", caveat="; NOT CALIBRATED: choice:11+"),
            _adapters(0.95),
            laya_threshold=0.55,
        )

        assert "NOT CALIBRATED" in decision.reason


def test_importing_autopsy_pulls_in_no_model_machinery():
    """The property the lazy imports exist to protect, asserted directly.

    ``tests/cli/test_chat_boundaries.py`` already imports the orchestrator in a
    clean interpreter and checks it for torch. This checks the same path for the
    rest of the loader stack, which that tuple does not list -- so a module-scope
    ``import laya`` here fails on the tier that added it rather than on whichever
    of its dependencies happens to be listed elsewhere.
    """
    import subprocess
    import sys

    heavy = ("torch", "transformers", "laya", "onnxruntime", "safetensors")
    probe = (
        "import sys, json, aorta.cia.autopsy.orchestrator, aorta.cia.autopsy.router;"
        f"heavy={heavy!r};"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in heavy)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == [], (
        f"importing Autopsy pulled in {out.stdout.strip()}; the Laya imports must "
        "stay inside the tier that loads a checkpoint."
    )
