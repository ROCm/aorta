"""The predictor seam: the contract three later tracks are written against.

Nothing here loads weights. What is pinned is the shape of the contract and the
two properties the integration tracks will lean on hardest -- that the option set
of a choice *is* the answer space, so a path or a mitigation the caller did not
offer cannot come back, and that a probability is never read as a boolean without
a threshold being named.

The real checkpoint path is covered by driving the loader seam with a stub. That
is not a substitute for running against weights, and it is not pretending to be:
it pins the translation between our question types and the library's, which is
the part that can be wrong without anyone noticing until a verdict is wrong.
"""

from __future__ import annotations

import pytest

from aorta.local_classifier.predictor import (
    CHECKPOINTS,
    VERIFIED_LAYA_VERSION,
    Calibration,
    Choice,
    ChoiceAnswer,
    ClampedBucket,
    ClassifierUnavailableError,
    DecisionPredictor,
    FakeDecisionPredictor,
    LayaAgentPredictor,
    Noul,
    NoulAnswer,
    ask_choice,
    ask_noul,
    ask_one,
    bucket_for,
    make_predictor,
)

_CLEAN = Noul(question="is this log healthy?")
_SIGNAL = Choice(question="which signal?", options=("nan", "hang", "oom"))


class _StubAgent:
    """Stands in for a loaded ``laya.Agent``, answering in its payload shape.

    Written from the answer format in the published 0.3.5 wheel --
    ``{"answers": {qid: {...}}}`` with ``noul`` carrying p(true) and ``choice``
    carrying a ``probabilities`` map keyed by the criteria keys -- so a change to
    that format fails here rather than in production. ``cfg`` and ``tok`` are here
    because ``context_length`` and ``token_count`` read them off the loaded agent.
    """

    cfg = {"max_len": 1024}

    def __init__(self, answers: dict | None = None) -> None:
        self._answers = answers or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.tok = lambda text, add_special_tokens=False: {"input_ids": text.split()}

    def predict(self, state: str, questions: dict) -> dict:
        self.calls.append((state, tuple(questions)))
        answers = {}
        for qid, spec in questions.items():
            override = self._answers.get(spec["instructions"])
            if override is not None:
                answers[qid] = override
            elif spec["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.75, "confidence": 0.75}
            else:
                options = list(spec["criteria"])
                share = 1.0 / len(options)
                answers[qid] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": dict.fromkeys(options, share),
                }
        return {"answers": answers, "usage": {"input_tokens": 1}}


def _agent_predictor(agent: _StubAgent) -> LayaAgentPredictor:
    return LayaAgentPredictor("laya-typed-decisions", load=lambda _checkpoint: agent)


class TestAnswers:
    def test_a_noul_carries_a_probability_and_no_verdict(self):
        """No ``.value``, on purpose: Watch's two thresholds gate opposite ways.

        ``confidence_threshold`` escalates *above* it and a local-classifier clean-gate skips
        the LLM *above* its own, so a default boolean read here is an inversion
        bug waiting for whichever caller forgot which way round it went.
        """
        answer = NoulAnswer(probability=0.8)
        assert not hasattr(answer, "value")
        assert answer.at(0.7)
        assert not answer.at(0.9)

    def test_at_the_threshold_counts_as_reaching_it(self):
        """The same ``>=`` ``should_alert`` uses, so both sides of the poll agree."""
        assert NoulAnswer(probability=0.7).at(0.7)

    def test_a_choice_reports_the_argmax_and_its_probability(self):
        answer = ChoiceAnswer(probabilities=(("nan", 0.2), ("hang", 0.5), ("oom", 0.3)))
        assert answer.option == "hang"
        assert answer.probability == pytest.approx(0.5)

    def test_a_choice_keeps_the_runners_up_for_a_ranker(self):
        """The log finder ranks a whole listing, so the argmax alone is not enough."""
        answer = ChoiceAnswer(probabilities=(("nan", 0.2), ("hang", 0.5), ("oom", 0.3)))
        assert [name for name, _ in answer.ranked()] == ["hang", "oom", "nan"]

    def test_an_option_never_offered_has_probability_zero(self):
        answer = ChoiceAnswer(probabilities=(("nan", 1.0),))
        assert answer.probability_of("throttle") == 0.0

    def test_a_tie_breaks_toward_the_order_offered(self):
        answer = ChoiceAnswer(probabilities=(("nan", 0.5), ("hang", 0.5)))
        assert answer.option == "nan"


class TestFake:
    def test_it_satisfies_the_protocol(self):
        predictor: DecisionPredictor = FakeDecisionPredictor()
        assert predictor.model_id() == "fake"

    def test_it_answers_one_list_per_state_aligned_with_the_questions(self):
        answers = FakeDecisionPredictor().ask(["a", "b"], [_CLEAN, _SIGNAL])
        assert len(answers) == 2
        assert all(len(row) == 2 for row in answers)
        assert isinstance(answers[0][0], NoulAnswer)
        assert isinstance(answers[0][1], ChoiceAnswer)

    def test_a_choice_answer_covers_exactly_the_options_offered(self):
        """The property the whole log-finder track rests on."""
        answer = ask_choice(FakeDecisionPredictor(), "listing", _SIGNAL)
        assert [name for name, _ in answer.probabilities] == list(_SIGNAL.options)

    def test_a_choice_answer_is_a_distribution(self):
        answer = ask_choice(FakeDecisionPredictor(), "listing", _SIGNAL)
        assert sum(value for _, value in answer.probabilities) == pytest.approx(1.0)

    def test_it_is_deterministic_across_processes(self):
        """blake2b rather than hash(), which is salted per process.

        A fake built on the built-in would answer differently on every run, and a
        test that pinned an answer would pass locally and fail in CI.
        """
        first = ask_noul(FakeDecisionPredictor(), "same log", _CLEAN).probability
        second = ask_noul(FakeDecisionPredictor(), "same log", _CLEAN).probability
        assert first == second

    def test_different_states_get_different_answers(self):
        one = ask_noul(FakeDecisionPredictor(), "log one", _CLEAN).probability
        two = ask_noul(FakeDecisionPredictor(), "log two", _CLEAN).probability
        assert one != two

    def test_a_pinned_answer_wins(self):
        """How a later track's test controls the model without owning a checkpoint."""
        predictor = FakeDecisionPredictor(pinned={_CLEAN.question: NoulAnswer(probability=0.99)})
        assert ask_noul(predictor, "anything", _CLEAN).probability == 0.99

    def test_no_questions_costs_no_answers(self):
        assert FakeDecisionPredictor().ask(["a"], []) == [[]]

    def test_its_calibration_is_unknown_rather_than_clean(self):
        """A test double whose calibration read as fine would be one more way for a
        hash-derived number to pass for a measured one."""
        calibration = FakeDecisionPredictor().calibration()
        assert not calibration.known
        assert "no checkpoint" in calibration.unknown_because
        assert "CALIBRATION UNKNOWN" in calibration.caveat(_CLEAN)

    def test_it_cannot_count_tokens(self):
        """Deliberate: a census without weights would decide the chunking question.

        ``aorta.local_classifier.gate.token_census`` looks for this method and refuses
        without it, so the absence is load-bearing rather than an omission.
        """
        assert not hasattr(FakeDecisionPredictor(), "token_count")


class TestHelpers:
    def test_ask_one_returns_the_answers_for_a_single_state(self):
        answers = ask_one(FakeDecisionPredictor(), "log", [_CLEAN, _SIGNAL])
        assert len(answers) == 2

    def test_asking_a_choice_as_a_noul_is_a_type_error(self):
        """A predictor answering the wrong shape must not be read as 0.5."""

        class Wrong(FakeDecisionPredictor):
            def ask(self, states, questions):
                return [[ChoiceAnswer(probabilities=(("a", 1.0),))] for _ in states]

        with pytest.raises(TypeError, match="answered a noul"):
            ask_noul(Wrong(), "log", _CLEAN)

    def test_asking_a_noul_as_a_choice_is_a_type_error(self):
        class Wrong(FakeDecisionPredictor):
            def ask(self, states, questions):
                return [[NoulAnswer(probability=0.5)] for _ in states]

        with pytest.raises(TypeError, match="answered a choice"):
            ask_choice(Wrong(), "log", _SIGNAL)


class TestTheRealTranslation:
    def test_a_noul_reaches_the_library_as_a_noul(self):
        agent = _StubAgent()
        answer = ask_noul(_agent_predictor(agent), "log text", _CLEAN)
        assert answer.probability == pytest.approx(0.75)
        assert agent.calls == [("log text", ("q0",))]

    def test_a_bare_noul_sends_no_criteria(self):
        """The library supplies its own two sentences, which the model was trained on.

        Passing an empty criteria dict would replace them with nothing.
        """
        sent: dict = {}

        class Recording(_StubAgent):
            def predict(self, state, questions):
                sent.update(questions)
                return super().predict(state, questions)

        ask_noul(_agent_predictor(Recording()), "log", _CLEAN)
        assert "criteria" not in sent["q0"]

    def test_a_glossed_noul_sends_both_sides(self):
        sent: dict = {}

        class Recording(_StubAgent):
            def predict(self, state, questions):
                sent.update(questions)
                return super().predict(state, questions)

        glossed = Noul(question="clean?", when_true="healthy", when_false="broken")
        ask_noul(_agent_predictor(Recording()), "log", glossed)
        assert sent["q0"]["criteria"] == {"true": "healthy", "false": "broken"}

    def test_every_option_becomes_a_key_whether_or_not_it_is_glossed(self):
        """The criteria keys are the answer space, so a missing one is not offered."""
        sent: dict = {}

        class Recording(_StubAgent):
            def predict(self, state, questions):
                sent.update(questions)
                return super().predict(state, questions)

        partly = Choice(
            question="which?",
            options=("nan", "hang", "oom"),
            criteria=(("nan", "a non-finite value"),),
        )
        ask_choice(_agent_predictor(Recording()), "log", partly)
        assert list(sent["q0"]["criteria"]) == ["nan", "hang", "oom"]

    def test_a_choice_answer_is_re_read_in_the_order_offered(self):
        """A caller zipping an answer against its own option list must not be misaligned."""
        agent = _StubAgent(
            answers={
                _SIGNAL.question: {
                    "type": "choice",
                    "choice": "oom",
                    # Returned in an order deliberately unlike the question's.
                    "probabilities": {"oom": 0.6, "nan": 0.3, "hang": 0.1},
                }
            }
        )
        answer = ask_choice(_agent_predictor(agent), "log", _SIGNAL)
        assert [name for name, _ in answer.probabilities] == ["nan", "hang", "oom"]
        assert answer.option == "oom"

    def test_many_questions_about_one_state_are_one_call(self):
        """The efficiency claim, asserted rather than assumed."""
        agent = _StubAgent()
        _agent_predictor(agent).ask(["one state"], [_CLEAN, _SIGNAL, Noul(question="again?")])
        assert len(agent.calls) == 1
        assert agent.calls[0][1] == ("q0", "q1", "q2")

    def test_many_states_are_many_calls(self):
        """N states are N forward passes, which is why batch 30 is the reranking blocker."""
        agent = _StubAgent()
        _agent_predictor(agent).ask(["a", "b", "c"], [_CLEAN])
        assert len(agent.calls) == 3

    def test_the_checkpoint_is_named_in_the_model_id(self):
        assert _agent_predictor(_StubAgent()).model_id() == "laya-typed-decisions"

    def test_the_device_travels_with_the_model_id(self):
        predictor = LayaAgentPredictor(
            "laya", device="cpu", load=lambda _c: _StubAgent()
        )
        assert predictor.model_id() == "laya@cpu"

    def test_the_context_length_comes_off_the_loaded_config(self):
        """Read rather than assumed: the checkpoints differ at 512 and 1024."""
        assert _agent_predictor(_StubAgent()).context_length() == 1024

    def test_loading_is_deferred_until_the_first_question(self):
        """A poll loop that never reaches an ambiguous log must not pay for weights."""
        loaded: list[str] = []
        predictor = LayaAgentPredictor(
            "laya", load=lambda checkpoint: loaded.append(checkpoint) or _StubAgent()
        )
        assert loaded == []
        predictor.ask(["log"], [_CLEAN])
        assert loaded == ["laya"]

    def test_the_checkpoint_is_loaded_once(self):
        loaded: list[str] = []
        predictor = LayaAgentPredictor(
            "laya", load=lambda checkpoint: loaded.append(checkpoint) or _StubAgent()
        )
        predictor.ask(["a"], [_CLEAN])
        predictor.ask(["b"], [_CLEAN])
        assert len(loaded) == 1


class TestTheRealFailures:
    def test_a_loader_failure_names_the_checkpoint(self):
        """The library's own error says which file it could not fetch, not which tier asked."""

        def _explode(_checkpoint: str):
            raise OSError("no route to huggingface.co")

        predictor = LayaAgentPredictor("laya", load=_explode)
        with pytest.raises(ClassifierUnavailableError, match="laya"):
            predictor.ask(["log"], [_CLEAN])

    def test_a_missing_answer_raises_rather_than_defaulting(self):
        """The opposite of how the LLM proposers fail, and deliberately.

        There, an unparseable reply is expected and has to still produce a
        report. Here it means the answer shape moved under us, and a 0.5 invented
        at this seam would travel into a threshold comparison as though it had
        been measured.
        """
        agent = _StubAgent(answers={_CLEAN.question: {"type": "noul"}})
        with pytest.raises(ClassifierUnavailableError, match="no 'noul' probability"):
            ask_noul(_agent_predictor(agent), "log", _CLEAN)

    def test_a_choice_answer_missing_an_offered_option_raises(self):
        agent = _StubAgent(
            answers={
                _SIGNAL.question: {
                    "type": "choice",
                    "probabilities": {"nan": 0.5, "hang": 0.5},
                }
            }
        )
        with pytest.raises(ClassifierUnavailableError, match="without the offered options"):
            ask_choice(_agent_predictor(agent), "log", _SIGNAL)

    def test_a_choice_answer_with_no_probabilities_raises(self):
        agent = _StubAgent(answers={_SIGNAL.question: {"type": "choice", "choice": "nan"}})
        with pytest.raises(ClassifierUnavailableError, match="no\n?.*'probabilities'"):
            ask_choice(_agent_predictor(agent), "log", _SIGNAL)

    def test_an_unknown_checkpoint_is_refused_at_construction(self):
        with pytest.raises(ClassifierUnavailableError, match="unknown Laya checkpoint"):
            LayaAgentPredictor("laya-multilingual")

    def test_a_local_directory_is_accepted_as_a_fine_tune(self, tmp_path):
        """The gate has to score a fine-tune, which exists on disk before anywhere else."""
        assert LayaAgentPredictor(str(tmp_path)).model_id() == str(tmp_path)

    def test_a_published_name_is_not_mistaken_for_a_relative_path(self):
        """``org/name`` also looks like a path, so the test is whether it is really there."""
        with pytest.raises(ClassifierUnavailableError, match="unknown Laya checkpoint"):
            LayaAgentPredictor("convaiinnovations/laya-multilingual")


class TestBucketFor:
    """Which shipped temperature answered a question. Pure, and offline on purpose.

    Two real callers fall on opposite sides of the interesting boundary, which is
    the whole argument for one definition: the probe agent's candidate choice is 21
    options, Autopsy's category choice is over the eleven-member
    ``AUTOPSY_CATEGORIES``, and the probe agent's own category choice is eight.
    """

    @pytest.mark.parametrize(
        "width, expected",
        [
            (2, "choice:2"),
            (3, "choice:3-5"),
            (5, "choice:3-5"),
            (6, "choice:6-10"),
            (10, "choice:6-10"),
            (11, "choice:11+"),
            (21, "choice:11+"),
        ],
    )
    def test_the_boundaries_match_the_librarys_own(self, width: int, expected: str):
        """Pinned at the boundary values, because this table is a copy of one upstream.

        Reading ``laya.common.temp_bucket`` instead would put torch on the import
        path of every caller that wants the answer before loading anything, which
        is where it is most useful. The cost of the copy is that it can go stale
        silently -- a caveat that stops firing looks exactly like a caveat with
        nothing to say -- so the boundaries are asserted rather than trusted.
        """
        question = Choice(question="which?", options=tuple(f"o{n}" for n in range(width)))
        assert bucket_for(question) == expected

    def test_a_noul_is_always_the_two_option_bucket(self):
        assert bucket_for(Noul(question="clean?")) == "noul:2"

    def test_a_noul_never_reaches_the_clamped_bucket(self):
        """The reason a per-candidate noul shape sidesteps the problem entirely."""
        assert bucket_for(Noul(question="would this help?")) != "choice:11+"

    def test_the_autopsy_category_choice_is_in_the_clamped_bucket(self):
        """Phase 4's choice, and the second caller. Eleven members is exactly the line.

        Derived from the real set rather than a literal eleven, so a category
        added or removed moves this assertion instead of leaving it describing a
        set that no longer exists.
        """
        from aorta.agent.llm import AUTOPSY_CATEGORIES

        question = Choice(question="which?", options=tuple(sorted(AUTOPSY_CATEGORIES)))
        assert bucket_for(question) == "choice:11+"

    def test_the_probe_category_choice_is_not(self):
        from aorta.agent.llm import PROBE_CATEGORIES

        question = Choice(question="which?", options=tuple(sorted(PROBE_CATEGORIES)))
        assert bucket_for(question) != "choice:11+"


class TestCalibration:
    """What a probability is a function of, besides the checkpoint.

    Decision 22 asks a report to record the temperature fit. Before this there was
    nothing real to record for it, and two callers had each started deriving
    "was this trustworthy" from their own option counts.
    """

    _CLAMPED = Calibration(
        model_id="stub",
        clamped=(ClampedBucket(bucket="choice:11+", shipped=0.1006, applied=0.5),),
        applied=(("choice:11+", 0.5), ("choice:3-5", 1.2)),
    )

    def test_a_clamped_bucket_is_reported_as_clamped(self):
        assert self._CLAMPED.is_clamped("choice:11+") is True

    def test_an_untouched_bucket_is_not(self):
        assert self._CLAMPED.is_clamped("choice:3-5") is False

    def test_an_unknown_calibration_answers_none_rather_than_false(self):
        """Collapsing None to False claims a checkpoint was fine without asking it."""
        unknown = Calibration(model_id="stub", unknown_because="not loaded")
        assert unknown.is_clamped("choice:11+") is None
        assert not unknown.known

    def test_the_caveat_names_the_bucket_and_both_temperatures(self):
        """It ends up in a hypothesis or a rationale, so it has to read as a sentence."""
        caveat = self._CLAMPED.caveat(Choice(question="q", options=tuple("abcdefghijkl")))
        assert "NOT CALIBRATED" in caveat
        assert "choice:11+" in caveat
        assert "0.1006" in caveat
        assert "0.5" in caveat

    def test_a_question_in_a_clean_bucket_gets_no_caveat(self):
        """Crying wolf on every line is noise, which is how a real warning gets ignored."""
        assert self._CLAMPED.caveat(Noul(question="clean?")) == ""
        assert self._CLAMPED.caveat(Choice(question="q", options=("a", "b", "c"))) == ""

    def test_an_unknown_calibration_says_so_rather_than_staying_silent(self):
        unknown = Calibration(
            model_id="stub", unknown_because="the checkpoint has not been loaded yet"
        )
        caveat = unknown.caveat(Noul(question="clean?"))
        assert "CALIBRATION UNKNOWN" in caveat
        assert "not been loaded" in caveat

    def test_it_serialises_for_a_report(self):
        payload = self._CLAMPED.to_dict()
        assert payload["known"] is True
        assert payload["clamped"][0]["bucket"] == "choice:11+"
        assert payload["clamped"][0]["shipped"] == 0.1006
        assert payload["applied"]["choice:3-5"] == 1.2


class TestTheRealCalibration:
    """Read by comparing the two tables 0.3.5 keeps, not by catching its warning.

    The warning is raised once inside ``__init__``, so anything catching it would
    have to have installed a filter before a load it does not control the timing
    of. The tables are still on the agent afterwards.
    """

    def _loaded(self, **attributes) -> LayaAgentPredictor:
        """A predictor whose stub agent is already loaded, and with it a table.

        The question asked here is the load: :meth:`calibration` deliberately does
        not trigger one, so a test that never asked anything would be exercising
        the unloaded path -- which is what every one of these did on first writing.
        """
        agent = _StubAgent()
        for name, value in attributes.items():
            setattr(agent, name, value)
        predictor = _agent_predictor(agent)
        predictor.ask(["state"], [_CLEAN])
        return predictor

    def test_a_clamped_bucket_is_found_by_diffing_the_tables(self):
        calibration = self._loaded(
            temperature_by_options_raw={"choice:11+": 0.1006, "choice:3-5": 1.2},
            temperature_by_options={"choice:11+": 0.5, "choice:3-5": 1.2},
        ).calibration()
        assert calibration.known
        assert [entry.bucket for entry in calibration.clamped] == ["choice:11+"]
        assert calibration.clamped[0].shipped == 0.1006
        assert calibration.clamped[0].applied == 0.5

    def test_an_unclamped_checkpoint_reports_nothing_clamped(self):
        calibration = self._loaded(
            temperature_by_options_raw={"choice:3-5": 1.2},
            temperature_by_options={"choice:3-5": 1.2},
        ).calibration()
        assert calibration.known
        assert calibration.clamped == ()
        assert calibration.applied == (("choice:3-5", 1.2),)

    def test_the_whole_applied_table_is_recorded_not_only_the_differences(self):
        """Decision 22 asks for the fit, and the buckets that survived are most of it."""
        calibration = self._loaded(
            temperature_by_options_raw={"choice:11+": 0.1, "noul:2": 1.4},
            temperature_by_options={"choice:11+": 0.5, "noul:2": 1.4},
        ).calibration()
        assert dict(calibration.applied) == {
            "choice:11+": 0.5,
            "noul:2": 1.4,
        }

    def test_a_clamped_per_type_fallback_is_reported_too(self):
        """Wider in effect than one bucket, so it would be the worse of the two to miss.

        The fallback applies to every bucket with no explicit entry.
        """
        clamped = self._loaded(
            temperature_by_options_raw={},
            temperature_by_options={},
            temperature_raw=[0.05, 1.0, 1.0],
            temperature=[0.5, 1.0, 1.0],
        ).calibration().clamped
        assert len(clamped) == 1
        assert clamped[0].bucket.endswith(":*")
        assert clamped[0].shipped == 0.05

    def test_an_unloaded_checkpoint_is_unknown_rather_than_clean(self):
        """And does not load: an accessor must not spend seconds building a model."""
        loaded: list[str] = []
        predictor = LayaAgentPredictor(
            "laya", load=lambda name: loaded.append(name) or _StubAgent()
        )
        calibration = predictor.calibration()
        assert loaded == []
        assert not calibration.known
        assert "not been loaded" in calibration.unknown_because

    def test_a_library_that_keeps_no_shipped_table_is_unknown_not_clean(self):
        """Laya below 0.3.5 applies the table verbatim and keeps no copy.

        On that version the answer is not that nothing was clamped -- it is that
        the sharpening was applied and nobody was told, which is why the extra
        floors above it.
        """
        # No temperature attributes at all, which is what 0.3.4's agent exposed.
        calibration = self._loaded().calibration()
        assert not calibration.known
        assert "0.3.5" in calibration.unknown_because

    def test_it_names_the_checkpoint_that_was_measured(self):
        predictor = self._loaded(temperature_by_options_raw={}, temperature_by_options={})
        assert predictor.calibration().model_id == "laya-typed-decisions"

    def test_the_caveat_off_a_real_load_reaches_the_offered_question(self):
        """End to end: a 21-option choice against a checkpoint that clamped that bucket."""
        predictor = self._loaded(
            temperature_by_options_raw={"choice:11+": 0.1006},
            temperature_by_options={"choice:11+": 0.5},
        )
        wide = Choice(question="which?", options=tuple(f"m{n}" for n in range(21)))
        predictor.ask(["state"], [wide])
        assert "NOT CALIBRATED" in predictor.calibration().caveat(wide)


class TestTheVerifiedRelease:
    """The floor and the verification are kept the same number, deliberately.

    0.3.3, 0.3.4 and 0.3.5 shipped on three consecutive days, moving both the
    Python floor and the transformers floor inside that window. The first version
    of this module was written against 0.3.4 and the extra floored at 0.3.4, which
    on every Python aorta supports resolves to 0.3.5 -- so the code was validated
    against a release no install would get. That is the gap this pins shut.
    """

    def test_the_extra_floors_at_the_release_that_was_verified(self, repo_root):
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        assert f'local-classifier = [\n    "laya>={VERIFIED_LAYA_VERSION}",' in pyproject, (
            f"the [local-classifier] extra does not floor at {VERIFIED_LAYA_VERSION}, which "
            "is the release aorta/local_classifier/predictor.py was read against. Re-read "
            "the wheel before moving either number."
        )

    def test_the_floor_is_a_floor_and_not_a_pin(self):
        """A later release is expected to work; nothing here refuses one at runtime.

        Pinning a dependency we do not control would trade a small, visible risk
        for a stale one, and there is no version assertion in the loader for the
        same reason.
        """
        import aorta.local_classifier.predictor as module

        source = module.__file__
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "__version__" not in text


class TestFactory:
    def test_fake_is_the_default_and_needs_nothing_installed(self):
        """Mirrors ``make_proposer``: the default has to work on a base install."""
        assert isinstance(make_predictor(), FakeDecisionPredictor)
        assert isinstance(make_predictor("fake"), FakeDecisionPredictor)

    def test_a_checkpoint_name_resolves_without_loading_it(self):
        predictor = make_predictor("laya-typed-decisions")
        assert isinstance(predictor, LayaAgentPredictor)
        assert predictor.model_id() == "laya-typed-decisions"

    def test_an_unknown_backend_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="unknown predictor backend"):
            make_predictor("gpt-4o")

    def test_a_local_checkpoint_overrides_the_backend(self, tmp_path):
        predictor = make_predictor("fake", checkpoint=str(tmp_path))
        assert isinstance(predictor, LayaAgentPredictor)
        assert predictor.model_id() == str(tmp_path)

    def test_the_multilingual_checkpoints_are_not_offered(self):
        """Skipped on purpose: the logs are English and pinning one avoids a reload.

        The card measures a 7.4 s median reload on CPU when traffic alternates
        checkpoints at the library's default of one resident model.
        """
        assert not any("multilingual" in name for name in CHECKPOINTS)

    def test_importing_the_seam_pulls_in_no_model_machinery(self):
        """The property the lazy imports exist to protect, asserted directly.

        ``_HEAVY_PREFIXES`` in ``tests/cli/test_chat_boundaries.py`` covers
        ``aorta.cli`` and ``aorta.agent.llm``; this covers the package those two
        will reach once the integration tracks land. A module-scope ``import
        laya`` here would put torch on the import path of every Watch poll and,
        through the chat router in Phase 5, of an install that CI fails for
        resolving torch at all.

        Measured in a subprocess, because ``sys.modules`` is already dirty by the
        time pytest runs this.
        """
        import json
        import subprocess
        import sys

        heavy = ("torch", "transformers", "laya", "onnxruntime", "numpy", "safetensors")
        probe = (
            "import sys, json, aorta.local_classifier, aorta.local_classifier.eval, "
            "aorta.local_classifier.gate, aorta.local_classifier.corpus, "
            "aorta.local_classifier.cli;"
            f"heavy={heavy!r};"
            "print(json.dumps(sorted(m for m in sys.modules "
            "if m.split('.')[0] in heavy)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout) == [], (
            f"importing aorta.local_classifier pulled in {out.stdout.strip()}. Every model import "
            "must live inside the function that loads a checkpoint."
        )

    def test_importing_the_corpus_builders_does_not_pull_in_the_agents_extra(self):
        """What the lazy imports in the Watch and log-finder builders are for.

        Both reach into ``aorta.cia.watch`` -- one for the questions its tier asks,
        one for the listing its tier is shown -- and that package imports dspy at
        module scope. Reaching either eagerly would make the whole ``aorta.local_classifier``
        package need the ``[cia]`` extra for the proposer builder and the scoring,
        which have no use for it.
        """
        import json
        import subprocess
        import sys

        probe = (
            "import sys, json, aorta.local_classifier.corpus, aorta.local_classifier.corpus.watch, "
            "aorta.local_classifier.corpus.proposer;"
            "print(json.dumps(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('dspy', 'litellm'))))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout) == [], (
            f"importing the corpus builders pulled in {out.stdout.strip()}. The "
            "agents' extra must stay optional for the builders that do not need it."
        )

    def test_no_score_primitive_is_offered_anywhere(self):
        """The third primitive is the weakest, and failure severity is the worst place for it."""
        import aorta.local_classifier.predictor as module

        assert not hasattr(module, "Score")
        assert "score" not in module.__all__
