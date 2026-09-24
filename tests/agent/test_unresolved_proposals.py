"""aorta#449: a dropped mitigation name must not be recorded as an agent stop.

The proposer filters the model's proposed names against the remaining
candidates *before* ``AgentPolicy.validate_step`` gets to reject them, so an
unregistered name empties ``next_mitigations``, and the loop's stop check
reads an empty list as "the model has no further hypotheses". The recorded
``stop_reason`` then contradicts the model's own ``stop: false``.

What is protected here:

* the two cases no longer share a ``stop_reason``;
* the dropped names survive in ``agent_log.jsonl``, which is what makes an
  affected run repairable instead of merely detectable;
* a run with nothing dropped writes the same bytes it wrote before this
  existed -- asserted against a literal golden log, because already-archived
  trajectories are compared against it.

``FakeLLMProposer`` picks from the candidate list it was handed and so cannot
produce an unresolvable name; the filter lives on the shared real-LLM path.
That is why these tests drive a real ``ChatProviderProposer`` with only the
chat model stubbed -- ``llm.py``, ``policy.py`` and ``loop.py`` are all real
code here, exactly as in the issue's reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aorta.agent.llm import AgentStep, ChatProviderProposer, FakeLLMProposer
from aorta.agent.loop import AgentConfig, _resolve_stop_outcome, run_agent_loop
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.state import wake

#: "rccl_p2p_disable" is not registered. It is the kind of name a model
#: invents because it reads exactly like one that exists.
UNREGISTERED = "rccl_p2p_disable"

REPO = Path(__file__).resolve().parents[2]

CANDIDATES = ["none", "tf32_off", "hsa_no_sdma"]

BASELINE_FAILED = [
    {
        "cell_name": "none-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier2:hang"],
        "warn_detectors_fired": [],
        "capture": {},
        "exit_code": None,
    }
]


def _reply(**payload) -> str:
    return json.dumps(payload)


@pytest.fixture()
def proposer(monkeypatch):
    """A real ChatProviderProposer whose only stub is the HTTP transport.

    Accepts one reply or a sequence of them: the loop keeps proposing until
    something stops it, and a partial rejection does not stop it.
    """

    def _install(content: str | list[str]) -> ChatProviderProposer:
        replies = [content] if isinstance(content, str) else list(content)
        calls = {"n": 0}

        def _invoke(_messages):
            reply = replies[min(calls["n"], len(replies) - 1)]
            calls["n"] += 1
            return SimpleNamespace(content=reply)

        monkeypatch.setattr(
            ChatProviderProposer,
            "_chat_model",
            lambda self: SimpleNamespace(invoke=_invoke),
        )
        return ChatProviderProposer("openai")

    return _install


def _propose(proposer, tried=None):
    return proposer.propose(
        symptom="training hangs at step 40",
        cell_summaries=BASELINE_FAILED,
        candidates=CANDIDATES,
        tried=tried or [],
    )


# ── the proposer keeps what it dropped ────────────────────────────────────


class TestTheFilterRecordsWhatItDrops:
    def test_an_unregistered_name_is_still_dropped(self, proposer):
        """Unchanged behaviour: the model does not get to widen its allowlist."""
        step = _propose(proposer(_reply(next_mitigations=[UNREGISTERED], stop=False)))
        assert step.next_mitigations == []

    def test_but_the_dropped_name_is_now_retained(self, proposer):
        """FAILS BEFORE THE FIX: AgentStep had no field to retain it."""
        step = _propose(proposer(_reply(next_mitigations=[UNREGISTERED], stop=False)))
        assert step.unresolved_mitigations == [UNREGISTERED]

    def test_a_partial_rejection_keeps_both_halves(self, proposer):
        """The quiet variant: the search runs half the plan the model stated.

        The loop continues here, so no stop event is ever written -- this is
        the only place the discarded half can be recorded.
        """
        step = _propose(
            proposer(_reply(next_mitigations=[UNREGISTERED, "tf32_off"], stop=False))
        )
        assert step.next_mitigations == ["tf32_off"]
        assert step.unresolved_mitigations == [UNREGISTERED]

    def test_the_model_proposal_is_reconstructible(self, proposer):
        """What makes an affected row repairable rather than discardable."""
        proposed = [UNREGISTERED, "tf32_off", "not_a_mitigation_either"]
        step = _propose(proposer(_reply(next_mitigations=proposed, stop=False)))
        reconstructed = [
            name
            for name in proposed
            if name in step.next_mitigations or name in step.unresolved_mitigations
        ]
        assert reconstructed == proposed

    def test_an_already_tried_name_counts_as_unresolved_too(self, proposer):
        """The filter drops on `remaining`, not on registry membership alone."""
        step = _propose(
            proposer(_reply(next_mitigations=["tf32_off"], stop=False)), tried=["tf32_off"]
        )
        assert step.next_mitigations == []
        assert step.unresolved_mitigations == ["tf32_off"]

    def test_the_baseline_is_dropped_even_though_it_is_registered(self, proposer):
        """The fourth cause the docs and the operator message now name."""
        step = _propose(proposer(_reply(next_mitigations=["none"], stop=False)))
        assert step.next_mitigations == []
        assert step.unresolved_mitigations == ["none"]

    def test_a_clean_proposal_records_no_rejections(self, proposer):
        step = _propose(proposer(_reply(next_mitigations=["tf32_off"], stop=False)))
        assert step.next_mitigations == ["tf32_off"]
        assert step.unresolved_mitigations == []

    def test_a_genuine_stop_records_no_rejections(self, proposer):
        step = _propose(
            proposer(_reply(hypothesis="no further hypotheses", next_mitigations=[], stop=True))
        )
        assert step.stop is True
        assert step.stop_reason == "agent_requested"
        assert step.unresolved_mitigations == []

    def test_the_drop_is_warned_about(self, proposer, caplog):
        """Nothing raised and nothing was logged before; silence was the defect."""
        with caplog.at_level("WARNING", logger="aorta.agent.llm"):
            _propose(proposer(_reply(next_mitigations=[UNREGISTERED], stop=False)))
        assert UNREGISTERED in caplog.text

    def test_the_model_cannot_claim_names_were_dropped(self, proposer):
        """The field is the agent's finding, so a reply must not be able to set it."""
        step = _propose(
            proposer(
                _reply(
                    next_mitigations=["tf32_off"],
                    unresolved_mitigations=["fabricated"],
                    stop=False,
                )
            )
        )
        assert step.unresolved_mitigations == []

    def test_the_model_cannot_claim_the_new_stop_reason(self, proposer):
        """Same rule as the downgraded baseline_pass: not the model's to assert."""
        step = _propose(
            proposer(_reply(next_mitigations=[], stop=True, stop_reason="proposal_unresolved"))
        )
        assert step.stop_reason == "agent_requested"

    def test_the_fake_proposer_cannot_trigger_any_of_this(self):
        """Why the defect spares synthetic runs and hits human-driven ones."""
        step = FakeLLMProposer().propose(
            symptom="training hangs at step 40",
            cell_summaries=BASELINE_FAILED,
            candidates=CANDIDATES,
            tried=[],
        )
        assert step.next_mitigations == ["tf32_off"]
        assert step.unresolved_mitigations == []


# ── policy must not be the thing that loses them ──────────────────────────


class TestValidationPreservesTheRejections:
    def test_validate_step_carries_the_field_through(self):
        """FAILS BEFORE THE FIX: validate_step rebuilds the step from scratch."""
        step = AgentStep(
            category="rccl_hang",
            hypothesis="peer-to-peer transport is wedged",
            next_mitigations=["tf32_off"],
            confidence=0.9,
            stop=False,
            unresolved_mitigations=[UNREGISTERED],
        )
        validated = AgentPolicy().validate_step(step)
        assert validated.unresolved_mitigations == [UNREGISTERED]
        assert validated.next_mitigations == ["tf32_off"]

    def test_validation_still_rejects_an_unregistered_name_that_reaches_it(self):
        """A custom proposer without the filter must still fail fast."""
        step = AgentStep(
            category="rccl_hang",
            hypothesis="",
            next_mitigations=[UNREGISTERED],
            confidence=0.5,
            stop=False,
        )
        with pytest.raises(PolicyViolation, match=UNREGISTERED):
            AgentPolicy().validate_step(step)


# ── attribution ───────────────────────────────────────────────────────────


class TestStopAttribution:
    def _step(self, **kw):
        base = {
            "category": "rccl_hang",
            "hypothesis": "peer-to-peer transport is wedged; disable it",
            "next_mitigations": [],
            "confidence": 0.9,
            "stop": False,
        }
        base.update(kw)
        return AgentStep(**base)

    def test_a_filtered_to_empty_proposal_is_not_an_agent_request(self):
        """FAILS BEFORE THE FIX: this returned agent_stop / agent_requested.

        The sharpest form of the defect -- the model set ``stop: false`` and
        the loop recorded that the agent requested the stop.
        """
        step = self._step(unresolved_mitigations=[UNREGISTERED])
        outcome, message, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert reason == "proposal_unresolved"
        assert outcome == "proposal_unresolved"
        assert UNREGISTERED in message

    def test_the_operator_is_not_shown_the_hypothesis_as_the_reason(self):
        """It is a plausible rationale for a stop the model never asked for."""
        step = self._step(unresolved_mitigations=[UNREGISTERED])
        _outcome, message, _reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert step.hypothesis not in message
        assert "NOT" in message

    def test_a_genuine_stop_is_untouched(self):
        step = self._step(
            hypothesis="no further hypotheses from this evidence",
            stop=True,
            stop_reason="agent_requested",
        )
        outcome, message, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")
        assert message == step.hypothesis

    def test_the_two_cases_are_now_distinguishable(self):
        """B and D from the issue's reproduction agreed on every recorded field."""
        dropped = _resolve_stop_outcome(
            self._step(unresolved_mitigations=[UNREGISTERED]), BASELINE_FAILED
        )
        genuine = _resolve_stop_outcome(
            self._step(stop=True, stop_reason="agent_requested"), BASELINE_FAILED
        )
        assert dropped[0] != genuine[0]
        assert dropped[2] != genuine[2]

    def test_an_explicit_stop_wins_even_when_names_were_dropped(self):
        """A self-contradicting reply: the model did ask to stop, so honour it.

        Control flow and attribution both stay as they are; the names are
        still written to the log, so the contradiction remains auditable.
        """
        step = self._step(stop=True, stop_reason="agent_requested",
                          unresolved_mitigations=[UNREGISTERED])
        outcome, _message, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_a_stop_with_no_reason_from_a_custom_proposer_is_its_own(self):
        """FAILS BEFORE THE FIX: recorded as proposal_unresolved.

        ``_step_from_content`` fills in ``agent_requested`` for a stop, but a
        proposer written against the protocol can return ``stop=True`` with no
        reason and dropped names. It asked to stop; the loop must say so.
        """
        step = self._step(stop=True, unresolved_mitigations=[UNREGISTERED])
        outcome, _message, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_the_operator_message_names_the_baseline_as_a_cause(self):
        """The filter also drops ``none``: a registered, allowed, untried name
        that is still never a candidate. The message must not tell the operator
        it was unregistered or already tried."""
        step = self._step(unresolved_mitigations=["none"])
        outcome, message, _reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert outcome == "proposal_unresolved"
        assert "`none` baseline" in message

    def test_a_passing_baseline_still_outranks_everything(self):
        """baseline_pass is a deterministic probe verdict, not an attribution call."""
        step = self._step(unresolved_mitigations=[UNREGISTERED])
        summaries = [{"cell_name": "none-none", "verdict": "pass"}]
        outcome, _message, reason = _resolve_stop_outcome(step, summaries)
        assert (outcome, reason) == ("baseline_pass", "baseline_pass")

    def test_an_empty_proposal_with_no_rejections_is_unchanged(self):
        """Not this defect: nothing was dropped, so there is nothing to attribute."""
        outcome, _message, reason = _resolve_stop_outcome(self._step(), BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_the_exhausted_heuristic_is_unchanged(self):
        step = self._step(hypothesis="No remaining registered mitigations to try.")
        outcome, _message, reason = _resolve_stop_outcome(step, [])
        assert (outcome, reason) == ("exhausted_candidates", "exhausted_candidates")

    @pytest.mark.parametrize(
        "doc", ["docs/agent/aorta-probe-agent.md", "docs/agent/agentic-testing-guide.md"]
    )
    def test_the_outcome_tables_name_the_baseline_as_a_cause(self, doc):
        row = next(
            line for line in (REPO / doc).read_text(encoding="utf-8").splitlines()
            if line.startswith("| `proposal_unresolved`")
        )
        assert "`none` baseline" in row

    def test_the_cli_has_a_headline_for_the_new_outcome(self):
        """An unmapped outcome degrades to a bare "Finished with outcome:" line."""
        from aorta.cli.agent_mitigate import _OUTCOME_HEADLINES

        assert "proposal_unresolved" in _OUTCOME_HEADLINES


# ── the log ───────────────────────────────────────────────────────────────

FROZEN_TS = "2026-09-17T00:00:00+00:00"


@pytest.fixture()
def loop_env(monkeypatch, tmp_path):
    """Run the real loop with only run_recipe, the verdicts and the clock stubbed."""
    import aorta.agent.loop as loop_mod
    import aorta.agent.state as state_mod

    run_dir = tmp_path / "out" / "A449"
    monkeypatch.setattr(loop_mod, "run_recipe", MagicMock(return_value=run_dir))
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", lambda _d: BASELINE_FAILED)
    monkeypatch.setattr(state_mod, "_utc_now_iso", lambda: FROZEN_TS)

    def _run(proposer):
        config = AgentConfig(
            output_dir=tmp_path / "out",
            ticket="A449",
            subprocess_argv=("echo", "hi"),
            policy=AgentPolicy(max_iterations=3),
            mitigations_allowlist=tuple(CANDIDATES),
            llm_backend="openai",
        )
        result = run_agent_loop(config, proposer=proposer)
        return result, (run_dir / "agent_log.jsonl").read_text(encoding="utf-8")

    return _run


def _records(log_text: str) -> list[dict]:
    return [json.loads(line) for line in log_text.splitlines() if line.strip()]


class TestTheLogRecord:
    def test_a_genuine_stop_writes_a_byte_identical_log(self, loop_env, proposer):
        """The archived-trajectory guarantee, as a literal comparison.

        A golden rather than a key-set assertion: anything added to these two
        events -- even an empty ``unresolved_mitigations: []`` -- changes the
        bytes of every run that ever recorded a genuine stop, and those are
        the runs already on disk.
        """
        _result, log_text = loop_env(
            proposer(
                _reply(
                    category="unknown",
                    hypothesis="no further hypotheses from this evidence",
                    next_mitigations=[],
                    confidence=0.3,
                    stop=True,
                )
            )
        )
        expected = (
            '{"argv": ["echo", "hi"], "llm_backend": "openai", "symptom": null, '
            '"ticket": "A449", "ticket_slug": "A449", '
            f'"ts": "{FROZEN_TS}", "type": "session_start"}}\n'
            '{"category": "unknown", "confidence": 0.3, '
            '"hypothesis": "no further hypotheses from this evidence", '
            '"next_mitigations": [], "stop": true, "stop_reason": "agent_requested", '
            f'"ts": "{FROZEN_TS}", "type": "llm_step"}}\n'
            '{"outcome": "agent_stop", "stop_reason": "agent_requested", '
            f'"ts": "{FROZEN_TS}", "type": "search_stopped"}}\n'
        )
        assert log_text == expected

    def test_the_new_key_is_absent_and_not_empty_when_nothing_is_dropped(
        self, loop_env, proposer
    ):
        """Absent, so a harvester's `"unresolved_mitigations" in record` still works."""
        _result, log_text = loop_env(
            proposer(_reply(next_mitigations=[], confidence=0.3, stop=True))
        )
        for record in _records(log_text):
            assert "unresolved_mitigations" not in record

    def test_a_filtered_to_empty_proposal_writes_the_names(self, loop_env, proposer):
        """FAILS BEFORE THE FIX: the names were retained nowhere at all."""
        result, log_text = loop_env(
            proposer(_reply(category="rccl_hang", next_mitigations=[UNREGISTERED], stop=False))
        )
        assert result.outcome == "proposal_unresolved"
        records = _records(log_text)
        llm_step = next(r for r in records if r["type"] == "llm_step")
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert llm_step["next_mitigations"] == []
        assert llm_step["unresolved_mitigations"] == [UNREGISTERED]
        assert llm_step["stop"] is False
        assert stopped["stop_reason"] == "proposal_unresolved"
        assert stopped["unresolved_mitigations"] == [UNREGISTERED]

    def test_the_recorded_stop_no_longer_contradicts_the_model(self, loop_env, proposer):
        """`stop: false` in the same record as a stop the agent attributed to it."""
        _result, log_text = loop_env(
            proposer(_reply(category="rccl_hang", next_mitigations=[UNREGISTERED], stop=False))
        )
        records = _records(log_text)
        llm_step = next(r for r in records if r["type"] == "llm_step")
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert llm_step["stop"] is False
        assert stopped["stop_reason"] != "agent_requested"

    def test_a_partial_rejection_is_recorded_and_the_search_continues(
        self, loop_env, proposer
    ):
        """The quiet variant. The loop runs the surviving half and carries on,
        so the ``llm_step`` record is the only place the discarded half can be
        written -- a stop event would come later, or never.
        """
        _result, log_text = loop_env(
            proposer(
                [
                    _reply(
                        category="rccl_hang",
                        next_mitigations=[UNREGISTERED, "tf32_off"],
                        stop=False,
                    ),
                    _reply(category="unknown", next_mitigations=[], stop=True),
                ]
            )
        )
        records = _records(log_text)
        first_step = next(r for r in records if r["type"] == "llm_step")
        assert first_step["next_mitigations"] == ["tf32_off"]
        assert first_step["unresolved_mitigations"] == [UNREGISTERED]
        assert first_step["stop"] is False
        # The search really did continue: the surviving name was run, and the
        # stop that eventually came is the second reply's, not this one's.
        assert [r["mitigation"] for r in records if r["type"] == "mitigation_tried"] == [
            "tf32_off"
        ]
        assert [r["iteration"] for r in records if r["type"] == "iteration_complete"] == [1]
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert stopped["stop_reason"] == "agent_requested"
        assert "unresolved_mitigations" not in stopped


# ── resume ────────────────────────────────────────────────────────────────


class TestWakeStillReconstructs:
    def test_a_dropped_name_is_never_treated_as_tried(self, tmp_path):
        """It was rejected, not attempted; a resumed run must still offer it."""
        run_dir = tmp_path / "A449"
        run_dir.mkdir()
        (run_dir / "agent_log.jsonl").write_text(
            "\n".join(
                json.dumps(record, sort_keys=True)
                for record in [
                    {"type": "session_start", "ticket": "A449", "ts": FROZEN_TS},
                    {
                        "type": "llm_step",
                        "category": "rccl_hang",
                        "hypothesis": "peer-to-peer transport is wedged",
                        "next_mitigations": [],
                        "confidence": 0.9,
                        "stop": False,
                        "stop_reason": None,
                        "unresolved_mitigations": [UNREGISTERED],
                        "ts": FROZEN_TS,
                    },
                    {
                        "type": "search_stopped",
                        "outcome": "proposal_unresolved",
                        "stop_reason": "proposal_unresolved",
                        "unresolved_mitigations": [UNREGISTERED],
                        "ts": FROZEN_TS,
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state = wake(run_dir, ticket="A449")
        assert state.tried_mitigations == []
        assert UNREGISTERED not in state.tried_mitigations
        assert state.last_category == "rccl_hang"
        assert state.last_hypothesis == "peer-to-peer transport is wedged"
        assert state.iterations_completed == 0
        assert state.converged is False

    def test_a_log_written_before_the_key_existed_still_replays(self, tmp_path):
        """The archived-trajectory read path: no new key, and none required."""
        run_dir = tmp_path / "OLD"
        run_dir.mkdir()
        (run_dir / "agent_log.jsonl").write_text(
            "\n".join(
                json.dumps(record, sort_keys=True)
                for record in [
                    {"type": "session_start", "ticket": "OLD", "ts": FROZEN_TS},
                    {
                        "type": "llm_step",
                        "category": "illegal_mem",
                        "hypothesis": "a race in the accumulation workspace",
                        "next_mitigations": ["tf32_off"],
                        "confidence": 0.65,
                        "stop": False,
                        "stop_reason": None,
                        "ts": FROZEN_TS,
                    },
                    {"type": "mitigation_tried", "mitigation": "tf32_off", "ts": FROZEN_TS},
                    {"type": "iteration_complete", "iteration": 1, "ts": FROZEN_TS},
                    {
                        "type": "converged",
                        "winning_mitigation": "tf32_off",
                        "ts": FROZEN_TS,
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state = wake(run_dir, ticket="OLD")
        assert state.tried_mitigations == ["tf32_off"]
        assert state.last_category == "illegal_mem"
        assert state.iterations_completed == 1
        assert state.converged is True
        assert state.winning_mitigation == "tf32_off"
