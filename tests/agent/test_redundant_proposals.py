"""aorta#501: a proposal validation empties must not be recorded as an agent stop.

``AgentPolicy.validate_step`` removes the ``none`` baseline and every repeated
name from ``next_mitigations``. Both removals are correct -- neither adds a
cell -- but neither was recorded, so a proposal of only ``none`` came back
empty and the loop read the empty list as the proposer choosing to stop.

The shipped real-LLM proposers never reach this: their candidate filter drops
``none`` first and records it under ``unresolved_mitigations`` (aorta#449).
The path that does is a proposer written against the ``LLMProposer`` protocol,
which is why most tests here hand ``validate_step`` an ``AgentStep`` directly.

What is protected here:

* the removed names survive, under their own key -- they resolved, and nothing
  declined them, so they are not ``unresolved_mitigations``;
* a baseline-only proposal no longer shares a ``stop_reason`` with a proposer
  that concluded;
* a run with nothing removed writes the log it wrote before.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aorta.agent.llm import AgentStep, ChatProviderProposer
from aorta.agent.loop import AgentConfig, _resolve_stop_outcome, run_agent_loop
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.state import wake

REPO = Path(__file__).resolve().parents[2]

CANDIDATES = ["tf32_off", "hsa_no_sdma"]

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


def _step(next_mitigations, *, stop=False, stop_reason=None, **kw) -> AgentStep:
    return AgentStep(
        category=kw.pop("category", "rccl_hang"),
        hypothesis=kw.pop("hypothesis", "the baseline already passes; nothing to mitigate"),
        next_mitigations=list(next_mitigations),
        confidence=kw.pop("confidence", 0.9),
        stop=stop,
        stop_reason=stop_reason,
        **kw,
    )


def _validated(next_mitigations, **kw) -> AgentStep:
    return AgentPolicy().validate_step(_step(next_mitigations, **kw))


# ── validation keeps what it removes ─────────────────────────────────────


class TestValidationRecordsWhatItRemoves:
    def test_the_baseline_is_still_removed(self):
        assert _validated(["none"]).next_mitigations == []

    def test_but_it_is_now_recorded(self):
        """FAILS BEFORE THE FIX: the removal was kept nowhere."""
        assert _validated(["none"]).redundant_mitigations == ["none"]

    def test_every_repeat_is_recorded_and_one_copy_is_kept(self):
        step = _validated(["tf32_off", "tf32_off", "tf32_off"])
        assert step.next_mitigations == ["tf32_off"]
        assert step.redundant_mitigations == ["tf32_off", "tf32_off"]

    def test_kept_plus_redundant_is_the_proposal(self):
        proposal = ["none", "tf32_off", "hsa_no_sdma", "tf32_off", "none"]
        step = _validated(proposal)
        assert step.next_mitigations == ["tf32_off", "hsa_no_sdma"]
        assert Counter(step.next_mitigations + step.redundant_mitigations) == Counter(proposal)

    def test_a_clean_proposal_records_nothing(self):
        assert _validated(["tf32_off", "hsa_no_sdma"]).redundant_mitigations == []

    def test_redundancy_is_not_folded_into_unresolved(self):
        """The two keys mean different things: these names resolved."""
        step = _validated(["none"], unresolved_mitigations=["rccl_p2p_disable"])
        assert step.unresolved_mitigations == ["rccl_p2p_disable"]
        assert step.redundant_mitigations == ["none"]

    def test_an_unregistered_name_still_raises(self):
        with pytest.raises(PolicyViolation):
            _validated(["none", "rccl_p2p_disable"])

    def test_a_second_pass_records_only_what_it_removed(self):
        """The record is per pass: validation is the field's only writer."""
        policy = AgentPolicy()
        once = policy.validate_step(_step(["none", "tf32_off", "tf32_off"]))
        twice = policy.validate_step(once)
        assert once.redundant_mitigations == ["none", "tf32_off"]
        assert twice.next_mitigations == ["tf32_off"]
        assert twice.redundant_mitigations == []

    def test_a_supplied_record_is_not_trusted(self):
        """FAILS BEFORE THE FIX: a proposer could hand validation its own record."""
        assert _validated([], redundant_mitigations=["none"]).redundant_mitigations == []

    def test_the_model_cannot_supply_the_field(self):
        step = AgentStep.from_dict(
            {"next_mitigations": ["tf32_off"], "redundant_mitigations": ["hsa_no_sdma"]}
        )
        assert step.redundant_mitigations == []


# ── the stop is attributed to what happened ──────────────────────────────


class TestStopAttribution:
    def test_a_baseline_only_proposal_is_not_an_agent_request(self):
        """FAILS BEFORE THE FIX: recorded as agent_stop / agent_requested."""
        outcome, _msg, reason = _resolve_stop_outcome(_validated(["none"]), BASELINE_FAILED)
        assert (outcome, reason) == ("proposal_redundant", "proposal_redundant")

    def test_twice_is_the_same_as_once(self):
        outcome, _msg, _reason = _resolve_stop_outcome(
            _validated(["none", "none"]), BASELINE_FAILED
        )
        assert outcome == "proposal_redundant"

    def test_the_operator_is_not_shown_the_hypothesis_as_the_reason(self):
        _outcome, msg, _reason = _resolve_stop_outcome(_validated(["none"]), BASELINE_FAILED)
        assert "nothing to mitigate" not in msg
        assert "['none']" in msg
        assert "redundant_mitigations" in msg

    def test_an_explicit_stop_is_still_the_proposers_own(self):
        """The narrowness case: `stop: true` is a request whatever else it carries."""
        outcome, _msg, reason = _resolve_stop_outcome(
            _validated(["none"], stop=True), BASELINE_FAILED
        )
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_a_repeat_alone_never_empties_the_proposal(self):
        step = _validated(["tf32_off", "tf32_off"])
        assert step.next_mitigations == ["tf32_off"]
        assert not step.stop

    def test_an_unresolved_name_outranks_a_redundant_one(self):
        step = _validated(["none"], unresolved_mitigations=["rccl_p2p_disable"])
        outcome, _msg, _reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert outcome == "proposal_unresolved"

    def test_a_proposer_cannot_claim_the_loops_reason(self):
        """FAILS BEFORE THE FIX: the claimed reason was recorded as given."""
        step = _step([], stop=True, stop_reason="proposal_redundant")
        outcome, _msg, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_a_claimed_reason_is_rederived_not_ignored(self):
        step = AgentPolicy().validate_step(
            _step(["none"], stop_reason="proposal_redundant")
        )
        outcome, _msg, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("proposal_redundant", "proposal_redundant")

    @pytest.mark.parametrize(
        "claimed", ["agent_requested", "exhausted_candidates", "baseline_pass"]
    )
    def test_a_reason_without_a_stop_is_not_taken_as_given(self, claimed):
        """FAILS BEFORE THE FIX: `stop: false` plus any reason skipped the derivation."""
        step = _validated(["none"], stop_reason=claimed)
        outcome, _msg, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("proposal_redundant", "proposal_redundant")

    def test_the_same_holds_for_an_unresolved_proposal(self):
        """FAILS BEFORE THE FIX: the aorta#449 twin of the case above."""
        step = _validated(
            [], stop_reason="agent_requested", unresolved_mitigations=["rccl_p2p_disable"]
        )
        outcome, _msg, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("proposal_unresolved", "proposal_unresolved")

    @pytest.mark.parametrize(
        "claimed", ["agent_requested", "exhausted_candidates", "baseline_pass"]
    )
    def test_without_a_stop_a_reason_changes_nothing(self, claimed):
        """Keyed on `stop`, not on what was removed: inert on any empty proposal."""
        assert _resolve_stop_outcome(
            _validated([], stop_reason=claimed), BASELINE_FAILED
        ) == _resolve_stop_outcome(_validated([]), BASELINE_FAILED)

    def test_a_stop_keeps_the_reason_it_gave(self):
        """The narrowness case: a proposer that did stop is taken at its word."""
        step = _validated(["none"], stop=True, stop_reason="exhausted_candidates")
        outcome, _msg, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("exhausted_candidates", "exhausted_candidates")

    def test_a_malformed_supplied_record_cannot_reach_the_stop_message(self):
        """FAILS BEFORE THE FIX: an unhashable entry raised out of the operator message."""
        step = _validated([], redundant_mitigations=[["none"]])
        outcome, _msg, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_the_model_cannot_claim_it_either(self):
        step = AgentStep.from_dict(
            {"next_mitigations": [], "stop": True, "stop_reason": "proposal_redundant"}
        )
        assert step.stop_reason is None

    def test_a_passing_baseline_still_outranks_it(self):
        outcome, _msg, _reason = _resolve_stop_outcome(
            _validated(["none"]), [{"cell_name": "none-none", "verdict": "pass"}]
        )
        assert outcome == "baseline_pass"

    @pytest.mark.parametrize(
        "doc", ["docs/agent/aorta-probe-agent.md", "docs/agent/agentic-testing-guide.md"]
    )
    def test_the_outcome_tables_carry_the_new_outcome(self, doc):
        rows = [
            line for line in (REPO / doc).read_text(encoding="utf-8").splitlines()
            if line.startswith("| `proposal_redundant`")
        ]
        assert len(rows) == 1
        assert "`redundant_mitigations`" in rows[0]

    def test_the_cli_has_a_headline_for_the_new_outcome(self):
        from aorta.cli.agent_mitigate import _OUTCOME_HEADLINES

        assert "proposal_redundant" in _OUTCOME_HEADLINES


# ── the log ───────────────────────────────────────────────────────────────

FROZEN_TS = "2026-09-29T00:00:00+00:00"


class _Scripted:
    """A proposer written against the protocol, bypassing the shipped filter."""

    def __init__(self, *steps: AgentStep) -> None:
        self._steps = list(steps)
        self.calls = 0

    def propose(self, *, symptom, cell_summaries, candidates, tried):
        step = self._steps[min(self.calls, len(self._steps) - 1)]
        self.calls += 1
        return step


@pytest.fixture()
def loop_env(monkeypatch, tmp_path):
    """Run the real loop with only run_recipe, the verdicts and the clock stubbed."""
    import aorta.agent.loop as loop_mod
    import aorta.agent.state as state_mod

    run_dir = tmp_path / "out" / "A501"
    monkeypatch.setattr(loop_mod, "run_recipe", MagicMock(return_value=run_dir))
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", lambda _d: BASELINE_FAILED)
    monkeypatch.setattr(state_mod, "_utc_now_iso", lambda: FROZEN_TS)

    def _run(proposer):
        config = AgentConfig(
            output_dir=tmp_path / "out",
            ticket="A501",
            subprocess_argv=("echo", "hi"),
            policy=AgentPolicy(max_iterations=3),
            mitigations_allowlist=tuple(CANDIDATES),
            llm_backend="openai",
        )
        result = run_agent_loop(config, proposer=proposer)
        return result, [
            json.loads(line)
            for line in (run_dir / "agent_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return _run


def _chat_proposer(monkeypatch, *replies: str) -> ChatProviderProposer:
    calls = {"n": 0}

    def _invoke(_messages):
        reply = replies[min(calls["n"], len(replies) - 1)]
        calls["n"] += 1
        return SimpleNamespace(content=reply)

    monkeypatch.setattr(
        ChatProviderProposer, "_chat_model", lambda self: SimpleNamespace(invoke=_invoke)
    )
    return ChatProviderProposer("openai")


class TestTheLogRecord:
    def test_a_baseline_only_proposal_writes_the_names(self, loop_env):
        """FAILS BEFORE THE FIX: recorded as a genuine agent stop."""
        result, records = loop_env(_Scripted(_step(["none"])))
        assert result.outcome == "proposal_redundant"
        llm_step = next(r for r in records if r["type"] == "llm_step")
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert llm_step["next_mitigations"] == []
        assert llm_step["redundant_mitigations"] == ["none"]
        assert llm_step["stop"] is False
        assert stopped["stop_reason"] == "proposal_redundant"
        assert stopped["redundant_mitigations"] == ["none"]
        assert "unresolved_mitigations" not in stopped

    @pytest.mark.parametrize(
        "claimed", ["agent_requested", "exhausted_candidates", "baseline_pass"]
    )
    def test_a_reason_without_a_stop_does_not_change_the_record(self, loop_env, claimed):
        """FAILS BEFORE THE FIX: recorded as agent_stop, or as exhausted_candidates."""
        result, records = loop_env(_Scripted(_step(["none"], stop_reason=claimed)))
        llm_step = next(r for r in records if r["type"] == "llm_step")
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert (llm_step["stop"], llm_step["stop_reason"]) == (False, claimed)
        assert (result.outcome, stopped["stop_reason"]) == (
            "proposal_redundant",
            "proposal_redundant",
        )

    def test_a_supplied_record_cannot_claim_the_loops_outcome(self, loop_env):
        """FAILS BEFORE THE FIX: recorded as proposal_redundant with nothing removed."""
        result, records = loop_env(_Scripted(_step([], redundant_mitigations=["none"])))
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert (result.outcome, stopped["stop_reason"]) == ("agent_stop", "agent_requested")
        for record in records:
            assert "redundant_mitigations" not in record

    def test_a_supplied_name_is_not_logged_as_removed(self, loop_env):
        """FAILS BEFORE THE FIX: a name validation never checked reached the log."""
        _result, records = loop_env(
            _Scripted(
                _step(["tf32_off"], redundant_mitigations=["not a name"]),
                _step([], stop=True, hypothesis="done"),
            )
        )
        for record in records:
            assert "redundant_mitigations" not in record

    def test_a_repeat_is_recorded_and_the_search_continues(self, loop_env, monkeypatch):
        """Through the shipped parser and filter, which keep repeats for validation."""
        result, records = loop_env(
            _chat_proposer(
                monkeypatch,
                json.dumps({"category": "rccl_hang", "hypothesis": "h",
                            "next_mitigations": ["tf32_off", "tf32_off"],
                            "confidence": 0.8, "stop": False}),
                json.dumps({"category": "unknown", "hypothesis": "done",
                            "next_mitigations": [], "confidence": 0.3, "stop": True}),
            )
        )
        first = next(r for r in records if r["type"] == "llm_step")
        assert first["next_mitigations"] == ["tf32_off"]
        assert first["redundant_mitigations"] == ["tf32_off"]
        assert [r["mitigation"] for r in records if r["type"] == "mitigation_tried"] == [
            "tf32_off"
        ]
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert (result.outcome, stopped["stop_reason"]) == ("agent_stop", "agent_requested")
        assert "redundant_mitigations" not in stopped

    def test_the_key_is_absent_and_not_empty_when_nothing_is_removed(self, loop_env):
        _result, records = loop_env(
            _Scripted(_step(["tf32_off"]), _step([], stop=True, hypothesis="done"))
        )
        for record in records:
            assert "redundant_mitigations" not in record

    def test_a_model_none_still_reaches_the_filter_first(self, loop_env, monkeypatch):
        """aorta#449's route is unchanged: the filter drops a model's `none`."""
        result, records = loop_env(
            _chat_proposer(
                monkeypatch,
                json.dumps({"category": "rccl_hang", "hypothesis": "h",
                            "next_mitigations": ["none"], "confidence": 0.8,
                            "stop": False}),
            )
        )
        assert result.outcome == "proposal_unresolved"
        llm_step = next(r for r in records if r["type"] == "llm_step")
        assert llm_step["unresolved_mitigations"] == ["none"]
        assert "redundant_mitigations" not in llm_step


class TestWake:
    def test_a_redundant_name_is_never_treated_as_tried(self, tmp_path):
        run_dir = tmp_path / "A501"
        run_dir.mkdir()
        (run_dir / "agent_log.jsonl").write_text(
            "\n".join(
                json.dumps(record, sort_keys=True)
                for record in [
                    {"type": "session_start", "ticket": "A501", "ts": FROZEN_TS},
                    {"type": "llm_step", "category": "rccl_hang", "hypothesis": "h",
                     "next_mitigations": [], "confidence": 0.9, "stop": False,
                     "stop_reason": None, "redundant_mitigations": ["none"],
                     "ts": FROZEN_TS},
                    {"type": "search_stopped", "outcome": "proposal_redundant",
                     "stop_reason": "proposal_redundant",
                     "redundant_mitigations": ["none"], "ts": FROZEN_TS},
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state = wake(run_dir, ticket="A501")
        assert state.tried_mitigations == []
        assert state.last_category == "rccl_hang"
