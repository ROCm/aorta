"""The diagnostic-axis twin of aorta#449: a dropped diagnostic is kept nowhere.

``_step_from_content`` filters the model's proposed diagnostics against the
offered set. A name that is not on it is discarded and retained nowhere: it
never reaches ``plan_axis_growth``, so it cannot appear in
``axis_growth.rejected_diagnostics``, and the loop-level allowlist guard that
would have raised on it sits *after* this filter has already removed it. The
episode therefore records a proposal the model did not make, and the discarded
half can be detected but not reconstructed.

Two things this file is careful about:

* **The naming.** ``axis_growth`` already carries ``rejected_diagnostics``,
  meaning a good name the cell budget had no room for. That name still appears
  in ``next_diagnostics``; an unresolved one never does. Merging them would
  put a cost decision and a name-resolution failure under one key, which is
  the conflation #449 exists to remove. Hence ``unresolved_diagnostics``.
* **Byte identity.** The key is written only when something was dropped, so a
  run with no drops emits the log it emitted before the key existed. The two
  archived episodes are compared against that, by hash.

There is no misattribution half being fixed here -- no ``stop_reason`` moves.
See ``test_a_diagnostics_only_stop_is_still_attributed_to_the_agent``, which
pins the current attribution rather than changing it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aorta.agent.llm import AgentStep, ChatProviderProposer, _step_from_content
from aorta.agent.loop import AgentConfig, _resolve_stop_outcome, run_agent_loop
from aorta.agent.policy import AgentPolicy
from aorta.agent.state import wake

#: Registered, but not on the offered diagnostic axis in these tests.
UNOFFERED = "hip_launch_blocking"
OFFERED = "amd_log_level_4"

MITIGATIONS = ["none", "tf32_off", "hsa_no_sdma"]

BASELINE_FAILED = [
    {
        "cell_name": "none-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier4:nan_signature"],
        "warn_detectors_fired": [],
        "capture": {},
        "exit_code": None,
    }
]


def _reply(**payload) -> str:
    body = {
        "category": "illegal_mem",
        "hypothesis": "uninitialised read; want the log to see it",
        "next_mitigations": [],
        "confidence": 0.6,
        "stop": False,
    }
    body.update(payload)
    return json.dumps(body)


@pytest.fixture()
def proposer(monkeypatch):
    """A real ChatProviderProposer with only the HTTP transport stubbed.

    ``llm.py``, ``policy.py`` and ``loop.py`` are all real code here, so the
    filter under test is the production one.
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


def _propose(proposer, tried_diagnostics=None):
    return proposer.propose(
        symptom="loss goes NaN",
        cell_summaries=BASELINE_FAILED,
        candidates=MITIGATIONS,
        tried=[],
        diagnostic_candidates=[OFFERED],
        tried_diagnostics=tried_diagnostics or [],
    )


# ── the filter keeps what it drops ────────────────────────────────────────


class TestTheDiagnosticFilterRecordsWhatItDrops:
    def test_an_unoffered_diagnostic_is_still_dropped(self, proposer):
        """Unchanged: the model does not get to arm an axis the operator didn't."""
        step = _propose(proposer(_reply(next_diagnostics=[UNOFFERED])))
        assert step.next_diagnostics == []

    def test_but_the_dropped_diagnostic_is_now_retained(self, proposer):
        """FAILS BEFORE THE FIX: AgentStep had no field to retain it."""
        step = _propose(proposer(_reply(next_diagnostics=[UNOFFERED])))
        assert step.unresolved_diagnostics == [UNOFFERED]

    def test_a_partial_rejection_keeps_both_halves(self, proposer):
        step = _propose(proposer(_reply(next_diagnostics=[UNOFFERED, OFFERED])))
        assert step.next_diagnostics == [OFFERED]
        assert step.unresolved_diagnostics == [UNOFFERED]

    def test_the_model_proposal_is_reconstructible(self, proposer):
        """What makes an affected episode repairable rather than discardable."""
        proposed = [UNOFFERED, OFFERED, "hsa_disable_cache"]
        step = _propose(proposer(_reply(next_diagnostics=proposed)))
        reconstructed = [
            d
            for d in proposed
            if d in step.next_diagnostics or d in step.unresolved_diagnostics
        ]
        assert reconstructed == proposed

    def test_an_already_tried_diagnostic_counts_as_unresolved_too(self, proposer):
        """The filter drops on the *remaining* set, not on registry membership."""
        step = _propose(
            proposer(_reply(next_diagnostics=[OFFERED])), tried_diagnostics=[OFFERED]
        )
        assert step.next_diagnostics == []
        assert step.unresolved_diagnostics == [OFFERED]

    def test_a_clean_diagnostic_proposal_records_no_drops(self, proposer):
        step = _propose(proposer(_reply(next_diagnostics=[OFFERED])))
        assert step.next_diagnostics == [OFFERED]
        assert step.unresolved_diagnostics == []

    def test_a_reply_naming_no_diagnostic_records_no_drops(self, proposer):
        """The single-axis shape, which is most existing replies."""
        step = _propose(proposer(_reply(next_mitigations=["tf32_off"])))
        assert step.next_diagnostics == []
        assert step.unresolved_diagnostics == []

    def test_the_drop_is_warned_about(self, proposer, caplog):
        """Nothing raised and nothing was logged before; silence was the defect."""
        with caplog.at_level("WARNING", logger="aorta.agent.llm"):
            _propose(proposer(_reply(next_diagnostics=[UNOFFERED])))
        assert UNOFFERED in caplog.text

    def test_the_model_cannot_claim_diagnostics_were_dropped(self, proposer):
        """The field is the agent's finding, not something a reply may assert."""
        step = _propose(
            proposer(
                _reply(next_diagnostics=[OFFERED], unresolved_diagnostics=["invented"])
            )
        )
        assert step.unresolved_diagnostics == []


# ── the remaining_diagnostics=None default ────────────────────────────────


class TestTheOfferedSetMustBeStatedExplicitly:
    def test_omitting_the_offered_set_is_now_an_error(self):
        """FAILS BEFORE THE FIX: the None default discarded 100%, silently.

        The dangerous property was not the policy but the ambiguity: a caller
        that deliberately offered nothing and a caller that forgot the
        argument produced the same total discard, on the one parameter that
        decides whether the model's diagnostic proposal survives at all.
        """
        with pytest.raises(TypeError):
            _step_from_content(  # type: ignore[call-arg]
                _reply(next_diagnostics=[OFFERED]), ["tf32_off"]
            )

    def test_an_explicit_empty_offer_still_discards_everything(self):
        """The policy is unchanged: an unarmed axis authorises nothing on it."""
        step = _step_from_content(
            _reply(next_diagnostics=[OFFERED]), ["tf32_off"], []
        )
        assert step.next_diagnostics == []

    def test_but_it_is_no_longer_silent(self):
        """FAILS BEFORE THE FIX. The discard is a policy, so it should be legible."""
        step = _step_from_content(
            _reply(next_diagnostics=[OFFERED]), ["tf32_off"], []
        )
        assert step.unresolved_diagnostics == [OFFERED]

    def test_the_two_real_proposers_were_already_passing_it(self, proposer):
        """Why requiring it changes no production behaviour.

        Both backends compute the remaining diagnostics and pass them, so the
        None default was reachable only from a new caller or a test.
        """
        step = _propose(proposer(_reply(next_diagnostics=[OFFERED])))
        assert step.next_diagnostics == [OFFERED]


# ── validation must not lose them ─────────────────────────────────────────


class TestValidationPreservesTheDrops:
    def test_validate_step_carries_the_field_through(self):
        """FAILS BEFORE THE FIX: validate_step rebuilds the step from scratch."""
        step = AgentStep(
            category="illegal_mem",
            hypothesis="h",
            next_mitigations=["tf32_off"],
            confidence=0.6,
            stop=False,
            next_diagnostics=[OFFERED],
            unresolved_diagnostics=[UNOFFERED],
        )
        validated = AgentPolicy().validate_step(step)
        assert validated.unresolved_diagnostics == [UNOFFERED]
        assert validated.next_diagnostics == [OFFERED]


# ── attribution is deliberately unchanged ─────────────────────────────────


class TestAttributionIsUnchanged:
    def test_a_diagnostics_only_stop_is_still_attributed_to_the_agent(self):
        """This PINS today's attribution; it does not change it.

        The loop's stop check is ``step.stop or not (next_mitigations or
        next_diagnostics)``, so a diagnostics-only proposal filtered to empty
        does reach it and is recorded as ``agent_requested`` -- the same
        misattribution shape as #449, on the other axis. Fixing that means a
        new stop reason, which is a decision for the branch owner, so this
        asserts the status quo rather than a preference, and will fail loudly
        if someone changes it without meaning to. The names are in the log
        either way, which is what makes the stop diagnosable meanwhile.

        (It cannot run on the unmodified tree only because the field it
        constructs does not exist there; the attribution it asserts is
        unchanged by this commit.)
        """
        step = AgentStep(
            category="illegal_mem",
            hypothesis="need visibility before testing a cause",
            next_mitigations=[],
            confidence=0.8,
            stop=False,
            next_diagnostics=[],
            unresolved_diagnostics=[UNOFFERED],
        )
        outcome, _message, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")

    def test_a_genuine_stop_is_untouched(self):
        step = AgentStep(
            category="unknown",
            hypothesis="no further hypotheses",
            next_mitigations=[],
            confidence=0.3,
            stop=True,
            stop_reason="agent_requested",
        )
        outcome, message, reason = _resolve_stop_outcome(step, BASELINE_FAILED)
        assert (outcome, reason) == ("agent_stop", "agent_requested")
        assert message == step.hypothesis


# ── the log ───────────────────────────────────────────────────────────────

FROZEN_TS = "2026-09-17T00:00:00+00:00"


@pytest.fixture()
def loop_env(monkeypatch, tmp_path):
    """The real loop with run_recipe, the verdicts and the clock stubbed."""
    import aorta.agent.loop as loop_mod
    import aorta.agent.state as state_mod

    run_dir = tmp_path / "out" / "DIAG-449"
    monkeypatch.setattr(loop_mod, "run_recipe", MagicMock(return_value=run_dir))
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", lambda _d: BASELINE_FAILED)
    monkeypatch.setattr(state_mod, "_utc_now_iso", lambda: FROZEN_TS)

    def _run(proposer, **config_kw):
        config = AgentConfig(
            output_dir=tmp_path / "out",
            ticket="DIAG-449",
            subprocess_argv=("false",),
            policy=AgentPolicy(max_iterations=3),
            mitigations_allowlist=tuple(MITIGATIONS),
            diagnostics_allowlist=(OFFERED,),
            llm_backend="openai",
            **config_kw,
        )
        result = run_agent_loop(config, proposer=proposer)
        return result, (run_dir / "agent_log.jsonl").read_text(encoding="utf-8")

    return _run


def _records(log_text: str) -> list[dict]:
    return [json.loads(line) for line in log_text.splitlines() if line.strip()]


class TestTheLogRecord:
    def test_a_run_with_no_drops_writes_a_byte_identical_log(self, loop_env, proposer):
        """The archived-episode guarantee, as a literal comparison.

        A golden rather than a key-set check: anything added unconditionally
        to these events -- even an empty ``unresolved_diagnostics: []`` --
        changes the bytes of every run already on disk.
        """
        _result, log_text = loop_env(
            proposer(
                _reply(
                    category="unknown",
                    hypothesis="no further hypotheses",
                    next_mitigations=[],
                    confidence=0.3,
                    stop=True,
                )
            )
        )
        expected = (
            '{"argv": ["false"], "diagnostic_candidates": ["amd_log_level_4"], '
            '"llm_backend": "openai", "max_probe_cells": null, "symptom": null, '
            '"ticket": "DIAG-449", "ticket_slug": "DIAG-449", '
            f'"ts": "{FROZEN_TS}", "type": "session_start"}}\n'
            '{"category": "unknown", "confidence": 0.3, '
            '"hypothesis": "no further hypotheses", "next_diagnostics": [], '
            '"next_mitigations": [], "stop": true, '
            '"stop_reason": "agent_requested", '
            f'"ts": "{FROZEN_TS}", "type": "llm_step"}}\n'
            '{"outcome": "agent_stop", "stop_reason": "agent_requested", '
            f'"ts": "{FROZEN_TS}", "type": "search_stopped"}}\n'
        )
        assert log_text == expected

    def test_the_key_is_absent_and_not_empty_when_nothing_is_dropped(
        self, loop_env, proposer
    ):
        """Absent, so `"unresolved_diagnostics" in record` still discriminates."""
        _result, log_text = loop_env(
            proposer(_reply(next_mitigations=[], confidence=0.3, stop=True))
        )
        for record in _records(log_text):
            assert "unresolved_diagnostics" not in record

    def test_a_dropped_diagnostic_is_written_to_the_llm_step(self, loop_env, proposer):
        """FAILS BEFORE THE FIX: the name was retained nowhere at all.

        The mitigation survives, so the loop carries on and this ``llm_step``
        is the only record of the dropped half.
        """
        _result, log_text = loop_env(
            proposer(
                [
                    _reply(next_mitigations=["tf32_off"], next_diagnostics=[UNOFFERED]),
                    _reply(category="unknown", next_mitigations=[], stop=True),
                ]
            )
        )
        records = _records(log_text)
        first_step = next(r for r in records if r["type"] == "llm_step")
        assert first_step["next_diagnostics"] == []
        assert first_step["unresolved_diagnostics"] == [UNOFFERED]
        # The search really did continue on the surviving mitigation.
        assert [r["mitigation"] for r in records if r["type"] == "mitigation_tried"] == [
            "tf32_off"
        ]

    def test_a_dropped_diagnostic_is_distinguishable_from_a_budget_refusal(
        self, loop_env, proposer
    ):
        """The naming claim, on one log: the two causes stay separable.

        ``axis_growth.rejected_diagnostics`` is a name that survived the
        filter and lost to the cell budget; it still appears in
        ``next_diagnostics``. An unresolved name appears in neither.
        """
        _result, log_text = loop_env(
            proposer(
                [
                    _reply(next_mitigations=["tf32_off"], next_diagnostics=[UNOFFERED]),
                    _reply(category="unknown", next_mitigations=[], stop=True),
                ]
            )
        )
        records = _records(log_text)
        growth = next(r for r in records if r["type"] == "axis_growth")
        step = next(r for r in records if r["type"] == "llm_step")
        assert growth["rejected_diagnostics"] == []
        assert step["unresolved_diagnostics"] == [UNOFFERED]
        assert UNOFFERED not in step["next_diagnostics"]

    def test_a_diagnostics_only_proposal_writes_the_names_on_the_stop(
        self, loop_env, proposer
    ):
        """FAILS BEFORE THE FIX. Recorded, not attributed: the reason is unchanged.

        A diagnostics-only proposal filtered to empty reaches the stop check
        through its second clause, so without these names the terminal event
        carries no trace of why the search ended.
        """
        result, log_text = loop_env(
            proposer(_reply(next_mitigations=[], next_diagnostics=[UNOFFERED]))
        )
        records = _records(log_text)
        step = next(r for r in records if r["type"] == "llm_step")
        stopped = next(r for r in records if r["type"] == "search_stopped")
        assert step["stop"] is False
        assert step["unresolved_diagnostics"] == [UNOFFERED]
        assert stopped["unresolved_diagnostics"] == [UNOFFERED]
        assert stopped["stop_reason"] == "agent_requested"
        assert result.outcome == "agent_stop"


# ── resume ────────────────────────────────────────────────────────────────


class TestWakeStillReconstructs:
    def test_a_dropped_diagnostic_is_never_treated_as_tried(self, tmp_path):
        """It was rejected before the axis was planned, so it was never run."""
        run_dir = tmp_path / "DIAG-449"
        run_dir.mkdir()
        (run_dir / "agent_log.jsonl").write_text(
            "\n".join(
                json.dumps(record, sort_keys=True)
                for record in [
                    {"type": "session_start", "ticket": "DIAG-449", "ts": FROZEN_TS},
                    {
                        "type": "llm_step",
                        "category": "illegal_mem",
                        "hypothesis": "need visibility",
                        "next_mitigations": ["tf32_off"],
                        "next_diagnostics": [],
                        "confidence": 0.6,
                        "stop": False,
                        "stop_reason": None,
                        "unresolved_diagnostics": [UNOFFERED],
                        "ts": FROZEN_TS,
                    },
                    {
                        "type": "mitigation_tried",
                        "mitigation": "tf32_off",
                        "ts": FROZEN_TS,
                    },
                    {
                        "type": "axis_growth",
                        "added_mitigations": ["tf32_off"],
                        "added_diagnostics": [],
                        "rejected_mitigations": [],
                        "rejected_diagnostics": [],
                        "cells_added": 1,
                        "cells_total": 2,
                        "mitigation_axis": ["none", "tf32_off"],
                        "diagnostic_axis": ["none"],
                        "ts": FROZEN_TS,
                    },
                    {"type": "iteration_complete", "iteration": 1, "ts": FROZEN_TS},
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state = wake(run_dir, ticket="DIAG-449")
        assert state.tried_diagnostics == []
        assert UNOFFERED not in state.tried_diagnostics
        assert state.tried_mitigations == ["tf32_off"]
        assert state.last_category == "illegal_mem"
        assert state.probe_cells_spent == 2
        assert state.iterations_completed == 1


# ── the harvester ─────────────────────────────────────────────────────────


class TestTheHarvesterCanReadThem:
    def _log(self, tmp_path: Path, records: list[dict]) -> Path:
        run_dir = tmp_path / "EP"
        run_dir.mkdir()
        (run_dir / "agent_log.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
            encoding="utf-8",
        )
        return run_dir

    def test_unresolved_names_reads_the_filter_drops(self, tmp_path):
        """FAILS BEFORE THE FIX: there was no such function and no such key."""
        from examples.rl.emit_demo_payload import refused_names, unresolved_names

        run_dir = self._log(
            tmp_path,
            [
                {
                    "type": "llm_step",
                    "next_diagnostics": [OFFERED],
                    "unresolved_diagnostics": [UNOFFERED],
                    "ts": FROZEN_TS,
                },
                {
                    "type": "axis_growth",
                    "rejected_mitigations": [],
                    "rejected_diagnostics": ["hsa_disable_cache"],
                    "ts": FROZEN_TS,
                },
            ],
        )
        assert unresolved_names(run_dir) == [UNOFFERED]
        # The budget refusal stays where it was, under its own name.
        assert refused_names(run_dir) == ["hsa_disable_cache"]

    def test_the_two_do_not_bleed_into_each_other(self, tmp_path):
        from examples.rl.emit_demo_payload import refused_names, unresolved_names

        run_dir = self._log(
            tmp_path,
            [
                {
                    "type": "llm_step",
                    "unresolved_diagnostics": [UNOFFERED],
                    "ts": FROZEN_TS,
                }
            ],
        )
        assert unresolved_names(run_dir) == [UNOFFERED]
        assert refused_names(run_dir) == []

    def test_a_log_with_no_drops_yields_nothing(self, tmp_path):
        """PASSES BOTH WAYS for refused_names; the archived episodes are this shape."""
        from examples.rl.emit_demo_payload import unresolved_names

        run_dir = self._log(
            tmp_path,
            [{"type": "llm_step", "next_diagnostics": [OFFERED], "ts": FROZEN_TS}],
        )
        assert unresolved_names(run_dir) == []


# ── the archived episodes ─────────────────────────────────────────────────

ARCHIVE = Path("/apps/vikhande/demo/episode")
#: sha256 of each archived agent_log.jsonl, recorded before this change.
ARCHIVED_LOGS = {
    ARCHIVE
    / "live"
    / "DEMO-JOINT-LIVE": (
        "litellm",
        "7c6c77c59f84a8af048234b35f9333cd50d8b38358e53547994a4dbc846a14f0",
    ),
    ARCHIVE
    / "fake"
    / "DEMO-JOINT-FAKE": (
        "fake",
        "ab4cc9bab4d05b35b3f10552b0676167820cfdc3bef0062e79f1ffa80187c029",
    ),
}


def _present_archives() -> dict[Path, tuple[str, str]]:
    return {d: v for d, v in ARCHIVED_LOGS.items() if (d / "agent_log.jsonl").is_file()}


def test_the_archived_episodes_are_unmodified_and_parse_unchanged():
    """PASSES ON BOTH TREES, which is what makes it evidence.

    Both episodes, including the litellm one -- not just the fake control. The
    live episode ran on the backend the filter is actually on, so "it used
    fake" is not the reason it is safe; the reason is that every diagnostic it
    proposed either reached the axis or was refused by the budget, and neither
    is a filter drop.

    Note the name in the refusal assertion: ``hip_launch_blocking`` is
    refused-by-budget in these episodes and is the filter-drop name used
    throughout this file. The same name reaching a log for two different
    reasons is exactly what separate keys are for -- under one key these
    episodes would be indistinguishable from ones where the policy's proposal
    never resolved.

    Skipped rather than failed when absent: the artifacts live outside the
    repo, and their absence is not a defect in this code.
    """
    from examples.rl.emit_demo_payload import refused_names

    present = _present_archives()
    if not present:
        pytest.skip(f"no archived agent_log.jsonl under {ARCHIVE}")
    for run_dir, (backend, expected_sha) in present.items():
        raw = (run_dir / "agent_log.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha, run_dir
        records = _records(raw.decode("utf-8"))
        assert records[0]["llm_backend"] == backend
        # Written by a code path that predates the key, so it cannot be there.
        assert not [r for r in records if "unresolved_diagnostics" in r]
        assert refused_names(run_dir) == ["hip_launch_blocking"]
        state = wake(run_dir, ticket=run_dir.name)
        assert state.winning_mitigation == (
            "pytorch_no_cuda_memory_caching" if backend == "litellm" else None
        )
        assert state.tried_diagnostics == ["amd_log_level_4"] + (
            ["hip_launch_blocking"] if backend == "fake" else []
        )


def test_the_archived_episodes_report_no_filter_drops_to_the_harvester():
    """The same claim through the new reader, so the caveat cannot misfire."""
    from examples.rl.emit_demo_payload import unresolved_names

    present = _present_archives()
    if not present:
        pytest.skip(f"no archived agent_log.jsonl under {ARCHIVE}")
    for run_dir in present:
        assert unresolved_names(run_dir) == []
