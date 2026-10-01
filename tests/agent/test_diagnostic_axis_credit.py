"""aorta#504: a pass is credited by what the passing name does, not by its axis.

Both probe axes resolve through one registry and stamp the same environment,
so ``none-hip_launch_blocking`` and ``hip_launch_blocking-none`` are the same
experiment -- and shipped recipes put ``hip_launch_blocking`` and ``xnack`` on
either axis. What is protected here:

* a built-in that changes behaviour is credited from either axis, while a
  pass under logging, or under a name the registry cannot classify, is not;
* a mitigation-axis win still decides the winner whenever there is one, so a
  matrix with a passing ``{m}-none`` cell reports the name it always did;
* a mitigation-axis convergence writes the log it always wrote -- asserted
  against a literal golden log, because archived trajectories are compared
  against it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aorta.agent.llm import AgentStep
from aorta.agent.loop import AgentConfig, run_agent_loop
from aorta.agent.policy import AgentPolicy
from aorta.agent.state import wake, winning_mitigation
from aorta.registry.mitigations import BUILTIN_MITIGATIONS

FROZEN_TS = "2026-09-29T00:00:00+00:00"

#: Registered by ``examples/probe-flag-sidecar.json``, not by the built-ins.
SIDECAR_NAME = "fbgemm_no_jk"


def _cell(name: str, verdict: str = "pass") -> dict:
    failed = verdict != "pass"
    return {
        "cell_name": name,
        "verdict": verdict,
        "failure_detectors_fired": ["tier1:exit_nonzero"] if failed else [],
        "warn_detectors_fired": [],
        "capture": {},
        "exit_code": 1 if failed else 0,
    }


BASELINE_FAILS = _cell("none-none", "fail")
BASELINE_FAILS_ON_DISK = ("none-none", "fail")


# ── the rule ──────────────────────────────────────────────────────────────


class TestTheCreditRule:
    @pytest.mark.parametrize("name", ["hip_launch_blocking", "xnack"])
    def test_a_behavioural_name_is_credited_from_either_axis(self, name):
        assert winning_mitigation(f"{name}-none", "pass") == name
        assert winning_mitigation(f"none-{name}", "pass") == name

    @pytest.mark.parametrize("name", sorted(set(BUILTIN_MITIGATIONS) - {"none"}))
    def test_the_mitigation_axis_credits_every_builtin_as_before(self, name):
        assert winning_mitigation(f"{name}-none", "pass") == name

    def test_a_pass_under_logging_is_not_a_fix(self):
        assert winning_mitigation("none-amd_log_level_4", "pass") is None

    def test_a_diagnostic_the_registry_cannot_classify_is_not_credited(self):
        assert SIDECAR_NAME not in BUILTIN_MITIGATIONS
        assert winning_mitigation(f"none-{SIDECAR_NAME}", "pass") is None

    @pytest.mark.parametrize(
        "cell", ["xnack-hip_launch_blocking", "hip_launch_blocking-xnack", "none-none"]
    )
    def test_a_pass_that_isolates_no_single_name_credits_nothing(self, cell):
        assert winning_mitigation(cell, "pass") is None

    def test_the_observability_set_names_only_builtins(self):
        """A renamed entry would leave its knob silently classified as behavioural."""
        from aorta.registry.mitigations import OBSERVABILITY_MITIGATIONS

        assert OBSERVABILITY_MITIGATIONS
        assert OBSERVABILITY_MITIGATIONS <= set(BUILTIN_MITIGATIONS) - {"none"}


class TestPrecedence:
    def test_a_mitigation_axis_win_outranks_an_earlier_diagnostic_axis_win(self):
        from aorta.agent.state import winning_cell

        cells = [
            ("none-none", "fail"),
            ("none-hip_launch_blocking", "pass"),
            ("xnack-none", "pass"),
        ]
        assert winning_cell(cells) == ("xnack-none", "xnack")

    def test_without_one_the_first_credited_diagnostic_wins(self):
        from aorta.agent.state import winning_cell

        cells = [
            ("none-none", "fail"),
            ("tf32_off-none", "fail"),
            ("none-amd_log_level_4", "pass"),
            ("none-hip_launch_blocking", "pass"),
            ("none-xnack", "pass"),
        ]
        assert winning_cell(cells) == ("none-hip_launch_blocking", "hip_launch_blocking")
        assert winning_cell(cells[:3]) is None


# ── resume ────────────────────────────────────────────────────────────────


def _write_cells(run_dir, cells):
    """Write each cell's trials; a tuple of verdicts writes one trial per entry."""
    for name, verdicts in cells:
        if isinstance(verdicts, str):
            verdicts = (verdicts,)
        for i, verdict in enumerate(verdicts):
            trial = run_dir / name / f"trial_{i}"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps({"cell_name": name, "verdict": verdict}), encoding="utf-8"
            )


class TestWake:
    def test_a_diagnostic_axis_pass_is_a_win_on_resume(self, tmp_path):
        _write_cells(tmp_path, [("none-none", "fail"), ("none-hip_launch_blocking", "pass")])
        state = wake(tmp_path, ticket="A504")
        assert state.converged is True
        assert state.winning_mitigation == "hip_launch_blocking"

    def test_a_mitigation_axis_winner_survives_a_cell_that_sorts_ahead_of_it(self, tmp_path):
        _write_cells(
            tmp_path,
            [("none-none", "fail"), ("none-hip_launch_blocking", "pass"), ("xnack-none", "pass")],
        )
        assert wake(tmp_path, ticket="A504").winning_mitigation == "xnack"

    @pytest.mark.parametrize("passing", ["none-hip_launch_blocking", "tf32_off-none"])
    def test_a_pass_beside_a_passing_baseline_is_not_rebuilt_as_a_win(self, tmp_path, passing):
        _write_cells(tmp_path, [("none-none", "pass"), (passing, "pass")])
        state = wake(tmp_path, ticket="A504")
        assert state.converged is False
        assert state.winning_mitigation is None

    @pytest.mark.parametrize(
        "baseline",
        [[], [("none-none", ("pass", "fail"))]],
        ids=["no baseline verdict yet", "baseline passing one trial of two"],
    )
    def test_a_win_beside_a_baseline_that_did_not_pass_is_rebuilt(self, tmp_path, baseline):
        _write_cells(tmp_path, [*baseline, ("none-hip_launch_blocking", "pass")])
        state = wake(tmp_path, ticket="A504")
        assert state.converged is True
        assert state.winning_mitigation == "hip_launch_blocking"


#: Matrices a live run can leave on disk. The baseline-pass rows are the ones a
#: resume could misread: the live loop stops on them before it looks for a win.
RESUME_MATRICES = {
    "baseline and a behavioural diagnostic": (
        [("none-none", "pass"), ("none-hip_launch_blocking", "pass")],
        ("baseline_pass", None),
    ),
    "baseline and a mitigation": (
        [("none-none", "pass"), ("tf32_off-none", "pass")],
        ("baseline_pass", None),
    ),
    "baseline and logging": (
        [("none-none", "pass"), ("none-amd_log_level_4", "pass")],
        ("baseline_pass", None),
    ),
    "baseline alone": ([("none-none", "pass")], ("baseline_pass", None)),
    "baseline passing one trial of two": (
        [("none-none", ("pass", "fail")), ("none-hip_launch_blocking", "pass")],
        ("converged", "hip_launch_blocking"),
    ),
    "no baseline verdict yet": (
        [("none-hip_launch_blocking", "pass")],
        ("converged", "hip_launch_blocking"),
    ),
    "a behavioural diagnostic": (
        [BASELINE_FAILS_ON_DISK, ("none-hip_launch_blocking", "pass")],
        ("converged", "hip_launch_blocking"),
    ),
    "a mitigation beside a diagnostic": (
        [BASELINE_FAILS_ON_DISK, ("none-hip_launch_blocking", "pass"), ("xnack-none", "pass")],
        ("converged", "xnack"),
    ),
    "logging alone": (
        [BASELINE_FAILS_ON_DISK, ("none-amd_log_level_4", "pass")],
        ("agent_stop", None),
    ),
    "both axes off baseline": (
        [BASELINE_FAILS_ON_DISK, ("xnack-hip_launch_blocking", "pass")],
        ("agent_stop", None),
    ),
    "nothing passes": (
        [BASELINE_FAILS_ON_DISK, ("none-hip_launch_blocking", "fail")],
        ("agent_stop", None),
    ),
}


class TestResumeAgreesWithTheLiveRun:
    """A resumed run reports the outcome and the winner the live run reported.

    The live run starts on an empty run dir and its matrix lands on disk only
    when the probe runs, as in production; the resume reads that same dir back
    through ``wake()``. Nothing but ``run_recipe`` is stubbed.
    """

    @pytest.mark.parametrize(
        ("cells", "expected"), RESUME_MATRICES.values(), ids=RESUME_MATRICES.keys()
    )
    def test_live_and_resumed_attribution_agree(self, monkeypatch, tmp_path, cells, expected):
        import aorta.agent.loop as loop_mod

        run_dir = tmp_path / "out" / "A504"

        def _probe(*_args, **_kwargs):
            if not (run_dir / cells[0][0]).exists():
                _write_cells(run_dir, cells)
            return run_dir

        monkeypatch.setattr(loop_mod, "run_recipe", _probe)
        config = AgentConfig(
            output_dir=tmp_path / "out",
            ticket="A504",
            subprocess_argv=("echo", "hi"),
            policy=AgentPolicy(max_iterations=2),
            mitigations_allowlist=("none", "tf32_off"),
        )

        def _attribution():
            result = run_agent_loop(config, proposer=_Stops())
            report = result.report_path.read_text(encoding="utf-8")
            return (
                result.outcome,
                result.state.converged,
                result.state.winning_mitigation,
                "Winning mitigation" in report,
            )

        live = _attribution()
        outcome, winner = expected
        assert live == (outcome, winner is not None, winner, winner is not None)
        assert _attribution() == live


class _NeverCalled:
    def propose(self, **kwargs):
        raise AssertionError("the matrix already holds the answer")


class _Stops:
    def __init__(self):
        self.calls = 0

    def propose(self, **kwargs):
        self.calls += 1
        return AgentStep(
            category="unknown",
            hypothesis="nothing else to try",
            next_mitigations=[],
            confidence=0.1,
            stop=True,
            stop_reason="agent_requested",
        )


@pytest.fixture()
def loop_env(monkeypatch, tmp_path):
    """Run the real loop with only run_recipe, the verdicts and the clock stubbed."""
    import aorta.agent.loop as loop_mod
    import aorta.agent.state as state_mod

    run_dir = tmp_path / "out" / "A504"
    recipe = MagicMock(return_value=run_dir)
    monkeypatch.setattr(loop_mod, "run_recipe", recipe)
    monkeypatch.setattr(state_mod, "_utc_now_iso", lambda: FROZEN_TS)

    def _run(summaries, proposer):
        monkeypatch.setattr(loop_mod, "_read_cell_summaries", lambda _d: summaries)
        config = AgentConfig(
            output_dir=tmp_path / "out",
            ticket="A504",
            subprocess_argv=("echo", "hi"),
            policy=AgentPolicy(max_iterations=2),
            mitigations_allowlist=("none", "tf32_off"),
        )
        result = run_agent_loop(config, proposer=proposer)
        log_text = (run_dir / "agent_log.jsonl").read_text(encoding="utf-8")
        return result, log_text, recipe.call_count

    return _run


def _records(log_text: str) -> list[dict]:
    return [json.loads(line) for line in log_text.splitlines() if line.strip()]


class TestTheLoop:
    def test_a_passing_behavioural_diagnostic_converges_before_any_proposal(self, loop_env):
        result, log_text, runs = loop_env(
            [BASELINE_FAILS, _cell("none-hip_launch_blocking")], _NeverCalled()
        )
        assert result.outcome == "converged"
        assert result.state.winning_mitigation == "hip_launch_blocking"
        assert runs == 1
        assert "(see cell `none-hip_launch_blocking` probe.env" in result.recommended_action
        converged = [r for r in _records(log_text) if r["type"] == "converged"]
        assert converged == [
            {
                "ts": FROZEN_TS,
                "type": "converged",
                "winning_cell": "none-hip_launch_blocking",
                "winning_mitigation": "hip_launch_blocking",
            }
        ]

    def test_a_pass_under_logging_does_not_end_the_search(self, loop_env):
        proposer = _Stops()
        result, _log_text, _runs = loop_env(
            [BASELINE_FAILS, _cell("none-amd_log_level_4")], proposer
        )
        assert result.outcome == "agent_stop"
        assert result.state.winning_mitigation is None
        assert proposer.calls == 1

    def test_a_mitigation_axis_convergence_writes_the_log_it_always_wrote(self, loop_env):
        result, log_text, _runs = loop_env(
            [BASELINE_FAILS, _cell("none-hip_launch_blocking"), _cell("xnack-none")],
            _NeverCalled(),
        )
        assert log_text == (
            '{"argv": ["echo", "hi"], "llm_backend": "fake", "symptom": null, '
            '"ticket": "A504", "ticket_slug": "A504", '
            f'"ts": "{FROZEN_TS}", "type": "session_start"}}\n'
            f'{{"ts": "{FROZEN_TS}", "type": "converged", "winning_mitigation": "xnack"}}\n'
        )
        assert result.recommended_action == (
            "Re-run the repro with mitigation `xnack` applied "
            "(see cell `xnack-none` probe.env or matrix)."
        )
