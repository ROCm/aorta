"""Agent log replay and wake()."""

from __future__ import annotations

import json
import os
import stat

from aorta.agent.state import agent_log_path, append_log_event, wake


def test_wake_replays_log_and_cell_verdicts(tmp_path):
    run_dir = tmp_path / "TICKET-1"
    run_dir.mkdir()
    append_log_event(run_dir, "llm_step", {"category": "illegal_mem", "hypothesis": "mem"})
    append_log_event(run_dir, "mitigation_tried", {"mitigation": "tf32_off"})

    cell = run_dir / "tf32_off-none" / "trial_0"
    cell.mkdir(parents=True)
    (cell / "result.json").write_text(
        json.dumps(
            {
                "cell_name": "tf32_off-none",
                "verdict": "pass",
                "failure_detectors_fired": [],
            }
        ),
        encoding="utf-8",
    )

    state = wake(run_dir, ticket="TICKET-1")
    assert state.last_category == "illegal_mem"
    assert "tf32_off" in state.tried_mitigations
    assert state.winning_mitigation == "tf32_off"
    assert state.converged is True


def test_append_log_event_narrows_existing_broad_perms(tmp_path):
    """An existing log with broad perms must be narrowed to 0600 on every write."""
    run_dir = tmp_path / "T"
    run_dir.mkdir()
    log = agent_log_path(run_dir)
    log.write_text('{"type": "stale"}\n', encoding="utf-8")
    os.chmod(log, 0o644)

    append_log_event(run_dir, "session_start", {"ticket": "T"})

    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_append_log_event_creates_owner_only(tmp_path):
    run_dir = tmp_path / "T2"
    run_dir.mkdir()
    append_log_event(run_dir, "session_start", {"ticket": "T2"})
    assert stat.S_IMODE(agent_log_path(run_dir).stat().st_mode) == 0o600
