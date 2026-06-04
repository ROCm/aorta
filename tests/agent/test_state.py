"""Agent log replay and wake()."""

from __future__ import annotations

import json

from aorta.agent.state import append_log_event, wake


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
