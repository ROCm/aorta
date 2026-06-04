"""Append-only agent event log and resume (``wake``).

State lives under ``<output>/<ticket>/agent_log.jsonl``. Each line is a
JSON object. The agent process is stateless across restarts: ``wake()``
replays the log and scans existing probe cell verdicts to rebuild
``AgentState``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_NAME = "agent_log.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class AgentState:
    """Reconstructed agent memory for a ticket run."""

    ticket: str
    tried_mitigations: list[str] = field(default_factory=list)
    last_category: str = "unknown"
    last_hypothesis: str = ""
    iterations_completed: int = 0
    winning_mitigation: str | None = None
    converged: bool = False


def agent_log_path(run_dir: Path) -> Path:
    return run_dir / _LOG_NAME


def append_log_event(run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    """Append one JSON line to ``agent_log.jsonl``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"ts": _utc_now_iso(), "type": event_type, **payload}
    path = agent_log_path(run_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_log_events(run_dir: Path) -> list[dict[str, Any]]:
    path = agent_log_path(run_dir)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _scan_cell_verdicts(run_dir: Path) -> dict[str, str]:
    """Map cell name -> verdict from ``trial_0/result.json`` when present."""
    verdicts: dict[str, str] = {}
    if not run_dir.is_dir():
        return verdicts
    for cell_dir in run_dir.iterdir():
        if not cell_dir.is_dir():
            continue
        result_path = cell_dir / "trial_0" / "result.json"
        if not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        verdict = data.get("verdict")
        if isinstance(verdict, str) and verdict:
            verdicts[cell_dir.name] = verdict
    return verdicts


def wake(run_dir: Path, *, ticket: str) -> AgentState:
    """Rebuild :class:`AgentState` from the event log and on-disk probe cells."""
    state = AgentState(ticket=ticket)
    for event in _read_log_events(run_dir):
        etype = event.get("type")
        if etype == "mitigation_tried":
            name = event.get("mitigation")
            if isinstance(name, str) and name not in state.tried_mitigations:
                state.tried_mitigations.append(name)
        elif etype == "llm_step":
            cat = event.get("category")
            if isinstance(cat, str):
                state.last_category = cat
            hyp = event.get("hypothesis")
            if isinstance(hyp, str):
                state.last_hypothesis = hyp
        elif etype == "iteration_complete":
            state.iterations_completed += 1
        elif etype == "converged":
            state.converged = True
            win = event.get("winning_mitigation")
            if isinstance(win, str):
                state.winning_mitigation = win

    verdicts = _scan_cell_verdicts(run_dir)
    for cell_name, verdict in verdicts.items():
        if cell_name == "none-none":
            continue
        if "-" not in cell_name:
            continue
        mitigation = cell_name.split("-", 1)[0]
        if mitigation not in state.tried_mitigations:
            state.tried_mitigations.append(mitigation)
        if verdict == "pass" and state.winning_mitigation is None:
            state.winning_mitigation = mitigation
            state.converged = True

    return state


__all__ = [
    "AgentState",
    "agent_log_path",
    "append_log_event",
    "wake",
]
