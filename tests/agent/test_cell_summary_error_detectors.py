"""A cell summary names its error detectors, and only where one fired (aorta#516).

A trial that ran into its deadline and one whose command never launched both
resolve to ``error`` with no failure or warn detector. What tells them apart is
``error_detectors_fired`` -- ``tier1:timeout`` against ``tier1:exec_failed`` --
so the summary the proposer and the report are built from has to carry it.

Two prompts must not move because of it:

* ``default`` sends each summary row verbatim, so a row carries the key only
  where an error detector fired, and every other row renders the bytes it did
  before the key existed.
* ``rl-episode`` sends five named fields, as the policy was trained on them, so
  the key never reaches it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from aorta.agent.llm import _profile_prompt
from aorta.agent.loop import _read_cell_summaries
from aorta.agent.prompt_profiles import get_prompt_profile
from aorta.agent.report import write_agent_report
from aorta.agent.state import AgentState

TIMEOUT = "tier1:timeout"
EXEC_FAILED = "tier1:exec_failed"
ENV_REJECTED = "meta:env_file_validation_failed"

#: The keys of a row no error detector fired on, in order. The default prompt
#: dumps the row as it is, so the order is part of the bytes.
SIX_KEYS = [
    "cell_name",
    "verdict",
    "failure_detectors_fired",
    "warn_detectors_fired",
    "capture",
    "exit_code",
]

SYMPTOM = "loss went to nan"
REMAINING = ["gpu_max_hw_queues_2", "pytorch_no_cuda_memory_caching"]
TRIED = ["xnack", "hsa_no_sdma"]


def _trial(verdict, exit_code, *, failures=(), errors=(), warns=(), timed_out=False):
    """One trial's ``result.json``, in the shape ``SubprocessWorkload`` writes."""
    return {
        "verdict": verdict,
        "exit_code": exit_code,
        "failure_detectors_fired": list(failures),
        "error_detectors_fired": list(errors),
        "warn_detectors_fired": list(warns),
        "capture": {},
        "timed_out": timed_out,
    }


#: A timeout the hang monitor did not recognise, a command that never launched,
#: and a rejected probe.env: the three ways the workload records an ``error``.
TIMED_OUT = _trial("error", -1, errors=[TIMEOUT], timed_out=True)
NEVER_LAUNCHED = _trial("error", 127, errors=[EXEC_FAILED])
ENV_FILE_REJECTED = _trial("error", 2, errors=[ENV_REJECTED])
#: A hang fails the trial, and the timeout it outranked is still recorded.
HUNG = _trial("fail", -1, failures=["tier2:hang"], errors=[TIMEOUT], timed_out=True)
FAILED = _trial("fail", 1, failures=["tier1:exit_nonzero", "tier4:nan_signature"])
PASSED = _trial("pass", 0)


def _write_cell(run_dir: Path, cell: str, *trials: dict) -> None:
    for index, doc in enumerate(trials):
        trial = run_dir / cell / f"trial_{index}"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"cell_name": cell, "trial_index": index, **doc}), encoding="utf-8"
        )


def _rows_by_cell(run_dir: Path) -> dict[str, dict]:
    return {row["cell_name"]: row for row in _read_cell_summaries(run_dir)}


def _user_message(profile: str, rows: list[dict]) -> str:
    _, user = _profile_prompt(get_prompt_profile(profile), SYMPTOM, rows, REMAINING, TRIED)
    return user


def _without_error_detectors(rows: list[dict]) -> list[dict]:
    rows = copy.deepcopy(rows)
    for row in rows:
        row.pop("error_detectors_fired", None)
    return rows


def _report_table(report_dir: Path, rows: list[dict]) -> list[str]:
    path = write_agent_report(
        report_dir,
        state=AgentState(ticket="T-516"),
        cell_summaries=rows,
        outcome="exhausted_candidates",
        recommended_action="Inspect the cells.",
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("| Cell | Verdict | Detectors |")
    return lines[start : lines.index("", start)]


# ── the summary ────────────────────────────────────────────────────────────


def test_a_timeout_and_a_command_that_never_launched_are_told_apart(tmp_path):
    """Without the key, only the exit code separates the rows."""
    _write_cell(tmp_path, "none-none", TIMED_OUT)
    _write_cell(tmp_path, "xnack-none", NEVER_LAUNCHED)
    _write_cell(tmp_path, "hsa_no_sdma-none", ENV_FILE_REJECTED)
    rows = _rows_by_cell(tmp_path)
    for row in rows.values():
        assert (row["verdict"], row["failure_detectors_fired"], row["warn_detectors_fired"]) == (
            "error",
            [],
            [],
        )
    assert {cell: row["error_detectors_fired"] for cell, row in rows.items()} == {
        "none-none": [TIMEOUT],
        "xnack-none": [EXEC_FAILED],
        "hsa_no_sdma-none": [ENV_REJECTED],
    }


def test_the_key_follows_the_detector_not_the_verdict(tmp_path):
    _write_cell(tmp_path, "none-none", HUNG)
    row = _rows_by_cell(tmp_path)["none-none"]
    assert row["verdict"] == "fail"
    assert row["failure_detectors_fired"] == ["tier2:hang"]
    assert row["error_detectors_fired"] == [TIMEOUT]


def test_error_detectors_are_unioned_across_trials_in_first_seen_order(tmp_path):
    """The rule the failure and warn lists follow, not the evidence trial's list."""
    _write_cell(tmp_path, "none-none", PASSED, TIMED_OUT, ENV_FILE_REJECTED, TIMED_OUT)
    assert _rows_by_cell(tmp_path)["none-none"]["error_detectors_fired"] == [
        TIMEOUT,
        ENV_REJECTED,
    ]


def test_a_row_no_error_detector_fired_on_keeps_its_six_keys(tmp_path):
    written_without_the_list = {k: v for k, v in FAILED.items() if k != "error_detectors_fired"}
    _write_cell(tmp_path, "none-none", FAILED)
    _write_cell(tmp_path, "xnack-none", PASSED, _trial("pass", 0, warns=["tier3:vram_growth"]))
    _write_cell(tmp_path, "hsa_no_sdma-none", written_without_the_list)
    rows = _read_cell_summaries(tmp_path)
    assert len(rows) == 3
    for row in rows:
        assert list(row) == SIX_KEYS, row["cell_name"]


# ── the prompts ────────────────────────────────────────────────────────────


def test_the_default_prompt_is_byte_identical_where_no_error_detector_fired(tmp_path):
    """Against the six-key rows written out, so key order counts as well as content."""
    _write_cell(tmp_path, "none-none", FAILED)
    _write_cell(tmp_path, "xnack-none", PASSED)
    six_key_rows = [
        {
            "cell_name": "none-none",
            "verdict": "fail",
            "failure_detectors_fired": ["tier1:exit_nonzero", "tier4:nan_signature"],
            "warn_detectors_fired": [],
            "capture": {},
            "exit_code": 1,
        },
        {
            "cell_name": "xnack-none",
            "verdict": "pass",
            "failure_detectors_fired": [],
            "warn_detectors_fired": [],
            "capture": {},
            "exit_code": 0,
        },
    ]
    rows = _read_cell_summaries(tmp_path)
    assert _user_message("default", rows) == _user_message("default", six_key_rows)


def test_the_default_prompt_gains_the_key_only_on_the_row_it_fired_on(tmp_path):
    _write_cell(tmp_path, "none-none", TIMED_OUT)
    _write_cell(tmp_path, "xnack-none", FAILED)
    shown = json.loads(_user_message("default", _read_cell_summaries(tmp_path)))["cell_summaries"]
    assert [list(row) for row in shown] == [[*SIX_KEYS, "error_detectors_fired"], SIX_KEYS]
    assert shown[0]["error_detectors_fired"] == [TIMEOUT]


def test_the_rl_episode_prompt_never_sees_it(tmp_path):
    _write_cell(tmp_path, "none-none", TIMED_OUT)
    _write_cell(tmp_path, "xnack-none", NEVER_LAUNCHED)
    _write_cell(tmp_path, "hsa_no_sdma-none", HUNG)
    rows = _read_cell_summaries(tmp_path)
    assert all("error_detectors_fired" in row for row in rows)
    user = _user_message("rl-episode", rows)
    assert user == _user_message("rl-episode", _without_error_detectors(rows))
    assert TIMEOUT not in user and EXEC_FAILED not in user


# ── the report ─────────────────────────────────────────────────────────────


def test_the_report_lists_error_detectors_after_failure_detectors(tmp_path):
    cells = tmp_path / "cells"
    _write_cell(cells, "none-none", TIMED_OUT)
    _write_cell(cells, "xnack-none", NEVER_LAUNCHED)
    _write_cell(cells, "hsa_no_sdma-none", HUNG)
    assert _report_table(tmp_path, _read_cell_summaries(cells)) == [
        "| Cell | Verdict | Detectors |",
        "|------|---------|-----------|",
        "| `hsa_no_sdma-none` | fail | tier2:hang, tier1:timeout |",
        "| `none-none` | error | tier1:timeout |",
        "| `xnack-none` | error | tier1:exec_failed |",
    ]


def test_a_report_with_no_error_detector_is_unchanged(tmp_path):
    cells = tmp_path / "cells"
    _write_cell(cells, "none-none", FAILED)
    _write_cell(cells, "xnack-none", _trial("pass", 0, warns=["tier3:vram_growth"]))
    assert _report_table(tmp_path, _read_cell_summaries(cells)) == [
        "| Cell | Verdict | Detectors |",
        "|------|---------|-----------|",
        "| `none-none` | fail | tier1:exit_nonzero, tier4:nan_signature |",
        "| `xnack-none` | pass | — |",
    ]
