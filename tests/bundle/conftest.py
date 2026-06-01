"""Shared fixtures for ``aorta bundle`` tests.

The bundle command operates on the per-ticket leaf of an ``aorta probe``
run dir (``<probe-output>/<ticket>/<cell>/trial_<n>/...``). Building
that tree by invoking ``aorta probe`` from every test would couple
bundle tests to probe behaviour (and slow the suite down). Instead
we synthesise a tree on disk that matches the documented artifact
shape -- the bundle pipeline does not care HOW the tree was
produced as long as it has ``trial_*/result.json`` entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_trial(
    cell_dir: Path,
    trial_idx: int,
    *,
    stdout: str = "hello\n",
    stderr: str = "",
    verdict: str = "pass",
    exit_code: int = 0,
    probe_env: str | None = None,
) -> None:
    trial = cell_dir / f"trial_{trial_idx}"
    trial.mkdir(parents=True, exist_ok=False)
    (trial / "stdout.log").write_text(stdout, encoding="utf-8")
    (trial / "stderr.log").write_text(stderr, encoding="utf-8")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "exit_code": exit_code,
                "walltime_sec": 0.1,
                "argv": ["bash", "-c", "echo hello"],
                "cell_name": cell_dir.name,
                "trial_index": trial_idx,
                "env_passthrough_mode": "inherit",
                "timed_out": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if probe_env is not None:
        (trial / "probe.env").write_text(probe_env, encoding="utf-8")


@pytest.fixture
def synthetic_run_dir(tmp_path: Path) -> Path:
    """A 2-cell probe-run leaf with 1 trial per cell + matrix metadata.

    Mirrors what ``aorta probe --ticket TKT-1 --output ./out -- bash``
    leaves under ``./out/TKT-1/``. Includes ``matrix.json`` and a
    minimal ``recipe.resolved.yaml`` so tests can exercise the
    "extra files are bundled too" path.
    """
    ticket = "TKT-1"
    run_dir = tmp_path / "probe-out" / ticket
    run_dir.mkdir(parents=True)

    _write_trial(run_dir / "none-none", 0)
    _write_trial(
        run_dir / "tf32_off-none",
        0,
        stdout="loss=0.5\n",
        verdict="fail",
        exit_code=1,
    )

    (run_dir / "matrix.md").write_text(
        "# Triage Matrix - _subprocess\n\n| Cell | Failures |\n",
        encoding="utf-8",
    )
    (run_dir / "matrix.json").write_text(
        json.dumps({"schema_version": 1, "ticket": ticket}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "recipe.resolved.yaml").write_text(
        "schema_version: 1\nmode: probe\n",
        encoding="utf-8",
    )
    (run_dir / "host_env.json").write_text("{}\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def no_ticket_run_dir(tmp_path: Path) -> Path:
    """A probe-run leaf whose basename is ``_no_ticket_``.

    Used to exercise the issue #196 acceptance criterion 2: bundle
    must refuse without --ticket when the source tree carries none.
    """
    run_dir = tmp_path / "probe-out" / "_no_ticket_"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none", 0)
    return run_dir


@pytest.fixture
def empty_run_dir(tmp_path: Path) -> Path:
    """An existing directory with no ``trial_*/result.json`` artifacts.

    Bundle command should refuse this with ``EmptyRunDirError``.
    """
    run_dir = tmp_path / "probe-out" / "TKT-EMPTY"
    run_dir.mkdir(parents=True)
    (run_dir / "matrix.md").write_text("placeholder\n", encoding="utf-8")
    return run_dir
