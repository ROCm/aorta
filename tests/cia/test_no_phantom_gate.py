"""There is no confirmation gate in this package, and nothing suggests there is.

``launch/confirm.py`` held ``proceed_gate`` and ``confirm_gate``, and nothing
imported either. ``docs/chat/README.md`` draws a line -- ``aorta chat`` is
conversational and watched, ``aorta agent`` is autonomous -- and that file read
like the thing keeping Launch on the chat side of it. It was not keeping Launch
anywhere.

It could not have, as written. ``proceed_gate`` asked for confirmation only when
stdin was a TTY and auto-proceeded otherwise, and nothing that serves a chat UI
has a TTY: under Chainlit it would have printed a summary nobody was reading and
returned True. A gate for chat-driven submission belongs where the chat tool
decides to submit, with a surface the user can actually answer, and wants
designing rather than reviving.

So the file is gone. A dead gate is worse than an absent one: the next reader
budgets for a confirmation that never happens.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CIA = ROOT / "src" / "aorta" / "cia"


def test_the_dead_gate_is_gone():
    assert not (CIA / "launch" / "confirm.py").exists()


@pytest.mark.parametrize("name", ["proceed_gate", "confirm_gate", "DeployError"])
def test_nothing_refers_to_it(name):
    """Including a stale import that would fail only when it ran."""
    hits = [
        str(p.relative_to(ROOT))
        for p in CIA.rglob("*.py")
        if name in p.read_text(encoding="utf-8")
    ]
    assert not hits, f"{name} still referenced in {hits}"


def test_the_package_still_imports_without_it():
    import importlib

    import aorta.cia.launch

    importlib.reload(aorta.cia.launch)


def test_submission_goes_through_the_seam(monkeypatch, tmp_path):
    """What launch() is for: one place a scheduler-less backend would branch.

    Asserted here as well as in test_launch_seam.py because the reason the gate
    looked plausible is that submission has a single documented entry point --
    so that entry point should stay the one that is used.
    """
    from aorta.cia import launch as launch_pkg

    called: list = []
    monkeypatch.setattr(
        "aorta.cia.launch.cluster.submit_sbatch",
        lambda **kw: called.append(kw) or ("42", ""),
    )

    job_id, error = launch_pkg.launch(
        command="echo hi",
        job_name="j",
        log_path=str(tmp_path / "l"),
        script_path=tmp_path / "s.sbatch",
    )

    assert (job_id, error) == ("42", "")
    assert called, "launch() must reach the scheduler through submit_sbatch"
