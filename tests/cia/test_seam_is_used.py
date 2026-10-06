"""The one thing that submits jobs goes through the seam.

``launch()`` exists so that a scheduler-less backend is a branch in one place
rather than an edit at every call site -- finding F4 of the port plan, and the
subject of ``test_launch_seam.py``. That test went through the seam while the
only production submitter, the triage driver, imported ``submit_sbatch`` and
called it directly. The seam was tested and bypassed, which is the state an
abstraction rots in: green tests, and a second call site to find later.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CIA = ROOT / "src" / "aorta" / "cia"
#: Where submit_sbatch is defined, and the seam that is allowed to call it.
_ALLOWED = {"launch/cluster.py", "launch/__init__.py"}


def _modules_calling(name: str) -> list[str]:
    """Modules whose code calls *name*, ignoring comments and docstrings."""
    hits = []
    for path in sorted(CIA.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            func = getattr(node, "func", None)
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            if isinstance(node, ast.Call) and called == name:
                hits.append(path.relative_to(CIA).as_posix())
                break
    return hits


def test_nothing_reaches_past_the_seam_to_the_scheduler():
    offenders = [m for m in _modules_calling("submit_sbatch") if m not in _ALLOWED]
    assert not offenders, (
        f"{offenders} call submit_sbatch directly. Submitting through launch() is "
        "what keeps a scheduler-less backend a branch in one place."
    )


def test_the_triage_driver_submits_through_it():
    assert "triage.py" in _modules_calling("launch")


def test_the_driver_no_longer_imports_the_scheduler_call():
    """A stale import is a call site waiting to be re-added."""
    source = (CIA / "triage.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "submit_sbatch" not in imported


def test_the_seam_still_forwards_everything_it_is_given(monkeypatch, tmp_path):
    """Routing through it is only worth anything if it passes the arguments on."""
    from aorta.cia import launch as launch_pkg

    seen: dict = {}
    monkeypatch.setattr(
        "aorta.cia.launch.cluster.submit_sbatch",
        lambda **kw: seen.update(kw) or ("7", ""),
    )
    launch_pkg.launch(
        command="echo hi",
        job_name="j",
        log_path="/tmp/l",
        script_path=tmp_path / "s.sbatch",
        working_dir="/tmp",
        env_vars={"A": "1"},
        node="node-01",
    )
    assert seen["command"] == "echo hi"
    assert seen["env_vars"] == {"A": "1"}
    assert seen["node"] == "node-01"
