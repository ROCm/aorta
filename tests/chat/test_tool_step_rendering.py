"""The UI side of the tool step: it stays open until the tool finishes.

Chainlit renders a step as completed the moment it carries an end timestamp,
so "is this step still running" is exactly "is ``end`` unset". These drive the
handler with the events the graph emits and assert on that.
"""

from __future__ import annotations

import pytest

cl = pytest.importorskip("chainlit", reason="requires the chat-ui extra")

from aorta.chat.ui import app


class FakeStep:
    """Records the lifecycle a real cl.Step would perform against the UI."""

    instances: list["FakeStep"] = []

    def __init__(self, name=None, type=None, **kwargs):
        self.name = name
        self.type = type
        self.output = ""
        self.start = None
        self.end = None
        self.sends = 0
        self.updates = 0
        FakeStep.instances.append(self)

    async def send(self):
        self.sends += 1

    async def update(self):
        self.updates += 1

    @property
    def running(self) -> bool:
        return self.sends > 0 and self.end is None


@pytest.fixture()
def steps(monkeypatch):
    FakeStep.instances = []
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    return FakeStep.instances


START = {"tool": "triage_kernel_source", "id": "triage_kernel_source:1"}
DONE = {**START, "done": True, "seconds": 312.0}


async def test_the_step_is_still_running_after_the_announcement(steps):
    """The regression: it used to be closed here, while the tool had minutes to go."""
    tracker = app._ToolSteps()
    await tracker.handle(START)
    assert steps[0].running, "the step was closed before the tool finished"


async def test_the_step_appears_immediately(steps):
    """Held open, but not held back -- the point is to show it during the wait."""
    tracker = app._ToolSteps()
    await tracker.handle(START)
    assert steps[0].sends == 1


async def test_the_completion_closes_it(steps):
    tracker = app._ToolSteps()
    await tracker.handle(START)
    await tracker.handle(DONE)
    assert not steps[0].running
    assert steps[0].end is not None


async def test_the_same_step_is_reused(steps):
    """Updated in place rather than a second step appearing beneath the first."""
    tracker = app._ToolSteps()
    await tracker.handle(START)
    await tracker.handle(DONE)
    assert len(steps) == 1, f"{len(steps)} steps rendered for one tool call"
    assert steps[0].updates == 1


async def test_the_wait_is_reported(steps):
    tracker = app._ToolSteps()
    await tracker.handle(START)
    await tracker.handle(DONE)
    assert "312s" in steps[0].output


async def test_concurrent_tools_close_independently(steps):
    """Closing by name would end whichever happened to match."""
    tracker = app._ToolSteps()
    first = {"tool": "read_file", "id": "read_file:1"}
    second = {"tool": "read_file", "id": "read_file:2"}
    await tracker.handle(first)
    await tracker.handle(second)
    await tracker.handle({**first, "done": True, "seconds": 1.0})
    assert not steps[0].running, "the finished call is still open"
    assert steps[1].running, "an unfinished call was closed by its twin"


async def test_a_completion_with_nothing_open_is_ignored(steps):
    """Progress reporting can drop the announcement; this must not raise."""
    await app._ToolSteps().handle(DONE)
    assert steps == []


async def test_close_all_ends_an_abandoned_step(steps):
    """A graph that dies between the two events leaves a step spinning."""
    tracker = app._ToolSteps()
    await tracker.handle(START)
    await tracker.close_all()
    assert not steps[0].running
    assert "Interrupted" in steps[0].output


async def test_close_all_survives_a_dead_session(steps, monkeypatch):
    """The tab is gone, so the update fails -- and must not mask the real error."""

    async def gone(self):
        raise RuntimeError("session closed")

    tracker = app._ToolSteps()
    await tracker.handle(START)
    monkeypatch.setattr(FakeStep, "update", gone)
    await tracker.close_all()  # must not raise


async def test_close_all_is_idempotent(steps):
    """It runs on both the success and the failure path."""
    tracker = app._ToolSteps()
    await tracker.handle(START)
    await tracker.close_all()
    await tracker.close_all()
    assert steps[0].updates == 1


async def test_a_finished_step_is_not_reopened_by_close_all(steps):
    """The normal path closes the step; cleanup must leave it alone."""
    tracker = app._ToolSteps()
    await tracker.handle(START)
    await tracker.handle(DONE)
    await tracker.close_all()
    assert "Interrupted" not in steps[0].output


def test_our_timestamp_matches_chainlits():
    """We reimplement chainlit.utils.utc_now rather than import a private module.

    Chainlit parses these, so the two must not drift apart.
    """
    chainlit_utils = pytest.importorskip("chainlit.utils")
    theirs, ours = chainlit_utils.utc_now(), app.utc_now()
    shape = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$"
    import re

    assert re.match(shape, theirs), f"chainlit's format changed: {theirs}"
    assert re.match(shape, ours), f"ours no longer matches: {ours}"
