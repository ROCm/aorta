"""The placeholder is the indicator until something better replaces it.

"Thinking..." was sent at the start and removed at the end, with the steps
accumulating in between. That is one indicator too many: a step names the node
that just ran, which is strictly more than "Thinking...", so from the first step
onward the placeholder said nothing the screen was not already saying better.

Removing it at the start instead would be worse. Steps arrive as nodes finish,
and the first node to finish is the router, which is an LLM call -- so there is
a gap at the beginning with nothing in it, which is the frozen-looking chat the
placeholder exists to prevent.

So it retires on the first step that actually renders. A node that produces no
step does not count: nothing has replaced it yet.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


class _Message:
    """Records sends and removals in order."""

    events: list[str] = []

    def __init__(self, content: str = "", **kwargs) -> None:
        self.content = content

    async def send(self) -> None:
        type(self).events.append(f"send:{self.content}")

    async def remove(self) -> None:
        type(self).events.append(f"remove:{self.content}")


class _Step:
    def __init__(self, name: str = "", **kwargs) -> None:
        self.name = name
        self.output = ""

    async def __aenter__(self):
        _Message.events.append(f"step:{self.name}")
        return self

    async def __aexit__(self, *exc):
        return False


class _UserSession:
    def __init__(self) -> None:
        self._values: dict = {}

    def get(self, key, default=None):
        return self._values.get(key, default)

    def set(self, key, value) -> None:
        self._values[key] = value


@pytest.fixture()
def app(monkeypatch):
    chainlit = ModuleType("chainlit")
    chainlit.Message = _Message
    chainlit.Step = _Step
    chainlit.user_session = _UserSession()
    chainlit.on_chat_start = lambda fn: fn
    chainlit.on_message = lambda fn: fn

    monkeypatch.setitem(sys.modules, "chainlit", chainlit)
    monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)
    import aorta.chat.ui.app as app_module

    app_module.cl.user_session = _UserSession()
    app_module.cl.user_session.set("history", [])
    app_module.cl.user_session.set("backend_error", None)
    _Message.events = []
    yield app_module
    monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)


def _answering(steps: list[tuple[str, dict]]):
    """An invoke_agent that reports *steps* and then answers."""

    async def invoke(question, history, on_step=None):
        for node, delta in steps:
            if on_step is not None:
                await on_step(node, delta)
        return "the kernel races on line 4", [], {}

    return invoke


class TestItRetiresOnTheFirstStep:
    async def test_the_placeholder_goes_before_the_step_is_drawn(self, app, monkeypatch):
        monkeypatch.setattr(
            app, "invoke_agent", _answering([("router", {"route": "action"})])
        )
        await app.on_message(SimpleNamespace(content="why?"))

        events = _Message.events
        assert "remove:Thinking..." in events
        first_step = next(i for i, e in enumerate(events) if e.startswith("step:"))
        assert events.index("remove:Thinking...") < first_step

    async def test_it_is_shown_first(self, app, monkeypatch):
        """Before any node has finished, it is all there is to show."""
        monkeypatch.setattr(
            app, "invoke_agent", _answering([("router", {"route": "action"})])
        )
        await app.on_message(SimpleNamespace(content="why?"))

        assert _Message.events[0] == "send:Thinking..."

    async def test_it_is_removed_exactly_once(self, app, monkeypatch):
        """Several steps, one placeholder."""
        monkeypatch.setattr(
            app,
            "invoke_agent",
            _answering(
                [
                    ("router", {"route": "action"}),
                    ("tool", {"tool": "triage_kernel_source"}),
                    ("act", {}),
                ]
            ),
        )
        await app.on_message(SimpleNamespace(content="why?"))

        assert _Message.events.count("remove:Thinking...") == 1

    async def test_a_tool_announcement_retires_it_too(self, app, monkeypatch):
        """The tool step can be the first thing rendered."""
        monkeypatch.setattr(
            app, "invoke_agent", _answering([("tool", {"tool": "triage_workload"})])
        )
        await app.on_message(SimpleNamespace(content="why?"))

        events = _Message.events
        first_step = next(i for i, e in enumerate(events) if e.startswith("step:"))
        assert events.index("remove:Thinking...") < first_step


class TestItSurvivesUntilSomethingReplacesIt:
    async def test_a_node_that_renders_nothing_does_not_retire_it(self, app, monkeypatch):
        """Otherwise the screen is empty while the first real node runs."""
        monkeypatch.setattr(app, "invoke_agent", _answering([("unknown_node", {})]))
        await app.on_message(SimpleNamespace(content="why?"))

        events = _Message.events
        assert not any(e.startswith("step:") for e in events)
        # Removed at the end, with the answer -- not during the run.
        assert events.index("remove:Thinking...") > events.index("send:Thinking...")

    async def test_a_turn_with_no_steps_still_removes_it(self, app, monkeypatch):
        monkeypatch.setattr(app, "invoke_agent", _answering([]))
        await app.on_message(SimpleNamespace(content="why?"))

        assert "remove:Thinking..." in _Message.events

    async def test_a_failing_turn_removes_it(self, app, monkeypatch):
        async def explode(question, history, on_step=None):
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", explode)
        await app.on_message(SimpleNamespace(content="why?"))

        events = _Message.events
        assert events.count("remove:Thinking...") == 1
        assert any("An error occurred" in e for e in events)


class TestTheAnswerStillArrivesLast:
    async def test_it_comes_after_the_steps(self, app, monkeypatch):
        monkeypatch.setattr(
            app,
            "invoke_agent",
            _answering([("router", {"route": "action"}), ("act", {})]),
        )
        await app.on_message(SimpleNamespace(content="why?"))

        events = _Message.events
        answer = next(i for i, e in enumerate(events) if "races on line 4" in e)
        assert answer == len(events) - 1
