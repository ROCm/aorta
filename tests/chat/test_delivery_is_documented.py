"""The delivery contract, and the docs that describe it.

The integration this chat came from documents how an answer arrives: a
``Thinking...`` placeholder replaced by one complete reply, not streamed token
by token. That is still true. What changed is what happens between those two
moments -- the browser now shows a step as each node finishes, and a tool
announces itself before it blocks -- and nothing said so.

These assert the two halves stay in step: that the code still keeps the part of
the contract the docs promise, and that the docs describe the part that is new.
A guide describing the pre-CIA assistant is how the next port re-learns these
collisions.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from aorta.chat import session

_README = Path(__file__).resolve().parents[2] / "docs" / "chat" / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    """The prose with its line wrapping flattened.

    Asserting on a phrase that the file happens to wrap is asserting on the
    wrapping: "streamed token by\ntoken" contains no "token by token".
    """
    return " ".join(_README.read_text(encoding="utf-8").split())


class TestTheContractTheDocsPromise:
    """Kept by the code, not only by the prose."""

    def test_no_callback_means_the_awaited_path(self):
        """The CLI passes none, and must not be routed through streaming."""
        source = inspect.getsource(session.invoke_agent)

        assert "if on_step is None:" in source
        assert "await agent_graph.ainvoke(initial)" in source

    def test_nothing_asks_for_token_level_streaming(self):
        """"messages" is the stream mode that emits tokens; it is not requested.

        Checked on the stream_mode list rather than the whole function, since
        the state dict has a "messages" key of its own.
        """
        source = inspect.getsource(session.invoke_agent)
        modes = source[source.index("stream_mode=") : source.index("stream_mode=") + 60]

        assert "messages" not in modes, modes
        assert "updates" in modes and "values" in modes and "custom" in modes

    def test_the_reply_is_assembled_once_at_the_end(self):
        source = inspect.getsource(session.invoke_agent)

        assert source.count("extract_reply(") == 1


class TestTheDocsDescribeWhatChanged:
    def test_the_placeholder_behaviour_is_still_stated(self, readme):
        assert "Thinking..." in readme
        assert "token by token" in readme

    def test_the_steps_are_described(self, readme):
        assert "step per node" in readme or "step as each node" in readme

    def test_it_says_the_cli_is_unchanged(self, readme):
        """The reason the CLI path was kept is worth being able to find."""
        assert "CLI" in readme

    def test_the_tool_announcement_is_described(self, readme):
        assert "announces itself" in readme

    def test_and_that_it_does_not_carry_the_arguments(self, readme):
        """Documented because the obvious next change is to add them back."""
        assert "does not carry the arguments" in readme

    def test_best_effort_progress_is_described(self, readme):
        assert "best-effort" in readme

    @pytest.mark.parametrize("front_door", ["aorta chat ui", "aorta chat ask"])
    def test_both_front_doors_are_named(self, readme, front_door):
        assert front_door in readme
