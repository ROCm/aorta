"""An approval that explains itself is still an approval.

The critic is asked to reply "exactly: VALID". Models asked to approve
something usually approve it *and say why*, and the whole-string comparison
read every one of those as a rejection. The turn then spent its entire retry
budget re-answering a question it had already answered, and what reached the
user was headed "I could not verify this answer against the tool output" --
about an answer the critic had just passed.

Found by running the assistant rather than by reading it: a question about
where redaction lives came back with the critic's own approval quoted under
that heading.

The comparison it replaced was a substring test, which failed the other way:
"INVALID" contains "VALID", so every rejection passed. The leading word tells
the two apart, since neither "INVALID" nor "not valid" begins with VALID.
"""

from __future__ import annotations

import pytest

from aorta.chat.graph.nodes import _critic_approved


class TestAnApprovalIsRecognised:
    @pytest.mark.parametrize(
        "verdict",
        [
            "VALID",
            "valid",
            "VALID\n\nThe response is well-grounded in the tool results.",
            "VALID, because every claim cites a file the tool read.",
            "VALID.",
            "**VALID**",
            "`VALID`",
            "  VALID  ",
        ],
    )
    def test_it_counts_as_one(self, verdict):
        assert _critic_approved(verdict) is True


class TestARejectionIsStillARejection:
    """The substring test this replaced passed every one of these."""

    @pytest.mark.parametrize(
        "verdict",
        [
            "INVALID",
            "INVALID - the file path does not exist",
            "invalid, the line numbers were invented",
            "This is not valid; the tool output says otherwise.",
            "NOT VALID",
            "The answer is VALID only for the first claim",
            "",
            "   ",
        ],
    )
    def test_it_does_not_count_as_approval(self, verdict):
        assert _critic_approved(verdict) is False

    def test_the_word_appearing_later_does_not_approve(self):
        """A critic explaining which part was valid is not approving the whole."""
        assert not _critic_approved("The second claim is VALID but the first is not")


class TestTheAnswerReachesTheUser:
    """What the bug actually cost: a good answer presented as unverified."""

    async def test_an_explained_approval_clears_the_critic(self, monkeypatch):
        from langchain_core.messages import AIMessage

        from aorta.chat.graph import nodes

        async def fake_send(llm, messages):
            return AIMessage(content="VALID\n\nEvery claim cites a tool result.")

        monkeypatch.setattr(nodes, "_send", fake_send)
        monkeypatch.setattr(nodes, "_get_llm", lambda **kwargs: object())

        state = {
            "messages": [],
            "command_output": "The redaction gate is in chat/redaction.py.",
            "tool_trace": ["read_file"],
            "iteration": 0,
        }
        out = await nodes.critic_node(state)

        assert out["critic_feedback"] is None, "an approval must not read as feedback"

    async def test_a_rejection_still_produces_feedback(self, monkeypatch):
        from langchain_core.messages import AIMessage

        from aorta.chat.graph import nodes

        async def fake_send(llm, messages):
            return AIMessage(content="INVALID - the path was invented")

        monkeypatch.setattr(nodes, "_send", fake_send)
        monkeypatch.setattr(nodes, "_get_llm", lambda **kwargs: object())

        state = {
            "messages": [],
            "command_output": "The gate is in a file I made up.",
            "tool_trace": ["read_file"],
            "iteration": 0,
        }
        out = await nodes.critic_node(state)

        assert out["critic_feedback"], "a rejection must still be reported"
