"""Agent state schema for the LangGraph workflow."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state that flows through the LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: str | None
    plan: str | None
    retrieved_context: str | None
    command_output: str | None
    critic_feedback: str | None
    iteration: int
    # What the tools actually returned during the act loop. Carried explicitly
    # rather than left in `messages`: the critic has to check the answer against
    # tool output, but `add_messages` only receives what a node returns, and act
    # returns just its final answer. Keeping it out of `messages` also avoids
    # replaying every tool result to the model on a retry.
    tool_trace: list[str] | None
    #: Tools the selector ranked, best first, and the reason it gave.
    candidate_tools: list[str] | None
    selection_rationale: str | None
