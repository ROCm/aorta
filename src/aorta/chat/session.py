"""Shared helpers used by both the Chainlit UI and the CLI entry points."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage

from aorta.chat.graph.graph import agent_graph
from aorta.chat.inference.callcount import count_llm_calls

logger = logging.getLogger(__name__)


async def wait_for_vllm(timeout: int = 300, interval: int = 5) -> None:
    """Deprecated alias for the local backend's preflight.

    Prefer ``get_backend().preflight()``, which honours ``llm_provider``.
    Imported inside the function so this module, which both entry points
    load, stays free of any single provider at import time.
    """
    from aorta.chat.inference.providers.local_vllm import LocalVLLMBackend

    await LocalVLLMBackend().preflight(timeout=timeout, interval=interval)


def extract_reply(messages: list[BaseMessage]) -> str:
    """Return the content of the last AIMessage, or a fallback string."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "I couldn't generate a response. Please try rephrasing."


async def invoke_agent(
    query: str,
    history: list[BaseMessage],
) -> tuple[str, list[BaseMessage], dict]:
    """Run a single query through the agent graph.

    Returns:
        (reply_text, updated_history, raw_result_dict)
    """
    from langchain_core.messages import HumanMessage

    # A copy, not the caller's list: both front doors catch a graph failure and
    # carry on with the history they passed in, so appending before the await
    # left the failed question in it as an unanswered user turn that every
    # later request then replayed.
    pending = [*history, HumanMessage(content=query)]
    with count_llm_calls("query"):
        result = await agent_graph.ainvoke(
            {
                "messages": pending,
                "route": None,
                "plan": None,
                "retrieved_context": None,
                "command_output": None,
                "critic_feedback": None,
                "iteration": 0,
            }
        )
    reply = extract_reply(result.get("messages", []))
    return reply, [*pending, AIMessage(content=reply)], result
