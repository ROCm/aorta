"""Shared helpers used by both the Chainlit UI and the CLI entry points."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import logging

from langchain_core.messages import AIMessage, BaseMessage

from aorta.chat.graph.graph import agent_graph
from aorta.chat.inference.callcount import count_llm_calls

logger = logging.getLogger(__name__)


async def wait_for_vllm(timeout: int = 300, interval: int = 5) -> None:
    """Deprecated alias for the local backend's preflight.

    Prefer ``get_backend().preflight()``, which honours ``LLM_PROVIDER``.
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
    on_step: Callable[[str, dict], Awaitable[None]] | None = None,
) -> tuple[str, list[BaseMessage], dict]:
    """Run a single query through the agent graph.

    *on_step* is awaited as the run proceeds: once per node with the state it
    produced, and once per tool with its name before that tool blocks. It is
    how a caller shows what the agents are doing rather than only what they
    concluded -- a diagnostic tool holds one node for minutes, and a node that
    reports only on completion says nothing for all of it.

    Omitting it awaits the graph exactly as before.

    Returns:
        (reply_text, updated_history, raw_result_dict)
    """
    from langchain_core.messages import HumanMessage

    history.append(HumanMessage(content=query))
    initial = {
        "messages": history,
        "route": None,
        "plan": None,
        "retrieved_context": None,
        "command_output": None,
        "critic_feedback": None,
        "iteration": 0,
    }
    with count_llm_calls("query"):
        if on_step is None:
            result = await agent_graph.ainvoke(initial)
        else:
            # "updates" names the node that just ran and "custom" carries a
            # tool announcing itself; "values" is the accumulated state, so the
            # last one matches what ainvoke would have returned.
            result = {}
            async for mode, chunk in agent_graph.astream(
                initial, stream_mode=["updates", "values", "custom"]
            ):
                if mode == "updates":
                    for node, delta in chunk.items():
                        await on_step(node, delta or {})
                elif mode == "custom":
                    await on_step("tool", chunk)
                else:
                    result = chunk
    reply = extract_reply(result.get("messages", []))
    history.append(AIMessage(content=reply))
    return reply, history, result
