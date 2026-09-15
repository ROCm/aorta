"""Shared helpers used by both the Chainlit UI and the CLI entry points."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import asyncio
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


async def _announce(
    on_step: Callable[[str, dict], Awaitable[None]],
    name: str,
    payload: dict,
    *,
    failed: list[BaseException],
) -> None:
    """Tell the caller a step happened, and carry on if it cannot hear it.

    Progress is not the answer. The awaits here were unguarded, so anything the
    callback raised came out of ``astream`` and ended the query -- and the
    callback the UI passes talks to Chainlit, which fails once the session is
    gone. A user closing the tab during a five-minute cluster job took the run
    down with it, leaving the job burning with nobody to receive the verdict.
    The ``ainvoke`` path this replaced had no such coupling, so anyone passing a
    callback was worse off than before.

    Cancellation is not swallowed: it is how the caller stops this on purpose,
    and catching it here would make the run unstoppable.
    """
    try:
        await on_step(name, payload)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any callback, any failure
        if not failed:
            logger.warning(
                "Progress reporting failed at step %r (%s: %s); the query "
                "continues and the answer is unaffected. Later steps this turn "
                "are not reported again.",
                name, type(exc).__name__, exc,
            )
        failed.append(exc)


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

    # A copy, not the caller's list: both front doors catch a graph failure and
    # carry on with the history they passed in, so appending before the await
    # left the failed question in it as an unanswered user turn that every
    # later request then replayed.
    pending = [*history, HumanMessage(content=query)]
    initial = {
        "messages": pending,
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
            #: Whether the stream ever carried state. "values" is the only mode
            #: that does, and without it ``result`` is still the empty dict it
            #: started as -- which reads downstream as a turn where the model
            #: said nothing, rather than as a graph that produced nothing.
            state_seen = False
            #: Progress failures seen this turn, so the warning is logged once
            #: rather than per chunk. A dead session fails every one of them.
            failed: list[BaseException] = []
            async for mode, chunk in agent_graph.astream(
                initial, stream_mode=["updates", "values", "custom"]
            ):
                if mode == "updates":
                    for node, delta in chunk.items():
                        await _announce(on_step, node, delta or {}, failed=failed)
                elif mode == "custom":
                    await _announce(on_step, "tool", chunk, failed=failed)
                else:
                    result = chunk
                    state_seen = True
            if not state_seen:
                # Distinguishable from a turn the model answered badly, which
                # is what falling through as an empty dict made this look like:
                # extract_reply would hand back "I couldn't generate a
                # response. Please try rephrasing.", and rephrasing cannot fix
                # a graph that streamed without ever yielding state. Raising
                # rather than returning that text keeps the failed turn out of
                # the transcript, which is what the copied history above is
                # for.
                raise RuntimeError(
                    "the agent graph streamed to completion without producing "
                    "any state, so there is no answer to return. This is a "
                    "malfunction rather than an unanswerable question."
                )

    reply = extract_reply(result.get("messages", []))
    return reply, [*pending, AIMessage(content=reply)], result
