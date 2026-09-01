"""Chainlit chat application -- entry point for the AORTA Agent."""

from __future__ import annotations

import logging

import chainlit as cl

from aorta.chat.inference.providers.factory import get_backend
from aorta.chat.session import invoke_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _unavailable_message(reason: str) -> str:
    return f"**LLM backend unavailable**\n\n```\n{reason}\n```"


@cl.on_chat_start
async def on_start():
    """Initialise per-session state and check the LLM backend is usable."""
    cl.user_session.set("history", [])
    cl.user_session.set("backend_error", None)

    try:
        backend = get_backend()
        await backend.preflight()
    except (ImportError, ValueError) as exc:
        logger.error("LLM backend unavailable: %s", exc)
        cl.user_session.set("backend_error", str(exc))
        await cl.Message(content=_unavailable_message(str(exc))).send()
        return

    await cl.Message(
        content=(
            "Welcome to the **AORTA Codebase Assistant**.\n\n"
            "I can help you understand, navigate, and work with the AORTA "
            "codebase. Ask me anything about it -- I can read files, search "
            "code, and even run commands in a sandbox.\n\n"
            f"_LLM backend: {backend.describe()}_\n\n"
            "_Type your question below to get started._"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle each user message by invoking the LangGraph agent."""
    # Without this the session keeps taking questions after a failed
    # preflight, and every one of them fails with the generic message below
    # instead of the actionable reason already reported at startup.
    backend_error = cl.user_session.get("backend_error")
    if backend_error:
        await cl.Message(content=_unavailable_message(backend_error)).send()
        return

    history: list = cl.user_session.get("history", [])

    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    try:
        reply, history, _result = await invoke_agent(message.content, history)
    except Exception:
        logger.exception("Agent graph error")
        await thinking_msg.remove()
        await cl.Message(
            content="An error occurred while processing your request. Please try again."
        ).send()
        return

    await thinking_msg.remove()
    cl.user_session.set("history", history)
    await cl.Message(content=reply).send()
