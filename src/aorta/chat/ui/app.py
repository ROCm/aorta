"""Chainlit chat application -- entry point for the AORTA Agent."""

from __future__ import annotations

import logging
import os

import chainlit as cl

from aorta.chat import redaction
from aorta.chat.config import UI_NO_WAIT_ENV, UI_VERBOSE_ENV
from aorta.chat.inference.providers.factory import get_backend
from aorta.chat.session import invoke_agent
from aorta.chat.ui.welcome import welcome_message

# Set by ``aorta chat ui`` from its group-level flags. This is a fresh
# interpreter, so they cannot arrive any other way.
_VERBOSE = os.environ.get(UI_VERBOSE_ENV) == "1"
_SKIP_PREFLIGHT = os.environ.get(UI_NO_WAIT_ENV) == "1"

logging.basicConfig(
    level=logging.DEBUG if _VERBOSE else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

#: Key under which each browser session keeps its :class:`redaction.NoticeState`.
_NOTICE_STATE_KEY = "redaction_notice_state"


def _unavailable_message(reason: str) -> str:
    return f"**LLM backend unavailable**\n\n```\n{reason}\n```"


@cl.on_chat_start
async def on_start():
    """Initialise per-session state and check the LLM backend is usable."""
    cl.user_session.set("history", [])
    cl.user_session.set("backend_error", None)
    # One state per browser session, not one per process: the notice is a
    # per-session disclosure and this server serves many at once.
    cl.user_session.set(_NOTICE_STATE_KEY, redaction.NoticeState())

    try:
        backend = get_backend()
        if not _SKIP_PREFLIGHT:
            await backend.preflight()
    except (ImportError, ValueError) as exc:
        logger.error("LLM backend unavailable: %s", exc)
        cl.user_session.set("backend_error", str(exc))
        await cl.Message(content=_unavailable_message(str(exc))).send()
        return

    await cl.Message(content=welcome_message(backend.describe())).send()


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
    # A session that predates this key (or a reconnect) still gets its own
    # state rather than falling back to the process-wide one, which would let
    # one browser session consume another's disclosure.
    notice_state = cl.user_session.get(_NOTICE_STATE_KEY)
    if notice_state is None:
        notice_state = redaction.NoticeState()
        cl.user_session.set(_NOTICE_STATE_KEY, notice_state)

    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    try:
        with redaction.use_notice_state(notice_state):
            reply, history, _result = await invoke_agent(message.content, history)
    except Exception:
        logger.exception("Agent graph error")
        await thinking_msg.remove()
        await cl.Message(
            content="An error occurred while processing your request. Please try again."
        ).send()
        # The send already happened before the graph raised, so what left the
        # machine has to be disclosed whether or not an answer came back. A
        # user who stops after the failure would otherwise never be told.
        await _deliver_notice(notice_state)
        return

    await thinking_msg.remove()
    cl.user_session.set("history", history)
    await cl.Message(content=reply).send()
    await _deliver_notice(notice_state)


async def _deliver_notice(notice_state: redaction.NoticeState) -> None:
    """Drain this session's pending redaction notice into the transcript.

    After the reply, not before: the notice reports what the request that just
    happened had removed, and draining it is what makes the session see it
    once. Without this the disclosure Decision 16 requires only ever reached
    the server's stderr.
    """
    notice = redaction.take_pending_notice(notice_state)
    if notice:
        await cl.Message(content=f"_{notice}_").send()
