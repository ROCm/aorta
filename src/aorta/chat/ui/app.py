"""Chainlit chat application -- entry point for the AORTA Agent."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone

import chainlit as cl

from aorta.chat import redaction
from aorta.chat.config import UI_NO_WAIT_ENV, UI_VERBOSE_ENV
from aorta.chat.inference.providers.factory import get_backend
from aorta.chat.session import invoke_agent
from aorta.chat.tools.cache import ToolCache, use_tool_cache
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

#: Key under which each browser session keeps its :class:`ToolCache`.
_TOOL_CACHE_KEY = "tool_result_cache"


def _unavailable_message(reason: str) -> str:
    return f"**LLM backend unavailable**\n\n```\n{reason}\n```"


#: What each node contributes that is worth showing. A node absent from this
#: map renders nothing, which keeps plumbing like retrieve out of the way.
_NODE_TITLES = {
    "router": "Deciding whether this needs a job",
    "select": "Choosing a diagnostic tool",
    "plan": "Planning the steps",
    "act": "Running tools",
    "critic": "Checking the answer",
}


def _as_code_block(text: str) -> str:
    """Fence tool output so that none of it can escape the block.

    Tool output is arbitrary -- ``read_file`` on any of this repository's
    Markdown returns fences of its own -- and a fixed ``` opener ends at the
    first one inside the content, leaving the rest to render as Markdown. The
    fence has to be longer than the longest run the content contains.

    An indented block would also contain it, but consecutive ones separated by
    a blank line are a single block in CommonMark, which would run each tool's
    output into the next.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _node_reasoning(node: str, delta: dict) -> str:
    """What a node recorded, in the words it recorded it."""
    if node == "router":
        return f"Route: **{delta.get('route') or 'unknown'}**"
    if node == "select":
        tools = delta.get("candidate_tools") or []
        why = delta.get("selection_rationale") or ""
        if not tools:
            return why or "No tool matched; the agent will see the full list."
        ranked = "\n".join(f"{i}. `{t}`" for i, t in enumerate(tools, 1))
        return f"{why}\n\n{ranked}" if why else ranked
    if node == "plan":
        return str(delta.get("plan") or "")
    if node == "act":
        trace = delta.get("tool_trace") or []
        return "\n\n".join(_as_code_block(entry[:1500]) for entry in trace)
    if node == "critic":
        return str(delta.get("critic_feedback") or "Accepted.")
    return ""


def utc_now() -> str:
    """A step timestamp in the form Chainlit writes.

    The same two lines as ``chainlit.utils.utc_now``, rather than an import of
    it: that is a private module, and importing it would also mean the tests
    that stand a stub in for ``chainlit`` could no longer load this one.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


class _ToolSteps:
    """Holds a step open for each tool that is still running.

    Chainlit renders a step as completed once it carries an end timestamp, and
    ``async with cl.Step`` stamps one on the way out of the block. Wrapping
    only the announcement in that block closed the step before the tool had
    started, so a five-minute cluster job rendered as already finished for the
    whole of the wait -- the opposite of what announcing it was for.

    A step therefore spans two events, and the lifecycle is split the way
    Chainlit's own context manager splits it: send on the way in, stamp
    ``end`` and update on the way out.
    """

    def __init__(self) -> None:
        self._open: dict[str, cl.Step] = {}

    async def handle(self, delta: dict) -> None:
        """Open a step on a tool's announcement, close it on its completion."""
        name = delta.get("tool")
        # Keyed by call rather than name: a turn runs the same tool more than
        # once with different arguments, and closing by name would end the
        # wrong one.
        call = str(delta.get("id") or name)
        if delta.get("done"):
            await self._finish(call, name, delta.get("seconds"))
            return
        step = cl.Step(name=f"Running {name}", type="tool")
        step.start = utc_now()
        step.output = f"`{name}`\n\nWork on the cluster can take several minutes."
        await step.send()
        self._open[call] = step

    async def _finish(self, call: str, name, seconds) -> None:
        step = self._open.pop(call, None)
        if step is None:
            return  # a completion with nothing open; nothing to close
        took = f" in {seconds:g}s" if isinstance(seconds, (int, float)) else ""
        step.output = f"`{name}` finished{took}."
        step.end = utc_now()
        await step.update()

    async def close_all(self) -> None:
        """Leave nothing rendering as running once the turn is over.

        The graph ends its own announcements in a ``finally``, so this is for
        the turn that never got that far: a graph that died between the two
        events, or progress reporting that failed partway. A step left open
        spins until the session is reloaded.
        """
        while self._open:
            _, step = self._open.popitem()
            step.end = utc_now()
            step.output = f"{step.output}\n\n_Interrupted._"
            try:
                await step.update()
            except Exception:  # noqa: BLE001 - a dead session must not mask why
                logger.debug("Could not close step %r", step.name, exc_info=True)


#: Whether this process has already waited for the backend. Reachability is a
#: property of the server, not of one conversation, and this wait was in
#: ``on_chat_start`` -- so every browser tab paid it again. With the backend
#: down that was the full timeout per tab, spent arriving at the same answer,
#: and the session then started regardless.
_preflight_done = False
_preflight_lock = asyncio.Lock()


async def _preflight_once(backend) -> None:
    """Wait for the backend at most once per process.

    Under the lock and checked twice, because several browsers reconnecting
    together -- which is what a restarted server looks like -- would otherwise
    each start their own wait before any of them had finished one.

    Set after the await whatever happened: ``preflight`` does not raise, it
    gives up and logs, and repeating a wait that already gave up once only
    delays the next session by the same amount to learn the same thing. A
    backend that comes up later needs no wait at all; the next request works.
    """
    global _preflight_done
    if _preflight_done:
        return
    async with _preflight_lock:
        if _preflight_done:
            return
        await backend.preflight()
        _preflight_done = True


@cl.on_chat_start
async def on_start():
    """Initialise per-session state and check the LLM backend is usable."""
    cl.user_session.set("history", [])
    cl.user_session.set("backend_error", None)
    # One state per browser session, not one per process: the notice is a
    # per-session disclosure and this server serves many at once.
    cl.user_session.set(
        _NOTICE_STATE_KEY, redaction.NoticeState(opt_out=redaction.UI_OPT_OUT)
    )
    # Same reason, and the reuse it controls is reported to the user as having
    # happened "in this conversation": a shared cache would answer one user's
    # paste from another user's cluster run and say so.
    cl.user_session.set(_TOOL_CACHE_KEY, ToolCache())

    try:
        backend = get_backend()
        if not _SKIP_PREFLIGHT:
            await _preflight_once(backend)
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
        notice_state = redaction.NoticeState(opt_out=redaction.UI_OPT_OUT)
        cl.user_session.set(_NOTICE_STATE_KEY, notice_state)

    tool_cache = cl.user_session.get(_TOOL_CACHE_KEY)
    if tool_cache is None:
        tool_cache = ToolCache()
        cl.user_session.set(_TOOL_CACHE_KEY, tool_cache)

    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    #: Whether "Thinking..." is still on screen. It is the indicator until the
    #: first step renders, and redundant afterwards -- a step says what is
    #: happening, which is strictly more than "Thinking...". Retiring it then,
    #: rather than at the start, is what keeps the gap before the first node
    #: finishes from looking like a chat that has frozen.
    thinking_shown = True

    async def _retire_thinking() -> None:
        nonlocal thinking_shown
        if thinking_shown:
            thinking_shown = False
            await thinking_msg.remove()

    running = _ToolSteps()

    async def show_step(node: str, delta: dict) -> None:
        # A tool announces itself before it runs. Showing that immediately is
        # the difference between a visible five-minute cluster job and a chat
        # that looks frozen.
        if node == "tool":
            await _retire_thinking()
            await running.handle(delta)
            return
        title = _NODE_TITLES.get(node)
        body = _node_reasoning(node, delta) if title else ""
        if not title or not body:
            # Nothing rendered, so nothing has replaced the placeholder yet.
            return
        await _retire_thinking()
        async with cl.Step(name=title) as step:
            step.output = body

    try:
        with redaction.use_notice_state(notice_state), use_tool_cache(tool_cache):
            reply, history, _result = await invoke_agent(
                message.content, history, on_step=show_step
            )
    except Exception:
        logger.exception("Agent graph error")
        await running.close_all()
        await _retire_thinking()
        await cl.Message(
            content="An error occurred while processing your request. Please try again."
        ).send()
        # The send already happened before the graph raised, so what left the
        # machine has to be disclosed whether or not an answer came back. A
        # user who stops after the failure would otherwise never be told.
        await _deliver_notice(notice_state)
        return

    await running.close_all()
    await _retire_thinking()
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
