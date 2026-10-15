"""Chainlit chat application -- entry point for the AORTA Agent."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import chainlit as cl

from aorta.chat import redaction
from aorta.chat.config import UI_NO_WAIT_ENV, UI_VERBOSE_ENV
from aorta.chat.inference.providers.factory import get_backend
from aorta.chat.session import invoke_agent
from aorta.chat.decision_log import new_session_id
from aorta.chat.tools.cache import ToolCache, use_tool_cache
from aorta.chat.ui.welcome import tool_protocol, welcome_message

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

#: Stable key for the opt-in decision log, and its monotonically increasing
#: turn number. Kept in Chainlit's per-browser session with the history.
_DECISION_SESSION_KEY = "decision_log_session_id"
_DECISION_TURN_KEY = "decision_log_turn"


def _unavailable_message(reason: str) -> str:
    return f"**LLM backend unavailable**\n\n```\n{reason}\n```"


#: What each node contributes that is worth showing. A node absent from this
#: map renders nothing, which keeps plumbing like retrieve out of the way.
_NODE_TITLES = {
    "router": "Understanding the request",
    "select": "Choosing a diagnostic approach",
    "plan": "Planning the investigation",
    "act": "Gathering diagnostic evidence",
    "critic": "Verifying the answer",
}

# User-facing activity, never Python function names. Unknown/plugin tools use
# the generic fallback rather than leaking an implementation identifier.
_TOOL_PROGRESS = {
    "list_files": (
        "Inspecting the project structure",
        "AORTA is checking which project files and directories are available.",
    ),
    "read_file": (
        "Reading relevant source",
        "AORTA is reading a relevant project file for concrete evidence.",
    ),
    "search_code": (
        "Searching the codebase",
        "AORTA is searching the codebase for semantically related implementation details.",
    ),
    "grep_code": (
        "Searching for exact matches",
        "AORTA is checking the codebase for exact names and patterns.",
    ),
    "search_repo_map": (
        "Locating relevant code",
        "AORTA is checking the project index for relevant functions and classes.",
    ),
    "list_runs": (
        "Finding previous runs",
        "AORTA is locating previous run artifacts that may contain useful evidence.",
    ),
    "read_run_matrix": (
        "Reviewing the run matrix",
        "AORTA is reviewing the recorded outcomes for each tested configuration.",
    ),
    "read_run_env": (
        "Reviewing the recorded environment",
        "AORTA is checking the software and hardware environment captured for the run.",
    ),
    "search_run_artifacts": (
        "Searching previous run evidence",
        "AORTA is searching recorded run artifacts for matching symptoms.",
    ),
    "list_cluster_jobs": (
        "Finding recent cluster diagnostics",
        "AORTA is checking recent cluster diagnostics for an existing result.",
    ),
    "read_autopsy_report": (
        "Reviewing the diagnostic report",
        "AORTA is reviewing the completed diagnostic report and its supporting evidence.",
    ),
    "triage_assembly_source": (
        "Analyzing the GPU assembly",
        "AORTA is checking diagnostic evidence for missing waits and instruction dependencies in the supplied GPU assembly.",
    ),
    "triage_kernel_source": (
        "Checking the GPU kernel for races",
        "AORTA is checking GPU race-diagnostic evidence for the supplied kernel.",
    ),
    "triage_workload": (
        "Reviewing the workload diagnosis",
        "AORTA is checking cluster logs and diagnostic artifacts for the supplied workload.",
    ),
    "run_terminal_command": (
        "Running an approved diagnostic command",
        "AORTA is running an allowlisted command to gather additional evidence.",
    ),
}
_DEFAULT_TOOL_PROGRESS = (
    "Gathering additional evidence",
    "AORTA is using an analysis capability to gather additional evidence.",
)


def _tool_progress(name: object) -> tuple[str, str]:
    """Human-facing title and sentence for a tool progress event."""
    return _TOOL_PROGRESS.get(str(name), _DEFAULT_TOOL_PROGRESS)


def _node_reasoning(node: str, delta: dict) -> str:
    """A plain-language progress sentence without internal state or names."""
    if node == "router":
        if delta.get("route") == "action":
            return "This request needs diagnostic evidence before AORTA can answer."
        if delta.get("route") == "question":
            return "This request can be answered from the available project context."
        return "AORTA is determining how to handle this request."
    if node == "select":
        tools = delta.get("candidate_tools") or []
        if not tools:
            return "AORTA did not identify a specialized diagnostic and will use general evidence gathering."
        activities = "\n".join(
            f"{index}. {_tool_progress(tool)[0]}"
            for index, tool in enumerate(tools, 1)
        )
        return f"AORTA selected this diagnostic approach:\n\n{activities}"
    if node == "plan":
        if not delta.get("plan"):
            return ""
        return "AORTA has planned which evidence to gather and in what order."
    if node == "act":
        trace = delta.get("tool_trace") or []
        if not trace:
            return ""
        noun = "result" if len(trace) == 1 else "results"
        return f"AORTA gathered {len(trace)} diagnostic {noun} for the answer."
    if node == "critic":
        if delta.get("critic_feedback"):
            return "The draft needed stronger evidence, so AORTA is revising it."
        return "The answer is supported by the evidence gathered during this turn."
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
        self._finished: list[cl.Step] = []

    async def handle(self, delta: dict) -> None:
        """Open a step on a tool's announcement, close it on its completion."""
        name = delta.get("tool")
        # Keyed by call rather than name: a turn runs the same tool more than
        # once with different arguments, and closing by name would end the
        # wrong one.
        call = str(delta.get("id") or name)
        if delta.get("done"):
            await self._finish(
                call,
                name,
                delta.get("seconds"),
                delta.get("cancelled"),
                delta.get("failed"),
            )
            return
        title, activity = _tool_progress(name)
        step = cl.Step(name=title, type="tool")
        step.start = utc_now()
        step.output = activity
        await step.send()
        self._open[call] = step

    async def _finish(
        self,
        call: str,
        name,
        seconds,
        cancellation=None,
        failed=False,
    ) -> None:
        step = self._open.pop(call, None)
        if step is None:
            return  # a completion with nothing open; nothing to close
        _, activity = _tool_progress(name)
        took = f" in {seconds:g}s" if isinstance(seconds, (int, float)) else ""
        if cancellation in {"stopped", True}:
            step.output = f"{activity}\n\nThe operation stopped after cancellation{took}."
        elif cancellation == "still running":
            step.output = (
                f"{activity}\n\nCancellation was requested{took}, but the "
                "underlying work is still running."
            )
        elif failed:
            step.output = f"{activity}\n\nThe operation could not complete{took}."
        else:
            step.output = f"{activity}\n\nFinished{took}."
        step.end = utc_now()
        await step.update()
        if not failed and cancellation not in {"stopped", True, "still running"}:
            self._finished.append(step)

    async def remove_finished(self) -> None:
        """Remove completed progress rows after the final answer is visible."""
        while self._finished:
            step = self._finished.pop()
            try:
                await step.remove()
            except Exception:  # noqa: BLE001 - a dead session must not affect the answer
                logger.debug("Could not remove finished step %r", step.name, exc_info=True)

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
    cl.user_session.set(_DECISION_SESSION_KEY, new_session_id())
    cl.user_session.set(_DECISION_TURN_KEY, 0)
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

    # The line ``aorta chat`` logs at startup. Here it is per session, as the
    # banner is: the protocol can move between one session and the next.
    logger.info("LLM backend: %s (tool protocol: %s)", backend.describe(), tool_protocol())
    await cl.Message(content=welcome_message(backend.describe())).send()


#: Above this an attachment is staged and named rather than folded into the
#: prompt. Not a size the tools cannot handle -- the assembler is happy with a
#: whole code object -- but the size the *model* cannot handle: ``source``
#: travels as a tool argument, so anything sent that way has to be written out
#: again in full by the model to make the call. A listing too big to paste is
#: exactly the one worth attaching, so it goes to disk and the model is given
#: its name.
_INLINE_ATTACHMENT_BYTES = 32 * 1024

#: The ceiling on staging too. Text, so generous; a disassembled GEMM code
#: object is about 9 MB and should still be analysable.
_MAX_ATTACHMENT_BYTES = 64 * 1024 * 1024

#: Read as source. Everything else is refused by name so the user learns why
#: rather than watching their file be ignored.
_SOURCE_SUFFIXES = (
    ".s", ".asm", ".S", ".isa", ".disasm", ".txt",
    ".hip", ".cpp", ".cc", ".c", ".cu", ".h", ".hpp", ".py",
)


def _attached_source(message: cl.Message) -> tuple[str, list[str]]:
    """Fold any attached text files into the prompt, and say what was skipped.

    Chainlit's upload button is on -- and until this existed, what it did was
    discard the file: ``on_message`` read ``content`` and nothing else, so an
    attached listing was answered from the covering sentence alone, with
    nothing anywhere saying the file had not been read.

    Folded in as a fenced block rather than routed anywhere new, so the paste
    path handles it: a fence is what the harness reads as "the user is
    pointing at this", and tool selection already knows what to do with a
    message carrying code.
    """
    notes: list[str] = []
    blocks: list[str] = []
    for element in getattr(message, "elements", None) or []:
        name = getattr(element, "name", None) or "attachment"
        path = getattr(element, "path", None)
        if not path:
            notes.append(f"`{name}` arrived without a readable path, so it was skipped.")
            continue
        candidate = Path(path)
        if candidate.suffix not in _SOURCE_SUFFIXES:
            notes.append(
                f"`{name}` is not a source or listing file "
                f"({candidate.suffix or 'no suffix'}), so it was not read."
            )
            continue
        try:
            raw = candidate.read_bytes()
        except OSError as exc:
            notes.append(f"`{name}` could not be read ({exc.__class__.__name__}).")
            continue
        if len(raw) > _MAX_ATTACHMENT_BYTES:
            notes.append(
                f"`{name}` is {len(raw) // (1024 * 1024)} MB, past what this "
                "will stage."
            )
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            notes.append(
                f"`{name}` is not text. A .hsaco has to be disassembled first; "
                "attach the listing."
            )
            continue
        if not text.strip():
            notes.append(f"`{name}` is empty.")
            continue
        if len(raw) > _INLINE_ATTACHMENT_BYTES:
            # Too big to travel as a tool argument, which is what folding it
            # into the prompt would commit it to. Staged instead, and the model
            # is handed the name: the tool reads the file itself, so the size
            # of the listing stops being a limit on asking about it.
            try:
                staged = _stage_attachment(candidate)
            except OSError as exc:
                notes.append(
                    f"`{name}` could not be staged ({type(exc).__name__})."
                )
                continue
            kind, tool = _routed_by_suffix(candidate.suffix)
            blocks.append(
                # Backticked, and the argument spelled out. Written bare it
                # ran into the sentence's full stop, and a name the model
                # copies with a trailing "." is a name that does not resolve.
                f"The user attached `{name}` ({len(raw) // 1024} KB of "
                f"{kind}), staged as `{staged}`\n\n"
                f"It is too large to quote. Call {tool} with "
                f"source_file=`{staged}` and no source argument. Do not try to "
                "reproduce its contents."
            )
            continue
        blocks.append(f"Attached file `{name}`:\n\n```\n{text.strip()}\n```")
    return "\n\n".join(blocks), notes


#: Which tool reads which kind of attachment, by suffix. The accepted set spans
#: three tools and the staged message named one of them for all of it, so a
#: large .hip was described as assembly and pointed at the assembly tool.
_ASSEMBLY_SUFFIXES = frozenset({".s", ".asm", ".S", ".isa", ".disasm", ".txt"})
_WORKLOAD_SUFFIXES = frozenset({".py"})


def _routed_by_suffix(suffix: str) -> tuple[str, str]:
    """What to call a staged attachment, and which tool should read it.

    Assembly listings and Python scripts are what they look like. Everything
    else in the accepted set is HIP or C++ that the kernel tool compiles, which
    is also the right default for a suffix that is new here: a kernel sent to
    the assembly tool fails on the first instruction, which is a clearer
    outcome than a listing quietly compiled as a program.
    """
    if suffix in _ASSEMBLY_SUFFIXES:
        return "assembly", "triage_assembly_source"
    if suffix in _WORKLOAD_SUFFIXES:
        return "Python", "triage_workload"
    return "HIP source", "triage_kernel_source"


def _stage_attachment(source: Path) -> str:
    """Copy *source* under the jobs root and return the name the tools take.

    Under the jobs root because that is the sandbox the triage tools resolve
    within, so a staged name can be handed to a model without giving it a way
    to name anything else. The returned value is relative for the same reason:
    an absolute path would be refused by the containment check on the way back
    in, and it keeps the model from reading a layout it has no use for.
    """
    from aorta.chat.config import settings

    root = Path(settings.jobs_root) / "chat-uploads"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="upload-", dir=root))
    target = work / source.name
    target.write_bytes(source.read_bytes())
    return str(target.relative_to(Path(settings.jobs_root)))


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

    attached, skipped = _attached_source(message)
    if skipped:
        # Said before the answer, not after: a user who attached the wrong
        # thing should learn that before reading a reply that did not use it.
        await cl.Message(
            content="\n".join(f"_{note}_" for note in skipped)
        ).send()
    question = f"{message.content}\n\n{attached}" if attached else message.content

    history: list = cl.user_session.get("history", [])
    decision_session = cl.user_session.get(_DECISION_SESSION_KEY)
    if not decision_session:
        decision_session = new_session_id()
        cl.user_session.set(_DECISION_SESSION_KEY, decision_session)
    decision_turn = int(cl.user_session.get(_DECISION_TURN_KEY) or 0) + 1
    cl.user_session.set(_DECISION_TURN_KEY, decision_turn)
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
                question,
                history,
                on_step=show_step,
                session_id=decision_session,
                turn=decision_turn,
                front_door="ui",
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
    await running.remove_finished()
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
