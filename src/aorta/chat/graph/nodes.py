"""LangGraph node implementations: Router, Plan, Retrieve, Act, Critic."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from aorta.chat.config import settings
from aorta.chat.graph.state import AgentState
from aorta.chat.inference.vllm_client import get_chat_llm
from aorta.chat.plugins import ChatTool, enabled_builtins, load_chat_tools
from aorta.chat.rag.repo_map import load_repo_map
from aorta.chat.rag.retriever import get_retriever
from aorta.chat.redaction import redact_for_send

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the AORTA Codebase Assistant, an AI agent that helps \
developers understand, navigate, and work with the AORTA codebase.

RULES:
1. Only answer questions about the AORTA codebase, or about the AORTA runs on \
   this machine. Politely refuse anything else.
2. When referencing code, always cite file paths and line numbers.
3. You have tools to explore the codebase: list_files, read_file, search_code, \
   grep_code and search_repo_map. You also have tools for this machine's own \
   AORTA run results: list_runs, read_run_matrix, read_run_env, and \
   search_run_artifacts. Use those when the question is about what a run did \
   rather than what the code says.
4. NEVER fabricate or guess commands. Before suggesting any command, you MUST first \
   use search_code or read_file to find the actual scripts, entry points, or \
   configuration in the codebase. Only generate commands that are grounded in real \
   files you have read or searched.
5. When generating commands for running scenarios, provide the exact command \
   derived from actual codebase files, explain what it does, and note any \
   prerequisites or expected output.
6. If you cannot find a relevant script or entry point in the codebase, say so \
   honestly rather than making one up.
7. Be concise and precise.
8. For "find all", "search for", or "list all" queries, do NOT stop after one search. \
   Use multiple search_code calls with different phrasings, use grep_code for exact \
   pattern matches, and use search_repo_map to check for entries you may have missed.
9. Before answering broad queries, use search_repo_map to cross-reference the \
   function/class index for any matching signatures you may have overlooked.
10. When multiple files are relevant, list ALL of them with file paths.
11. A run artifact that reports a field as "unknown" or "NOT RECORDED" did not \
   record it. That is not zero and not a pass -- say the run did not record it.

RETRIEVED CONTEXT:
{context}
"""

#: answer_node has no tool-execution loop, so its prompt must not offer tools.
#: SYSTEM_PROMPT advertises six of them and says to call them before answering,
#: which a reasoning model will try to obey: gpt-oss emits the tool call into its
#: reasoning channel, returns empty content, and the query dead-ends. Queries
#: that genuinely need tools are what the router's "action" path is for.
ANSWER_PROMPT = """\
You are the AORTA Codebase Assistant, an AI agent that helps \
developers understand, navigate, and work with the AORTA codebase.

You have NO tools available. Answer using only the RETRIEVED CONTEXT below \
and the conversation so far.

RULES:
1. Only answer questions about the AORTA codebase, or about the AORTA runs on \
   this machine. Politely refuse anything else.
2. When referencing code, always cite file paths and line numbers.
3. NEVER fabricate file paths, commands, flags, or behaviour. Everything you \
   state must be visible in the RETRIEVED CONTEXT.
4. If the RETRIEVED CONTEXT does not contain the answer, say so plainly and name \
   what is missing, so the user can ask a more specific question. Do not guess, \
   and do not attempt to call tools or emit tool-call syntax.
5. When multiple files are relevant, list ALL of them with file paths.
6. Be concise and precise.

RETRIEVED CONTEXT:
{context}
"""

ROUTER_PROMPT = """\
Classify the user's latest message into one of two categories:
- "question": a simple, specific question that can be answered with retrieved context \
  alone (e.g. "What does function X do?", "How is class Y structured?")
- "action": requires using tools to search, list, read files, run commands, or find \
  multiple items (e.g. "Find all functions that ...", "Search for ...", "List all ...", \
  "How do I run ...", "Show me the files in ...")

If the message asks to find, search, list, or enumerate multiple items, classify as action.
If in doubt, classify as action.

Reply with ONLY the single word: question or action
"""

_BUILTIN_PLAN_PROMPT = """\
You are a planning agent. Given the user request and the repository map, \
break down the task into concrete steps.

Available tools:
1. list_files(path) - List files and directories under a path.
2. read_file(file_path) - Read the contents of a file.
3. search_code(query) - Semantic search for code related to the query.
4. grep_code(pattern, path) - Regex search across files (e.g. "def.*config").
5. search_repo_map(query) - Search the function/class index for matching entries.
6. list_runs(path) - List this machine's AORTA run directories.
7. read_run_matrix(path) - Read a run's per-cell pass/fail matrix.
8. read_run_env(path) - Read a run's environment snapshot.
9. search_run_artifacts(query) - Semantic search over indexed run artifacts.

For each step, specify which tool to use and what arguments. For "find all" or \
"search for" queries, plan MULTIPLE searches with different phrasings and tools \
(semantic search + regex grep + repo map lookup) to ensure completeness.

Output your plan as a numbered list. Be specific about file paths.
"""

#: Resolved once, at import: the graph is built per process and re-reading the
#: entry points on every act round would buy nothing. A plugin installed while
#: a REPL session is open is not picked up until the next one.
CHAT_TOOLS: dict[str, ChatTool] = load_chat_tools()

TOOL_REGISTRY: dict[str, BaseTool] = {name: entry.tool for name, entry in CHAT_TOOLS.items()}

_BUILTIN_TOOL_DESCRIPTIONS = """\
You have these tools. To call one, output EXACTLY this format on its own line:

ACTION: tool_name(arg1="value1", arg2="value2")

Available tools:

1. list_files(path=".") - List files and directories under a path in the AORTA codebase.
2. read_file(file_path="path/to/file") - Read the contents of a file in the AORTA codebase.
3. search_code(query="search terms", k=10) - Semantic search for code related to the query (k = max results).
4. grep_code(pattern="regex", path=".") - Regex search across files (e.g. "def.*config", "class.*Config").
5. search_repo_map(query="keyword") - Search the function/class index for matching entries.
6. list_runs(path=".") - List this machine's AORTA run directories and their artifacts.
7. read_run_matrix(path="run_dir") - Read a run's matrix.json: per-cell pass/fail and failure hints.
8. read_run_env(path="run_dir") - Read a run's environment snapshot (ROCm version, probe completeness).
9. search_run_artifacts(query="terms") - Semantic search over this machine's indexed run artifacts.

After receiving tool results, you may call another tool or provide your final answer.
When you are done gathering information and ready to answer, output your final response \
WITHOUT any ACTION: lines.

IMPORTANT:
- If the RETRIEVED CONTEXT fully and completely answers the question, respond directly.
- If the question asks to "find all", "list all", or "search for" something, the \
  RETRIEVED CONTEXT is likely incomplete. Use search_code with multiple different \
  query phrasings, grep_code for exact pattern matches, and search_repo_map to \
  check the function/class index for completeness.
- If you need more information (listing files, reading specific files, counting things, \
  exploring directories), call tools FIRST before answering.
- NEVER fabricate file paths, commands, or flags -- only use what the tools return or \
  what is in the RETRIEVED CONTEXT.
- For questions about file counts, directory contents, or running experiments, ALWAYS \
  use the listing and search tools to get real data.
- Run artifacts report fields they did not record as "unknown", listed under \
  "NOT RECORDED". Never read one as zero or as a pass. If the field that decides \
  the question is unknown, say the run did not record it.
"""


def _summary_line(tool: BaseTool) -> str:
    """First line of a tool's description, which is the model's only summary.

    ``@tool`` refuses a function with no docstring, so an empty description
    means a hand-built ``BaseTool`` -- possible, and not worth an IndexError.
    """
    lines = (tool.description or "").strip().splitlines()
    return lines[0].strip() if lines else "no description"


def _plugin_tool_help(tools: dict[str, ChatTool]) -> str:
    """Advertise plugin-contributed tools, or return "" when there are none.

    The hand-written lists above cover the built-ins; a tool discovered from the
    ``aorta.chat_tools`` entry-point group has to describe itself. Only the text
    protocol needs this -- ``bind_tools()`` sends every tool's real schema, so
    the native protocol offers plugin tools whether or not the prompt says so.

    Returns the empty string when nothing is installed, so both prompts stay
    byte-identical to what a user with no plugins had before.
    """
    extra = [entry for entry in tools.values() if entry.source_package != "aorta"]
    if not extra:
        return ""
    lines = [
        f"{index}. {entry.name}(...) - {_summary_line(entry.tool)} "
        f"[from {entry.source_package}]"
        # Counted from the built-ins actually registered, not from
        # BUILTIN_CHAT_TOOLS, whose length stopped saying how many the prompt
        # listed once the shell tool became conditional.
        for index, entry in enumerate(extra, start=len(enabled_builtins()) + 1)
    ]
    return "\nAdditional tools contributed by installed plugins:\n\n" + "\n".join(lines) + "\n"


#: Appended to the prompts only when the shell tool is registered. Keeping it
#: out of the numbered lists is the point: a prompt that advertises a tool the
#: registry does not hold teaches the model to emit a call that can only ever be
#: refused, and tells prompt-injected text that a shell is worth asking for.
_SHELL_TOOL_ACT_HELP = (
    "\nAlso available:\n\n"
    '- run_terminal_command(command="cmd") - Run one allowlisted command, '
    "starting in the AORTA directory. Not a filesystem sandbox; prefer "
    "read_file, list_files and grep_code where they will do.\n"
)
_SHELL_TOOL_PLAN_HELP = (
    "\nAlso available:\n\n"
    "- run_terminal_command(command) - Run one allowlisted command. Not a "
    "filesystem sandbox; prefer the file and search tools where they will do.\n"
)


def _shell_tool_help(fragment: str) -> str:
    """*fragment* when the shell tool is switched on, otherwise nothing."""
    from aorta.chat.config import settings

    return fragment if settings.enable_shell_tool else ""


TOOL_DESCRIPTIONS = (
    _BUILTIN_TOOL_DESCRIPTIONS
    + _shell_tool_help(_SHELL_TOOL_ACT_HELP)
    + _plugin_tool_help(CHAT_TOOLS)
)
PLAN_PROMPT = (
    _BUILTIN_PLAN_PROMPT
    + _shell_tool_help(_SHELL_TOOL_PLAN_HELP)
    + _plugin_tool_help(CHAT_TOOLS)
)


def _get_llm(**kwargs):
    return get_chat_llm(**kwargs)


async def _send(llm: Any, messages: list[Any]) -> Any:
    """Invoke *llm*, redacting the messages on their way out (Decision 16).

    Every node sends through here rather than calling ``ainvoke`` directly, so
    the gate cannot be bypassed by a node added later. It takes the already-
    bound model, so the tool-calling path is covered too.
    """
    return await llm.ainvoke(redact_for_send(messages))


def _build_system_message(context: str = "") -> SystemMessage:
    return SystemMessage(
        content=SYSTEM_PROMPT.format(context=context)
    )


def _build_answer_message(context: str = "") -> SystemMessage:
    """System message for the tool-free Q&A path."""
    return SystemMessage(content=ANSWER_PROMPT.format(context=context))


#: Fields a serving stack may expose the model's reasoning on.
#: ``langchain-openai`` surfaces gpt-oss's channel as ``reasoning``; vLLM's own
#: OpenAI server calls it ``reasoning_content``. Both are read because neither
#: is guaranteed to be there.
_REASONING_FIELDS = ("reasoning", "reasoning_content")


def _reasoning_channel(response: Any) -> str:
    """The model's reasoning text, from whichever field the stack exposes it on.

    Read defensively and used only as a signal. Whether any particular gateway
    populates either field is not knowable from here, so nothing may *depend* on
    it being present -- see :func:`_is_reasoning_dead_end`.
    """
    extra = getattr(response, "additional_kwargs", None) or {}
    for field in _REASONING_FIELDS:
        value = extra.get(field)
        if value:
            return str(value)
    return ""


def _output_tokens(response: Any) -> int:
    """Output tokens *response* reports, or 0 when it reports none usably."""
    usage = getattr(response, "usage_metadata", None) or {}
    try:
        return int(usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        return 0


#: Finish reasons that say the reply was cut off or filtered rather than
#: completed. OpenAI spells them ``length`` and ``content_filter``; Anthropic
#: reports ``max_tokens`` on ``stop_reason``. Used only to *disqualify* the
#: token signal, never as a requirement -- see :func:`_is_reasoning_dead_end`.
_INCOMPLETE_FINISH_REASONS = frozenset({"length", "content_filter", "max_tokens"})


def _finish_reason(response: Any) -> str:
    """Why the provider says generation stopped, lowercased, or "" if unsaid.

    ``finish_reason`` is the OpenAI field and ``stop_reason`` the Anthropic one;
    both arrive on ``response_metadata`` and neither is guaranteed, so "" means
    "the stack did not say" rather than "it completed".
    """
    metadata = getattr(response, "response_metadata", None) or {}
    for field in ("finish_reason", "stop_reason"):
        value = metadata.get(field)
        if value:
            return str(value).strip().lower()
    return ""


def _log_empty_content(response: Any, node: str) -> None:
    """Record why a node produced no text, when the model still spent tokens.

    Reasoning models can put everything in a side channel and return empty
    content. The token counts make that diagnosable; the channel itself is read
    through :func:`_reasoning_channel`, which tries every field in
    :data:`_REASONING_FIELDS` because no stack was known to populate one when
    this was written -- so it is read defensively in case a version or a
    gateway does.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    logger.warning(
        "%s produced no text despite %s output tokens. A reasoning model may "
        "have returned only internal reasoning.",
        node,
        usage.get("output_tokens", "an unknown number of"),
    )
    reasoning = _reasoning_channel(response)
    if reasoning:
        logger.debug("%s reasoning channel: %s", node, reasoning)


def _close_paren(text: str, start: int) -> int | None:
    """Index of the ``)`` closing the ``(`` at *start*, or None if unbalanced.

    Quote-aware, because the argument values are shell and Python fragments:
    stopping at the first ``)`` truncated every call whose value contained one,
    of which ``run_terminal_command(command="python -c 'print(1)'")`` is the
    ordinary case rather than a corner one. A truncated call still parsed, so
    the tool ran with a silently different argument.

    Escape-aware for the same reason: without it a ``\\"`` closed the quote it
    was written to sit inside, and the scan resumed treating the rest of the
    value as code.
    """
    quote: str | None = None
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif quote is not None:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _parse_action(text: str) -> tuple[str, dict] | None:
    """Extract a tool call from ACTION: tool_name(k="v", ...) format."""
    import re

    match = re.search(r"ACTION:\s*(\w+)\(", text)
    if not match:
        return None
    end = _close_paren(text, match.end() - 1)
    if end is None:
        return None
    tool_name = match.group(1)
    if tool_name not in TOOL_REGISTRY:
        return None
    kwargs = _parse_action_kwargs(text[match.end() : end])
    if kwargs is None:
        return None
    return tool_name, kwargs


def _parse_action_kwargs(args: str) -> dict | None:
    """Parse the argument list as keyword literals, or ``None`` to refuse it.

    Matching values with ``[^"]*`` ended them at the first quote character,
    escaped or not, so ``search_code(query="say \\"hi\\"")`` parsed as ``say \\``
    and searched for that -- the tool ran, and the answer was built from the
    wrong query with nothing on screen to say so. Scanning with ``finditer``
    also meant anything it failed to match was passed over in silence.

    The syntax the prompt asks for is Python's, so Python parses it, and
    anything that does not parse is refused whole. A refusal reaches the critic
    and gets another attempt; a half-read call cannot be noticed by anything.
    """
    import ast

    args = args.strip()
    if not args:
        return {}
    try:
        call = ast.parse(f"_tool({args})", mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(call, ast.Call) or call.args:
        # Positional arguments have no name to map onto the tool's schema.
        return None
    kwargs: dict = {}
    for keyword in call.keywords:
        if keyword.arg is None:  # ``**rest``
            return None
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            # A bare name, an f-string, a concatenation: not a literal, so its
            # value is not knowable here.
            return None
    return kwargs


def _normalise_tool_name(tool_name: str) -> str:
    """Strip harmony control markers a provider may leak into the tool name.

    gpt-oss speaks the harmony format internally, and a serving stack whose
    parser is imperfect emits names like ``search_code<|channel|>commentary``.
    Everything from the first ``<|`` is protocol, not part of the name.
    """
    return tool_name.split("<|", 1)[0].strip()


async def _execute_tool(tool_name: str, kwargs: dict) -> str:
    """Call a tool and return its string result, or an error the model can read.

    An unknown name must not raise: in the native protocol the name comes
    straight from the provider, so a hallucinated or mangled one would abort the
    whole graph run instead of giving the model a chance to correct itself.

    ``ainvoke``, because a tool call is the longest blocking thing in the act
    loop and the loop is awaited from a Chainlit request handler: ``grep_code``
    walks the tree, ``search_code`` embeds a query, and
    ``run_terminal_command`` runs a subprocess up to its timeout. ``BaseTool``
    hands a tool with no coroutine to ``run_in_executor``, so a synchronous
    tool still runs off the event loop without needing its own wrapper here.
    """
    name = _normalise_tool_name(tool_name)
    tool_fn = TOOL_REGISTRY.get(name)
    if tool_fn is None:
        logger.warning("Model asked for unknown tool %r", tool_name)
        return (
            f"Error: there is no tool named {name!r}. Available tools: "
            f"{', '.join(sorted(TOOL_REGISTRY))}."
        )
    try:
        result = await tool_fn.ainvoke(kwargs)
    except Exception as exc:
        result = f"Tool error: {exc}"
    return str(result)


# ──────────────────── Router ─────────────────────


#: Where a reply that names no route goes. ``question`` because that branch is
#: retrieval plus a single answer call and needs no tool protocol at all, so its
#: worst case is a retrieval-only answer rather than none. ``action`` was the
#: old fallback, and it is the branch a model returning empty content cannot
#: drive -- so an empty router reply, which is exactly what such a model
#: produces, was routed into the one branch guaranteed to fail.
_ROUTER_FALLBACK_ROUTE = "question"


def _parse_route(route_text: str) -> str | None:
    """The route *route_text* names, or ``None`` if it names neither or both.

    Substring matching is generous enough for a one-word reply, but it has to be
    symmetric. Testing only for ``question`` made every other string mean
    ``action`` by omission -- including ``""``, which is not a classification.
    """
    named = [name for name in ("question", "action") if name in route_text]
    return named[0] if len(named) == 1 else None


async def router_node(state: AgentState) -> dict[str, Any]:
    """Classify intent: pure Q&A vs action-requiring."""
    llm = _get_llm(temperature=0.0, streaming=False)
    last_msg = state["messages"][-1]

    response = await _send(
        llm,
        [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_msg.content),
        ]
    )
    route_text = str(response.content or "").strip().lower()
    route = _parse_route(route_text)
    if route is None:
        route = _ROUTER_FALLBACK_ROUTE
        logger.warning(
            "Router reply %r does not name exactly one route; classifying as "
            "%s.",
            route_text,
            route,
        )
    else:
        logger.info("Router classified as: %s (raw: %r)", route, route_text)
    return {"route": route}


# ──────────────────── Plan ───────────────────────


async def plan_node(state: AgentState) -> dict[str, Any]:
    """Generate a step-by-step plan for action-type requests."""
    llm = _get_llm(temperature=0.1)
    # Offloaded for the same reason as retrieval: this reads the whole repo
    # map off disk, which is around 3 MB for AORTA, on a coroutine a Chainlit
    # request handler is awaiting.
    repo_map = await asyncio.to_thread(load_repo_map)
    last_msg = state["messages"][-1]

    response = await _send(
        llm,
        [
            SystemMessage(
                content=PLAN_PROMPT + f"\n\nREPOSITORY MAP:\n{repo_map}"
            ),
            HumanMessage(content=last_msg.content),
        ]
    )
    return {"plan": response.content}


# ──────────────────── Retrieve ───────────────────


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Gather context for the user's query: source chunks, plus run artifacts.

    Both, because this feeds the branch that has no tools. Retrieving only
    source left a run question answerable only from code.

    Neither retrieval may run on the event loop. This coroutine is awaited from
    a Chainlit request handler, and with remote embeddings each blocks on network
    I/O while with local embeddings each blocks on CPU work in the ONNX model --
    so a synchronous call here stalls every concurrent session behind whichever
    one is retrieving.
    """
    last_human = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"retrieved_context": ""}

    try:
        retriever = get_retriever()
        # ``ainvoke``, not ``invoke``. This is enough for the CPU-bound local
        # provider as well as the remote one: ``SqliteVecStore`` implements only
        # the synchronous search, and ``VectorStore``'s async default hands that
        # to ``run_in_executor`` -- so the embedding work lands on a thread
        # either way, which is what the issue's ``asyncio.to_thread`` caveat
        # asks for. ``search_run_docs`` below is a plain function with no such
        # shim, so that one is dispatched explicitly.
        docs = await retriever.ainvoke(last_human)
    except FileNotFoundError:
        return {
            "retrieved_context": "(Index not built yet -- run indexing first.)"
        }

    chunks = []
    for doc in docs:
        src = doc.metadata.get("source", "?")
        chunks.append(f"### {src}\n```\n{doc.page_content}\n```")
    chunks.extend(await asyncio.to_thread(_run_artifact_chunks, last_human))

    if not chunks:
        return {"retrieved_context": "(No relevant code found.)"}
    return {"retrieved_context": "\n\n".join(chunks)}


def _run_artifact_chunks(query: str) -> list[str]:
    """Run-artifact context for the branch that has no tools. Blocking.

    Deliberately left synchronous and dispatched with :func:`asyncio.to_thread`
    by its caller. ``search_run_docs`` goes straight to
    ``max_marginal_relevance_search`` rather than through a retriever, so there
    is no ``ainvoke`` shim to offload the embedding work -- and the work is the
    same network or ONNX-CPU cost the retriever pays.

    The router sends a specific question -- "why did this sweep fail?" -- down
    the ``question`` path by design, because it is specific. That branch never
    calls a tool, so without this it could only answer from source code, and
    the run artifacts the command exists to explain were unreachable by the
    route most likely to ask about them (issue #433).

    A missing collection is the common case, not an error: most installs have
    never run ``aorta chat index runs``. This is supplementary context, so
    *nothing* it can raise -- an absent collection, an unreadable index, a
    remote embedding provider that is down -- may take the answer with it; the
    query is still answerable from source, which is what it did before this
    existed. Hence the broad catch, at debug level so it is still diagnosable.
    """
    from aorta.chat.rag.runs import search_run_docs

    try:
        docs = search_run_docs(query, settings.search_tool_k)
    except Exception as exc:  # supplementary context; see the docstring
        logger.debug("No run-artifact context for this query: %s", exc)
        return []
    return [
        f"### run artifact: {doc.metadata.get('source', '?')} "
        f"({doc.metadata.get('artifact_kind', '?')})\n```\n{doc.page_content}\n```"
        for doc in docs
    ]


# ──────────────────── Search-query detection ─────


_SEARCH_KEYWORDS = [
    "find all", "find every", "find the",
    "search for", "search the",
    "list all", "list every", "list the",
    "show all", "show every", "show me all",
    "which functions", "which classes", "which files", "which modules",
    "where is", "where are", "where does",
    "what functions", "what classes", "what files",
]


def _is_search_query(text: str) -> bool:
    """Heuristic: does the user query ask to find/search/list multiple items?"""
    lower = text.lower()
    return any(kw in lower for kw in _SEARCH_KEYWORDS)


_SEARCH_FORCE_MSG = (
    "SEARCH QUERY DETECTED: This query asks to find, search, or list items "
    "across the codebase. The RETRIEVED CONTEXT is likely incomplete.\n\n"
    "You MUST:\n"
    "1. Call grep_code with a relevant regex pattern (e.g. 'def.*config', "
    "'class.*Config') to find exact matches.\n"
    "2. Call search_repo_map with relevant keywords to check the function/class index.\n"
    "3. Optionally call search_code with different query phrasings for broader coverage.\n"
    "4. Only provide your final answer AFTER using at least 2 different tools.\n\n"
    "Do NOT answer from RETRIEVED CONTEXT alone."
)

_SEARCH_REPROMPT_BODY = (
    "You responded without using any tools. This is a search query that requires "
    "thorough exploration of the codebase. You MUST call grep_code or search_repo_map "
    "FIRST to find all relevant results. "
)

#: One nudge per protocol. The text loop asks for a syntax it parses; the native
#: loop asks for a tool call. Sending the ``ACTION:`` wording to a model bound
#: through the function-calling API asked it for the one thing that path does not
#: read -- and the auto-escalation below means users who never chose the native
#: protocol now reach it.
_SEARCH_REPROMPT_MSG = _SEARCH_REPROMPT_BODY + "Please issue an ACTION: line now."
_SEARCH_REPROMPT_NATIVE_MSG = _SEARCH_REPROMPT_BODY + "Call one of them now."

#: Consecutive rounds where the model neither called a tool nor said anything.
#: A model that cannot drive the protocol will not learn it by round eight, and
#: each round is a billed call: gpt-oss burned 11 on one query before this cap.
_MAX_UNPRODUCTIVE_ROUNDS = 2

#: The same cap for the one native round the escalation below buys, and it is
#: 1 rather than 2 on purpose. A model that can drive the function-calling API
#: returns structured ``tool_calls`` on its first round, which counts as
#: productive; a model that returns nothing again has answered the only question
#: the retry was asked. A second round could not add information, and this
#: escalation must not reopen the call-count problem the cap above exists for.
_MAX_ESCALATED_ROUNDS = 1

#: Require the reasoning channel before escalating, rather than merely letting
#: it confirm. **Left False deliberately, and it is one line to flip.**
#:
#: Review asked for the trigger to be sharpened with "and the reasoning
#: channel is populated", because "empty content plus non-zero output tokens"
#: also matches a truncation, a content filter or a stop-sequence bug. That
#: needs one fact this repository cannot supply: whether the reporter's gateway
#: populates the field. Checking it needs a live query against their AMD APIM
#: endpoint, and **that check has not been run** -- there are no credentials for
#: it here.
#:
#: So the trigger runs on the documented token signature and *tightens itself*
#: when the channel turns out to be there (see
#: :func:`_is_reasoning_dead_end`). Flipping this to True is the whole of the
#: change if the live check later says the field is reliable; until then,
#: requiring an unverified field would mean the escalation silently never fires
#: on the very gateway it was written for.
_REQUIRE_REASONING_CHANNEL = False


def _is_reasoning_dead_end(response: Any) -> bool:
    """Whether an empty reply is the documented reasoning-model signature.

    ``docs/chat/providers.md`` calls the signature distinctive: ``finish_reason:
    stop``, non-zero output tokens, empty ``content``. The model spent tokens
    and returned no text, so the text went somewhere this protocol cannot read.

    A populated reasoning channel is near-proof of that, and its absence proves
    nothing -- a gateway may strip it, and ``langchain-openai`` may not surface
    it. Hence the asymmetry: either fact alone is enough by default, and
    :data:`_REQUIRE_REASONING_CHANNEL` turns the channel into a requirement in
    one line if it is ever shown to be dependable.

    The token half carries the finish reason with it, because "spent tokens and
    said nothing" also describes a length cutoff and a content filter, and
    neither is a protocol problem: switching the process to native would not
    make a truncated reply complete. A *stated* cutoff therefore disqualifies
    it. An unstated one does not, for the same reason the reasoning channel is
    not required -- the field is optional and differently named per provider,
    and demanding it would mean the escalation never fires on a gateway that
    omits it. The reasoning channel stays sufficient on its own either way: it
    is direct evidence the answer went somewhere this protocol cannot read.
    """
    has_reasoning = bool(_reasoning_channel(response))
    cut_short = _finish_reason(response) in _INCOMPLETE_FINISH_REASONS
    spent_tokens = _output_tokens(response) > 0 and not cut_short
    if _REQUIRE_REASONING_CHANNEL:
        return has_reasoning and spent_tokens
    return has_reasoning or spent_tokens


def _dead_end_signature(response: Any) -> str:
    """What was actually observed, for a log line that must not overstate it.

    :func:`_is_reasoning_dead_end` has two sufficient halves and either fires
    alone, so a message naming the reasoning channel unconditionally reported a
    fact that had usually not been observed.
    """
    parts = []
    tokens = _output_tokens(response)
    if tokens:
        parts.append(f"{tokens} output tokens spent on empty content")
    if _reasoning_channel(response):
        parts.append("reasoning returned on a side channel")
    return ", ".join(parts) or "empty content"


@dataclass
class _EscalationState:
    """What this process has learned about which tool protocol the endpoint serves.

    Process-wide on purpose, and deliberately *not* the per-session shape
    :class:`aorta.chat.redaction.NoticeState` uses. That notice is a disclosure
    owed to one user, so one Chainlit session's must not consume another's.
    This is a fact about the endpoint every session in the process talks to, so
    scoping it per session would put the two wasted rounds back on every
    browser tab -- which is the cost it exists to remove.

    A mutable object rather than two module globals so that the commit and the
    failure count read and write one thing. There is no rollback to pair them
    with: the switch is thrown only once native has answered, so the state it
    would have had to undo is never reached.
    """

    #: Set once native has actually answered, never merely on deciding to try
    #: it. What moves *later* queries straight to native, through
    #: :func:`_resolved_tool_mode`.
    escalated: bool = False
    #: Escalated attempts that did not answer: one that raised, and one that
    #: came back as empty as the text round it was sent to rescue. Counted
    #: rather than latched because one failure does not say which kind it was: a
    #: stock local vLLM without ``--enable-auto-tool-choice`` and a matching
    #: ``--tool-call-parser`` refuses the protocol permanently, while a timeout
    #: or a 503 refuses it for a minute.
    #:
    #: A silent attempt counts the same as a raised one so that the retry cannot
    #: be billed on every query to a model that answers on neither protocol.
    #: Nothing was learned about the endpoint either way, and the budget is the
    #: same budget.
    native_failures: int = 0
    #: Escalated probes begun, monotonic. Each probe keeps the value it read
    #: here, which is what lets :func:`_record_escalation_failure` tell a second
    #: *sequential* attempt from a second *concurrent* one.
    probes_begun: int = 0
    #: ``probes_begun`` as it stood when a failure was last counted. Any probe
    #: that began at or before this was already in flight then, so it belongs to
    #: the same wave and must not be billed again.
    counted_watermark: int = 0


#: Remediation for a native request that failed outright, appended by
#: :func:`_record_escalation_failure` when its caller asks. One copy, because
#: two callers print it and a third deliberately does not -- which is a rule
#: about when it applies, and rules with copies drift.
_ENDPOINT_HINT = (
    " If it is a local vLLM, it must be started with --enable-auto-tool-choice "
    "and a matching --tool-call-parser to serve this protocol."
)

#: Failed escalated attempts before native is written off for the process.
#: Two, not one: writing it off on a single failure lets a bad minute strand a
#: long-lived ``aorta chat ui`` server on the protocol its model cannot drive,
#: and the cost of being wrong the other way is one extra failed call across
#: the whole process. An endpoint restarted with the flag is not picked up
#: either way; restarting chat is the way back.
_MAX_NATIVE_FAILURES = 2

_escalation = _EscalationState()


def reset_tool_mode_escalation() -> None:
    """Forget what was learned about the protocol. For tests, which share one process.

    Restores every field from a fresh :class:`_EscalationState` rather than
    naming them, because it had already fallen behind once: two fields were
    added for the concurrent-failure fix and an enumerated reset would have
    leaked both into the next test, as state that looks like a previous
    session's probe.
    """
    for field in dataclass_fields(_EscalationState):
        setattr(_escalation, field.name, field.default)


def _tool_mode_is_explicit() -> bool:
    """Whether ``llm_tool_mode`` was chosen by the user rather than defaulted.

    ``AORTA_CHAT_LLM_TOOL_MODE`` is a documented knob, and a user who set
    ``text`` deliberately must not be overridden -- a stock local vLLM without
    ``--enable-auto-tool-choice`` and a matching ``--tool-call-parser`` cannot
    serve the native protocol at all, so escalating there would trade a bad
    answer for a failed request.

    pydantic-settings records which fields a source supplied, so "the user asked
    for text" is distinguishable from "text is the default". Every source
    counts, which is the intended reading: each one is someone stating a
    preference. Today that means ``AORTA_CHAT_LLM_TOOL_MODE`` and the profile
    file; ``aorta chat`` has no flag for the protocol, but a future one would
    count too, because :func:`aorta.chat.config.configure` passes flags as
    constructor arguments and those land in ``model_fields_set`` as well.
    """
    fields_set = getattr(settings, "model_fields_set", None)
    if not isinstance(fields_set, (set, frozenset)):
        # Not a real pydantic model -- a stand-in settings object in a test.
        # Answer "explicit", because this gate exists to protect a stated
        # preference and ``False`` is the answer that overrides one. Failing
        # open here would let the escalation move a mode the user chose, which
        # is the single thing the docstring above promises it will not do.
        return True
    return "llm_tool_mode" in fields_set


def _resolved_tool_mode() -> str:
    """The tool protocol to use now, after any auto-escalation."""
    mode = settings.llm_tool_mode.strip().lower()
    if mode == "text" and _escalation.escalated:
        return "native"
    return mode


def _escalate_to_native(response: Any) -> bool:
    """Whether to retry this round in the native protocol.

    A decision and nothing else: it records no state and writes no log, so two
    Chainlit sessions that dead-end at once both get their retry. Making this
    the place the switch was thrown meant the second one read its own dead end
    as somebody else's business -- it saw the flag already set, returned False,
    and fell through to the degraded fallback without ever trying the protocol
    that had just been chosen for it.

    The switch is thrown in :func:`_commit_escalation` instead, once native has
    answered. Three things have to hold here:

    * the failure has to look like the protocol rather than the query, or a
      truncation would silently change the user's configured protocol;
    * the user must not have chosen the protocol themselves;
    * and native must not already have failed to answer
      :data:`_MAX_NATIVE_FAILURES` times, or every query would pay a round to
      rediscover that -- *unless* native has since been proven to work, which
      answers the budget's question before it is asked.

    That last exception is not bookkeeping. :func:`_resolved_tool_mode` routes
    every request that *arrives* after a commit straight to native, but a
    request already inside :func:`_act_text` resolved its mode before the
    commit and still reaches this gate. Without the exception a spent budget
    refuses it, and it answers from retrieved context instead of retrying on
    the protocol another request has just demonstrated -- one degraded answer
    per request in flight across the commit. The budget exists to stop queries
    paying to rediscover that native does not work; once it does, there is
    nothing left to rediscover.
    """
    if _tool_mode_is_explicit():
        return False
    if not _escalation.escalated and _escalation.native_failures >= _MAX_NATIVE_FAILURES:
        return False
    return _is_reasoning_dead_end(response)


#: What proved the protocol works, for :func:`_commit_escalation` to announce.
#: The switch moves on either, but they are not the same event and the log must
#: not say the first when it was the second: a retry can demonstrate structured
#: tool calling and *still* fail to answer, if the call that would have
#: synthesised the results is the one the backend dropped.
_ANSWERED = "Native function calling answered it"
_CALLED_TOOLS = (
    "Native function calling drove real tool calls before the backend failed, "
    "so the protocol works even though this query got no answer"
)
#: Native drove tool calls, nothing failed, and the summarising call still came
#: back empty. Distinct from :data:`_CALLED_TOOLS`, which says "before the
#: backend failed" -- no backend failure happened on this path.
_CALLED_TOOLS_NO_SYNTHESIS = (
    "Native function calling drove real tool calls and then returned no text "
    "to summarise them, so the protocol works even though this query got no "
    "answer"
)
#: The same outcome reached without executing anything: every structured call
#: repeated one the text protocol had already run, so the duplicate guard
#: answered them from the recorded results. Kept apart from
#: :data:`_CALLED_TOOLS_NO_SYNTHESIS` because "all of them repeats" is a claim
#: about what native did, and it is false the moment one call was fresh --
#: which is the ordinary shape of the empty-synthesis path, not a rare one.
_CALLED_TOOLS_QUIET = (
    "Native function calling emitted structured tool calls -- all of them "
    "repeats of calls the text protocol had already run -- and then returned "
    "no text, so the protocol works even though this query got no answer"
)


def _commit_escalation(signature: str, evidence: str = _ANSWERED) -> None:
    """Move the rest of the process to native, once native has proved it works.

    Committed on evidence rather than on the decision to try, because an
    endpoint that cannot serve the protocol must not be able to select it: a
    switch thrown up front sent every later query into the same refusal, and
    the refusal was the thing being recovered from.

    *evidence* is what the caller observed -- :data:`_ANSWERED` or
    :data:`_CALLED_TOOLS`. It exists because this used to hard-code "answered
    it" and one of the two callers reaches here after the request *raised*, so
    the announcement was false on exactly the path where an operator reading
    the log most needs it to be true.

    Announced once. Later queries reach native through
    :func:`_resolved_tool_mode` and are then ordinary native queries -- a
    provider failure on one of those surfaces the way it does for a user who
    configured native themselves, because by then the protocol is known to work
    and the failure is not about the protocol.
    """
    if _escalation.escalated:
        return
    _escalation.escalated = True
    logger.warning(
        # "A round", not "the model": escalation is triggered by the round that
        # gave up, and earlier rounds of the same query may well have run tools
        # -- which is why the retry is seeded with their results at all. Phrased
        # as a claim about the query, this line contradicted the tool trace
        # recorded on the very same turn.
        "A round returned no text and no tool call (%s), which is how a "
        "reasoning model behaves on the 'text' tool protocol; any tools the "
        "earlier rounds ran are on the turn's trace. %s, so this process will "
        "use native from here. Set AORTA_CHAT_LLM_TOOL_MODE to choose the "
        "protocol yourself; this process started on the 'text' protocol.",
        signature,
        evidence,
    )


def _begin_escalated_probe() -> int:
    """Claim an identity for one escalated probe, before it is sent.

    Read before the request so the number says *when the probe began*, which is
    the fact :func:`_record_escalation_failure` needs. Reading it afterwards
    would make every failure look sequential.
    """
    _escalation.probes_begun += 1
    return _escalation.probes_begun


def _record_escalation_failure(probe: int, *, endpoint_hint: bool = False) -> str:
    """Count an escalated attempt that did not answer; return what follows next.

    Shared by the two ways the retry can fail to rescue a query -- the request
    raised, or it came back as silent as the text round before it. Both spend
    the same budget, and both leave the protocol where it was, so counting them
    in one place is what stops a caller from spending it forever by forgetting.

    Returns the **whole** clause, not a suffix, and that is the point. Each
    caller used to build its own sentence around this and so had to restate
    what had happened to the protocol and to the budget. The defect that
    started this was one of those sentences saying the ``text`` protocol stayed
    in force after a concurrent probe had already moved it -- a call site
    asserting something no longer true, which is the exact failure this
    function exists to prevent. Three sites reciting one fact cannot be kept
    honest by review; computing it here means there is nothing to keep in step.

    ``endpoint_hint`` asks for the local-vLLM tool-parser advice, which suits a
    caller whose request *failed* and not one whose model merely went quiet. It
    is a request rather than a decision: the pairing rule -- that the advice is
    wrong once native is known to work -- lives below with the state it depends
    on, because that rule is itself a claim that could drift.

    Counts *waves*, not attempts. The budget is two so that one bad minute does
    not disable the escalation for the process -- a permanent refusal and a 503
    look identical on a single failure, so the second attempt is what tells
    them apart. Two Chainlit sessions failing inside the same outage is not
    that second attempt: nothing was probed in between, so it carries no
    information the first failure did not. Counting it anyway spent the whole
    budget on one outage, and it was measured doing exactly that -- two
    concurrent sessions, one transient 503, and every later query in the
    process denied its retry.

    A failure also stops being evidence once native has been committed at all
    -- not merely once a probe in the same wave has. In that case the process
    is already on native, so incrementing the budget would write off a proven
    protocol and the caller would log the opposite of the state later queries
    now observe. Scoping this to the wave was too narrow: a request that
    resolved ``text`` before the commit probes *after* it, and a failure on
    that probe reported "the 'text' protocol stays in force" while
    :func:`_resolved_tool_mode` returned native, over a count that had run past
    its own cap.

    So a failure counts only if its probe began *after* the last counted one
    was recorded. The watermark moves to :attr:`_EscalationState.probes_begun`
    rather than to this probe's own number, which is what absorbs the rest of
    the wave: every probe already in flight is at or below it.

    No lock, deliberately. Serialising the probe would make one session's
    provider call block another's, which is the event-loop stall issue #444 is
    about -- and this runs on the recovery path, where a queue behind a timing
    out request is the worst place to put one. Coalescing needs no mutual
    exclusion: these are plain integer reads and writes between ``await``
    points on one event loop.
    """
    if _escalation.escalated:
        # No endpoint hint on this branch even when the caller asks for one:
        # native demonstrably works here, so tool-parser advice would send the
        # operator to fix something that is not broken.
        return (
            "Another retry has already proved native works on this "
            "endpoint, so the protocol has moved to native regardless and this "
            "failure is not counted against the budget."
        )
    if probe > _escalation.counted_watermark:
        _escalation.native_failures += 1
        _escalation.counted_watermark = _escalation.probes_begun
    budget = (
        " native will not be tried again in this process."
        if _escalation.native_failures >= _MAX_NATIVE_FAILURES
        else " a later query may try again."
    )
    return (
        f"The 'text' protocol stays in force. This is attempt "
        f"{_escalation.native_failures} of {_MAX_NATIVE_FAILURES};{budget}"
        f"{_ENDPOINT_HINT if endpoint_hint else ''}"
    )


@dataclass(frozen=True)
class _EscalateToNative:
    """Sentinel: the text loop gave up and the protocol, not the query, is why.

    Returned in place of a state update so that ``act_node`` owns the retry --
    the escalation is then visible at the dispatch point rather than buried
    inside one protocol's implementation, and ``_act_text`` keeps its single
    job of driving one protocol.

    It carries the trace the abandoned loop accumulated because the label on a
    fallback answer depends on it: if the native retry cannot run at all, the
    query still has to land on :func:`_abandoned_result`, and a loop that ran a
    tool before going quiet must not be told there that it used none. It
    carries the observed signature for the same reason -- the announcement now
    happens after the retry, and the evidence for it was seen before.

    ``calls`` is the same tool history as ``trace``, structured rather than
    rendered, so the retry can seed its duplicate guard and show the model what
    already came back. ``trace`` cannot serve that: it is display text the
    critic scans line-by-line, and re-parsing it to recover the arguments would
    be a second parser to keep in step with the first.
    """

    trace: tuple[str, ...] = ()
    signature: str = ""
    calls: tuple[tuple[str, dict, str], ...] = ()

#: Goes into the answer slot, so it names nothing internal: no environment
#: variable, neither tool protocol, and no class of model. The user asked a
#: question and must not get a configuration lecture back. Everything specific
#: is still recorded on the log lines beside the abandon branch, including the
#: tool protocol, which the ``LLM backend`` line names at startup. ``aorta chat
#: doctor`` is named because it is where an operator looks first and it covers
#: the configuration faults that reach this message by *some* of its routes --
#: an unreachable backend, a missing index.
#:
#: It does not assert one, though, and used to: "something in my own
#: configuration is stopping me" was wrong on most of the paths that reach
#: here. A model that returns empty content on both protocols is not a
#: misconfiguration, and neither is a backend that falls over mid-loop -- the
#: troubleshooting table in ``docs/chat/providers.md`` says of one of them, in
#: as many words, that nothing there is a configuration fault. Naming a cause
#: the code has not established sends the operator to `doctor` to audit
#: settings that are all correct. So the text says what is true on every path
#: (no answer, and the reason is recorded next to it) and points at both places
#: the specifics actually live.
_NO_ANSWER_MSG = (
    "I wasn't able to answer that -- check the warning logged with this reply "
    "for the specific reason, or run `aorta chat doctor` to rule out a "
    "configuration problem."
)

#: Prefixed to a fallback answer, and the labelling is not decoration. Silently
#: answering from retrieval when the user's question needed tools is exactly how
#: https://github.com/ROCm/aorta/issues/433 misled people, and a fallback that
#: did it unlabelled would trade one defect for its mirror image. Says what was
#: not done and what that costs, in the user's terms.
_DEGRADED_ANSWER_PREFIX = (
    "I could not use my tools for this question, so what follows comes only "
    "from the documentation and run records I already have indexed. Anything "
    "that needed me to go and look is missing from it."
)


async def _fallback_retrieval_answer(state: AgentState) -> str:
    """One tool-free answer attempt after the act loop abandoned, or "".

    The reporter proved this works before it was written: the same information
    need, asked twice a minute apart against one model and one index, cost 4
    billed calls and returned nothing through the ``action`` route and 2 calls
    and a correct answer through ``question``. Only the classification differed
    -- the index, the retrieval and the model were all fine, and the ``question``
    route never touches a tool protocol, so the empty-content behaviour that
    kills the act loop does not apply to it.

    This is the same single call that route makes, on the context ``retrieve``
    already gathered, so it adds one call and no retrieval work. It is also
    independent of the auto-escalation above and worth having alongside it: it
    covers a provider hiccup on one protocol, an endpoint that refuses the
    other, and any future model that can drive neither.

    It does *not* cover a tool outage, which the surrounding docs used to claim.
    A failing tool still appends its error to the trace, and
    :func:`_abandoned_result` sends any non-empty trace to the plain notice --
    because "I could not use my tools" is false once a tool has run, and an
    untrue label is what this fallback exists to avoid.

    The trailing user turn this needs is :data:`_FALLBACK_RETRY_NUDGE`, not the
    default: this request carries no tools, so the default's "ground every claim
    in output you obtained from a tool in this turn" would be an instruction it
    makes impossible to follow.

    Returns "" when the model produces nothing *or when the call fails*, so the
    caller still reports the dead end rather than an empty answer or an error.

    That second half is the same rule as the escalated native retry in
    :func:`_escalated_native_attempt`, and for the same reason: this is an
    extra call the user did not ask for, added to a path that previously made
    none, so the worst it may do is fail to improve the notice they were
    already getting. The likeliest reason the act loop dead-ended is a backend
    that is unwell, which is exactly when this call is most likely to raise --
    letting it through would turn the give-up message into a traceback on the
    query that needed the message most. Broad for the same reason as there:
    every provider client spells its failures differently, and the one not
    enumerated is the one that reaches the user.
    """
    # Building the model is inside the guard, not before it. ``_get_llm``
    # resolves the backend and can fail for the same reasons the call can, and
    # the docstring above promises "" for a failure rather than for one of two
    # failures.
    try:
        llm = _get_llm(temperature=0.1, streaming=False)
        messages = [
            _build_answer_message(state.get("retrieved_context", "")),
            *state["messages"],
        ]
        _ensure_ends_with_user(messages, _FALLBACK_RETRY_NUDGE)
        response = await _send(llm, messages)
    except Exception as exc:  # last-resort extra call; see the docstring
        logger.warning(
            "The fallback answer from retrieved context failed too (%s: %s), so "
            "this query has no answer to give.",
            type(exc).__name__,
            exc,
        )
        return ""
    text = str(response.content or "").strip()
    if not text:
        _log_empty_content(response, "act_node retrieval fallback")
    return text


async def _abandoned_result(state: AgentState, trace: list[str]) -> dict[str, Any]:
    """What the act loop returns once it has given up: a fallback, or the notice.

    ``command_output`` is deliberately left empty even when the fallback
    answered. It is what ``critic_node`` judges, and an empty value makes the
    critic return no feedback, which sends the graph to ``END`` -- so the
    fallback cannot be rejected into a retry that re-enters the act loop, which
    is what keeps :data:`_MAX_UNPRODUCTIVE_ROUNDS` meaningful. It is also
    honest: no command was run and no tool output exists for a critic to check
    the answer against.

    One attempt per entry to the act loop, and on the ordinary path the empty
    ``command_output`` means there is only ever one entry.

    A loop that ran a tool before going quiet gets the plain notice instead. The
    label would be false there -- tools *did* run -- and an inaccurate label is
    the thing this fallback is careful about in the first place.
    """
    if trace:
        return {
            "messages": [AIMessage(content=_NO_ANSWER_MSG)],
            "command_output": "",
            "tool_trace": trace,
        }
    answer = await _fallback_retrieval_answer(state)
    if answer:
        logger.info(
            "Answered from retrieved context after the act loop abandoned. The "
            "answer is labelled as degraded: no tool ran for this query."
        )
        answer = f"{_DEGRADED_ANSWER_PREFIX}\n\n{answer}"
    return {
        "messages": [AIMessage(content=answer or _NO_ANSWER_MSG)],
        "command_output": "",
        "tool_trace": trace,
    }


#: Sent with the final synthesis call, which runs without tools bound. Offered
#: tools, a model that has not yet found what it wants keeps calling them and
#: returns no prose, so a loop that gathered plenty still answered nothing.
#:
#: This call only happens when the round budget ran out, so the model is often
#: mid-task. Saying "you have gathered enough information" invited it to carry
#: on narrating -- one run ended with the whole answer being "Installed. Now let
#: me build the HIP binary and confirm the CLI works." Naming the exhaustion and
#: asking for what is unfinished gets a self-contained reply instead.
_FINAL_ANSWER_MSG = (
    "Your tool budget for this turn is now exhausted -- no further tools are "
    "available, and there will be no further turns. Write your complete final "
    "answer now, as prose the user will read on its own, with no reference to "
    "continuing. Include: what you established, citing the file paths and "
    "command output you saw; the answer to the question as far as you can give "
    "it; and, if you were part-way through something, exactly what remains and "
    "the commands the user should run to finish it. Do not promise further "
    "work, and do not ask to continue."
)


# ──────────────────── Act ────────────────────────


async def act_node(state: AgentState) -> dict[str, Any]:
    """Tool-using loop, in whichever protocol ``llm_tool_mode`` selects.

    Also the one place the protocol can change: the text loop reports back that
    the model cannot drive it, and the retry in the native protocol is issued
    from here rather than from inside the loop that gave up.
    """
    mode = _resolved_tool_mode()
    if mode == "native":
        return await _act_native(state)
    if mode == "text":
        result = await _act_text(state)
        if isinstance(result, _EscalateToNative):
            return await _escalated_native_attempt(
                state, list(result.trace), result.signature, list(result.calls)
            )
        return result
    raise ValueError(
        f"unknown llm_tool_mode: {settings.llm_tool_mode!r} "
        "(expected one of native, text)"
    )


async def _escalated_native_attempt(
    state: AgentState,
    trace: list[str],
    signature: str,
    calls: list[tuple[str, dict, str]] | None = None,
) -> dict[str, Any]:
    """The native retry, and what happens when the endpoint will not serve it.

    The retry is speculative: the user asked a question, not for a protocol
    change, so it must not be able to leave them worse off than the dead end it
    is trying to rescue. A stock local vLLM started without
    ``--enable-auto-tool-choice`` and a matching ``--tool-call-parser`` refuses
    any request carrying ``tools``, and that user is precisely the one the
    escalation targets -- they never set
    ``llm_tool_mode``, so nothing marks their endpoint as text-only until a
    native request comes back refused. Unhandled, that refusal turned a poor
    answer into a traceback out of the graph. Caught, the query lands on the
    same fallback it would have had if the escalation had never fired.

    The catch is broad on purpose. Which exception a refused ``tools`` payload
    raises depends on the provider client, the gateway and the LangChain
    wrapper between them -- a ``BadRequestError``, an ``APIError``, an httpx
    status error -- and enumerating them means the one that was missed is the
    one that reaches the user as a traceback, which is the whole failure being
    fixed. ``Exception`` leaves ``CancelledError`` and ``KeyboardInterrupt``
    fatal, so an interrupted query is still interrupted.

    Being unable to name the exception is also why the failure is *counted*
    rather than read. The same broad catch sees a permanent refusal and a
    timeout, and the log line says so instead of diagnosing on the caller's
    behalf; :data:`_MAX_NATIVE_FAILURES` is what tells them apart, by whether
    the failure repeats.

    There is one thing the catch *can* read, and it changes the verdict:
    whether the model emitted structured ``tool_calls`` before the failure,
    which :class:`_NativeLoopError` carries out with the trace. A failure with a
    tool call behind it is not a failure of the protocol -- native demonstrably
    worked and the backend then fell over -- so it commits the switch and
    spends none of the budget, which exists to write off an endpoint native is
    not *served* on. Without that split, two transient 503s after working tool
    calls would strand the process on ``text`` for good. A refused ``tools``
    payload raises on the first request, before any tool call, so the shape the
    budget is aimed at still lands in it.

    A request that comes back *silent* is the second way the retry fails, and it
    is the reason this reads :attr:`_NativeOutcome.answered` rather than simply
    committing on a return. Every exit from the native loop is a state update,
    including the two that gave up, so "did not raise" says nothing about
    whether the protocol worked -- and committing on it handed the rest of the
    process to a protocol that had answered nothing. It counts against the same
    budget as a raise: nothing was learned about the endpoint, and a model that
    is silent on both protocols must not be billed for a native round on every
    query from here on.
    """
    # Claimed before the request, so a failure knows whether it began before or
    # after the last counted one -- see `_record_escalation_failure`.
    probe = _begin_escalated_probe()
    try:
        outcome = await _run_native_loop(
            state, escalated=True, prior_trace=trace, prior_calls=calls
        )
    except _NativeLoopError as failure:
        # Everything the native loop achieved before it broke, in front of what
        # the text loop had achieved before it gave up. Both belong to the same
        # query, and `_abandoned_result` reads the pair to decide whether "I
        # could not use my tools" is a true thing to tell this user.
        whole_trace = [*trace, *failure.trace]
        if failure.tool_called:
            # Native drove structured `tool_calls` and *then* the backend fell
            # over. The protocol is not what failed, so this must not spend the
            # budget that exists to write off an endpoint native is not served
            # on -- two transient 503s after a working tool call would
            # otherwise strand the process on `text` for good. Committed for
            # the same reason `_NativeOutcome.answered` counts a tool call as
            # proof: the question the escalation asks has been answered yes.
            _commit_escalation(signature, _CALLED_TOOLS)
            logger.warning(
                "The escalated native tool-calling request drove structured "
                "tool calls and then failed (%s: %s). Structured tool calling "
                "works on this endpoint, so the protocol moves to native "
                "anyway and this does not count against the %d-failure budget. "
                "%s, but the call that would have turned them into an answer is "
                "the one that failed, so this query still has no answer to "
                "give. Retrying the question will now go straight to native.",
                type(failure.cause).__name__,
                failure.cause,
                _MAX_NATIVE_FAILURES,
                # Not `len(failure.trace)`: that counts results newly executed
                # by this loop, and a call the duplicate guard answered from the
                # text protocol's recorded result appends none. Reporting it
                # would have logged "drove 0 tool call(s)" one clause before
                # committing native because a call proved the protocol works.
                f"The {len(whole_trace)} tool result(s) gathered for this "
                f"question are recorded on the turn"
                if whole_trace
                else "Every call repeated one already answered above",
            )
            return await _abandoned_result(state, whole_trace)
        logger.warning(
            "The escalated native tool-calling request failed (%s: %s) without "
            "making a tool call. Answering from retrieved context instead. %s",
            type(failure.cause).__name__,
            failure.cause,
            _record_escalation_failure(probe, endpoint_hint=True),
        )
        return await _abandoned_result(state, whole_trace)
    except Exception as exc:
        # Still broad, and for the reason the docstring gives: only the calls
        # inside the loop are wrapped as `_NativeLoopError`, and the loop also
        # resolves the backend and binds the tool schemas before it makes any.
        # Those raise plainly, and letting them through would put the traceback
        # back on the query this whole path exists to keep an answer on.
        logger.warning(
            "The escalated native tool-calling request failed before it could "
            "call anything (%s: %s). Answering from retrieved context "
            "instead. %s",
            type(exc).__name__,
            exc,
            _record_escalation_failure(probe, endpoint_hint=True),
        )
        return await _abandoned_result(state, trace)
    if not outcome.answered:
        # The retry ran and the model was as silent on native as it had been on
        # text. Committing here is what the switch has to refuse to do: a
        # protocol that has never answered would then be selected for every
        # later query on the strength of an attempt that failed.
        logger.warning(
            "The escalated native tool-calling request returned no answer and "
            "no tool call either. %s",
            _record_escalation_failure(probe),
        )
        return outcome.result
    _commit_escalation(signature, outcome.evidence)
    return outcome.result


def _act_messages(state: AgentState) -> list[Any]:
    """System framing shared by both protocols."""
    context = state.get("retrieved_context", "")
    plan = state.get("plan", "")
    critic_fb = state.get("critic_feedback", "")

    messages: list[Any] = [_build_system_message(context)]
    if plan:
        messages.append(SystemMessage(content=f"PLAN:\n{plan}"))
    if critic_fb:
        messages.append(
            SystemMessage(
                content=f"PREVIOUS COMMAND FAILED:\n{critic_fb}\n"
                "Analyze the error and retry with a corrected command."
            )
        )
    return messages


def _last_human(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


_RETRY_NUDGE = (
    "Your previous answer was rejected by the validation step. Revise it, and "
    "ground every claim in output you obtained from a tool in this turn."
)

#: The same trailing-user-turn job as :data:`_RETRY_NUDGE`, for the one caller
#: that has no tools to ground anything in. The retrieval fallback runs on the
#: unbound model under a prompt that forbids tool use, so telling it to ground
#: every claim "in output you obtained from a tool in this turn" set it a task
#: the request it arrived in makes impossible -- and it landed exactly on the
#: critic-retry path, where a rejected answer is the last thing in state and so
#: the nudge is guaranteed to be appended. Names the source that *is* available.
_FALLBACK_RETRY_NUDGE = (
    "Your previous answer was rejected by the validation step, and no tools are "
    "available for this attempt. Answer from the documentation and run records "
    "quoted above, and say plainly which parts of the question they do not cover."
)


def _ensure_ends_with_user(messages: list[Any], nudge: str = _RETRY_NUDGE) -> None:
    """Append a user turn when the conversation ends with an assistant one.

    Anthropic treats a trailing assistant message as a prefill to continue, and
    models that disallow prefill reject the request outright: *"This model does
    not support assistant message prefill. The conversation must end with a user
    message."* That is exactly the shape a critic-triggered retry produces, since
    the rejected answer is the last thing in state.

    *nudge* is what that turn says. It is a parameter because the turn is not
    only padding -- it is an instruction the model will follow -- so a caller
    that cannot honour the default must pass one it can; see
    :data:`_FALLBACK_RETRY_NUDGE`.
    """
    if messages and isinstance(messages[-1], AIMessage):
        messages.append(HumanMessage(content=nudge))


@dataclass(frozen=True)
class _NativeOutcome:
    """A native loop's state update, and whether the protocol actually answered.

    The second field exists because the first cannot carry it. Every way out of
    the loop returns a state update, including the ones that gave up, so
    "returned without raising" is not the same as "native works" -- and
    :func:`_commit_escalation` needs the second question answered, not the
    first. Reading it back off the result was tried and does not work either:
    :func:`_abandoned_result` blanks ``command_output`` by design, and the
    final-synthesis path substitutes :data:`_NO_ANSWER_MSG` for empty text, so
    both a rescued query and an abandoned one can present the same shape.
    """

    result: dict[str, Any]
    #: Whether native demonstrably drove the protocol: it returned prose, or it
    #: made at least one tool call. False only when the loop gave up having seen
    #: neither -- which is the same model behaviour that started the escalation,
    #: now observed on the protocol that was supposed to fix it.
    answered: bool
    #: Which of those two it was, for the announcement to quote. Kept beside the
    #: flag rather than re-derived at the commit site, because by then the
    #: distinction is gone: a loop that answered and one whose calls were all
    #: duplicates both arrive as ``answered=True``, and announcing "answered it"
    #: for the second tells the operator the query succeeded when it returned
    #: the give-up notice. Ignored when :attr:`answered` is false.
    evidence: str = _ANSWERED


class _NativeLoopError(Exception):
    """A backend failure inside the native loop, plus what the loop had achieved.

    The loop's progress lives in locals, so an exception used to discard it: a
    query whose first native round drove a real tool call and whose *second*
    round hit a 503 came back through the caller's ``except`` with the tool
    result gone -- which meant the plain "I could not use my tools" notice on a
    query where one had, and a native failure counted against
    :data:`_MAX_NATIVE_FAILURES` on an endpoint that had just demonstrated the
    protocol works. Carrying both facts out with the exception is what lets the
    caller tell "native is not served here" from "native worked and the backend
    then fell over", which are the two things that budget exists to separate.
    """

    def __init__(self, cause: Exception, trace: list[str], tool_called: bool):
        super().__init__(str(cause))
        #: The provider/gateway error, for the caller's log line.
        self.cause = cause
        #: Tool results gathered before the failure, in loop order.
        self.trace = list(trace)
        #: Whether the model emitted structured ``tool_calls`` at least once.
        #: The same "the protocol works" test :attr:`_NativeOutcome.answered`
        #: applies, and deliberately not "a tool succeeded" -- a tool that
        #: errored still proves the model drove ``tools``, which is the only
        #: question the escalation is asking.
        self.tool_called = tool_called


async def _act_native(state: AgentState, escalated: bool = False) -> dict[str, Any]:
    """Tool loop over the OpenAI function-calling API.

    What reasoning models expect. The provider returns structured ``tool_calls``
    and ``finish_reason=tool_calls``, so nothing depends on the model reproducing
    a text syntax, and there is no parsing to fail.

    *escalated* marks the retry :func:`act_node` issues after the text loop
    reported that the model cannot drive it. It buys one round rather than two
    (:data:`_MAX_ESCALATED_ROUNDS`), because by then the query has already been
    paid for twice.
    """
    return (await _run_native_loop(state, escalated=escalated)).result


def _call_signature(name: str, kwargs: dict | None) -> str:
    """Identity of a tool call, for the duplicate guard.

    One function because the guard now compares calls made through *different*
    protocols -- the text loop's parsed kwargs against the native loop's
    structured ``args`` -- and two separately-written format strings that have
    to agree is how the seeding silently stops matching.

    Normalised through :func:`_normalise_tool_name` for the same reason, and
    this is the part that has teeth. :func:`_execute_tool` dispatches on the
    normalised name, so ``list_files<|channel|>commentary`` *runs*
    ``list_files``; a signature built from the raw name would call that a
    different call and let the duplicate through. The two functions have to
    decide identity the same way, and the population this matters for is
    exactly the one that reaches the escalation: gpt-oss behind a serving stack
    whose harmony parser leaks the channel marker is both why
    :func:`_normalise_tool_name` exists and why the retry runs at all. It also
    covers the within-loop case, where two channel markers on one tool would
    otherwise be two calls.
    """
    return f"{_normalise_tool_name(name)}({sorted((kwargs or {}).items())})"


def _prior_results_msg(prior_calls: list[tuple[str, dict, str]]) -> str:
    """Hand the escalated retry what the text protocol already ran.

    Without this the retry starts from a bare prompt, so a model that wants the
    same tool call the text round already made gets it *executed a second time*
    -- measured: one query, ``list_files(path="src")`` run twice, and the same
    result recorded twice in the trace handed to the critic. With the shell tool
    enabled (``enable_shell_tool``) the repeated call is
    ``run_terminal_command``, so this is a duplicated side effect and not merely
    a duplicated bill.

    Results are passed as text rather than as reconstructed
    ``AIMessage(tool_calls=...)``/``ToolMessage`` pairs. The text protocol never
    had structured call IDs to reconstruct from, and inventing them risks a
    provider rejecting the request for an assistant turn whose ``tool_calls`` do
    not match the ``ToolMessage`` ids that follow -- which would turn a recovery
    path into a new failure. The model does not need to believe *it* made these
    calls; it needs to know they were made and what came back.
    """
    lines = [
        "Tool results already gathered for this question, before switching to "
        "function calling. Do not repeat these calls -- use the results:",
        "",
    ]
    for name, kwargs, result in prior_calls:
        lines.append(f"{name}({kwargs}) returned:")
        lines.append(result)
        lines.append("")
    lines.append(
        "Call a different tool if you still need something, or answer with "
        "what is here."
    )
    return "\n".join(lines)


async def _run_native_loop(
    state: AgentState,
    escalated: bool = False,
    prior_trace: list[str] | None = None,
    prior_calls: list[tuple[str, dict, str]] | None = None,
) -> _NativeOutcome:
    """:func:`_act_native`, plus the answer to "did native work?".

    Split out so the escalated retry can tell a rescue from a second dead end.
    Ordinary native queries go through the wrapper and discard the extra fact.

    *prior_trace* is what already ran before this loop started -- the text
    protocol's tool results, when this is the escalated retry. It is kept as a
    separate variable rather than seeding ``trace``, because the two answer
    different questions. The loop's own ``trace`` decides whether *this*
    protocol gathered anything worth synthesising, and whether it may be said to
    have driven the protocol; the two together describe the *query*, which is
    what ``tool_trace`` reports and what the "no tool ran" label is about.
    :func:`whole_trace` is the merge, and every reporting exit uses it; seeding
    would conflate the two and buy a synthesis call off the back of a tool the
    other protocol ran.

    *prior_calls* is the same history in structured form -- ``(name, kwargs,
    result)`` -- and does two things ``prior_trace`` cannot, because it is
    display text the critic scans rather than data. It seeds ``seen``, so a
    repeat of a call the text protocol already made is refused by the duplicate
    guard rather than executed again, and it is rendered into the prompt by
    :func:`_prior_results_msg` so the model can use the result instead of
    needing to ask for it. Both matter: seeding alone would tell the model "you
    already asked that" about a result it had never been shown.
    """
    plain = _get_llm(temperature=0.1, streaming=False)
    llm = plain.bind_tools(list(TOOL_REGISTRY.values()))

    messages = _act_messages(state)
    is_search = _is_search_query(_last_human(state))
    if is_search:
        messages.append(SystemMessage(content=_SEARCH_FORCE_MSG))
    messages.extend(state["messages"])
    if prior_calls:
        messages.append(HumanMessage(content=_prior_results_msg(prior_calls)))
    _ensure_ends_with_user(messages)

    max_rounds = (
        settings.max_act_rounds_search if is_search else settings.max_act_rounds
    )
    unproductive_cap = _MAX_ESCALATED_ROUNDS if escalated else _MAX_UNPRODUCTIVE_ROUNDS
    unproductive = 0
    # Seeded with what the other protocol ran, so the duplicate guard below
    # covers the whole query rather than only this loop's own calls.
    seen: set[str] = {
        _call_signature(name, kwargs) for name, kwargs, _result in (prior_calls or [])
    }
    trace: list[str] = []
    # Whether the model ever emitted structured `tool_calls`, which is the
    # question "does this endpoint serve native?" reduces to. Tracked separately
    # from `trace` because it has to survive an exception -- see
    # `_NativeLoopError` -- and because the two can diverge: a round whose
    # every call was a duplicate of one already made appends nothing.
    tool_called = False
    # Distinguishes the two ways out of the loop below. Falling out of the range
    # means the model was still working when the round budget ran out, which is
    # what the final synthesis call is for; giving up on empty rounds means it
    # produced nothing, and asking a model that has said nothing twice for a
    # summary of nothing is one more billed call for the same result.
    gave_up = False

    def whole_trace() -> list[str]:
        """Every tool result this *query* has, not just this protocol's.

        What goes into ``tool_trace``, on every way out of this loop. The state
        update is read by ``critic_node`` and carried into the next turn, and it
        describes the query -- so a tool the text protocol ran before the dead
        end belongs in it even when native is the protocol that answered.
        Distinct from the bare ``trace`` on purpose: that one decides whether
        *this* protocol gathered enough to be worth a synthesis call, and must
        not be able to buy one off the back of the other protocol's work.
        """
        return [*(prior_trace or []), *trace]

    async def send(model: Any) -> Any:
        """:func:`_send`, with the loop's progress attached to a failure.

        Every outbound call in this loop goes through here so that a backend
        that falls over mid-loop cannot take the tool results already gathered
        down with it. Reads ``trace`` and ``tool_called`` as free variables, so
        it reports whatever had been achieved at the moment of the failure.
        """
        try:
            return await _send(model, messages)
        except Exception as exc:
            raise _NativeLoopError(exc, trace, tool_called) from exc

    for round_num in range(max_rounds):
        response = await send(llm)
        tool_calls = getattr(response, "tool_calls", None) or []
        text = str(response.content or "").strip()

        if not tool_calls:
            if text:
                return _NativeOutcome(
                    result={
                        "messages": [AIMessage(content=text)],
                        "command_output": text,
                        "tool_trace": whole_trace(),
                    },
                    answered=True,
                )
            unproductive += 1
            _log_empty_content(response, f"act_node round {round_num + 1}")
            if unproductive >= unproductive_cap:
                gave_up = True
                break
            messages.append(HumanMessage(content=_SEARCH_REPROMPT_NATIVE_MSG))
            continue

        unproductive = 0
        tool_called = True
        messages.append(response)
        for call in tool_calls:
            signature = _call_signature(call["name"], call["args"])
            logger.info(
                "Act round %d: %s(%s)", round_num + 1, call["name"], call["args"]
            )
            if signature in seen:
                # Models loop on an identical query when a result disappoints.
                # Each repeat is a billed round that cannot teach it anything
                # new, so say so instead of running the tool again.
                messages.append(
                    ToolMessage(
                        content=(
                            "This exact call was already made and returned the "
                            "result above. Try different arguments or a different "
                            "tool, or answer with what you have."
                        ),
                        tool_call_id=call["id"],
                    )
                )
                continue
            seen.add(signature)
            result = await _execute_tool(call["name"], call["args"])
            trace.append(f"{_TOOL_RESULT_PREFIX}{call['name']}:\n{result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # A loop that gathered results before going quiet still has material, so it
    # keeps the final synthesis call below; one that produced nothing at all has
    # nothing to synthesise from and would only be billed for saying so again.
    if gave_up and not trace:
        logger.warning(
            "Act loop abandoned after %d round(s) in native mode with no text "
            "and no %s.%s",
            unproductive,
            "tool result" if tool_called else "tool call",
            " Escalating from the text protocol did not help, so the model is "
            "returning nothing on either."
            if escalated and not tool_called
            else (
                " Structured tool calling worked -- every call repeated one the "
                "text protocol had already made, so the duplicate guard "
                "answered them from the recorded results."
                if tool_called
                else ""
            ),
        )
        # `trace` is provably empty on this branch, so this is `prior_trace`.
        # Passing it matters: a text loop that ran a tool before dead-ending
        # must not have its query labelled "I could not use my tools", which is
        # what an empty trace tells `_abandoned_result` to do.
        return _NativeOutcome(
            result=await _abandoned_result(state, whole_trace()),
            # Not `False`: an empty `trace` stopped meaning "made no tool call"
            # once the duplicate guard began answering repeats from the recorded
            # result instead of re-running them. A native round whose every call
            # duplicated a text-protocol call emits structured `tool_calls`,
            # appends nothing, and had proved the protocol works -- reporting it
            # as a failed probe spent the failure budget on a working endpoint
            # and could disable escalation for the process.
            answered=tool_called,
            evidence=_CALLED_TOOLS_QUIET,
        )

    # Reaching here means the loop never produced a tool-free reply, and there
    # are two ways that happens. Both get the synthesis call below, and they are
    # logged apart because the operator reading the line is diagnosing one or
    # the other: a budget that wants raising, or a model that went quiet.
    if gave_up:
        # `gave_up` with a non-empty trace: the model stopped answering after
        # gathering results, so the budget was *not* exhausted and it was not
        # still calling tools. Saying it was sent the reader to MAX_ACT_ROUNDS,
        # which is the one knob that would not have helped.
        logger.warning(
            "Act loop gave up after %d empty round(s) in native mode with "
            "%d tool result(s) already gathered; asking for a final answer from "
            "those. The round budget was not the limit -- the model stopped "
            "producing tool calls and text.",
            unproductive,
            len(trace),
        )
    else:
        # The budget really did run out mid-task. Say so: the answer will read
        # as truncated, and the cause is a knob the user can turn.
        logger.warning(
            "Act loop hit its %d-round budget with the model still calling "
            "tools; asking for a final answer now. Raise MAX_ACT_ROUNDS%s if "
            "this query needs more steps.",
            max_rounds,
            "_SEARCH" if is_search else "",
        )
    # Final synthesis runs on the *unbound* model: offered tools, the model
    # keeps calling them and returns no prose, which is how a completed loop
    # still ended in an empty answer.
    #
    # The instruction goes in as a *user* turn, not a system one. LiteLLM hoists
    # every system message into Anthropic's `system` parameter, so appending one
    # here never made it the last thing the model saw -- it merged into the
    # system prompt while the conversation still ended on tool results, and the
    # model carried on working. As a user turn it is positionally last.
    messages.append(HumanMessage(content=_FINAL_ANSWER_MSG))
    final = await send(plain)
    text = str(final.content or "").strip()
    # Read before the substitution below, which would otherwise make an empty
    # synthesis indistinguishable from a real answer to the caller.
    synthesised = bool(text)
    if not text:
        _log_empty_content(final, "act_node final")
        text = _NO_ANSWER_MSG
    return _NativeOutcome(
        result={
            "messages": [AIMessage(content=text)],
            "command_output": text,
            "tool_trace": whole_trace(),
        },
        # A tool call is proof the protocol works even when the synthesis that
        # followed it came back empty: the model drove `tools` successfully, and
        # what failed after that is not the protocol.
        #
        # `tool_called` rather than `bool(trace)`, because those stopped being
        # the same question once the duplicate guard started answering repeats
        # from the recorded result: a round whose every call duplicated a
        # text-protocol call emits structured `tool_calls` and appends nothing.
        # It is still this loop's own flag, set only inside it, so the property
        # `trace` was chosen for holds -- the other protocol's work cannot
        # answer for this one.
        answered=tool_called or synthesised,
        # Three outcomes reach here, and the announcement must not merge them:
        # prose, fresh tool results the synthesis call failed to summarise, and
        # a round whose every call the duplicate guard answered. `trace` is what
        # separates the last two -- it is empty only when nothing new ran.
        evidence=(
            _ANSWERED
            if synthesised
            else (_CALLED_TOOLS_NO_SYNTHESIS if trace else _CALLED_TOOLS_QUIET)
        ),
    )


async def _act_text(state: AgentState) -> dict[str, Any] | _EscalateToNative:
    """ReAct-style loop: LLM outputs ACTION lines, we execute and feed back.

    Returns an :class:`_EscalateToNative` instead of a state update when the model
    cannot drive this protocol at all; :func:`act_node` turns that into the
    retry.
    """
    llm = _get_llm(temperature=0.1, streaming=False)

    context = state.get("retrieved_context", "")
    plan = state.get("plan", "")
    critic_fb = state.get("critic_feedback", "")

    system = _build_system_message(context)
    messages = [system, SystemMessage(content=TOOL_DESCRIPTIONS)]

    if plan:
        messages.append(SystemMessage(content=f"PLAN:\n{plan}"))
    if critic_fb:
        messages.append(
            SystemMessage(
                content=f"PREVIOUS COMMAND FAILED:\n{critic_fb}\n"
                "Analyze the error and retry with a corrected command."
            )
        )

    last_human = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    is_search = _is_search_query(last_human)
    max_rounds = (
        settings.max_act_rounds_search if is_search else settings.max_act_rounds
    )

    if is_search:
        messages.append(SystemMessage(content=_SEARCH_FORCE_MSG))
        logger.info("Search query detected, forcing tool usage (max_rounds=%d)", max_rounds)

    messages.extend(state["messages"])
    _ensure_ends_with_user(messages)

    tool_trace: list[str] = []
    tool_calls_made: list[tuple[str, dict, str]] = []
    # The earliest response in this loop that carried the reasoning dead-end
    # signature, which is not necessarily the one that hits the round cap.
    dead_end_evidence: Any = None
    unproductive = 0

    for round_num in range(max_rounds):
        response = await _send(llm, messages)
        text = str(response.content or "").strip()

        action = _parse_action(text)
        if not action:
            # An empty response is not an answer. Returning it produced the
            # "I couldn't generate a response" dead end, because extract_reply
            # skips empty AIMessages.
            if not text:
                unproductive += 1
                _log_empty_content(response, f"act_node round {round_num + 1}")
                # Kept because the decision below is made on the round that
                # happens to hit the cap, and the signature does not have to be
                # on that one. A model that returned reasoning and 105 tokens
                # in round 1 and a plain empty reply in round 2 has dead-ended
                # just as clearly, but reading only round 2 saw nothing to act
                # on and abandoned the query.
                if dead_end_evidence is None and _is_reasoning_dead_end(response):
                    dead_end_evidence = response
                if unproductive >= _MAX_UNPRODUCTIVE_ROUNDS:
                    logger.warning(
                        "Act loop abandoned after %d rounds with no tool call and "
                        "no text. If this is a reasoning model, set "
                        "AORTA_CHAT_LLM_TOOL_MODE=native.",
                        unproductive,
                    )
                    # The one place the protocol gets to change. Signalled
                    # rather than done here so act_node issues the retry: this
                    # function's job is to drive one protocol, not to pick one.
                    # The earliest round that showed the signature is the
                    # evidence, falling back to this one.
                    evidence = dead_end_evidence or response
                    if _escalate_to_native(evidence):
                        return _EscalateToNative(
                            tuple(tool_trace),
                            _dead_end_signature(evidence),
                            tuple(tool_calls_made),
                        )
                    return await _abandoned_result(state, tool_trace)
                messages.append(HumanMessage(content=_SEARCH_REPROMPT_MSG))
                continue

            if is_search and not tool_trace:
                unproductive += 1
                logger.info(
                    "Act round %d: search query but no tools used, re-prompting",
                    round_num + 1,
                )
                # Give up rather than spending the whole budget re-asking a
                # model that is not going to comply.
                if unproductive >= _MAX_UNPRODUCTIVE_ROUNDS:
                    logger.warning(
                        "Model would not use tools after %d rounds; answering "
                        "from context instead of spending the remaining budget.",
                        unproductive,
                    )
                    return {
                        "messages": [AIMessage(content=text)],
                        "command_output": text,
                        "tool_trace": tool_trace,
                    }
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=_SEARCH_REPROMPT_MSG))
                continue

            return {
                "messages": [AIMessage(content=text)],
                "command_output": text,
                "tool_trace": tool_trace,
            }

        unproductive = 0
        tool_name, kwargs = action
        logger.info("Act round %d: %s(%s)", round_num + 1, tool_name, kwargs)
        result = await _execute_tool(tool_name, kwargs)
        # The result starts on its own line so that ``Exit code: N`` stays at
        # the start of one: ``critic_node`` scans this trace with
        # ``line.startswith(_EXIT_CODE_PREFIX)``, so putting the result after
        # the arrow hid every failed command in text tool mode and let the
        # critic approve an answer built on one.
        tool_trace.append(f"[{tool_name}({kwargs})] →\n{result}")
        # Structured alongside the rendered form, for the escalated retry's
        # duplicate guard. See `_EscalateToNative.calls`.
        tool_calls_made.append((tool_name, kwargs, result))

        messages.append(AIMessage(content=text))
        messages.append(HumanMessage(
            content=f"TOOL RESULT from {tool_name}:\n{result}\n\n"
            "Use this result to continue. Call another tool if needed, "
            "or provide your final answer (no ACTION: line)."
        ))

    logger.warning(
        "Act loop hit its %d-round budget with the model still calling tools; "
        "asking for a final answer now. Raise MAX_ACT_ROUNDS%s if this query "
        "needs more steps.",
        max_rounds,
        "_SEARCH" if is_search else "",
    )
    # Same user-turn instruction as the native path, and for the same reason.
    messages.append(HumanMessage(content=_FINAL_ANSWER_MSG))
    final = await _send(llm, messages)
    text = str(final.content or "").strip()
    if not text:
        _log_empty_content(final, "act_node final")
        text = _NO_ANSWER_MSG
    return {
        "messages": [AIMessage(content=text)],
        "command_output": text,
        "tool_trace": tool_trace,
    }


# ──────────────────── Critic ─────────────────────

_EXIT_CODE_PREFIX = "Exit code: "


_CRITIC_VALIDATION_PROMPT = """\
You are a command-validation critic for the AORTA codebase. Your job is to check \
whether the generated response is grounded in real codebase files.

Review the GENERATED RESPONSE and the TOOL RESULTS that were gathered.

Check for these problems:
1. Commands referencing scripts or files that were NOT found by the tools
2. Invented flags, arguments, or paths not present in the actual codebase
3. Commands that contradict what the tools revealed about the codebase

If the response is well-grounded in the tool results, reply with exactly: VALID

If there are problems, explain what is wrong and what the correct command should \
be based on the tool results. Do NOT invent information yourself -- only use what \
the tools found.
"""

_CRITIC_FAILURE_PROMPT = """\
You are a command-execution analyst. A command was run inside the AORTA codebase \
and failed. Analyze the error output below and provide:
1. Root cause of the failure
2. A corrected command or clear fix instructions

ERROR OUTPUT:
{errors}
"""


_TOOL_RESULT_PREFIX = "TOOL RESULT from "


async def critic_node(state: AgentState) -> dict[str, Any]:
    """Validate generated commands against codebase reality and check failures.

    Two validation passes:
    1. Check if any executed commands returned non-zero exit codes.
    2. Check if the generated response is grounded in actual tool results
       (not hallucinated).
    """
    iteration = state.get("iteration", 0) + 1

    if iteration > settings.max_retry_iterations:
        return {"iteration": iteration, "critic_feedback": None}

    command_output = state.get("command_output", "")
    if not command_output:
        return {"iteration": iteration, "critic_feedback": None}

    failures: list[str] = []
    # Read the trace act_node returns rather than scanning `messages`. The scan
    # never matched: act appends tool results to its own working list, and only
    # its final answer is returned into state, so the critic always concluded
    # "no tool results gathered" and rejected any answer citing a file -- which
    # then cost a retry. The scan is kept as a fallback in case a future node
    # does put them in the conversation.
    tool_results: list[str] = list(state.get("tool_trace") or [])
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and msg.content.startswith(_TOOL_RESULT_PREFIX):
            tool_results.append(msg.content)

    for content in tool_results:
        if _EXIT_CODE_PREFIX not in content:
            continue
        for line in content.splitlines():
            if not line.startswith(_EXIT_CODE_PREFIX):
                continue
            try:
                exit_code = int(line[len(_EXIT_CODE_PREFIX):].strip())
            except ValueError:
                continue
            if exit_code != 0:
                failures.append(content)

    llm = _get_llm(temperature=0.0, streaming=False)

    if failures:
        error_summary = "\n---\n".join(failures)
        analysis = await _send(
            llm,
            [SystemMessage(content=_CRITIC_FAILURE_PROMPT.format(errors=error_summary))]
        )
        logger.info("Critic found %d failure(s), iteration %d", len(failures), iteration)
        return {"iteration": iteration, "critic_feedback": analysis.content}

    if command_output:
        tool_context = "\n---\n".join(tool_results[:10]) if tool_results else "(no tool results gathered)"
        validation = await _send(
            llm,
            [
                SystemMessage(content=_CRITIC_VALIDATION_PROMPT),
                HumanMessage(
                    content=(
                        f"TOOL RESULTS:\n{tool_context}\n\n"
                        f"GENERATED RESPONSE:\n{command_output}"
                    )
                ),
            ]
        )
        verdict = validation.content.strip()
        # Exact match, not a substring test: "INVALID" contains "VALID", and so
        # does "not valid", so the substring form passed every rejection the
        # critic exists to catch. The prompt asks for exactly "VALID", which
        # makes anything else -- including an explanation -- a retry.
        if verdict.upper() != "VALID":
            logger.info("Critic rejected response, iteration %d: %s", iteration, verdict[:200])
            return {"iteration": iteration, "critic_feedback": verdict}

    return {"iteration": iteration, "critic_feedback": None}


# ──────────────────── Exhausted retries ─────────


#: Prefix on the answer the critic rejected for the last time. The rejected text
#: is kept rather than dropped -- it is usually partly right, and the user waited
#: for it -- but it must not be the only thing they see, which is what happened
#: when this edge went straight to END.
#: ``iteration`` counts critic passes, so it is *attempts*, not retries -- the
#: first pass is the initial validation. Naming it "retries" would overstate the
#: budget by one every time.
_UNRESOLVED_HEADER = (
    "I could not verify this answer against the tool output, and I have used all "
    "{allowed} attempts. Treat it as unconfirmed:"
)

_UNRESOLVED_FOOTER = (
    "The check that failed: {feedback}\n\n"
    "Asking something narrower, or naming the file you care about, usually "
    "gets a grounded answer."
)


def _last_ai_text(messages: list[Any]) -> str:
    """The most recent assistant text in *messages*, or empty."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


async def finalize_node(state: AgentState) -> dict[str, Any]:
    """Report an unresolved criticism instead of passing the answer off as good.

    Reached only when the critic rejected the answer on the final permitted
    attempt. Before this node existed the edge went to ``END`` with the rejected
    ``AIMessage`` still last in state, so :func:`aorta.chat.session.extract_reply`
    returned it as an ordinary reply and the validator's verdict reached nobody:
    the one case the critic exists to catch was the one case it could not report.

    No LLM call. The wording is fixed so that exhausting the budget cannot
    itself fail, and so this node adds no spend to a query that has already
    paid for ``max_retry_iterations`` rounds.
    """
    feedback = (state.get("critic_feedback") or "").strip()
    # `command_output` is the text the critic judged, so it is the rejected
    # answer by definition. The message scan is only a fallback: the critic
    # cannot produce feedback without it, so this path is not reachable today.
    rejected = state.get("command_output") or _last_ai_text(state["messages"])
    header = _UNRESOLVED_HEADER.format(allowed=settings.max_retry_iterations)
    parts = [header, rejected.strip() or "(no answer was produced)"]
    if feedback:
        parts.append(_UNRESOLVED_FOOTER.format(feedback=feedback))
    logger.info(
        "Retry budget exhausted with the critic still rejecting; reporting "
        "the unresolved criticism (iteration %s)",
        state.get("iteration", 0),
    )
    return {"messages": [AIMessage(content="\n\n".join(parts))]}


# ──────────────────── End (Q&A shortcut) ─────────


async def answer_node(state: AgentState) -> dict[str, Any]:
    """Directly answer a question using retrieved context (no tools)."""
    llm = _get_llm(temperature=0.1)
    context = state.get("retrieved_context", "")
    system = _build_answer_message(context)
    messages = [system] + list(state["messages"])

    response = await _send(llm, messages)
    if not str(response.content).strip():
        _log_empty_content(response, "answer_node")
        response = AIMessage(
            content=(
                "I could not answer that from the retrieved context. Try asking "
                "something more specific, or phrase it as a search (for example "
                '"find all ..." or "list all ...") so I can use tools to explore '
                "the codebase."
            )
        )
    return {"messages": [response]}
