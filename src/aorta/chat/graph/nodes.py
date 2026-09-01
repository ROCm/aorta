"""LangGraph node implementations: Router, Plan, Retrieve, Act, Critic."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from aorta.chat.config import settings
from aorta.chat.graph.state import AgentState
from aorta.chat.inference.vllm_client import get_chat_llm
from aorta.chat.rag.repo_map import load_repo_map
from aorta.chat.rag.retriever import get_retriever
from aorta.chat.redaction import redact_for_send
from aorta.chat.tools.artifacts import (
    list_runs,
    read_run_env,
    read_run_matrix,
    search_run_artifacts,
)
from aorta.chat.tools.files import list_files, read_file
from aorta.chat.tools.run import run_terminal_command
from aorta.chat.tools.search import grep_code, search_code, search_repo_map

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the AORTA Codebase Assistant, an AI agent that helps \
developers understand, navigate, and work with the AORTA codebase.

RULES:
1. Only answer questions about the AORTA codebase. Politely refuse anything else.
2. When referencing code, always cite file paths and line numbers.
3. You have tools to explore the codebase: list_files, read_file, search_code, \
   grep_code, search_repo_map, and run_terminal_command (sandboxed). You also have \
   tools for this machine's own AORTA run results: list_runs, read_run_matrix, \
   read_run_env, and search_run_artifacts. Use those when the question is about \
   what a run did rather than what the code says.
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
1. Only answer questions about the AORTA codebase. Politely refuse anything else.
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

PLAN_PROMPT = """\
You are a planning agent. Given the user request and the repository map, \
break down the task into concrete steps.

Available tools:
1. list_files(path) - List files and directories under a path.
2. read_file(file_path) - Read the contents of a file.
3. search_code(query) - Semantic search for code related to the query.
4. grep_code(pattern, path) - Regex search across files (e.g. "def.*config").
5. search_repo_map(query) - Search the function/class index for matching entries.
6. run_terminal_command(command) - Run a sandboxed terminal command.
7. list_runs(path) - List this machine's AORTA run directories.
8. read_run_matrix(path) - Read a run's per-cell pass/fail matrix.
9. read_run_env(path) - Read a run's environment snapshot.
10. search_run_artifacts(query) - Semantic search over indexed run artifacts.

For each step, specify which tool to use and what arguments. For "find all" or \
"search for" queries, plan MULTIPLE searches with different phrasings and tools \
(semantic search + regex grep + repo map lookup) to ensure completeness.

Output your plan as a numbered list. Be specific about file paths.
"""

TOOL_REGISTRY: dict[str, callable] = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "grep_code": grep_code,
    "search_repo_map": search_repo_map,
    "run_terminal_command": run_terminal_command,
    "list_runs": list_runs,
    "read_run_matrix": read_run_matrix,
    "read_run_env": read_run_env,
    "search_run_artifacts": search_run_artifacts,
}

TOOL_DESCRIPTIONS = """\
You have these tools. To call one, output EXACTLY this format on its own line:

ACTION: tool_name(arg1="value1", arg2="value2")

Available tools:

1. list_files(path=".") - List files and directories under a path in the AORTA codebase.
2. read_file(file_path="path/to/file") - Read the contents of a file in the AORTA codebase.
3. search_code(query="search terms", k=10) - Semantic search for code related to the query (k = max results).
4. grep_code(pattern="regex", path=".") - Regex search across files (e.g. "def.*config", "class.*Config").
5. search_repo_map(query="keyword") - Search the function/class index for matching entries.
6. run_terminal_command(command="cmd") - Run a sandboxed terminal command in the AORTA directory.
7. list_runs(path=".") - List this machine's AORTA run directories and their artifacts.
8. read_run_matrix(path="run_dir") - Read a run's matrix.json: per-cell pass/fail and failure hints.
9. read_run_env(path="run_dir") - Read a run's environment snapshot (ROCm version, probe completeness).
10. search_run_artifacts(query="terms") - Semantic search over this machine's indexed run artifacts.

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
  use list_files or run_terminal_command to get real data.
- Run artifacts report fields they did not record as "unknown", listed under \
  "NOT RECORDED". Never read one as zero or as a pass. If the field that decides \
  the question is unknown, say the run did not record it.
"""


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


def _log_empty_content(response: Any, node: str) -> None:
    """Record why a node produced no text, when the model still spent tokens.

    Reasoning models can put everything in a side channel and return empty
    content. The token counts make that diagnosable; ``langchain-openai`` does
    not currently surface gpt-oss's ``reasoning`` field, so it is read
    defensively in case a future version does.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    logger.warning(
        "%s produced no text despite %s output tokens. A reasoning model may "
        "have returned only internal reasoning.",
        node,
        usage.get("output_tokens", "an unknown number of"),
    )
    reasoning = (getattr(response, "additional_kwargs", None) or {}).get("reasoning")
    if reasoning:
        logger.debug("%s reasoning channel: %s", node, reasoning)


def _parse_action(text: str) -> tuple[str, dict] | None:
    """Extract a tool call from ACTION: tool_name(k="v", ...) format."""
    import re
    match = re.search(r'ACTION:\s*(\w+)\(([^)]*)\)', text)
    if not match:
        return None
    tool_name = match.group(1)
    args_str = match.group(2).strip()
    if tool_name not in TOOL_REGISTRY:
        return None
    kwargs: dict = {}
    if args_str:
        for part in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+))', args_str):
            key = part.group(1)
            val = part.group(2) if part.group(2) is not None else (
                part.group(3) if part.group(3) is not None else int(part.group(4))
            )
            kwargs[key] = val
    return tool_name, kwargs


def _normalise_tool_name(tool_name: str) -> str:
    """Strip harmony control markers a provider may leak into the tool name.

    gpt-oss speaks the harmony format internally, and a serving stack whose
    parser is imperfect emits names like ``search_code<|channel|>commentary``.
    Everything from the first ``<|`` is protocol, not part of the name.
    """
    return tool_name.split("<|", 1)[0].strip()


def _execute_tool(tool_name: str, kwargs: dict) -> str:
    """Call a tool and return its string result, or an error the model can read.

    An unknown name must not raise: in the native protocol the name comes
    straight from the provider, so a hallucinated or mangled one would abort the
    whole graph run instead of giving the model a chance to correct itself.
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
        result = tool_fn.invoke(kwargs)
    except Exception as exc:
        result = f"Tool error: {exc}"
    return str(result)


# ──────────────────── Router ─────────────────────


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
    route_text = response.content.strip().lower()
    route = "question" if "question" in route_text else "action"
    logger.info("Router classified as: %s (raw: %r)", route, route_text)
    return {"route": route}


# ──────────────────── Plan ───────────────────────


async def plan_node(state: AgentState) -> dict[str, Any]:
    """Generate a step-by-step plan for action-type requests."""
    llm = _get_llm(temperature=0.1)
    repo_map = load_repo_map()
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
    """Run RAG retrieval to gather context for the user's query."""
    last_human = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"retrieved_context": ""}

    try:
        retriever = get_retriever()
        docs = retriever.invoke(last_human)
    except FileNotFoundError:
        return {
            "retrieved_context": "(Index not built yet -- run indexing first.)"
        }

    if not docs:
        return {"retrieved_context": "(No relevant code found.)"}

    chunks = []
    for doc in docs:
        src = doc.metadata.get("source", "?")
        chunks.append(f"### {src}\n```\n{doc.page_content}\n```")
    return {"retrieved_context": "\n\n".join(chunks)}


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

_SEARCH_REPROMPT_MSG = (
    "You responded without using any tools. This is a search query that requires "
    "thorough exploration of the codebase. You MUST call grep_code or search_repo_map "
    "FIRST to find all relevant results. Please issue an ACTION: line now."
)

#: Consecutive rounds where the model neither called a tool nor said anything.
#: A model that cannot drive the protocol will not learn it by round eight, and
#: each round is a billed call: gpt-oss burned 11 on one query before this cap.
_MAX_UNPRODUCTIVE_ROUNDS = 2

_NO_ANSWER_MSG = (
    "I could not complete that request. The model did not produce an answer or "
    "call any tools. If you are using a reasoning model, set LLM_TOOL_MODE=native "
    "so it can call tools through the OpenAI function-calling API instead of the "
    "ACTION: text protocol."
)

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
    """Tool-using loop, in whichever protocol LLM_TOOL_MODE selects."""
    mode = settings.llm_tool_mode.strip().lower()
    if mode == "native":
        return await _act_native(state)
    if mode == "text":
        return await _act_text(state)
    raise ValueError(
        f"unknown LLM_TOOL_MODE: {settings.llm_tool_mode!r} "
        "(expected one of native, text)"
    )


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


def _ensure_ends_with_user(messages: list[Any]) -> None:
    """Append a user turn when the conversation ends with an assistant one.

    Anthropic treats a trailing assistant message as a prefill to continue, and
    models that disallow prefill reject the request outright: *"This model does
    not support assistant message prefill. The conversation must end with a user
    message."* That is exactly the shape a critic-triggered retry produces, since
    the rejected answer is the last thing in state.
    """
    if messages and isinstance(messages[-1], AIMessage):
        messages.append(HumanMessage(content=_RETRY_NUDGE))


async def _act_native(state: AgentState) -> dict[str, Any]:
    """Tool loop over the OpenAI function-calling API.

    What reasoning models expect. The provider returns structured ``tool_calls``
    and ``finish_reason=tool_calls``, so nothing depends on the model reproducing
    a text syntax, and there is no parsing to fail.
    """
    plain = _get_llm(temperature=0.1, streaming=False)
    llm = plain.bind_tools(list(TOOL_REGISTRY.values()))

    messages = _act_messages(state)
    is_search = _is_search_query(_last_human(state))
    if is_search:
        messages.append(SystemMessage(content=_SEARCH_FORCE_MSG))
    messages.extend(state["messages"])
    _ensure_ends_with_user(messages)

    max_rounds = (
        settings.max_act_rounds_search if is_search else settings.max_act_rounds
    )
    unproductive = 0
    seen: set[str] = set()
    trace: list[str] = []

    for round_num in range(max_rounds):
        response = await _send(llm, messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        text = str(response.content or "").strip()

        if not tool_calls:
            if text:
                return {
                    "messages": [AIMessage(content=text)],
                    "command_output": text,
                    "tool_trace": trace,
                }
            unproductive += 1
            _log_empty_content(response, f"act_node round {round_num + 1}")
            if unproductive >= _MAX_UNPRODUCTIVE_ROUNDS:
                break
            messages.append(HumanMessage(content=_SEARCH_REPROMPT_MSG))
            continue

        unproductive = 0
        messages.append(response)
        for call in tool_calls:
            signature = f"{call['name']}({sorted((call['args'] or {}).items())})"
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
            result = _execute_tool(call["name"], call["args"])
            trace.append(f"{_TOOL_RESULT_PREFIX}{call['name']}:\n{result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # Reaching here means the loop never produced a tool-free reply, so the
    # budget ran out mid-task. Say so: the answer will read as truncated, and
    # the cause is a knob the user can turn.
    logger.warning(
        "Act loop hit its %d-round budget with the model still calling tools; "
        "asking for a final answer now. Raise MAX_ACT_ROUNDS%s if this query "
        "needs more steps.",
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
    final = await _send(plain, messages)
    text = str(final.content or "").strip()
    if not text:
        _log_empty_content(final, "act_node final")
        text = _NO_ANSWER_MSG
    return {
        "messages": [AIMessage(content=text)],
        "command_output": text,
        "tool_trace": trace,
    }


async def _act_text(state: AgentState) -> dict[str, Any]:
    """ReAct-style loop: LLM outputs ACTION lines, we execute and feed back."""
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
                if unproductive >= _MAX_UNPRODUCTIVE_ROUNDS:
                    logger.warning(
                        "Act loop abandoned after %d rounds with no tool call and "
                        "no text. If this is a reasoning model, set "
                        "LLM_TOOL_MODE=native.",
                        unproductive,
                    )
                    return {
                        "messages": [AIMessage(content=_NO_ANSWER_MSG)],
                        "command_output": "",
                        "tool_trace": tool_trace,
                    }
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
        result = _execute_tool(tool_name, kwargs)
        tool_trace.append(f"[{tool_name}({kwargs})] → {result}")

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
        if "VALID" not in verdict.upper():
            logger.info("Critic rejected response, iteration %d: %s", iteration, verdict[:200])
            return {"iteration": iteration, "critic_feedback": verdict}

    return {"iteration": iteration, "critic_feedback": None}


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
