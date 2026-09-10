"""Chat-tool registry: built-ins + entry-point discovery + collision detection.

`load_chat_tools()` returns the merged registry of the tools this package ships
and anything an installed third party contributed to the ``aorta.chat_tools``
entry-point group, keyed by name. Each entry carries its `source_package` so a
collision error can name the conflicting parties and so the ACTION: protocol's
prompt can say where an unfamiliar tool came from.

Built-ins register **directly** in `BUILTIN_CHAT_TOOLS`; the entry-point group
is reserved for packages outside this repo. That is the same split
`BUILTIN_MITIGATIONS` / `BUILTIN_AGENTS` already use, and for the same reason: a
tool that ships with aorta gains nothing from being discovered at runtime, and
an entry point it does not need is one more thing that can be mis-registered.

The payload is a langchain `BaseTool` -- what the `@tool` decorator produces --
because both consumers need one. `act_node` hands the whole list to
`bind_tools()` for the native function-calling protocol, and calls `.invoke()`
on a single entry for the ACTION: text protocol.

**On the entry-point name.** It IS the tool name, as in `aorta.mitigations`, and
it must equal the `BaseTool`'s own `name`. Those are two different strings in
two different files, and only the second one reaches the model: `bind_tools`
sends `tool.name`, so the provider echoes `tool.name` back, and a registry keyed
on a different entry-point name would never find it. A mismatch is therefore a
tool that silently fails on every call, which is worth refusing at load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import entry_points

from langchain_core.tools import BaseTool

from aorta.chat.tools.artifacts import (
    list_runs,
    read_run_env,
    read_run_matrix,
    search_run_artifacts,
)
from aorta.chat.tools.files import list_files, read_file
from aorta.chat.tools.run import run_terminal_command
from aorta.chat.tools.search import grep_code, search_code, search_repo_map
from aorta.registry.errors import RegistryCollisionError, RegistryError

logger = logging.getLogger(__name__)

_GROUP = "aorta.chat_tools"

#: The tools `aorta chat` ships unconditionally. Keys must match each tool's own
#: ``name``; :func:`load_chat_tools` raises if they drift, since that is an
#: aorta bug.
BUILTIN_CHAT_TOOLS: dict[str, BaseTool] = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "grep_code": grep_code,
    "search_repo_map": search_repo_map,
    "list_runs": list_runs,
    "read_run_matrix": read_run_matrix,
    "read_run_env": read_run_env,
    "search_run_artifacts": search_run_artifacts,
}

#: Shipped, but only registered when ``enable_shell_tool`` is set. Kept out of
#: :data:`BUILTIN_CHAT_TOOLS` rather than filtered later so that a disabled
#: shell is absent from the registry the prompts are built from, not merely
#: refused at call time -- a tool the model is never told about is one no
#: prompt-injected text can talk it into reaching for.
OPTIONAL_CHAT_TOOLS: dict[str, BaseTool] = {
    "run_terminal_command": run_terminal_command,
}


def enabled_builtins() -> dict[str, BaseTool]:
    """Built-ins plus whichever optional tools the profile switched on."""
    from aorta.chat.config import settings

    tools = dict(BUILTIN_CHAT_TOOLS)
    if settings.enable_shell_tool:
        tools.update(OPTIONAL_CHAT_TOOLS)
    return tools


@dataclass(frozen=True)
class ChatTool:
    """One entry in the merged registry."""

    name: str
    tool: BaseTool
    source_package: str  # "aorta" for built-ins, dist name for contributors


def load_chat_tools() -> dict[str, ChatTool]:
    """Discover and merge all chat tools: built-ins, then entry-point plugins.

    A *plugin* entry-point that fails to load, does not resolve to a `BaseTool`,
    or whose entry-point name disagrees with the tool's own name is logged via
    the `aorta.chat.plugins` logger and skipped, so one broken third-party
    package cannot take the assistant's whole tool surface down with it (same
    rule as `aorta.run.discovery` and `aorta.registry.agents`). A name collision
    is not a load failure and still raises: shadowing `run_terminal_command`
    with someone else's idea of it is precisely the failure the sibling
    registries refuse to allow.

    No caching -- re-reads entry-points each call, mirroring the other loaders.
    `aorta.chat.graph.nodes` calls it once at import and keeps the result.

    Raises:
        RegistryCollisionError: two contributors registered the same tool name.
        RegistryError: a built-in is not a `BaseTool`, or its key and its own
            name disagree.
    """
    registry: dict[str, ChatTool] = {}
    for name, tool in enabled_builtins().items():
        if not isinstance(tool, BaseTool):
            raise RegistryError(
                f"built-in chat tool '{name}' must be a langchain BaseTool; "
                f"got {type(tool).__name__}"
            )
        if tool.name != name:
            raise RegistryError(
                f"built-in chat tool is registered as '{name}' but names itself "
                f"'{tool.name}'; the model only ever sees the latter"
            )
        registry[name] = ChatTool(name=name, tool=tool, source_package="aorta")

    for ep in entry_points(group=_GROUP):
        plugin_name = ep.dist.name if ep.dist else "<unknown>"
        try:
            tool = ep.load()
        except Exception:
            # ``exc_info=True`` keeps the full traceback on the warning record;
            # plugin load failures are most often ImportError chains and are
            # undiagnosable without it.
            logger.warning(
                "Failed to load chat tool '%s' from '%s'", ep.name, plugin_name, exc_info=True
            )
            continue
        if not isinstance(tool, BaseTool):
            # A plain function is the likely mistake -- the author forgot
            # langchain's @tool decorator. Without this it would reach
            # bind_tools() and fail there, several frames from the cause.
            logger.warning(
                "Entry point '%s' from '%s' = %r is not a langchain BaseTool; "
                "skipping. Decorate the function with @tool.",
                ep.name,
                plugin_name,
                tool,
            )
            continue
        if tool.name != ep.name:
            logger.warning(
                "Chat tool from '%s' is registered as '%s' but names itself "
                "'%s'; skipping. The model is offered '%s' and asks for it by "
                "that name, so the two must agree -- rename the entry point.",
                plugin_name,
                ep.name,
                tool.name,
                tool.name,
            )
            continue
        if ep.name in registry:
            existing = registry[ep.name].source_package
            raise RegistryCollisionError(
                f"chat tool '{ep.name}' registered by both '{existing}' "
                f"and '{plugin_name}' — rename one or remove the duplicate"
            )
        registry[ep.name] = ChatTool(name=ep.name, tool=tool, source_package=plugin_name)

    return registry


__all__ = [
    "BUILTIN_CHAT_TOOLS",
    "OPTIONAL_CHAT_TOOLS",
    "ChatTool",
    "enabled_builtins",
    "load_chat_tools",
]
