"""Agents registry: built-ins + entry-point discovery + collision detection.

`load_agents()` returns the merged registry of built-in agents and plugin
contributions, keyed by name. Each entry carries its `source_package` so
collision errors can name the conflicting parties.

An *agent* here is an autonomous workflow: it runs unattended to a verdict and
writes artifacts. Interactive, human-in-the-loop work is a different surface and
does not belong in this group. The payload is the `click.Command` implementing
the agent, since the registry's consumer is the `aorta agent` command group.

Built-ins register **directly** in `BUILTIN_AGENTS`; the `aorta.agents`
entry-point group is reserved for third-party packages. That is the same split
`BUILTIN_MITIGATIONS` / `BUILTIN_ENVIRONMENTS` already use. Plugin authors
register one entry-point per agent, where the entry-point name IS the name
`aorta agent <name>` dispatches on.
"""

import logging
from importlib import import_module
from importlib.metadata import entry_points

import click

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownAgentError,
)
from aorta.registry.types import Agent

logger = logging.getLogger(__name__)

_GROUP = "aorta.agents"

# Values are ``"module:attr"`` target strings, not imported objects: `aorta.
# registry` must not import `aorta.cli`, and the string form is exactly what an
# entry-point value looks like, so built-ins and plugins resolve through one
# code path. Click code stays under `aorta/cli/` per the repo layering rule.
BUILTIN_AGENTS: dict[str, str] = {
    "mitigate": "aorta.cli.agent_mitigate:mitigate",
}


def _resolve(target: str) -> object:
    """Import a ``"module:attr"`` target string, matching entry-point semantics."""
    module_name, sep, attr = target.partition(":")
    if not sep or not attr:
        raise RegistryError(f"agent target {target!r} is not of the form 'module:attr'")
    obj: object = import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_agents() -> dict[str, Agent]:
    """Discover and merge all agents: built-ins, then entry-point plugins.

    A built-in that fails to resolve raises -- that is an aorta bug and hiding it
    would silently drop a shipped command. A *plugin* entry-point that fails to
    load, or that does not resolve to a `click.Command`, is logged via the
    `aorta.registry.agents` logger and skipped, so one broken third-party package
    cannot take the whole `aorta agent` namespace down with it (same rule as
    `aorta.run.discovery`). A name collision is not a load failure and still
    raises: silently shadowing an agent is exactly the failure the sibling
    registries refuse to allow.

    No caching -- re-reads entry-points each call, mirroring the mitigation and
    environment loaders.

    Raises:
        RegistryCollisionError: two contributors registered the same agent name.
        RegistryError: a built-in target was malformed or did not resolve to a
            `click.Command`.
    """
    registry: dict[str, Agent] = {}
    for name, target in BUILTIN_AGENTS.items():
        command = _resolve(target)
        if not isinstance(command, click.Command):
            raise RegistryError(
                f"built-in agent '{name}' target {target!r} must resolve to a "
                f"click.Command; got {type(command).__name__}"
            )
        registry[name] = Agent(name=name, command=command, source_package="aorta")

    for ep in entry_points(group=_GROUP):
        plugin_name = ep.dist.name if ep.dist else "<unknown>"
        try:
            command = ep.load()
        except Exception:
            # ``exc_info=True`` keeps the full traceback on the warning record;
            # plugin load failures are most often ImportError chains and are
            # undiagnosable without it.
            logger.warning(
                "Failed to load agent '%s' from '%s'", ep.name, plugin_name, exc_info=True
            )
            continue
        if not isinstance(command, click.Command):
            # A mis-registered plugin returning a function or a class would
            # otherwise fail much later inside Click's dispatch with a confusing
            # AttributeError.
            logger.warning(
                "Entry point '%s' from '%s' = %r is not a click.Command; skipping.",
                ep.name,
                plugin_name,
                command,
            )
            continue
        if ep.name in registry:
            existing = registry[ep.name].source_package
            raise RegistryCollisionError(
                f"agent '{ep.name}' registered by both '{existing}' "
                f"and '{plugin_name}' — rename one or remove the duplicate"
            )
        registry[ep.name] = Agent(name=ep.name, command=command, source_package=plugin_name)

    return registry


def get_agent(name: str) -> Agent:
    """Return the registry entry for an agent name.

    Unlike `get_mitigation` (which returns a dict), the `Agent` dataclass IS the
    public surface -- callers need both the command and its `source_package`.
    """
    registry = load_agents()
    if name not in registry:
        raise UnknownAgentError(
            f"unknown agent '{name}'; available: {sorted(registry)}; "
            f"if you expected a plugin-contributed entry, ensure the plugin is installed"
        )
    return registry[name]
