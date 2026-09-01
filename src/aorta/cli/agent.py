"""``aorta agent`` -- namespace for autonomous, run-to-verdict agents.

``aorta agent <name>`` dispatches through the ``aorta.agents`` registry
(:mod:`aorta.registry.agents`), so adding an agent means registering a Click
command -- as a built-in, or from a third-party package via the entry-point
group -- rather than editing this module.

The line this namespace draws is autonomous versus interactive: an agent runs
unattended, reaches a verdict, and writes artifacts. Conversational,
human-in-the-loop work is a different front door and does not belong here.

``aorta agent -- <command>`` was the whole command before the namespace existed
(it *was* the mitigation search). It keeps working for one release as a
deprecated alias for ``aorta agent mitigate``, following the same shim shape
``aorta probe`` / ``aorta triage`` use for ``aorta sweep``.
"""

from __future__ import annotations

import click

from aorta.cli._deprecation import emit_deprecation
from aorta.registry import RegistryError, UnknownAgentError, get_agent, load_agents

#: The agent the pre-namespace ``aorta agent -- <cmd>`` form resolves to.
_LEGACY_AGENT = "mitigate"

# ``--help`` / ``-h`` address the group itself. Every other leading dash-token
# is either an option of the old flat command (``--output``, ``-v``) or its
# mandatory ``--`` separator, so a leading dash is a reliable discriminator:
# agent names never start with one, and this cannot shadow a registered agent.
_GROUP_TOKENS = frozenset({"--help", "-h"})


def _is_legacy_invocation(args: list[str]) -> bool:
    return bool(args) and args[0].startswith("-") and args[0] not in _GROUP_TOKENS


class _AgentGroup(click.Group):
    """Registry-backed group, plus the deprecation shim for the pre-namespace form."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not ctx.resilient_parsing and _is_legacy_invocation(args):
            # Rewrite the argv rather than keep a second copy of the command:
            # the deprecated form then runs the exact same object as
            # ``aorta agent mitigate``, so it cannot drift. The notice goes to
            # stderr, so a scripted caller's stdout stays parseable and its
            # exit code is unchanged.
            emit_deprecation("aorta agent -- <command>", "aorta agent mitigate -- <command>")
            args = [_LEGACY_AGENT, *args]
        return super().parse_args(ctx, args)

    def list_commands(self, ctx: click.Context) -> list[str]:
        try:
            return sorted(load_agents())
        except RegistryError as exc:
            raise click.ClickException(str(exc)) from exc

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        try:
            return get_agent(cmd_name).command
        except UnknownAgentError as exc:
            if ctx.resilient_parsing:
                return None  # shell completion: no output, no error
            # Returning None here would get Click's bare "No such command",
            # dropping the registry's "is the plugin installed?" hint -- the
            # single most common cause of a missing agent name.
            raise click.UsageError(str(exc), ctx=ctx) from exc
        except RegistryError as exc:
            raise click.ClickException(str(exc)) from exc


@click.group(name="agent", cls=_AgentGroup)
def agent() -> None:
    """Autonomous agents: run unattended to a verdict and write artifacts."""


__all__ = ["agent"]
