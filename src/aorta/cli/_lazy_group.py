"""A ``click.Group`` that imports a subcommand's module only when it is used.

Every ``aorta`` invocation used to import all ten command modules, and with
them most of the package -- roughly 190 ms of the 255 ms that ``aorta --help``
took (issue #417). The cost is breadth rather than weight: no single module is
expensive, but ``aorta --help`` and shell completion paid for all of them.

:class:`LazyGroup` keeps a registry mapping a command name to the import path
of its ``click.Command`` and resolves that path on first use. The registry also
carries each command's help text, and that duplication is the whole point:
:meth:`click.Group.format_commands` and :meth:`click.Group.shell_complete` call
:meth:`get_command` for every entry just to read a short help string, so
without a registry copy the help and completion paths would import exactly the
modules this group exists to avoid.

``tests/cli/test_lazy_commands.py`` fails if a registry help string drifts from
the command's own, and if any command module is imported eagerly.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from click.shell_completion import CompletionItem


def _named(command: click.Command, name: str) -> click.Command:
    """Return ``command`` carrying ``name``, copying it if it is named otherwise.

    A registry key need not match the loaded object's own ``name``: ``bench``
    registers ``hw_queue_eval`` against a group defined as ``cli``. Click 8.0.0
    builds the child context from ``cmd.name`` rather than the invoked name, so
    the mismatch renders as ``Usage: aorta bench cli ...`` and points the user
    at a command line that does not exist. That was fixed in Click 8.0.1, but
    ``click>=8.0.0`` still resolves 8.0.0.

    Copying leaves the module-level command object unmutated -- ``cli`` is also
    the entry point of ``python -m aorta.hw_queue_eval``, which must keep its
    own name. The copy is shallow, so subcommands and params stay shared.
    """
    if command.name == name:
        return command
    renamed = copy(command)
    renamed.name = name
    return renamed


@dataclass(frozen=True)
class LazyCommand:
    """Registry entry: where a command lives, and what ``--help`` says about it.

    Attributes:
        import_path: ``"module:attribute"`` locating the ``click.Command``.
        help: The command's help text. Only the first paragraph is ever
            rendered, so registering just that paragraph is enough; see the
            module docstring for why it is duplicated here at all.
    """

    import_path: str
    help: str

    def load(self) -> click.Command:
        """Import the module and return the command object."""
        module_name, _, attribute = self.import_path.partition(":")
        command: click.Command = getattr(import_module(module_name), attribute)
        return command


class LazyGroup(click.Group):
    """Group whose subcommands come from a name -> :class:`LazyCommand` registry.

    Commands added the usual way with ``add_command`` keep working and are
    listed alongside the lazy ones.
    """

    def __init__(
        self,
        *args: Any,
        lazy_commands: dict[str, LazyCommand] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_commands: dict[str, LazyCommand] = dict(lazy_commands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self.lazy_commands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        entry = self.lazy_commands.get(cmd_name)
        if entry is None:
            return super().get_command(ctx, cmd_name)
        return _named(entry.load(), cmd_name)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and args[0] in self.lazy_commands:
            return super().resolve_command(ctx, args)
        # Click draws its "Did you mean ...?" candidates from ``self.commands``,
        # which lazy names never enter -- so an eager group answers
        # ``aorta enviroments`` with ``Did you mean 'environments'?`` and this
        # one would not. Only the keys are read, and only on this
        # unknown-command path, so bare stand-ins are enough and nothing is
        # imported. Registered commands keep precedence over the registry.
        registered = self.commands
        self.commands = {**{name: click.Command(name) for name in self.lazy_commands}, **registered}
        try:
            return super().resolve_command(ctx, args)
        finally:
            self.commands = registered

    def _summary(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Return a command object good enough to read a short help off, no import.

        For a registry entry that is a throwaway ``click.Command`` carrying the
        registered help text. Building a real Click command rather than
        returning the string means Click applies its own truncation rules, so
        the rendered output is identical to an eagerly constructed group's.
        """
        entry = self.lazy_commands.get(cmd_name)
        if entry is None:
            return super().get_command(ctx, cmd_name)
        return click.Command(cmd_name, help=entry.help)

    def _visible_summaries(
        self,
        ctx: click.Context,
        prefix: str = "",
    ) -> list[tuple[str, click.Command]]:
        summaries = []
        for name in self.list_commands(ctx):
            if not name.startswith(prefix):
                continue
            cmd = self._summary(ctx, name)
            if cmd is not None and not cmd.hidden:
                summaries.append((name, cmd))
        return summaries

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Mirrors click.Group.format_commands, but sources each row from the
        # registry instead of an imported command.
        summaries = self._visible_summaries(ctx)
        if not summaries:
            return
        limit = formatter.width - 6 - max(len(name) for name, _ in summaries)
        with formatter.section("Commands"):
            formatter.write_dl([(name, cmd.get_short_help_str(limit)) for name, cmd in summaries])

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        # Mirrors click.Group.shell_complete. Completion runs on every TAB, so
        # it matters at least as much as --help that it imports nothing; the
        # option completions still come from click.Command.
        from click.shell_completion import CompletionItem  # noqa: PLC0415

        results = [
            CompletionItem(name, help=cmd.get_short_help_str())
            for name, cmd in self._visible_summaries(ctx, incomplete)
        ]
        results.extend(click.Command.shell_complete(self, ctx, incomplete))
        return results


__all__ = ["LazyCommand", "LazyGroup"]
