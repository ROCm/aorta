"""Regression tests for the lazy top-level command group (issue #417).

``aorta`` used to import every command module on every invocation, which was
roughly 190 ms of a 255 ms ``aorta --help`` (measured when there were ten). The laziness is only worth
anything if it holds, and it is easy to lose by accident: a convenience import
at the top of ``aorta/cli/__init__.py``, or a help/completion path that reads
``short_help`` off a real command, quietly drags the whole graph back in.

The import assertions run in a subprocess. By the time this module executes,
the pytest session has already imported most of ``aorta`` for other tests, so
the in-process ``sys.modules`` says nothing about what a real invocation loads.
"""

from __future__ import annotations

import json
import subprocess
import sys

import click
import pytest
from click.testing import CliRunner

from aorta.cli import _COMMANDS, main
from aorta.cli._lazy_group import LazyGroup

_COMMAND_MODULES = {entry.import_path.partition(":")[0] for entry in _COMMANDS.values()}

_REPORT = (
    'import json, sys; print(json.dumps(sorted(m for m in sys.modules if m.startswith("aorta."))))'
)


def _modules_loaded_by(*statements: str) -> set[str]:
    """Run statements in a fresh interpreter and report the ``aorta.*`` modules loaded."""
    completed = subprocess.run(
        [sys.executable, "-c", "\n".join([*statements, _REPORT])],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return set(json.loads(completed.stdout.splitlines()[-1]))


@pytest.mark.parametrize(
    ("scenario", "statements"),
    [
        ("import", ("import aorta.cli",)),
        (
            "help",
            (
                "from click.testing import CliRunner",
                "from aorta.cli import main",
                "result = CliRunner().invoke(main, ['--help'])",
                "assert result.exit_code == 0, result.output",
            ),
        ),
        (
            "completion",
            (
                "import click",
                "from aorta.cli import main",
                "completions = main.shell_complete(click.Context(main), '')",
                "assert completions, 'no completions offered'",
            ),
        ),
    ],
)
def test_no_command_module_is_imported(scenario: str, statements: tuple[str, ...]) -> None:
    """Neither importing the CLI, nor ``--help``, nor completion may load a command."""
    eager = _modules_loaded_by(*statements) & _COMMAND_MODULES
    assert not eager, f"{scenario} eagerly imported: {sorted(eager)}"


@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_registered_help_matches_the_command(name: str) -> None:
    """The registry's help must render the same short help as the command's own.

    The registry duplicates each command's help so ``--help`` need not import
    anything; this is the test that keeps the copy honest. Comparing rendered
    short help across several widths (rather than the raw strings) is what the
    user actually sees, and lets the registry hold just the first paragraph.
    """
    entry = _COMMANDS[name]
    command = entry.load()
    assert command.name == name
    stand_in = click.Command(name, help=entry.help)
    for limit in (30, 45, 62, 200):
        assert stand_in.get_short_help_str(limit) == command.get_short_help_str(limit)


def test_help_lists_exactly_the_registered_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    listed = [
        line.split()[0]
        for line in result.output.partition("Commands:\n")[2].splitlines()
        if line.strip()
    ]
    assert listed == sorted(_COMMANDS)


def _commands_section(help_output: str) -> str:
    return help_output.partition("Commands:\n")[2]


def test_commands_section_matches_an_eagerly_built_group() -> None:
    """The registry must render the Commands block an eager group would.

    ``LazyGroup.format_commands`` says it mirrors ``click.Group``'s while
    sourcing rows from the registry rather than imported commands. Checking
    that against a real eager group is what turns the claim from a comment
    into a test, and it pins the property this whole change rests on: that
    ``aorta --help`` stays byte-for-byte what it was, Click's column width and
    ``...`` truncation included.
    """
    eager = click.Group(
        main.name,
        commands={name: entry.load() for name, entry in _COMMANDS.items()},
    )
    lazy = _commands_section(main.get_help(click.Context(main, terminal_width=80)))
    assert lazy, "no Commands section rendered"
    assert lazy == _commands_section(eager.get_help(click.Context(eager, terminal_width=80)))


@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_every_command_resolves_through_the_group(name: str) -> None:
    """Covers the deprecated probe/triage aliases (issue #248) along with the rest."""
    result = CliRunner().invoke(main, [name, "--help"])
    assert result.exit_code == 0, result.output


def test_shell_completion_offers_every_command() -> None:
    completions = main.shell_complete(click.Context(main), "")
    assert sorted(item.value for item in completions) == sorted(_COMMANDS)


def test_shell_completion_carries_help_from_the_registry() -> None:
    """Every completion item ships the help column, sourced from the registry.

    ``zsh_complete`` renders this column, and the registry is the only place it
    can come from without importing the command -- so an empty help here means
    either the laziness or the description in the shell menu has been lost.
    """
    completions = {item.value: item.help for item in main.shell_complete(click.Context(main), "")}
    for name, entry in _COMMANDS.items():
        rendered = completions.get(name)
        assert rendered, f"{name} completes with no help text: {rendered!r}"
        # Tolerate Click's short-help truncation instead of pinning a width.
        assert entry.help.startswith(rendered.removesuffix("...").rstrip()), (name, rendered)


def test_shell_completion_filters_on_the_incomplete_prefix() -> None:
    completions = main.shell_complete(click.Context(main), "en")
    assert [item.value for item in completions] == ["env", "environments"]


def test_unknown_command_still_suggests_a_registered_one() -> None:
    """A typo must keep getting Click's "Did you mean ...?" hint.

    Click sources the suggestions from ``Group.commands``, which lazy names
    never enter, so laziness silently drops them; an eagerly built group is the
    reference for what the user used to see. Click only grew suggestions in
    8.4, hence comparing against an eager group rather than pinning the text.
    """
    eager = click.Group(main.name, commands={name: click.Command(name) for name in _COMMANDS})
    expected = CliRunner().invoke(eager, ["enviroments"]).output
    if "Did you mean" not in expected:
        pytest.skip("this Click does not offer unknown-command suggestions")
    result = CliRunner().invoke(main, ["enviroments"])
    assert "environments" in result.output, result.output


def test_eagerly_added_commands_keep_working_alongside_lazy_ones() -> None:
    """``add_command`` must survive the swap ``resolve_command`` does.

    ``LazyGroup``'s docstring promises eagerly added commands keep working
    alongside lazy ones, and the unknown-command path now temporarily replaces
    ``self.commands`` to recover Click's suggestions. Nothing else covers the
    two together.
    """
    group = LazyGroup(
        "eager-and-lazy",
        lazy_commands={"agent": _COMMANDS["agent"]},
    )

    @click.command("added")
    def added() -> None:
        """Added the usual way."""

    group.add_command(added)
    assert group.list_commands(click.Context(group)) == ["added", "agent"]
    result = CliRunner().invoke(group, ["added", "--help"])
    assert result.exit_code == 0, result.output
    assert group.get_command(click.Context(group), "added") is added
    # The stand-ins must not outlive the lookup: a group whose ``commands``
    # advertises `agent` would hand a caller an empty command with no callback.
    assert set(group.commands) == {"added"}


def test_resolved_commands_carry_their_registry_name() -> None:
    """The object handed to Click must be named after the key it was reached by.

    Click 8.0.0 builds the child context from ``cmd.name`` rather than the
    invoked name, so a key that disagrees renders a usage line the user cannot
    type. Every entry here happens to agree already; the assertion is on
    ``get_command``, which is what ``bench``'s ``hw_queue_eval`` -> ``cli``
    entry needs and what keeps a future mismatch from going unnoticed.
    """
    ctx = click.Context(main)
    for name in _COMMANDS:
        resolved = main.get_command(ctx, name)
        assert resolved is not None, name
        assert resolved.name == name
