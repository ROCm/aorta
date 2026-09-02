"""Regression tests for the lazy top-level command group (issue #417).

``aorta`` used to import all ten command modules on every invocation, which
was roughly 190 ms of a 255 ms ``aorta --help``. The laziness is only worth
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


@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_every_command_resolves_through_the_group(name: str) -> None:
    """Covers the deprecated probe/run/triage aliases (issue #248) along with the rest."""
    result = CliRunner().invoke(main, [name, "--help"])
    assert result.exit_code == 0, result.output


def test_shell_completion_offers_every_command() -> None:
    completions = main.shell_complete(click.Context(main), "")
    assert sorted(item.value for item in completions) == sorted(_COMMANDS)


def test_shell_completion_filters_on_the_incomplete_prefix() -> None:
    completions = main.shell_complete(click.Context(main), "en")
    assert [item.value for item in completions] == ["env", "environments"]
