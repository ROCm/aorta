"""``aorta bench`` — perf-characterization micro-benchmarks."""

from __future__ import annotations

import click

from aorta.cli._lazy_group import LazyCommand, LazyGroup

_INSTALL_HINT = (
    "'aorta bench hw_queue_eval' requires the hw-queue extra.\n"
    "Install it with:  pip install 'amd-aorta[hw-queue]'"
)


def _install_hint_command(name: str) -> click.Command:
    """Stand in for ``hw_queue_eval`` when the [hw-queue] extra is absent.

    Accepts whatever arguments follow so that every invocation -- bare or with
    a subcommand -- reports the install hint rather than Click's generic "No
    such command". ``--help`` still exits 0; the hint is the signal, not the
    exit code.
    """

    @click.command(
        name=name,
        help=_INSTALL_HINT,
        context_settings={"ignore_unknown_options": True},
    )
    @click.argument("argv", nargs=-1, type=click.UNPROCESSED, metavar="[COMMAND]...")
    def hint(argv: tuple[str, ...]) -> None:
        raise click.ClickException(_INSTALL_HINT)

    return hint


class _BenchGroup(LazyGroup):
    """Reports an absent [hw-queue] extra as an install hint.

    Only a missing *external* dependency (e.g. torch, numpy) means "extra not
    installed". A ModuleNotFoundError for an aorta.hw_queue_eval.* sub-module
    -- or any other ImportError -- propagates so real bugs surface.
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        try:
            return super().get_command(ctx, cmd_name)
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.startswith("aorta.hw_queue_eval"):
                raise
            return _install_hint_command(cmd_name)


@click.group(
    cls=_BenchGroup,
    lazy_commands={
        # hw_queue_eval (and transitively torch) is imported only when the
        # subcommand is actually invoked.
        "hw_queue_eval": LazyCommand(
            "aorta.hw_queue_eval.cli:cli",
            "Hardware queue evaluation harness (needs the hw-queue extra).",
        ),
    },
)
def bench() -> None:
    """Micro-benchmarks for perf characterization (hw_queue_eval)."""
