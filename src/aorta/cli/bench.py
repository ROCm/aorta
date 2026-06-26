"""``aorta bench`` — perf-characterization micro-benchmarks."""

from __future__ import annotations

import click

_INSTALL_HINT = (
    "'aorta bench hw_queue_eval' requires the hw-queue extra.\n"
    "Install it with:  pip install 'amd-aorta[hw-queue]'"
)


def _load_hw_queue_cli() -> click.Group | None:
    """Return the hw_queue_eval CLI group, or None if the [hw-queue] extra is absent.

    Raises any ImportError that is NOT caused by a missing optional dependency
    (e.g. a syntax error or broken internal import inside hw_queue_eval) so
    real bugs are not silently swallowed.
    """
    try:
        from aorta.hw_queue_eval.cli import cli  # noqa: PLC0415

        return cli
    except ModuleNotFoundError:
        # A missing module means an optional dependency (torch, numpy, …) is
        # absent — treat as "extra not installed" and return None.
        return None
    # Any other ImportError (bad internal import, syntax error, …) propagates.


class _LazyHwQueueGroup(click.Group):
    """Lazy proxy for the hw_queue_eval Click group.

    Defers the hw_queue_eval import (and transitively torch) until the user
    actually invokes ``aorta bench hw_queue_eval``. Group-level metadata
    (help text, params like ``--version``) is forwarded from the inner group
    so the surface matches ``python -m aorta.hw_queue_eval`` exactly.
    When the extra is absent, every entry point shows a clear install hint.
    """

    def _inner(self) -> click.Group | None:
        return _load_hw_queue_cli()

    def _require(self) -> click.Group:
        inner = self._inner()
        if inner is None:
            raise click.ClickException(_INSTALL_HINT)
        return inner

    @property
    def params(self) -> list[click.Parameter]:  # type: ignore[override]
        inner = self._inner()
        return inner.params if inner is not None else super().params

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        inner = self._inner()
        if inner is not None:
            inner.format_help(ctx, formatter)
        else:
            # Show the install hint instead of an empty proxy help page so
            # `aorta bench hw_queue_eval --help` is useful on a base install.
            with formatter.section("Error"):
                formatter.write_text(_INSTALL_HINT)

    def list_commands(self, ctx: click.Context) -> list[str]:
        inner = self._inner()
        return inner.list_commands(ctx) if inner is not None else []

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.BaseCommand | None:
        return self._require().get_command(ctx, cmd_name)

    def invoke(self, ctx: click.Context) -> None:
        # Ensures the install hint fires on bare `aorta bench hw_queue_eval`
        # (no subcommand) rather than Click's generic "Missing command" error.
        self._require()
        super().invoke(ctx)


@click.group()
def bench() -> None:
    """Micro-benchmarks for perf characterization (hw_queue_eval)."""


# Lazy proxy: hw_queue_eval imported only when the subcommand is actually invoked.
bench.add_command(_LazyHwQueueGroup(name="hw_queue_eval"), name="hw_queue_eval")
