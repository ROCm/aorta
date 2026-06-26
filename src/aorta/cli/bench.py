"""``aorta bench`` — perf-characterization micro-benchmarks."""

from __future__ import annotations

import click


class _LazyHwQueueGroup(click.Group):
    """Thin lazy wrapper: imports hw_queue_eval only when the subcommand is invoked.

    This avoids pulling ``torch`` (via hw_queue_eval.__init__) on every
    ``aorta`` command even when the [hw-queue] extra is installed.
    """

    def _load(self) -> click.Group | None:
        try:
            from aorta.hw_queue_eval.cli import cli  # noqa: PLC0415

            return cli
        except ImportError:
            return None

    def list_commands(self, ctx: click.Context) -> list[str]:
        inner = self._load()
        return inner.list_commands(ctx) if inner is not None else []

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.BaseCommand | None:
        inner = self._load()
        if inner is None:
            raise click.ClickException(
                "'aorta bench hw_queue_eval' requires the hw-queue extra.\n"
                "Install it with:  pip install 'amd-aorta[hw-queue]'"
            )
        return inner.get_command(ctx, cmd_name)


@click.group()
def bench() -> None:
    """Micro-benchmarks for perf characterization (hw_queue_eval)."""


# Mount as a lazy group: hw_queue_eval is imported only when the user
# actually invokes `aorta bench hw_queue_eval`, not on every aorta command.
bench.add_command(_LazyHwQueueGroup(name="hw_queue_eval"), name="hw_queue_eval")
