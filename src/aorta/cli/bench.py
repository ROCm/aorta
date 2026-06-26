"""``aorta bench`` — perf-characterization micro-benchmarks."""

import click

try:
    from aorta.hw_queue_eval.cli import cli as hw_queue_eval_cli

    _hw_queue_available = True
except ImportError:
    _hw_queue_available = False


@click.group()
def bench() -> None:
    """Micro-benchmarks for perf characterization (hw_queue_eval)."""


if _hw_queue_available:
    # Mount the existing hw_queue_eval group verbatim. No logic added here.
    bench.add_command(hw_queue_eval_cli, name="hw_queue_eval")
else:

    @bench.command(name="hw_queue_eval")
    def _hw_queue_unavailable() -> None:
        """Hardware queue evaluation (requires amd-aorta[hw-queue])."""
        raise click.ClickException(
            "'aorta bench hw_queue_eval' requires the hw-queue extra.\n"
            "Install it with:  pip install 'amd-aorta[hw-queue]'"
        )
