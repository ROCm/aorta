"""`aorta env` - environment capture and comparison."""

import click


@click.group()
def env() -> None:
    """Capture and compare GPU/library environment for trial reproducibility."""


@env.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default="env.json",
    show_default=True,
    help="Path to write env.json.",
)
def probe(output: str) -> None:
    """Capture current GPU + library + env-var state to env.json.

    Implemented by task A1 (instrumentation/environment.py).
    """
    raise click.ClickException("aorta env probe - not yet implemented (task A1)")
