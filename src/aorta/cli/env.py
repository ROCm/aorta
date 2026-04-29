"""``aorta env`` - thin CLI wrapper around :func:`collect_env`.

The library function in :mod:`aorta.instrumentation.environment` does all
the probing; this module only handles arg parsing and writing the JSON
snapshot to disk. Per #147 acceptance: this file does no probing of its
own and stays under ~30 lines of substantive code (excluding the
docstring above).
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def env() -> None:
    """Capture and compare GPU/library environment for trial reproducibility."""


@env.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=Path("env.json"),
    show_default=True,
    help="Path to write env.json.",
)
def probe(output: Path) -> None:
    """Capture trial-environment state to env.json (issue #147)."""
    from aorta.instrumentation.environment import collect_env

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = collect_env()
    output.write_text(json.dumps(snapshot.to_dict(), indent=2, default=str))
    partial = " [PARTIAL]" if snapshot.partial else ""
    click.echo(
        f"Wrote env probe to {output} "
        f"(schema_version={snapshot.schema_version}){partial}"
    )
    click.echo(snapshot.summary())
