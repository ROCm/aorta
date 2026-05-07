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
    # NOTE: deliberately not setting ``writable=True`` -- Click validates
    # that during arg parsing, which fails for ``-o newdir/env.json``
    # *before* we get a chance to ``mkdir`` the parent. We create the
    # parent ourselves and let the actual write surface any I/O error.
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("env.json"),
    show_default=True,
    help="Path to write env.json.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="After the brief, also print the full snapshot JSON to stdout.",
)
@click.option(
    "--buck-target",
    type=str,
    default=None,
    help=(
        "Buck2 label to introspect for library identity (issue #163, "
        "A1.2b). When given, the snapshot's library_introspection list "
        "is populated from `buck2 audit dependencies <label> --transitive "
        "--json`. Ignored if buck2 isn't on PATH."
    ),
)
@click.option(
    "--buck-timeout",
    type=int,
    default=10,
    show_default=True,
    help="Per-call timeout (seconds) for `buck2 audit dependencies`.",
)
def probe(
    output: Path, verbose: bool, buck_target: str | None, buck_timeout: int
) -> None:
    """Capture trial-environment state to env.json (issue #147)."""
    from aorta.instrumentation.environment import collect_env

    output = output.expanduser().resolve()
    # Wrap the two filesystem ops so common operator errors (unwritable
    # parent, read-only mount, full disk, etc.) surface as a clean
    # one-line Click error instead of a Python traceback. collect_env()
    # itself never raises, so it stays outside the try blocks.
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise click.ClickException(
            f"Failed to create parent directory for {output}: {exc}"
        ) from exc

    snapshot = collect_env(buck_target=buck_target, buck_timeout=buck_timeout)

    # NOTE: deliberately not passing ``default=str`` -- the snapshot is
    # supposed to be JSON-native (str/int/bool/None/list/dict). If a
    # non-serializable type sneaks in (e.g. a Path or datetime), we want
    # the failure to be loud rather than silently stringified.
    snapshot_dict = snapshot.to_dict()
    try:
        output.write_text(json.dumps(snapshot_dict, indent=2))
    except OSError as exc:
        raise click.ClickException(
            f"Failed to write env probe to {output}: {exc}"
        ) from exc

    partial = " [PARTIAL]" if snapshot.partial else ""
    click.echo(
        f"Wrote env probe to {output} "
        f"(schema_version={snapshot.schema_version}){partial}"
    )
    click.echo(snapshot.summary())

    # Always inline the partial_reasons -- they are action items and
    # forcing the operator to ``jq env.json`` to read them is friction.
    # Costs nothing (the list is already in memory).
    if snapshot.partial:
        click.echo("\nPartial reasons:")
        for reason in snapshot.partial_reasons:
            click.echo(f"  - {reason}")

    if verbose:
        click.echo("\n--- Full snapshot ---")
        click.echo(json.dumps(snapshot_dict, indent=2))

    # Closing marker -- repeats the [PARTIAL] state at end-of-output so
    # an operator scrolled to the bottom (especially after --verbose
    # dumps the full JSON) immediately sees probe status without
    # scrolling back up. Matches the marker shown next to the "Wrote
    # env probe..." line.
    if snapshot.partial:
        click.echo(
            f"\n[PARTIAL, {len(snapshot.partial_reasons)} reason(s)]"
        )
    else:
        click.echo("\n[OK]")
