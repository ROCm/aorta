"""``aorta env`` - thin CLI wrapper around :func:`collect_env`.

The library function in :mod:`aorta.instrumentation.environment` does all
the probing; this module only handles arg parsing and writing the JSON
snapshot to disk. Per #147 acceptance: this file does no probing of its
own and stays under ~30 lines of substantive code per subcommand
(excluding the docstring above).

Subcommands:

* ``probe``  -- capture trial-environment state to env.json (#147,
  extended by A1.2a/b for Buck2 environments).
* ``recipe`` -- emit a best-effort build-recipe fragment from an
  existing env.json (#163, A1.2c). Currently supports
  ``--format buck``; ``--format dockerfile`` is a placeholder.
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


# Supported --format values for ``aorta env recipe``. ``buck`` is the
# A1.2c deliverable; ``dockerfile`` is a placeholder so the CLI surface
# doesn't lock us in (per the issue spec's acceptance criterion).
# Future formats slot in here and below in ``recipe()``.
_RECIPE_FORMATS = ("buck", "dockerfile")


@env.command()
@click.argument(
    "env_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--format",
    "fmt",  # ``format`` is a builtin; rename to dodge shadowing in the body.
    type=click.Choice(_RECIPE_FORMATS),
    required=True,
    help=(
        "Output format. `buck` emits a BUCK file fragment (#163, A1.2c). "
        "`dockerfile` is a placeholder -- see help text for `--format "
        "dockerfile` for the current status."
    ),
)
def recipe(env_json: Path, fmt: str) -> None:
    """Emit a best-effort build-recipe fragment from an env.json snapshot.

    Reads ``ENV_JSON`` produced by ``aorta env probe`` and writes the
    emitted fragment to stdout. Output is BEST-EFFORT, NOT EXACT --
    env.json captures observed state, not a complete build recipe.

    Per the A1.2c acceptance criteria (issue #163):

    * Output for ``--format buck`` starts with a loud "BEST-EFFORT, NOT
      EXACT" header comment block.
    * Each ``library_introspection`` entry with ``source == "buck"``
      becomes one ``prebuilt_cxx_library(...)`` rule pinning the
      captured revision.
    * ``--format dockerfile`` exits with a clear "not yet implemented"
      error message (placeholder so the CLI surface doesn't lock us in
      to a Buck-only world).
    """
    try:
        env_dict = json.loads(env_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        # Wrap both filesystem and JSON-parse errors as a single
        # Click error so the operator sees a clean one-liner instead
        # of a Python traceback.
        raise click.ClickException(
            f"Failed to read env.json from {env_json}: {exc}"
        ) from exc

    if not isinstance(env_dict, dict):
        raise click.ClickException(
            f"{env_json} is not a JSON object (parsed as "
            f"{type(env_dict).__name__}); expected an env.json snapshot."
        )

    if fmt == "buck":
        from aorta.instrumentation.recipes import emit_buck_recipe

        click.echo(emit_buck_recipe(env_dict), nl=False)
        return

    if fmt == "dockerfile":
        # Placeholder per the acceptance criterion. Tracked separately;
        # don't generate something half-shaped that locks the surface.
        raise click.ClickException(
            "--format dockerfile is not yet implemented. The CLI surface "
            "is reserved for it; file a follow-up issue if you need it."
        )

    # ``click.Choice`` should make this unreachable; the explicit raise
    # is a defence-in-depth against the choice list and the dispatch
    # block drifting out of sync.
    raise click.ClickException(  # pragma: no cover -- click.Choice guards
        f"Unknown --format {fmt!r}; expected one of {_RECIPE_FORMATS}."
    )
