"""``aorta bundle`` -- thin Click shim around :func:`aorta.bundle.bundle_run_dir`.

Mirrors the discipline tested in ``tests/run/test_cli_parsing.py`` and
``tests/probe/test_cli_parsing.py``: this handler does no
orchestration. It validates inputs, builds a ``review_callback``
when ``--review`` is set, calls :func:`bundle_run_dir`, and maps
:class:`aorta.bundle.errors.BundleError` subclasses to a
:class:`click.ClickException`. The acceptance criteria from issue
#196 are split between the writer (the actual bundling) and this
shim (CLI surface, exit codes, ``--review`` interactive pause).

The handler is intentionally short -- under the same ~60-line
ceiling the ``aorta probe`` handler keeps -- so reviewers can
audit it without scrolling.
"""

from __future__ import annotations

from pathlib import Path

import click

from aorta.bundle import (
    BundleAbortedError,
    BundleError,
    Manifest,
    bundle_run_dir,
)


def _render_review_summary(manifest: Manifest) -> str:
    """Build the ``--review`` summary text shown right before the prompt.

    Renders a short header (ticket, source dir, redactor, totals) plus
    a per-file-count table. Per-file rows are truncated to the first
    ten files when there are more than twelve total -- the manifest
    JSON is always written to disk for full inspection, the
    review pause is just an "is this what you meant?" sanity
    check. The truncated-rows note tells the operator how to see the
    rest if they need to.
    """
    lines: list[str] = []
    lines.append("aorta bundle: review pause")
    lines.append(f"  ticket            : {manifest.ticket}")
    lines.append(f"  source run dir    : {manifest.source_run_dir}")
    lines.append(f"  redactor          : {manifest.redactor_kind}")
    lines.append(f"  redaction applied : {manifest.redaction_applied}")
    lines.append(f"  files             : {len(manifest.files)}")
    lines.append(f"  total bytes in    : {manifest.total_bytes_in()}")
    lines.append(f"  total bytes out   : {manifest.total_bytes_out()}")
    lines.append("")
    lines.append("  per-file counts (env / paths / ips, bytes_in -> bytes_out):")
    rows = manifest.files
    truncated = len(rows) > 12
    shown = rows[:10] if truncated else rows
    for f in shown:
        lines.append(
            f"    {f.path}: "
            f"{f.env_keys_removed}/{f.paths_rewritten}/{f.ips_rewritten} "
            f"({f.bytes_in} -> {f.bytes_out})"
        )
    if truncated:
        lines.append(f"    ... ({len(rows) - 10} more; full set in manifest.json)")
    return "\n".join(lines)


@click.command(name="bundle")
@click.argument(
    "run_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--ticket",
    default=None,
    help=(
        "Bundle ticket. When omitted, inferred from the basename of "
        "<run-dir>. Required when the basename is '_no_ticket_' "
        "(otherwise the bundle has no routing target downstream)."
    ),
)
@click.option(
    "--review",
    is_flag=True,
    help=(
        "Print the manifest summary and pause for [y/N] confirmation "
        "before writing the tarball. Answering 'n' aborts the bundle "
        "with exit 1 and writes nothing."
    ),
)
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Where to write the tarball. Default: "
        "'<safe-slug(ticket)>-<UTC-timestamp>.tar.gz' in the current "
        "directory (the ticket is slugified for filesystem safety, so "
        "spaces/slashes become underscores). If PATH is an EXISTING directory the "
        "default filename is dropped inside it; otherwise PATH is "
        "treated verbatim as the tarball filename (the parent is "
        "created if missing). 'aorta bundle ./bundles' will write a "
        "file literally named 'bundles' unless that directory already "
        "exists -- create it first with mkdir if you want directory "
        "semantics."
    ),
)
@click.option(
    "--redaction-from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Recipe whose 'redaction:' block governs scrubber behaviour. "
        "NOTE: until 'aorta probe' Phase 3 (issue #188) ships "
        "'aorta.probe.redaction', this flag is logged but not yet "
        "consumed -- bundles run with the IdentityRedactor (no "
        "scrubbing, zero per-file counts). Phase 3 will also add "
        "the auto-fallback to '<run-dir>/recipe.resolved.yaml' "
        "when this flag is omitted; today an explicit path is "
        "required to log a redaction-from line at all."
    ),
)
def bundle(
    run_dir: Path,
    ticket: str | None,
    review: bool,
    output: Path | None,
    redaction_from: Path | None,
) -> None:
    """Package an ``aorta probe`` run directory into a redacted tarball.

    Issue tracker: https://github.com/ROCm/aorta/issues/196 .
    Design + manifest schema: ``docs/probe-188/bundle.md``.
    """
    review_callback = (lambda manifest: _prompt_review(manifest)) if review else None
    try:
        path = bundle_run_dir(
            run_dir,
            ticket=ticket,
            output=output,
            redaction_from=redaction_from,
            review_callback=review_callback,
        )
    except BundleAbortedError as exc:
        # Operator answered 'n' at the review pause. Distinct from
        # other BundleErrors because we want exit 1 (per issue #196
        # acceptance criterion 3) without a "ClickException" prefix
        # cluttering the abort message.
        click.echo(str(exc), err=True)
        raise click.exceptions.Exit(1) from exc
    except BundleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote bundle to {path}")


def _prompt_review(manifest: Manifest) -> bool:
    """Show the manifest summary and prompt for ``[y/N]``.

    Used as the ``review_callback`` argument to :func:`bundle_run_dir`
    when ``--review`` is set. Returning ``False`` makes the writer
    raise :class:`BundleAbortedError`, which the CLI handler maps
    to ``Exit(1)``.

    ``click.confirm(default=False)`` matches the issue #196
    contract -- a bare press of ``Enter`` (or any non-``y`` answer)
    aborts. Tests inject the answer through the invoke call, e.g.
    ``runner.invoke(cli, [...], input="y\\n")`` /
    ``input="n\\n"`` (the ``input=`` kwarg belongs to ``invoke``,
    not the ``CliRunner`` constructor).
    """
    click.echo(_render_review_summary(manifest))
    click.echo("")
    return click.confirm("Write the bundle?", default=False)


__all__ = ["bundle"]
