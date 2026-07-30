"""``aorta env`` - thin CLI wrapper around :func:`collect_env`.

The library function in :mod:`aorta.instrumentation.environment` does all
the probing; this module only handles arg parsing, dispatch between
output modes (full snapshot, brief summary, or one-field lookup), and
writing the JSON snapshot to disk. Per #147 acceptance: this file does
no probing of its own.

Subcommands:

* ``probe``  -- capture trial-environment state to env.json (#147,
  extended by A1.2a/b for Buck2 environments and PR #177 for source-
  install introspection + ninja_hipcc legacy-FindHIP fallback).
* ``recipe`` -- emit a best-effort build-recipe fragment from an
  existing env.json (#163, A1.2c). Currently supports
  ``--format buck``; ``--format dockerfile`` is a placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

# --execution-context choices. Mirrors EXECUTION_CONTEXT_INVOCATIONS in
# aorta.instrumentation.environment (the canonical source). We hard-code the
# literal here rather than importing it, so that `aorta env --help` / CLI
# startup does NOT eagerly import the large environment probing module (it
# is imported lazily inside probe() only when the command actually runs).
# tests/instrumentation/test_environment.py asserts these stay in sync.
_EXECUTION_CONTEXT_CHOICES = ["direct", "buck2_run", "buck2_action"]


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
    help="Path to write env.json (ignored with --summary / --field).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="After the brief, also print the full snapshot JSON to stdout.",
)
@click.option(
    "--summary",
    is_flag=True,
    help=(
        "Print only the one-screen brief and exit (skip JSON write). "
        "Use for quick eyeballing without producing an artifact."
    ),
)
@click.option(
    "--field",
    "field_path",
    type=str,
    default=None,
    metavar="DOTTED.PATH",
    help=(
        "Print one snapshot field as JSON and exit (skip file write). "
        "Example: --field pytorch_build.ninja_hipcc.targets.ck_sdpa."
        "use_defines_present.USE_ROCM_CK_SDPA. For keys containing "
        "'.' (e.g. 'libaotriton_v2.so'), use jq on a full snapshot."
    ),
)
@click.option(
    "--extended",
    is_flag=True,
    help=(
        "Keep the full per-file kernel-catalog lists (tensile_catalog / "
        "miopen_catalog menu.files) in the JSON. Default (compact) drops "
        "them -- they dominate env.json size -- but keeps every count and "
        "content hash, so a diff still detects a changed catalog; "
        "--extended localizes which file changed."
    ),
)
@click.option(
    "--buck-target",
    type=str,
    default=None,
    help=(
        "Buck2 label to introspect for library identity (issue #163, "
        "A1.2b). When given, the snapshot's library_introspection list "
        "is populated from bound `buck2 cquery 'deps(%s)' <label> --json` (each "
        "matched entry carries both a stripped `target` and the raw "
        "`configured_target`, schema 1.6). If buck2 isn't on PATH (or "
        "the cquery otherwise fails), the library_introspection list "
        "stays empty and a human-readable reason is recorded in "
        "`partial_reasons`."
    ),
)
@click.option(
    "--buck-mode-file",
    "buck_mode_files",
    type=str,
    multiple=True,
    metavar="PATH",
    help=(
        "Buck mode/flag file to apply to the cquery (repeatable, ordered). "
        "Pass a Buck cell path such as root//mode/debug; a leading '@' is "
        "accepted but not required."
    ),
)
@click.option(
    "--buck-config",
    "buck_configs",
    type=str,
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Buck -c config override for the cquery (repeatable, ordered). "
        "Only each key and an aggregate context fingerprint are persisted; "
        "raw values are never written to env.json."
    ),
)
@click.option(
    "--buck-modifier",
    "buck_modifiers",
    type=str,
    multiple=True,
    metavar="MODIFIER",
    help="Buck -m configuration modifier for the cquery (repeatable, ordered).",
)
@click.option(
    "--buck-default-context",
    is_flag=True,
    help=(
        "Confirm that --buck-target should use Buck's default invocation "
        "context. Mutually exclusive with --buck-mode-file, --buck-config, "
        "and --buck-modifier."
    ),
)
@click.option(
    "--buck-timeout",
    # ``IntRange(min=1)`` rejects 0 / negative values at arg-parse time
    # rather than letting them flow into ``subprocess.run(timeout=...)``
    # where they raise ``ValueError`` and confuse the never-raises
    # fallback path. Per Copilot review on PR #165.
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Per-call timeout (seconds, must be >= 1) for `buck2 cquery 'deps(...)'`.",
)
@click.option(
    "--execution-context",
    type=click.Choice(_EXECUTION_CONTEXT_CHOICES),
    default="direct",
    show_default=True,
    help=(
        "Self-declared label for HOW the probe was launched, stamped into "
        "execution_context.probe_invocation. Use buck2_action when running "
        "the probe as a Buck2 action/genrule so it captures the executor's "
        "environment. Captures nothing new -- but if you claim a "
        "container/RE capture (non-'direct') and no isolation is detected "
        "AND neither $AORTA_RE_IMAGE nor $AORTA_DOCKER_IMAGE is set, a "
        "loud warning is printed to stderr (you may have probed the wrong "
        "place)."
    ),
)
def probe(
    output: Path,
    verbose: bool,
    summary: bool,
    field_path: str | None,
    extended: bool,
    buck_target: str | None,
    buck_mode_files: tuple[str, ...],
    buck_configs: tuple[str, ...],
    buck_modifiers: tuple[str, ...],
    buck_default_context: bool,
    buck_timeout: int,
    execution_context: str,
) -> None:
    """Capture trial-environment state to env.json (issue #147)."""
    from aorta.instrumentation.buck_invocation import BuckInvocationContext
    from aorta.instrumentation.environment import (
        collect_env,
        execution_context_warning,
    )

    # --summary and --field both bypass the file write -- only one
    # output mode at a time makes sense.
    if summary and field_path is not None:
        raise click.ClickException("--summary and --field are mutually exclusive")

    explicit_buck_context = bool(buck_mode_files or buck_configs or buck_modifiers)
    if (explicit_buck_context or buck_default_context) and buck_target is None:
        raise click.UsageError(
            "--buck-mode-file, --buck-config, --buck-modifier, and "
            "--buck-default-context require --buck-target"
        )
    if buck_default_context and explicit_buck_context:
        raise click.UsageError(
            "--buck-default-context is mutually exclusive with "
            "--buck-mode-file, --buck-config, and --buck-modifier"
        )
    try:
        buck_context = (
            BuckInvocationContext(
                mode_files=buck_mode_files,
                config_overrides=buck_configs,
                modifiers=buck_modifiers,
                default_context_confirmed=buck_default_context,
            )
            if buck_target is not None
            else None
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="Buck context options") from exc

    # Capture once; both short-circuit modes and the default mode read
    # from this single snapshot. Buck-related kwargs flow through to
    # collect_env() regardless of output mode -- the buck audit cost
    # is paid only when --buck-target is set. detail="full" only when
    # --extended: keeps the heavy per-file catalog lists in the snapshot.
    snapshot = collect_env(
        buck_target=buck_target,
        buck_timeout=buck_timeout,
        buck_context=buck_context,
        detail="full" if extended else "compact",
        probe_invocation=execution_context,
    )
    snapshot_dict = snapshot.to_dict()

    # Validation guardrail (shared predicate with _probe_main): if the caller
    # claims a container/RE capture but we saw zero isolation signal and no
    # launcher image env var, they may have probed the host shell instead of
    # the workload's context. Warn loudly to stderr (never a hard error --
    # the snapshot is still valid and written).
    _ec_warning = execution_context_warning(execution_context, snapshot.container_detected)
    if _ec_warning is not None:
        click.echo(_ec_warning, err=True)

    # --field: print one value as JSON and exit. Skips the file write
    # entirely; pair with `jq` / `xargs` for scripting.
    if field_path is not None:
        value = _lookup_field(snapshot_dict, field_path)
        # Compact JSON so a scalar (bool / int / str) prints as one
        # line with the value's type preserved. `default=str` would
        # mask a non-serializable leak; we want loud failure instead.
        click.echo(json.dumps(value))
        return

    # --summary: print only the brief + partial reasons. Skips the
    # file write -- pair with `aorta env probe -o env.json` separately
    # when you need both.
    if summary:
        click.echo(snapshot.summary())
        if snapshot.partial:
            click.echo("\nPartial reasons:")
            for reason in snapshot.partial_reasons:
                click.echo(f"  - {reason}")
        return

    # Default mode: write the JSON artifact AND print the brief. The
    # full snapshot stays the single source of truth for downstream
    # diffing; the brief is courtesy stdout for the operator who just
    # ran the command.
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

    # NOTE: deliberately not passing ``default=str`` -- the snapshot is
    # supposed to be JSON-native (str/int/bool/None/list/dict). If a
    # non-serializable type sneaks in (e.g. a Path or datetime), we want
    # the failure to be loud rather than silently stringified.
    #
    # ``encoding="utf-8"`` is set explicitly to stay symmetric with the
    # ``recipe`` reader (which also forces utf-8). Without it,
    # ``write_text`` would use the platform default (e.g. cp1252 on
    # some Windows locales / containers), which could produce a file
    # that ``aorta env recipe`` later refuses to decode. Per Copilot
    # round-1 review on PR #181.
    try:
        output.write_text(
            json.dumps(snapshot_dict, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise click.ClickException(f"Failed to write env probe to {output}: {exc}") from exc

    partial = " [PARTIAL]" if snapshot.partial else ""
    click.echo(
        f"Wrote env probe to {output} " f"(schema_version={snapshot.schema_version}){partial}"
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
        click.echo(f"\n[PARTIAL, {len(snapshot.partial_reasons)} reason(s)]")
    else:
        click.echo("\n[OK]")


def _lookup_field(snapshot_dict: dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted-path into a nested dict, ClickException on miss.

    Walks ``snapshot_dict[a][b][c]`` for path ``"a.b.c"``. Surfaces a
    helpful error when:

    * A segment is not present (lists the keys actually available at
      that level, capped at ~10 so the message fits one screen).
    * A non-dict is encountered mid-path (e.g. the user tried to
      descend into a list or a scalar leaf).

    Limitation: dotted-path notation cannot reference keys that
    themselves contain a ``.``. The only such key in the current
    schema is ``"libaotriton_v2.so"`` under
    ``pytorch_build.binary_introspection.torch_lib_bundled``; reach it
    via ``jq`` on a full snapshot.
    """
    if not dotted_path:
        raise click.ClickException("--field path must be non-empty")
    parts = dotted_path.split(".")
    cur: Any = snapshot_dict
    for i, part in enumerate(parts):
        prefix = ".".join(parts[:i]) or "<root>"
        if cur is None:
            raise click.ClickException(f"Cannot descend into '{part}' at '{prefix}': value is null")
        if not isinstance(cur, dict):
            raise click.ClickException(
                f"Cannot descend into '{part}' at '{prefix}': "
                f"value is {type(cur).__name__}, not an object"
            )
        if part not in cur:
            available = sorted(cur.keys())
            shown = ", ".join(available[:10])
            more = f" (+ {len(available) - 10} more)" if len(available) > 10 else ""
            raise click.ClickException(
                f"Key '{part}' not found at '{prefix}'. " f"Available keys: {shown}{more}"
            )
        cur = cur[part]
    return cur


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
        # ``read_text()`` defaults to the platform encoding (usually
        # UTF-8 on Linux). A snapshot written on a host with a stricter
        # encoding (or a hand-edited file with stray bytes) can raise
        # ``UnicodeDecodeError`` -- catch that alongside the obvious
        # OSError/JSONDecodeError. Per Copilot review on PR #178.
        env_dict = json.loads(env_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Wrap filesystem, decode, and JSON-parse errors as a single
        # Click error so the operator sees a clean one-liner instead
        # of a Python traceback.
        raise click.ClickException(f"Failed to read env.json from {env_json}: {exc}") from exc

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
