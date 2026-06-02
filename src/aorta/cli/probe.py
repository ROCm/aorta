"""``aorta probe`` -- wrap-and-collect command for opaque user launch commands.

Thin Click shim per the rubric (FR 1.15: handler body <= 60 lines, no
orchestration). The CLI's whole job is:

1. Validate the trailing argv and the env-passthrough mode.
2. Load the recipe (must be ``mode: probe``).
3. Override ``recipe.probe_extras.env_passthrough_mode`` from the CLI
   flag **only when the user actually passed it** (FR 1.10: the flag
   wins when present, otherwise the recipe's ``env_passthrough_mode:``
   takes effect; if neither is set the recipe-builder default
   ``"inherit"`` applies).
4. Call :func:`aorta.triage.runner.run_recipe` with the probe-mode
   knobs (``layout="flat_resume"``, ``resume_existing=True``,
   ``subprocess_argv=...``).
5. Map recipe-load and runner exceptions to ``ClickException``.
   ``LookupError`` is in the catch list so that
   ``UnknownMitigationError`` / ``UnknownEnvironmentError`` (KeyError
   subclasses, not ``RegistryError``) bubble up as a clean
   ClickException -- the most common failure path when an operator
   forgets ``--mitigations-file`` for a recipe that references a
   sidecar-only name. Mirrors ``aorta.cli.run``.

The shared-engine test in ``tests/probe/test_shared_engine.py`` mocks
``run_recipe`` and invokes both ``aorta probe`` and ``aorta triage
run`` -- both must reach the mock.

Phase 2 adds two short-circuit flags:

* ``--list-patterns`` -- print the built-in Tier-4 pattern catalogue
  and exit 0 without loading any recipe. This is the rubric's
  ``aorta probe list-patterns`` subcommand expressed as a flag so the
  Phase 1 CLI surface (``aorta probe -- <argv>``) stays
  byte-equivalent. See the PR description for the rationale.
* ``--version`` (paired with ``--list-patterns``) -- print
  ``aorta probe pattern library v<N> (aorta <pkg-version>)`` and
  exit.
"""

from __future__ import annotations

from pathlib import Path

import click

from aorta.probe.classifier.tier4_patterns import (
    BUILTIN_PATTERN_VERSION,
    all_patterns,
)
from aorta.probe.cli_helpers import (
    ProbeUsageError,
    apply_recipe_overrides,
    help_token_in_option_zone,
    parse_env_passthrough_mode,
    reject_flag_shaped_value,
    require_double_dash_separator,
    validate_trailing_argv,
)
from aorta.probe.recipe_builder import ProbeExtras
from aorta.registry import RegistryError
from aorta.run.cli_helpers import configure_verbose_logging
from aorta.triage.output import RunDirLockedError
from aorta.triage.recipe import (
    RecipeCellError,
    RecipeSchemaError,
    load_recipe,
)
from aorta.triage.runner import run_recipe


def _aorta_package_version() -> str:
    """Return the installed ``aorta`` package version (best-effort).

    ``importlib.metadata.version`` is the supported API; falls back
    to ``"unknown"`` only when the package metadata is missing
    (editable install on a Python that doesn't expose dist-info,
    very rare).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("aorta")
    except PackageNotFoundError:
        return "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _print_list_patterns(show_version: bool) -> None:
    """Render the Tier-4 catalogue or the version banner (rubric FR 2.5).

    Plain stdout writes (no JSON) so the output is greppable from a
    shell. Each pattern emits a stable three-line entry: ID,
    description, sample regex match line.
    """
    if show_version:
        click.echo(
            f"aorta probe pattern library v{BUILTIN_PATTERN_VERSION} "
            f"(aorta {_aorta_package_version()})"
        )
        return
    click.echo(f"aorta probe built-in pattern library (v{BUILTIN_PATTERN_VERSION})")
    click.echo("")
    for pattern in all_patterns():
        click.echo(f"  {pattern.detector_id}")
        click.echo(f"      description: {pattern.description}")
        click.echo(f"      regex     : {pattern.regex.pattern}")
        click.echo(f"      sample    : {pattern.sample}")
        click.echo("")


def _reject_bare_version_flag() -> None:
    """Raise a targeted ``click.UsageError`` for bare ``--version``.

    The flag is documented as meaningful only when paired with
    ``--list-patterns`` (it prints the built-in pattern-library
    version next to the aorta package version). Without
    ``--list-patterns`` it has no semantics; silently falling
    through to the ``--recipe`` check would surface a confusing
    "Missing option '--recipe'" error, suggesting the operator
    needs a recipe just to see a version string. Reject up-front
    with a targeted usage message that names the intended pairing.
    """
    raise click.UsageError(
        "--version is only meaningful with --list-patterns. "
        "Run 'aorta probe --list-patterns --version' to print the "
        "pattern-library and package versions, or drop --version "
        "to run a probe trial."
    )


def _reject_flag_shaped_callback(
    ctx: click.Context, param: click.Parameter, value: object
) -> object:
    """Click callback wiring :func:`reject_flag_shaped_value` onto string options.

    Translates :class:`ProbeUsageError` into :class:`click.BadParameter`
    so the standard Click usage rendering kicks in (with the option name
    + usage hint).
    """
    if value is None or not isinstance(value, (str, Path)):
        return value
    option_name = f"--{param.name.replace('_', '-')}" if param.name else "<option>"
    try:
        reject_flag_shaped_value(option_name, str(value))
    except ProbeUsageError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc
    return value


class _ProbeCommand(click.Command):
    """``probe`` command that requires an explicit ``--`` separator in raw argv.

    Implemented as a ``parse_args`` override so the check sees the same
    argv that Click sees -- whether that's ``sys.argv[1:]`` from the real
    entry point or the ``args=[...]`` list from ``CliRunner.invoke`` in
    tests. ``--help``/``-h`` and ``--list-patterns`` short-circuit the
    check, but ONLY when the bypass token sits in the aorta-option
    zone (before the user-command boundary). A naive ``"--help" in
    args`` bypass is wrong because ``aorta probe --recipe r --output o
    echo --help`` would silently skip the separator check -- ``--help``
    is the user-command's flag there, not aorta's. See
    :func:`help_token_in_option_zone` for the scoping rule.

    ``--list-patterns`` is grouped with the help bypass because it
    also prints info and exits without consuming a user command --
    the rubric's ``aorta probe list-patterns`` shape, expressed as
    a flag to preserve the Phase 1 CLI byte-equivalently.

    ``--version`` similarly short-circuits: it is documented as only
    meaningful when paired with ``--list-patterns``, and the body
    rejects ``--version`` alone with a targeted usage message. Without
    this short-circuit, ``aorta probe --version`` would die at the
    ``--`` separator check first and the operator would never see
    the more helpful "use --list-patterns" hint.
    """

    _BYPASS_TOKENS: frozenset[str] = frozenset({"--help", "-h", "--list-patterns", "--version"})

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not ctx.resilient_parsing and not help_token_in_option_zone(
            args, self._value_taking_option_tokens(), self._BYPASS_TOKENS
        ):
            try:
                require_double_dash_separator(args)
            except ProbeUsageError as exc:
                raise click.UsageError(str(exc), ctx=ctx) from exc
        return super().parse_args(ctx, args)

    def _value_taking_option_tokens(self) -> frozenset[str]:
        """Long-form option tokens (``--recipe`` etc.) that consume a value.

        Derived from the Click command's own ``params`` so adding a new
        option that takes a value (without ``is_flag``/``count``) won't
        silently re-open the bot-flagged misparse. Boolean flags and
        ``count`` options are excluded -- they don't consume the next
        token.
        """
        tokens: set[str] = set()
        for param in self.params:
            if not isinstance(param, click.Option):
                continue
            if param.is_flag or param.count:
                continue
            for opt in param.opts:
                if opt.startswith("--"):
                    tokens.add(opt)
        return frozenset(tokens)


@click.command(
    name="probe",
    cls=_ProbeCommand,
    context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False},
)
@click.option(
    "--list-patterns",
    "list_patterns",
    is_flag=True,
    help=(
        "Print the built-in Tier-4 pattern catalogue and exit. "
        "Combines with --version to print the pattern-library version "
        "and the aorta package version."
    ),
)
@click.option(
    "--version",
    "show_version",
    is_flag=True,
    help=(
        "With --list-patterns: print 'aorta probe pattern library "
        "v<N> (aorta <pkg-version>)' and exit. Rejected with a usage "
        "error when passed without --list-patterns (see "
        "_reject_bare_version_flag for the rationale)."
    ),
)
@click.option(
    "--recipe",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    callback=_reject_flag_shaped_callback,
    help="Path to a 'mode: probe' YAML or JSON recipe file.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("probe_results"),
    show_default=True,
    callback=_reject_flag_shaped_callback,
    help="Top-level output directory; <output>/<ticket>/<cell>/trial_<n>/ artifacts land here.",
)
@click.option(
    "--ticket",
    default=None,
    callback=_reject_flag_shaped_callback,
    help=(
        "Ticket ID for output-dir grouping. When omitted, output is grouped "
        "under '_no_ticket_/'. Required for 'aorta bundle' downstream (Phase 3)."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the recipe and print the planned cell list + argv without executing.",
)
@click.option(
    "--env-passthrough-mode",
    type=click.Choice(["inherit", "file"]),
    default=None,
    help=(
        "How per-cell env vars reach the user command: 'inherit' stamps "
        "them on os.environ in-process (the child Popen inherits); 'file' "
        "additionally writes a chmod-0600 probe.env file in the trial dir "
        "and exports AORTA_ENV_FILE for the user's argv to reference "
        "(e.g. 'docker run --env-file $AORTA_ENV_FILE ...'). "
        "Aorta never modifies the user's argv. "
        "When omitted, the recipe's 'env_passthrough_mode:' value is used "
        "(falling back to 'inherit' if the recipe also omits it). When "
        "present, this flag overrides the recipe."
    ),
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "JSON sidecar file with ad-hoc mitigations and/or environments "
        "(repeatable). Merged with built-ins at recipe load time."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Stream per-cell progress to stderr. -v = INFO, -vv = DEBUG.",
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def probe(
    list_patterns: bool,
    show_version: bool,
    recipe: Path | None,
    output: Path,
    ticket: str | None,
    dry_run: bool,
    env_passthrough_mode: str | None,
    mitigation_files: tuple[Path, ...],
    verbose: int,
    argv: tuple[str, ...],
) -> None:
    """Run an opaque user launch command across a mitigation x diagnostic matrix.

    All arguments after ``--`` are forwarded byte-for-byte to the user
    command. Aorta never parses them; the only "boundary" is the
    optional ``probe.env`` file written under ``--env-passthrough-mode file``.
    """
    if list_patterns:
        _print_list_patterns(show_version=show_version)
        return
    if show_version:
        # ``--version`` is only meaningful with ``--list-patterns``;
        # see :func:`_reject_bare_version_flag` for the rationale.
        _reject_bare_version_flag()

    configure_verbose_logging(verbose)
    if recipe is None:
        raise click.UsageError(
            "Missing option '--recipe'. Pass --recipe <path>, or run "
            "with --list-patterns to print the built-in pattern catalogue."
        )
    try:
        # ``--env-passthrough-mode`` defaults to None so the handler can
        # distinguish "user passed the flag" from "user omitted it"; per
        # FR 1.10 the CLI wins only when present (see apply_recipe_overrides).
        cli_passthrough_mode = (
            parse_env_passthrough_mode(env_passthrough_mode)
            if env_passthrough_mode is not None
            else None
        )
        clean_argv = validate_trailing_argv(argv)
        r = load_recipe(recipe, sidecar_files=mitigation_files or None)
        if r.probe_extras is None:
            raise ProbeUsageError(
                f"recipe {recipe} is not a probe-mode recipe; "
                "set 'mode: probe' at the recipe top level"
            )
        r = apply_recipe_overrides(r, ticket=ticket, cli_passthrough_mode=cli_passthrough_mode)
    except (ProbeUsageError, RecipeSchemaError, RecipeCellError, RegistryError, LookupError) as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        run_dir = run_recipe(
            r,
            output_dir=output,
            dry_run=dry_run,
            layout="flat_resume",
            resume_existing=True,
            subprocess_argv=clean_argv,
        )
    except RunDirLockedError as exc:
        # Friendlier surface than a stack trace: the exception message
        # already names the holder host/PID and the lockfile path to
        # remove. Wrap into ClickException so click exits 1 with the
        # operator-visible message rather than a Python traceback.
        raise click.ClickException(str(exc)) from exc
    except (RegistryError, RecipeCellError, RecipeSchemaError, LookupError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not dry_run:
        click.echo(f"Wrote probe artifacts to {run_dir}")


__all__ = ["probe", "ProbeExtras"]
