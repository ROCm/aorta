"""Pure helpers for the ``aorta probe`` CLI.

Kept out of :mod:`aorta.cli.probe` so the Click handler stays a thin shim
(see FR 1.15 -- handler body is bounded at 60 lines so it can't silently
grow business logic). Every function here is pure: no FS, no env mutation,
no subprocess.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from aorta.probe.classifier.disables import (
    DetectorSpecError,
    normalize_detector_ids,
    normalize_tiers,
)

if TYPE_CHECKING:
    from aorta.triage.recipe import Recipe

VALID_ENV_PASSTHROUGH_MODES: tuple[Literal["inherit", "file"], ...] = ("inherit", "file")


class ProbeUsageError(ValueError):
    """User-input error that the CLI should surface as ``ClickException``.

    Kept as a plain ``ValueError`` subclass (not a ``click.ClickException``)
    so non-CLI callers -- tests, the recipe-builder, future programmatic
    consumers -- don't need to depend on Click to catch it. The Click
    handler bridges this into a ``ClickException`` at the CLI boundary.
    """


def reject_flag_shaped_value(option_name: str, value: str | None) -> None:
    """Reject option values that look like another flag.

    Defends against the classic ``--output $TMPDIR --ticket X`` bug where
    ``$TMPDIR`` is unset: the shell collapses two spaces into one, Click
    sees ``--output --ticket`` and silently accepts ``--ticket`` as the
    *value* of ``--output``. The user's run then writes to a directory
    literally named ``--ticket`` and ``--ticket X`` is lost. ``X`` and
    everything after gets swept into the trailing argv.

    Only the leading ``--`` is treated as suspicious. A single ``-foo`` or
    a path that genuinely starts with ``-`` (rare but legal) is allowed
    through; the user can quote or use ``./-foo`` to disambiguate if
    needed. The leading-``--`` check covers ~all real-world recurrences
    of the bug without false-positive risk on legitimate paths.
    """
    if value is None:
        return
    if value.startswith("--"):
        raise ProbeUsageError(
            f"{option_name}: value {value!r} looks like another flag. "
            "Did you forget to set the variable or to quote the value? "
            f"(common cause: '{option_name} $VAR ...' where $VAR is unset)"
        )


_DEFAULT_BYPASS_TOKENS: frozenset[str] = frozenset({"--help", "-h"})


def help_token_in_option_zone(
    args: Sequence[str],
    value_taking_options: frozenset[str],
    bypass_tokens: frozenset[str] = _DEFAULT_BYPASS_TOKENS,
) -> bool:
    """True iff a bypass token (default ``--help``/``-h``) appears in the aorta-option zone.

    The "aorta-option zone" is the prefix of ``args`` that comes BEFORE
    either the explicit ``--`` separator or the first non-option
    positional argument (the user-command executable). Bypass tokens
    appearing AFTER the user command (``aorta probe --recipe r -- echo
    --help`` or ``aorta probe --recipe r echo --help`` with
    ``allow_interspersed_args=False``) are part of the user command and
    MUST NOT short-circuit the mandatory ``--`` separator check.

    Walks the token list left-to-right and consumes
    ``--opt value`` pairs for options that take a value (passed as
    ``value_taking_options``, derived from the Click command's
    ``params``). ``--opt=value`` is consumed as a single token. Flag
    options (``-v``, ``--dry-run``) are single tokens. The first token
    that doesn't start with ``-`` (and isn't an option value) marks
    the user-command boundary; a bypass token at or after that boundary
    is user-command content.

    ``bypass_tokens`` defaults to ``{"--help", "-h"}`` for callers that
    only need the help short-circuit. Phase 2 callers pass an extended
    set to also include ``--list-patterns`` (a flag that prints the
    Tier-4 catalogue and exits without consuming a user command).

    Defends against the bot-flagged misparse where ``aorta probe
    --recipe r --output o echo --help`` would silently skip the
    separator check.
    """
    i = 0
    while i < len(args):
        token = args[i]
        if token in bypass_tokens:
            return True
        if token == "--":
            return False
        if token.startswith("--"):
            opt = token.split("=", 1)[0]
            if "=" in token or opt not in value_taking_options:
                i += 1
                continue
            i += 2  # consume the value as well
            continue
        if token.startswith("-"):
            i += 1
            continue
        return False  # first user-command positional
    return False


def require_double_dash_separator(raw_argv: Sequence[str]) -> None:
    """Require an explicit ``--`` separator in the raw process argv.

    Without ``--``, Click cheerfully sweeps any leftover positional
    arguments into the trailing-argv list, masking flag-misparse bugs
    (e.g. a stray ``SMOKE-1`` becoming the user-command executable name).
    Requiring ``--`` makes the boundary explicit: aorta options on the
    left, opaque user command on the right.

    ``raw_argv`` is the full process argv (``sys.argv[1:]`` or the equivalent
    for a tested CLI invocation). The check passes as long as ``--`` appears
    somewhere; the trailing-argv emptiness check is a separate concern
    handled by :func:`validate_trailing_argv`.
    """
    if "--" not in raw_argv:
        raise ProbeUsageError(
            "missing '--' separator. The user command must come after a "
            "literal '--' so aorta knows where its own flags end. "
            "Usage: aorta probe [options] -- <command> [args...]"
        )


def parse_env_passthrough_mode(value: str) -> Literal["inherit", "file"]:
    """Validate the ``--env-passthrough-mode`` value.

    Both modes share the same in-process env-var application (the
    dispatcher sets per-cell mitigation + diagnostic env vars on
    ``os.environ`` before the workload's ``run()``); ``file`` mode
    additionally drops a ``probe.env`` file in the trial dir and exports
    ``AORTA_ENV_FILE`` to point at it. See ``docs/probe-188/usage.md``
    §"Env-passthrough modes" for the F6 rationale.
    """
    if value not in VALID_ENV_PASSTHROUGH_MODES:
        raise ProbeUsageError(
            f"--env-passthrough-mode: must be one of "
            f"{list(VALID_ENV_PASSTHROUGH_MODES)}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def parse_env_passthrough_mode_opt(value: str | None) -> Literal["inherit", "file"] | None:
    """Parse ``--env-passthrough-mode`` preserving the "flag omitted" signal.

    ``None`` in -> ``None`` out so :func:`apply_recipe_overrides` can tell
    "user passed the flag" from "user omitted it" (FR 1.10 precedence).
    Keeps the Click handler a thin shim by hosting the None-guard here.
    """
    return None if value is None else parse_env_passthrough_mode(value)


def validate_trailing_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Reject an empty / clearly-misparsed trailing-argv list.

    ``aorta probe -- <argv>`` is the only legal channel for the user
    command; ``aorta probe`` without a trailing ``--`` (or with ``--``
    followed by nothing) is a usage error. The "no parsing" invariant
    means we don't otherwise inspect ``argv`` -- it's forwarded
    byte-for-byte to :class:`SubprocessWorkload`.

    A second guard rejects ``argv[0]`` that starts with ``-``: a real user
    command's executable name is essentially never dash-prefixed, but a
    leaked aorta option (e.g. ``-v`` smuggled past Click) is. Catching
    this here turns a 127 exit-code "fail" trial into a clear usage error.
    """
    if not argv:
        raise ProbeUsageError(
            "no trailing argv supplied. Usage: aorta probe [options] -- <command> [args...]"
        )
    if argv[0].startswith("-"):
        raise ProbeUsageError(
            f"user command starts with {argv[0]!r}, which looks like a flag. "
            "Place all aorta options before '--' and the user command after. "
            "Usage: aorta probe [options] -- <command> [args...]"
        )
    return argv


def apply_recipe_overrides(
    recipe: Recipe,
    *,
    ticket: str | None,
    cli_passthrough_mode: Literal["inherit", "file"] | None,
    cli_stop_after_events: int | None = None,
    cli_max_trials: int | None = None,
    cli_disable_detectors: tuple[str, ...] = (),
) -> Recipe:
    """Layer CLI flags on top of a loaded probe-mode ``Recipe``.

    Overlays today:

    * ``--ticket`` -- when set, replaces ``recipe.ticket``;
    * ``--env-passthrough-mode`` -- when set (i.e. the user actually
      passed the flag), replaces ``recipe.probe_extras.env_passthrough_mode``;
    * ``--stop-after-events`` / ``--max-trials`` -- when either is passed,
      builds (or overlays) the recipe's ``stop_after`` rule (issue #232).
      Missing halves fall back to the recipe's existing ``stop_after``;
      a target with no cap (neither flag nor recipe supplies ``max_trials``)
      is rejected so the loop is never unbounded.
    * ``--disable-detector`` -- when passed (repeatable), each token is
      a whole-tier name (``tier3``) or a ``<tier>:<id>`` detector id
      (``tier2:hang``). Tokens are validated + classified here and
      UNIONed onto whatever the recipe already disables, so the CLI is
      additive rather than a replacement (an operator silencing one
      more detector on top of a recipe shouldn't have to restate the
      recipe's list).

    The caller must verify ``recipe.probe_extras is not None`` before
    invoking this helper (probe-mode discriminator is the CLI's
    responsibility). Living in ``aorta.probe.cli_helpers`` so the
    Click handler stays a thin shim per FR 1.15 (handler ≤ 60 lines,
    enforced by ``tests/probe/test_cli_parsing.py::test_handler_is_thin_shim``).
    """
    probe_extras = recipe.probe_extras
    assert probe_extras is not None, "apply_recipe_overrides: not a probe-mode recipe"
    if ticket is not None:
        recipe = dataclasses.replace(recipe, ticket=ticket)
    if cli_passthrough_mode is not None:
        recipe = dataclasses.replace(
            recipe,
            probe_extras=dataclasses.replace(
                probe_extras, env_passthrough_mode=cli_passthrough_mode
            ),
        )
    if cli_stop_after_events is not None or cli_max_trials is not None:
        recipe = _overlay_stop_after(recipe, cli_stop_after_events, cli_max_trials)
    if cli_disable_detectors:
        recipe = _overlay_disable_detectors(recipe, cli_disable_detectors)
    return recipe


def _overlay_stop_after(
    recipe: Recipe,
    cli_events: int | None,
    cli_max_trials: int | None,
) -> Recipe:
    """Build the ``stop_after`` overlay from the CLI flags (issue #232).

    CLI halves win over the recipe's existing block; the other half (and
    ``event_verdict``) fall back to the recipe. Validation mirrors the
    recipe loader: positive ints, ``max_trials >= events``, and a
    mandatory cap. Errors raise :class:`ProbeUsageError` for a friendly
    CLI message.
    """
    from aorta.triage.recipe import StopAfter

    base = recipe.stop_after
    events = cli_events if cli_events is not None else (base.events if base else None)
    max_trials = cli_max_trials if cli_max_trials is not None else (base.max_trials if base else None)
    if events is None:
        raise ProbeUsageError("--max-trials requires --stop-after-events")
    if max_trials is None:
        raise ProbeUsageError("--stop-after-events requires --max-trials (the loop needs a hard cap)")
    if events < 1 or max_trials < 1:
        raise ProbeUsageError("--stop-after-events and --max-trials must be >= 1")
    if max_trials < events:
        raise ProbeUsageError(
            f"--max-trials ({max_trials}) must be >= --stop-after-events ({events})"
        )
    event_verdict = base.event_verdict if base else "fail"
    return dataclasses.replace(
        recipe,
        stop_after=StopAfter(events=events, max_trials=max_trials, event_verdict=event_verdict),
    )


def _overlay_disable_detectors(recipe: Recipe, tokens: tuple[str, ...]) -> Recipe:
    """Classify ``--disable-detector`` tokens and union them onto the recipe.

    A token with a ``:`` is a detector id (``tier2:hang``); a bare
    token is a whole-tier name (``tier3``). Invalid tokens raise
    :class:`ProbeUsageError` so the CLI surfaces a friendly message
    rather than a stack trace.
    """
    probe_extras = recipe.probe_extras
    assert probe_extras is not None
    tier_tokens = [t for t in tokens if ":" not in t]
    id_tokens = [t for t in tokens if ":" in t]
    try:
        new_ids = normalize_detector_ids(list(probe_extras.disable_detectors) + id_tokens)
        new_tiers = normalize_tiers(list(probe_extras.disable_detector_tiers) + tier_tokens)
    except DetectorSpecError as exc:
        raise ProbeUsageError(f"--disable-detector: {exc}") from exc
    return dataclasses.replace(
        recipe,
        probe_extras=dataclasses.replace(
            probe_extras,
            disable_detectors=new_ids,
            disable_detector_tiers=new_tiers,
        ),
    )


def format_dry_run_line(
    cell_name: str,
    env: dict[str, str],
    argv: tuple[str, ...],
) -> str:
    """Render one cell's dry-run line.

    Stable key order (sorted) so snapshot-style tests are deterministic
    across runs. Kept in the helper module so the dry-run formatter can
    be unit-tested without invoking the CLI.
    """
    env_part = " ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(no env)"
    return f"  {cell_name}: env=[{env_part}] argv={list(argv)}"


__all__ = [
    "VALID_ENV_PASSTHROUGH_MODES",
    "ProbeUsageError",
    "apply_recipe_overrides",
    "format_dry_run_line",
    "help_token_in_option_zone",
    "parse_env_passthrough_mode",
    "parse_env_passthrough_mode_opt",
    "reject_flag_shaped_value",
    "require_double_dash_separator",
    "validate_trailing_argv",
]
