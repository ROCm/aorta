"""Probe-mode recipe builder (issue #188 Phase 1).

Parses a ``mode: probe`` recipe dict into a :class:`aorta.triage.recipe.Recipe`
whose ``cells`` are the cartesian product of
``mitigation_axis x diagnostic_axis``, with ``workload`` fixed to the
reserved internal name ``_subprocess`` and ``probe_extras`` populated for
the runner to honour layout / resume / env-passthrough semantics.

This module is the SINGLE place that knows how a probe-mode recipe shape
maps to the existing Recipe / Cell shape; the rest of the platform stays
mode-agnostic (the runner just iterates ``recipe.cells``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aorta.probe.classifier.disables import (
    DetectorSpecError,
    normalize_detector_ids,
    normalize_tiers,
)
from aorta.probe.classifier.tier2_hang import (
    DEFAULT_HANG_GRACE_SEC,
    DEFAULT_HANG_WINDOW_SEC,
)
from aorta.probe.classifier.tier5_custom import (
    CompiledPattern,
    validate_custom_patterns,
)
from aorta.probe.redaction import RedactionCfg, parse_redaction
from aorta.registry import get_mitigation
from aorta.registry.errors import UnknownMitigationError
from aorta.triage.recipe import (
    _VALID_CONFOUND_KEYS,
    SCHEMA_VERSION,
    Cell,
    ConfoundCfg,
    Recipe,
    RecipeCellError,
    RecipeSchemaError,
    RetainPolicy,
    _parse_retain,
    _parse_stop_after,
)

# Fixed workload name for probe-mode cells. Leading underscore is
# deliberate: it cannot collide with a user-facing workload (B1's
# discovery doesn't reserve the prefix but no public workload uses
# it) and it surfaces as "platform-internal" when ``aorta triage
# list-workloads`` lands. Registered in pyproject.toml under
# ``[project.entry-points."aorta.workloads"]``.
SUBPROCESS_WORKLOAD_NAME = "_subprocess"

# Default for ``timeout_per_trial`` when the recipe omits it. None means
# no timeout -- the subprocess runs until exit. Wired through to
# SubprocessWorkload via ProbeExtras; the dispatcher does not look at it.
DEFAULT_TIMEOUT_PER_TRIAL: float | None = None


@dataclass(frozen=True)
class ProbeExtras:
    """Probe-mode-only knobs attached to :class:`Recipe.probe_extras`.

    Carried as a typed sidecar (rather than smuggled into
    ``workload_config``) so the runner can read them without
    re-parsing the YAML and so the triage code path stays untouched.
    Phase 2 added the classifier knobs (``hang_window_sec``,
    ``hang_grace_period_at_start``, ``custom_patterns``) -- see the
    fields below. Phase 3 will add the ``redaction`` block.

    Attributes:
        step_time_regex: Optional regex string; Phase 1 keeps this on
            the recipe for forward-compat only -- the Phase 2
            classifier consumes it. Validated as compileable at load
            time so a typo surfaces immediately.
        collect_paths: List of glob patterns. Phase 1 always collects
            ``stdout.log`` / ``stderr.log`` / ``result.json``; the
            generic collector lands in Phase 2.
        timeout_per_trial: Wall-clock cap per trial in seconds; None
            means no cap. Passed verbatim to ``SubprocessWorkload``.
        env_passthrough_mode: ``"inherit"`` (default) or ``"file"``.
            Carried here so a recipe can pin the mode. Precedence: the
            CLI flag ``--env-passthrough-mode`` wins ONLY when the
            user actually passes it; otherwise the recipe's
            ``env_passthrough_mode:`` value (this field) is honored,
            and if neither is set the recipe-builder default
            ``"inherit"`` applies. The CLI handler in
            ``aorta.cli.probe.probe`` distinguishes "user omitted the
            flag" from "user passed the flag" by defaulting the
            Click option to ``None`` and only overlaying when not
            ``None``; see
            ``tests/probe/test_cli_parsing.py::test_env_passthrough_mode_precedence_*``
            for the pinned contract.
        mitigation_axis: The mitigation-axis names that produced the
            cells, preserved for audit + the dry-run formatter.
        diagnostic_axis: Same, for the diagnostic axis.
    """

    step_time_regex: str | None = None
    collect_paths: tuple[str, ...] = ()
    timeout_per_trial: float | None = DEFAULT_TIMEOUT_PER_TRIAL
    env_passthrough_mode: Literal["inherit", "file"] = "inherit"
    mitigation_axis: tuple[str, ...] = ()
    diagnostic_axis: tuple[str, ...] = ()
    # Env vars the mitigation_axis + diagnostic_axis stamps resolve
    # to, keyed by cell name. The dispatcher applies them via the
    # existing ``mitigations=`` channel on RunRequest, so this field
    # exists for the dry-run formatter and Phase 2 / 3 inspection
    # only. ``dict`` rather than ``frozendict`` to keep the dataclass
    # JSON-serialisable for matrix.json.
    cell_envs: dict[str, dict[str, str]] = field(default_factory=dict)
    # Phase 2 additions (issue #188): compiled custom_patterns +
    # hang knobs threaded onto the recipe so the runner can attach
    # them to each cell's ``RunRequest.probe_extras`` without
    # re-parsing the YAML. ``custom_patterns`` is empty by default
    # so probe recipes that don't set the block round-trip exactly
    # as Phase 1 produced them.
    custom_patterns: tuple[CompiledPattern, ...] = ()
    hang_window_sec: float = DEFAULT_HANG_WINDOW_SEC
    hang_grace_period_at_start: float = DEFAULT_HANG_GRACE_SEC
    # When False, skip the Tier-3 pre/post VRAM delta check. Useful for
    # opaque docker wrappers where GPU allocation is normal workload
    # behaviour rather than a leak signal.
    tier3_vram_growth: bool = True
    # Issue #229: operator detector-disable knobs. ``disable_detectors``
    # holds ``<tier>:<id>`` tokens (e.g. ``"tier2:hang"``);
    # ``disable_detector_tiers`` holds whole-tier tokens (``"tier3"``).
    # Validated at load time by ``aorta.probe.classifier.disables``; a
    # disabled detector is never evaluated and never counts toward the
    # verdict. Empty tuples are the no-op default so recipes that don't
    # set the knobs round-trip exactly.
    disable_detectors: tuple[str, ...] = ()
    disable_detector_tiers: tuple[str, ...] = ()
    # Phase 3 (issue #188): redaction block consumed by ``aorta bundle``.
    redaction: RedactionCfg | None = None
    # Issue #231: verdict-keyed per-trial artifact retention. ``None``
    # preserves the legacy keep-everything behaviour; when set,
    # SubprocessWorkload prunes each trial dir to the level mapped from
    # that trial's verdict (full/summary/log/none) after classification,
    # never dropping the trial record (``result.json``).
    retain: RetainPolicy | None = None


def _ensure_str_list(path_hint: str, raw: Any, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RecipeSchemaError(
            f"{path_hint}: must be a list[str], got {type(raw).__name__} ({raw!r})"
        )
    if not raw and not allow_empty:
        raise RecipeSchemaError(f"{path_hint}: must be a non-empty list")
    return list(raw)


def _validate_axis_names(
    path_hint: str,
    axis: list[str],
    sidecar_files: tuple[Path, ...] | None,
) -> None:
    """Resolve every axis name through the mitigations registry.

    Probe-mode treats both axes as mitigation names: ``none`` is the
    no-op baseline, ``tf32_off`` flips ``DISABLE_TF32`` etc. Unknown
    names bubble up as :class:`aorta.registry.errors.UnknownMitigationError`
    at load time rather than mid-run.
    """
    extra = list(sidecar_files) if sidecar_files else None
    for name in axis:
        try:
            get_mitigation(name, extra_files=extra)
        except UnknownMitigationError as exc:
            raise UnknownMitigationError(f"{path_hint}: {exc}") from exc


def _safe_cell_segment(name: str) -> str:
    """Slug an axis-value into a cell-name path segment.

    Cell names ride through the existing ``_validate_cell_name`` regex
    (``^[A-Za-z0-9_][A-Za-z0-9_.\\-]*$``) so the resulting
    ``<mitigation>-<diagnostic>`` string can be used as a path
    component without an extra slugging layer. Most B3 mitigation
    names already conform; we light-weight scrub anything that
    doesn't.
    """
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    if not cleaned or cleaned[0] in {".", "-"}:
        cleaned = "_" + cleaned
    return cleaned


def _validate_collect_paths(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RecipeSchemaError(
            f"recipe.collect_paths: must be a list[str], got {type(raw).__name__}"
        )
    return tuple(raw)


def _validate_step_time_regex(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RecipeSchemaError(
            f"recipe.step_time_regex: must be a string, got {type(raw).__name__}"
        )
    # Phase-1 hardening: compile-validate the regex at load time so a
    # typo surfaces immediately rather than at first match (Phase 2's
    # classifier is the actual consumer).
    import re

    try:
        re.compile(raw)
    except re.error as exc:
        raise RecipeSchemaError(f"recipe.step_time_regex: invalid regex ({exc}): {raw!r}") from exc
    return raw


def _validate_timeout_per_trial(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RecipeSchemaError(
            f"recipe.timeout_per_trial: must be a number or null, got {type(raw).__name__}"
        )
    if raw <= 0:
        raise RecipeSchemaError(f"recipe.timeout_per_trial: must be > 0 when set, got {raw}")
    return float(raw)


def _validate_env_passthrough_mode(raw: Any) -> Literal["inherit", "file"]:
    if raw is None:
        return "inherit"
    if raw not in ("inherit", "file"):
        raise RecipeSchemaError(
            f"recipe.env_passthrough_mode: must be 'inherit' or 'file', got {raw!r}"
        )
    return "file" if raw == "file" else "inherit"


def _validate_positive_seconds(
    path_hint: str,
    raw: Any,
    *,
    default: float,
    allow_zero: bool = False,
) -> float:
    """Validate a positive-seconds Phase-2 knob (hang_window_sec, etc.).

    Returns ``default`` when ``raw`` is None / missing; otherwise
    coerces to ``float`` after rejecting bools (which would pass
    ``isinstance(int)`` checks) and non-positive numbers.

    ``allow_zero`` flips the lower bound from strict (``> 0``) to
    inclusive (``>= 0``). Used for ``hang_grace_period_at_start``,
    which the runtime predicate explicitly supports as "no grace
    period -- a hang can be detected immediately after the window
    elapses" (useful for short-running repros where 60s of grace
    swallows the trial). ``hang_window_sec`` keeps the strict bound
    because a zero window would re-trip the predicate on every poll.
    """
    if raw is None:
        return float(default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RecipeSchemaError(f"{path_hint}: must be a number, got {type(raw).__name__}")
    if allow_zero:
        if raw < 0:
            raise RecipeSchemaError(f"{path_hint}: must be >= 0 when set, got {raw}")
    elif raw <= 0:
        raise RecipeSchemaError(f"{path_hint}: must be > 0 when set, got {raw}")
    return float(raw)


def build_probe_recipe_from_dict(
    data: dict,
    sidecar_files: tuple[Path, ...] | None,
    source_path: Path | None = None,
    source_sha256: str | None = None,
) -> Recipe:
    """Build a probe-mode :class:`Recipe` from a parsed YAML/JSON dict.

    Called from :func:`aorta.triage.recipe._build_recipe` when
    ``data["mode"] == "probe"``. The caller has already run
    :func:`aorta.triage.recipe._validate_top_level`, so the dict is
    known to be schema-valid at the outer layer (mode set, required
    probe-mode keys present, no triage-mode-only keys, no Phase 2/3
    keys).

    Synthesises cells as ``<mitigation>-<diagnostic>`` and pre-computes
    each cell's resolved env-var bundle on the returned
    :class:`ProbeExtras` so the dry-run formatter doesn't need to call
    back into the registry.
    """
    trials_raw = data["trials"]
    if isinstance(trials_raw, bool) or not isinstance(trials_raw, int) or trials_raw < 1:
        raise RecipeSchemaError(f"recipe.trials: must be a positive int, got {trials_raw!r}")

    ticket = data.get("ticket")
    if ticket is not None and not isinstance(ticket, str):
        raise RecipeSchemaError(
            f"recipe.ticket: must be a string or absent, got {type(ticket).__name__}"
        )

    mitigation_axis = _ensure_str_list("recipe.mitigation_axis", data["mitigation_axis"])
    diagnostic_axis = _ensure_str_list("recipe.diagnostic_axis", data["diagnostic_axis"])

    _validate_axis_names("recipe.mitigation_axis", mitigation_axis, sidecar_files)
    _validate_axis_names("recipe.diagnostic_axis", diagnostic_axis, sidecar_files)

    step_time_regex = _validate_step_time_regex(data.get("step_time_regex"))
    collect_paths = _validate_collect_paths(data.get("collect_paths"))
    timeout_per_trial = _validate_timeout_per_trial(data.get("timeout_per_trial"))
    env_passthrough_mode = _validate_env_passthrough_mode(data.get("env_passthrough_mode"))
    custom_patterns = validate_custom_patterns(data.get("custom_patterns"))
    hang_window_sec = _validate_positive_seconds(
        "recipe.hang_window_sec",
        data.get("hang_window_sec"),
        default=DEFAULT_HANG_WINDOW_SEC,
    )
    hang_grace_period_at_start = _validate_positive_seconds(
        "recipe.hang_grace_period_at_start",
        data.get("hang_grace_period_at_start"),
        default=DEFAULT_HANG_GRACE_SEC,
        allow_zero=True,
    )
    tier3_vram_growth_raw = data.get("tier3_vram_growth", True)
    if not isinstance(tier3_vram_growth_raw, bool):
        raise RecipeSchemaError(
            "recipe.tier3_vram_growth: must be a boolean, got "
            f"{type(tier3_vram_growth_raw).__name__} ({tier3_vram_growth_raw!r})"
        )
    tier3_vram_growth = tier3_vram_growth_raw
    stop_after = _parse_stop_after("recipe", data.get("stop_after"))
    try:
        disable_detectors = normalize_detector_ids(data.get("disable_detectors"))
        disable_detector_tiers = normalize_tiers(data.get("disable_detector_tiers"))
    except DetectorSpecError as exc:
        # ``DetectorSpecError`` messages either already carry a field prefix
        # (e.g. ``disable_detectors: ...``) or are a bare token diagnostic
        # (e.g. ``unknown tier 'tier9'``). ``recipe: {exc}`` reads cleanly for
        # both, whereas ``recipe.{exc}`` produced ``recipe.unknown tier`` for
        # the bare case. Mirrors the ``recipe:`` prefix used in the triage
        # loader (aorta.triage.recipe).
        raise RecipeSchemaError(f"recipe: {exc}") from exc
    retain = _parse_retain("recipe", data.get("retain"))
    # Use ``in`` rather than ``.get(...) is not None`` so an explicit
    # ``redaction: null`` is treated as present-but-invalid (parse_redaction
    # rejects None) instead of being silently conflated with "no redaction
    # block" -- a null block should fail validation, not quietly disable
    # scrubbing.
    redaction = parse_redaction(data["redaction"]) if "redaction" in data else None

    # confound block is permitted (already in _PROBE_TOP_LEVEL) but
    # not meaningful for Tier-1-only Phase 1 -- pass it through with
    # an explicit default so the runner's existing baseline-resolution
    # code keeps working.
    confound_raw = data.get("confound")
    if confound_raw is not None and not isinstance(confound_raw, dict):
        raise RecipeSchemaError(
            f"recipe.confound: must be a mapping, got {type(confound_raw).__name__}"
        )
    if confound_raw is not None:
        # Mirror the triage-mode loader's strict-schema contract
        # (:func:`aorta.triage.recipe._parse_confound`): reject typo'd
        # keys like ``baseline_cel`` up-front instead of silently
        # ignoring them and using the auto-discovered baseline. Sharing
        # ``_VALID_CONFOUND_KEYS`` with the triage parser means a new
        # confound key added there is automatically rejected here too
        # until probe explicitly opts in.
        unknown = set(confound_raw) - _VALID_CONFOUND_KEYS
        if unknown:
            raise RecipeSchemaError(
                f"recipe.confound: unknown keys {sorted(unknown)}; "
                f"allowed: {sorted(_VALID_CONFOUND_KEYS)}"
            )

    # Synthesise the cartesian product of the two axes. Cell-name
    # collisions (two distinct (m, d) pairs that slug-collapse to the
    # same name) are rejected up-front so the per-trial directory
    # layout never silently overwrites a sibling.
    cells: list[Cell] = []
    seen_cell_names: dict[str, tuple[str, str]] = {}
    extra = list(sidecar_files) if sidecar_files else None
    cell_envs: dict[str, dict[str, str]] = {}
    for m in mitigation_axis:
        m_env = get_mitigation(m, extra_files=extra)
        for d in diagnostic_axis:
            d_env = get_mitigation(d, extra_files=extra)
            cell_name = f"{_safe_cell_segment(m)}-{_safe_cell_segment(d)}"
            if cell_name in seen_cell_names:
                prior = seen_cell_names[cell_name]
                raise RecipeCellError(
                    f"probe-mode cell synthesis: ({m!r}, {d!r}) and "
                    f"({prior[0]!r}, {prior[1]!r}) both slug to {cell_name!r}; "
                    "rename one of the axis values"
                )
            seen_cell_names[cell_name] = (m, d)

            # Resolved env bundle for the cell, for the dry-run formatter
            # and matrix.json audit. The dispatcher computes the same
            # bundle internally from ``mitigations=(m, d)``; we capture
            # it here so the cell-collision check on overlapping keys
            # surfaces at recipe-load time rather than mid-run.
            merged_env: dict[str, str] = {}
            for key, val in m_env.items():
                merged_env[key] = val
            for key, val in d_env.items():
                prior_val = merged_env.get(key)
                if prior_val is not None and prior_val != val:
                    raise RecipeCellError(
                        f"probe-mode cell {cell_name!r}: mitigation axis "
                        f"{m!r} sets {key}={prior_val!r}, diagnostic axis "
                        f"{d!r} sets {key}={val!r}. Pick a non-overlapping "
                        "pair or split the conflicting axis."
                    )
                merged_env[key] = val
            cell_envs[cell_name] = merged_env

            # Construct the Cell. ``mitigations`` carries BOTH axes so
            # the dispatcher unions them itself (the same code path B2
            # already exercises for stacked-mitigation cells); we don't
            # smuggle the merged env via ``extra_env`` because that
            # would lose the per-axis provenance the matrix wants.
            cells.append(
                Cell(
                    name=cell_name,
                    mitigations=(m, d),
                    environment="local",
                )
            )

    # Probe-mode bypasses the speed-confound classifier (Phase 1 is
    # Tier-1 only), but ``run_recipe`` unconditionally resolves a
    # baseline cell -- so we set one here so the existing resolver
    # (which expects either an explicit name, a ``baseline-*`` cell, or
    # a ``mitigations == ['none']`` cell) doesn't trip on probe-mode
    # cell names like ``none-none`` / ``tf32_off-none`` whose
    # mitigation tuples have length 2. Prefer a cell whose both axes
    # are ``none`` (the canonical no-op baseline); fall back to the
    # first cell so a recipe without a none-none pair still resolves.
    none_none_name: str | None = None
    for cell in cells:
        if all(m == "none" for m in cell.mitigations):
            none_none_name = cell.name
            break
    default_baseline = none_none_name if none_none_name is not None else cells[0].name

    confound = ConfoundCfg(baseline_cell=default_baseline)
    if confound_raw is not None:
        # Honor an explicit threshold if the operator set one (cheap
        # forward-compat for Phase 2 step-time-regex aggregation).
        threshold = confound_raw.get("threshold", confound.threshold)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
            raise RecipeSchemaError(f"recipe.confound.threshold: must be > 0, got {threshold!r}")
        explicit_baseline = confound_raw.get("baseline_cell", default_baseline)
        if explicit_baseline is not None and not isinstance(explicit_baseline, str):
            raise RecipeSchemaError(
                f"recipe.confound.baseline_cell: must be a string, got "
                f"{type(explicit_baseline).__name__}"
            )
        confound = ConfoundCfg(threshold=float(threshold), baseline_cell=explicit_baseline)

    probe_extras = ProbeExtras(
        step_time_regex=step_time_regex,
        collect_paths=collect_paths,
        timeout_per_trial=timeout_per_trial,
        env_passthrough_mode=env_passthrough_mode,
        mitigation_axis=tuple(mitigation_axis),
        diagnostic_axis=tuple(diagnostic_axis),
        cell_envs=cell_envs,
        custom_patterns=custom_patterns,
        hang_window_sec=hang_window_sec,
        hang_grace_period_at_start=hang_grace_period_at_start,
        tier3_vram_growth=tier3_vram_growth,
        disable_detectors=disable_detectors,
        disable_detector_tiers=disable_detector_tiers,
        redaction=redaction,
        retain=retain,
    )

    return Recipe(
        schema_version=SCHEMA_VERSION,
        workload=SUBPROCESS_WORKLOAD_NAME,
        trials=trials_raw,
        # ``steps`` is required by the triage-mode dataclass; probe
        # mode doesn't use it (subprocess workloads don't iterate),
        # so we pass 1 as a non-meaningful placeholder. The runner
        # threads it through but ``SubprocessWorkload.run()`` ignores
        # the value.
        steps=1,
        cells=tuple(cells),
        ticket=ticket,
        confound=confound,
        inline_environments=(),
        sidecar_files=tuple(sidecar_files) if sidecar_files else (),
        source_path=source_path,
        source_sha256=source_sha256,
        workload_config={},
        save_logs=False,
        stop_after=stop_after,
        probe_extras=probe_extras,
    )


__all__ = [
    "DEFAULT_TIMEOUT_PER_TRIAL",
    "SUBPROCESS_WORKLOAD_NAME",
    "ProbeExtras",
    "build_probe_recipe_from_dict",
]
