"""Recipe schema, loader, and flag-mode builder for `aorta triage run`.

The recipe is the authoritative description of a triage matrix invocation:
which cells to run (cartesian or hand-picked mitigation x environment pairs),
per-cell trial / step counts, the ticket the matrix belongs to, and the
speed-confound detection config.

Two entry points converge on the same `Recipe` dataclass:

* :func:`load_recipe` - parses a YAML or JSON recipe file.
* :func:`build_recipe_from_flags` - constructs an in-memory `Recipe` from the
  CLI flag shim (``aorta triage run --mode matrix --mitigation-axis ... --environment-axis ...``).

The runner consumes a validated `Recipe` and does not branch on the origin
of it - both paths produce the same structure.

Schema version: 1. Unknown ``schema_version`` values raise
:class:`RecipeSchemaError` with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aorta.registry import (
    get_environment,
    get_mitigation,
)

SCHEMA_VERSION = 1

# Top-level keys accepted by the recipe loader regardless of ``mode``.
# Two banks: triage-mode keys (the historical set, also accepted in
# probe-mode as a strict subset where they overlap) and probe-mode keys
# (issue #188) which are ONLY accepted when ``mode == "probe"``. The
# loader uses the union for unknown-key rejection and re-checks the
# subset per mode below.
_TRIAGE_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "ticket",
        "workload",
        "trials",
        "steps",
        "confound",
        "cells",
        "workload_config",
        "save_logs",
    }
)
_PROBE_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "ticket",
        "mode",
        "trials",
        "mitigation_axis",
        "diagnostic_axis",
        "step_time_regex",
        "collect_paths",
        "timeout_per_trial",
        "env_passthrough_mode",
        "confound",
        # Phase 2 additions (issue #188): custom_patterns runs after
        # the built-in classifier, hang knobs tune the Tier 2 monitor.
        # Accepted only in mode: probe (the triage-mode check below
        # rejects them when mode != "probe"); a top-level ``condition``
        # key OUTSIDE a custom_patterns[*] entry is still Phase 3 and
        # rejected via _PHASE_3_KEYS.
        "custom_patterns",
        "hang_window_sec",
        "hang_grace_period_at_start",
    }
)
# Phase 3 keys -- intercepted at load time BEFORE the unknown-keys
# rejection so the operator sees a clear "deferred to Phase 3"
# pointer instead of a generic "unknown key" message. Phase 2 keys
# (custom_patterns, hang knobs) are now in _PROBE_TOP_LEVEL and
# parsed normally when mode == "probe".
#
# A top-level ``condition:`` (outside a ``custom_patterns[*].match.condition``)
# is reserved for a future Phase 3-style "fail-the-trial-by-condition"
# block; rejected here so a typo'd indent of a custom_patterns
# condition can't fall through to a silent no-op.
_PHASE_3_KEYS = frozenset({"redaction", "condition"})
# Union of valid keys -- used for the unknown-key rejection. Phase 2 keys
# (``custom_patterns``, ``hang_window_sec``, ``hang_grace_period_at_start``)
# ARE included via ``_PROBE_TOP_LEVEL`` and parsed normally when
# ``mode == "probe"``. Only Phase 3 keys (``_PHASE_3_KEYS``) are kept
# OUT of this set; they are intercepted earlier so the operator sees
# a "deferred to Phase 3" pointer instead of a generic "unknown key".
_VALID_TOP_LEVEL = _TRIAGE_TOP_LEVEL | _PROBE_TOP_LEVEL | {"mode"}
_VALID_CONFOUND_KEYS = frozenset({"threshold", "baseline_cell"})
_VALID_CELL_KEYS = frozenset(
    {"name", "mitigations", "environment", "extra_env", "trials", "steps", "workload_config"}
)
_VALID_INLINE_ENV_KEYS = frozenset({"docker"})

# Keys that workload_config (recipe- or cell-scope) is NOT allowed to set.
# - "steps" is a first-class recipe/cell field; the dispatcher writes
#   ``config["steps"] = request.steps`` AFTER spreading config_overrides
#   (src/aorta/run/dispatcher.py), so a workload_config["steps"] would be
#   silently clobbered. Reject at load time so the recipe author finds out
#   immediately instead of debugging a step count that mysteriously ignores
#   their override.
# - The ``_aorta_*`` prefix is reserved for platform-supplied keys
#   (currently ``_aorta_environment``); the dispatcher already rejects them
#   at runtime. Mirror the rejection at recipe-load time so the failure
#   surfaces before any trial runs.
_RESERVED_WORKLOAD_CONFIG_KEYS = frozenset({"steps"})
_RESERVED_WORKLOAD_CONFIG_PREFIX = "_aorta_"

# Cell names become directory components under cells/<name>/ in the run output
# tree.  Reject characters that would let a recipe escape its own cells/
# directory (path traversal) or collide with sibling artifacts written by the
# runner (matrix.md, host_env.json, etc).  Keeping the allowed character set
# tight here means the runner can use the cell name as a path component
# without an extra layer of slugging that would silently rename cells.
_CELL_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_RESERVED_CELL_NAMES = frozenset({".", "..", "matrix.md", "matrix.json"})


class RecipeSchemaError(ValueError):
    """Raised when a recipe fails top-level schema validation (bad keys, bad types, bad version)."""


class RecipeCellError(ValueError):
    """Raised when a cell fails validation (duplicate name, empty mitigations, env-var collision)."""


@dataclass(frozen=True)
class InlineEnv:
    """An environment declared inline in a recipe as ``{docker: <ref>}``.

    Auto-named ``_inline_<hash>`` where ``<hash>`` is the first 8 chars of
    blake2b over the image ref. Two cells that reference the same image ref
    produce the same auto-name (deterministic), so the environment probe for
    that ref is captured exactly once.
    """

    name: str
    docker: str


@dataclass(frozen=True)
class ConfoundCfg:
    """Speed-confound detection configuration."""

    threshold: float = 1.15
    baseline_cell: str | None = None


@dataclass(frozen=True)
class Cell:
    """One row of the triage matrix.

    Attributes:
        name: Unique row label within the recipe (used as the matrix.md row
            label and the cells/<name>/ directory name).
        mitigations: Names to resolve through ``aorta.registry.get_mitigation``.
            Each name contributes an env-var bundle; bundles are unioned in
            list order. Cross-mitigation collisions (two bundles setting the
            same key to *different* values) are rejected at recipe-construction
            time by :func:`_validate_no_mitigation_collisions` -- no
            ``dict.update`` silent-wins. Use ``extra_env`` if you intentionally
            want to override a mitigation's value.
        environment: Either a registered environment name OR an inline-docker
            auto-name ``_inline_<hash>``. The recipe loader normalizes the
            ``{docker: <ref>}`` mapping shorthand into the auto-name and
            records the mapping on the parent :class:`Recipe`.
        extra_env: Ad-hoc env-var overrides applied AFTER the mitigation bundle
            (so this cell can override a registered mitigation for one-off
            experiments without polluting the registry). ``extra_env`` overrides
            are intentional and stay silent; only mitigation-vs-mitigation
            disagreements are flagged as errors.
        trials: Optional per-cell override of the recipe-level ``trials``.
        steps: Optional per-cell override of the recipe-level ``steps``.
        workload_config: Arbitrary ``dict[str, Any]`` forwarded to the
            workload constructor via ``Request.config_overrides``. Merged
            over the recipe-level ``workload_config`` (cell wins on key
            collision). Keys must be strings; ``"steps"`` and any
            ``_aorta_*`` key are rejected at load time -- ``steps`` is a
            first-class field the dispatcher would silently overwrite, and
            ``_aorta_*`` is reserved for platform-supplied keys.
    """

    name: str
    mitigations: tuple[str, ...]
    environment: str
    extra_env: dict[str, str] = field(default_factory=dict)
    trials: int | None = None
    steps: int | None = None
    workload_config: dict[str, Any] = field(default_factory=dict)

    def effective_trials(self, recipe_trials: int) -> int:
        return self.trials if self.trials is not None else recipe_trials

    def effective_steps(self, recipe_steps: int) -> int:
        return self.steps if self.steps is not None else recipe_steps


@dataclass(frozen=True)
class Recipe:
    """An in-memory, pre-validated triage-matrix recipe.

    Produced by :func:`load_recipe` or :func:`build_recipe_from_flags` and
    consumed by the runner. A ``Recipe`` is only constructed after all
    name-resolution, schema validation, and inline-docker normalization has
    succeeded, so downstream code can assume every cell references a name
    that will resolve at runtime.

    Attributes:
        schema_version: Always ``1`` for this build.
        workload: Workload name (resolved via ``aorta.workloads`` entry-point
            group at runtime by B1).
        trials: Recipe-level trial count. Cells override via ``cell.trials``.
        steps: Recipe-level step count. Cells override via ``cell.steps``.
        cells: Tuple of :class:`Cell` rows, in the order they appear in the
            source (preserved for matrix.md row ordering).
        ticket: Optional ticket ID; drives output-dir grouping. ``None`` is
            routed to ``_no_ticket_`` at write time.
        confound: Speed-confound detection configuration.
        inline_environments: Auto-registered inline envs referenced by cells.
            The runner writes a temporary sidecar JSON containing these so
            B1's ``get_environment`` resolves the auto-names.
        sidecar_files: Operator-supplied ``--mitigations-file`` paths that
            were used to validate this recipe's names. Carried on the
            ``Recipe`` so a programmatic ``load_recipe(path,
            sidecar_files=...) -> run_recipe(recipe)`` flow works without
            the caller threading sidecars through twice. The runner unions
            this with its own ``extra_sidecar_files`` argument and also
            snapshots each path into ``<run_dir>/sidecars/<basename>`` for
            replay. Empty tuple when no sidecars were supplied.
        source_path: Path of the source file if loaded from disk (None for
            flag-mode). Surfaced in ``matrix.md``.
        source_sha256: SHA-256 of the source file text (None for flag-mode).
            Surfaced in ``matrix.md`` for reproducibility.
        workload_config: Recipe-scope ``dict[str, Any]`` applied to every
            cell as the base ``Request.config_overrides``. Per-cell
            ``Cell.workload_config`` merges over this on a per-key basis
            (cell wins on collision; non-collision keys union). Empty dict
            when the recipe omits the field -- behaviourally identical to
            today's recipes.
        save_logs: When ``True``, the dispatcher captures the workload's
            in-process ``stdout`` / ``stderr`` writes (via
            ``contextlib.redirect_*``) into per-trial files alongside the
            trial JSON. Default ``False`` preserves today's behaviour
            (writes are not captured -- they appear on the parent
            process's TTY). Workloads that spawn subprocesses don't have
            their subprocess output captured by ``redirect_*``; those
            wrappers can opt in by reading the platform-supplied
            ``_aorta_save_logs`` / ``_aorta_log_prefix`` config keys
            the dispatcher injects and writing their own capture to a
            sibling path derived from the prefix (e.g.
            ``<prefix>.subprocess.stdout.log``). The prefix is an
            absolute path-with-stem so wrappers don't need to know
            ``results_dir``. Both file capture and key injection are
            **rank-0 only** -- matches the trial-JSON write guarantee.
            Wrappers running on non-rank-0 won't see the keys and
            should treat capture as off there.
    """

    schema_version: int
    workload: str
    trials: int
    steps: int
    cells: tuple[Cell, ...]
    ticket: str | None = None
    confound: ConfoundCfg = field(default_factory=ConfoundCfg)
    inline_environments: tuple[InlineEnv, ...] = ()
    sidecar_files: tuple[Path, ...] = ()
    source_path: Path | None = None
    source_sha256: str | None = None
    workload_config: dict[str, Any] = field(default_factory=dict)
    save_logs: bool = False
    # Probe-mode (issue #188) extras attached to the recipe so the
    # runner can branch on layout / resume / env-passthrough without
    # re-parsing the YAML. ``None`` for every triage-mode recipe;
    # populated only by :func:`aorta.probe.recipe_builder.build_probe_recipe_from_dict`.
    # Typed as ``Any`` to keep this module's import graph cycle-free --
    # the actual dataclass is :class:`aorta.probe.recipe_builder.ProbeExtras`.
    probe_extras: Any | None = None


def inline_env_name(docker_ref: str) -> str:
    """Deterministic auto-name for an inline docker environment.

    The first 8 hex chars of blake2b(image-ref). Matches the spec in issue
    #151 so two cells with the same ``docker_ref`` share a single
    auto-registered environment and therefore a single env-probe.
    """
    digest = hashlib.blake2b(docker_ref.encode("utf-8"), digest_size=4).hexdigest()
    return f"_inline_{digest}"


def _ensure_type(path_hint: str, value: Any, expected: type, label: str) -> None:
    if not isinstance(value, expected):
        raise RecipeSchemaError(
            f"{path_hint}: {label} must be {expected.__name__}, got "
            f"{type(value).__name__} ({value!r})"
        )


def _parse_confound(path_hint: str, raw: Any) -> ConfoundCfg:
    if raw is None:
        return ConfoundCfg()
    if not isinstance(raw, dict):
        raise RecipeSchemaError(
            f"{path_hint}.confound: must be a mapping, got {type(raw).__name__}"
        )
    unknown = set(raw) - _VALID_CONFOUND_KEYS
    if unknown:
        raise RecipeSchemaError(
            f"{path_hint}.confound: unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_VALID_CONFOUND_KEYS)}"
        )
    threshold = raw.get("threshold", 1.15)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise RecipeSchemaError(
            f"{path_hint}.confound.threshold: must be a number, got {type(threshold).__name__}"
        )
    if threshold <= 0:
        # Match flag-mode validation in `build_recipe_from_flags`. A non-positive
        # threshold makes ``classify_all`` flag every non-baseline cell as a
        # speed confound (any positive ratio >= threshold), which is never the
        # intent. Reject at load time so the two entry points agree on what
        # constitutes a valid recipe.
        raise RecipeSchemaError(f"{path_hint}.confound.threshold: must be > 0, got {threshold}")
    baseline = raw.get("baseline_cell")
    if baseline is not None and not isinstance(baseline, str):
        raise RecipeSchemaError(
            f"{path_hint}.confound.baseline_cell: must be a string, got {type(baseline).__name__}"
        )
    return ConfoundCfg(threshold=float(threshold), baseline_cell=baseline)


def _parse_environment(path_hint: str, raw: Any, inline_envs: dict[str, InlineEnv]) -> str:
    """Normalize a cell's environment field into a registered name.

    String -> returned as-is (registry lookup happens at runtime via B1).
    Mapping ``{docker: <ref>}`` -> auto-registered as ``_inline_<hash>``,
    recorded in ``inline_envs``, and the auto-name returned.
    """
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise RecipeSchemaError(
            f"{path_hint}.environment: must be a string (registered name) "
            f"or a mapping {{docker: <ref>}}, got {type(raw).__name__}"
        )
    unknown = set(raw) - _VALID_INLINE_ENV_KEYS
    if unknown:
        raise RecipeSchemaError(
            f"{path_hint}.environment: inline-docker mapping only accepts "
            f"{sorted(_VALID_INLINE_ENV_KEYS)}; got unknown keys "
            f"{sorted(unknown)}. (There is intentionally no 'name:' field -- "
            f"anything you'd want to name belongs in the registry.)"
        )
    if "docker" not in raw:
        raise RecipeSchemaError(
            f"{path_hint}.environment: inline-docker mapping missing required key 'docker'"
        )
    ref = raw["docker"]
    if not isinstance(ref, str) or not ref:
        raise RecipeSchemaError(
            f"{path_hint}.environment.docker: must be a non-empty string, "
            f"got {type(ref).__name__} ({ref!r})"
        )
    auto_name = inline_env_name(ref)
    existing = inline_envs.get(auto_name)
    if existing is None:
        inline_envs[auto_name] = InlineEnv(name=auto_name, docker=ref)
    elif existing.docker != ref:
        raise RecipeSchemaError(
            f"{path_hint}.environment: inline-env hash collision for "
            f"{auto_name!r}: {existing.docker!r} vs {ref!r}. "
            "Rename one ref or register a named environment explicitly."
        )
    return auto_name


def _parse_workload_config(path_hint: str, raw: Any) -> dict[str, Any]:
    """Validate and copy a ``workload_config`` mapping.

    Returns an empty dict when ``raw`` is None / missing so the caller can
    just call ``_parse_workload_config(hint, data.get("workload_config"))``
    without branching. The returned dict is a shallow copy -- values are
    forwarded as-is to the workload constructor (``dict[str, Any]``),
    keys MUST be strings, and the reserved set in
    :data:`_RESERVED_WORKLOAD_CONFIG_KEYS` / the ``_aorta_`` prefix is
    rejected at load time (see their docstring for the rationale).
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RecipeSchemaError(
            f"{path_hint}.workload_config: must be a mapping, got {type(raw).__name__}"
        )
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise RecipeSchemaError(
                f"{path_hint}.workload_config: keys must be strings, "
                f"got {type(k).__name__} ({k!r})"
            )
        if k in _RESERVED_WORKLOAD_CONFIG_KEYS:
            raise RecipeSchemaError(
                f"{path_hint}.workload_config: key {k!r} is reserved -- "
                "it is a first-class recipe/cell field and would be silently "
                "overwritten by the dispatcher. Set it at the recipe/cell scope instead."
            )
        if k.startswith(_RESERVED_WORKLOAD_CONFIG_PREFIX):
            raise RecipeSchemaError(
                f"{path_hint}.workload_config: key {k!r} uses the reserved "
                f"{_RESERVED_WORKLOAD_CONFIG_PREFIX!r} prefix "
                "(platform-supplied; not a user override)."
            )
        out[k] = v
    return out


def _parse_cell(idx: int, raw: Any, inline_envs: dict[str, InlineEnv]) -> Cell:
    path_hint = f"cells[{idx}]"
    if not isinstance(raw, dict):
        raise RecipeSchemaError(f"{path_hint}: must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _VALID_CELL_KEYS
    if unknown:
        raise RecipeSchemaError(
            f"{path_hint}: unknown keys {sorted(unknown)}; allowed: {sorted(_VALID_CELL_KEYS)}"
        )
    for required in ("name", "mitigations", "environment"):
        if required not in raw:
            raise RecipeSchemaError(f"{path_hint}: missing required key '{required}'")

    name = raw["name"]
    _ensure_type(path_hint, name, str, "name")
    _validate_cell_name(path_hint, name)

    mitigations = raw["mitigations"]
    if not isinstance(mitigations, list) or not all(isinstance(m, str) for m in mitigations):
        raise RecipeSchemaError(
            f"{path_hint}.mitigations: must be a list[str], got {mitigations!r}"
        )
    if not mitigations:
        raise RecipeSchemaError(
            f"{path_hint}.mitigations: empty list not allowed -- use ['none'] "
            "for the explicit baseline"
        )

    environment = _parse_environment(path_hint, raw["environment"], inline_envs)

    extra_env_raw = raw.get("extra_env", {})
    if extra_env_raw is None:
        extra_env_raw = {}
    if not isinstance(extra_env_raw, dict):
        raise RecipeSchemaError(
            f"{path_hint}.extra_env: must be a mapping of str -> str, got "
            f"{type(extra_env_raw).__name__}"
        )
    extra_env: dict[str, str] = {}
    for k, v in extra_env_raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise RecipeSchemaError(
                f"{path_hint}.extra_env[{k!r}]: keys and values must be strings, "
                f"got {type(k).__name__} -> {type(v).__name__}"
            )
        extra_env[k] = v

    trials = raw.get("trials")
    if trials is not None and (
        not isinstance(trials, int) or isinstance(trials, bool) or trials < 1
    ):
        raise RecipeSchemaError(f"{path_hint}.trials: must be a positive int, got {trials!r}")
    steps = raw.get("steps")
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1):
        raise RecipeSchemaError(f"{path_hint}.steps: must be a positive int, got {steps!r}")

    workload_config = _parse_workload_config(path_hint, raw.get("workload_config"))

    return Cell(
        name=name,
        mitigations=tuple(mitigations),
        environment=environment,
        extra_env=extra_env,
        trials=trials,
        steps=steps,
        workload_config=workload_config,
    )


def _reject_phase_3_keys(data: dict) -> None:
    """Intercept Phase 3 keys with a "deferred to phase N" pointer.

    Phase 2 keys (``custom_patterns`` / ``hang_window_sec`` /
    ``hang_grace_period_at_start``) are NOT intercepted here — they
    are valid in ``mode: probe`` recipes and parsed downstream. Only
    Phase 3 keys (``redaction``, top-level ``condition``) hit this
    rejection. Without it, a copy-pasted-from-the-design-doc recipe
    carrying ``redaction:`` would silently fall into the generic
    "unknown top-level key" path and the operator would have no
    signal that the key WILL be valid once Phase 3 ships.

    Renamed from ``_reject_phase_2_3_keys`` in PR #197 after Phase 2
    keys became valid (Copilot review): keeping the old name would
    have mis-suggested that Phase 2 keys are still trapped here.
    """
    phase3 = set(data) & _PHASE_3_KEYS
    if phase3:
        raise RecipeSchemaError(
            f"recipe: keys {sorted(phase3)} are deferred to Phase 3 "
            "(bundle integration + redaction + handout templates); see "
            "docs/plans/aorta-probe-188-rubric.md §PHASE 3. Remove these "
            "keys from your recipe."
        )


def _validate_top_level(data: Any) -> None:
    if not isinstance(data, dict):
        raise RecipeSchemaError(f"recipe top-level must be a mapping, got {type(data).__name__}")

    # Phase 3 keys are intercepted BEFORE either the unknown-key or
    # the per-mode required-key checks so the operator sees a
    # "deferred to Phase 3" pointer instead of a generic
    # "unknown top-level key" error. Phase 2 keys
    # (``custom_patterns``, ``hang_window_sec``,
    # ``hang_grace_period_at_start``) are NOT intercepted here; they
    # live in ``_PROBE_TOP_LEVEL`` and parse normally when
    # ``mode == "probe"``.
    _reject_phase_3_keys(data)

    mode = data.get("mode", "triage")
    if mode not in ("triage", "probe"):
        raise RecipeSchemaError(f"recipe.mode: must be 'triage' or 'probe', got {mode!r}")

    unknown = set(data) - _VALID_TOP_LEVEL
    if unknown:
        raise RecipeSchemaError(
            f"recipe: unknown top-level keys {sorted(unknown)}; allowed: {sorted(_VALID_TOP_LEVEL)}"
        )

    # Per-mode bank check: each mode has its own subset of valid keys.
    # Probe-mode keys (mitigation_axis, diagnostic_axis, step_time_regex,
    # collect_paths, timeout_per_trial, env_passthrough_mode) only make
    # sense when ``mode == 'probe'`` -- in triage mode they would be
    # silently ignored, which is exactly the silent-no-op footgun the
    # strict-unknown-key policy is supposed to forbid.
    if mode == "triage":
        misplaced = set(data) & (_PROBE_TOP_LEVEL - _TRIAGE_TOP_LEVEL - {"mode"})
        if misplaced:
            raise RecipeSchemaError(
                f"recipe: keys {sorted(misplaced)} are probe-mode only; "
                "set 'mode: probe' at the recipe top level or remove these keys"
            )
        for required in ("schema_version", "workload", "trials", "steps", "cells"):
            if required not in data:
                raise RecipeSchemaError(f"recipe: missing required key '{required}'")
    else:  # mode == "probe"
        misplaced = set(data) & (_TRIAGE_TOP_LEVEL - _PROBE_TOP_LEVEL - {"mode"})
        if misplaced:
            raise RecipeSchemaError(
                f"recipe: keys {sorted(misplaced)} are triage-mode only and "
                "are not valid in a 'mode: probe' recipe (probe-mode synthesises "
                "cells from mitigation_axis x diagnostic_axis and fixes the "
                "workload internally). Remove these keys."
            )
        for required in ("schema_version", "trials", "mitigation_axis", "diagnostic_axis"):
            if required not in data:
                raise RecipeSchemaError(f"recipe (mode: probe): missing required key '{required}'")

    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise RecipeSchemaError(
            f"recipe.schema_version: must be an integer, got {type(version).__name__} ({version!r})"
        )
    if version != SCHEMA_VERSION:
        raise RecipeSchemaError(
            f"recipe.schema_version: unsupported version {version}; "
            f"this build understands version {SCHEMA_VERSION}"
        )


def _validate_cell_name(path_hint: str, name: str) -> None:
    """Reject cell names that are unsafe as filesystem path components.

    Cell names land in the run-output tree as ``cells/<name>/``. Without this
    check, a recipe with ``name: ../foo`` or ``name: a/b`` could write its
    trial JSONs outside its own cell directory or clobber sibling artifacts
    like ``matrix.md``. Reject up-front so the runner can use ``cell.name``
    as a path component without an extra slugging layer that would silently
    rename cells (and, by renaming, break the unique-name contract too).
    """
    if not name:
        raise RecipeSchemaError(f"{path_hint}.name: must be non-empty")
    if name in _RESERVED_CELL_NAMES:
        raise RecipeCellError(
            f"{path_hint}.name {name!r}: reserved name (would clobber a sibling "
            "matrix artifact or escape the cell directory)"
        )
    if not _CELL_NAME_RE.match(name):
        raise RecipeCellError(
            f"{path_hint}.name {name!r}: must match {_CELL_NAME_RE.pattern} "
            "(used directly as the cells/<name>/ directory component; path "
            "separators, '..', leading '-', etc. are rejected to keep the run "
            "directory layout safe)"
        )


def _validate_unique_cell_names(cells: list[Cell]) -> None:
    seen: set[str] = set()
    for c in cells:
        if c.name in seen:
            raise RecipeCellError(
                f"duplicate cell name {c.name!r}; cell names must be unique "
                "within a recipe (they are used as matrix row labels and dir names)"
            )
        seen.add(c.name)


def _validate_names_resolve(
    cells: tuple[Cell, ...],
    inline_envs: dict[str, InlineEnv],
    sidecar_files: tuple[Path, ...] | None,
) -> None:
    """Pre-flight check that every mitigation + non-inline environment is known.

    Bubbles up B3's ``UnknownMitigationError`` / ``UnknownEnvironmentError``
    at load time instead of letting the runner hit it half-way through a
    multi-cell matrix (fail-fast).
    """
    extra = list(sidecar_files) if sidecar_files else None
    seen_mitigations: set[str] = set()
    seen_environments: set[str] = set()
    for cell in cells:
        for m in cell.mitigations:
            if m in seen_mitigations:
                continue
            get_mitigation(m, extra_files=extra)
            seen_mitigations.add(m)
        if cell.environment in inline_envs:
            continue
        if cell.environment in seen_environments:
            continue
        get_environment(cell.environment, extra_files=extra)
        seen_environments.add(cell.environment)


def _validate_no_mitigation_collisions(
    cells: tuple[Cell, ...],
    sidecar_files: tuple[Path, ...] | None,
) -> None:
    """Reject cells whose stacked mitigations disagree on the same env-var key.

    Per the Cell docstring, stacked mitigations are unioned in list order.
    Silently letting the later one win (``dict.update`` semantics) makes the
    cell's effective configuration order-dependent on the recipe author's
    typing -- that's exactly the kind of foot-gun the recipe contract is
    supposed to forbid. Flag it up front with the conflicting cell, key, and
    bundles so the recipe author either reorders, drops one, or pins the
    intended value via ``extra_env`` (which is documented as the override
    knob; its precedence over mitigations is intentional and stays silent).

    Sidecar resolution is performed once per unique mitigation name so an
    expensive sidecar isn't re-parsed per cell.
    """
    extra = list(sidecar_files) if sidecar_files else None
    bundle_cache: dict[str, dict[str, str]] = {}

    def _bundle(name: str) -> dict[str, str]:
        if name not in bundle_cache:
            bundle_cache[name] = dict(get_mitigation(name, extra_files=extra))
        return bundle_cache[name]

    for cell in cells:
        if len(cell.mitigations) < 2:
            continue
        applied: dict[str, tuple[str, str]] = {}  # key -> (value, source mitigation)
        for mit_name in cell.mitigations:
            for env_key, env_val in _bundle(mit_name).items():
                prior = applied.get(env_key)
                if prior is not None and prior[0] != env_val:
                    raise RecipeCellError(
                        f"cell {cell.name!r}: mitigation {mit_name!r} sets "
                        f"{env_key}={env_val!r}, but mitigation {prior[1]!r} "
                        f"already set {env_key}={prior[0]!r}. Stacked mitigations "
                        "must agree on overlapping keys; reorder, drop one, or "
                        "use extra_env to pin the intended value."
                    )
                applied[env_key] = (env_val, mit_name)


def _sha256_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_recipe(
    path: Path,
    sidecar_files: tuple[Path, ...] | None = None,
) -> Recipe:
    """Load, validate, and normalize a YAML or JSON recipe file.

    Args:
        path: Path to the recipe file. Extension ``.yaml``, ``.yml``, or
            ``.json``; the loader dispatches on extension. Anything else
            falls through to YAML (which accepts JSON as a subset).
        sidecar_files: Optional tuple of JSON sidecar paths forwarded to the
            registry so ad-hoc mitigations / environments defined in a
            sidecar resolve at validation time.

    Returns:
        A fully validated :class:`Recipe` with ``source_path`` and
        ``source_sha256`` populated for reproducibility metadata.

    Raises:
        RecipeSchemaError: Top-level schema violation (bad keys, bad types,
            unsupported ``schema_version``).
        RecipeCellError: Cell-level semantic violation (duplicate names, etc.).
        UnknownMitigationError / UnknownEnvironmentError: A referenced
            registry name is not known. Bubbled up from B3's resolver.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeSchemaError(f"recipe {path}: cannot read file ({exc})") from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise RecipeSchemaError(f"recipe {path}: parse error ({exc})") from exc

    recipe = _build_recipe(
        data,
        sidecar_files=sidecar_files,
        source_path=path,
        source_sha256=_sha256_bytes(text),
    )
    return recipe


def _build_recipe(
    data: Any,
    sidecar_files: tuple[Path, ...] | None,
    source_path: Path | None,
    source_sha256: str | None,
) -> Recipe:
    _validate_top_level(data)

    # Probe-mode recipes go through a separate builder so the
    # triage-mode path here stays byte-equivalent to today's loader.
    # The import is local because aorta.probe.recipe_builder imports
    # back into aorta.triage.recipe for Recipe / Cell / ConfoundCfg --
    # a top-level import would be a cycle.
    if data.get("mode", "triage") == "probe":
        from aorta.probe.recipe_builder import build_probe_recipe_from_dict

        return build_probe_recipe_from_dict(
            data,
            sidecar_files=sidecar_files,
            source_path=source_path,
            source_sha256=source_sha256,
        )

    workload = data["workload"]
    _ensure_type("recipe", workload, str, "workload")
    if not workload:
        raise RecipeSchemaError("recipe.workload: must be non-empty")

    trials = data["trials"]
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise RecipeSchemaError(f"recipe.trials: must be a positive int, got {trials!r}")

    steps = data["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise RecipeSchemaError(f"recipe.steps: must be a positive int, got {steps!r}")

    ticket = data.get("ticket")
    if ticket is not None and not isinstance(ticket, str):
        raise RecipeSchemaError(
            f"recipe.ticket: must be a string or absent, got {type(ticket).__name__}"
        )

    confound = _parse_confound("recipe", data.get("confound"))
    workload_config = _parse_workload_config("recipe", data.get("workload_config"))

    raw_save_logs = data.get("save_logs", False)
    if not isinstance(raw_save_logs, bool):
        raise RecipeSchemaError(
            f"recipe.save_logs: must be a boolean, got {type(raw_save_logs).__name__}"
        )

    raw_cells = data["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise RecipeSchemaError(f"recipe.cells: must be a non-empty list, got {raw_cells!r}")

    inline_envs: dict[str, InlineEnv] = {}
    cells = [_parse_cell(i, c, inline_envs) for i, c in enumerate(raw_cells)]
    _validate_unique_cell_names(cells)
    cells_tuple = tuple(cells)

    _validate_names_resolve(cells_tuple, inline_envs, sidecar_files)
    _validate_no_mitigation_collisions(cells_tuple, sidecar_files)

    if confound.baseline_cell is not None:
        names = {c.name for c in cells_tuple}
        if confound.baseline_cell not in names:
            raise RecipeCellError(
                f"confound.baseline_cell {confound.baseline_cell!r} does not "
                f"match any cell name; cells: {sorted(names)}"
            )

    return Recipe(
        schema_version=SCHEMA_VERSION,
        workload=workload,
        trials=trials,
        steps=steps,
        cells=cells_tuple,
        ticket=ticket,
        confound=confound,
        inline_environments=tuple(inline_envs.values()),
        sidecar_files=tuple(sidecar_files) if sidecar_files else (),
        source_path=source_path,
        source_sha256=source_sha256,
        workload_config=workload_config,
        save_logs=raw_save_logs,
    )


def build_recipe_from_flags(
    workload: str,
    mitigation_axis: str,
    environment_axis: str,
    trials: int,
    steps: int | None,
    ticket: str | None = None,
    baseline_cell: str | None = None,
    confound_threshold: float = 1.15,
    sidecar_files: tuple[Path, ...] | None = None,
) -> Recipe:
    """Construct an in-memory :class:`Recipe` from the CLI flag shim.

    The flag shim builds the full cartesian product of
    ``mitigation_axis x environment_axis``, naming each cell
    ``<mitigation>-<environment>``. The runner does not branch on mode after
    this point -- both the recipe path and the flag path funnel into
    :func:`aorta.triage.runner.run_recipe`.

    Environment-axis item parsing (Option B from the spec):

    * ``image:<ref>`` -> inline-docker cell using the same ``{docker: <ref>}``
      normalisation as recipe-mode. Cell name embeds the auto-name so
      ``<mitigation>-_inline_<hash>`` disambiguates multiple images on the
      same axis.
    * Anything else -> registered environment name (resolved against the
      registry at validation time).

    Every primitive is validated up-front to the same standard as
    :func:`load_recipe` (positive trials/steps, non-empty workload, sane
    confound threshold) so flag mode and recipe mode reject the same set of
    invalid inputs.
    """
    if not isinstance(workload, str) or not workload:
        raise RecipeSchemaError(f"--workload: must be a non-empty string, got {workload!r}")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise RecipeSchemaError(f"--trials: must be a positive int, got {trials!r}")
    if steps is None:
        raise RecipeSchemaError(
            "--steps is required in flag mode (ditto recipe mode). Pass --steps N explicitly."
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise RecipeSchemaError(f"--steps: must be a positive int, got {steps!r}")
    if not isinstance(confound_threshold, (int, float)) or isinstance(confound_threshold, bool):
        raise RecipeSchemaError(
            f"--confound-threshold: must be a number, got {type(confound_threshold).__name__}"
        )
    if confound_threshold <= 0:
        raise RecipeSchemaError(
            f"--confound-threshold: must be > 0, got {confound_threshold!r} "
            "(threshold is a step-time ratio; values <= 0 would flag every cell)"
        )
    if ticket is not None and (not isinstance(ticket, str) or not ticket):
        raise RecipeSchemaError(
            f"--ticket: must be a non-empty string when provided, got {ticket!r}"
        )

    mitigations = _split_axis(mitigation_axis, name="--mitigation-axis")
    raw_envs = _split_axis(environment_axis, name="--environment-axis")

    inline_envs: dict[str, InlineEnv] = {}
    env_cell_names: list[tuple[str, str]] = []
    for raw in raw_envs:
        if raw.startswith("image:"):
            ref = raw[len("image:") :]
            if not ref:
                raise RecipeSchemaError(
                    "--environment-axis item 'image:' requires a ref after the colon"
                )
            auto = inline_env_name(ref)
            inline_envs.setdefault(auto, InlineEnv(name=auto, docker=ref))
            env_cell_names.append((auto, auto))
        else:
            env_cell_names.append((raw, raw))

    cells: list[Cell] = []
    for m in mitigations:
        for env_name, display in env_cell_names:
            cell_name = f"{m}-{display}"
            _validate_cell_name(
                f"--mitigation-axis x --environment-axis ({m!r}, {display!r})", cell_name
            )
            cells.append(
                Cell(
                    name=cell_name,
                    mitigations=(m,),
                    environment=env_name,
                )
            )
    _validate_unique_cell_names(cells)
    cells_tuple = tuple(cells)

    inline_envs_tuple = tuple(inline_envs.values())
    _validate_names_resolve(cells_tuple, inline_envs, sidecar_files)
    _validate_no_mitigation_collisions(cells_tuple, sidecar_files)

    if baseline_cell is not None:
        names = {c.name for c in cells_tuple}
        if baseline_cell not in names:
            raise RecipeCellError(
                f"--baseline-cell {baseline_cell!r} does not match any cell; cells: {sorted(names)}"
            )

    return Recipe(
        schema_version=SCHEMA_VERSION,
        workload=workload,
        trials=trials,
        steps=steps,
        cells=cells_tuple,
        ticket=ticket,
        confound=ConfoundCfg(threshold=confound_threshold, baseline_cell=baseline_cell),
        inline_environments=inline_envs_tuple,
        sidecar_files=tuple(sidecar_files) if sidecar_files else (),
        source_path=None,
        source_sha256=None,
    )


def _split_axis(value: str, name: str) -> list[str]:
    if not value:
        raise RecipeSchemaError(f"{name}: must be non-empty")
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise RecipeSchemaError(f"{name}: no non-empty items after splitting on ','")
    return items


__all__ = [
    "SCHEMA_VERSION",
    "Cell",
    "ConfoundCfg",
    "InlineEnv",
    "Recipe",
    "RecipeCellError",
    "RecipeSchemaError",
    "build_recipe_from_flags",
    "inline_env_name",
    "load_recipe",
]
