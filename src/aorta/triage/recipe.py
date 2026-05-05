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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aorta.registry import (
    get_environment,
    get_mitigation,
)

SCHEMA_VERSION = 1

_VALID_TOP_LEVEL = frozenset(
    {"schema_version", "ticket", "workload", "trials", "steps", "confound", "cells"}
)
_VALID_CONFOUND_KEYS = frozenset({"threshold", "baseline_cell"})
_VALID_CELL_KEYS = frozenset({"name", "mitigations", "environment", "extra_env", "trials", "steps"})
_VALID_INLINE_ENV_KEYS = frozenset({"docker"})


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
            list order (later names win on collision WITHIN a cell only -- the
            runner re-detects cross-mitigation collisions at env-application
            time and raises :class:`RecipeCellError` if two bundles disagree
            on the same key).
        environment: Either a registered environment name OR an inline-docker
            auto-name ``_inline_<hash>``. The recipe loader normalizes the
            ``{docker: <ref>}`` mapping shorthand into the auto-name and
            records the mapping on the parent :class:`Recipe`.
        extra_env: Ad-hoc env-var overrides applied AFTER the mitigation bundle
            (so this cell can override a registered mitigation for one-off
            experiments without polluting the registry).
        trials: Optional per-cell override of the recipe-level ``trials``.
        steps: Optional per-cell override of the recipe-level ``steps``.
    """

    name: str
    mitigations: tuple[str, ...]
    environment: str
    extra_env: dict[str, str] = field(default_factory=dict)
    trials: int | None = None
    steps: int | None = None

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
        source_path: Path of the source file if loaded from disk (None for
            flag-mode). Surfaced in ``matrix.md``.
        source_sha256: SHA-256 of the source file text (None for flag-mode).
            Surfaced in ``matrix.md`` for reproducibility.
    """

    schema_version: int
    workload: str
    trials: int
    steps: int
    cells: tuple[Cell, ...]
    ticket: str | None = None
    confound: ConfoundCfg = field(default_factory=ConfoundCfg)
    inline_environments: tuple[InlineEnv, ...] = ()
    source_path: Path | None = None
    source_sha256: str | None = None


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
            f"{path_hint}.confound.threshold: must be a number, got " f"{type(threshold).__name__}"
        )
    baseline = raw.get("baseline_cell")
    if baseline is not None and not isinstance(baseline, str):
        raise RecipeSchemaError(
            f"{path_hint}.confound.baseline_cell: must be a string, got "
            f"{type(baseline).__name__}"
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
            f"{path_hint}.environment: inline-docker mapping missing required " f"key 'docker'"
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


def _parse_cell(idx: int, raw: Any, inline_envs: dict[str, InlineEnv]) -> Cell:
    path_hint = f"cells[{idx}]"
    if not isinstance(raw, dict):
        raise RecipeSchemaError(f"{path_hint}: must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _VALID_CELL_KEYS
    if unknown:
        raise RecipeSchemaError(
            f"{path_hint}: unknown keys {sorted(unknown)}; " f"allowed: {sorted(_VALID_CELL_KEYS)}"
        )
    for required in ("name", "mitigations", "environment"):
        if required not in raw:
            raise RecipeSchemaError(f"{path_hint}: missing required key '{required}'")

    name = raw["name"]
    _ensure_type(path_hint, name, str, "name")
    if not name:
        raise RecipeSchemaError(f"{path_hint}.name: must be non-empty")

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

    return Cell(
        name=name,
        mitigations=tuple(mitigations),
        environment=environment,
        extra_env=extra_env,
        trials=trials,
        steps=steps,
    )


def _validate_top_level(data: Any) -> None:
    if not isinstance(data, dict):
        raise RecipeSchemaError(f"recipe top-level must be a mapping, got {type(data).__name__}")
    for required in ("schema_version", "workload", "trials", "steps", "cells"):
        if required not in data:
            raise RecipeSchemaError(f"recipe: missing required key '{required}'")
    unknown = set(data) - _VALID_TOP_LEVEL
    if unknown:
        raise RecipeSchemaError(
            f"recipe: unknown top-level keys {sorted(unknown)}; "
            f"allowed: {sorted(_VALID_TOP_LEVEL)}"
        )
    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise RecipeSchemaError(
            f"recipe.schema_version: must be an integer, got "
            f"{type(version).__name__} ({version!r})"
        )
    if version != SCHEMA_VERSION:
        raise RecipeSchemaError(
            f"recipe.schema_version: unsupported version {version}; "
            f"this build understands version {SCHEMA_VERSION}"
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
            f"recipe.ticket: must be a string or absent, got " f"{type(ticket).__name__}"
        )

    confound = _parse_confound("recipe", data.get("confound"))

    raw_cells = data["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise RecipeSchemaError(f"recipe.cells: must be a non-empty list, got {raw_cells!r}")

    inline_envs: dict[str, InlineEnv] = {}
    cells = [_parse_cell(i, c, inline_envs) for i, c in enumerate(raw_cells)]
    _validate_unique_cell_names(cells)
    cells_tuple = tuple(cells)

    _validate_names_resolve(cells_tuple, inline_envs, sidecar_files)

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
        source_path=source_path,
        source_sha256=source_sha256,
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

    Steps is optional at CLI (per the flag spec), so when ``steps is None``
    we still require a positive int for the Recipe (default 1 makes the
    schema happy; per-cell overrides aren't used in flag mode so the
    effective value flows through unchanged).
    """
    mitigations = _split_axis(mitigation_axis, name="--mitigation-axis")
    raw_envs = _split_axis(environment_axis, name="--environment-axis")
    if steps is None:
        raise RecipeSchemaError(
            "--steps is required in flag mode (ditto recipe mode). " "Pass --steps N explicitly."
        )

    inline_envs: dict[str, InlineEnv] = {}
    env_cell_names: list[tuple[str, str]] = []
    for raw in raw_envs:
        if raw.startswith("image:"):
            ref = raw[len("image:") :]
            if not ref:
                raise RecipeSchemaError(
                    "--environment-axis item 'image:' requires a ref after " "the colon"
                )
            auto = inline_env_name(ref)
            inline_envs.setdefault(auto, InlineEnv(name=auto, docker=ref))
            env_cell_names.append((auto, auto))
        else:
            env_cell_names.append((raw, raw))

    cells: list[Cell] = []
    for m in mitigations:
        for env_name, display in env_cell_names:
            cells.append(
                Cell(
                    name=f"{m}-{display}",
                    mitigations=(m,),
                    environment=env_name,
                )
            )
    _validate_unique_cell_names(cells)
    cells_tuple = tuple(cells)

    inline_envs_tuple = tuple(inline_envs.values())
    _validate_names_resolve(cells_tuple, inline_envs, sidecar_files)

    if baseline_cell is not None:
        names = {c.name for c in cells_tuple}
        if baseline_cell not in names:
            raise RecipeCellError(
                f"--baseline-cell {baseline_cell!r} does not match any cell; "
                f"cells: {sorted(names)}"
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
