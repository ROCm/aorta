"""Adapter between probe-mode recipes and ``aorta bundle`` (issue #188 Phase 3).

Resolves the ``redaction:`` block from a probe recipe and constructs the
:class:`~aorta.probe.redaction.RedactingRedactor` the bundle writer expects.
When no ``redaction:`` block is present, returns
:class:`~aorta.bundle.redactor.IdentityRedactor`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aorta.bundle.redactor import IdentityRedactor, Redactor
from aorta.probe.redaction import RedactingRedactor, RedactionCfg, parse_redaction
from aorta.triage.recipe import RecipeSchemaError, load_recipe_mapping

log = logging.getLogger(__name__)

_RECIPE_RESOLVED_NAME = "recipe.resolved.yaml"


def load_redaction_cfg(recipe_path: Path) -> RedactionCfg | None:
    """Parse only the ``redaction:`` block from a recipe file.

    Bundling needs nothing but the ``redaction:`` mapping, so this parses
    just that key (via :func:`~aorta.triage.recipe.load_recipe_mapping`)
    rather than running the full recipe loader. A full
    :func:`~aorta.triage.recipe.load_recipe` resolves the mitigation /
    diagnostic axes against the registry, which fails for a perfectly
    valid probe run whose recipe referenced sidecar-defined mitigations or
    environments (the ``recipe.resolved.yaml`` fallback has no sidecar
    paths to thread back in). Parsing the block directly decouples the
    bundle redaction-resolve from recipe axis validity.

    Returns ``None`` only when a *valid* recipe mapping has no ``redaction:``
    key. A file that does not parse to a mapping at all (empty file, list, or
    scalar -- i.e. a corrupted recipe / ``--redaction-from`` target) raises
    :class:`~aorta.triage.recipe.RecipeSchemaError` rather than silently
    returning ``None``: failing open there would emit an unredacted bundle the
    operator believed was scrubbed. An explicit ``redaction: null`` is likewise
    rejected by :func:`parse_redaction` (a null block is invalid, not "no
    redaction"), matching the probe recipe builder.
    """
    data = load_recipe_mapping(recipe_path)
    if not isinstance(data, dict):
        raise RecipeSchemaError(
            f"recipe {recipe_path}: expected a top-level mapping, got "
            f"{type(data).__name__}; refusing to fall back to no redaction"
        )
    if "redaction" not in data:
        return None
    return parse_redaction(data["redaction"])


def build_redactor_from_recipe(
    recipe_path: Path | None,
    run_dir: Path,
) -> Redactor:
    """Resolve recipe path and return the appropriate :class:`Redactor`.

    Precedence:

    1. Explicit ``--redaction-from`` path when provided.
    2. ``<run-dir>/recipe.resolved.yaml`` when present.
    3. :class:`IdentityRedactor` when neither yields a ``redaction:`` block.
    """
    resolved_path: Path | None = None
    if recipe_path is not None:
        resolved_path = recipe_path
    else:
        fallback = run_dir / _RECIPE_RESOLVED_NAME
        if fallback.is_file():
            resolved_path = fallback
            log.info(
                "aorta bundle: using redaction recipe fallback %s",
                fallback,
            )

    if resolved_path is None:
        return IdentityRedactor()

    cfg = load_redaction_cfg(resolved_path)
    if cfg is None:
        log.info(
            "aorta bundle: recipe %s has no redaction: block; "
            "using IdentityRedactor",
            resolved_path,
        )
        return IdentityRedactor()

    return RedactingRedactor(cfg)


__all__ = [
    "build_redactor_from_recipe",
    "load_redaction_cfg",
]
