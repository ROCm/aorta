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
# Run-state fallback. ``recipe.resolved.yaml`` is emitted in the triage shape so
# it stays loadable by ``load_recipe``, and ``redaction:`` is a probe-mode-only
# key -- so the resolved recipe of a real probe run never carries the block.
# ``matrix.json`` records the rule that was actually in force; without this
# second candidate, ``aorta bundle <run-dir>`` silently produced an unredacted
# bundle for every run that did not pass ``--redaction-from``.
_MATRIX_JSON_NAME = "matrix.json"


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

    1. Explicit ``--redaction-from`` path when provided. An explicit path is
       authoritative: if it carries no ``redaction:`` block, no fallback is
       consulted.
    2. ``<run-dir>/recipe.resolved.yaml``, then ``<run-dir>/matrix.json`` --
       the first that carries a ``redaction:`` block wins.
    3. :class:`IdentityRedactor` when none yields a block.
    """
    if recipe_path is not None:
        cfg = load_redaction_cfg(recipe_path)
        if cfg is None:
            log.info(
                "aorta bundle: recipe %s has no redaction: block; "
                "using IdentityRedactor",
                recipe_path,
            )
            return IdentityRedactor()
        return RedactingRedactor(cfg, run_root=run_dir)

    for name in (_RECIPE_RESOLVED_NAME, _MATRIX_JSON_NAME):
        candidate = run_dir / name
        if not candidate.is_file():
            continue
        cfg = load_redaction_cfg(candidate)
        if cfg is None:
            continue
        log.info("aorta bundle: using redaction block from %s", candidate)
        return RedactingRedactor(cfg, run_root=run_dir)

    log.info(
        "aorta bundle: no redaction: block found under %s; using IdentityRedactor",
        run_dir,
    )
    return IdentityRedactor()


__all__ = [
    "build_redactor_from_recipe",
    "load_redaction_cfg",
]
