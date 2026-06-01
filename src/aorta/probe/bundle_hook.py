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
from aorta.probe.redaction import RedactingRedactor, RedactionCfg
from aorta.triage.recipe import load_recipe

log = logging.getLogger(__name__)

_RECIPE_RESOLVED_NAME = "recipe.resolved.yaml"


def load_redaction_cfg(recipe_path: Path) -> RedactionCfg | None:
    """Load ``redaction:`` from a probe-mode recipe file."""
    recipe = load_recipe(recipe_path)
    if recipe.probe_extras is None or recipe.probe_extras.redaction is None:
        return None
    cfg: RedactionCfg = recipe.probe_extras.redaction
    return cfg


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
