"""Triage matrix runner (B2): recipe + flag mode.

Public API:

* :func:`aorta.triage.runner.run_recipe` -- the shared execution path.
* :func:`aorta.triage.recipe.load_recipe` -- YAML / JSON loader.
* :func:`aorta.triage.recipe.build_recipe_from_flags` -- flag-shim builder.

Matrix-mode only for MVP per D11; optimize mode is deferred to P1. See issue
#151 and ``recipes/README.md`` for the full reference.
"""

from aorta.triage.recipe import (
    Cell,
    ConfoundCfg,
    InlineEnv,
    Recipe,
    RecipeCellError,
    RecipeSchemaError,
    build_recipe_from_flags,
    load_recipe,
)
from aorta.triage.runner import run_recipe

__all__ = [
    "Cell",
    "ConfoundCfg",
    "InlineEnv",
    "Recipe",
    "RecipeCellError",
    "RecipeSchemaError",
    "build_recipe_from_flags",
    "load_recipe",
    "run_recipe",
]
