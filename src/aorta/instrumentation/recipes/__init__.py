"""Recipe emitters for ``aorta env recipe`` (issue #163, A1.2c).

Read-only consumers of env.json that emit text fragments approximating
the captured environment in some build system's own syntax. The output
is BEST-EFFORT, NOT EXACT -- env.json captures observed state, not a
complete build recipe.

Current formats:

* ``buck`` -- emits a BUCK file fragment with one ``prebuilt_cxx_library``
  per ``library_introspection`` entry whose ``source == "buck"``.

The CLI entry point lives in ``src/aorta/cli/env.py`` as
``aorta env recipe --format <format> <env.json>``.

Per the issue spec (`A1.2c out-of-scope`):

* No Buck rule / macro / library code vendored.
* No ``buck2 build`` invocation -- text generation only.
* No reconstruction of internal targets, host driver state, mounted
  source trees, local patches, or private toolchains. Those are not
  recoverable from env.json by construction.
"""

from __future__ import annotations

from aorta.instrumentation.recipes.buck import emit_buck_recipe

__all__ = ["emit_buck_recipe"]
