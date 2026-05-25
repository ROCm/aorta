"""``aorta probe`` -- wrap-and-collect command for opaque user launch commands.

Phase 1 (issue #188) ships an MVP that:

* loads a ``mode: probe`` recipe;
* synthesises cells from ``mitigation_axis x diagnostic_axis``;
* runs every cell through :func:`aorta.triage.runner.run_recipe`
  (NO parallel runner -- the shared-engine test in
  ``tests/probe/test_shared_engine.py`` enforces this);
* writes per-trial artifacts under
  ``<output>/<safe_slug(ticket)>/<safe_slug(cell)>/trial_<n>/``;
* re-running with the same ``--output`` skips cells whose
  ``result.json`` is valid and carries a non-empty ``verdict``.

Phase 2 (built-in 5-tier classifier + sandboxed ``custom_patterns``) and
Phase 3 (``aorta bundle`` integration + redaction + handout templates)
are tracked separately and are NOT implemented in this module.
"""

from aorta.probe.recipe_builder import (
    ProbeExtras,
    build_probe_recipe_from_dict,
)
from aorta.probe.resume import is_trial_complete

__all__ = [
    "ProbeExtras",
    "build_probe_recipe_from_dict",
    "is_trial_complete",
]
