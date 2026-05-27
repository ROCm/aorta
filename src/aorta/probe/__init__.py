"""``aorta probe`` -- wrap-and-collect command for opaque user launch commands.

Phase 1 (issue #188) shipped the MVP:

* loads a ``mode: probe`` recipe;
* synthesises cells from ``mitigation_axis x diagnostic_axis``;
* runs every cell through :func:`aorta.triage.runner.run_recipe`
  (NO parallel runner -- the shared-engine test in
  ``tests/probe/test_shared_engine.py`` enforces this);
* writes per-trial artifacts under
  ``<output>/<safe_slug(ticket)>/<safe_slug(cell)>/trial_<n>/``;
* re-running with the same ``--output`` skips cells whose
  ``result.json`` is valid and carries a non-empty ``verdict``.

Phase 2 has shipped on top of Phase 1: the five-tier classifier lives
in :mod:`aorta.probe.classifier` and the AST-whitelisted ``condition``
evaluator in :mod:`aorta.probe.sandbox`. ``custom_patterns`` are
compiled at recipe-load time and resolved post-exit by the workload;
detector IDs land in ``result.json::failure_detectors_fired`` /
``warn_detectors_fired``.

Phase 3 (``aorta bundle`` integration + redaction + handout templates)
is tracked separately (issue #196) and is NOT yet implemented.
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
