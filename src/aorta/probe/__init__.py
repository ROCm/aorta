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

Phase 2: five-tier classifier in :mod:`aorta.probe.classifier` and the
AST-whitelisted ``condition`` evaluator in :mod:`aorta.probe.sandbox`.

Phase 3: redaction scrubbers in :mod:`aorta.probe.redaction`, bundle
integration via :mod:`aorta.probe.bundle_hook`, and handout templates
under ``recipes/probe-template-*.yaml``.
"""

from aorta.probe.bundle_hook import build_redactor_from_recipe, load_redaction_cfg
from aorta.probe.recipe_builder import (
    ProbeExtras,
    build_probe_recipe_from_dict,
)
from aorta.probe.redaction import RedactingRedactor, RedactionCfg, parse_redaction
from aorta.probe.resume import is_trial_complete

__all__ = [
    "ProbeExtras",
    "RedactionCfg",
    "RedactingRedactor",
    "build_probe_recipe_from_dict",
    "build_redactor_from_recipe",
    "is_trial_complete",
    "load_redaction_cfg",
    "parse_redaction",
]
