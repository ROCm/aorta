"""Validate the shipped draft recipe without running a checker or a kernel.

Run: python -m pytest tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_skill_recipe.py
The temporary object is inert text, not a qualification control.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe
from aorta.triage.recipe import RecipeSchemaError

_REPO = Path(__file__).resolve().parents[3]
_TEMPLATE = (
    _REPO / ".agents/skills/rocm-sanitizer-triage/assets/waitcheck-exact-entry.yaml"
)


@pytest.mark.parametrize("resolved_index", [False, True])
def test_unresolved_exact_entry_cannot_load(tmp_path: Path, resolved_index: bool) -> None:
    data = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    if resolved_index:
        data["sanitizer_plan"]["source"]["kernel"]["code_object_index"] = 0
    recipe_path = tmp_path / "draft.yaml"
    recipe_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    field = "entry_offset" if resolved_index else "code_object_index"
    with pytest.raises(RecipeSchemaError, match=field):
        load_sanitizer_recipe(recipe_path)


def test_resolved_template_preserves_exact_identity_and_bounds(tmp_path: Path) -> None:
    # Only exercise recipe loading. These bytes/offsets are synthetic, not GPU code.
    payload = b"inert recipe identity fixture"
    (tmp_path / "fixture.co").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    data = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    plan = data["sanitizer_plan"]
    plan["target"] = "gfx1250"
    plan["source"]["kernel"] = {
        "name": "synthetic_entry",
        "code_object": "fixture.co",
        "code_object_sha256": digest,
        "code_object_index": 1,
        "entry_offset": "0x40",
    }
    recipe_path = tmp_path / "prepared.yaml"
    recipe_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    recipe = load_sanitizer_recipe(recipe_path)
    identity = recipe.kernel_specs[0]
    assert identity.code_object == str((tmp_path / "fixture.co").resolve())
    assert identity.code_object_sha256 == digest
    assert identity.code_object_index == 1
    assert identity.entry_offset == 0x40
    assert recipe.sanitizers == ("waitcheck",)
    assert recipe.top_n == 1
    assert recipe.timeout_seconds == 120
    assert recipe.waitcheck_max_diagnostics == 64
    assert recipe.report_name == "sanitizer_report.json"
