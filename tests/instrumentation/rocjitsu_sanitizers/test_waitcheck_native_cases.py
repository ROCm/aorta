"""Local schema/argv checks for draft selector and cap cases; no native execution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from aorta.instrumentation.rocjitsu_sanitizers.models import KernelIdentity
from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe
from aorta.instrumentation.rocjitsu_sanitizers.waitcheck import waitcheck_argv
from aorta.triage.recipe import RecipeSchemaError

ASSETS = Path(__file__).resolve().parents[3] / ".agents/skills/rocm-sanitizer-triage/assets"
CASES = ("invalid-image", "invalid-entry", "cap-high", "cap-low")


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recipe checks cannot launch a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("resolved_image", [False, True])
def test_unresolved_identity_is_not_bypassed(tmp_path, case, resolved_image):
    data = yaml.safe_load((ASSETS / f"waitcheck-{case}.yaml").read_text(encoding="utf-8"))
    if resolved_image:
        data["sanitizer_plan"]["source"]["kernel"]["code_object_index"] = 0
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(RecipeSchemaError, match="entry_offset" if resolved_image else "code_object_index"):
        load_sanitizer_recipe(path)


@pytest.mark.parametrize("case,image,entry,cap", [
    ("invalid-image", 1, 256, 64),
    ("invalid-entry", 0, 20480, 64),
    ("cap-high", 0, 128, 64),
    ("cap-low", 0, 128, 1),
])
def test_resolved_recipe_preserves_explicit_scan_scope(tmp_path, case, image, entry, cap):
    # Selectors here test schema/argv only, not whether a kernel really exists.
    artifact = tmp_path / "inert.co"
    artifact.write_bytes(b"inert local recipe fixture")
    data = yaml.safe_load((ASSETS / f"waitcheck-{case}.yaml").read_text(encoding="utf-8"))
    data["sanitizer_plan"]["source"]["kernel"].update(
        code_object=artifact.name, code_object_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        code_object_index=image, entry_offset=entry,
    )
    path = tmp_path / "prepared.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    recipe = load_sanitizer_recipe(path)
    assert recipe.target == "gfx1250" and recipe.sanitizers == ("waitcheck",)
    assert recipe.scope_kind == "kernel" and recipe.top_n == 1
    assert recipe.timeout_seconds == 120 and recipe.waitcheck_max_diagnostics == cap
    assert len(recipe.kernel_specs) == 1
    identity = KernelIdentity(target=recipe.target, **recipe.kernel_specs[0].to_entry())
    assert identity.exact and not identity.code_object_scan
    argv = waitcheck_argv(Path("rj_waitcheck"), identity, max_diagnostics=cap)
    assert argv[argv.index("--code-object-index") + 1] == str(image)
    assert argv[argv.index("--kernel-entry") + 1] == hex(entry)
    assert argv[argv.index("--max-diagnostics") + 1] == str(cap)


def test_cap_comparison_changes_only_cap_and_metadata():
    high = yaml.safe_load((ASSETS / "waitcheck-cap-high.yaml").read_text(encoding="utf-8"))
    low = yaml.safe_load((ASSETS / "waitcheck-cap-low.yaml").read_text(encoding="utf-8"))
    normalized = deepcopy(low)
    normalized["ticket"], normalized["description"] = high["ticket"], high["description"]
    normalized["sanitizer_plan"]["policy"]["waitcheck_max_diagnostics"] = 64
    assert normalized == high
