"""Check draft two-kernel recipes without compiling, launching or scanning code."""
from __future__ import annotations

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

ASSETS = Path(__file__).resolve().parents[3] / "examples/sanitizers/waitcheck/assets"


@pytest.fixture(autouse=True)
def prohibit_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("these recipe checks must not launch any process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)


@pytest.mark.parametrize("label", ["missing", "corrected"])
@pytest.mark.parametrize("resolved_index", [False, True])
def test_unresolved_isolation_selector_is_refused(tmp_path, label, resolved_index):
    data = yaml.safe_load((ASSETS / f"waitcheck-isolation-{label}.yaml").read_text(encoding="utf-8"))
    if resolved_index:
        data["sanitizer_plan"]["source"]["kernel"]["code_object_index"] = 0
    path = tmp_path / "unresolved.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(RecipeSchemaError, match="entry_offset" if resolved_index else "code_object_index"):
        load_sanitizer_recipe(path)


def test_prepared_pair_selects_distinct_entries_in_same_object(tmp_path):
    # Inert local data and synthetic selectors test the recipe/argv contract only.
    # These offsets are NOT proposed entry values for the HIP source.
    artifact = tmp_path / "two-kernel-fixture.co"
    artifact.write_bytes(b"inert shared-object recipe identity fixture")
    fingerprint = hashlib.sha256(artifact.read_bytes()).hexdigest()
    identities, argv_by_label = {}, {}
    for label, entry in (("missing", 0), ("corrected", 0x80)):
        data = yaml.safe_load((ASSETS / f"waitcheck-isolation-{label}.yaml").read_text(encoding="utf-8"))
        spec = data["sanitizer_plan"]["source"]["kernel"]
        spec.update(code_object=artifact.name, code_object_sha256=fingerprint,
                    code_object_index=2, entry_offset=entry)
        path = tmp_path / f"{label}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        loaded = load_sanitizer_recipe(path)
        assert loaded.target == "gfx1250"
        assert loaded.sanitizers == ("waitcheck",)
        assert loaded.scope_kind == "kernel" and loaded.top_n == 1
        assert loaded.timeout_seconds == 120 and loaded.waitcheck_max_diagnostics == 64
        assert len(loaded.kernel_specs) == 1
        kernel = loaded.kernel_specs[0]
        identity = KernelIdentity(target=loaded.target, **kernel.to_entry())
        assert identity.name == f"waitcheck_{label}"
        assert identity.exact and not identity.code_object_scan
        assert identity.entry_offset == entry
        identities[label] = identity
        command = waitcheck_argv(Path("rj_waitcheck"), identity, max_diagnostics=loaded.waitcheck_max_diagnostics)
        assert command[1] == str(artifact.resolve())
        assert command[command.index("--code-object-index") + 1] == "2"
        assert command[command.index("--kernel-entry") + 1] == hex(entry)
        assert command[command.index("--max-diagnostics") + 1] == "64"
        argv_by_label[label] = command
    missing, corrected = identities["missing"], identities["corrected"]
    assert missing.code_object == corrected.code_object
    assert missing.code_object_sha256 == corrected.code_object_sha256 == fingerprint
    assert missing.code_object_index == corrected.code_object_index == 2
    assert missing.entry_offset != corrected.entry_offset
    assert missing.stable_key != corrected.stable_key
    assert argv_by_label["missing"] != argv_by_label["corrected"]
