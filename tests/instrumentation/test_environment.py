"""Tests for ``aorta.instrumentation.environment`` (issue #147 acceptance).

Strategy: load the module by file path so the test does not pull in the
torch-dependent ``aorta.utils`` package. Every subprocess and filesystem
touchpoint is monkeypatched, so tests run on any host without ROCm,
RDHC, hipconfig, hipblaslt, or torch.

Coverage matrix (per the updated A1 spec):

* ``EnvSnapshot`` shape + ``to_dict`` / ``from_dict`` / ``summary``
* ``collect_env`` orchestration: never raises, populates ``partial`` /
  ``partial_reasons``, idempotent
* B1/B2-style integration: snapshot embeds losslessly into a fake trial
  result dict and round-trips
* Per-probe behaviour: RDHC happy/error paths, ROCm version files, HIP
  toolchain, hipBLASLt introspection, runtime context, Docker metadata,
  env vars, PyTorch version
* Schema invariants: required top-level keys, schema_version constant,
  no GPU compute, workload config not captured, ``partial`` reflected in
  the persisted JSON
* CLI invariant: ``cli/env.py`` stays thin
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# -- Direct module load (avoid torch via aorta.utils) -------------------------

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "src",
    "aorta",
    "instrumentation",
    "environment.py",
)
_spec = importlib.util.spec_from_file_location(
    "aorta.instrumentation.environment", _MODULE_PATH
)
env_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = env_mod
_spec.loader.exec_module(env_mod)


collect_env = env_mod.collect_env
EnvSnapshot = env_mod.EnvSnapshot
SCHEMA_VERSION = env_mod.SCHEMA_VERSION
CANONICAL_ENV_VARS = env_mod.CANONICAL_ENV_VARS


# -- Shared fixtures ---------------------------------------------------------


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip env vars that would leak host state into the snapshot."""
    for name in CANONICAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.delenv("SINGULARITY_NAME", raising=False)
    monkeypatch.delenv("AORTA_DOCKER_IMAGE", raising=False)
    monkeypatch.delenv("AORTA_DOCKER_DIGEST", raising=False)
    return monkeypatch


@pytest.fixture
def all_disabled(isolated_env, tmp_path: Path, monkeypatch):
    """Force every external dep into its 'unavailable' branch.

    Result: ``collect_env`` exercises only pure-Python paths and every
    block returns its null-shaped form. Triggers ``partial=True`` with
    one reason per fallback.
    """
    monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "no_rocm")
    monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "no_rocm_dev")
    monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "no_kmd")
    monkeypatch.setattr(
        env_mod, "HIPBLASLT_VERSION_HEADER", tmp_path / "no_header.h"
    )
    monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", tmp_path / "no_libs")
    monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tmp_path / "no_tensile")
    monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
    monkeypatch.setattr(
        env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv"
    )
    monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")
    monkeypatch.setattr(env_mod, "SELF_CGROUP_FILE", tmp_path / "no_self_cgroup")
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
    # Force pytorch import to fail so its fallback path is exercised too
    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict
    ) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    return monkeypatch


# ---------------------------------------------------------------------------
# Schema completeness + versioning
# ---------------------------------------------------------------------------


class TestPathConstants:
    """Structural guard for the filesystem path constants.

    Catches accidental typos / relative-path mistakes; does NOT verify
    that the paths exist on the test host (they are host-state and are
    monkeypatched by every test that uses them).
    """

    @pytest.mark.parametrize(
        "constant_name",
        [
            "ROCM_VERSION_FILE",
            "ROCM_VERSION_DEV_FILE",
            "KMD_VERSION_FILE",
            "HIPBLASLT_VERSION_HEADER",
            "HIPBLASLT_LIB_DIR",
            "HIPBLASLT_TENSILE_DIR",
            "DOCKERENV_MARKER",
            "PODMAN_CONTAINERENV_MARKER",
            "CGROUP_FILE",
            "SELF_CGROUP_FILE",
        ],
    )
    def test_path_is_absolute(self, constant_name: str):
        path = getattr(env_mod, constant_name)
        assert isinstance(path, Path), f"{constant_name} must be a Path"
        assert path.is_absolute(), (
            f"{constant_name} = {path!r} is not absolute. The probe "
            "looks at well-known system locations; relative paths would "
            "be resolved against pytest's CWD and produce nonsense."
        )

    def test_known_constant_set_is_stable(self):
        """Reasoned guard: adding/removing a path constant is a schema
        change that should also touch the structural test above and the
        provenance comments in environment.py.
        """
        path_attrs = {
            name for name in dir(env_mod)
            if isinstance(getattr(env_mod, name, None), Path)
            and not name.startswith("_")
        }
        assert path_attrs == {
            "ROCM_VERSION_FILE",
            "ROCM_VERSION_DEV_FILE",
            "KMD_VERSION_FILE",
            "HIPBLASLT_VERSION_HEADER",
            "HIPBLASLT_LIB_DIR",
            "HIPBLASLT_TENSILE_DIR",
            "DOCKERENV_MARKER",
            "PODMAN_CONTAINERENV_MARKER",
            "CGROUP_FILE",
            "SELF_CGROUP_FILE",
        }, (
            "FS path constants set drifted; update test_path_is_absolute "
            "parametrize list AND the provenance comments in "
            "src/aorta/instrumentation/environment.py."
        )

    def test_self_cgroup_distinct_from_init_cgroup(self):
        """Regression guard: the two cgroup files are different on purpose.

        ``CGROUP_FILE`` (``/proc/1/cgroup``) is the init process's cgroup
        and is sniffed for the runtime *type* (docker/podman/singularity).
        ``SELF_CGROUP_FILE`` (``/proc/self/cgroup``) is the current
        process's cgroup and is parsed for the container *ID*. Conflating
        them would either misclassify the runtime or fail to extract
        an ID inside k8s pods where /proc/1 belongs to the host.
        """
        assert env_mod.CGROUP_FILE != env_mod.SELF_CGROUP_FILE
        assert "1" in env_mod.CGROUP_FILE.parts
        assert "self" in env_mod.SELF_CGROUP_FILE.parts


REQUIRED_TOP_KEYS = {
    "schema_version",
    "captured_at",
    "partial",
    "partial_reasons",
    "system_health",
    "rocm",
    "hip",
    "hipblaslt",
    "runtime_context",
    "docker",
    "env_vars",
    "python_version",
    "pytorch_version",
}


class TestSchemaCompleteness:
    def test_all_top_level_keys_present_when_everything_unavailable(
        self, all_disabled
    ):
        snapshot = collect_env()
        assert set(snapshot.to_dict().keys()) == REQUIRED_TOP_KEYS
        assert snapshot.schema_version == "1.0"
        assert snapshot.system_health is None
        assert snapshot.rocm == {
            "version": None,
            "version_dev": None,
            "kmd_version": None,
        }
        assert snapshot.hip == {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

    def test_schema_version_constant_is_emitted(self, all_disabled):
        snapshot = collect_env()
        assert snapshot.schema_version == SCHEMA_VERSION

    def test_captured_at_is_iso8601_utc(self, all_disabled):
        snapshot = collect_env()
        assert snapshot.captured_at.endswith("Z")
        assert "T" in snapshot.captured_at

    def test_persisted_json_includes_partial_keys(self, all_disabled, tmp_path: Path):
        """``partial`` and ``partial_reasons`` must be present in the on-disk JSON."""
        snapshot = collect_env()
        out = tmp_path / "env.json"
        # Deliberately no default=str -- the schema is supposed to be
        # JSON-native. If anything sneaks in this should fail loudly.
        out.write_text(json.dumps(snapshot.to_dict()))
        on_disk = json.loads(out.read_text())
        assert "partial" in on_disk
        assert "partial_reasons" in on_disk
        assert on_disk["partial"] is True
        assert isinstance(on_disk["partial_reasons"], list)
        assert on_disk["partial_reasons"]  # non-empty since all_disabled


# ---------------------------------------------------------------------------
# EnvSnapshot dataclass: round-trip + summary
# ---------------------------------------------------------------------------


def _example_snapshot(**overrides) -> object:
    """Build a fully-populated EnvSnapshot for round-trip testing."""
    base = {
        "schema_version": "1.0",
        "captured_at": "2026-04-28T12:00:00Z",
        "system_health": {"rdhc_version": "1.4.0", "tests": {}},
        "rocm": {
            "version": "7.2.1",
            "version_dev": "7.2.1-43",
            "kmd_version": "6.16.13",
        },
        "hip": {
            "version": "7.2.5",
            "platform": "amd",
            "compiler": "clang",
            "runtime": "rocclr",
            "cpp_config": "-D__HIP_PLATFORM_AMD__",
        },
        "hipblaslt": {
            "commit": "dabb6df2b9",
            "package_version": "1.2.2",
            "lib_hash": "sha256:abc",
            "tensile_yaml_revision": "filenames-sha256:def",
            "applied_prs": {},
        },
        "runtime_context": {
            "type": "docker",
            "python_env": "venv",
            "venv_path": "/home/u/.venv",
            "conda_env_name": None,
        },
        "docker": {
            "image": "rocm/pytorch:7.2",
            "digest": "sha256:deadbeef",
            "container_id": "abcd1234",
        },
        "env_vars": dict.fromkeys(CANONICAL_ENV_VARS),
        "python_version": "3.12.3",
        "pytorch_version": "2.12.0",
        "partial": False,
        "partial_reasons": [],
    }
    base.update(overrides)
    return EnvSnapshot(**base)


class TestEnvSnapshot:
    def test_to_dict_keys_are_complete(self):
        snap = _example_snapshot()
        d = snap.to_dict()
        assert set(d.keys()) == REQUIRED_TOP_KEYS

    def test_round_trip_via_dict(self):
        original = _example_snapshot()
        rebuilt = EnvSnapshot.from_dict(original.to_dict())
        assert rebuilt == original

    def test_round_trip_via_json(self):
        """B1/B2 path: serialise via JSON, embed in a result, deserialise back."""
        original = _example_snapshot()
        as_json = json.dumps(original.to_dict())
        rebuilt = EnvSnapshot.from_dict(json.loads(as_json))
        assert rebuilt == original

    def test_from_dict_tolerates_extra_keys_forward_compat(self):
        """Future schema additions in env.json shouldn't break old code reading it."""
        d = _example_snapshot().to_dict()
        d["future_field_not_yet_added"] = {"hello": "world"}
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.schema_version == "1.0"

    def test_from_dict_defaults_partial_reasons_when_missing(self):
        """Older env.json without partial_reasons still loads (defaults to [])."""
        d = _example_snapshot().to_dict()
        del d["partial_reasons"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.partial_reasons == []

    def test_summary_includes_partial_marker_when_partial(self):
        partial_snap = _example_snapshot(partial=True, partial_reasons=["x: y"])
        clean_snap = _example_snapshot()
        assert "PARTIAL" in partial_snap.summary()
        assert "PARTIAL" not in clean_snap.summary()

    def test_summary_treats_empty_system_health_as_present(self):
        """Regression guard: RDHC may legitimately return an empty dict
        ``{}`` (subprocess succeeded, nothing to report). The earlier
        truthiness check ``if self.system_health`` would summarise that as
        unavailable -- ``is not None`` is the right check.
        """
        snap_empty = _example_snapshot(system_health={})
        snap_null = _example_snapshot(system_health=None)
        snap_populated = _example_snapshot(system_health={"rdhc_version": "1.4.0"})

        # Empty dict and populated dict should both render as 'present'
        assert "present" in snap_empty.summary()
        assert "unavailable" not in snap_empty.summary()
        assert "present" in snap_populated.summary()
        # Only None should render as 'unavailable'
        assert "unavailable" in snap_null.summary()
        assert "present" not in snap_null.summary()

    def test_summary_is_multiline_human_readable(self):
        snap = _example_snapshot()
        s = snap.summary()
        # 6 lines per the implementation; loose lower-bound
        assert s.count("\n") >= 4
        assert "rocm:" in s
        assert "hipblaslt:" in s

    def test_dataclass_is_frozen(self):
        """Callers can safely embed the snapshot without mutation hazards."""
        snap = _example_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.schema_version = "2.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# collect_env contract: never raises + partial semantics + idempotency
# ---------------------------------------------------------------------------


class TestCollectEnvContract:
    def test_collect_env_never_raises_when_all_probes_fail(self, all_disabled):
        """Acceptance: monkeypatch every probe to fail, still get an EnvSnapshot."""
        snapshot = collect_env()
        assert isinstance(snapshot, EnvSnapshot)
        assert snapshot.partial is True
        assert snapshot.partial_reasons, "partial=True must include at least one reason"

    def test_partial_reasons_have_field_prefixes(self, all_disabled):
        """Each reason should name the field it relates to (e.g. ``rocm.version: ...``)."""
        snapshot = collect_env()
        # Every reason should look like "<top.field>: <cause>" or "<top>: <cause>"
        for reason in snapshot.partial_reasons:
            head = reason.split(":", 1)[0]
            assert head, f"reason missing prefix: {reason!r}"

    def test_partial_false_on_clean_full_probe(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """When every probe succeeds, partial is False and reasons is empty."""
        # Stand up a fully-populated mock environment under tmp.
        rocm_info = tmp_path / ".info"
        rocm_info.mkdir()
        (rocm_info / "version").write_text("7.2.1\n")
        (rocm_info / "version_dev").write_text("7.2.1-43\n")
        kmd = tmp_path / "kmd"
        kmd.write_text("6.16.13\n")

        header_dir = tmp_path / "include" / "hipblaslt"
        header_dir.mkdir(parents=True)
        (header_dir / "hipblaslt-version.h").write_text(
            "#define HIPBLASLT_VERSION_MAJOR 1\n"
            "#define HIPBLASLT_VERSION_MINOR 2\n"
            "#define HIPBLASLT_VERSION_PATCH 2\n"
            "#define HIPBLASLT_VERSION_TWEAK abc1234\n"
        )

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "libhipblaslt.so").write_bytes(b"binary")

        tensile_dir = tmp_path / "tensile"
        tensile_dir.mkdir()
        (tensile_dir / "TensileLibrary_X.dat").write_bytes(b"x")

        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", rocm_info / "version")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", rocm_info / "version_dev")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", kmd)
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_VERSION_HEADER", header_dir / "hipblaslt-version.h"
        )
        monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", lib_dir)
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tensile_dir)
        monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
        monkeypatch.setattr(
            env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv"
        )
        monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")

        # rdhc happy path: pretend it's installed and writes valid JSON
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/" + name)

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sudo" and "rdhc" in cmd[3]:
                Path(cmd[-1]).write_text('{"rdhc_version": "1.4.0"}')
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            if cmd[0] == "hipconfig":
                outs = {
                    "--version": "7.2.5",
                    "--platform": "amd",
                    "--compiler": "clang",
                    "--runtime": "rocclr",
                    "--cpp_config": "-D__HIP_PLATFORM_AMD__",
                }
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=outs[cmd[1]], stderr=""
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        # Pretend torch is importable with a version
        import builtins
        import types

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return types.SimpleNamespace(__version__="2.12.0")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        snapshot = collect_env()
        assert snapshot.partial is False, (
            f"clean probe should not be partial; reasons: {snapshot.partial_reasons}"
        )
        assert snapshot.partial_reasons == []
        # Verify the success values landed
        assert snapshot.rocm["version"] == "7.2.1"
        assert snapshot.hipblaslt["commit"] == "abc1234"
        assert snapshot.system_health == {"rdhc_version": "1.4.0"}
        assert snapshot.hip["version"] == "7.2.5"
        assert snapshot.pytorch_version == "2.12.0"

    def test_idempotent_two_calls_produce_equivalent_snapshots(self, all_disabled):
        """B1 may collect once per trial; B2 may collect once per matrix start.

        Calling twice in the same process must produce equivalent snapshots
        (modulo timestamp). No cross-call state contamination.
        """
        snap1 = collect_env()
        snap2 = collect_env()
        # Compare every field except captured_at (which is a wall-clock stamp)
        d1 = snap1.to_dict()
        d2 = snap2.to_dict()
        d1.pop("captured_at")
        d2.pop("captured_at")
        assert d1 == d2

    def test_baremetal_does_not_trigger_partial_for_docker_block(
        self, isolated_env, monkeypatch, tmp_path: Path
    ):
        """``docker == None`` on baremetal is the documented contract, NOT a fallback."""
        monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
        monkeypatch.setattr(env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv")
        monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")
        # Confirm via the probe directly
        rt = {"type": "baremetal"}
        reasons: list[str] = []
        block = env_mod._capture_docker_metadata(rt, reasons)
        assert block is None
        assert reasons == []  # NOT partial

    def test_unset_env_vars_do_not_trigger_partial(self, isolated_env):
        """Individual env_vars values being None is the documented contract."""
        # All canonical vars are unset (cleared by isolated_env fixture).
        # _capture_env_vars doesn't take a reasons list -- by design.
        block = env_mod._capture_env_vars()
        assert all(v is None for v in block.values())

    def test_runtime_context_never_partial(self, all_disabled):
        """runtime_context.* fields are documented absences, not fallbacks.

        The other top-level blocks (rocm, hipblaslt, etc.) DO show up in
        partial_reasons under all_disabled -- this test only asserts that
        nothing prefixed with ``runtime_context`` ever appears.
        """
        snapshot = collect_env()
        runtime_reasons = [
            r for r in snapshot.partial_reasons if r.startswith("runtime_context")
        ]
        assert runtime_reasons == [], (
            f"runtime_context fields should never trigger partial; got: {runtime_reasons}"
        )

    def test_collect_env_returns_snapshot_when_probe_unexpectedly_raises(
        self, all_disabled, monkeypatch
    ):
        """Hard never-raises guarantee: a probe that raises an unexpected
        exception (i.e. not handled internally) MUST NOT propagate.

        Sabotage `_capture_hipblaslt` to raise. Without the top-level
        try/except in collect_env, this would bubble up and break B1/B2.
        With the guard, we get a fully-shaped EnvSnapshot back, marked
        partial, with the exception captured in partial_reasons.
        """

        def boom(reasons: list[str]) -> dict:
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(env_mod, "_capture_hipblaslt", boom)

        snapshot = collect_env()  # must not raise
        assert isinstance(snapshot, EnvSnapshot)
        assert snapshot.partial is True
        # The unexpected-failure reason is appended to whatever earlier
        # probes already recorded. Find the recovery one specifically.
        recovery_reasons = [
            r for r in snapshot.partial_reasons if r.startswith("collect_env:")
        ]
        assert len(recovery_reasons) == 1
        assert "RuntimeError" in recovery_reasons[0]
        assert "simulated probe failure" in recovery_reasons[0]
        # Schema must still be complete -- callers should not see missing keys
        assert set(snapshot.to_dict().keys()) == REQUIRED_TOP_KEYS

    def test_disaster_snapshot_emits_complete_schema(self):
        """The disaster path must still produce a full env.json shape."""
        snap = env_mod._disaster_snapshot(
            preceding_reasons=["earlier: thing"],
            unexpected_reason="collect_env: boom",
        )
        d = snap.to_dict()
        assert set(d.keys()) == REQUIRED_TOP_KEYS
        assert snap.partial is True
        # Both the earlier reasons and the new disaster reason are present
        assert "earlier: thing" in snap.partial_reasons
        assert "collect_env: boom" in snap.partial_reasons
        # JSON-native check (no default=str needed)
        json.dumps(d)

    def test_disaster_snapshot_populates_every_envsnapshot_field(self):
        """Hard guard against a future PR adding a field to EnvSnapshot
        without updating _disaster_snapshot.

        If a field is added to the dataclass and _disaster_snapshot is not
        updated, the missing-arg ``TypeError`` would fire from inside
        collect_env's ``except`` block, get caught silently, and we'd be
        stuck with a half-broken safety net. This test enumerates the
        dataclass fields and asserts every one is present in the disaster
        snapshot's ``to_dict()`` output.
        """
        from dataclasses import fields as dc_fields

        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="test: dummy"
        )
        snap_dict = snap.to_dict()
        expected_fields = {f.name for f in dc_fields(EnvSnapshot)}
        missing = expected_fields - set(snap_dict.keys())
        assert not missing, (
            f"_disaster_snapshot did not populate fields {missing}. "
            "If you added a field to EnvSnapshot, update _disaster_snapshot "
            "in src/aorta/instrumentation/environment.py to give it a sane "
            "default."
        )

    def test_disaster_snapshot_constructs_when_collect_env_helpers_raise(
        self, monkeypatch
    ):
        """Even the disaster path must not crash if its own helpers blow up.

        Sabotage both ``_utc_now_iso`` and ``platform.python_version`` so
        the disaster fallback's defensive ``try/except`` fires twice.
        Both fields fall back to empty strings; the snapshot is still
        constructible.
        """
        monkeypatch.setattr(
            env_mod, "_utc_now_iso", lambda: (_ for _ in ()).throw(RuntimeError("no time"))
        )
        monkeypatch.setattr(
            env_mod.platform, "python_version",
            lambda: (_ for _ in ()).throw(RuntimeError("no python"))
        )

        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="test: chained failure"
        )
        assert snap.captured_at == ""
        assert snap.python_version == ""
        assert snap.partial is True
        # Schema completeness preserved
        from dataclasses import fields as dc_fields
        assert set(snap.to_dict().keys()) == {f.name for f in dc_fields(EnvSnapshot)}


# ---------------------------------------------------------------------------
# B1 / B2 integration-style: snapshot embeds in a fake trial result
# ---------------------------------------------------------------------------


class TestB1B2Integration:
    """Mirrors how B1 (per-trial runner) and B2 (matrix runner) will use this.

    B1's pattern:
        trial_result = {
            "trial_id": "...",
            "passed": True,
            "metrics": {...},
            "env": collect_env().to_dict(),  # embedded inline
        }
        write(trial_result_json, trial_result)

    B2's pattern (host scope):
        host_env = collect_env()
        write(matrix_dir / "host_env.json", host_env.to_dict())

    Both must round-trip cleanly so post-mortem tools can reconstruct an
    EnvSnapshot from the persisted JSON.
    """

    def test_snapshot_embeds_in_trial_result_and_round_trips(self, all_disabled, tmp_path: Path):
        snapshot = collect_env()

        trial_result = {
            "trial_id": "exp1-trial0",
            "passed": True,
            "metrics": {"loss": 0.42, "step_times_ms": [10.1, 9.8]},
            "env": snapshot.to_dict(),
        }
        out = tmp_path / "trial_result.json"
        out.write_text(json.dumps(trial_result, indent=2))

        loaded = json.loads(out.read_text())
        assert loaded["trial_id"] == "exp1-trial0"
        # Reconstruct the typed snapshot from the embedded dict
        reconstructed = EnvSnapshot.from_dict(loaded["env"])
        assert reconstructed == snapshot

    def test_b2_host_env_file_round_trips(self, all_disabled, tmp_path: Path):
        """B2 writes host_env.json once at matrix start."""
        snapshot = collect_env()
        host_env_path = tmp_path / "host_env.json"
        host_env_path.write_text(json.dumps(snapshot.to_dict()))

        loaded = EnvSnapshot.from_dict(json.loads(host_env_path.read_text()))
        assert loaded == snapshot
        assert loaded.partial == snapshot.partial
        assert loaded.partial_reasons == snapshot.partial_reasons


# ---------------------------------------------------------------------------
# CLI thin-wrapper invariant
# ---------------------------------------------------------------------------


class TestCliIsThinWrapper:
    """Per #147 acceptance: ``src/aorta/cli/env.py`` does no probing of its own
    and stays under ~30 lines of substantive code."""

    @pytest.fixture
    def cli_path(self) -> Path:
        return Path(env_mod.__file__).parent.parent / "cli" / "env.py"

    def test_total_file_size_is_bounded(self, cli_path: Path):
        # Total file budget (incl. docstring/imports/blank lines/error handling).
        # The spec target is ~30 lines of substantive code; the cushion
        # accommodates the module docstring, Click decorators, and the
        # try/except blocks that surface filesystem errors as
        # ``click.ClickException`` (per Copilot review). The real
        # "no-probing-in-CLI" guard is `test_cli_does_no_probing_imports`
        # below -- this one is a soft canary against the file ballooning.
        line_count = sum(1 for _ in cli_path.read_text().splitlines())
        assert line_count <= 80, (
            f"cli/env.py is {line_count} lines; soft budget is 80. "
            "If you need more, check that the new code is genuinely "
            "wiring/error-handling and not probing -- "
            "test_cli_does_no_probing_imports is the strict guard."
        )

    def test_cli_does_no_probing_imports(self, cli_path: Path):
        """CLI must not import anything that would let it probe directly."""
        text = cli_path.read_text()
        forbidden = ["import subprocess", "import shutil", "import platform", "import hashlib"]
        for token in forbidden:
            assert token not in text, (
                f"cli/env.py imports {token!r} -- probing belongs in the library"
            )

    def test_cli_calls_collect_env(self, cli_path: Path):
        """Sanity check: the CLI references the library function."""
        text = cli_path.read_text()
        assert "collect_env" in text

    def test_cli_creates_missing_parent_directory(self, all_disabled, tmp_path: Path):
        """Regression guard: ``-o newdir/env.json`` must work for a non-existent
        parent. With ``click.Path(writable=True)`` Click would reject that
        before our ``mkdir`` ran.
        """
        from click.testing import CliRunner

        # Import the CLI symbol the same way the entrypoint does
        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location("aorta.cli.env", cli_path)
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)

        out_path = tmp_path / "deeply" / "nested" / "new" / "env.json"
        assert not out_path.parent.exists()
        runner = CliRunner()
        result = runner.invoke(cli_mod.env, ["probe", "-o", str(out_path)])
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        # And the JSON is loadable
        json.loads(out_path.read_text())

    def test_cli_surfaces_filesystem_errors_as_click_exception(
        self, all_disabled, tmp_path: Path
    ):
        """Regression guard: an unwritable output path must surface as a
        clean ``click.ClickException``, not a Python traceback.

        Two scenarios: (a) parent ``mkdir`` fails (read-only mount); (b)
        the write itself fails (e.g. parent exists but is not writable).
        Both should yield a non-zero CLI exit + a one-line error
        starting with ``Error:``.
        """
        from click.testing import CliRunner

        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location("aorta.cli.env", cli_path)
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)
        runner = CliRunner()

        # Scenario (a): parent mkdir blows up. Achieve by sabotaging mkdir.
        target = tmp_path / "no_perm" / "env.json"

        original_mkdir = Path.mkdir

        def fake_mkdir(self, *args, **kwargs):
            if "no_perm" in str(self):
                raise PermissionError(13, "Permission denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", fake_mkdir):
            result = runner.invoke(cli_mod.env, ["probe", "-o", str(target)])
        assert result.exit_code != 0
        assert "Failed to create parent directory" in result.output
        # Belt-and-suspenders: no Python traceback header in the output
        assert "Traceback" not in result.output

        # Scenario (b): write itself fails.
        target_b = tmp_path / "env_b.json"
        original_write_text = Path.write_text

        def fake_write_text(self, *args, **kwargs):
            if str(self) == str(target_b.resolve()):
                raise OSError(28, "No space left on device")
            return original_write_text(self, *args, **kwargs)

        with patch.object(Path, "write_text", fake_write_text):
            result = runner.invoke(cli_mod.env, ["probe", "-o", str(target_b)])
        assert result.exit_code != 0
        assert "Failed to write env probe" in result.output
        assert "Traceback" not in result.output

    def test_cli_emits_json_native_output_no_default_str(
        self, all_disabled
    ):
        """Regression guard: the CLI does not pass ``default=str`` to json.dumps.

        Stringifying non-JSON types would mask schema regressions (Path,
        datetime, etc. accidentally leaking into EnvSnapshot fields).
        Verified by:
        1. Confirming ``collect_env()``'s output is json.dumps-able with
           the strict default (no ``default=`` argument).
        2. Scanning the CLI source for the *call-site* pattern -- not the
           bare token, which appears in our explanatory comment.
        """
        snapshot = collect_env()
        # Must succeed without any default= fallback
        json.dumps(snapshot.to_dict())

        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        text = cli_path.read_text()
        # Match the call-site pattern (',' + space + key=val + ')') so we
        # don't false-positive on the comment that documents this very
        # invariant.
        assert ", default=str)" not in text, (
            "cli/env.py json.dumps call uses default=str -- this masks "
            "non-JSON types in the schema. Remove it so the failure is loud."
        )


# ---------------------------------------------------------------------------
# RDHC wrapper
# ---------------------------------------------------------------------------


class TestRdhcWrapper:
    def test_rdhc_unavailable_returns_none_and_records_reason(self, all_disabled):
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        assert any("rdhc" in r for r in reasons)

    def test_rdhc_unavailable_reason_includes_install_hint(self, all_disabled):
        """Operator-facing affordance: the rdhc-not-on-PATH reason must point
        at the install docs so users hitting `system_health: null` for the
        first time know how to fix it without reading source.
        """
        reasons: list[str] = []
        env_mod._run_rdhc(reasons)
        assert any(
            "docs/env-probe.md#installing-rdhc" in r for r in reasons
        ), f"install hint missing from reasons: {reasons}"

    def test_rdhc_present_but_sudo_n_fails_returns_none(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="sudo: a password is required"
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        assert any("sudo" in r.lower() or "exited 1" in r for r in reasons)

    def test_rdhc_nonzero_exit_includes_stderr_in_reason(
        self, isolated_env, monkeypatch
    ):
        """Regression guard: when rdhc fails for a reason OTHER than
        sudo-n-needs-password, the partial_reason must surface the actual
        stderr so operators can debug. The earlier hardcoded
        "(likely sudo-n unavailable)" was misleading.
        """
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=2,
                stdout="",
                stderr="rdhc: ERROR: amdgpu kernel module not loaded\n",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        rdhc_reason = next(r for r in reasons if "system_health" in r)
        assert "exited 2" in rdhc_reason
        assert "amdgpu kernel module not loaded" in rdhc_reason
        # And the misleading boilerplate should NOT be present when stderr was given
        assert "likely sudo-n unavailable" not in rdhc_reason
        # The install hint is also NOT appended when there's actionable stderr
        # -- we don't want to bury a real diagnostic under a generic link.
        assert "docs/env-probe.md#installing-rdhc" not in rdhc_reason

    def test_rdhc_nonzero_exit_no_stderr_keeps_sudo_hint(
        self, isolated_env, monkeypatch
    ):
        """When rdhc prints nothing to stderr (the typical sudo-n no-password
        case), the reason names sudo-n AND points at the install/sudo
        recipe in the docs."""
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=""
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        rdhc_reason = next(r for r in reasons if "system_health" in r)
        assert "no stderr" in rdhc_reason
        assert "sudo-n" in rdhc_reason
        assert "docs/env-probe.md#installing-rdhc" in rdhc_reason

    def test_rdhc_timeout_returns_none(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        assert any("timeout" in r.lower() for r in reasons)

    def test_rdhc_happy_path_returns_parsed_json(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        rdhc_payload = {
            "rdhc_version": "1.4.0",
            "tests": {"gpu_present": "PASS"},
            "general_info": {"hostname": "test-host"},
            "gpu_info": [{"name": "MI300X"}],
            "firmware": [],
        }

        captured: dict[str, Path] = {}

        def fake_run(cmd, **kwargs):
            assert cmd[0] == "sudo"
            assert "-n" in cmd
            assert "--quick" in cmd
            assert "--json" in cmd
            out_path = Path(cmd[-1])
            captured["path"] = out_path
            out_path.write_text(json.dumps(rdhc_payload))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        result = env_mod._run_rdhc(reasons)
        assert result == rdhc_payload
        assert reasons == []  # happy path -> no partial reason
        assert "path" in captured
        assert not captured["path"].exists()  # tempfile cleaned up

    def test_rdhc_malformed_json_returns_none(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_text("not valid json {{{")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        assert any("parseable" in r for r in reasons)

    def test_rdhc_temp_file_cleaned_up_on_failure(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")
        captured: dict[str, Path] = {}

        def fake_run(cmd, **kwargs):
            captured["path"] = Path(cmd[-1])
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="boom"
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        assert env_mod._run_rdhc(reasons) is None
        assert "path" in captured
        assert not captured["path"].exists()

    def test_rdhc_handles_tempfile_oserror(self, isolated_env, monkeypatch):
        """Regression guard: a read-only or full /tmp must not break collect_env.

        Without the try/except around tempfile.NamedTemporaryFile, OSError
        would bubble up and break the never-raises contract.
        """
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def boom(*a, **kw):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(env_mod.tempfile, "NamedTemporaryFile", boom)
        reasons: list[str] = []
        # Must not raise; must record a system_health: reason
        assert env_mod._run_rdhc(reasons) is None
        assert any("temp file" in r for r in reasons)


# ---------------------------------------------------------------------------
# ROCm version files
# ---------------------------------------------------------------------------


class TestRocmVersionFiles:
    def test_all_present_no_reasons(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "version"
        v.write_text("7.2.1\n")
        vdev = tmp_path / "version-dev"
        vdev.write_text("7.2.1.50311-abc1234\n")
        kmd = tmp_path / "kmd_version"
        kmd.write_text("6.16.13\n")

        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", v)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", vdev)
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", kmd)

        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons)
        assert result == {
            "version": "7.2.1",
            "version_dev": "7.2.1.50311-abc1234",
            "kmd_version": "6.16.13",
        }
        assert reasons == []

    def test_all_missing_appends_three_reasons(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "nope1")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "nope2")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "nope3")
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons)
        assert result == {"version": None, "version_dev": None, "kmd_version": None}
        assert len(reasons) == 3
        assert all(r.startswith("rocm.") for r in reasons)

    def test_partial_missing_appends_only_for_missing(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "version"
        v.write_text("7.2.1\n")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", v)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "nope")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "also_nope")
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons)
        assert result["version"] == "7.2.1"
        assert result["version_dev"] is None
        assert result["kmd_version"] is None
        assert len(reasons) == 2
        assert any("version_dev" in r for r in reasons)
        assert any("kmd_version" in r for r in reasons)

    def test_empty_file_treated_as_none(self, tmp_path: Path, monkeypatch):
        empty = tmp_path / "version-dev"
        empty.write_text("")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", empty)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "nope")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "nope")
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons)
        assert result["version_dev"] is None

    def test_non_utf8_file_returns_none_no_raise(self, tmp_path: Path, monkeypatch):
        """Regression guard: a corrupt/non-UTF8 version file must not raise.

        Without the UnicodeDecodeError catch in _read_text_file, a single
        rogue byte in /sys/module/amdgpu/version (or a locale-mismatched
        file) would abort the whole env probe and break the never-raises
        contract.
        """
        bad = tmp_path / "non_utf8"
        bad.write_bytes(b"\xff\xfe\x80not-utf8")  # invalid UTF-8 lead bytes
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", bad)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "nope")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "nope")
        reasons: list[str] = []
        # Must not raise, must return None for the bad file
        result = env_mod._capture_rocm_version_files(reasons)
        assert result["kmd_version"] is None


# ---------------------------------------------------------------------------
# HIP toolchain
# ---------------------------------------------------------------------------


class TestHipToolchain:
    def test_hipconfig_missing_returns_all_none_and_one_reason(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        reasons: list[str] = []
        result = env_mod._capture_hip_toolchain(reasons)
        assert all(v is None for v in result.values())
        assert len(reasons) == 1
        assert "hip" in reasons[0]

    def test_hipconfig_happy_path_no_reasons(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/hipconfig")

        outputs = {
            "--version": "7.2.53211-e1a6bc5663",
            "--platform": "amd",
            "--compiler": "clang",
            "--runtime": "rocclr",
            "--cpp_config": "-D__HIP_PLATFORM_AMD__",
        }

        def fake_run(cmd, **kwargs):
            assert cmd[0] == "hipconfig"
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=outputs[cmd[1]] + "\n", stderr=""
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        result = env_mod._capture_hip_toolchain(reasons)
        assert result["version"] == "7.2.53211-e1a6bc5663"
        assert reasons == []

    def test_hipconfig_one_field_fails_appends_one_reason(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/hipconfig")

        def fake_run(cmd, **kwargs):
            if cmd[1] == "--cpp_config":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="boom"
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ok\n", stderr=""
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        result = env_mod._capture_hip_toolchain(reasons)
        assert result["cpp_config"] is None
        assert len(reasons) == 1
        assert "cpp_config" in reasons[0]


# ---------------------------------------------------------------------------
# hipBLASLt introspection
# ---------------------------------------------------------------------------


class TestHipblasltHeaderParsing:
    def test_parse_full_header(self):
        text = """
        #ifndef _HIPBLASLT_VERSION_H_
        #define _HIPBLASLT_VERSION_H_
        #define HIPBLASLT_VERSION_MAJOR     1
        #define HIPBLASLT_VERSION_MINOR     2
        #define HIPBLASLT_VERSION_PATCH     2
        #define HIPBLASLT_VERSION_TWEAK     dabb6df2b9
        #endif
        """
        commit, version = env_mod._parse_hipblaslt_header(text)
        assert commit == "dabb6df2b9"
        assert version == "1.2.2"

    def test_parse_missing_tweak_returns_none_commit(self):
        text = """
        #define HIPBLASLT_VERSION_MAJOR 1
        #define HIPBLASLT_VERSION_MINOR 2
        #define HIPBLASLT_VERSION_PATCH 0
        """
        commit, version = env_mod._parse_hipblaslt_header(text)
        assert commit is None
        assert version == "1.2.0"

    def test_parse_empty_returns_none_pair(self):
        commit, version = env_mod._parse_hipblaslt_header("")
        assert (commit, version) == (None, None)
        commit, version = env_mod._parse_hipblaslt_header(None)
        assert (commit, version) == (None, None)


class TestHipblasltLibHash:
    def test_hash_resolved_through_symlink(self, tmp_path: Path, monkeypatch):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        real = lib_dir / "libhipblaslt.so.1.2.70201"
        real.write_bytes(b"hello hipblaslt")
        symlink_a = lib_dir / "libhipblaslt.so.1"
        symlink_b = lib_dir / "libhipblaslt.so"
        symlink_a.symlink_to(real.name)
        symlink_b.symlink_to(symlink_a.name)

        monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", lib_dir)
        digest = env_mod._hash_hipblaslt_library()
        expected = "sha256:" + hashlib.sha256(b"hello hipblaslt").hexdigest()
        assert digest == expected

    def test_no_library_returns_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", tmp_path / "empty")
        assert env_mod._hash_hipblaslt_library() is None


class TestTensileFingerprint:
    def test_fingerprint_changes_when_filenames_change(
        self, tmp_path: Path, monkeypatch
    ):
        d = tmp_path / "library"
        d.mkdir()
        (d / "TensileLibrary_A.dat").write_bytes(b"x")
        (d / "TensileLibrary_B.dat").write_bytes(b"y")
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", d)
        fp1 = env_mod._tensile_fingerprint()
        assert fp1 is not None and fp1.startswith("filenames-sha256:")

        (d / "TensileLibrary_C.dat").write_bytes(b"z")
        fp2 = env_mod._tensile_fingerprint()
        assert fp2 != fp1

    def test_no_dir_returns_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tmp_path / "nope")
        assert env_mod._tensile_fingerprint() is None

    def test_dir_with_no_kernel_files_returns_none(
        self, tmp_path: Path, monkeypatch
    ):
        d = tmp_path / "library"
        d.mkdir()
        (d / "README.txt").write_text("not a kernel file")
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", d)
        assert env_mod._tensile_fingerprint() is None


class TestHipblasltBlockShape:
    def test_applied_prs_is_empty_dict_initially(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_hipblaslt(reasons)
        assert block["applied_prs"] == {}

    def test_block_keys_stable(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_hipblaslt(reasons)
        assert set(block.keys()) == {
            "commit",
            "package_version",
            "lib_hash",
            "tensile_yaml_revision",
            "applied_prs",
        }

    def test_partial_reasons_contain_hipblaslt_prefix(self, all_disabled):
        reasons: list[str] = []
        env_mod._capture_hipblaslt(reasons)
        assert all(r.startswith("hipblaslt.") for r in reasons)

    def test_reason_when_header_unreadable(self, all_disabled):
        """Header file missing -> reason should say 'not readable'."""
        reasons: list[str] = []
        env_mod._capture_hipblaslt(reasons)
        commit_reason = next(r for r in reasons if r.startswith("hipblaslt.commit"))
        assert "not readable" in commit_reason

    def test_reason_when_header_present_but_tweak_missing(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Header readable but no TWEAK define -> reason names that explicitly.

        Regression guard: prior to this fix, both failure modes used the
        same "not readable" reason -- misleading when the header *was*
        readable but missing the specific define.
        """
        header = tmp_path / "hipblaslt-version.h"
        header.write_text(
            "#define HIPBLASLT_VERSION_MAJOR 1\n"
            "#define HIPBLASLT_VERSION_MINOR 2\n"
            "#define HIPBLASLT_VERSION_PATCH 0\n"
            # Note: no HIPBLASLT_VERSION_TWEAK -- a real-world case where a
            # build config emits MAJOR/MINOR/PATCH only.
        )
        monkeypatch.setattr(env_mod, "HIPBLASLT_VERSION_HEADER", header)
        monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", tmp_path / "no_libs")
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tmp_path / "no_tensile")

        reasons: list[str] = []
        block = env_mod._capture_hipblaslt(reasons)
        # commit failed but package_version succeeded
        assert block["commit"] is None
        assert block["package_version"] == "1.2.0"
        commit_reason = next(r for r in reasons if r.startswith("hipblaslt.commit"))
        assert "not readable" not in commit_reason
        assert "HIPBLASLT_VERSION_TWEAK" in commit_reason


# ---------------------------------------------------------------------------
# Runtime context detection
# ---------------------------------------------------------------------------


class TestRuntimeContext:
    def test_baremetal_no_markers(self, all_disabled, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", sys.prefix)
        rt = env_mod._detect_runtime_context()
        assert rt["type"] == "baremetal"
        assert rt["python_env"] == "system"
        assert rt["venv_path"] is None
        assert rt["conda_env_name"] is None

    def test_docker_via_dockerenv_marker(self, all_disabled, tmp_path: Path):
        marker = tmp_path / ".dockerenv"
        marker.write_text("")
        all_disabled.setattr(env_mod, "DOCKERENV_MARKER", marker)
        assert env_mod._detect_container_type() == "docker"

    def test_podman_via_containerenv_marker(self, all_disabled, tmp_path: Path):
        marker = tmp_path / ".containerenv"
        marker.write_text("engine=podman")
        all_disabled.setattr(env_mod, "PODMAN_CONTAINERENV_MARKER", marker)
        assert env_mod._detect_container_type() == "podman"

    def test_singularity_via_env_var(self, all_disabled):
        all_disabled.setenv("SINGULARITY_NAME", "myapp.sif")
        assert env_mod._detect_container_type() == "singularity"

    def test_docker_via_cgroup_fallback(self, all_disabled, tmp_path: Path):
        cgroup = tmp_path / "cgroup"
        cgroup.write_text("12:freezer:/docker/abc123def456\n0::/init.scope\n")
        all_disabled.setattr(env_mod, "CGROUP_FILE", cgroup)
        assert env_mod._detect_container_type() == "docker"

    def test_podman_via_cgroup_fallback(self, all_disabled, tmp_path: Path):
        cgroup = tmp_path / "cgroup"
        cgroup.write_text("0::/machine.slice/libpod-podman-abc.scope\n")
        all_disabled.setattr(env_mod, "CGROUP_FILE", cgroup)
        assert env_mod._detect_container_type() == "podman"

    def test_dockerenv_takes_precedence_over_cgroup(
        self, all_disabled, tmp_path: Path
    ):
        marker = tmp_path / ".dockerenv"
        marker.write_text("")
        cgroup = tmp_path / "cgroup"
        cgroup.write_text("0::/machine.slice/libpod-podman-abc.scope\n")
        all_disabled.setattr(env_mod, "DOCKERENV_MARKER", marker)
        all_disabled.setattr(env_mod, "CGROUP_FILE", cgroup)
        assert env_mod._detect_container_type() == "docker"

    def test_singularity_wins_over_docker_in_cgroup_fallback(
        self, all_disabled, tmp_path: Path
    ):
        """Regression guard: when /proc/1/cgroup mentions both 'singularity'
        and 'docker' (e.g. a Singularity instance whose underlying cgroup
        was created by a docker-shim), the documented precedence says
        Singularity wins. Earlier code iterated docker first and would
        misclassify.
        """
        cgroup = tmp_path / "cgroup"
        cgroup.write_text(
            "12:freezer:/docker/abc123\n"
            "0::/singularity/instance-xyz\n"
        )
        all_disabled.setattr(env_mod, "CGROUP_FILE", cgroup)
        assert env_mod._detect_container_type() == "singularity"

    def test_singularity_wins_over_podman_in_cgroup_fallback(
        self, all_disabled, tmp_path: Path
    ):
        """Same precedence rule against podman tokens."""
        cgroup = tmp_path / "cgroup"
        cgroup.write_text(
            "0::/machine.slice/libpod-podman-xxx.scope\n"
            "0::/singularity/instance-xyz\n"
        )
        all_disabled.setattr(env_mod, "CGROUP_FILE", cgroup)
        assert env_mod._detect_container_type() == "singularity"

    def test_python_env_venv(self, isolated_env, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "prefix", "/tmp/myvenv")
        assert env_mod._detect_python_env() == "venv"

    def test_python_env_conda(self, isolated_env):
        isolated_env.setenv("CONDA_DEFAULT_ENV", "myenv")
        assert env_mod._detect_python_env() == "conda"

    def test_python_env_system(self, isolated_env, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", sys.prefix)
        assert env_mod._detect_python_env() == "system"

    def test_runtime_context_venv_path_populated(self, all_disabled, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "prefix", "/home/user/.venv")
        rt = env_mod._detect_runtime_context()
        assert rt["python_env"] == "venv"
        assert rt["venv_path"] == "/home/user/.venv"
        assert rt["conda_env_name"] is None

    def test_runtime_context_conda_name_populated(self, all_disabled):
        all_disabled.setenv("CONDA_DEFAULT_ENV", "rocm-7.2")
        rt = env_mod._detect_runtime_context()
        assert rt["python_env"] == "conda"
        assert rt["conda_env_name"] == "rocm-7.2"
        assert rt["venv_path"] is None


# ---------------------------------------------------------------------------
# Docker metadata
# ---------------------------------------------------------------------------


class TestDockerMetadata:
    def test_baremetal_returns_none_no_reasons(self):
        reasons: list[str] = []
        assert env_mod._capture_docker_metadata({"type": "baremetal"}, reasons) is None
        assert reasons == []

    def test_docker_picks_up_aorta_env_vars(self, isolated_env):
        isolated_env.setenv("AORTA_DOCKER_IMAGE", "rocm/pytorch:7.2")
        isolated_env.setenv("AORTA_DOCKER_DIGEST", "sha256:deadbeef")
        reasons: list[str] = []
        block = env_mod._capture_docker_metadata({"type": "docker"}, reasons)
        assert block["image"] == "rocm/pytorch:7.2"
        assert block["digest"] == "sha256:deadbeef"
        assert reasons == []  # both populated -> no partial

    def test_docker_in_container_without_env_vars_appends_reasons(self, isolated_env):
        reasons: list[str] = []
        block = env_mod._capture_docker_metadata({"type": "docker"}, reasons)
        assert set(block.keys()) == {"image", "digest", "container_id"}
        assert block["image"] is None
        assert block["digest"] is None
        # Both image and digest missing -> two reasons
        assert len(reasons) == 2
        assert any("image" in r for r in reasons)
        assert any("digest" in r for r in reasons)

    def test_container_id_extracted_from_cgroup(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # Cleaner now: monkeypatch SELF_CGROUP_FILE directly instead of
        # the global _read_text_file helper. Exercises the same
        # constant the production code reads.
        cgroup = tmp_path / "self_cgroup"
        cid = "abc123def456789012345678901234567890abcd"
        cgroup.write_text(f"12:freezer:/docker/{cid}\n")
        monkeypatch.setattr(env_mod, "SELF_CGROUP_FILE", cgroup)
        reasons: list[str] = []
        # Provide image/digest so they don't add their own reasons
        isolated_env.setenv("AORTA_DOCKER_IMAGE", "x")
        isolated_env.setenv("AORTA_DOCKER_DIGEST", "y")
        block = env_mod._capture_docker_metadata({"type": "docker"}, reasons)
        assert block["container_id"] == cid

    def test_container_id_returns_none_when_self_cgroup_missing(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """If /proc/self/cgroup isn't readable (e.g. heavily sandboxed
        container), container_id is None but the function does not raise."""
        monkeypatch.setattr(env_mod, "SELF_CGROUP_FILE", tmp_path / "no_self_cgroup")
        assert env_mod._read_container_id() is None


# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------


class TestEnvVars:
    def test_canonical_vars_captured_when_set(self, isolated_env):
        isolated_env.setenv("HSA_XNACK", "1")
        isolated_env.setenv("GPU_MAX_HW_QUEUES", "4")
        isolated_env.setenv("FBGEMM_TBE_V2", "1")
        result = env_mod._capture_env_vars()
        assert result["HSA_XNACK"] == "1"
        assert result["GPU_MAX_HW_QUEUES"] == "4"
        assert result["FBGEMM_TBE_V2"] == "1"

    def test_canonical_vars_null_when_unset(self, isolated_env):
        result = env_mod._capture_env_vars()
        for var in CANONICAL_ENV_VARS:
            assert result[var] is None, f"{var} should be None when unset"

    def test_workload_config_vars_are_not_captured(self, isolated_env):
        # Per acceptance criteria, these are workload state, not env probe state
        isolated_env.setenv("AMP_DTYPE", "bf16")
        isolated_env.setenv("MODEL_DTYPE", "fp32")
        isolated_env.setenv("SHAMPOO_PRECONDITIONER_DTYPE", "fp64")
        result = env_mod._capture_env_vars()
        for forbidden in ("AMP_DTYPE", "MODEL_DTYPE", "SHAMPOO_PRECONDITIONER_DTYPE"):
            assert forbidden not in result, f"{forbidden} leaked into env_vars"

    def test_canonical_var_names_stable(self):
        # Reasoned guard: changing this list is a schema change. If a future
        # PR adds a var, this test forces an explicit acknowledgement.
        assert set(CANONICAL_ENV_VARS) == {
            "HSA_XNACK",
            "HSA_KERNARG_POOL_SIZE",
            "HSA_NO_SCRATCH_RECLAIM",
            "GPU_MAX_HW_QUEUES",
            "AMDGCN_USE_BUFFER_OPS",
            "DISABLE_TF32",
            "NCCL_MAX_NCHANNELS",
            "FBGEMM_NO_JK",
            "FBGEMM_TBE_V2",
            "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL",
            "FBGEMM_BOUNDS_CHECK_INDICES_V2",
            "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
            "PYTORCH_CUDA_ALLOC_CONF",
        }


# ---------------------------------------------------------------------------
# PyTorch version + 'no GPU compute' guard
# ---------------------------------------------------------------------------


class TestPytorchVersion:
    def test_torch_unavailable_returns_none_and_records_reason(self, isolated_env):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            reasons: list[str] = []
            assert env_mod._capture_pytorch_version(reasons) is None
            assert any("torch" in r for r in reasons)

    def test_torch_present_without_version_returns_none_not_string(
        self, isolated_env
    ):
        """Regression guard: never emit the string "None" as the version."""
        import builtins
        import types

        real_import = builtins.__import__
        fake_torch = types.SimpleNamespace()  # no __version__ attr

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            reasons: list[str] = []
            result = env_mod._capture_pytorch_version(reasons)
        assert result is None
        assert result != "None"
        assert any("__version__" in r for r in reasons)

    def test_torch_with_version_returns_string_no_reason(self, isolated_env):
        import builtins
        import types

        real_import = builtins.__import__
        fake_torch = types.SimpleNamespace(__version__="2.12.0")

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            reasons: list[str] = []
            assert env_mod._capture_pytorch_version(reasons) == "2.12.0"
            assert reasons == []


class TestNoGpuCompute:
    """Guard against introducing GPU work into the env probe.

    True GPU-zero verification is via rocprofv3 in CI; here we assert
    that the orchestrator never reaches into ``torch.cuda`` (which would
    initialise a HIP context).
    """

    def test_torch_cuda_never_called(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Inject a fake torch into ``sys.modules`` so the test runs even
        when the host venv has no torch installed (the previous
        sys.modules-sniff version always skipped, defeating the guard).

        Then exercise the full ``collect_env()`` orchestration with all
        external probes disabled, and assert that ``torch.cuda.is_available``
        and ``torch.cuda.device_count`` were never called.
        """
        import types

        fake_cuda = types.SimpleNamespace(
            is_available=MagicMock(name="is_available"),
            device_count=MagicMock(name="device_count"),
        )
        fake_torch = types.SimpleNamespace(__version__="2.12.0", cuda=fake_cuda)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        # Disable external probes (rdhc, hipconfig, rocm files, hipblaslt,
        # container markers) so the test is fast and host-independent.
        # Deliberately NOT using the `all_disabled` fixture here -- that
        # one sabotages `import torch` and would defeat the whole test.
        for attr in (
            "ROCM_VERSION_FILE", "ROCM_VERSION_DEV_FILE", "KMD_VERSION_FILE",
            "HIPBLASLT_VERSION_HEADER", "HIPBLASLT_LIB_DIR",
            "HIPBLASLT_TENSILE_DIR", "DOCKERENV_MARKER",
            "PODMAN_CONTAINERENV_MARKER", "CGROUP_FILE", "SELF_CGROUP_FILE",
        ):
            monkeypatch.setattr(env_mod, attr, tmp_path / f"no_{attr.lower()}")
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)

        snapshot = collect_env()

        # Sanity: probe ran, picked up our fake torch's version
        assert snapshot.pytorch_version == "2.12.0"

        # The actual guard
        fake_cuda.is_available.assert_not_called()
        fake_cuda.device_count.assert_not_called()
