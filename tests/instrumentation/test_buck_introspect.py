"""Tests for ``aorta.instrumentation.buck_introspect`` (issue #163, A1.2b).

Strategy mirrors ``test_build_system.py``: load the modules by file path
so the test does not pull in torch via ``aorta.utils``. Every
subprocess and ``shutil.which`` lookup is monkeypatched so tests run
on any host -- with or without buck2.

Coverage matrix (per the A1.2b acceptance criteria):

* ``buck2`` absent -> empty entries, single ``reasons`` line.
* Happy path: fixture JSON yields the expected library set with
  ``source: "buck"`` and the passed-through repo revision.
* Pattern matching: every documented label shape resolves to the
  right library name; unrelated labels resolve to ``None``.
* Failure modes: timeout, OSError, non-zero exit (with stderr
  truncation), invalid JSON, unexpected JSON shape -- all surface as
  ``reasons`` lines without raising.
* ``collect_env()`` integration: buck kwargs wire through, alternates
  are synthesised only for libraries A1 also captures, partial reasons
  propagate, an unexpected exception in introspection is swallowed,
  and the introspection lists round-trip via ``to_dict()`` /
  ``from_dict()``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# -- Direct module load (mirrors test_build_system.py) -----------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, _REPO_ROOT / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bi_mod = _load_module(
    "src/aorta/instrumentation/buck_introspect.py",
    "aorta.instrumentation.buck_introspect",
)
bs_mod = _load_module(
    "src/aorta/instrumentation/build_system.py",
    "aorta.instrumentation.build_system",
)
env_mod = _load_module(
    "src/aorta/instrumentation/environment.py",
    "aorta.instrumentation.environment",
)


FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "buck" / "audit_dependencies.json"
)


def _make_completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["buck2"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# _match_library: pattern matching against the known library list
# ---------------------------------------------------------------------------


class TestMatchLibrary:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("//third-party/rocm:hipblaslt_lib", "hipblaslt"),
            ("//third-party/rocm:hipblaslt-lib", "hipblaslt"),
            ("//some/path:hipblaslt", "hipblaslt"),
            ("hipblaslt-foo:lib", "hipblaslt"),
            ("//third-party/rocm:rccl_lib", "rccl"),
            ("//third-party/rocm:rccl-lib", "rccl"),
            ("//third-party/pytorch:torch", "pytorch"),
            ("//third-party/pytorch:pytorch", "pytorch"),
            ("//third-party/pytorch:torch_hip", "pytorch"),
            ("//third-party/pytorch:torch_cuda", "pytorch"),
            ("//third-party/rocm:hip_runtime", "rocm"),
            ("//third-party/rocm:rocm_runtime", "rocm"),
            ("//myproj/util:logging", None),
            ("//some/other:unrelated_target", None),
            ("", None),
        ],
    )
    def test_pattern_matching(self, label, expected):
        assert bi_mod._match_library(label) == expected


# ---------------------------------------------------------------------------
# introspect_libraries_via_buck: subprocess wrapper end-to-end
# ---------------------------------------------------------------------------


class TestBuck2Absent:
    def test_returns_empty_with_reason_when_buck2_missing(self, monkeypatch):
        monkeypatch.setattr(bi_mod.shutil, "which", lambda _name: None)
        entries, reasons = bi_mod.introspect_libraries_via_buck(
            target="//foo:bar", repo_revision="rev"
        )
        assert entries == []
        assert len(reasons) == 1
        assert "buck2 not on PATH" in reasons[0]
        assert "//foo:bar" in reasons[0]


class TestBuck2HappyPath:
    def test_fixture_yields_expected_library_set(self, monkeypatch):
        monkeypatch.setattr(bi_mod.shutil, "which", lambda _name: "/usr/bin/buck2")
        fixture = FIXTURE_PATH.read_text()
        with patch.object(
            bi_mod.subprocess, "run", return_value=_make_completed(stdout=fixture)
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//myproj:training_main", repo_revision="cafef00d"
            )

        assert reasons == []
        names = sorted(e["name"] for e in entries)
        assert names == ["hipblaslt", "pytorch", "rccl", "rocm"]
        for entry in entries:
            assert entry["source"] == "buck"
            assert entry["revision"] == "cafef00d"
            assert entry["target"].startswith("//")

    def test_dedup_keeps_first_match_only(self, monkeypatch):
        monkeypatch.setattr(bi_mod.shutil, "which", lambda _name: "/usr/bin/buck2")
        payload = {
            "//foo": [
                "//a:hipblaslt_lib",
                "//b:hipblaslt-lib",
            ]
        }
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(stdout=json.dumps(payload)),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision=None
            )
        assert reasons == []
        assert len(entries) == 1
        assert entries[0]["target"] == "//a:hipblaslt_lib"

    def test_revision_passes_through_when_none(self, monkeypatch):
        monkeypatch.setattr(bi_mod.shutil, "which", lambda _name: "/usr/bin/buck2")
        payload = {"//foo": ["//x:rccl_lib"]}
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(stdout=json.dumps(payload)),
        ):
            entries, _ = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision=None
            )
        assert entries[0]["revision"] is None

    def test_empty_match_set_is_not_partial(self, monkeypatch):
        monkeypatch.setattr(bi_mod.shutil, "which", lambda _name: "/usr/bin/buck2")
        payload = {"//foo": ["//util:logging", "//util:metrics"]}
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(stdout=json.dumps(payload)),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r"
            )
        assert entries == []
        assert reasons == []  # success with no matches != failure


class TestBuck2FailureModes:
    def setup_method(self):
        self._patch = patch.object(bi_mod.shutil, "which", lambda _name: "/usr/bin/buck2")
        self._patch.start()

    def teardown_method(self):
        self._patch.stop()

    def test_timeout_returns_reason(self):
        with patch.object(
            bi_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="buck2", timeout=10),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r", timeout=10
            )
        assert entries == []
        assert any("timed out after 10s" in r for r in reasons)

    def test_oserror_returns_reason(self):
        with patch.object(
            bi_mod.subprocess, "run", side_effect=OSError("permission denied")
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r"
            )
        assert entries == []
        assert any("failed to launch" in r for r in reasons)

    def test_nonzero_exit_includes_truncated_stderr(self):
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(
                stdout="", stderr="target //foo not found", returncode=2
            ),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r"
            )
        assert entries == []
        assert any("exited 2" in r and "target //foo not found" in r for r in reasons)

    def test_nonzero_exit_with_empty_stderr_renders_placeholder(self):
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(stdout="", stderr="", returncode=1),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r"
            )
        assert entries == []
        assert any("(empty)" in r for r in reasons)

    def test_invalid_json_returns_reason(self):
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(stdout="not-json"),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r"
            )
        assert entries == []
        assert any("non-JSON output" in r for r in reasons)

    def test_unexpected_json_shape_returns_reason(self):
        with patch.object(
            bi_mod.subprocess,
            "run",
            return_value=_make_completed(stdout=json.dumps(["//a", "//b"])),
        ):
            entries, reasons = bi_mod.introspect_libraries_via_buck(
                target="//foo", repo_revision="r"
            )
        assert entries == []
        assert any("unexpected JSON shape" in r for r in reasons)


# ---------------------------------------------------------------------------
# collect_env() integration: kwargs path + alternates synthesis + round trip
# ---------------------------------------------------------------------------


class TestCollectEnvIntegration:
    """Integration uses the env_mod helper directly to avoid the
    ~150-test all_disabled fixture machinery in test_environment.py.
    The introspection function itself is patched, so we don't need to
    fake any of A1's per-library probes -- we only validate the wiring.
    """

    def test_no_buck_target_yields_empty_lists(self):
        snap = env_mod.collect_env()
        assert snap.library_introspection == []
        assert snap.library_introspection_alternates == []

    def test_buck_target_populates_library_introspection(self, monkeypatch):
        fake_entries = [
            {
                "name": "hipblaslt",
                "source": "buck",
                "revision": "cafef00d",
                "target": "//rocm:hipblaslt_lib",
            },
            {
                "name": "pytorch",
                "source": "buck",
                "revision": "cafef00d",
                "target": "//pytorch:torch",
            },
        ]
        monkeypatch.setattr(
            bi_mod, "introspect_libraries_via_buck", lambda **_: (fake_entries, [])
        )
        snap = env_mod.collect_env(buck_target="//foo:bar")
        assert snap.library_introspection == fake_entries

    def test_alternates_synthesised_only_for_a1_overlap(self, monkeypatch):
        fake_entries = [
            {
                "name": "hipblaslt",
                "source": "buck",
                "revision": "cafef00d",
                "target": "//rocm:hipblaslt_lib",
            },
            {
                "name": "pytorch",
                "source": "buck",
                "revision": "cafef00d",
                "target": "//pytorch:torch",
            },
        ]
        monkeypatch.setattr(
            bi_mod, "introspect_libraries_via_buck", lambda **_: (fake_entries, [])
        )
        snap = env_mod.collect_env(buck_target="//foo:bar")
        names = [a["name"] for a in snap.library_introspection_alternates]
        # hipblaslt has an A1 block, pytorch does not => only one alt
        assert names == ["hipblaslt"]
        assert snap.library_introspection_alternates[0]["source"] == "package"

    def test_buck_reasons_propagate_to_partial_reasons(self, monkeypatch):
        monkeypatch.setattr(
            bi_mod,
            "introspect_libraries_via_buck",
            lambda **_: ([], ["library_introspection: synthetic failure"]),
        )
        snap = env_mod.collect_env(buck_target="//foo:bar")
        assert any(
            "library_introspection: synthetic failure" in r
            for r in snap.partial_reasons
        )

    def test_unexpected_exception_in_introspection_is_swallowed(self, monkeypatch):
        def boom(**_kwargs):
            raise RuntimeError("buck2 went sideways")

        monkeypatch.setattr(bi_mod, "introspect_libraries_via_buck", boom)
        snap = env_mod.collect_env(buck_target="//foo:bar")
        assert snap.library_introspection == []
        assert any(
            "buck introspection raised" in r and "RuntimeError" in r
            for r in snap.partial_reasons
        )

    def test_round_trip_preserves_introspection_lists(self, monkeypatch):
        fake_entries = [
            {
                "name": "rccl",
                "source": "buck",
                "revision": "abc",
                "target": "//rocm:rccl_lib",
            }
        ]
        monkeypatch.setattr(
            bi_mod, "introspect_libraries_via_buck", lambda **_: (fake_entries, [])
        )
        snap = env_mod.collect_env(buck_target="//foo:bar")
        d = snap.to_dict()
        assert d["library_introspection"] == fake_entries
        rebuilt = env_mod.EnvSnapshot.from_dict(d)
        assert rebuilt.library_introspection == fake_entries
        assert (
            rebuilt.library_introspection_alternates
            == snap.library_introspection_alternates
        )

    def test_disconnect_buck_target_without_buck2_kind_records_reason(
        self, monkeypatch
    ):
        # Force build_system to look non-buck so the reconciliation
        # branch fires regardless of whether the host has buck2 installed.
        monkeypatch.setattr(
            env_mod, "_detect_build_system_safe", lambda: {"kind": "none"}
        )
        monkeypatch.setattr(
            bi_mod, "introspect_libraries_via_buck", lambda **_: ([], [])
        )
        snap = env_mod.collect_env(buck_target="//foo:bar")
        assert any(
            "build_system.kind=" in r and "running anyway" in r
            for r in snap.partial_reasons
        )
