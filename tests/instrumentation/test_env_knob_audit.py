"""Tests for scripts/audit_env_knobs.py -- the env-knob coverage audit.

The audit is the only check that can show the *installed* libraries are covered,
so its own logic has to be trustworthy. These tests exercise it against fixture
"libraries" (plain files containing env-var-shaped strings), which keeps them in
the CPU gate: no ROCm install, no real shared objects.

The case that matters most is ``uncovered`` -- a knob a library exposes that the
manifest omits. That is the direction a hand-written list cannot check, and the
direction that found ``ROCBLAS_API_BENCH``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).parent.parent.parent / "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
try:
    import audit_env_knobs as audit  # noqa: E402
finally:
    sys.path.remove(_SCRIPTS_DIR)

_DOCS = Path(__file__).parent.parent.parent / "docs" / "env-probe.md"


def _fake_lib(directory: Path, soname: str, names: list[str], *, real_name: str | None = None):
    """A fixture 'library': a file with env-var strings, reached via its soname.

    ``real_name`` plus a symlink models the layout that trips a glob-based
    resolver -- a stale versioned file sitting beside the active one.
    """
    target = directory / (real_name or soname)
    padding = "\x00some other unrelated binary content\x00"
    target.write_text(padding + "\n".join(names) + padding)
    if real_name:
        link = directory / soname
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.name)
    return target


class TestStringExtraction:
    def test_extracts_only_env_var_shaped_names(self, tmp_path):
        lib = _fake_lib(
            tmp_path,
            "libhipblaslt.so",
            [
                "HIPBLASLT_LOG_LEVEL",
                "TENSILE_DB2",
                "ROCBLAS_LAYER",
                "hipblasltSomeSymbol",  # not env-var shaped
                "HIPBLASLT",  # prefix alone, no suffix
                "some random string",
                "MIOPEN_FIND_MODE",  # audited prefixes only
            ],
        )
        assert audit.extract_names(lib) == frozenset(
            {"HIPBLASLT_LOG_LEVEL", "TENSILE_DB2", "ROCBLAS_LAYER"}
        )

    def test_resolves_through_the_soname_not_a_glob(self, tmp_path):
        """A stale ``libhipblaslt.so.1.0.70002`` beside the active
        ``1.4.70002`` is a real layout in these images. Globbing
        ``libhipblaslt.so.*`` can pick the stale file and give a wrong answer, so
        resolution follows the soname symlink."""
        _fake_lib(
            tmp_path, "libhipblaslt.so", ["HIPBLASLT_ACTIVE"], real_name="libhipblaslt.so.1.4.70002"
        )
        (tmp_path / "libhipblaslt.so.1.0.70002").write_text("HIPBLASLT_STALE_ONLY")

        resolved = audit.resolve_library(tmp_path, "libhipblaslt.so")

        assert resolved.name == "libhipblaslt.so.1.4.70002"
        assert audit.extract_names(resolved) == frozenset({"HIPBLASLT_ACTIVE"})

    def test_missing_library_resolves_to_none(self, tmp_path):
        assert audit.resolve_library(tmp_path, "libnope.so") is None


class TestAuditReport:
    """End-to-end through ``main`` in bootstrap mode, which is how the script is
    driven when the registry is not importable."""

    def _run(self, tmp_path, capsys, lib_names, manifest_names, extra_argv=()):
        _fake_lib(tmp_path, "libhipblaslt.so", lib_names)
        _fake_lib(tmp_path, "librocblas.so", [])
        manifest = tmp_path / "names.txt"
        manifest.write_text("\n".join(manifest_names) + "\n")
        report = tmp_path / "report.json"
        argv = [
            "audit_env_knobs.py",
            "--rocm-lib",
            str(tmp_path),
            "--names-file",
            str(manifest),
            "--json",
            str(report),
            *extra_argv,
        ]
        sys.argv = argv
        code = audit.main()
        capsys.readouterr()
        return code, json.loads(report.read_text())

    def test_covered_and_not_present_are_separated(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", sys.argv[:])
        code, report = self._run(
            tmp_path,
            capsys,
            lib_names=["HIPBLASLT_LOG_LEVEL", "TENSILE_DB2"],
            manifest_names=["HIPBLASLT_LOG_LEVEL", "TENSILE_DB2", "TENSILE_STREAMK_TILES"],
        )
        assert code == 0
        assert report["covered"] == ["HIPBLASLT_LOG_LEVEL", "TENSILE_DB2"]
        # A knob this build does not ship is NOT an error: the probe reports null.
        assert report["not_present"] == ["TENSILE_STREAMK_TILES"]
        assert report["uncovered"] == []

    def test_uncovered_knob_is_reported(self, tmp_path, capsys, monkeypatch):
        """The ROCBLAS_API_BENCH case: the library exposes a knob no list has."""
        monkeypatch.setattr(sys, "argv", sys.argv[:])
        code, report = self._run(
            tmp_path,
            capsys,
            lib_names=["HIPBLASLT_LOG_LEVEL", "ROCBLAS_API_BENCH"],
            manifest_names=["HIPBLASLT_LOG_LEVEL"],
        )
        assert report["uncovered"] == ["ROCBLAS_API_BENCH"]
        assert code == 0, "an uncovered knob is only fatal under --strict"

    def test_strict_makes_an_uncovered_knob_fatal(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", sys.argv[:])
        code, report = self._run(
            tmp_path,
            capsys,
            lib_names=["ROCBLAS_API_BENCH"],
            manifest_names=["HIPBLASLT_LOG_LEVEL"],
            extra_argv=("--strict",),
        )
        assert report["uncovered"] == ["ROCBLAS_API_BENCH"]
        assert code == 1

    def test_non_gemm_manifest_entries_are_out_of_scope(self, tmp_path, capsys, monkeypatch):
        """HSA_ / NCCL_ / MIOPEN_ knobs live in libraries this audit does not
        read, so they must not be reported as missing from hipBLASLt."""
        monkeypatch.setattr(sys, "argv", sys.argv[:])
        code, report = self._run(
            tmp_path,
            capsys,
            lib_names=["HIPBLASLT_LOG_LEVEL"],
            manifest_names=["HIPBLASLT_LOG_LEVEL", "HSA_XNACK", "NCCL_IB_TC", "LD_LIBRARY_PATH"],
        )
        assert report["not_present"] == []
        assert report["uncovered"] == []

    def test_missing_rocm_lib_dir_is_a_setup_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            ["audit_env_knobs.py", "--rocm-lib", str(tmp_path / "nope")],
        )
        assert audit.main() == 2
        capsys.readouterr()


class TestRegistryIsAuditable:
    def test_the_script_can_load_the_real_registry(self):
        """Bootstrap mode aside, the script's normal input is the manifest
        itself; if that import breaks, the audit silently stops being run."""
        names, source = audit.load_registry()
        assert len(names) > 100
        assert "env_knobs" in source
        assert names["TENSILE_DB2"]  # every name maps to a library

    def test_docs_env_var_count_matches_registry(self):
        """The docs' knob count is asserted, not maintained by hand.

        It said "58 names" from schema 1.13 through 1.15 while the real count
        went 58 -> 72 -> 132, because nothing checked it."""
        names, _ = audit.load_registry()
        text = _DOCS.read_text()
        match = re.search(r"currently (\d+) names", text)
        assert match, "docs no longer state a knob count in the expected form"
        assert int(match.group(1)) == len(
            names
        ), f"docs say {match.group(1)} names, registry has {len(names)}"
