"""Tests for scripts/audit_env_knobs.py -- the env-knob coverage audit.

The audit is the only check that can show the *installed* libraries are covered,
so its own logic has to be trustworthy. These tests exercise it against fixture
"libraries" (plain files containing env-var-shaped strings), which keeps them in
the CPU gate: no ROCm install, no real shared objects.

The case that matters most is ``uncovered`` -- a verified environment-variable
string that a library exposes and the manifest omits. Exact printable-run matching
is essential: splitting command templates on whitespace once invented a fake
``ROCBLAS_API_BENCH`` variable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    padding = b"\x00some other unrelated binary content\x00"
    target.write_bytes(
        b"\x7fELF" + padding + b"\x00".join(name.encode("ascii") for name in names) + padding
    )
    if real_name:
        link = directory / soname
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.name)
    return target


def _mark_fixture_as_reference(monkeypatch, directory: Path) -> None:
    monkeypatch.setattr(
        audit,
        "REFERENCE_LIBRARY_SHA256",
        {
            soname: audit._sha256_file(directory / audit.REFERENCE_LIBRARY_BASENAMES[soname])
            for soname in audit.DEFAULT_SONAMES
        },
    )


class TestStringExtraction:
    def test_extracts_only_env_var_shaped_names(self, tmp_path):
        lib = _fake_lib(
            tmp_path,
            "libhipblaslt.so",
            [
                "HIPBLASLT_LOG_LEVEL",
                "TENSILE_DB2",
                "ROCBLAS_LAYER",
                "ANALYTICAL_GEMM_HEURISTICS",
                "ANALYTICAL_GEMM_HEURISTICS_VARIANCE",
                "ORIGAMI_LOG_FILE",
                "hipblasltSomeSymbol",  # not env-var shaped
                "HIPBLASLT",  # prefix alone, no suffix
                "some random string",
                # Command template, not an environment variable. Whitespace
                # tokenization used to invent ROCBLAS_API_BENCH from this.
                "ROCBLAS_API_BENCH -f gemmt -r",
                "MIOPEN_FIND_MODE",  # audited prefixes only
            ],
        )
        assert audit.extract_names(lib) == frozenset(
            {
                "ANALYTICAL_GEMM_HEURISTICS",
                "ANALYTICAL_GEMM_HEURISTICS_VARIANCE",
                "HIPBLASLT_LOG_LEVEL",
                "ORIGAMI_LOG_FILE",
                "ROCBLAS_LAYER",
                "TENSILE_DB2",
            }
        )

    def test_does_not_trim_a_padded_help_string_into_a_fake_name(self, tmp_path):
        lib = tmp_path / "libhipblaslt.so"
        lib.write_bytes(b"\x7fELF\x00 ROCBLAS_PADDED_HELP_TOKEN \x00")

        assert audit.extract_names(lib) == frozenset()

    def test_does_not_split_newline_help_text_into_a_fake_name(self, tmp_path):
        lib = tmp_path / "libhipblaslt.so"
        lib.write_bytes(b"\x7fELF\x00Usage:\nHIPBLASLT_HELP_ONLY\n\x00")

        assert audit.extract_names(lib) == frozenset()

    @pytest.mark.parametrize("payload", [b"", b"INPUT(libhipblaslt.so.4)\n"])
    def test_rejects_empty_and_non_elf_inputs(self, tmp_path, payload):
        lib = tmp_path / "libhipblaslt.so"
        lib.write_bytes(payload)

        with pytest.raises(ValueError, match="not an ELF"):
            audit.extract_names(lib)

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

    def test_runtime_only_tree_resolves_the_highest_major(self, tmp_path):
        """A runtime-only container ships no ``.so`` devel link.

        The fallback then has to choose among ``<soname>.<major>`` links, and
        ``sorted()`` is lexicographic -- it picked the OLDEST co-installed
        major and silently audited the wrong library.
        """
        _fake_lib(
            tmp_path,
            "libhipblaslt.so.4",
            ["HIPBLASLT_STALE"],
            real_name="libhipblaslt.so.4.0.60000",
        )
        _fake_lib(
            tmp_path,
            "libhipblaslt.so.5",
            ["HIPBLASLT_ACTIVE"],
            real_name="libhipblaslt.so.5.1.70002",
        )

        resolved = audit.resolve_library(tmp_path, "libhipblaslt.so")

        assert resolved.name == "libhipblaslt.so.5.1.70002"
        assert audit.extract_names(resolved) == frozenset({"HIPBLASLT_ACTIVE"})

    def test_two_digit_major_is_reachable(self, tmp_path):
        """``{soname}.[0-9]`` never matched a two-digit major, so the library
        resolved to nothing and every one of its knobs was reported as absent
        from the install rather than as a setup error about an unread file."""
        _fake_lib(
            tmp_path,
            "libhipblaslt.so.10",
            ["HIPBLASLT_ACTIVE"],
            real_name="libhipblaslt.so.10.0.80000",
        )

        resolved = audit.resolve_library(tmp_path, "libhipblaslt.so")

        assert resolved is not None
        assert resolved.name == "libhipblaslt.so.10.0.80000"

    def test_missing_library_resolves_to_none(self, tmp_path):
        assert audit.resolve_library(tmp_path, "libnope.so") is None


class TestRegistryCoupling:
    def test_registry_exports_the_sentinels_the_audit_reads(self):
        """The provenance check reaches into the registry module by name.

        Every reference-path test injects a fake module, so a rename in
        ``env_knobs`` would only surface as an ``AttributeError`` on a real
        reference build -- which no CI runner has.
        """
        module = audit._import_registry_module()

        for sentinel in (
            "REF_BOTH_SO",
            "REF_HIPBLASLT_SO",
            "REF_ROCBLAS_SO",
            "ABSENT_FROM_REFERENCE_BUILD",
        ):
            assert isinstance(getattr(module, sentinel), str), sentinel


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
        # A knob this build does not ship is NOT an audit error. Capture is
        # independent: an exported value is preserved; null means only unset.
        assert report["not_present"] == ["TENSILE_STREAMK_TILES"]
        assert report["uncovered"] == []

    def test_uncovered_knob_is_reported(self, tmp_path, capsys, monkeypatch):
        """An exact env-var string exposed by a library must be reported."""
        monkeypatch.setattr(sys, "argv", sys.argv[:])
        code, report = self._run(
            tmp_path,
            capsys,
            lib_names=["HIPBLASLT_LOG_LEVEL", "HIPBLASLT_NEW_KNOB"],
            manifest_names=["HIPBLASLT_LOG_LEVEL"],
        )
        assert report["uncovered"] == ["HIPBLASLT_NEW_KNOB"]
        assert code == 0, "an uncovered knob is only fatal under --strict"

    def test_strict_makes_an_uncovered_knob_fatal(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", sys.argv[:])
        code, report = self._run(
            tmp_path,
            capsys,
            lib_names=["HIPBLASLT_NEW_KNOB"],
            manifest_names=["HIPBLASLT_LOG_LEVEL"],
            extra_argv=("--strict",),
        )
        assert report["uncovered"] == ["HIPBLASLT_NEW_KNOB"]
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

    def test_missing_one_requested_library_is_a_setup_error(self, tmp_path, monkeypatch, capsys):
        _fake_lib(tmp_path, "libhipblaslt.so", ["HIPBLASLT_LOG_LEVEL"])
        report = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--names-file",
                str(tmp_path / "names.txt"),
                "--json",
                str(report),
                "--strict",
            ],
        )
        (tmp_path / "names.txt").write_text("HIPBLASLT_LOG_LEVEL\n")

        assert audit.main() == 2

        capsys.readouterr()
        assert json.loads(report.read_text())["missing_sonames"] == ["librocblas.so"]

    @pytest.mark.parametrize("payload", [b"", b"INPUT(libhipblaslt.so.4)\n"])
    def test_invalid_library_is_a_reported_setup_error(
        self, tmp_path, monkeypatch, capsys, payload
    ):
        (tmp_path / "libhipblaslt.so").write_bytes(payload)
        _fake_lib(tmp_path, "librocblas.so", [])
        names = tmp_path / "names.txt"
        names.write_text("HIPBLASLT_LOG_LEVEL\n")
        report = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--names-file",
                str(names),
                "--json",
                str(report),
                "--strict",
            ],
        )

        assert audit.main() == 2

        captured = capsys.readouterr()
        assert "cannot audit" in captured.err
        parsed = json.loads(report.read_text())
        assert parsed["library_errors"][0]["soname"] == "libhipblaslt.so"
        assert parsed["uncovered"] == []

    def test_library_read_failure_is_exit_two_and_still_writes_report(
        self, tmp_path, monkeypatch, capsys
    ):
        _fake_lib(tmp_path, "libhipblaslt.so", ["HIPBLASLT_LOG_LEVEL"])
        _fake_lib(tmp_path, "librocblas.so", [])
        names = tmp_path / "names.txt"
        names.write_text("HIPBLASLT_LOG_LEVEL\n")
        report = tmp_path / "report.json"
        real_extract = audit.extract_names

        def fail_one(path):
            if path.name.startswith("libhipblaslt"):
                raise PermissionError("fixture unreadable")
            return real_extract(path)

        monkeypatch.setattr(audit, "extract_names", fail_one)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--names-file",
                str(names),
                "--json",
                str(report),
                "--strict",
            ],
        )

        assert audit.main() == 2

        capsys.readouterr()
        parsed = json.loads(report.read_text())
        assert parsed["library_errors"] == [
            {
                "error": "PermissionError: fixture unreadable",
                "resolved": str((tmp_path / "libhipblaslt.so").resolve()),
                "soname": "libhipblaslt.so",
            }
        ]

    def test_strict_detects_declared_library_owner_mismatch(self, tmp_path, monkeypatch, capsys):
        _fake_lib(
            tmp_path,
            "libhipblaslt.so",
            ["HIPBLASLT_LOG_LEVEL"],
            real_name="libhipblaslt.so.1.4.70002",
        )
        _fake_lib(
            tmp_path,
            "librocblas.so",
            [],
            real_name="librocblas.so.5.0.70002",
        )
        knob = SimpleNamespace(
            name="HIPBLASLT_LOG_LEVEL",
            library="rocblas",
            category="gemm_diagnostics",
            source_reference="hipblaslt",
        )
        module = SimpleNamespace(
            ENV_KNOB_REGISTRY=(knob,),
            REF_BOTH_SO="both",
            REF_HIPBLASLT_SO="hipblaslt",
            REF_ROCBLAS_SO="rocblas",
            ABSENT_FROM_REFERENCE_BUILD="absent",
        )
        monkeypatch.setattr(audit, "_import_registry_module", lambda: module)
        _mark_fixture_as_reference(monkeypatch, tmp_path)
        report = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--json",
                str(report),
                "--strict",
            ],
        )

        assert audit.main() == 1

        capsys.readouterr()
        mismatches = json.loads(report.read_text())["library_mismatches"]
        assert mismatches == [
            {
                "declared": "rocblas",
                "name": "HIPBLASLT_LOG_LEVEL",
                "observed": "hipblaslt",
                "present_in": ["libhipblaslt.so"],
            }
        ]

    def test_non_reference_library_ownership_variation_is_not_an_error(
        self, tmp_path, monkeypatch, capsys
    ):
        _fake_lib(tmp_path, "libhipblaslt.so", ["TENSILE_DB"])
        _fake_lib(tmp_path, "librocblas.so", [])
        knob = SimpleNamespace(
            name="TENSILE_DB",
            library="hipblaslt+rocblas",
            category="gemm_diagnostics",
            source_reference="reference build only",
        )
        module = SimpleNamespace(ENV_KNOB_REGISTRY=(knob,))
        monkeypatch.setattr(audit, "_import_registry_module", lambda: module)
        report = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--json",
                str(report),
                "--strict",
            ],
        )

        assert audit.main() == 0

        capsys.readouterr()
        parsed = json.loads(report.read_text())
        assert parsed["reference_build_validation"] is False
        # ``null``, not ``[]``: the check did not run, and an empty list reads
        # as "ran and found nothing".
        assert parsed["library_mismatches"] is None

    def test_reference_filenames_without_reference_hashes_do_not_activate_checks(
        self, tmp_path, monkeypatch, capsys
    ):
        _fake_lib(
            tmp_path,
            "libhipblaslt.so",
            ["HIPBLASLT_LOG_LEVEL"],
            real_name="libhipblaslt.so.1.4.70002",
        )
        _fake_lib(
            tmp_path,
            "librocblas.so",
            [],
            real_name="librocblas.so.5.0.70002",
        )
        knob = SimpleNamespace(
            name="HIPBLASLT_LOG_LEVEL",
            library="wrong-on-purpose",
            category="gemm_diagnostics",
            source_reference="wrong-on-purpose",
        )
        monkeypatch.setattr(
            audit,
            "_import_registry_module",
            lambda: SimpleNamespace(ENV_KNOB_REGISTRY=(knob,)),
        )
        report = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--json",
                str(report),
                "--strict",
            ],
        )

        assert audit.main() == 0

        capsys.readouterr()
        parsed = json.loads(report.read_text())
        assert parsed["reference_build_validation"] is False
        assert parsed["library_mismatches"] is None
        assert parsed["provenance_mismatches"] is None

    def test_strict_detects_reference_provenance_mismatch(self, tmp_path, monkeypatch, capsys):
        _fake_lib(
            tmp_path,
            "libhipblaslt.so",
            ["HIPBLASLT_LOG_LEVEL"],
            real_name="libhipblaslt.so.1.4.70002",
        )
        _fake_lib(
            tmp_path,
            "librocblas.so",
            [],
            real_name="librocblas.so.5.0.70002",
        )
        knob = SimpleNamespace(
            name="HIPBLASLT_LOG_LEVEL",
            library="hipblaslt",
            category="gemm_diagnostics",
            source_reference="incorrectly absent",
        )
        module = SimpleNamespace(
            ENV_KNOB_REGISTRY=(knob,),
            REF_BOTH_SO="both",
            REF_HIPBLASLT_SO="hipblaslt",
            REF_ROCBLAS_SO="rocblas",
            ABSENT_FROM_REFERENCE_BUILD="absent",
        )
        monkeypatch.setattr(audit, "_import_registry_module", lambda: module)
        _mark_fixture_as_reference(monkeypatch, tmp_path)
        report = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(tmp_path),
                "--json",
                str(report),
                "--strict",
            ],
        )

        assert audit.main() == 1

        capsys.readouterr()
        assert json.loads(report.read_text())["provenance_mismatches"] == [
            {
                "declared": "incorrectly absent",
                "expected": "hipblaslt",
                "name": "HIPBLASLT_LOG_LEVEL",
                "present_in": ["libhipblaslt.so"],
            }
        ]


class TestRegistryIsAuditable:
    def test_the_script_can_load_the_real_registry(self):
        """Bootstrap mode aside, the script's normal input is the manifest
        itself; if that import breaks, the audit silently stops being run."""
        names, source = audit.load_registry()
        assert len(names) > 100
        assert "env_knobs" in source
        assert names["TENSILE_DB2"]  # every name maps to a library

    def test_docs_knob_inventory_matches_registry(self):
        """The docs' knob inventory is GENERATED, and this is the check that keeps
        it that way. Without it the table is just a second hand-written copy of
        the manifest -- the exact thing the manifest exists to eliminate."""
        text = _DOCS.read_text()
        begin, end = audit.DOCS_TABLE_BEGIN, audit.DOCS_TABLE_END
        assert begin in text and end in text, "generated inventory block missing from docs"
        in_docs = text[text.index(begin) : text.index(end) + len(end)]

        assert in_docs == audit.render_docs_table(), (
            "docs inventory is stale -- regenerate with "
            "`python scripts/audit_env_knobs.py --emit-docs-table`"
        )

    def test_emit_docs_table_needs_no_rocm_install(self, monkeypatch, capsys):
        """The table comes from the manifest, so it must render on a host with no
        ROCm: --emit-docs-table returns before any library resolution."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["audit_env_knobs.py", "--emit-docs-table", "--rocm-lib", "/nonexistent"],
        )

        assert audit.main() == 0

        out = capsys.readouterr().out
        assert audit.DOCS_TABLE_BEGIN in out and audit.DOCS_TABLE_END in out

    def test_docs_env_var_count_matches_registry(self):
        """The docs' knob count is asserted, not maintained by hand.

        It said "58 names" from schema 1.13 through 1.15 while the real count
        went 58 -> 72 -> 136, because nothing checked it."""
        names, _ = audit.load_registry()
        text = _DOCS.read_text()
        match = re.search(r"currently (\d+) names", text)
        assert match, "docs no longer state a knob count in the expected form"
        assert int(match.group(1)) == len(
            names
        ), f"docs say {match.group(1)} names, registry has {len(names)}"


# ---------------------------------------------------------------------------
# Layout-agnostic default for --rocm-lib (issue #381)
# ---------------------------------------------------------------------------


class TestDefaultRocmLib:
    """The audit must find the GEMM libraries on both install layouts.

    Before #381 the default was the literal ``/opt/rocm/lib``, so on a wheel
    install the audit found no libraries and exited 2 -- indistinguishable
    from a genuinely broken image.
    """

    @staticmethod
    def _patch_resolver(monkeypatch, lib_dir: Path):
        from aorta.instrumentation import rocm_paths

        monkeypatch.setattr(
            rocm_paths,
            "resolve_rocm_roots",
            lambda environ=None: SimpleNamespace(lib_dir=lib_dir),
        )

    def test_uses_the_resolved_lib_dir_on_a_classic_install(self, monkeypatch):
        self._patch_resolver(monkeypatch, Path("/opt/rocm/lib"))
        assert audit.default_rocm_lib() == Path("/opt/rocm/lib")

    def test_uses_the_resolved_lib_dir_on_a_wheel_install(self, monkeypatch):
        wheel_lib = Path("/opt/venv/lib/python3.14/site-packages/_rocm_sdk_libraries/lib")
        self._patch_resolver(monkeypatch, wheel_lib)
        assert audit.default_rocm_lib() == wheel_lib

    def test_falls_back_to_the_classic_path_when_aorta_is_not_importable(
        self, monkeypatch
    ):
        """The script is also run standalone, outside an aorta install.

        A wrong default beats a traceback there: a caller in that situation can
        still pass ``--rocm-lib`` explicitly. Note gpu-tests.yml no longer does
        (see ``test_the_gpu_gate_exercises_the_resolved_default``), so in CI
        this fallback is not the path taken -- aorta is installed there.
        """
        monkeypatch.setitem(sys.modules, "aorta.instrumentation.rocm_paths", None)
        assert audit.default_rocm_lib() == audit.DEFAULT_ROCM_LIB
        assert audit.DEFAULT_ROCM_LIB == Path("/opt/rocm/lib")

    def test_explicit_rocm_lib_argument_bypasses_discovery(
        self, monkeypatch, tmp_path, capsys
    ):
        """``--rocm-lib`` wins when passed, and must not be second-guessed.

        Still supported and still tested -- it is how you audit a tree that is
        NOT the resolved install. It is simply no longer what CI passes.
        """

        def fail(environ=None):
            raise AssertionError("resolver consulted despite an explicit --rocm-lib")

        from aorta.instrumentation import rocm_paths

        monkeypatch.setattr(rocm_paths, "resolve_rocm_roots", fail)
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        _fake_lib(explicit, "libhipblaslt.so", ["HIPBLASLT_LOG_LEVEL"])
        _fake_lib(explicit, "librocblas.so", [])
        manifest = tmp_path / "names.txt"
        manifest.write_text("HIPBLASLT_LOG_LEVEL\n")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_env_knobs.py",
                "--rocm-lib",
                str(explicit),
                "--names-file",
                str(manifest),
            ],
        )
        assert audit.main() == 0
        capsys.readouterr()

    def test_audit_finds_libraries_under_a_wheel_layout_without_arguments(
        self, monkeypatch, tmp_path, capsys
    ):
        """#381 acceptance: the audit works on a wheel install with no --rocm-lib."""
        wheel_lib = tmp_path / "site-packages" / "_rocm_sdk_libraries" / "lib"
        wheel_lib.mkdir(parents=True)
        # Runtime-only trees ship only the versioned soname, no bare symlink.
        _fake_lib(
            wheel_lib,
            "libhipblaslt.so",
            ["HIPBLASLT_LOG_LEVEL"],
            real_name="libhipblaslt.so.1",
        )
        _fake_lib(wheel_lib, "librocblas.so", [], real_name="librocblas.so.5")
        self._patch_resolver(monkeypatch, wheel_lib)
        manifest = tmp_path / "names.txt"
        manifest.write_text("HIPBLASLT_LOG_LEVEL\n")
        monkeypatch.setattr(
            sys,
            "argv",
            ["audit_env_knobs.py", "--names-file", str(manifest)],
        )
        assert audit.main() == 0
        capsys.readouterr()

    def test_the_gpu_gate_exercises_the_resolved_default(self):
        """The one job that can prove ``default_rocm_lib()`` must not bypass it.

        Not reachable from a unit test of the code it breaks, so it is pinned
        here the same way this branch pins the other workflow invariants.

        gpu-tests.yml used to run ``--rocm-lib /opt/rocm/lib``, which left the
        resolved default with no CI coverage anywhere: green on today's classic
        base, an immediate exit 2 at the deferred wheel-layout flip (#383), and
        the failure would have read as "the audit broke" rather than "the
        workflow pinned a path the resolver was supposed to supply". Omitting
        the argument IS the resolver path, so this job is now the regression
        test for ``default_rocm_lib()`` -- and this assertion is what keeps it
        one.
        """
        workflow = (
            Path(__file__).parent.parent.parent / ".github" / "workflows" / "gpu-tests.yml"
        ).read_text(encoding="utf-8")
        invocations = [
            line.strip()
            for line in workflow.splitlines()
            if "audit_env_knobs.py" in line and not line.strip().startswith("#")
        ]
        assert invocations, "the GPU gate no longer runs the env-knob audit at all"
        for invocation in invocations:
            assert "--rocm-lib" not in invocation, (
                "pinning --rocm-lib here bypasses default_rocm_lib() and removes its "
                f"only CI coverage: {invocation}"
            )
