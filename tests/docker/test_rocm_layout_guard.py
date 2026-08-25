"""Pin ``docker/rocm_layout_guard.py`` to the resolver it mirrors (issue #381).

The guard runs inside a docker build, where the repo does not exist: the build
context is ``docker/`` and the checkout is only mounted at runtime. It therefore
re-implements ``aorta.instrumentation.rocm_paths`` resolution rules instead of
importing them.

Duplicated logic drifts, and the failure mode is expensive -- a guard that
disagrees with the resolver either passes a base image the probe cannot read
(silent nulls in CI) or fails a build that would have worked fine. So rather
than trusting the copy, these tests run **both** implementations over the same
synthetic trees and require identical answers. Add a resolution rule to one
side without the other and this file goes red.

The trees are built rather than mocked. Wheel discovery is driven by stubbing
``importlib.util.find_spec``, which BOTH implementations call, so each still
runs its own spec-interpretation logic (``spec.origin`` for a regular package,
``submodule_search_locations`` for a namespace one) -- that is the part that
can drift. Stubbing rather than using ``sys.path`` is deliberate: these tests
run inside the CI image, which on the wheel layout has a real
``_rocm_sdk_core`` installed, and a namespace portion placed earlier on the
path loses to a regular package found later, so a path-based test would
silently resolve to the image's own ROCm instead of the fixture. That
discovery works against a genuinely installed wheel is verified directly
against real images, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aorta.instrumentation import rocm_paths
from aorta.instrumentation.rocm_paths import resolve_rocm_roots

_GUARD_PATH = Path(__file__).resolve().parents[2] / "docker" / "rocm_layout_guard.py"


def _load_module_from_path(name: str, path: Path):
    """Import a standalone script by path.

    Neither the guard nor ``scripts/audit_env_knobs.py`` is importable as a
    package member, and both are read here for the same reason: a test that
    restates their constants instead of reading them cannot catch drift.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_module_from_path("rocm_layout_guard", _GUARD_PATH)


def place_gemm_sonames(lib_dir: Path) -> Path:
    """The GEMM libraries ``audit_env_knobs.py`` resolves, as a runtime tree ships them.

    ``.so.<major>`` rather than a bare ``.so``: the bare link is the devel
    package's, and a runtime-only image has only the versioned file. The guard
    has to accept that shape, so it is what the healthy fixtures use.
    """
    lib_dir.mkdir(parents=True, exist_ok=True)
    for soname in guard.AUDIT_SONAMES:
        (lib_dir / f"{soname}.1").write_bytes(b"\x7fELF")
    return lib_dir


def build_classic(
    root: Path,
    *,
    version: str | None = "7.2.4",
    lib: bool = True,
    sonames: bool = True,
) -> Path:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    if lib:
        (root / "lib").mkdir(exist_ok=True)
        if sonames:
            place_gemm_sonames(root / "lib")
    if version is not None:
        info = root / ".info"
        info.mkdir(exist_ok=True)
        (info / "version").write_text(f"{version}\n", encoding="utf-8")
    return root


def build_wheel(
    site: Path,
    *,
    libraries: bool = True,
    devel: bool = False,
    version: str | None = "7.14.0",
    sonames: bool = True,
) -> Path:
    core = site / "_rocm_sdk_core"
    (core / "bin").mkdir(parents=True)
    if version is not None:
        info = core / ".info"
        info.mkdir()
        (info / "version").write_text(f"{version}\n", encoding="utf-8")
    if libraries:
        libraries_lib = site / "_rocm_sdk_libraries" / "lib"
        libraries_lib.mkdir(parents=True)
        if sonames:
            place_gemm_sonames(libraries_lib)
    if devel:
        (site / "_rocm_sdk_devel" / "include").mkdir(parents=True)
    return core


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch):
    """Hide host ROCm from BOTH implementations, symmetrically.

    That includes an importable wheel: these tests run inside the CI image,
    which on the wheel layout has one installed, so without stubbing
    ``find_spec`` the "nothing found" cases would discover the image's own
    ROCm.
    """
    absent = tmp_path / "absent_opt_rocm"
    monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", absent)
    monkeypatch.setattr(guard, "CLASSIC_ROCM_ROOT", absent)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    for name in ("ROCM_PATH", "ROCM_HOME"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def classic_at(sandbox):
    """Place a classic install where autodetection will find it."""

    def _place(root: Path) -> Path:
        sandbox.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", root)
        sandbox.setattr(guard, "CLASSIC_ROCM_ROOT", root)
        return root

    return _place


@pytest.fixture
def importable_wheel(sandbox, tmp_path: Path):
    """Make a wheel tree discoverable via ``importlib.util.find_spec``.

    ``regular_package`` picks which spec shape the resolvers have to
    interpret: a real ``_rocm_sdk_core`` ships an ``__init__.py`` (so
    ``spec.origin`` is set, and that is the production path), while a
    namespace package leaves ``origin`` None and exposes only
    ``submodule_search_locations``.
    """

    def _install(*, regular_package: bool = True, **kwargs) -> Path:
        site = tmp_path / "site-packages"
        core = build_wheel(site, **kwargs)
        if regular_package:
            init = core / "__init__.py"
            init.write_text("", encoding="utf-8")
            spec = SimpleNamespace(origin=str(init), submodule_search_locations=[str(core)])
        else:
            spec = SimpleNamespace(origin=None, submodule_search_locations=[str(core)])
        sandbox.setattr(
            "importlib.util.find_spec",
            lambda name: spec if name == "_rocm_sdk_core" else None,
        )
        return core

    return _install


@pytest.fixture
def agree(sandbox):
    """Run both implementations over the same state and require agreement."""

    def _agree(env: dict[str, str] | None = None):
        env = env or {}
        for name in ("ROCM_PATH", "ROCM_HOME"):
            sandbox.delenv(name, raising=False)
        for name, value in env.items():
            sandbox.setenv(name, value)

        core, libraries, include, layout, source = guard.resolve()
        expected = resolve_rocm_roots(env)

        assert layout == expected.layout
        assert source == expected.source
        assert core == expected.core
        assert libraries == expected.libraries
        assert include == expected.include
        # The consumers read these, not the roots themselves.
        assert core / ".info" / "version" == expected.version_file
        assert libraries / "lib" == expected.lib_dir
        return expected

    return _agree


class TestParity:
    def test_classic_via_rocm_path(self, agree, tmp_path: Path):
        root = build_classic(tmp_path / "rocm")
        assert agree({"ROCM_PATH": str(root)}).source == "ROCM_PATH"

    def test_classic_via_rocm_home(self, agree, tmp_path: Path):
        root = build_classic(tmp_path / "rocm")
        assert agree({"ROCM_HOME": str(root)}).source == "ROCM_HOME"

    def test_rocm_path_beats_rocm_home(self, agree, tmp_path: Path):
        preferred = build_classic(tmp_path / "preferred")
        other = build_classic(tmp_path / "other")
        result = agree({"ROCM_PATH": str(preferred), "ROCM_HOME": str(other)})
        assert result.core == preferred

    def test_stale_env_var_falls_through(self, agree, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        real = build_classic(tmp_path / "rocm")
        result = agree({"ROCM_PATH": str(empty), "ROCM_HOME": str(real)})
        assert result.core == real

    def test_opt_rocm_autodetected(self, agree, classic_at, tmp_path: Path):
        root = classic_at(build_classic(tmp_path / "opt_rocm"))
        assert agree().core == root

    def test_wheel_via_import(self, agree, importable_wheel):
        core = importable_wheel()
        result = agree()
        assert result.core == core
        assert result.source == "import:_rocm_sdk_core"

    def test_wheel_via_import_namespace_package(self, agree, importable_wheel):
        """No ``__init__.py``: ``spec.origin`` is None and both sides must
        fall back to ``submodule_search_locations``."""
        core = importable_wheel(regular_package=False)
        assert agree().core == core

    def test_empty_opt_rocm_does_not_shadow_a_wheel(
        self, agree, classic_at, importable_wheel, tmp_path: Path
    ):
        stale = tmp_path / "opt_rocm"
        stale.mkdir()
        classic_at(stale)
        core = importable_wheel()
        assert agree().core == core

    def test_opt_rocm_beats_an_importable_wheel(
        self, agree, classic_at, importable_wheel, tmp_path: Path
    ):
        importable_wheel()
        root = classic_at(build_classic(tmp_path / "opt_rocm"))
        assert agree().core == root

    def test_bin_only_opt_rocm_stub_does_not_shadow_a_wheel(
        self, agree, classic_at, importable_wheel, tmp_path: Path
    ):
        """Autodetection is held to the stricter 'usable' test on both sides.

        If only one implementation adopted it, the guard would fail a build the
        resolver reads fine (or pass one it cannot), which is precisely the drift
        this parity suite exists to catch.
        """
        stub = tmp_path / "opt_rocm"
        (stub / "bin").mkdir(parents=True)
        classic_at(stub)
        core = importable_wheel()
        result = agree()
        assert result.core == core
        assert result.layout == "wheel"

    def test_a_lib_dir_alone_is_still_a_usable_classic_root(
        self, agree, classic_at, tmp_path: Path
    ):
        root = tmp_path / "opt_rocm"
        (root / "lib").mkdir(parents=True)
        classic_at(root)
        assert agree().source == "opt_rocm"

    @pytest.mark.parametrize(
        "marker_kind",
        ["empty-file", "whitespace-only", "directory"],
    )
    def test_an_unusable_version_marker_does_not_shadow_a_wheel(
        self, agree, classic_at, importable_wheel, tmp_path: Path, marker_kind
    ):
        """Both sides must agree that `exists()` is not enough (#387).

        If only one adopted the readable-and-non-empty rule, the guard would fail
        a build the resolver reads fine (or pass one it cannot) -- exactly the
        drift this suite exists to catch.
        """
        stub = tmp_path / "opt_rocm"
        (stub / ".info").mkdir(parents=True)
        (stub / "bin").mkdir()
        marker = stub / ".info" / "version"
        if marker_kind == "empty-file":
            marker.write_text("", encoding="utf-8")
        elif marker_kind == "whitespace-only":
            marker.write_text("\n \n", encoding="utf-8")
        else:
            marker.mkdir()
        classic_at(stub)
        core = importable_wheel()
        assert agree().core == core

    def test_a_stale_mount_degrades_identically_on_both_sides(
        self, agree, sandbox, tmp_path: Path
    ):
        """Neither implementation may propagate an OSError from a probe.

        The resolver documents that resolution never raises; the guard has to
        reach its own diagnostics rather than dying with a traceback, since a
        traceback fails the build without saying what it looked for.
        """

        def boom(self):
            raise OSError("stale NFS handle")

        sandbox.setattr(Path, "is_dir", boom)
        assert agree({"ROCM_PATH": str(tmp_path / "gone")}).source == "none"

    def test_env_var_on_a_component_directory(self, agree, tmp_path: Path):
        site = tmp_path / "site-packages"
        core = build_wheel(site, devel=True)
        result = agree({"ROCM_PATH": str(site / "_rocm_sdk_devel")})
        assert result.core == core
        assert result.include == site / "_rocm_sdk_devel"

    def test_libraries_and_include_fall_back_to_core(self, agree, tmp_path: Path):
        site = tmp_path / "site-packages"
        core = build_wheel(site, libraries=False, devel=False)
        result = agree({"ROCM_PATH": str(core)})
        assert result.libraries == core
        assert result.include == core

    def test_lone_component_without_core_sibling(self, agree, tmp_path: Path):
        lone = tmp_path / "site-packages" / "_rocm_sdk_devel"
        lone.mkdir(parents=True)
        assert agree({"ROCM_PATH": str(lone)}).core == lone

    def test_nothing_found(self, agree):
        assert agree().source == "none"

    def test_a_relative_override_is_anchored_on_both_sides(
        self, agree, tmp_path: Path, monkeypatch
    ):
        """Both must anchor a relative override, or they disagree on the roots."""
        build_classic(tmp_path / "rocm")
        monkeypatch.chdir(tmp_path)
        result = agree({"ROCM_PATH": "rocm"})
        assert result.core.is_absolute()
        assert result.core == tmp_path / "rocm"

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"   \n\t", id="ascii-whitespace"),
            pytest.param("\u00a0\u2003".encode(), id="unicode-whitespace"),
            pytest.param(b"\xff\xfe", id="non-utf8"),
            # Valid UTF-8 for the first 64 bytes, invalid at offset 64: unusable
            # only to a reader that looks past the old 64-byte probe window.
            pytest.param(
                b"7.2.4-" + b"x" * 58 + b"\xff" + b"tail", id="non-utf8-past-old-probe"
            ),
        ],
    )
    def test_marker_usability_agrees_on_degenerate_content(
        self, agree, classic_at, tmp_path: Path, content: bytes
    ):
        """Both must reach the same verdict on a marker that is not a version.

        The unicode-whitespace case is the one that actually diverged: bytes
        .strip() only strips ASCII, so a marker holding a non-breaking space read
        as usable in the guard and unusable in the resolver. A guard that is more
        lenient than the probe is the bad direction -- it passes the image, then
        the probe reports no version and nothing explains why.

        Deliberately NO ``lib/`` here: ``_is_usable_rocm_root`` accepts a version
        marker OR a lib dir, so creating one makes the root usable regardless of
        the marker and the assertion stops testing anything. (It did, first
        attempt -- reverting the guard fix still passed.)
        """
        root = classic_at(tmp_path / "rocm")
        (root / ".info").mkdir(parents=True)
        (root / ".info" / "version").write_bytes(content)
        # Not a usable marker on either side, and no lib/ to fall back on, so
        # both must decline the root. `agree` raises if they disagree.
        assert agree().source == "none"

    @pytest.mark.parametrize(
        ("content", "usable"),
        [
            pytest.param(b"7.2.4", True, id="clean"),
            pytest.param("7.2.4\u00e9".encode(), True, id="valid-multibyte"),
            # Truncated multi-byte sequences at the REAL end of the file. An
            # unconditional final=False buffers the incomplete tail and drops
            # it, so these decoded to "7.2.4" and a corrupt marker was
            # published as a valid version -- contradicting the reader's own
            # docstring and the guard error text that promises to match it.
            pytest.param(b"7.2.4\xc3", False, id="corrupt-2-byte-seq-at-eof"),
            pytest.param(b"7.2.4\xe2\x82", False, id="corrupt-3-byte-seq-at-eof"),
            pytest.param(b"7.2.4\xff", False, id="never-a-lead-byte"),
        ],
    )
    def test_marker_decoding_agrees_on_truncated_sequences(
        self, tmp_path: Path, content: bytes, usable: bool
    ):
        """Both copies must finalise the decoder at a real EOF (#387).

        The incremental decoder is there so a character the BYTE LIMIT cut in
        half is not reported as corruption. ``final=False`` cannot tell that
        from corruption at the genuine end of the file, so it accepted both.
        Reading limit + 1 bytes distinguishes them.
        """
        marker = tmp_path / f"version_{len(content)}_{content[-1]}"
        marker.write_bytes(content)
        assert (rocm_paths.read_version_marker(marker) is not None) is usable
        assert (guard.read_version_marker(marker) is not None) is usable

    def test_a_character_split_by_the_byte_limit_is_still_tolerated(
        self, tmp_path: Path
    ):
        """The property the incremental decoder exists for, kept intact.

        Rejecting this would make a long-but-valid marker unreadable, which is
        the over-correction the finalisation fix has to avoid: the tail is
        incomplete only because the read stopped, not because the file is bad.
        """
        marker = tmp_path / "version"
        limit = 32
        # The 2-byte character starts at the last byte inside the limit.
        marker.write_bytes(b"a" * (limit - 1) + "\u00e9".encode())

        for reader in (rocm_paths.read_version_marker, guard.read_version_marker):
            value = reader(marker, limit)
            assert value == "a" * (limit - 1), reader

    def test_a_mojibake_marker_does_not_hide_a_present_install(
        self, agree, classic_at, tmp_path: Path
    ):
        """An undecodable marker is unusable; the *install* is still found.

        The two verdicts are separate, and conflating them is how an earlier
        revision of this went wrong. The marker is rejected (so the reported
        version is null rather than U+FFFD), but the root still resolves via its
        lib/ directory -- reporting "no ROCm found" for an install that is
        plainly present would be the worse error.
        """
        root = classic_at(tmp_path / "rocm")
        (root / ".info").mkdir(parents=True)
        (root / ".info" / "version").write_bytes(b"\xff\xfe")
        (root / "lib").mkdir()
        assert agree().core == root
        assert rocm_paths.read_version_marker(root / ".info" / "version") is None

    def test_the_guard_never_rejects_a_marker_discovery_accepted(
        self, classic_at, tmp_path: Path, capsys
    ):
        """resolve() and main() must not disagree about the same file (#387, round 9).

        They did: discovery accepted a non-UTF-8 marker with errors="replace"
        while main() rejected it, so the guard selected the root and then failed
        the build with "no readable .info/version" naming the file it had just
        read. Both now share one reader.
        """
        root = classic_at(tmp_path / "rocm")
        (root / ".info").mkdir(parents=True)
        (root / ".info" / "version").write_bytes(b"\xff\xfe")
        (root / "lib").mkdir()
        # Discovery finds the root, main() reports the version as missing, and
        # the two agree the marker is the unusable part.
        assert guard.resolve()[0] == root
        assert guard.main() == 1
        assert "no readable" in capsys.readouterr().err

    def test_the_guard_uses_one_byte_limit_for_probe_and_value(
        self, classic_at, tmp_path: Path, capsys
    ):
        """The same disagreement as above, reached via the byte limit (#387).

        Sharing the reader fixed the predicate but not the limit: validation
        probed 64 bytes while every value read took 4096, so a marker that is
        valid UTF-8 only for its first 64 bytes was accepted by discovery and
        rejected by main() -- the guard failing the build about the very file
        discovery had just read. One limit now, so the two cannot part company.
        """
        root = classic_at(tmp_path / "rocm")
        (root / ".info").mkdir(parents=True)
        marker = root / ".info" / "version"
        marker.write_bytes(b"7.2.4-" + b"x" * 58 + b"\xff" + b"tail")
        (root / "lib").mkdir()

        assert guard._has_readable_version(root) is False
        assert guard.read_version_marker(marker) is None
        # Still found via lib/, and still reported as having no readable
        # version -- the two verdicts agree instead of contradicting.
        assert guard.resolve()[0] == root
        assert guard.main() == 1
        assert "no readable" in capsys.readouterr().err

    def test_an_unanchorable_override_degrades_identically(
        self, agree, sandbox, monkeypatch
    ):
        """A failing getcwd() must make both fall through, not one crash.

        If only the guard raised, a base image with a relative ROCM_PATH would
        fail the build with a traceback instead of the diagnostic; if only the
        resolver raised, the guard would pass an image that then breaks at
        collection time. Same degradation on both sides or the guard lies.
        """

        def _no_cwd(_path):
            raise OSError(2, "No such file or directory")

        for module in (rocm_paths, guard):
            monkeypatch.setattr(module.os.path, "abspath", _no_cwd)
        assert agree({"ROCM_PATH": "relative/rocm"}).source == "none"

    @pytest.mark.parametrize(
        "exc",
        [
            ImportError("half-installed package"),
            ValueError("__spec__ is not set"),
            OSError("stale NFS handle on site-packages"),
            RuntimeError("broken meta path finder"),
        ],
    )
    def test_a_raising_find_spec_degrades_identically(self, agree, sandbox, exc):
        """Both sides must swallow the same set, or one fails a build the other passes.

        find_spec runs arbitrary path-finder code, so a damaged install can raise
        more than ImportError/ValueError. Neither implementation may propagate it:
        the resolver documents that resolution never raises, and the guard has to
        reach its own diagnostics rather than dying with a traceback.
        """

        def boom(name):
            raise exc

        sandbox.setattr("importlib.util.find_spec", boom)
        assert agree().source == "none"


class TestGuardVerdict:
    """The guard's own contract: accept both layouts, still fail closed."""

    def test_classic_install_passes(self, classic_at, tmp_path: Path, capsys):
        classic_at(build_classic(tmp_path / "opt_rocm"))
        assert guard.main() == 0
        out = capsys.readouterr().out
        assert "ROCm layout : classic" in out
        assert "7.2.4" in out

    def test_wheel_install_passes(self, importable_wheel, capsys):
        importable_wheel()
        assert guard.main() == 0
        out = capsys.readouterr().out
        assert "ROCm layout : wheel" in out
        assert "7.14.0" in out

    def test_no_rocm_at_all_fails(self, sandbox, capsys):
        assert guard.main() == 1
        err = capsys.readouterr().err
        assert "no ROCm install found in either layout" in err
        assert "source=none" in err
        assert "neither was found" in err

    def test_an_incomplete_install_does_not_claim_neither_layout_was_found(
        self, classic_at, tmp_path: Path, capsys
    ):
        """The diagnostic must match the state it just printed (#387).

        When discovery DID resolve a tree that is merely incomplete, saying
        "neither layout was found" contradicts the layout/source printed two
        lines above and sends the reader hunting the wrong problem.
        """
        classic_at(build_classic(tmp_path / "opt_rocm", version=None))
        assert guard.main() == 1
        err = capsys.readouterr().err
        assert "found a classic ROCm install" in err
        assert "source=opt_rocm" in err
        assert "incomplete or damaged" in err
        assert "neither was found" not in err

    def test_install_without_a_version_file_fails(self, classic_at, tmp_path: Path, capsys):
        """The exact silent-degradation case the guard exists to catch.

        A tree that looks like ROCm but carries no version marker is what
        makes the dashboard's ``rocm`` column go null while CI stays green.
        """
        classic_at(build_classic(tmp_path / "opt_rocm", version=None))
        assert guard.main() == 1
        err = capsys.readouterr().err
        assert "nightly_eval.py" in err

    def test_install_without_a_lib_dir_fails(self, classic_at, tmp_path: Path, capsys):
        classic_at(build_classic(tmp_path / "opt_rocm", lib=False))
        assert guard.main() == 1
        err = capsys.readouterr().err
        assert "audit_env_knobs.py" in err

    def test_empty_version_file_counts_as_missing(self, classic_at, tmp_path: Path, capsys):
        root = build_classic(tmp_path / "opt_rocm", version=None)
        (root / ".info").mkdir(exist_ok=True)
        (root / ".info" / "version").write_text("", encoding="utf-8")
        classic_at(root)
        assert guard.main() == 1
        assert "nightly_eval.py" in capsys.readouterr().err

    def test_version_dev_alone_is_sufficient(self, classic_at, tmp_path: Path, capsys):
        root = build_classic(tmp_path / "opt_rocm", version=None)
        info = root / ".info"
        info.mkdir(exist_ok=True)
        (info / "version-dev").write_text("7.2.4.50311-abc\n", encoding="utf-8")
        classic_at(root)
        assert guard.main() == 0
        assert "7.2.4.50311-abc" in capsys.readouterr().out

    def test_a_wheel_without_the_libraries_component_fails(
        self, importable_wheel, capsys
    ):
        """A directory check cannot see a missing libraries component (#387).

        ``_wheel_roots`` falls back ``libraries -> core`` when
        ``_rocm_sdk_libraries`` is absent, and ``core/lib`` exists anyway
        because the LLVM toolchain lives at ``core/lib/llvm/bin``. So the old
        "is ``lib/`` a directory" assertion passed an image carrying neither
        ``libhipblaslt`` nor ``librocblas``, and the breakage surfaced later in
        ``audit_env_knobs.py`` -- the "a bad digest quietly guts the evidence"
        case arriving THROUGH the fallback rather than around it.
        """
        core = importable_wheel(libraries=False)
        # What makes core/lib exist on a real wheel image regardless.
        (core / "lib" / "llvm" / "bin").mkdir(parents=True)

        assert guard.main() == 1
        err = capsys.readouterr().err
        for soname in guard.AUDIT_SONAMES:
            assert soname in err
        assert "audit_env_knobs.py" in err
        # The diagnostic names the likely cause rather than only the symptom.
        assert "_rocm_sdk_libraries" in err

    def test_a_wheel_with_the_libraries_component_still_passes(
        self, importable_wheel, capsys
    ):
        """The fix must not reject the layout the canary actually runs."""
        importable_wheel(libraries=True)
        assert guard.main() == 0
        assert "GEMM libs" in capsys.readouterr().out

    def test_a_runtime_only_soname_is_accepted(self, classic_at, tmp_path: Path):
        """Only the devel package ships the bare ``.so`` link.

        Requiring it would fail every runtime-only image, which is most of
        them -- ``resolve_library`` falls back to ``<soname>.<major>`` for
        exactly this reason and the guard has to agree.
        """
        root = build_classic(tmp_path / "opt_rocm", sonames=False)
        for soname in guard.AUDIT_SONAMES:
            (root / "lib" / f"{soname}.5").write_bytes(b"\x7fELF")
        classic_at(root)
        assert guard.main() == 0

    @pytest.mark.parametrize(
        ("name", "make_dir"),
        [
            # A permissive check is invisible against a well-formed tree, so
            # every case here is a DECOY: something that looks like the library
            # to a loose matcher and is not one to the audit.
            pytest.param("libhipblaslt.so", True, id="directory-named-like-the-link"),
            pytest.param("libhipblaslt.so.1", True, id="directory-named-like-a-major"),
            pytest.param("libhipblaslt.so.debug", False, id="separate-debug-info-file"),
        ],
    )
    def test_a_decoy_that_the_audit_would_not_resolve_does_not_satisfy_the_guard(
        self, classic_at, tmp_path: Path, capsys, name: str, make_dir: bool
    ):
        """The guard must not be more permissive than the consumer it mirrors (#387).

        Round 11 replaced the lib/-is-a-directory proxy precisely so the guard
        would mirror the audit's real failure. The first version still used
        ``exists()`` and an unrestricted ``{soname}.*`` glob, so each of these
        passed the guard while ``resolve_library`` found nothing -- guard exits
        0, audit exits 2, and the build-time check hands the problem downstream.
        The debug-info file is the realistic one.
        """
        root = build_classic(tmp_path / "opt_rocm", sonames=False)
        lib = root / "lib"
        # rocBLAS present and well-formed, so the ONLY thing under test is
        # whether the hipBLASLt decoy satisfies the guard.
        (lib / "librocblas.so.5").write_bytes(b"\x7fELF")
        decoy = lib / name
        decoy.mkdir() if make_dir else decoy.write_bytes(b"\x7fELF")
        classic_at(root)

        assert guard.main() == 1
        err = capsys.readouterr().err
        assert "libhipblaslt.so" in err
        # rocBLAS resolved, so it must not be blamed.
        assert "librocblas.so" not in err

    def test_the_guard_and_the_audit_agree_on_every_soname_shape(self, tmp_path: Path):
        """Cross-check the two implementations directly, not case by case.

        ``_has_soname`` claims to mirror ``resolve_library``; this asserts it
        over the shapes that distinguish a loose matcher from a faithful one,
        so a future relaxation of either side shows up here rather than as a
        build that passes and an audit that exits 2.
        """
        audit = _load_module_from_path(
            "audit_env_knobs",
            Path(__file__).resolve().parents[2] / "scripts" / "audit_env_knobs.py",
        )
        soname = "libhipblaslt.so"
        shapes = {
            "bare-link-is-a-file": lambda d: (d / soname).write_bytes(b"\x7fELF"),
            "bare-link-is-a-dir": lambda d: (d / soname).mkdir(),
            "single-digit-major": lambda d: (d / f"{soname}.4").write_bytes(b"\x7fELF"),
            "two-digit-major": lambda d: (d / f"{soname}.12").write_bytes(b"\x7fELF"),
            "major-is-a-dir": lambda d: (d / f"{soname}.1").mkdir(),
            "non-numeric-suffix": lambda d: (d / f"{soname}.debug").write_bytes(b"\x7fELF"),
            "debug-beside-real-major": lambda d: (
                (d / f"{soname}.debug").write_bytes(b"\x7fELF"),
                (d / f"{soname}.3").write_bytes(b"\x7fELF"),
            ),
            "empty": lambda d: None,
        }
        for label, build in shapes.items():
            lib = tmp_path / label / "lib"
            lib.mkdir(parents=True)
            build(lib)
            assert guard._has_soname(lib, soname) is (
                audit.resolve_library(lib, soname) is not None
            ), label

    def test_guard_requires_the_sonames_the_audit_consumes(self):
        """Read out of the audit, not restated, so the two cannot drift.

        The guard's whole claim is that it mirrors a real consumer. If the
        audit starts resolving a third library, a hardcoded copy here would
        keep passing images the audit then fails on.
        """
        audit = _load_module_from_path(
            "audit_env_knobs",
            Path(__file__).resolve().parents[2] / "scripts" / "audit_env_knobs.py",
        )
        assert guard.AUDIT_SONAMES == audit.DEFAULT_SONAMES


class TestGuardIsSelfContained:
    def test_imports_only_the_standard_library(self):
        """It runs before any pip install, on the base image's interpreter."""
        source = _GUARD_PATH.read_text(encoding="utf-8")
        imported = {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "__future__" not in line
        }
        assert imported <= {"codecs", "importlib", "os", "sys", "pathlib"}
