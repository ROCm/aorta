"""Tests for layout-agnostic ROCm root discovery (issue #381).

``resolve_rocm_roots`` decides where every ROCm path in the tree comes from,
so its resolution ORDER is the behaviour worth pinning: an explicit operator
override must beat autodetection, and the classic system install must keep
priority over a wheel that merely happens to be importable.

Two things make these tests hermetic, which matters because the module reads
real host state:

* ``resolve_rocm_roots(environ=...)`` takes an injected mapping, so nothing
  here needs ``monkeypatch.setenv``;
* the ``no_rocm`` fixture points ``CLASSIC_ROCM_ROOT`` at an absent directory
  and stubs wheel discovery, so a developer laptop and a GPU box with a real
  ``/opt/rocm`` produce the same result. Without it these tests would pass or
  fail depending on the machine -- exactly the ambiguity #381 set out to end.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from aorta.instrumentation import rocm_paths
from aorta.instrumentation.rocm_paths import (
    LAYOUT_CLASSIC,
    LAYOUT_WHEEL,
    resolve_rocm_roots,
)


def build_classic(root: Path, *, version: str | None = "7.2.4", lib: bool = True) -> Path:
    """A ``/opt/rocm``-shaped tree: one root holding bin/, lib/ and .info/."""
    (root / "bin").mkdir(parents=True, exist_ok=True)
    if lib:
        (root / "lib").mkdir(exist_ok=True)
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
    version: str = "7.14.0",
) -> Path:
    """A TheRock site-packages tree; returns the ``_rocm_sdk_core`` directory."""
    core = site / "_rocm_sdk_core"
    (core / "bin").mkdir(parents=True)
    info = core / ".info"
    info.mkdir()
    (info / "version").write_text(f"{version}\n", encoding="utf-8")
    therock = core / "share" / "therock"
    therock.mkdir(parents=True)
    (therock / "therock_manifest.json").write_text("{}", encoding="utf-8")
    if libraries:
        (site / "_rocm_sdk_libraries" / "lib").mkdir(parents=True)
    if devel:
        (site / "_rocm_sdk_devel" / "include").mkdir(parents=True)
    return core


@pytest.fixture
def no_rocm(tmp_path: Path, monkeypatch):
    """Neutralise host ROCm so only what a test builds can be discovered."""
    monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", tmp_path / "absent_opt_rocm")
    monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda package: None)
    return monkeypatch


class TestResolutionOrder:
    def test_rocm_path_is_honoured(self, no_rocm, tmp_path: Path):
        root = build_classic(tmp_path / "rocm")
        roots = resolve_rocm_roots({"ROCM_PATH": str(root)})
        assert roots.core == root
        assert roots.source == "ROCM_PATH"
        assert roots.layout == LAYOUT_CLASSIC

    def test_rocm_home_is_honoured_when_rocm_path_is_unset(self, no_rocm, tmp_path: Path):
        root = build_classic(tmp_path / "rocm")
        roots = resolve_rocm_roots({"ROCM_HOME": str(root)})
        assert roots.core == root
        assert roots.source == "ROCM_HOME"

    def test_rocm_path_beats_rocm_home(self, no_rocm, tmp_path: Path):
        preferred = build_classic(tmp_path / "preferred")
        other = build_classic(tmp_path / "other")
        roots = resolve_rocm_roots({"ROCM_PATH": str(preferred), "ROCM_HOME": str(other)})
        assert roots.core == preferred
        assert roots.source == "ROCM_PATH"

    def test_empty_env_value_is_ignored(self, no_rocm, tmp_path: Path):
        root = build_classic(tmp_path / "rocm")
        roots = resolve_rocm_roots({"ROCM_PATH": "", "ROCM_HOME": str(root)})
        assert roots.source == "ROCM_HOME"

    def test_env_var_pointing_at_a_non_rocm_directory_falls_through(self, no_rocm, tmp_path: Path):
        """A stale override must not shadow a real install.

        The candidate is validated rather than trusted, so ROCM_PATH left over
        from another image loses to the ROCm that is actually present.
        """
        empty = tmp_path / "empty"
        empty.mkdir()
        real = build_classic(tmp_path / "rocm")
        roots = resolve_rocm_roots({"ROCM_PATH": str(empty), "ROCM_HOME": str(real)})
        assert roots.core == real
        assert roots.source == "ROCM_HOME"

    def test_opt_rocm_used_when_no_env_vars_are_set(self, tmp_path: Path, monkeypatch):
        root = build_classic(tmp_path / "opt_rocm")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", root)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        roots = resolve_rocm_roots({})
        assert roots.core == root
        assert roots.source == "opt_rocm"
        assert roots.layout == LAYOUT_CLASSIC

    def test_empty_opt_rocm_does_not_shadow_a_wheel(self, tmp_path: Path, monkeypatch):
        """A leftover empty /opt/rocm must not win over a real wheel install.

        This is why ``/opt/rocm`` is marker-validated instead of merely
        existence-checked: uninstalling classic ROCm can leave the directory
        behind, and trusting it would report a null version on a working box.
        """
        stale = tmp_path / "opt_rocm"
        stale.mkdir()
        core = build_wheel(tmp_path / "site-packages")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", stale)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: core)
        roots = resolve_rocm_roots({})
        assert roots.core == core
        assert roots.layout == LAYOUT_WHEEL
        assert roots.source == "import:_rocm_sdk_core"

    def test_bin_only_opt_rocm_stub_does_not_shadow_a_wheel(
        self, tmp_path: Path, monkeypatch
    ):
        """A compat shim is not an install.

        Stricter than the empty-directory case above, and the one that actually
        bites: a wheel-based image can reasonably ship a bin-only ``/opt/rocm``
        so ``hipcc`` stays on ``PATH``. "Has a bin/" is not evidence enough to
        outrank a working wheel -- doing so reports a null version on a healthy
        box, relocating the #381 failure instead of removing it.
        """
        stub = tmp_path / "opt_rocm"
        (stub / "bin").mkdir(parents=True)
        (stub / "bin" / "hipcc").write_text("#!/bin/sh\n", encoding="utf-8")
        core = build_wheel(tmp_path / "site-packages")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", stub)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: core)
        roots = resolve_rocm_roots({})
        assert roots.core == core
        assert roots.layout == LAYOUT_WHEEL

    def test_bin_only_stub_with_no_wheel_reports_nothing_found(
        self, tmp_path: Path, monkeypatch
    ):
        """With no wheel to fall back to, the stub is still not an install."""
        stub = tmp_path / "opt_rocm"
        (stub / "bin").mkdir(parents=True)
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", stub)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        roots = resolve_rocm_roots({})
        assert roots.source == "none"
        assert roots.core == stub  # still absolute, still the classic root

    @pytest.mark.parametrize("marker", ["version", "version-dev"])
    def test_a_version_marker_alone_makes_opt_rocm_usable(
        self, tmp_path: Path, monkeypatch, marker
    ):
        stub = tmp_path / "opt_rocm"
        (stub / "bin").mkdir(parents=True)
        (stub / ".info").mkdir()
        (stub / ".info" / marker).write_text("7.2.4\n", encoding="utf-8")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", stub)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        assert resolve_rocm_roots({}).source == "opt_rocm"

    def test_a_lib_dir_alone_makes_opt_rocm_usable(self, tmp_path: Path, monkeypatch):
        """A tarball install with no ``.info/`` must still be accepted.

        The stricter autodetect test must not narrow the classic layouts that
        genuinely work -- only reject the ones nothing can be read from.
        """
        root = tmp_path / "opt_rocm"
        (root / "lib").mkdir(parents=True)
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", root)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        assert resolve_rocm_roots({}).source == "opt_rocm"

    def test_an_explicit_override_still_wins_on_a_bin_only_stub(
        self, tmp_path: Path, monkeypatch
    ):
        """Stated intent beats autodetection, and ``source`` records whose it was.

        The loose/strict asymmetry is deliberate: an operator who names a root
        gets it even if we cannot read anything from it, because
        ``root_source: "ROCM_HOME"`` makes the resulting null attributable to
        their override rather than to a discovery failure.
        """
        stub = tmp_path / "stub"
        (stub / "bin").mkdir(parents=True)
        core = build_wheel(tmp_path / "site-packages")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", tmp_path / "absent")
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: core)
        roots = resolve_rocm_roots({"ROCM_HOME": str(stub)})
        assert roots.core == stub
        assert roots.source == "ROCM_HOME"
        assert roots.layout == LAYOUT_CLASSIC

    def test_importable_wheel_used_when_nothing_else_is_found(self, tmp_path: Path, monkeypatch):
        core = build_wheel(tmp_path / "site-packages")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", tmp_path / "absent")
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: core)
        roots = resolve_rocm_roots({})
        assert roots.core == core
        assert roots.libraries == core.parent / "_rocm_sdk_libraries"
        assert roots.source == "import:_rocm_sdk_core"

    def test_opt_rocm_beats_an_importable_wheel(self, tmp_path: Path, monkeypatch):
        """Classic keeps priority: a venv wheel must not hijack a system install."""
        classic = build_classic(tmp_path / "opt_rocm")
        core = build_wheel(tmp_path / "site-packages")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", classic)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: core)
        roots = resolve_rocm_roots({})
        assert roots.core == classic
        assert roots.layout == LAYOUT_CLASSIC

    def test_a_relative_override_is_anchored_to_an_absolute_path(
        self, no_rocm, tmp_path: Path, monkeypatch
    ):
        """Relative roots would change meaning with the working directory (#387).

        ``environment.py`` freezes these into module constants at import, so a
        relative ``$ROCM_PATH`` would make the same snapshot describe different
        trees before and after a ``chdir`` -- and it breaks the absolute-path
        invariant ``TestPathConstants`` asserts over those constants.
        """
        build_classic(tmp_path / "rocm")
        monkeypatch.chdir(tmp_path)
        roots = resolve_rocm_roots({"ROCM_PATH": "rocm"})
        assert roots.core.is_absolute()
        assert roots.core == tmp_path / "rocm"
        # Every derived path inherits it.
        for derived in (roots.lib_dir, roots.include_dir, roots.version_file):
            assert derived.is_absolute()

    def test_an_unanchorable_relative_override_falls_through(
        self, no_rocm, tmp_path: Path, monkeypatch
    ):
        """A deleted working directory must not turn import into a traceback.

        Anchoring a relative override calls ``getcwd()``, which raises when the
        working directory was removed underneath the process. ``environment.py``
        resolves at *import* time, so an escaping OSError would break collection
        for every caller -- including the ones with no ROCm interest at all.
        The override is simply unusable, so resolution continues past it.
        """
        classic = build_classic(tmp_path / "opt_rocm")
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", classic)

        def _no_cwd(_path):
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(rocm_paths.os.path, "abspath", _no_cwd)
        roots = resolve_rocm_roots({"ROCM_PATH": "relative/rocm"})
        # Fell through to the classic root rather than raising.
        assert roots.core == classic
        assert roots.source == "opt_rocm"

    def test_anchoring_does_not_resolve_symlinks(self, tmp_path: Path, monkeypatch):
        """``/opt/rocm`` is normally a symlink and must stay reported as such.

        Lexical anchoring only -- ``resolve()`` would report the versioned target
        (``/opt/rocm-7.2.4``), which is exactly what CLASSIC_ROCM_ROOT documents
        it does not do, so two hosts on one release would stop comparing equal.
        """
        target = build_classic(tmp_path / "rocm-7.2.4")
        link = tmp_path / "rocm"
        link.symlink_to(target, target_is_directory=True)
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", tmp_path / "absent")
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        monkeypatch.chdir(tmp_path)
        assert resolve_rocm_roots({"ROCM_PATH": "rocm"}).core == link

    def test_nothing_found_returns_classic_root_with_source_none(self, no_rocm, tmp_path: Path):
        """The all-absent case: never raise, never return None.

        ``source="none"`` is what makes a downstream null attributable --
        "no ROCm install found" rather than "install found, but no version
        file in it".
        """
        roots = resolve_rocm_roots({})
        assert roots.source == "none"
        assert roots.layout == LAYOUT_CLASSIC
        assert roots.core == rocm_paths.CLASSIC_ROCM_ROOT
        assert roots.core.is_absolute()
        assert roots.lib_dir.is_absolute()


class TestWheelLayout:
    def test_env_var_on_a_component_directory_assembles_its_siblings(self, no_rocm, tmp_path: Path):
        """``rocm/pytorch:latest`` sets ROCM_PATH to ``_rocm_sdk_devel``.

        The pointed-at component is not the whole install, so the siblings
        beside it have to be picked up or the math libraries go missing.
        """
        site = tmp_path / "site-packages"
        core = build_wheel(site, devel=True)
        roots = resolve_rocm_roots({"ROCM_PATH": str(site / "_rocm_sdk_devel")})
        assert roots.layout == LAYOUT_WHEEL
        assert roots.source == "ROCM_PATH"
        assert roots.core == core
        assert roots.libraries == site / "_rocm_sdk_libraries"
        assert roots.include == site / "_rocm_sdk_devel"

    def test_libraries_falls_back_to_core_when_the_component_is_absent(
        self, no_rocm, tmp_path: Path
    ):
        site = tmp_path / "site-packages"
        core = build_wheel(site, libraries=False)
        roots = resolve_rocm_roots({"ROCM_PATH": str(core)})
        assert roots.libraries == core

    def test_include_falls_back_to_core_on_a_runtime_only_install(self, no_rocm, tmp_path: Path):
        """Runtime-only images ship no ``_rocm_sdk_devel``, hence no headers.

        The include root still has to be an absolute path so the header reads
        simply miss instead of crashing the probe.
        """
        site = tmp_path / "site-packages"
        core = build_wheel(site, devel=False)
        roots = resolve_rocm_roots({"ROCM_PATH": str(core)})
        assert roots.include == core
        assert not (roots.include_dir / "hipblaslt").exists()

    def test_lone_component_without_a_core_sibling_is_used_for_every_root(
        self, no_rocm, tmp_path: Path
    ):
        lone = tmp_path / "site-packages" / "_rocm_sdk_devel"
        lone.mkdir(parents=True)
        roots = resolve_rocm_roots({"ROCM_PATH": str(lone)})
        assert roots.layout == LAYOUT_WHEEL
        assert roots.core == roots.libraries == roots.include == lone


class TestClassicRegression:
    """#381 acceptance: no behaviour change on a classic image."""

    def test_all_three_roots_coincide(self, no_rocm, tmp_path: Path):
        root = build_classic(tmp_path / "rocm")
        roots = resolve_rocm_roots({"ROCM_PATH": str(root)})
        assert roots.core == roots.libraries == roots.include == root

    def test_derived_paths_are_the_literals_they_replaced(self):
        """Byte-for-byte the hardcoded constants removed from environment.py.

        If this drifts, every classic ROCm install starts reading a different
        path than it did before #381 -- which is the one outcome the rewrite
        was not allowed to produce.
        """
        roots = rocm_paths._classic_roots("opt_rocm")
        assert roots.bin_dir == Path("/opt/rocm/bin")
        assert roots.version_file == Path("/opt/rocm/.info/version")
        assert roots.version_dev_file == Path("/opt/rocm/.info/version-dev")
        assert roots.lib_dir == Path("/opt/rocm/lib")
        assert roots.include_dir == Path("/opt/rocm/include")
        assert roots.manifest_file == Path("/opt/rocm/share/therock/therock_manifest.json")
        assert roots.llvm_bin_dir == Path("/opt/rocm/lib/llvm/bin")

    def test_opt_rocm_symlink_is_not_resolved(self, tmp_path: Path, monkeypatch):
        """``/opt/rocm`` normally points at ``/opt/rocm-7.2.4``.

        The snapshot should report the stable path an operator would type, not
        the versioned target, so two boxes on the same release compare equal.
        """
        target = build_classic(tmp_path / "rocm-7.2.4")
        link = tmp_path / "rocm"
        link.symlink_to(target, target_is_directory=True)
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", link)
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        roots = resolve_rocm_roots({})
        assert roots.core == link


class TestDerivedPaths:
    def test_properties_hang_off_the_right_root(self, no_rocm, tmp_path: Path):
        site = tmp_path / "site-packages"
        core = build_wheel(site, devel=True)
        roots = resolve_rocm_roots({"ROCM_PATH": str(core)})
        libraries = site / "_rocm_sdk_libraries"
        devel = site / "_rocm_sdk_devel"

        assert roots.bin_dir == core / "bin"
        assert roots.version_file == core / ".info" / "version"
        assert roots.version_dev_file == core / ".info" / "version-dev"
        assert roots.manifest_file == core / "share" / "therock" / "therock_manifest.json"
        assert roots.lib_dir == libraries / "lib"
        assert roots.include_dir == devel / "include"
        # The LLVM tools ship with the core component in both layouts, not
        # with the math libraries and not with the devel headers.
        assert roots.llvm_bin_dir == core / "lib" / "llvm" / "bin"

    def test_roots_are_frozen(self, no_rocm, tmp_path: Path):
        roots = resolve_rocm_roots({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            roots.core = Path("/somewhere/else")  # type: ignore[misc]


class TestVersionMarkerMustBeReadable:
    """``exists()`` is too weak to grant classic autodetection priority (#387).

    Each of these would otherwise let a stale ``/opt/rocm`` outrank a healthy
    importable wheel and then report a null version -- the same class as the
    bin-only ``bin/`` shim, one marker further in.
    """

    @pytest.fixture
    def wheel_beside(self, tmp_path: Path, monkeypatch):
        """A real wheel install that a stale /opt/rocm must not outrank."""
        core = build_wheel(tmp_path / "site-packages")
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: core)
        return core

    def _stub_root(self, tmp_path: Path, monkeypatch) -> Path:
        stub = tmp_path / "opt_rocm"
        (stub / "bin").mkdir(parents=True)
        (stub / ".info").mkdir()
        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", stub)
        return stub

    def test_zero_byte_version_file_is_not_usable(
        self, tmp_path: Path, monkeypatch, wheel_beside
    ):
        """An interrupted install leaves exactly this behind."""
        stub = self._stub_root(tmp_path, monkeypatch)
        (stub / ".info" / "version").write_text("", encoding="utf-8")
        assert resolve_rocm_roots({}).core == wheel_beside

    def test_whitespace_only_version_file_is_not_usable(
        self, tmp_path: Path, monkeypatch, wheel_beside
    ):
        stub = self._stub_root(tmp_path, monkeypatch)
        (stub / ".info" / "version").write_text("\n  \n", encoding="utf-8")
        assert resolve_rocm_roots({}).core == wheel_beside

    def test_a_directory_named_version_is_not_usable(
        self, tmp_path: Path, monkeypatch, wheel_beside
    ):
        """``exists()`` accepts a directory; a read raises IsADirectoryError."""
        stub = self._stub_root(tmp_path, monkeypatch)
        (stub / ".info" / "version").mkdir()
        assert resolve_rocm_roots({}).core == wheel_beside

    def test_unreadable_version_file_is_not_usable(
        self, tmp_path: Path, monkeypatch, wheel_beside
    ):
        stub = self._stub_root(tmp_path, monkeypatch)
        marker = stub / ".info" / "version"
        marker.write_text("7.2.4\n", encoding="utf-8")
        marker.chmod(0o000)
        try:
            if os.access(marker, os.R_OK):  # running as root: chmod proves nothing
                pytest.skip("cannot make a file unreadable as this user")
            assert resolve_rocm_roots({}).core == wheel_beside
        finally:
            marker.chmod(0o644)

    def test_a_real_version_file_is_still_usable(self, tmp_path: Path, monkeypatch):
        """The whole point: this must not narrow a working classic install."""
        stub = self._stub_root(tmp_path, monkeypatch)
        (stub / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
        monkeypatch.setattr(rocm_paths, "_installed_wheel_component", lambda p: None)
        assert resolve_rocm_roots({}).source == "opt_rocm"

    def test_invalid_bytes_past_the_old_probe_window_are_not_usable(
        self, tmp_path: Path, monkeypatch, wheel_beside
    ):
        """Validation and the value read must use the SAME byte limit (#387).

        Validation once probed 64 bytes while every consumer read 4096. A larger
        read sees strictly more bytes, so it can reject content the probe never
        reached: this marker is valid UTF-8 for its first 64 bytes and has an
        invalid byte at offset 64. Discovery therefore accepted it, autodetected
        ``/opt/rocm`` outranked a healthy wheel, and every reader then reported
        ``None`` -- ``rocm.version: null`` with ``root_source: "opt_rocm"`` on a
        working box, which is the #381 failure mode relocated rather than
        removed.

        The guard parity suite could not catch it: both copies carried the same
        constant pair, so they agreed with each other and were wrong together.
        """
        stub = self._stub_root(tmp_path, monkeypatch)
        marker = stub / ".info" / "version"
        marker.write_bytes(b"7.2.4-" + b"x" * 58 + b"\xff" + b"tail")

        assert rocm_paths.read_version_marker(marker) is None
        assert rocm_paths._has_readable_version(stub) is False
        assert resolve_rocm_roots({}).core == wheel_beside

    def test_validation_and_value_read_agree_on_every_marker(self, tmp_path: Path):
        """The invariant behind the bug, asserted directly rather than by case.

        ``_has_readable_version`` is exactly "some marker reads as non-None", so
        a future limit or predicate split shows up here even for content nobody
        thought to enumerate.
        """
        payloads = {
            "short": b"7.2.4\n",
            "long-valid": b"7.2.4-rocm-rel-7.2-24" + b"-pad" * 40,
            "invalid-at-0": b"\xff\xfe",
            "invalid-past-64": b"7.2.4-" + b"x" * 58 + b"\xff" + b"tail",
            "invalid-past-4096": b"7.2.4" + b"x" * 4995 + b"\xff",
            "empty": b"",
            "whitespace": b"  \n\t ",
        }
        for name, payload in payloads.items():
            root = tmp_path / name
            (root / ".info").mkdir(parents=True)
            marker = root / ".info" / "version"
            marker.write_bytes(payload)
            assert rocm_paths._has_readable_version(root) is (
                rocm_paths.read_version_marker(marker) is not None
            ), name


class TestNeverRaises:
    def test_unreadable_candidate_is_not_a_root(self, tmp_path: Path, monkeypatch):
        def boom(self):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "is_dir", boom)
        assert rocm_paths._is_rocm_root(tmp_path) is False

    def test_every_filesystem_probe_is_fail_soft(self, tmp_path: Path, monkeypatch):
        """A stale mount must not escape as an exception (#387).

        The module is imported at module scope by ``environment.py`` to build its
        path constants, so an ``OSError`` from any probe here does not just
        mis-resolve -- it can stop the env probe importing at all, on exactly the
        damaged hosts it exists to describe. Blows up ``is_dir`` globally so
        every probe on every resolution path is covered at once, rather than
        naming the ones that happened to be flagged.
        """

        def boom(self):
            raise OSError("stale NFS handle")

        monkeypatch.setattr(Path, "is_dir", boom)
        for environ in (
            {},
            {"ROCM_PATH": str(tmp_path / "rocm")},
            {"ROCM_HOME": str(tmp_path / "rocm")},
            {"ROCM_PATH": str(tmp_path / "site" / "_rocm_sdk_devel")},
        ):
            roots = resolve_rocm_roots(environ)
            assert roots.source == "none", environ
            assert roots.core.is_absolute()

    def test_wheel_component_probe_is_fail_soft(self, tmp_path: Path, monkeypatch):
        def boom(self):
            raise OSError("stale NFS handle")

        monkeypatch.setattr(Path, "is_dir", boom)
        assert rocm_paths._wheel_component(tmp_path, "_rocm_sdk_core") is None

    def test_version_marker_probe_is_fail_soft(self, tmp_path: Path, monkeypatch):
        def boom(self, *args, **kwargs):
            raise OSError("stale NFS handle")

        monkeypatch.setattr(Path, "open", boom)
        assert rocm_paths._has_readable_version(tmp_path) is False

    def test_resolution_survives_an_unreadable_env_path(self, no_rocm, tmp_path: Path):
        def boom(self):
            raise OSError("stale NFS handle")

        no_rocm.setattr(Path, "is_dir", boom)
        roots = resolve_rocm_roots({"ROCM_PATH": "/mnt/gone"})
        assert roots.source == "none"

    @pytest.mark.parametrize(
        "exc",
        [
            ImportError("half-installed package"),
            ValueError("__spec__ is not set"),
            # find_spec runs arbitrary path-finder code, so a damaged install or
            # an unreadable site-packages mount can raise these instead. The
            # module documents that resolution never raises, so a narrow catch
            # would break that contract on exactly the broken installs the probe
            # exists to describe.
            OSError("stale NFS handle on site-packages"),
            RuntimeError("broken meta path finder"),
        ],
    )
    def test_find_spec_failure_is_treated_as_absent(self, monkeypatch, exc):
        def boom(name):
            raise exc

        monkeypatch.setattr("importlib.util.find_spec", boom)
        assert rocm_paths._installed_wheel_component("_rocm_sdk_core") is None

    def test_resolution_never_raises_when_find_spec_explodes(
        self, tmp_path: Path, monkeypatch
    ):
        """The end-to-end contract, not just the helper."""

        def boom(name):
            raise OSError("site-packages unreadable")

        monkeypatch.setattr(rocm_paths, "CLASSIC_ROCM_ROOT", tmp_path / "absent")
        monkeypatch.setattr("importlib.util.find_spec", boom)
        roots = resolve_rocm_roots({})
        assert roots.source == "none"
        assert roots.core.is_absolute()

    def test_injected_environ_is_used_instead_of_os_environ(self, no_rocm, tmp_path: Path):
        """The injected mapping must fully replace the process environment.

        The standalone CI scripts rely on this to resolve deterministically.
        """
        root = build_classic(tmp_path / "rocm")
        no_rocm.setenv("ROCM_PATH", str(tmp_path / "should_be_ignored"))
        roots = resolve_rocm_roots({"ROCM_HOME": str(root)})
        assert roots.core == root
        assert roots.source == "ROCM_HOME"


class TestSourceVocabulary:
    def test_source_is_always_one_of_the_documented_values(self, no_rocm, tmp_path: Path):
        """``root_source`` is a documented enum in the env.json schema (1.16)."""
        documented = {
            "ROCM_PATH",
            "ROCM_HOME",
            "opt_rocm",
            "import:_rocm_sdk_core",
            "none",
        }
        classic = build_classic(tmp_path / "rocm")
        core = build_wheel(tmp_path / "site-packages")
        observed = {
            resolve_rocm_roots({}).source,
            resolve_rocm_roots({"ROCM_PATH": str(classic)}).source,
            resolve_rocm_roots({"ROCM_HOME": str(core)}).source,
        }
        assert observed <= documented
