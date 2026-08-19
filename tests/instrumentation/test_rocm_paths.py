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


class TestNeverRaises:
    def test_unreadable_candidate_is_not_a_root(self, tmp_path: Path, monkeypatch):
        def boom(self):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "is_dir", boom)
        assert rocm_paths._is_rocm_root(tmp_path) is False

    def test_resolution_survives_an_unreadable_env_path(self, no_rocm, tmp_path: Path):
        def boom(self):
            raise OSError("stale NFS handle")

        no_rocm.setattr(Path, "is_dir", boom)
        roots = resolve_rocm_roots({"ROCM_PATH": "/mnt/gone"})
        assert roots.source == "none"

    def test_find_spec_failure_is_treated_as_absent(self, monkeypatch):
        def boom(name):
            raise ValueError("__spec__ is not set")

        monkeypatch.setattr("importlib.util.find_spec", boom)
        assert rocm_paths._installed_wheel_component("_rocm_sdk_core") is None

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
