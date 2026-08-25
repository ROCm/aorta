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


def _load_guard():
    spec = importlib.util.spec_from_file_location("rocm_layout_guard", _GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def build_classic(root: Path, *, version: str | None = "7.2.4", lib: bool = True) -> Path:
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
    version: str | None = "7.14.0",
) -> Path:
    core = site / "_rocm_sdk_core"
    (core / "bin").mkdir(parents=True)
    if version is not None:
        info = core / ".info"
        info.mkdir()
        (info / "version").write_text(f"{version}\n", encoding="utf-8")
    if libraries:
        (site / "_rocm_sdk_libraries" / "lib").mkdir(parents=True)
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


class TestGuardIsSelfContained:
    def test_imports_only_the_standard_library(self):
        """It runs before any pip install, on the base image's interpreter."""
        source = _GUARD_PATH.read_text(encoding="utf-8")
        imported = {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "__future__" not in line
        }
        assert imported <= {"importlib", "os", "sys", "pathlib"}
