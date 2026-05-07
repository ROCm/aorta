"""Tests for ``aorta.instrumentation.build_system`` (issue #163, A1.2a).

Strategy mirrors ``test_environment.py``: load the module by file path
so the test does not pull in torch via ``aorta.utils``. Every
subprocess and ``shutil.which`` lookup is monkeypatched, so tests run
on any host -- with or without buck2, hg, or git.

Coverage matrix (per the A1.2a acceptance criteria):

* ``buck2`` absent -> ``{"kind": "none"}`` (no subprocess work).
* ``buck2`` present + ``hg`` resolves -> populated dict, hg revision wins.
* ``buck2`` present + ``hg`` absent + ``git`` resolves -> populated dict,
  git revision used.
* ``buck2`` present + neither hg nor git resolves -> populated dict
  with ``revision: None``.
* ``buck2`` present but ``buck2 root`` fails (typical "buck2 installed
  on host, cwd not inside a Buck checkout") -> ``{"kind": "none"}``;
  no revision lookup runs against the wrong directory.
* ``buck2`` present but ``buck2 --version`` fails (binary is on PATH
  but non-functional) -> ``{"kind": "none"}``.
* ``buck2`` present but both ``--version`` and ``root`` fail (broken
  install) -> ``{"kind": "none"}``.
* Subprocess timeout -> degrades cleanly without raising.
* The wrapper in ``environment.py`` (``_detect_build_system_safe``)
  surfaces ``{"kind": "none"}`` if ``detect_build_system`` itself
  raises (defence in depth around the never-raises contract).
* ``EnvSnapshot`` schema includes the field; ``collect_env`` populates
  it; round-trip via ``to_dict`` / ``from_dict`` preserves it.
* ``EnvSnapshot.from_dict`` tolerates schema 1.1 dicts that predate
  the ``build_system`` field (defaults to ``{"kind": "none"}``).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

# -- Direct module load (mirrors test_environment.py) ------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, _REPO_ROOT / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bs_mod = _load_module(
    "src/aorta/instrumentation/build_system.py",
    "aorta.instrumentation.build_system",
)
env_mod = _load_module(
    "src/aorta/instrumentation/environment.py",
    "aorta.instrumentation.environment",
)


# -- Host-isolation fixtures -------------------------------------------------
#
# Tests in this file that go through ``env_mod.collect_env()`` must NOT
# execute the real env-probe subprocesses (sudo/rdhc, hipconfig, nm,
# rocm_agent_enumerator, ...) -- they would slow the suite down, become
# flaky on hosts that have ROCm but not rdhc, and produce different
# results in CI vs. on a developer laptop.
#
# The fixture below mirrors the ``all_disabled`` pattern in
# ``test_environment.py``: it points every filesystem path constant at a
# nonexistent ``tmp_path`` location, forces ``shutil.which`` to ``None``
# so every external tool reports "not on PATH", and sabotages
# ``import torch`` so the optional torch branches take their fallback.
# What's left is pure-Python orchestration -- which is exactly the
# subset these tests want to exercise.


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip env vars that would otherwise leak host state."""
    for name in env_mod.CANONICAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    monkeypatch.delenv("SINGULARITY_NAME", raising=False)
    monkeypatch.delenv("AORTA_DOCKER_IMAGE", raising=False)
    monkeypatch.delenv("AORTA_DOCKER_DIGEST", raising=False)
    return monkeypatch


@pytest.fixture
def all_disabled(isolated_env, tmp_path: Path, monkeypatch):
    """Force every external dep into its 'unavailable' branch.

    Result: ``collect_env`` exercises only pure-Python paths, every
    block returns its null-shaped form, and the only externally-driven
    field left is ``build_system`` -- which is exactly what these
    tests want to drive. Triggers ``partial=True`` with one reason per
    fallback, but the build_system tests don't assert on
    ``partial_reasons`` so that's fine.

    The patched path constants are kept in lock-step with
    ``test_environment.py::all_disabled`` -- if you add a new path
    constant there, mirror it here too (or move both into a shared
    conftest).
    """
    for attr, leaf in (
        ("ROCM_VERSION_FILE", "no_rocm"),
        ("ROCM_VERSION_DEV_FILE", "no_rocm_dev"),
        ("KMD_VERSION_FILE", "no_kmd"),
        ("HIPBLASLT_VERSION_HEADER", "no_header.h"),
        ("HIPBLASLT_LIB_DIR", "no_libs"),
        ("HIPBLASLT_TENSILE_DIR", "no_tensile"),
        ("ROCBLAS_VERSION_HEADER", "no_rocblas_header.h"),
        ("ROCBLAS_LIB_DIR", "no_rocblas_libs"),
        ("ROCBLAS_TENSILE_DIR", "no_rocblas_tensile"),
        ("CK_VERSION_HEADER", "no_ck.h"),
        ("CK_TILE_CONFIG_HEADER", "no_ck_tile.hpp"),
        ("MIOPEN_VERSION_HEADER", "no_miopen_version.h"),
        ("MIOPEN_LIB_DIR", "no_miopen_libs"),
        ("MIOPEN_KERNEL_DB_DIR", "no_miopen_db"),
        ("RCCL_VERSION_HEADER", "no_rccl.h"),
        ("RCCL_LIB_DIR", "no_rccl_libs"),
        ("DOCKERENV_MARKER", "no_dockerenv"),
        ("PODMAN_CONTAINERENV_MARKER", "no_podmanenv"),
        ("CGROUP_FILE", "no_cgroup"),
        ("SELF_CGROUP_FILE", "no_self_cgroup"),
    ):
        monkeypatch.setattr(env_mod, attr, tmp_path / leaf)
    # Disable every external binary lookup. Tests that DO want buck2
    # present override this via their own ``bs_mod.detect_build_system``
    # monkeypatch (i.e., they bypass shutil.which entirely).
    #
    # Why patch BOTH ``env_mod.shutil`` and ``bs_mod.shutil``: at
    # runtime ``env_mod.shutil is bs_mod.shutil is shutil`` (they are
    # all references to the one stdlib module), so a single
    # monkeypatch on either target is functionally enough. We patch
    # both anyway, for two reasons: (1) the fixture's intent --
    # "neither the env probe NOR the build_system module sees any
    # external binary" -- is then spelled out for the next reader,
    # who shouldn't need to reason about Python's import-time module
    # identity; (2) it future-proofs against a refactor that rebinds
    # ``shutil`` locally inside ``build_system.py``.
    def _no_which(name: str) -> None:
        return None

    monkeypatch.setattr(env_mod.shutil, "which", _no_which)
    monkeypatch.setattr(bs_mod.shutil, "which", _no_which)
    # Force the optional ``import torch`` path inside collect_env to
    # take its fallback branch even on hosts where torch is installed.
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    return monkeypatch


# -- Helpers -----------------------------------------------------------------


def _which_factory(present: set[str]) -> "callable":
    """Build a ``shutil.which`` replacement that yields a path for any
    name in ``present`` and ``None`` otherwise. Matches stdlib's
    name-only lookup contract.
    """

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return fake_which


def _run_factory(table: dict[tuple[str, ...], object]) -> "callable":
    """Build a ``subprocess.run`` replacement keyed by argv prefix.

    ``table`` maps a tuple-prefix of the command (e.g., ``("buck2",
    "--version")``) to either a ``CompletedProcess``-style tuple
    ``(returncode, stdout)``, an ``Exception`` instance to raise, or a
    callable to invoke with ``(cmd, kwargs)``.

    Matching is by full-prefix on the argv basenames, so the test stays
    agnostic to whether the cmd is ``buck2`` or ``/usr/bin/buck2``.
    """

    def fake_run(cmd, **kwargs):
        # Normalise: strip directory prefixes so tests can match by
        # basename. Real cmd[0] from production code is the absolute
        # path returned by shutil.which.
        key = tuple([os.path.basename(cmd[0])] + list(cmd[1:]))
        # Look for the longest matching prefix.
        for prefix, response in sorted(
            table.items(), key=lambda kv: -len(kv[0])
        ):
            if key[: len(prefix)] == prefix:
                if isinstance(response, Exception):
                    raise response
                if callable(response):
                    return response(cmd, kwargs)
                returncode, stdout = response
                return subprocess.CompletedProcess(
                    args=cmd, returncode=returncode, stdout=stdout, stderr=""
                )
        raise AssertionError(f"unexpected subprocess call in test: {cmd}")

    return fake_run


# -- detect_build_system: buck2 absent ---------------------------------------


class TestBuck2Absent:
    def test_returns_none_kind_when_buck2_not_on_path(self, monkeypatch):
        monkeypatch.setattr(bs_mod.shutil, "which", _which_factory(set()))
        # No subprocess calls should happen.
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            lambda *a, **kw: pytest.fail("subprocess.run called when buck2 absent"),
        )
        assert bs_mod.detect_build_system() == {"kind": "none"}


# -- detect_build_system: buck2 present, VCS variants ------------------------


class TestBuck2Present:
    def test_hg_revision_wins_when_hg_present(self, monkeypatch):
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2", "hg", "git"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (0, "buck2 2025-04-15\n"),
                    ("buck2", "root"): (0, "/data/users/me/monorepo\n"),
                    # hg present and resolves -- git must NOT be invoked.
                    ("hg", "id", "-i"): (0, "deadbeefcafe\n"),
                }
            ),
        )
        result = bs_mod.detect_build_system()
        assert result == {
            "kind": "buck2",
            "buck2_version": "buck2 2025-04-15",
            "repo_root": "/data/users/me/monorepo",
            "revision": "deadbeefcafe",
        }

    def test_hg_dirty_suffix_preserved(self, monkeypatch):
        """``hg id -i`` appends ``+`` for an uncommitted-changes working
        copy. Dropping it would lose triage signal."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2", "hg"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (0, "buck2 2025-04-15"),
                    ("buck2", "root"): (0, "/repo"),
                    ("hg", "id", "-i"): (0, "abc1234+\n"),
                }
            ),
        )
        result = bs_mod.detect_build_system()
        assert result["revision"] == "abc1234+"

    def test_git_revision_used_when_hg_absent(self, monkeypatch):
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2", "git"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (0, "buck2 2025-04-15"),
                    ("buck2", "root"): (0, "/data/users/me/monorepo"),
                    ("git", "rev-parse", "HEAD"): (0, "0123456789abcdef" * 2 + "\n"),
                }
            ),
        )
        result = bs_mod.detect_build_system()
        assert result["kind"] == "buck2"
        assert result["revision"] == "0123456789abcdef" * 2

    def test_git_used_when_hg_present_but_returns_nothing(self, monkeypatch):
        """Outside an hg repo, ``hg id -i`` exits non-zero. Detector
        must fall through to git."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2", "hg", "git"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (0, "buck2 2025-04-15"),
                    ("buck2", "root"): (0, "/repo"),
                    ("hg", "id", "-i"): (255, ""),  # not in an hg repo
                    ("git", "rev-parse", "HEAD"): (0, "abc123\n"),
                }
            ),
        )
        result = bs_mod.detect_build_system()
        assert result["revision"] == "abc123"

    def test_revision_is_none_when_no_vcs_resolves(self, monkeypatch):
        """A fresh Buck2 sample repo with no VCS should still produce a
        valid build_system dict with ``revision: None``, NOT crash and
        NOT downgrade to ``kind: none``."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (0, "buck2 2025-04-15"),
                    ("buck2", "root"): (0, "/tmp/sample"),
                }
            ),
        )
        result = bs_mod.detect_build_system()
        assert result == {
            "kind": "buck2",
            "buck2_version": "buck2 2025-04-15",
            "repo_root": "/tmp/sample",
            "revision": None,
        }


# -- detect_build_system: degraded buck2 -------------------------------------


class TestBuck2Degraded:
    def test_both_buck2_calls_failing_downgrades_to_none(self, monkeypatch):
        """``buck2`` on PATH but completely broken (e.g., orphaned
        wrapper script after toolchain removal) should NOT produce a
        half-shaped buck2 dict; downgrade to ``kind: none``."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (1, ""),
                    ("buck2", "root"): (1, ""),
                }
            ),
        )
        assert bs_mod.detect_build_system() == {"kind": "none"}

    def test_buck2_installed_but_cwd_not_in_repo_downgrades_to_none(
        self, monkeypatch
    ):
        """The dominant non-buck case on a developer laptop: buck2 is
        on PATH (system-wide install, vendored toolchain) but the
        current working directory is not inside any Buck checkout, so
        ``buck2 root`` exits non-zero. The detector MUST report
        ``kind=none`` rather than emit a half-shaped buck2 dict --
        otherwise downstream consumers (B1/B2, recipe emitter) treat
        the env as Buck2 and the revision lookup runs against an
        arbitrary cwd, producing a misleading SHA.
        """
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2", "hg", "git"})
        )
        # `buck2 --version` succeeds, `buck2 root` fails (not in a Buck
        # repo). hg and git would both resolve a SHA against cwd if we
        # let them -- but we must NOT, because cwd is not the relevant
        # repo. The fake_run table omits hg/git entries to assert
        # neither is invoked.
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (0, "buck2 2025-04-15"),
                    ("buck2", "root"): (1, "not in a buck2 project\n"),
                }
            ),
        )
        assert bs_mod.detect_build_system() == {"kind": "none"}

    def test_buck2_root_succeeds_but_version_fails_downgrades_to_none(
        self, monkeypatch
    ):
        """Symmetric inverse: ``buck2 root`` somehow resolves but
        ``buck2 --version`` is broken (orphaned wrapper, missing
        toolchain). The buck2 install isn't trustworthy -- downgrade
        rather than emit ``buck2_version=None``."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): (1, ""),
                    ("buck2", "root"): (0, "/repo"),
                }
            ),
        )
        assert bs_mod.detect_build_system() == {"kind": "none"}

    def test_subprocess_timeout_does_not_raise(self, monkeypatch):
        """TimeoutExpired must be caught -- the env probe's never-raises
        contract extends through this module."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): subprocess.TimeoutExpired(
                        cmd=["buck2", "--version"], timeout=5
                    ),
                    ("buck2", "root"): subprocess.TimeoutExpired(
                        cmd=["buck2", "root"], timeout=5
                    ),
                }
            ),
        )
        # Both core calls timed out -> downgrade to kind=none, no exception.
        assert bs_mod.detect_build_system() == {"kind": "none"}

    def test_oserror_does_not_raise(self, monkeypatch):
        """OSError (e.g., transient ENOMEM at fork) must be caught."""
        monkeypatch.setattr(
            bs_mod.shutil, "which", _which_factory({"buck2"})
        )
        monkeypatch.setattr(
            bs_mod.subprocess,
            "run",
            _run_factory(
                {
                    ("buck2", "--version"): OSError("Cannot allocate memory"),
                    ("buck2", "root"): OSError("Cannot allocate memory"),
                }
            ),
        )
        assert bs_mod.detect_build_system() == {"kind": "none"}


# -- _detect_build_system_safe wrapper in environment.py ---------------------


class TestEnvironmentWrapper:
    def test_safe_wrapper_swallows_unexpected_exception(self, monkeypatch):
        """If detect_build_system itself raises (e.g., a future
        regression), the wrapper must still return a fully-shaped dict
        rather than break collect_env's never-raises contract."""

        def boom():
            raise RuntimeError("synthetic future regression")

        # Patch the import target inside the wrapper. The wrapper
        # imports detect_build_system from the build_system module each
        # call, so monkeypatching the attribute there is sufficient.
        monkeypatch.setattr(bs_mod, "detect_build_system", boom)
        result = env_mod._detect_build_system_safe()
        assert result == {"kind": "none"}

    def test_safe_wrapper_returns_buck2_dict_when_detector_succeeds(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            bs_mod,
            "detect_build_system",
            lambda: {
                "kind": "buck2",
                "buck2_version": "x",
                "repo_root": "y",
                "revision": "z",
            },
        )
        assert env_mod._detect_build_system_safe() == {
            "kind": "buck2",
            "buck2_version": "x",
            "repo_root": "y",
            "revision": "z",
        }


# -- EnvSnapshot integration -------------------------------------------------


class TestSnapshotIntegration:
    def test_envsnapshot_has_build_system_field(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(env_mod.EnvSnapshot)}
        assert "build_system" in names

    def test_build_system_field_has_default_factory(self):
        """``build_system`` was added in schema 1.3 as an additive
        field. To avoid breaking direct ``EnvSnapshot(...)`` callers
        that predate 1.3 (e.g., the triage test fixtures pinned to
        schema_version="1.1"), the field must have a
        ``default_factory`` returning ``{"kind": "none"}`` -- the
        same back-compat default ``from_dict`` uses for missing keys.
        Mirrors the existing ``partial_reasons`` defaulting policy:
        additive schema fields don't break constructors.
        """
        import dataclasses

        bs_field = next(
            f
            for f in dataclasses.fields(env_mod.EnvSnapshot)
            if f.name == "build_system"
        )
        assert bs_field.default_factory is not dataclasses.MISSING, (
            "build_system must have a default_factory; otherwise direct "
            "EnvSnapshot(...) callers (triage test fixtures, downstream "
            "tools pinning an older schema) get TypeError when this PR "
            "lands."
        )
        assert bs_field.default_factory() == {"kind": "none"}

    def test_envsnapshot_constructible_without_build_system_kwarg(self):
        """Concrete back-compat check that mirrors the way pre-1.3
        callers (e.g. ``tests/triage/test_output_layout.py``'s
        ``_clean_snapshot`` fixture) construct an ``EnvSnapshot``: a
        long list of explicit kwargs but NO ``build_system``. This
        must not raise, and the resulting snapshot must surface a
        valid ``build_system`` so downstream consumers can branch
        unconditionally on the field.
        """
        snap = env_mod.EnvSnapshot(
            schema_version="1.1",
            captured_at="2026-04-28T14:12:03Z",
            system_health=None,
            rocm={},
            hip={},
            hipblaslt={},
            rocblas={},
            composable_kernel={},
            tensile={},
            triton={},
            fbgemm={},
            aiter={},
            aotriton={},
            miopen={},
            rccl={},
            gpu_arch={},
            runtime_context={},
            host={},
            docker=None,
            env_vars={},
            python_version="3.11.0",
            pytorch_version=None,
            pytorch_build={},
            partial=False,
        )
        assert snap.build_system == {"kind": "none"}

    def test_disaster_snapshot_populates_build_system(self):
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="synthetic"
        )
        assert snap.build_system == {"kind": "none"}

    def test_collect_env_populates_build_system(self, all_disabled):
        """``shutil.which`` is forced to ``None`` by ``all_disabled``,
        so buck2 is absent and detect_build_system reports kind=none
        even on hosts that legitimately have buck2 installed.
        """
        snap = env_mod.collect_env()
        assert snap.build_system == {"kind": "none"}

    def test_to_dict_round_trip_preserves_build_system(
        self, all_disabled, monkeypatch
    ):
        monkeypatch.setattr(
            bs_mod,
            "detect_build_system",
            lambda: {
                "kind": "buck2",
                "buck2_version": "buck2 2025-04-15",
                "repo_root": "/repo",
                "revision": "abc",
            },
        )
        snap = env_mod.collect_env()
        d = snap.to_dict()
        assert d["build_system"]["kind"] == "buck2"
        rebuilt = env_mod.EnvSnapshot.from_dict(d)
        assert rebuilt.build_system == d["build_system"]

    def test_schema_version_includes_build_system_bump(self):
        # A1.2a bumped 1.2 -> 1.3 to add ``build_system``. Subsequent
        # A1.2 phases keep bumping (A1.2b -> 1.4 with
        # ``library_introspection``), so this assertion pins the floor
        # rather than the exact value: ``build_system`` must be present
        # in the current schema version's required key set.
        assert tuple(int(p) for p in env_mod.SCHEMA_VERSION.split(".")) >= (1, 3)

    def test_from_dict_defaults_missing_build_system_to_none_kind(
        self, all_disabled
    ):
        """Loading a schema-1.1 env.json (no ``build_system`` key) into
        a 1.3 reader must NOT raise. A missing field is back-filled
        with the only honest default -- ``{"kind": "none"}`` -- since
        the producer didn't run the build_system probe at all.
        Mirrors the existing ``partial_reasons`` default tolerance.
        """
        snap = env_mod.collect_env()
        d = snap.to_dict()
        d.pop("build_system")
        d["schema_version"] = "1.1"
        rebuilt = env_mod.EnvSnapshot.from_dict(d)
        assert rebuilt.build_system == {"kind": "none"}
        assert rebuilt.schema_version == "1.1"
        assert rebuilt.python_version == snap.python_version

    def test_summary_renders_none_for_baremetal(self, all_disabled):
        snap = env_mod.collect_env()
        assert "build_sys: none" in snap.summary()

    def test_summary_renders_buck2_signature(self, all_disabled, monkeypatch):
        monkeypatch.setattr(
            bs_mod,
            "detect_build_system",
            lambda: {
                "kind": "buck2",
                "buck2_version": "buck2 2025-04-15",
                "repo_root": "/repo",
                "revision": "abcdef0123",
            },
        )
        snap = env_mod.collect_env()
        line = next(
            ln for ln in snap.summary().splitlines() if "build_sys:" in ln
        )
        assert "buck2=buck2 2025-04-15" in line
        assert "repo_root=/repo" in line
        assert "rev=abcdef01" in line  # short hash
