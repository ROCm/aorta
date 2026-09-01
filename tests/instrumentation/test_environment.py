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

import builtins
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import os
import struct
import subprocess
import sys
import types
import zipfile
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
# The provenance manifest CANONICAL_ENV_VARS is generated from. Reached through
# sys.modules because environment.py has already imported it -- so this adds no
# import that the module under test did not already require.
env_knobs = sys.modules["aorta.instrumentation.env_knobs"]
# Same reasoning: the resolver is already imported by environment.py, and the
# version-marker reader has to be the SAME object the probe reads through for a
# consistency assertion over it to mean anything.
rocm_paths = sys.modules["aorta.instrumentation.rocm_paths"]


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
    # AORTA_RE_IMAGE also suppresses the --execution-context warning, so a
    # host/CI that sets it must not silently defuse the warning tests.
    monkeypatch.delenv("AORTA_RE_IMAGE", raising=False)
    return monkeypatch


@pytest.fixture
def all_disabled(isolated_env, tmp_path: Path, monkeypatch):
    """Force every external dep into its 'unavailable' branch.

    Result: ``collect_env`` exercises only pure-Python paths and every
    block returns its null-shaped form. Triggers ``partial=True`` with
    one reason per fallback.
    """
    # Resolved ROCm roots (#381). Patched so the snapshot is identical whether
    # or not the test host happens to have a real /opt/rocm or an importable
    # TheRock wheel -- otherwise `rocm.root_source` alone would differ between
    # a developer laptop and a GPU box.
    monkeypatch.setattr(env_mod, "ROCM_ROOT", tmp_path / "no_rocm_root")
    monkeypatch.setattr(env_mod, "ROCM_LIB_ROOT", tmp_path / "no_rocm_root")
    monkeypatch.setattr(env_mod, "ROCM_INCLUDE_ROOT", tmp_path / "no_rocm_root")
    monkeypatch.setattr(env_mod, "ROCM_ROOT_SOURCE", "none")
    monkeypatch.setattr(env_mod, "ROCM_LAYOUT", "classic")
    monkeypatch.setattr(
        env_mod, "THEROCK_MANIFEST_FILE", tmp_path / "no_therock_manifest.json"
    )
    _disable_rocm_version_fallbacks(monkeypatch)
    monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "no_rocm")
    monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "no_rocm_dev")
    monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "no_kmd")
    monkeypatch.setattr(
        env_mod, "HIPBLASLT_VERSION_HEADER", tmp_path / "no_header.h"
    )
    monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", tmp_path / "no_libs")
    monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tmp_path / "no_tensile")
    monkeypatch.setattr(
        env_mod, "ROCBLAS_VERSION_HEADER", tmp_path / "no_rocblas_header.h"
    )
    monkeypatch.setattr(env_mod, "ROCBLAS_LIB_DIR", tmp_path / "no_rocblas_libs")
    monkeypatch.setattr(
        env_mod, "ROCBLAS_TENSILE_DIR", tmp_path / "no_rocblas_tensile"
    )
    monkeypatch.setattr(env_mod, "CK_VERSION_HEADER", tmp_path / "no_ck.h")
    monkeypatch.setattr(
        env_mod, "CK_TILE_CONFIG_HEADER", tmp_path / "no_ck_tile.hpp"
    )
    monkeypatch.setattr(
        env_mod, "MIOPEN_VERSION_HEADER", tmp_path / "no_miopen_version.h"
    )
    monkeypatch.setattr(env_mod, "MIOPEN_LIB_DIR", tmp_path / "no_miopen_libs")
    monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", tmp_path / "no_miopen_db")
    monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path / "no_rocfft")
    monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
    monkeypatch.delenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, raising=False)
    monkeypatch.delenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, raising=False)
    monkeypatch.setattr(
        env_mod, "RCCL_VERSION_HEADER", tmp_path / "no_rccl.h"
    )
    monkeypatch.setattr(env_mod, "RCCL_LIB_DIR", tmp_path / "no_rccl_libs")
    monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
    monkeypatch.setattr(
        env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv"
    )
    monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")
    monkeypatch.setattr(env_mod, "SELF_CGROUP_FILE", tmp_path / "no_self_cgroup")
    monkeypatch.setattr(env_mod, "SELF_MNT_NS", tmp_path / "no_self_mnt_ns")
    monkeypatch.setattr(env_mod, "SELF_CGROUP_NS", tmp_path / "no_self_cgroup_ns")
    monkeypatch.setattr(env_mod, "BOOT_ID_FILE", tmp_path / "no_boot_id")
    monkeypatch.setattr(env_mod, "KFD_DEVICE_NODE", tmp_path / "no_kfd")
    monkeypatch.setattr(env_mod, "KFD_SYSFS_DIR", tmp_path / "no_kfd_sysfs")
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
            "ROCM_ROOT",
            "ROCM_LIB_ROOT",
            "ROCM_CORE_LIB_DIR",
            "ROCM_INCLUDE_ROOT",
            "ROCM_BIN_DIR",
            "ROCM_VERSION_FILE",
            "ROCM_VERSION_DEV_FILE",
            "THEROCK_MANIFEST_FILE",
            "KMD_VERSION_FILE",
            "KFD_DEVICE_NODE",
            "KFD_SYSFS_DIR",
            "HIPBLASLT_VERSION_HEADER",
            "HIPBLASLT_LIB_DIR",
            "HIPBLASLT_TENSILE_DIR",
            "ROCBLAS_VERSION_HEADER",
            "ROCBLAS_LIB_DIR",
            "ROCBLAS_TENSILE_DIR",
            "CK_VERSION_HEADER",
            "CK_TILE_CONFIG_HEADER",
            "MIOPEN_VERSION_HEADER",
            "MIOPEN_LIB_DIR",
            "MIOPEN_KERNEL_DB_DIR",
            "ROCFFT_LIB_DIR",
            "RCCL_VERSION_HEADER",
            "RCCL_LIB_DIR",
            "SYS_CLASS_NET",
            "SYS_CLASS_INFINIBAND",
            "DOCKERENV_MARKER",
            "PODMAN_CONTAINERENV_MARKER",
            "CGROUP_FILE",
            "SELF_CGROUP_FILE",
            "INIT_MNT_NS",
            "SELF_MNT_NS",
            "SELF_CGROUP_NS",
            "BOOT_ID_FILE",
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
            # Roots resolved by aorta.instrumentation.rocm_paths (#381); every
            # ROCm path below is derived from one of them.
            "ROCM_ROOT",
            "ROCM_LIB_ROOT",
            # Schema 1.17: the CORE lib dir, which is a DIFFERENT directory
            # from ROCM_LIB_ROOT/lib in the wheel layout -- it is where
            # libamdhip64 lives, and the linkage probe reads its tags there.
            "ROCM_CORE_LIB_DIR",
            "ROCM_INCLUDE_ROOT",
            "ROCM_BIN_DIR",
            "ROCM_VERSION_FILE",
            "ROCM_VERSION_DEV_FILE",
            "THEROCK_MANIFEST_FILE",
            "KMD_VERSION_FILE",
            "KFD_DEVICE_NODE",
            "KFD_SYSFS_DIR",
            "HIPBLASLT_VERSION_HEADER",
            "HIPBLASLT_LIB_DIR",
            "HIPBLASLT_TENSILE_DIR",
            "ROCBLAS_VERSION_HEADER",
            "ROCBLAS_LIB_DIR",
            "ROCBLAS_TENSILE_DIR",
            "CK_VERSION_HEADER",
            "CK_TILE_CONFIG_HEADER",
            "MIOPEN_VERSION_HEADER",
            "MIOPEN_LIB_DIR",
            "MIOPEN_KERNEL_DB_DIR",
            "ROCFFT_LIB_DIR",
            "RCCL_VERSION_HEADER",
            "RCCL_LIB_DIR",
            "SYS_CLASS_NET",
            "SYS_CLASS_INFINIBAND",
            "DOCKERENV_MARKER",
            "PODMAN_CONTAINERENV_MARKER",
            "CGROUP_FILE",
            "SELF_CGROUP_FILE",
            "INIT_MNT_NS",
            "SELF_MNT_NS",
            "SELF_CGROUP_NS",
            "BOOT_ID_FILE",
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
    "therock",
    "hip",
    "hipblaslt",
    "rocblas",
    "composable_kernel",
    "tensile",
    "tensile_catalog",
    "miopen_catalog",
    "rocfft_catalog",
    "triton",
    "fbgemm",
    "torchrec",
    "aiter",
    "aotriton",
    "miopen",
    "rccl",
    "gpu_arch",
    "host",
    "runtime_context",
    "docker",
    "env_vars",
    "python_version",
    "pytorch_version",
    "pytorch_build",
    "build_system",
    "buck_invocation",
    "library_introspection",
    "library_introspection_alternates",
    "pytorch_sdpa",
    "nics",
    "amdgpu_driver",
    "container_detected",
    "execution_context",
    "probe_namespace",
    "library_linkage",
}


class TestSchemaCompleteness:
    def test_all_top_level_keys_present_when_everything_unavailable(
        self, all_disabled
    ):
        snapshot = collect_env()
        assert set(snapshot.to_dict().keys()) == REQUIRED_TOP_KEYS
        assert snapshot.schema_version == SCHEMA_VERSION
        assert snapshot.system_health is None
        # The three version fields are null, and the schema-1.16 attribution
        # keys say why: no ROCm install was located at all (`root_source:
        # "none"`), as opposed to one found without a version file.
        assert snapshot.rocm == {
            "version": None,
            "version_dev": None,
            "kmd_version": None,
            "version_source": None,
            "root": str(env_mod.ROCM_ROOT),
            "lib_root": str(env_mod.ROCM_LIB_ROOT),
            "root_source": "none",
            "layout": "classic",
        }
        assert snapshot.therock["status"] == "absent"
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

    def test_public_docs_match_current_schema_version(self):
        """The public env-probe docs must advertise the CURRENT SCHEMA_VERSION as
        the "current" schema -- a stale doc makes operators think live env.json
        files are wrong or that a new field shouldn't exist yet. Guards against the
        doc/code drift that shipped probe_namespace while docs still said 1.11."""
        docs_dir = Path(env_mod.__file__).resolve().parents[3] / "docs"
        env_probe = (docs_dir / "env-probe.md").read_text(encoding="utf-8")
        # The schema-version table row and the changelog "(current)" heading must
        # both name the live constant.
        assert f'Currently `"{SCHEMA_VERSION}"`' in env_probe
        assert f"### `{SCHEMA_VERSION}` (current)" in env_probe

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
    """Build a fully-populated EnvSnapshot for round-trip testing.

    The ``schema_version`` field uses the live ``SCHEMA_VERSION`` constant
    rather than a hard-coded string so this fixture (and every round-trip
    test that depends on it) keeps working whenever the schema version is
    bumped in ``environment.py``. The schema-version-change tests live
    elsewhere in this module and pin the literal version intentionally.
    """
    base = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": "2026-04-28T12:00:00Z",
        "system_health": {"rdhc_version": "1.4.0", "tests": {}},
        # Full schema-1.16 shape. It has to be complete for the round-trip
        # tests to mean anything: from_dict() merges these blocks over their
        # 1.16 null shapes (#387), so a fixture that omitted the added keys
        # would round-trip unequal and, worse, would let a genuinely short
        # emission from the probe pass unnoticed.
        "rocm": {
            "version": "7.2.1",
            "version_dev": "7.2.1-43",
            "kmd_version": "6.16.13",
            "version_source": "version_file",
            "root": "/opt/rocm",
            "lib_root": "/opt/rocm",
            "root_source": "opt_rocm",
            "layout": "classic",
        },
        "hip": {
            "version": "7.2.5",
            "platform": "amd",
            "compiler": "clang",
            "runtime": "rocclr",
            "cpp_config": "-D__HIP_PLATFORM_AMD__",
        },
        "hipblaslt": {
            "rocm_release_tweak": "dabb6df2b9",
            "package_version": "1.2.2",
            "lib_hash": "sha256:abc",
            "kernel_db_revision": "filenames-sha256:def",
            "upstream_commit": None,
            "upstream_commit_matches_tweak": None,
            "applied_prs": {},
        },
        "rocblas": {
            "rocm_release_tweak": "dabb6df2b9",
            "package_version": "5.2.0",
            "lib_hash": "sha256:bbb",
            "kernel_db_revision": "filenames-sha256:ccc",
            "upstream_commit": None,
            "upstream_commit_matches_tweak": None,
            "applied_prs": {},
        },
        "composable_kernel": {
            "system": {
                "version": "1.2.0",
                "commit": "23d531c8ae9721ac990116751542ab63e11d27c8",
                "ck_tile_present": True,
            },
            "pytorch_bundled": {"present": True, "symbol_count": 4067},
            "pytorch_use_ck_sdpa": True,
            "pytorch_use_ck_gemm": True,
        },
        "tensile": {
            "package_version": None,
            "kernel_db_combined_hash": "filenames-sha256:eee",
        },
        "triton": {
            "package_version": "3.5.1+rocm7.2.1.gita272dfa8",
            "commit": "a272dfa8",
        },
        "fbgemm": {
            "package_version": None,
            "commit": None,
            "pytorch_use_fbgemm": True,
            "pytorch_use_fbgemm_genai": True,
        },
        "torchrec": {
            "package_version": "1.4.0",
            "commit": None,
            "source_version": None,
            "source_commit": None,
            "distribution_version": "1.4.0",
        },
        "aiter": {
            "package_version": None,
            "package_dist_name": None,
            "commit": None,
            "hsa_tree": None,
        },
        "aotriton": {
            "bundled_present": True,
            "bundled_version": "0.11.1",
            "bundled_lib_hash": "sha256:abc",
            "bundled_images_dir_present": True,
            "installed_prefix": None,
        },
        "miopen": {
            "rocm_release_tweak": "dabb6df2b9",
            "package_version": "3.5.1",
            "lib_hash": "sha256:miopenhash",
            "kernel_db_revision": "filenames-sha256:miopendb",
        },
        "rccl": {
            "version_code": 22707,
            "version": "2.27.7",
            "lib_hash": "sha256:rcclhash",
            "net_plugin_mode": "external",
            "plugin_path": "/apps/build/amd-anp/build/librccl-net.so",
            "plugin_lib_hash": "sha256:pluginhash",
            "anp_lib_hash": None,
            "net_lib_hash": None,
        },
        "gpu_arch": {
            "agent_count": 8,
            "gfx_targets": ["gfx942"],
            "agent_arch_counts": {"gfx942": 8},
        },
        "host": {
            "kernel_release": "5.15.0-174-generic",
            "kernel_version": "#184-Ubuntu SMP Fri Mar 13 18:41:50 UTC 2026",
            "machine": "x86_64",
            "glibc_version": "2.35",
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
        "pytorch_build": {
            "git_commit": "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf",
            "hip_version": "7.2.53211-e1a6bc5663",
            "cuda_version": None,
            "debug": False,
            "install_kind": "wheel",
            "source_path": None,
            "submodule_commits": {
                "_source": None,
                "composable_kernel": None,
                "aiter": None,
                "fbgemm": None,
            },
            "flags": {
                "build_settings": None,
                "cxx_defines": None,
                "cxx_flags_raw": None,
                "cuda_flags_raw": None,
                "gpu_arch_list": None,
            },
            "binary_introspection": {
                "libtorch_hip_symbol_counts": {
                    m: None for m in env_mod._LIBTORCH_HIP_SYMBOL_MARKERS
                },
                "torch_lib_bundled": None,
                "cxx_flags_use_defines": None,
            },
            "build_flags": {
                name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES
            },
            "cmake_cache": {"_source_file": None, "entries": None},
            "ninja_hipcc": {"_source_file": None, "targets": None},
        },
        "build_system": {"kind": "none"},
        "buck_invocation": {
            "status": "success",
            "target": "//app:trainer",
            "context_source": "explicit",
            "mode_files": ["root//mode/debug"],
            "config_keys": ["build.profile"],
            "modifiers": ["//constraints:linux"],
            "option_order": ["mode", "config", "modifier"],
            "context_fingerprint": "sha256:" + "a" * 64,
            "configured_root_target": (
                "root//app:trainer "
                "(prelude//platforms:default#configured)"
            ),
            "comparison": "not_compared",
        },
        "partial": False,
        "partial_reasons": [],
        "library_introspection": [],
        "library_introspection_alternates": [],
        "pytorch_sdpa": {
            "backends_enabled": {
                name: None for name in env_mod._PYTORCH_SDPA_GETTERS
            }
        },
        "nics": {
            "ainic": {"present": False},
            "broadcom": {
                "present": True,
                "driver_version": "6.9.0-0_fbk10_brcmrdma13_141_g9",
                "firmware": "232.0.219.16/pkg 232.1.196.16",
                "rdma_devices": ["bnxt_re0"],
                "links": [
                    {"device": "bnxt_re0", "state": "ACTIVE", "netdev": "benic7p1"}
                ],
            },
            "cx7": {"present": True, "driver_version": None, "firmware": None,
                    "rdma_devices": [], "links": []},
        },
        "amdgpu_driver": {
            "scope": "host_kernel",
            "status": "present",
            "package_name": "amdgpu-dkms",
            "package_version": "1:6.14.14-2212064.24.04",
            "package_full_name": "amdgpu-dkms=1:6.14.14-2212064.24.04",
            "package_manager": "dpkg",
            "module_version": "6.14.14",
            "module_srcversion": "A1B2C3D4E5F6A7B8C9D0",
            "kmd_version": "6.16.13",
            "kfd_device_present": True,
            "kfd_sysfs_present": True,
        },
        "container_detected": True,
        "execution_context": {
            "probe_invocation": "buck2_action",
            "likely_execution_platform": None,
        },
        "probe_namespace": "mnt:0123456789abcdef",
        # Schema 1.17. Deliberately a MIXED reading -- ROCm objects on
        # DT_RPATH while torch's own library is on DT_RUNPATH -- because that
        # is the real ROCm 10 shape (issue #413) and a single-tag fixture
        # would let a "collapse to one verdict" regression round-trip clean.
        "library_linkage": {
            "status": "ok",
            "rocm_rpath": True,
            "tags_observed": ["rpath", "runpath"],
            # The census over the whole ROCm lib dir -- what ``rocm_rpath``
            # actually aggregates -- rather than the named sample below,
            # which is a reading aid. Counts mirror the measured gate image
            # (61 objects, all DT_RPATH), and the examples name
            # libhipblas.so.3 on purpose: it is the object #413 measured
            # defeating a substitution and it is NOT in the named sample.
            "census": {
                "dirs": ["/opt/rocm/lib"],
                "unlistable_dirs": [],
                "scanned": 61,
                "rpath": 61,
                "runpath": 0,
                "none": 0,
                "unreadable": 0,
                "truncated": False,
                "rpath_examples": ["libhipblas.so.3", "libhipblaslt.so.1"],
            },
            "libraries": [
                {
                    "name": "libhipblaslt.so",
                    "scope": "rocm",
                    "path": "/opt/rocm/lib/libhipblaslt.so.1",
                    "dt_tags": ["rpath"],
                    "effective_tag": "rpath",
                    "reason": None,
                },
                {
                    "name": "libtorch_hip.so",
                    "scope": "pytorch",
                    "path": "/site-packages/torch/lib/libtorch_hip.so",
                    "dt_tags": ["runpath"],
                    "effective_tag": "runpath",
                    "reason": None,
                },
            ],
        },
    }
    base.update(overrides)
    return EnvSnapshot(**base)


class TestEnvSnapshot:
    def test_to_dict_keys_are_complete(self):
        snap = _example_snapshot()
        d = snap.to_dict()
        assert set(d.keys()) == REQUIRED_TOP_KEYS

    def test_partial_reasons_stays_the_trailer(self):
        """The documented trailer convention, asserted rather than assumed.

        Any field missing from ``_OUTPUT_KEY_ORDER`` is appended after it, so a
        new block silently lands *past* ``partial_reasons`` -- which is exactly
        what happened to ``therock`` (#387, 8th pass). Pinning the tail catches
        the next one without needing a rule per block.
        """
        keys = list(_example_snapshot().to_dict())
        assert keys[-2:] == ["partial", "partial_reasons"]

    def test_therock_is_emitted_with_the_rocm_block(self):
        """It is that install's build provenance; it belongs next to `rocm`."""
        keys = list(_example_snapshot().to_dict())
        assert keys[keys.index("rocm") + 1] == "therock"

    def test_every_field_is_ordered_explicitly_or_appended_once(self):
        """No duplicates and no drops, whichever path a field takes."""
        keys = list(_example_snapshot().to_dict())
        assert len(keys) == len(set(keys))
        assert set(keys) == REQUIRED_TOP_KEYS

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
        assert rebuilt.schema_version == SCHEMA_VERSION

    def test_from_dict_defaults_partial_reasons_when_missing(self):
        """Older env.json without partial_reasons still loads (defaults to [])."""
        d = _example_snapshot().to_dict()
        del d["partial_reasons"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.partial_reasons == []

    @pytest.mark.parametrize(
        "key,empty_factory",
        [
            ("tensile_catalog", env_mod._empty_tensile_catalog),
            ("miopen_catalog", env_mod._empty_miopen_catalog),
            ("rocfft_catalog", env_mod._empty_rocfft_catalog),
            ("amdgpu_driver", env_mod._empty_amdgpu_driver),
            ("execution_context", env_mod._empty_execution_context),
            ("buck_invocation", env_mod._empty_buck_invocation),
        ],
    )
    def test_from_dict_backfills_missing_catalog_key(self, key, empty_factory):
        """A pre-1.9 env.json predates the three catalog blocks entirely.

        ``docs/env-probe.md`` promises that ``from_dict()`` back-fills the
        "not captured" empty shape for each; this is the regression guard
        for that promise (nothing previously exercised the deletion case,
        only the "key present with real data" round-trip).
        """
        d = _example_snapshot().to_dict()
        del d[key]
        rebuilt = EnvSnapshot.from_dict(d)
        assert getattr(rebuilt, key) == empty_factory()

    def test_from_dict_backfills_missing_container_detected(self):
        # A pre-1.11 env.json predates container_detected (a bare bool, so
        # it isn't in the factory-parametrized test above). from_dict must
        # default it to False rather than raising.
        d = _example_snapshot().to_dict()
        del d["container_detected"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.container_detected is False

    def test_from_dict_backfills_missing_probe_namespace(self):
        # A pre-1.12 env.json predates probe_namespace. from_dict must
        # default it to None rather than raising.
        d = _example_snapshot().to_dict()
        del d["probe_namespace"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.probe_namespace is None

    # -- schema 1.16 added keys INSIDE existing blocks (#387) ----------------
    #
    # Every back-fill test above deletes a whole top-level key. 1.16 is the
    # first bump that grew existing nested blocks, so a 1.15 artifact loads
    # with the key present but SHORT -- which none of those tests would catch.

    #: A schema-1.15 `rocm` / `hipblaslt` / `rocblas` block, verbatim: the keys
    #: a 1.15 producer wrote, and nothing 1.16 added.
    _V115_ROCM = {"version": "7.2.4", "version_dev": None, "kmd_version": "6.16.13"}
    _V115_GEMM = {
        "rocm_release_tweak": "5b515cf1bc",
        "package_version": "1.2.2",
        "lib_hash": "sha256:abc",
        "kernel_db_revision": "filenames-sha256:def",
        "applied_prs": {},
    }

    def _v115_dict(self):
        d = _example_snapshot().to_dict()
        d["schema_version"] = "1.15"
        d["rocm"] = dict(self._V115_ROCM)
        d["hipblaslt"] = dict(self._V115_GEMM)
        d["rocblas"] = dict(self._V115_GEMM)
        d.pop("therock", None)
        return d

    def test_from_dict_backfills_the_1_16_rocm_keys(self):
        """A 1.15 snapshot must gain the documented 1.16 keys, not a short dict."""
        rebuilt = EnvSnapshot.from_dict(self._v115_dict())
        assert set(rebuilt.rocm) == set(env_mod._empty_rocm())
        # Real 1.15 data survives the merge...
        assert rebuilt.rocm["version"] == "7.2.4"
        assert rebuilt.rocm["kmd_version"] == "6.16.13"
        # ...and the added keys read null rather than claiming THIS host's
        # roots, which the older producer never recorded.
        for key in ("root", "lib_root", "root_source", "layout", "version_source"):
            assert rebuilt.rocm[key] is None, key

    def test_from_dict_backfills_the_1_16_gemm_keys(self):
        rebuilt = EnvSnapshot.from_dict(self._v115_dict())
        for block in (rebuilt.hipblaslt, rebuilt.rocblas):
            assert set(block) == set(env_mod._empty_gemm_library())
            assert block["rocm_release_tweak"] == "5b515cf1bc"
            assert block["upstream_commit"] is None
            assert block["upstream_commit_matches_tweak"] is None

    def test_a_1_15_snapshot_exposes_the_whole_documented_1_16_shape(self):
        """The reviewer's actual concern: no KeyError for a 1.16 consumer.

        Indexes every key the 1.16 docs promise, the way a jq pipeline or the
        dashboard would, instead of asserting key sets abstractly.
        """
        s = EnvSnapshot.from_dict(self._v115_dict())
        for key in ("root", "lib_root", "root_source", "layout", "version_source"):
            s.rocm[key]
        for block in (s.hipblaslt, s.rocblas):
            block["upstream_commit"]
            block["upstream_commit_matches_tweak"]
        s.therock["status"]
        s.therock["gemm_libraries_commit"]

    def test_a_pre_1_16_snapshot_gets_unknown_therock_not_absent(self):
        """`absent` is a positive claim the old producer never made (#387).

        It means "this install ships no manifest". A pre-1.16 snapshot may well
        have been a wheel layout with a perfectly good manifest nobody looked
        for, so asserting `absent` -- and stamping the READING host's
        manifest_path onto someone else's capture -- invents both facts. Same
        reasoning as `_null_rocm` for the attribution keys.
        """
        d = _example_snapshot().to_dict()
        d["schema_version"] = "1.15"
        del d["therock"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.therock["status"] == "unknown"
        assert rebuilt.therock["manifest_path"] is None
        # Still the full shape, so a 1.16 consumer can index it.
        assert set(rebuilt.therock) == set(env_mod._empty_therock())

    def test_a_locally_built_snapshot_still_says_absent(self, all_disabled):
        """The local default is unchanged: here we DID look and found none."""
        assert collect_env().therock["status"] == "absent"

    def test_a_short_therock_block_is_backfilled_too(self):
        d = _example_snapshot().to_dict()
        d["therock"] = {"status": "present", "rocm_version": "7.14.0"}
        rebuilt = EnvSnapshot.from_dict(d)
        assert set(rebuilt.therock) == set(env_mod._empty_therock())
        assert rebuilt.therock["rocm_version"] == "7.14.0"
        assert rebuilt.therock["submodules"] == []

    @pytest.mark.parametrize("key", ["rocm", "hipblaslt", "rocblas"])
    def test_a_genuinely_missing_required_block_still_raises(self, key):
        """The back-fill must not silence a malformed dict.

        ``from_dict``'s docstring is explicit that the schema-1.0/1.1 required
        set is NOT defaulted, because absence there means a broken artifact
        rather than an older producer. Merging only when the key is present
        keeps that true.
        """
        d = _example_snapshot().to_dict()
        del d[key]
        with pytest.raises(TypeError):
            EnvSnapshot.from_dict(d)

    def test_backfill_shapes_match_what_the_probe_emits(self):
        """The null and locally-attributed rocm shapes must not drift apart."""
        assert set(env_mod._null_rocm()) == set(env_mod._empty_rocm())
        assert all(v is None for v in env_mod._null_rocm().values())

    def test_summary_does_not_duplicate_partial_marker(self):
        """The brief returned by ``summary()`` is the *body* of what the
        CLI prints. The CLI itself frames the brief with a header line
        (``Wrote env probe to ... [PARTIAL]``) and a closing line
        (``[PARTIAL, N reason(s)]``), so a third copy of "PARTIAL"
        embedded in the summary body would be redundant. Asserts the
        body stays clean of PARTIAL markers regardless of state.
        """
        partial_snap = _example_snapshot(partial=True, partial_reasons=["x: y"])
        clean_snap = _example_snapshot()
        # Neither should leak "PARTIAL" into the body. The marker is
        # the CLI's job, not summary()'s. (We do still want the field
        # values themselves to differ -- the partial vs clean snapshot
        # produce different `partial_reasons` lengths visible to the
        # caller via .partial_reasons, which is what the CLI prints.)
        assert "PARTIAL" not in partial_snap.summary()
        assert "PARTIAL" not in clean_snap.summary()

    def test_summary_treats_empty_system_health_as_present(self):
        """Regression guard: RDHC may legitimately return an empty dict
        ``{}`` (subprocess succeeded, nothing to report). The earlier
        truthiness check ``if self.system_health`` would summarise that as
        unavailable -- ``is not None`` is the right check.

        Asserts on the **rdhc:** line specifically rather than the
        whole brief, because other lines (e.g. ``aotriton``) use
        "present" as a field-name key (``bundled_present=True``) which
        would false-positive a substring search across the full text.
        """
        snap_empty = _example_snapshot(system_health={})
        snap_null = _example_snapshot(system_health=None)
        snap_populated = _example_snapshot(system_health={"rdhc_version": "1.4.0"})

        def rdhc_line(snap) -> str:
            for line in snap.summary().splitlines():
                if line.lstrip().startswith("rdhc:"):
                    return line
            raise AssertionError("no rdhc: line in summary")

        # Empty dict and populated dict should both render as 'present'
        # in the rdhc line
        assert "present" in rdhc_line(snap_empty)
        assert "unavailable" not in rdhc_line(snap_empty)
        assert "present" in rdhc_line(snap_populated)
        # Only None should render as 'unavailable'
        assert "unavailable" in rdhc_line(snap_null)
        assert "present" not in rdhc_line(snap_null)

    def test_summary_is_multiline_human_readable(self):
        snap = _example_snapshot()
        s = snap.summary()
        # 6 lines per the implementation; loose lower-bound
        assert s.count("\n") >= 4
        assert "rocm:" in s
        assert "hipblaslt:" in s

    def test_summary_surfaces_probe_namespace_observation(self):
        snap = _example_snapshot(probe_namespace="mnt:0123456789abcdef")
        runtime_line = next(
            line for line in snap.summary().splitlines()
            if line.lstrip().startswith("runtime:")
        )
        assert "ns=mnt:012345…" in runtime_line

    def test_summary_surfaces_buck_invocation_status_source_and_fingerprint(self):
        snap = _example_snapshot()
        buck_line = next(
            line
            for line in snap.summary().splitlines()
            if line.lstrip().startswith("buck ctx:")
        )
        assert "status=success" in buck_line
        assert "source=explicit" in buck_line
        assert "fingerprint=sha256:aaaaaa…" in buck_line

    def test_dataclass_is_frozen(self):
        """Callers can safely embed the snapshot without mutation hazards."""
        snap = _example_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.schema_version = "2.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# collect_env contract: never raises + partial semantics + idempotency
# ---------------------------------------------------------------------------


class TestProbeStdioRedirect:
    """fd-level capture of benign HIP/C-runtime probe noise (#220).

    On a multi-GPU ROCm host the in-process ``import torch`` during the env
    probe makes the HIP runtime ``dlopen`` write one
    ``(null): No such file or directory`` line per GPU straight to fd 2.
    ``_ProbeStdioRedirect`` must intercept those raw ``write(2, ...)`` syscalls
    (which ``contextlib.redirect_stderr`` cannot) so they never reach the
    operator's terminal, and re-emit them at DEBUG for post-hoc debugging.
    """

    def _with_outer_terminal(self, tmp_path: Path):
        """Install a temp file on real fds 1/2; return (path, restore())."""
        outer = tmp_path / "outer_terminal.txt"
        sys.stdout.flush()
        sys.stderr.flush()
        outer_fd = os.open(
            str(outer), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )
        saved1, saved2 = os.dup(1), os.dup(2)
        os.dup2(outer_fd, 1)
        os.dup2(outer_fd, 2)

        def restore():
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved1, 1)
            os.dup2(saved2, 2)
            os.close(saved1)
            os.close(saved2)
            os.close(outer_fd)

        return outer, restore

    def test_raw_fd2_writes_are_captured_not_leaked(self, tmp_path, caplog):
        outer, restore = self._with_outer_terminal(tmp_path)
        try:
            redirect = env_mod._ProbeStdioRedirect()
            redirect.start()
            for _ in range(8):  # one (null) line per GPU on an 8-GPU host
                os.write(2, b"(null): No such file or directory\n")
            os.write(1, b"some toolchain chatter\n")
            with caplog.at_level("DEBUG", logger=env_mod.log.name):
                redirect.stop()
        finally:
            restore()

        assert outer.read_text() == "", "probe noise leaked to the terminal"
        assert any(
            "(null): No such file or directory" in rec.getMessage()
            for rec in caplog.records
        ), "captured noise should be re-emitted at DEBUG"

    @pytest.mark.parametrize(
        "logger_name, propagate",
        [
            # utils.setup_logging: stderr handler on the root logger.
            ("", True),
            # CLI configure_verbose_logging (-v/-vv): handler on the "aorta"
            # logger with propagate=False -- a root-only reroute would miss it.
            ("aorta", False),
        ],
    )
    def test_aorta_logs_reach_terminal_during_redirect(
        self, tmp_path, logger_name, propagate
    ):
        """Aorta's own logs must survive the fd redirect (#221 review).

        The redirect captures *all* of fd 1/2 for the probe body, which would
        otherwise swallow aorta's own ``log.info`` (vanishing at -v, mislabeled
        as benign HIP noise at -vv). ``_reroute_loggers`` repoints the stderr
        StreamHandler at the real terminal for the window, so a real diagnostic
        still reaches the operator while the C-level ``(null)`` noise does not.
        Covers both wiring styles: the root logger (``setup_logging``) and the
        "aorta" logger with ``propagate=False`` (the CLI verbosity path).
        """
        target = logging.getLogger(logger_name)
        saved_handlers = target.handlers[:]
        saved_level = target.level
        saved_propagate = target.propagate
        stream_handler = logging.StreamHandler(sys.stderr)
        outer, restore = self._with_outer_terminal(tmp_path)
        try:
            for handler in saved_handlers:
                target.removeHandler(handler)
            stream_handler.setLevel(logging.INFO)
            target.addHandler(stream_handler)
            target.setLevel(logging.INFO)
            target.propagate = propagate

            redirect = env_mod._ProbeStdioRedirect()
            redirect.start()
            env_mod.log.info("real diagnostic from a probe")
            os.write(2, b"(null): No such file or directory\n")
            redirect.stop()
        finally:
            target.removeHandler(stream_handler)
            for handler in saved_handlers:
                target.addHandler(handler)
            target.setLevel(saved_level)
            target.propagate = saved_propagate
            restore()

        terminal_text = outer.read_text()
        assert "real diagnostic from a probe" in terminal_text, (
            "aorta's own INFO log was swallowed by the probe fd redirect"
        )
        assert "(null): No such file or directory" not in terminal_text, (
            "benign HIP noise leaked to the terminal"
        )

    def test_stop_is_idempotent(self, tmp_path):
        outer, restore = self._with_outer_terminal(tmp_path)
        try:
            redirect = env_mod._ProbeStdioRedirect()
            redirect.start()
            os.write(2, b"noise\n")
            redirect.stop()
            redirect.stop()  # must be a no-op, not crash or re-restore
        finally:
            restore()
        assert outer.read_text() == ""

    def test_start_partial_dup_failure_does_not_leak_fd(self, monkeypatch):
        """A partial ``dup()`` must not orphan the first descriptor (#221).

        If ``os.dup(1)`` succeeds but ``os.dup(2)`` raises (fd 2 already
        closed/detached), ``start()`` falls back to probing unredirected -- but
        the fd from the successful first ``dup()`` must be closed, not leaked.
        """
        redirect = env_mod._ProbeStdioRedirect()
        real_dup = os.dup
        duped: list[int] = []

        def flaky_dup(fd):
            if not duped:
                new = real_dup(fd)
                duped.append(new)
                return new
            raise OSError("simulated: fd 2 closed/detached")

        monkeypatch.setattr(env_mod.os, "dup", flaky_dup)
        redirect.start()

        assert redirect._saved_out is None
        assert redirect._saved_err is None
        assert duped, "test should have exercised the first dup()"
        with pytest.raises(OSError):
            os.fstat(duped[0])  # orphan fd was closed, not leaked
        redirect.stop()  # unredirected instance -> no-op, must not crash

    def test_start_dup2_failure_closes_capture_file(self, monkeypatch):
        """A ``dup2()`` failure after the temp file is created must close it (#221).

        ``start()`` dups fds 1/2 successfully, then ``dup2`` raises. The
        fallback restores the real fds (clearing ``_saved_out/_saved_err``),
        which makes a later ``stop()`` short-circuit -- so ``start()`` itself
        must close the capture file or it leaks the handle.
        """
        redirect = env_mod._ProbeStdioRedirect()
        real_dup2 = os.dup2
        calls: list[int] = []

        def flaky_dup2(*args, **kwargs):
            # Fail the first capture redirect; let _restore_fds() succeed.
            if not calls:
                calls.append(1)
                raise OSError("simulated: dup2 failed")
            return real_dup2(*args, **kwargs)

        monkeypatch.setattr(env_mod.os, "dup2", flaky_dup2)
        redirect.start()

        assert redirect._saved_out is None
        assert redirect._saved_err is None
        assert redirect._capture is None, "capture temp file was leaked"
        redirect.stop()  # no-op, must not crash

    def test_start_fail_soft_on_non_oserror_flush(self, monkeypatch):
        """A closed stream's flush() raises ValueError; start() must fail soft.

        ``start()`` runs *before* ``collect_env()``'s try/except, so any
        exception escaping it (here a ``ValueError`` from ``sys.stdout.flush``)
        would break the never-raises contract (#221).
        """
        redirect = env_mod._ProbeStdioRedirect()

        class _ClosedStream:
            def flush(self):
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(env_mod.sys, "stdout", _ClosedStream())
        redirect.start()  # must not raise
        redirect.stop()
        # Setup aborted -> nothing left dangling.
        assert redirect._saved_out is None
        assert redirect._saved_err is None
        assert redirect._capture is None

    def test_stop_never_raises_when_streams_closed(self, tmp_path, monkeypatch):
        """stop() runs in collect_env()'s finally and must never raise (#221)."""
        outer, restore = self._with_outer_terminal(tmp_path)
        try:
            redirect = env_mod._ProbeStdioRedirect()
            redirect.start()
            os.write(2, b"noise\n")

            class _ClosedStream:
                def flush(self):
                    raise ValueError("closed")

            monkeypatch.setattr(env_mod.sys, "stdout", _ClosedStream())
            try:
                redirect.stop()  # must not raise despite flush ValueError
            finally:
                # Undo the closed-stdout patch before the outer-terminal
                # teardown, which flushes sys.stdout itself.
                monkeypatch.undo()
        finally:
            restore()
        assert redirect._capture is None

    def test_restore_fds_swallows_dup2_failure(self, tmp_path, monkeypatch):
        """_restore_fds() must be best-effort: a dup2 EBADF must not escape."""
        outer, restore = self._with_outer_terminal(tmp_path)
        try:
            redirect = env_mod._ProbeStdioRedirect()
            redirect.start()
            real_dup2 = os.dup2

            def boom_dup2(*_a, **_k):
                raise OSError("simulated: EBADF on restore")

            monkeypatch.setattr(env_mod.os, "dup2", boom_dup2)
            try:
                redirect.stop()  # must not raise even though restore dup2 fails
            finally:
                # Restore the real dup2 before the outer terminal teardown,
                # which itself dup2()s fds 1/2 back to the test runner.
                monkeypatch.setattr(env_mod.os, "dup2", real_dup2)
        finally:
            restore()
        assert redirect._saved_out is None
        assert redirect._saved_err is None

    def test_collect_env_does_not_leak_probe_noise(
        self, all_disabled, tmp_path, monkeypatch
    ):
        """End-to-end: fd-2 noise from a probe never reaches the terminal."""
        original = env_mod._detect_runtime_context

        def noisy_detect():
            for _ in range(8):
                os.write(2, b"(null): No such file or directory\n")
            return original()

        monkeypatch.setattr(env_mod, "_detect_runtime_context", noisy_detect)

        outer, restore = self._with_outer_terminal(tmp_path)
        try:
            snapshot = collect_env()
        finally:
            restore()

        assert isinstance(snapshot, EnvSnapshot)
        assert outer.read_text() == "", "env probe leaked HIP noise to terminal"
        # The benign noise is NOT a probe failure -> must not inflate partial.
        assert not any(
            "(null)" in reason for reason in snapshot.partial_reasons
        )


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
        (lib_dir / "librocblas.so").write_bytes(b"rocblas-binary")

        tensile_dir = tmp_path / "tensile"
        tensile_dir.mkdir()
        (tensile_dir / "TensileLibrary_X.dat").write_bytes(b"x")

        # rocblas inputs (header in its own internal/ subdir to mirror prod layout)
        rocblas_header_dir = tmp_path / "include" / "rocblas" / "internal"
        rocblas_header_dir.mkdir(parents=True)
        (rocblas_header_dir / "rocblas-version.h").write_text(
            "#define ROCBLAS_VERSION_MAJOR 5\n"
            "#define ROCBLAS_VERSION_MINOR 2\n"
            "#define ROCBLAS_VERSION_PATCH 0\n"
            "#define ROCBLAS_VERSION_TWEAK dabb6df2b9\n"
        )
        rocblas_tensile_dir = tmp_path / "rocblas_tensile"
        rocblas_tensile_dir.mkdir()
        (rocblas_tensile_dir / "Kernels.so-000-gfx942.hsaco").write_bytes(b"k")
        (rocblas_tensile_dir / "TensileLibrary_gfx942.dat").write_bytes(b"t")

        # CK inputs (header-only)
        ck_header_dir = tmp_path / "include" / "ck"
        ck_header_dir.mkdir(parents=True)
        (ck_header_dir / "version.h").write_text(
            "#define CK_VERSION 1.2.0\n"
            "#define CK_VERSION_MAJOR 1\n"
            "#define CK_VERSION_MINOR 2\n"
            "#define CK_VERSION_PATCH 0\n"
            "#define CK_COMMIT_ID 23d531c8ae9721ac990116751542ab63e11d27c8\n"
        )
        ck_tile_dir = tmp_path / "include" / "ck_tile" / "core"
        ck_tile_dir.mkdir(parents=True)
        (ck_tile_dir / "config.hpp").write_text("// ck_tile config\n")

        # MIOpen inputs
        miopen_header_dir = tmp_path / "include" / "miopen"
        miopen_header_dir.mkdir(parents=True)
        (miopen_header_dir / "version.h").write_text(
            "#define MIOPEN_VERSION_MAJOR 3\n"
            "#define MIOPEN_VERSION_MINOR 5\n"
            "#define MIOPEN_VERSION_PATCH 1\n"
            "#define MIOPEN_VERSION_TWEAK dabb6df2b9\n"
        )
        (lib_dir / "libMIOpen.so").write_bytes(b"miopen-binary")
        miopen_db_dir = tmp_path / "miopen_db"
        miopen_db_dir.mkdir()
        (miopen_db_dir / "gfx942_64.db.txt").write_text("kernel db")

        # RCCL inputs
        rccl_header_dir = tmp_path / "include" / "rccl"
        rccl_header_dir.mkdir(parents=True)
        (rccl_header_dir / "rccl.h").write_text(
            "#define NCCL_VERSION_CODE 22707\n"
        )
        (lib_dir / "librccl.so").write_bytes(b"rccl-binary")

        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", rocm_info / "version")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", rocm_info / "version_dev")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", kmd)
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_VERSION_HEADER", header_dir / "hipblaslt-version.h"
        )
        monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", lib_dir)
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tensile_dir)
        monkeypatch.setattr(
            env_mod,
            "ROCBLAS_VERSION_HEADER",
            rocblas_header_dir / "rocblas-version.h",
        )
        monkeypatch.setattr(env_mod, "ROCBLAS_LIB_DIR", lib_dir)
        monkeypatch.setattr(env_mod, "ROCBLAS_TENSILE_DIR", rocblas_tensile_dir)
        monkeypatch.setattr(env_mod, "CK_VERSION_HEADER", ck_header_dir / "version.h")
        monkeypatch.setattr(env_mod, "CK_TILE_CONFIG_HEADER", ck_tile_dir / "config.hpp")
        monkeypatch.setattr(
            env_mod, "MIOPEN_VERSION_HEADER", miopen_header_dir / "version.h"
        )
        monkeypatch.setattr(env_mod, "MIOPEN_LIB_DIR", lib_dir)
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", miopen_db_dir)
        monkeypatch.setattr(
            env_mod, "RCCL_VERSION_HEADER", rccl_header_dir / "rccl.h"
        )
        monkeypatch.setattr(env_mod, "RCCL_LIB_DIR", lib_dir)
        monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
        monkeypatch.setattr(
            env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv"
        )
        monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")
        # Redirect KFD paths off the real filesystem: on a host that
        # actually has an AMD GPU, /dev/kfd exists and the fake dpkg/rpm/
        # modinfo below never resolve a package -- without this, the
        # amdgpu_driver conflict check would fire a real partial reason
        # and break the "clean probe -> partial is False" contract.
        monkeypatch.setattr(env_mod, "KFD_DEVICE_NODE", tmp_path / "no_kfd")
        monkeypatch.setattr(env_mod, "KFD_SYSFS_DIR", tmp_path / "no_kfd_sysfs")

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
        # Pretend torch is importable with a version. Also stand up the
        # surfaces the new fbgemm / composable_kernel probes peek at:
        #   * torch.__config__.show() returns a build string mentioning
        #     -DUSE_FBGEMM and -DUSE_FBGEMM_GENAI (so fbgemm flags = True)
        #   * torch.__file__ + a tmp libtorch_hip.so so the CK-bundled
        #     probe finds something to run nm/c++filt against. We
        #     redirect both subprocesses through ``fake_run`` so it
        #     reports a tiny stdout containing one ck:: symbol.
        import builtins
        import types

        fake_torch_dir = tmp_path / "fake_torch"
        (fake_torch_dir / "lib").mkdir(parents=True)
        (fake_torch_dir / "lib" / "libtorch_hip.so").write_bytes(b"fake")
        # Stand up a fake bundled AOTriton (the new aotriton probe
        # would otherwise add a partial reason for "no libaotriton_v2.so*").
        (fake_torch_dir / "lib" / "libaotriton_v2.so.0.11.1").write_bytes(b"aot")
        (fake_torch_dir / "lib" / "aotriton.images").mkdir()
        fake_torch_init = fake_torch_dir / "__init__.py"
        fake_torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __version__="2.12.0",
            __file__=str(fake_torch_init),
            __config__=types.SimpleNamespace(
                show=lambda: "CXX_FLAGS=-DUSE_FBGEMM -DUSE_FBGEMM_GENAI"
            ),
            # torch.version.* surface required by _capture_pytorch_build.
            # On a clean full probe we want no partial reasons, so set
            # all four fields. install_kind will be "source" because we
            # also create a fake third_party tree below.
            version=types.SimpleNamespace(
                git_version="ff65f5bc672795c5e5033900ea0a0c4f8566c8cf",
                hip="7.2.5",
                cuda=None,
                debug=False,
            ),
            # backends.cuda surface required by _capture_pytorch_sdpa
            # (issue #176). Without this, the SDPA probe reports
            # "torch.backends.cuda unavailable" and the clean-probe
            # contract (no partial reasons) fails.
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    flash_sdp_enabled=lambda: True,
                    mem_efficient_sdp_enabled=lambda: True,
                    math_sdp_enabled=lambda: True,
                    cudnn_sdp_enabled=lambda: False,
                ),
            ),
        )

        # Stand up a fake source tree with .git + third_party so
        # _detect_pytorch_install_kind walks up from torch.__file__ and
        # finds it. AORTA_PYTORCH_SRC is the explicit path; that's the
        # cleaner option for tests.
        fake_src = tmp_path / "fake_pytorch_src"
        (fake_src / "third_party").mkdir(parents=True)
        for name in ("composable_kernel", "aiter", "fbgemm"):
            sub = fake_src / "third_party" / name
            sub.mkdir()
            (sub / ".git").write_text("gitdir: ../../.git/modules/" + name)
        monkeypatch.setenv("AORTA_PYTORCH_SRC", str(fake_src))

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Have shutil.which find nm/c++filt so the bundled-CK probe runs.
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/" + name)

        # Wrap fake_run so nm + c++filt produce a synthetic ck:: hit.
        original_fake_run = fake_run

        def fake_run_with_nm(cmd, **kwargs):
            if cmd[0].endswith("nm"):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout="0000 T mangled_symbol\n", stderr="",
                )
            if cmd[0].endswith("c++filt"):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout="ck::tensor_operation::SomeKernel\n", stderr="",
                )
            if cmd[0].endswith("rocm_agent_enumerator"):
                # Mimic the real binary's stdout: one gfx target per
                # GPU. The clean-probe fixture asserts no partial
                # reasons fire, so we need a non-empty result here.
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout="gfx942\ngfx942\n", stderr="",
                )
            if cmd[0].endswith("git") and "rev-parse" in cmd:
                # Synthesize a deterministic 40-char hex SHA per submodule
                # path so _git_rev_parse_head's hex-validity check passes.
                sub_name = Path(cmd[2]).name
                fake_sha = (sub_name + "0" * 40)[:40].lower().replace("_", "0")
                fake_sha = "".join(c if c in "0123456789abcdef" else "0" for c in fake_sha)
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=fake_sha + "\n", stderr="",
                )
            return original_fake_run(cmd, **kwargs)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run_with_nm)

        snapshot = collect_env()
        assert snapshot.partial is False, (
            f"clean probe should not be partial; reasons: {snapshot.partial_reasons}"
        )
        assert snapshot.partial_reasons == []
        # Verify the success values landed
        assert snapshot.rocm["version"] == "7.2.1"
        assert snapshot.hipblaslt["rocm_release_tweak"] == "abc1234"
        assert snapshot.rocblas["rocm_release_tweak"] == "dabb6df2b9"
        assert snapshot.composable_kernel["system"]["version"] == "1.2.0"
        assert snapshot.composable_kernel["pytorch_bundled"]["present"] is True
        assert snapshot.fbgemm["pytorch_use_fbgemm"] is True
        assert snapshot.fbgemm["pytorch_use_fbgemm_genai"] is True
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

        def boom(reasons: list[str], gemm_libraries_commit: str | None = None) -> dict:
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

    def test_disaster_snapshot_preserves_namespace_captured_before_later_failure(
        self, all_disabled, monkeypatch
    ):
        observed = "mnt:0123456789abcdef"
        monkeypatch.setattr(
            env_mod, "_capture_probe_namespace", lambda reasons: observed
        )

        def boom(reasons):
            raise RuntimeError("failure after namespace capture")

        monkeypatch.setattr(env_mod, "_run_rdhc", boom)
        snapshot = collect_env()
        assert snapshot.partial is True
        assert snapshot.probe_namespace == observed

    def test_disaster_snapshot_probes_namespace_if_happy_path_never_reached(
        self, monkeypatch
    ):
        observed = "cgroup-ns:0123456789abcdef"
        monkeypatch.setattr(
            env_mod,
            "_capture_probe_namespace_safe",
            lambda reasons: observed,
        )
        snapshot = env_mod._disaster_snapshot(
            preceding_reasons=[],
            unexpected_reason="collect_env: early failure",
        )
        assert snapshot.probe_namespace == observed

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
        assert snap.buck_invocation == env_mod._empty_buck_invocation()

    def test_disaster_snapshot_preserves_redacted_buck_request(self):
        hidden_value = "not-persisted-value"
        context = env_mod.BuckInvocationContext(
            config_overrides=(f"build.profile={hidden_value}",),
        )
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[],
            unexpected_reason="collect_env: boom",
            buck_target="//app:trainer",
            buck_context=context,
        )

        assert snap.buck_invocation["status"] == "failure"
        assert snap.buck_invocation["target"] == "//app:trainer"
        assert snap.buck_invocation["context_source"] == "explicit"
        assert snap.buck_invocation["config_keys"] == ["build.profile"]
        assert hidden_value not in json.dumps(snap.buck_invocation)

    def test_disaster_snapshot_nics_block_is_fully_shaped(self):
        """The disaster path shapes nics like every other block (all
        vendor keys present, undeterminable presence) -- not an empty
        dict -- so downstream diffs/parsers stay predictable."""
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="collect_env: boom"
        )
        assert set(snap.nics.keys()) == {"ainic", "broadcom", "cx7"}
        assert all(snap.nics[v] == {"present": None} for v in snap.nics)

    def test_disaster_snapshot_nested_blocks_match_the_happy_path_shape(
        self, all_disabled
    ):
        """Guard the NESTED shapes, not just the top-level field list.

        ``test_disaster_snapshot_populates_every_envsnapshot_field`` catches a new
        top-level field, but a new key inside an existing block slips past it: the
        disaster path built ``rocm`` / ``hipblaslt`` / ``rocblas`` from inline
        literals, so schema 1.16's ``root``/``layout``/``upstream_commit`` keys were
        added to the happy path only. A crash artifact then advertised
        ``schema_version: "1.16"`` while omitting keys a 1.16 consumer indexes,
        which is the exact opposite of what the disaster path is for.

        Comparing against the all-unavailable happy-path snapshot ties the two
        together for every future bump instead of re-listing keys here.
        """
        happy = collect_env()
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="collect_env: boom"
        )
        for block in ("rocm", "hipblaslt", "rocblas", "hip", "therock"):
            assert set(getattr(snap, block)) == set(getattr(happy, block)), block

    def test_disaster_snapshot_keeps_rocm_root_attribution(self):
        """A crash must not lose WHERE the probe looked.

        Root resolution happens at import and cannot fail, so the attribution
        keys are answerable even when the version reads never ran -- and they are
        what makes the resulting nulls interpretable.
        """
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="collect_env: boom"
        )
        assert snap.rocm["root"] == str(env_mod.ROCM_ROOT)
        assert snap.rocm["lib_root"] == str(env_mod.ROCM_LIB_ROOT)
        assert snap.rocm["root_source"] == env_mod.ROCM_ROOT_SOURCE
        assert snap.rocm["layout"] == env_mod.ROCM_LAYOUT
        assert snap.rocm["version"] is None
        assert snap.rocm["version_source"] is None

    def test_disaster_snapshot_gemm_blocks_carry_the_1_16_fields(self):
        """The hipBLASLt commit is the evidence the NaN escalations rest on."""
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="collect_env: boom"
        )
        for block in (snap.hipblaslt, snap.rocblas):
            assert block["upstream_commit"] is None
            assert block["upstream_commit_matches_tweak"] is None
            assert block["applied_prs"] == {}

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
        # The original #147 spec target was ~30 lines of substantive code
        # for the single ``probe`` subcommand. Two later additions both
        # grew the budget legitimately (none of them probing):
        #
        # * A1.2c added a second subcommand (``recipe``) with its own
        #   click decorators, --format dispatch block, and error-
        #   handling envelope.
        # * PR #177 added --summary / --field output modes plus the
        #   ``_lookup_field`` helper (dotted-path resolution with
        #   friendly errors that list available keys).
        # * The compact/--extended catalog-detail work added one more
        #   click option block + its detail= wiring into collect_env().
        # * The --execution-context work (schema 1.11) added one more click
        #   option block plus a stderr validation-warning envelope (still
        #   pure wiring/validation -- no probing).
        # * The Buck invocation-context work (schema 1.13) added four Click
        #   options and their mutual-requirement validation. The typed argv
        #   construction and fingerprinting remain in the library module.
        #
        # The real "no-probing-in-CLI" guard is
        # `test_cli_does_no_probing_imports` below -- this one is a
        # soft canary against the file ballooning beyond pure wiring.
        line_count = sum(1 for _ in cli_path.read_text().splitlines())
        assert line_count <= 500, (
            f"cli/env.py is {line_count} lines; soft budget is 500. "
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

    def test_cli_does_not_eager_import_environment(self, cli_path: Path):
        """The CLI must NOT import the heavy environment probing module at
        module top level -- that would pull subprocess/hashlib/platform into
        every ``aorta env --help`` / startup and defeat the thin-wrapper
        goal. ``environment`` may only be imported lazily inside handlers.

        Parse the AST and check only module-level ``import`` nodes, so a
        mere mention of the module in a docstring or comment does not trip.
        """
        import ast

        tree = ast.parse(cli_path.read_text())
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.ImportFrom) and node.module and (
                "aorta.instrumentation.environment" in node.module
            ):
                raise AssertionError(
                    f"cli/env.py imports environment at module top "
                    f"(from {node.module}); import it lazily inside the handler."
                )
            if isinstance(node, ast.Import) and any(
                "aorta.instrumentation.environment" in alias.name
                for alias in node.names
            ):
                raise AssertionError(
                    "cli/env.py imports environment at module top; "
                    "import it lazily inside the handler."
                )

    def test_cli_execution_context_choices_match_canonical(self, cli_path: Path):
        """The hard-coded --execution-context choices in the CLI must stay in
        sync with the canonical EXECUTION_CONTEXT_INVOCATIONS. The CLI
        hard-codes (rather than imports) the list to avoid the eager
        environment import guarded above, so this is the drift guard.
        """
        spec = importlib.util.spec_from_file_location("aorta.cli.env", cli_path)
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)
        assert (
            cli_mod._EXECUTION_CONTEXT_CHOICES
            == list(env_mod.EXECUTION_CONTEXT_INVOCATIONS)
        ), (
            "cli/env.py _EXECUTION_CONTEXT_CHOICES drifted from "
            "EXECUTION_CONTEXT_INVOCATIONS; keep them identical."
        )

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

    def test_cli_echoes_partial_reasons_inline(
        self, all_disabled, tmp_path: Path
    ):
        """Operator running the probe should see WHY it's partial without
        having to ``jq env.json``. partial_reasons is already in memory;
        the CLI must print each one inline.
        """
        from click.testing import CliRunner

        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location("aorta.cli.env", cli_path)
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)

        runner = CliRunner()
        out_path = tmp_path / "env.json"
        result = runner.invoke(cli_mod.env, ["probe", "-o", str(out_path)])
        assert result.exit_code == 0, result.output
        assert "Partial reasons:" in result.output
        # At least one rdhc-style reason will appear (rdhc not on PATH
        # under all_disabled). Each reason rendered as a bullet line.
        bullet_lines = [
            line for line in result.output.splitlines()
            if line.startswith("  - ")
        ]
        assert bullet_lines, (
            f"no '  - <reason>' bullet lines in output: {result.output}"
        )

    def test_cli_closing_marker_partial(self, all_disabled, tmp_path: Path):
        """Closing line repeats the [PARTIAL, N] state at end-of-output
        so it's visible after a long --verbose dump or a long
        partial_reasons list.
        """
        from click.testing import CliRunner

        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location("aorta.cli.env", cli_path)
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)

        runner = CliRunner()
        out_path = tmp_path / "env.json"
        result = runner.invoke(cli_mod.env, ["probe", "-o", str(out_path)])
        assert result.exit_code == 0, result.output
        # Last non-empty line is the closing marker.
        last_line = next(
            line for line in reversed(result.output.splitlines()) if line.strip()
        )
        assert last_line.startswith("[PARTIAL, ") and "reason(s)]" in last_line, (
            f"closing marker missing or malformed: {last_line!r}"
        )

    def test_cli_verbose_flag_dumps_full_json(
        self, all_disabled, tmp_path: Path
    ):
        """``aorta env probe -v`` should print the full JSON snapshot to
        stdout in addition to the brief, so an operator on a remote box
        can copy-paste without reading the JSON file.
        """
        from click.testing import CliRunner

        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location("aorta.cli.env", cli_path)
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)

        runner = CliRunner()
        out_path = tmp_path / "env.json"

        # Without -v: no full snapshot block in stdout
        plain = runner.invoke(cli_mod.env, ["probe", "-o", str(out_path)])
        assert plain.exit_code == 0
        assert "--- Full snapshot ---" not in plain.output

        # With -v: full snapshot block appears and parses back as JSON
        verbose = runner.invoke(
            cli_mod.env, ["probe", "-o", str(out_path), "-v"]
        )
        assert verbose.exit_code == 0
        assert "--- Full snapshot ---" in verbose.output
        # The block after the marker must be the same JSON we wrote to file.
        marker = "--- Full snapshot ---"
        json_block = verbose.output.split(marker, 1)[1]
        # Trim everything after the closing marker.
        if "[PARTIAL, " in json_block:
            json_block = json_block.split("\n[PARTIAL, ", 1)[0]
        elif "[OK]" in json_block:
            json_block = json_block.split("\n[OK]", 1)[0]
        # Must be parseable JSON, and its keys match REQUIRED_TOP_KEYS.
        parsed = json.loads(json_block)
        assert set(parsed.keys()) == REQUIRED_TOP_KEYS

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


class TestCliSummaryAndFieldFlags:
    """1.4: --summary and --field CLI modes (no file write).

    Both short-circuit the JSON write so operators can quickly eyeball
    the brief or script a one-field lookup. The default mode (no flag)
    is unchanged.
    """

    @staticmethod
    def _split_stderr_runner():
        """A CliRunner whose result exposes stderr separately, across Click
        versions. Click < 8.2 combines the streams unless given
        ``mix_stderr=False``; Click >= 8.2 removed that kwarg (streams are
        always separate). Pass it only when the running Click accepts it."""
        from click.testing import CliRunner

        try:
            return CliRunner(mix_stderr=False)
        except TypeError:
            return CliRunner()

    @staticmethod
    def _cli_mod():
        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location(
            "aorta.cli.env", cli_path,
        )
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)
        return cli_mod

    def test_summary_flag_prints_brief_and_skips_file_write(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        # Run in tmp_path so the default env.json output path won't
        # land in the project root if the flag fails to suppress it.
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli_mod.env, ["probe", "--summary"])
            assert result.exit_code == 0, result.output
            # Brief lines from EnvSnapshot.summary() must be present.
            assert "runtime:" in result.output
            assert "rocm:" in result.output
            # No JSON dumped (the default mode would include the JSON
            # file path; --summary should not).
            assert "Wrote env probe to" not in result.output
            # File MUST NOT be written -- the whole point of the flag.
            assert not (Path.cwd() / "env.json").exists()

    def test_execution_context_claim_without_isolation_warns(
        self, all_disabled, tmp_path: Path, monkeypatch,
    ):
        # Claiming buck2_action on a host with no isolation signal and no
        # launcher image env var must warn loudly (the core misdiagnosis
        # guardrail) -- but still exit 0 and write the file. The warning
        # must land on STDERR so it never corrupts stdout scripting; use a
        # split-stderr runner to assert the stream explicitly.
        cli_mod = self._cli_mod()
        # Force container_detected False regardless of the test host.
        monkeypatch.setattr(
            env_mod, "_detect_container_detected", lambda: False
        )
        runner = self._split_stderr_runner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            out = Path.cwd() / "env.json"
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--execution-context", "buck2_action", "-o", str(out)],
            )
            assert result.exit_code == 0, result.stdout
            assert "WARNING" in result.stderr
            assert "WARNING" not in result.stdout
            assert out.exists()

    def test_execution_context_claim_with_image_does_not_warn(
        self, all_disabled, tmp_path: Path, monkeypatch,
    ):
        # Same claim, but $AORTA_DOCKER_IMAGE is set -> no warning (the
        # launcher asserted the context, which is the sanctioned pattern).
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        monkeypatch.setattr(
            env_mod, "_detect_container_detected", lambda: False
        )
        monkeypatch.setenv("AORTA_DOCKER_IMAGE", "rocm/pytorch:tag")
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            out = Path.cwd() / "env.json"
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--execution-context", "buck2_action", "-o", str(out)],
            )
            assert result.exit_code == 0, result.output
            assert "WARNING" not in result.output

    def test_direct_execution_context_never_warns(
        self, all_disabled, tmp_path: Path, monkeypatch,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        monkeypatch.setattr(
            env_mod, "_detect_container_detected", lambda: False
        )
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            out = Path.cwd() / "env.json"
            result = runner.invoke(cli_mod.env, ["probe", "-o", str(out)])
            assert result.exit_code == 0, result.output
            assert "WARNING" not in result.output

    def test_default_probe_writes_compact_catalog(
        self, all_disabled, tmp_path: Path,
    ):
        # Default `probe` (no --extended) writes a JSON whose catalog
        # menus carry no per-file lists -- the size fix. all_disabled
        # yields empty catalogs (files already None), so this asserts the
        # key is present-and-null rather than counting entries.
        import json as _json
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            out = Path.cwd() / "env.json"
            result = runner.invoke(cli_mod.env, ["probe", "-o", str(out)])
            assert result.exit_code == 0, result.output
            doc = _json.loads(out.read_text(encoding="utf-8"))
            assert doc["tensile_catalog"]["hipblaslt"]["menu"]["files"] is None
            assert doc["miopen_catalog"]["menu"]["files"] is None

    def test_extended_flag_is_accepted(
        self, all_disabled, tmp_path: Path,
    ):
        # --extended must parse and still produce a valid snapshot. (Under
        # all_disabled the catalogs are empty so files stays None either
        # way; this guards the flag wiring, not populated-catalog output,
        # which TestCatalogCompactDetail covers at the library layer.)
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            out = Path.cwd() / "env.json"
            result = runner.invoke(
                cli_mod.env, ["probe", "--extended", "-o", str(out)]
            )
            assert result.exit_code == 0, result.output
            assert out.exists()

    def test_field_flag_returns_top_level_scalar_as_json(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env, ["probe", "--field", "schema_version"],
            )
            assert result.exit_code == 0, result.output
            # JSON-typed: a string surfaces with surrounding quotes.
            assert result.output.strip() == f'"{env_mod.SCHEMA_VERSION}"'
            # No file write.
            assert not (Path.cwd() / "env.json").exists()

    def test_field_flag_returns_nested_value(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # all_disabled means most fields are null; pick one we know
            # the default-shape snapshot populates (the new 1.4 keys).
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--field", "pytorch_build.ninja_hipcc._parser"],
            )
            assert result.exit_code == 0, result.output
            assert result.output.strip() == "null"

    def test_field_flag_returns_subdict_as_json(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env, ["probe", "--field", "env_vars"],
            )
            assert result.exit_code == 0, result.output
            # Must be valid JSON and must be a dict.
            payload = json.loads(result.output.strip())
            assert isinstance(payload, dict)
            # Spot-check one canonical key is present.
            assert "HIP_VISIBLE_DEVICES" in payload

    def test_field_flag_missing_top_level_key_lists_available(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--field", "does_not_exist"],
            )
            assert result.exit_code != 0
            # Error message must be helpful: name the missing segment,
            # the path it failed at, and (a sample of) available keys.
            assert "does_not_exist" in result.output
            assert "<root>" in result.output
            assert "Available keys" in result.output

    def test_field_flag_missing_nested_key_scopes_error_to_parent_path(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--field", "pytorch_build.nonsense_key"],
            )
            assert result.exit_code != 0
            assert "nonsense_key" in result.output
            # Parent path explicitly named, not "<root>".
            assert "pytorch_build" in result.output

    def test_field_flag_descending_into_scalar_explains_type(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--field", "schema_version.try_descend"],
            )
            assert result.exit_code != 0
            # Must call out the actual mid-path type so user sees why.
            assert "str" in result.output
            assert "not an object" in result.output

    def test_summary_and_field_are_mutually_exclusive(
        self, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--summary", "--field", "schema_version"],
            )
            assert result.exit_code != 0
            assert "mutually exclusive" in result.output

    def test_default_mode_unchanged_writes_file(
        self, all_disabled, tmp_path: Path,
    ):
        """Schema-stability: invoking probe without the new flags must
        still write env.json + print the brief, like in 1.3 and earlier.
        """
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        out = tmp_path / "env.json"
        result = runner.invoke(cli_mod.env, ["probe", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "Wrote env probe to" in result.output
        # Snapshot is parseable JSON with the expected schema version.
        payload = json.loads(out.read_text())
        assert payload["schema_version"] == env_mod.SCHEMA_VERSION


class TestProbeBuckContextOptions:
    _module_load_count = 0

    @classmethod
    def _cli_mod(cls):
        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        cls._module_load_count += 1
        module_name = f"aorta.cli.env_test_{cls._module_load_count}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            cli_path,
        )
        assert spec is not None and spec.loader is not None
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = cli_mod
        try:
            spec.loader.exec_module(cli_mod)
        finally:
            sys.modules.pop(module_name, None)
        return cli_mod

    @pytest.mark.parametrize(
        "context_args",
        [
            ["--buck-option", "mode=root//mode/debug"],
            ["--buck-mode-file", "root//mode/debug"],
            ["--buck-config", "build.profile=debug"],
            ["--buck-modifier", "//constraints:linux"],
            ["--buck-default-context"],
        ],
    )
    def test_context_options_require_buck_target(self, context_args):
        from click.testing import CliRunner

        result = CliRunner().invoke(
            self._cli_mod().env,
            ["probe", *context_args, "--summary"],
        )
        assert result.exit_code != 0
        assert "require --buck-target" in result.output
        assert "Traceback" not in result.output

    def test_default_confirmation_rejects_explicit_context(self):
        from click.testing import CliRunner

        result = CliRunner().invoke(
            self._cli_mod().env,
            [
                "probe",
                "--buck-target",
                "//app:trainer",
                "--buck-default-context",
                "--buck-modifier",
                "//constraints:linux",
                "--summary",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_repeatable_options_preserve_each_axis_order(self, monkeypatch):
        from click.testing import CliRunner

        captured = {}

        def fake_collect_env(**kwargs):
            captured.update(kwargs)
            return _example_snapshot()

        monkeypatch.setattr(env_mod, "collect_env", fake_collect_env)
        result = CliRunner().invoke(
            self._cli_mod().env,
            [
                "probe",
                "--buck-target",
                "//app:trainer",
                "--buck-mode-file",
                "root//mode/debug",
                "--buck-mode-file",
                "root//mode/gpu",
                "--buck-config",
                "build.profile=debug",
                "--buck-config",
                "scheduler.policy=local",
                "--buck-modifier",
                "//constraints:linux",
                "--buck-modifier",
                "//constraints:gfx",
                "--summary",
            ],
        )

        assert result.exit_code == 0, result.output
        context = captured["buck_context"]
        assert context.mode_files == (
            "root//mode/debug",
            "root//mode/gpu",
        )
        assert context.config_overrides == (
            "build.profile=debug",
            "scheduler.policy=local",
        )
        assert context.modifiers == (
            "//constraints:linux",
            "//constraints:gfx",
        )
        assert context.to_buck_args() == [
            "@root//mode/debug",
            "@root//mode/gpu",
            "-c",
            "build.profile=debug",
            "-c",
            "scheduler.policy=local",
            "-m",
            "//constraints:linux",
            "-m",
            "//constraints:gfx",
        ]

    def test_ordered_option_preserves_cross_type_order(self, monkeypatch):
        from click.testing import CliRunner

        captured = {}

        def fake_collect_env(**kwargs):
            captured.update(kwargs)
            return _example_snapshot()

        monkeypatch.setattr(env_mod, "collect_env", fake_collect_env)
        result = CliRunner().invoke(
            self._cli_mod().env,
            [
                "probe",
                "--buck-target",
                "//app:trainer",
                "--buck-option",
                "config=build.profile=debug",
                "--buck-option",
                "mode=root//mode/override",
                "--buck-option",
                "modifier=//constraints:gfx",
                "--summary",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["buck_context"].to_buck_args() == [
            "-c",
            "build.profile=debug",
            "@root//mode/override",
            "-m",
            "//constraints:gfx",
        ]

    def test_ordered_option_rejects_grouped_options(self):
        from click.testing import CliRunner

        result = CliRunner().invoke(
            self._cli_mod().env,
            [
                "probe",
                "--buck-target",
                "//app:trainer",
                "--buck-option",
                "mode=root//mode/debug",
                "--buck-config",
                "build.profile=debug",
                "--summary",
            ],
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output

    def test_default_confirmation_and_unspecified_target_are_distinct(
        self, monkeypatch
    ):
        from click.testing import CliRunner

        contexts = []

        def fake_collect_env(**kwargs):
            contexts.append(kwargs["buck_context"])
            return _example_snapshot()

        monkeypatch.setattr(env_mod, "collect_env", fake_collect_env)
        cli_mod = self._cli_mod()
        runner = CliRunner()

        unspecified = runner.invoke(
            cli_mod.env,
            ["probe", "--buck-target", "//app:trainer", "--summary"],
        )
        confirmed = runner.invoke(
            cli_mod.env,
            [
                "probe",
                "--buck-target",
                "//app:trainer",
                "--buck-default-context",
                "--summary",
            ],
        )

        assert unspecified.exit_code == 0, unspecified.output
        assert confirmed.exit_code == 0, confirmed.output
        assert contexts[0].source == "unspecified"
        assert contexts[1].source == "default_confirmed"

    def test_invalid_config_shape_is_clean_click_error(self):
        from click.testing import CliRunner

        result = CliRunner().invoke(
            self._cli_mod().env,
            [
                "probe",
                "--buck-target",
                "//app:trainer",
                "--buck-config",
                "build.profile",
                "--summary",
            ],
        )
        assert result.exit_code != 0
        assert "KEY=VALUE" in result.output
        assert "--buck-config" in result.output
        assert "Traceback" not in result.output


class TestProbeBuckTimeoutValidation:
    """Per Copilot review on PR #165: ``--buck-timeout`` must reject
    values <= 0 at arg-parse time. ``subprocess.run(timeout=...)``
    raises ``ValueError`` on a non-positive timeout, which would
    short-circuit the buck audit into the never-raises fallback path
    instead of failing loudly. Catching it in Click means the operator
    sees the typo, not a silent "no introspection done".
    """

    @staticmethod
    def _cli_mod():
        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location(
            "aorta.cli.env", cli_path,
        )
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)
        return cli_mod

    @pytest.mark.parametrize("bad_value", ["0", "-1", "-100"])
    def test_buck_timeout_non_positive_rejected_by_click(
        self, bad_value, all_disabled, tmp_path: Path,
    ):
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--buck-timeout", bad_value, "--summary"],
            )
            assert result.exit_code != 0
            # Click's IntRange formatting names the option and the
            # offending value; assert on stable text.
            lower = result.output.lower()
            assert "buck-timeout" in lower
            assert "invalid value" in lower or "not in the range" in lower
            # No Python traceback should leak.
            assert "Traceback" not in result.output

    def test_buck_timeout_positive_value_accepted(
        self, all_disabled, tmp_path: Path,
    ):
        """Sanity: any value >= 1 still parses (no regression on the
        documented default of 10 or any other reasonable override).
        """
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli_mod.env,
                ["probe", "--buck-timeout", "5", "--summary"],
            )
            assert result.exit_code == 0, result.output


class TestProbeWritesUtf8:
    """Per Copilot review on PR #181: `aorta env probe` must write
    env.json with `encoding="utf-8"` so the symmetric
    `aorta env recipe` reader (which forces utf-8) can always read it
    back, regardless of the host's default encoding. Without this,
    a non-UTF-8 default locale (e.g. cp1252) on the producing host
    could silently produce a file that the consumer refuses to decode.
    """

    @staticmethod
    def _cli_mod():
        cli_path = Path(env_mod.__file__).parent.parent / "cli" / "env.py"
        spec = importlib.util.spec_from_file_location(
            "aorta.cli.env", cli_path,
        )
        cli_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cli_mod
        spec.loader.exec_module(cli_mod)
        return cli_mod

    def test_probe_passes_utf8_encoding_to_write_text(
        self, all_disabled, tmp_path: Path,
    ):
        """Direct regression guard: patch ``Path.write_text`` and
        assert ``encoding="utf-8"`` is in the call. This is the only
        unambiguous way to assert the writer side of the round-trip
        contract without manufacturing a non-UTF-8 host environment.
        """
        from click.testing import CliRunner

        captured_kwargs: dict = {}
        original_write_text = Path.write_text

        def fake_write_text(self, *args, **kwargs):
            # Capture the kwargs the CLI passed and then delegate to
            # the real implementation so the file actually lands and
            # the rest of the CLI path (echoes, summary, etc.) runs.
            captured_kwargs.update(kwargs)
            return original_write_text(self, *args, **kwargs)

        cli_mod = self._cli_mod()
        runner = CliRunner()
        out = tmp_path / "env.json"
        with patch.object(Path, "write_text", fake_write_text):
            result = runner.invoke(cli_mod.env, ["probe", "-o", str(out)])

        assert result.exit_code == 0, result.output
        assert captured_kwargs.get("encoding") == "utf-8", (
            f"probe must pass encoding='utf-8' to write_text; "
            f"got kwargs={captured_kwargs}"
        )

    def test_probe_then_recipe_round_trip_succeeds(
        self, all_disabled, tmp_path: Path,
    ):
        """End-to-end: write a snapshot with ``probe``, then read it
        with ``recipe`` and confirm the recipe is emitted. This is the
        operator-visible contract Copilot's comment was protecting.
        """
        from click.testing import CliRunner
        cli_mod = self._cli_mod()
        runner = CliRunner()
        out = tmp_path / "env.json"

        probe_result = runner.invoke(cli_mod.env, ["probe", "-o", str(out)])
        assert probe_result.exit_code == 0, probe_result.output
        assert out.exists()

        # The file must be valid UTF-8.
        decoded = out.read_text(encoding="utf-8")
        assert "schema_version" in decoded

        recipe_result = runner.invoke(
            cli_mod.env, ["recipe", str(out), "--format", "buck"]
        )
        assert recipe_result.exit_code == 0, recipe_result.output
        assert "BEST-EFFORT, NOT EXACT" in recipe_result.output


class TestCaptureTo:
    """Programmatic writer for Buck ``.par`` applications that own __main__."""

    def test_writes_complete_json_and_returns_snapshot(
        self, all_disabled, tmp_path: Path
    ):
        output = tmp_path / "nested" / "env.action.json"
        snapshot = env_mod.capture_to(
            output,
            probe_invocation="buck2_run",
        )
        assert output.is_file()
        assert output.stat().st_size > 0
        on_disk = json.loads(output.read_text(encoding="utf-8"))
        assert on_disk == snapshot.to_dict()

    def test_forwards_buck_context_without_persisting_config_value(
        self, all_disabled, tmp_path: Path
    ):
        hidden = "private-placeholder"
        context = env_mod.BuckInvocationContext(
            config_overrides=(f"build.profile={hidden}",),
        )
        output = tmp_path / "env.action.json"
        snapshot = env_mod.capture_to(
            output,
            buck_target="//app:trainer",
            buck_context=context,
            probe_invocation="buck2_run",
        )
        serialized = output.read_text(encoding="utf-8")
        assert snapshot.buck_invocation["config_keys"] == ["build.profile"]
        assert hidden not in serialized

    def test_write_failure_is_not_silenced(
        self, all_disabled, tmp_path: Path, monkeypatch
    ):
        def fail_write(*args, **kwargs):
            raise OSError("synthetic write failure")

        monkeypatch.setattr(Path, "write_text", fail_write)
        with pytest.raises(OSError, match="synthetic write failure"):
            env_mod.capture_to(tmp_path / "env.json")


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


def _disable_rocm_version_fallbacks(monkeypatch):
    """Silence the non-file sources in the ROCm version chain.

    Without this, a test asserting "no version resolved" would pass or fail
    depending on whether the host happens to have a ROCm torch or the
    ``rocm`` distribution installed.
    """
    monkeypatch.setattr(env_mod, "_distribution_version", lambda name: None)
    monkeypatch.setattr(env_mod, "_rocm_version_from_torch", lambda: None)


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
        assert result["version"] == "7.2.1"
        assert result["version_dev"] == "7.2.1.50311-abc1234"
        assert result["kmd_version"] == "6.16.13"
        # The file is the most authoritative source, so no fallback ran.
        assert result["version_source"] == "version_file"
        assert reasons == []

    def test_all_missing_appends_three_reasons(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "nope1")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "nope2")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "nope3")
        _disable_rocm_version_fallbacks(monkeypatch)
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons)
        assert result["version"] is None
        assert result["version_dev"] is None
        assert result["kmd_version"] is None
        assert result["version_source"] is None
        assert len(reasons) == 3
        assert all(r.startswith("rocm.") for r in reasons)

    def test_attribution_keys_are_always_populated(self, monkeypatch):
        """A null version must say which root was searched and how it was found.

        Before #381 a null was unattributable: "found an install with no
        version file" and "found no install" produced identical output.
        """
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons)
        assert result["root"] == str(env_mod.ROCM_ROOT)
        assert result["lib_root"] == str(env_mod.ROCM_LIB_ROOT)
        assert result["root_source"] == env_mod.ROCM_ROOT_SOURCE
        assert result["layout"] in {"classic", "wheel"}

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
# ROCm version fallback chain + TheRock provenance (schema 1.16, issue #381)
# ---------------------------------------------------------------------------


class TestRocmVersionFallbackChain:
    """Each source in the chain, isolated (#381 acceptance).

    Ordered most to least authoritative: the version file, then TheRock's
    build manifest, then what pip says is installed, then the ROCm release
    torch happens to have been compiled against. ``version_source`` has to
    name the winner, because "7.14.0 read from a file" and "7.14.0 inferred
    from a torch wheel" are not equally strong claims.
    """

    @pytest.fixture
    def no_version_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "absent")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "absent_dev")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "absent_kmd")
        monkeypatch.setattr(env_mod, "_distribution_version", lambda name: None)
        monkeypatch.setattr(env_mod, "_rocm_version_from_torch", lambda: None)
        return monkeypatch

    def test_version_file_wins_and_short_circuits(self, tmp_path: Path, monkeypatch):
        version = tmp_path / "version"
        version.write_text("7.2.4\n", encoding="utf-8")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", version)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "absent")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "absent_kmd")

        def fail(*args, **kwargs):
            raise AssertionError("a later fallback ran despite the file resolving")

        monkeypatch.setattr(env_mod, "_distribution_version", fail)
        monkeypatch.setattr(env_mod, "_rocm_version_from_torch", fail)

        result = env_mod._capture_rocm_version_files([], {"rocm_version": "9.9.9"})
        assert result["version"] == "7.2.4"
        assert result["version_source"] == "version_file"

    def test_manifest_used_when_no_version_file(self, no_version_file):
        result = env_mod._capture_rocm_version_files([], {"rocm_version": "7.14.0"})
        assert result["version"] == "7.14.0"
        assert result["version_source"] == "therock_manifest"

    def test_pip_rocm_used_when_no_file_and_no_manifest(self, no_version_file):
        no_version_file.setattr(
            env_mod,
            "_distribution_version",
            lambda name: "7.15.0a20260716" if name == "rocm" else None,
        )
        result = env_mod._capture_rocm_version_files([], None)
        assert result["version"] == "7.15.0a20260716"
        assert result["version_source"] == "pip:rocm"

    def test_pip_falls_through_to_rocm_sdk_core(self, no_version_file):
        no_version_file.setattr(
            env_mod,
            "_distribution_version",
            lambda name: "7.15.0" if name == "rocm-sdk-core" else None,
        )
        result = env_mod._capture_rocm_version_files([], None)
        assert result["version_source"] == "pip:rocm-sdk-core"

    def test_torch_is_the_last_resort(self, no_version_file):
        no_version_file.setattr(env_mod, "_rocm_version_from_torch", lambda: "7.14.0")
        result = env_mod._capture_rocm_version_files([], None)
        assert result["version"] == "7.14.0"
        assert result["version_source"] == "torch"

    def test_all_sources_absent_yields_null_and_a_reason(self, no_version_file):
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons, None)
        assert result["version"] is None
        assert result["version_source"] is None
        assert any(r.startswith("rocm.version:") for r in reasons)

    def test_version_dev_falls_back_to_the_manifest_package_version(
        self, no_version_file
    ):
        """The wheel layout ships no ``version-dev``.

        ``rocm_package_version`` is its analogue: the precise build rather
        than the release tag.
        """
        result = env_mod._capture_rocm_version_files(
            [], {"rocm_version": "7.14.0", "rocm_package_version": "7.14.0rc20260801"}
        )
        assert result["version_dev"] == "7.14.0rc20260801"

    @pytest.mark.parametrize(
        "pip_value", [pytest.param("", id="empty"), pytest.param("   ", id="whitespace")]
    )
    def test_an_empty_pip_version_does_not_block_the_torch_fallback(
        self, no_version_file, pip_value: str
    ):
        """A falsy-but-not-None pip value used to end the chain silently (#387).

        The loop assigned into ``version`` before testing it, while every step
        around it is guarded on ``is None``. An empty version from the LAST
        distribution consulted therefore left ``version == ""``: falsy, so
        ``version_source`` was never set and the loop did not break, but not
        ``None``, so torch was skipped and the reason loop appended nothing.
        """
        no_version_file.setattr(
            env_mod,
            "_distribution_version",
            lambda name: pip_value if name == "rocm-sdk-core" else None,
        )
        no_version_file.setattr(env_mod, "_rocm_version_from_torch", lambda: "7.14.0")

        result = env_mod._capture_rocm_version_files([], None)
        assert result["version"] == "7.14.0"
        assert result["version_source"] == "torch"

    @pytest.mark.parametrize(
        "pip_value", [pytest.param("", id="empty"), pytest.param("  \n ", id="whitespace")]
    )
    def test_an_empty_pip_version_with_no_torch_is_null_with_a_reason(
        self, no_version_file, pip_value: str
    ):
        """Never ``version: ""`` with ``version_source: null`` and no reason.

        That shape is a non-null nobody can attribute, which is exactly what
        the schema-1.16 attribution keys exist to eliminate.
        """
        no_version_file.setattr(
            env_mod, "_distribution_version", lambda name: pip_value
        )
        reasons: list[str] = []
        result = env_mod._capture_rocm_version_files(reasons, None)

        assert result["version"] is None
        assert result["version_source"] is None
        assert any(r.startswith("rocm.version:") for r in reasons)

    @pytest.mark.parametrize(
        ("source", "setter"),
        [
            pytest.param("pip:rocm", "_distribution_version", id="pip"),
            pytest.param("torch", "_rocm_version_from_torch", id="torch"),
        ],
    )
    def test_non_file_sources_are_stripped_like_the_file_sources(
        self, no_version_file, source: str, setter: str
    ):
        """``_distribution_version`` and torch are the only unnormalised sources.

        ``read_version_marker`` and ``_clean_manifest_string`` both strip, so a
        padded value from either of these would otherwise be reported verbatim
        into a field consumers compare across hosts.
        """
        value = " 7.14.0 "
        if setter == "_distribution_version":
            no_version_file.setattr(
                env_mod, setter, lambda name: value if name == "rocm" else None
            )
        else:
            no_version_file.setattr(env_mod, setter, lambda: value)

        result = env_mod._capture_rocm_version_files([], None)
        assert result["version"] == "7.14.0"
        assert result["version_source"] == source

    def test_version_markers_are_read_by_the_same_rule_discovery_validated_them(
        self, tmp_path: Path, monkeypatch
    ):
        """The probe must not reject a marker discovery accepted (#387).

        These two files decide whether an autodetected ``/opt/rocm`` outranks an
        importable wheel, so reading them with the unbounded ``_read_text_file``
        while discovery validated them with the bounded
        ``rocm_paths.read_version_marker`` reintroduced the divergence one module
        out: valid UTF-8 through the 4096-byte read and invalid past it was
        accepted as evidence of an install and then reported as ``None`` here.
        """
        version = tmp_path / "version"
        version.write_bytes(b"7.2.4" + b"x" * 4995 + b"\xff")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", version)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "absent_dev")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "absent_kmd")
        monkeypatch.setattr(env_mod, "_distribution_version", lambda name: None)
        monkeypatch.setattr(env_mod, "_rocm_version_from_torch", lambda: None)

        assert rocm_paths.read_version_marker(version) is not None
        result = env_mod._capture_rocm_version_files([], None)
        assert result["version"] == rocm_paths.read_version_marker(version)
        assert result["version_source"] == "version_file"


class TestRocmVersionFromTorch:
    @staticmethod
    def _fake_torch(monkeypatch, version: str, hip: str | None):
        module = types.SimpleNamespace(
            __version__=version, version=types.SimpleNamespace(hip=hip)
        )
        monkeypatch.setitem(sys.modules, "torch", module)

    def test_local_segment_is_preferred_over_torch_version_hip(self, monkeypatch):
        """``+rocm...`` carries the full package version; ``.hip`` truncates it."""
        self._fake_torch(monkeypatch, "2.10.0+rocm7.15.0a20260716", "7.15.0")
        assert env_mod._rocm_version_from_torch() == "7.15.0a20260716"

    def test_falls_back_to_torch_version_hip(self, monkeypatch):
        self._fake_torch(monkeypatch, "2.12.0", "7.14.60850")
        assert env_mod._rocm_version_from_torch() == "7.14.60850"

    def test_cpu_only_torch_yields_none(self, monkeypatch):
        self._fake_torch(monkeypatch, "2.12.0+cpu", None)
        assert env_mod._rocm_version_from_torch() is None

    def test_broken_torch_import_does_not_raise(self, monkeypatch):
        def boom(name, *args, **kwargs):
            if name == "torch":
                raise RuntimeError("half-installed torch")
            return original(name, *args, **kwargs)

        original = builtins.__import__
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", boom)
        assert env_mod._rocm_version_from_torch() is None


class TestTheRockManifest:
    """Manifest parsing (#381 acceptance: full SHA recorded when present)."""

    MANIFEST = {
        "rocm_version": "7.14.0",
        "rocm_package_version": "7.14.0",
        "the_rock_commit": "418cd5f63abb7a604bad5874cd7b2e29334e640f",
        "github_run_id": "29052710811",
        "github_job": "build_stage",
        "submodules": [
            {
                "submodule_name": "llvm-project",
                "submodule_url": "https://github.com/ROCm/llvm-project.git",
                "pin_sha": "46fcb339fb61119b337f973c7ca9e710a319fdd0",
                "patches": ["patches/amd-mainline/llvm-project/0002-hipcc.patch"],
            },
            {
                "submodule_name": "rocm-libraries",
                "submodule_url": "https://github.com/ROCm/rocm-libraries",
                "pin_sha": "cd9574023093742434e8c992d13b89ab9a6c1cf8",
                "patches": [],
            },
        ],
    }

    def test_absent_manifest_is_a_documented_absence(self, tmp_path, monkeypatch):
        """Classic installs have no manifest and never will."""
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", tmp_path / "absent.json")
        assert env_mod._read_therock_manifest() == (None, None)
        reasons: list[str] = []
        block = env_mod._capture_therock(None, reasons)
        assert block["status"] == "absent"
        assert block["gemm_libraries_commit"] is None
        assert block["submodules"] == []
        # A documented absence must NOT raise a partial.
        assert reasons == []

    def test_manifest_is_read_and_parsed(self, tmp_path, monkeypatch):
        path = tmp_path / "therock_manifest.json"
        path.write_text(json.dumps(self.MANIFEST), encoding="utf-8")
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", path)
        assert env_mod._read_therock_manifest() == (self.MANIFEST, None)

    @pytest.mark.parametrize(
        "content,expected_in_error",
        [
            ("{not json", "not valid JSON"),
            ("[1, 2, 3]", "not an object"),
            ('"just a string"', "not an object"),
            ("", "present but empty or unreadable"),
        ],
    )
    def test_a_broken_manifest_is_distinguished_from_absence(
        self, tmp_path, monkeypatch, content, expected_in_error
    ):
        """A corrupt manifest must not read as the classic-layout absence (#387).

        ``status="absent"`` deliberately raises no partial, so collapsing a
        damaged wheel install into it would report the expected classic reading
        and silently drop the provenance -- on a wheel image this block is the
        only source of the full 40-char GEMM pin, which is the evidence the NaN
        escalations are argued from.
        """
        path = tmp_path / "therock_manifest.json"
        path.write_text(content, encoding="utf-8")
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", path)

        manifest, error = env_mod._read_therock_manifest()
        assert manifest is None
        assert error is not None and expected_in_error in error

        reasons: list[str] = []
        block = env_mod._capture_therock(manifest, reasons, error)
        assert block["status"] == "invalid"
        assert any(r.startswith("therock:") for r in reasons)
        # Same key set as every other reading, so consumers index one shape.
        assert set(block) == set(env_mod._empty_therock())

    def test_a_non_utf8_manifest_is_reported_as_invalid(self, tmp_path, monkeypatch):
        """`_read_text_file` returns None for non-UTF8 too, which is not absence."""
        path = tmp_path / "therock_manifest.json"
        path.write_bytes(b"\xff\xfe\x80not-utf8")
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", path)
        manifest, error = env_mod._read_therock_manifest()
        assert manifest is None
        assert error is not None
        assert env_mod._capture_therock(manifest, [], error)["status"] == "invalid"

    def test_collect_env_flags_a_broken_manifest_as_partial(
        self, all_disabled, tmp_path, monkeypatch
    ):
        """End-to-end: the reason reaches the snapshot, not just the helper."""
        path = tmp_path / "therock_manifest.json"
        path.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", path)
        snapshot = collect_env()
        assert snapshot.therock["status"] == "invalid"
        assert snapshot.partial is True
        assert any(r.startswith("therock:") for r in snapshot.partial_reasons)

    def test_capture_records_build_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            env_mod, "THEROCK_MANIFEST_FILE", tmp_path / "therock_manifest.json"
        )
        block = env_mod._capture_therock(self.MANIFEST)
        assert block["status"] == "present"
        assert block["rocm_version"] == "7.14.0"
        assert block["the_rock_commit"] == "418cd5f63abb7a604bad5874cd7b2e29334e640f"
        assert block["github_run_id"] == "29052710811"
        assert block["github_job"] == "build_stage"

    def test_gemm_libraries_commit_is_the_full_forty_char_pin(self):
        """The whole point of the manifest.

        The classic header exposes only a truncated tweak; this is the same
        commit at full width, and it is the hipBLASLt provenance the NaN
        escalations are argued from.
        """
        block = env_mod._capture_therock(self.MANIFEST)
        assert block["gemm_libraries_commit"] == (
            "cd9574023093742434e8c992d13b89ab9a6c1cf8"
        )
        assert len(block["gemm_libraries_commit"]) == 40

    def test_patches_are_recorded_against_their_pin(self):
        """A non-empty patch list means the build is NOT that commit alone."""
        block = env_mod._capture_therock(self.MANIFEST)
        llvm = next(s for s in block["submodules"] if s["name"] == "llvm-project")
        assert llvm["patches"] == [
            "patches/amd-mainline/llvm-project/0002-hipcc.patch"
        ]
        gemm = next(s for s in block["submodules"] if s["name"] == "rocm-libraries")
        assert gemm["patches"] == []

    def test_malformed_submodule_entries_are_skipped(self):
        block = env_mod._capture_therock(
            {"submodules": ["not-a-dict", {"submodule_name": "x", "pin_sha": None}]}
        )
        assert [s["name"] for s in block["submodules"]] == ["x"]
        assert block["gemm_libraries_commit"] is None

    def test_non_list_submodules_does_not_raise(self):
        block = env_mod._capture_therock({"submodules": "nope"})
        assert block["submodules"] == []

    def test_an_unstatable_manifest_is_invalid_not_absent(self, tmp_path, monkeypatch):
        """Only a path confirmed missing earns "absent" (#387, 3rd pass).

        A stale mount or an unreadable parent makes even ``exists()`` raise;
        calling that the documented classic absence is the same conflation this
        function exists to undo.
        """
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", tmp_path / "m.json")
        monkeypatch.setattr(env_mod, "_read_text_file", lambda path: None)

        def boom(self):
            raise OSError("stale NFS handle")

        # The presence probe is lstat(), not exists() -- see the dangling-symlink
        # test below for why.
        monkeypatch.setattr(Path, "lstat", boom)
        manifest, error = env_mod._read_therock_manifest()
        assert manifest is None
        assert error is not None and "could not be checked" in error
        assert env_mod._capture_therock(manifest, [], error)["status"] == "invalid"

    def test_a_dangling_symlink_manifest_is_invalid_not_absent(self, tmp_path, monkeypatch):
        """`exists()` follows symlinks, so a broken link looked like absence.

        A dangling symlink IS a present directory entry on a damaged install, so
        it belongs in "invalid" -- the same absent/invalid conflation as the
        previous two rounds, one filesystem call further in (#387).
        """
        link = tmp_path / "therock_manifest.json"
        link.symlink_to(tmp_path / "gone.json")
        assert not link.exists()          # the old check said "absent"
        assert link.is_symlink()          # ...but the entry is right there
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", link)
        manifest, error = env_mod._read_therock_manifest()
        assert manifest is None
        assert error is not None
        assert env_mod._capture_therock(manifest, [], error)["status"] == "invalid"

    def test_a_truly_absent_manifest_is_still_absent(self, tmp_path, monkeypatch):
        """The lstat change must not turn the normal classic reading into a partial."""
        monkeypatch.setattr(env_mod, "THEROCK_MANIFEST_FILE", tmp_path / "nope.json")
        assert env_mod._read_therock_manifest() == (None, None)
        reasons: list[str] = []
        assert env_mod._capture_therock(None, reasons)["status"] == "absent"
        assert reasons == []

    def test_malformed_submodule_entries_are_surfaced_not_swallowed(self):
        """Kept-with-null-name is visible; dropped is indistinguishable from absent.

        ``submodules`` is provenance, so under-reporting what the manifest
        listed is its own silent degradation. The entries stay, and a reason
        makes the malformed shape actionable (#387, 3rd pass).
        """
        reasons: list[str] = []
        block = env_mod._capture_therock(
            {
                "submodules": [
                    "not-a-dict",
                    {"submodule_name": "rocm-libraries", "pin_sha": "a" * 40},
                    {"pin_sha": "b" * 40},
                ]
            },
            reasons,
        )
        # The unnamed entry is retained, with its pin, and visibly unnamed.
        assert [s["name"] for s in block["submodules"]] == ["rocm-libraries", None]
        assert block["submodules"][1]["pin_sha"] == "b" * 40
        # ...and the shape problem is reported rather than inferred from counts.
        assert any(r.startswith("therock.submodules:") for r in reasons)
        # A nameless entry can never be mistaken for the GEMM pin.
        assert block["gemm_libraries_commit"] == "a" * 40

    def test_a_non_list_submodules_key_is_surfaced(self):
        """Distinct from "no submodules": the key was present and unusable."""
        reasons: list[str] = []
        env_mod._capture_therock({"submodules": "nope"}, reasons)
        assert any(r.startswith("therock.submodules:") for r in reasons)

    def test_a_well_formed_manifest_raises_no_submodule_reason(self):
        reasons: list[str] = []
        env_mod._capture_therock(self.MANIFEST, reasons)
        assert reasons == []

    @pytest.mark.parametrize(
        "pin",
        [
            pytest.param("cd95740", id="abbreviated"),
            pytest.param("c" * 39, id="one-short"),
            pytest.param("c" * 41, id="one-long"),
            pytest.param("z" * 40, id="right-length-not-hex"),
            pytest.param("cd9574023093742434e8c992d13b89ab9a6c1cf ", id="embedded-space"),
            pytest.param(12345, id="numeric-scalar"),
            pytest.param("HEAD", id="symbolic"),
        ],
    )
    def test_a_pin_that_is_not_a_full_sha_is_not_reported_as_provenance(self, pin):
        """``pin_sha`` makes a checkable claim, so a half-valid one is worse than null.

        ``_clean_manifest_string`` accepted any scalar, so an abbreviated or
        non-hex pin became ``pin_sha`` *and* was promoted to
        ``gemm_libraries_commit`` -> ``upstream_commit``, while the block stayed
        ``status="present"`` with no reason. The schema documents these as full
        40-char SHAs, so that was provenance which looked authoritative and was
        not -- and it fed the tweak prefix comparison (#387, 8th pass).
        """
        reasons: list[str] = []
        block = env_mod._capture_therock(
            {"submodules": [{"submodule_name": "rocm-libraries", "pin_sha": pin}]},
            reasons,
        )
        # The entry survives (it names a real submodule) but claims no commit.
        assert [s["name"] for s in block["submodules"]] == ["rocm-libraries"]
        assert block["submodules"][0]["pin_sha"] is None
        # Never promoted -- this is what upstream_commit reads.
        assert block["gemm_libraries_commit"] is None
        assert any("full 40-char pin_sha" in r for r in reasons)

    def test_an_uppercase_pin_is_accepted_and_normalised(self):
        """Case is not a defect: normalise so it compares against lowercase tweaks."""
        reasons: list[str] = []
        block = env_mod._capture_therock(
            {"submodules": [{"submodule_name": "rocm-libraries", "pin_sha": "AB" * 20}]},
            reasons,
        )
        assert block["gemm_libraries_commit"] == "ab" * 20
        assert reasons == []

    def test_an_invalid_pin_cannot_satisfy_the_tweak_invariant(self):
        """The reason the promotion matters, asserted end to end.

        A tweak is compared as a prefix of the manifest pin. Before this, a
        manifest whose pin was merely the tweak itself would "match" and report
        ``upstream_commit_matches_tweak=True`` off nothing.
        """
        block = env_mod._capture_therock(
            {"submodules": [{"submodule_name": "rocm-libraries", "pin_sha": "cd95740230"}]},
            [],
        )
        assert (
            env_mod._gemm_provenance_matches("cd95740230", block["gemm_libraries_commit"])
            is None
        )


class TestGemmProvenanceInvariant:
    """The tweak/manifest cross-check (#381 acceptance).

    Synthetic data only, and deliberately so: no real image ships both sides.
    The classic layout has the headers but no manifest, and the runtime wheel
    has the manifest but no headers, so on every image available today the
    field is ``None``. It exists for the devel-wheel case and as a mis-parse
    guard.
    """

    MANIFEST_SHA = "5b515cf1bcaa0d2f3e4c5d6a7b8c9d0e1f2a3b4c"

    def test_matching_prefix_is_true(self):
        assert env_mod._gemm_provenance_matches("5b515cf1bc", self.MANIFEST_SHA) is True

    def test_prefix_is_compared_at_the_tweaks_own_length(self):
        """Not a fixed 8 chars.

        Measured on ROCm 7.2.4 the tweak is 10 characters, and the field is
        documented as 7-12, so a hardcoded width would either truncate the
        comparison or fail a correct install.
        """
        assert len("5b515cf1bc") == 10
        assert env_mod._gemm_provenance_matches("5b515cf", self.MANIFEST_SHA) is True
        assert env_mod._gemm_provenance_matches("5b515cf1bcaa", self.MANIFEST_SHA) is True

    def test_mismatch_is_false(self):
        assert (
            env_mod._gemm_provenance_matches("deadbeef12", self.MANIFEST_SHA) is False
        )

    @pytest.mark.parametrize(
        "tweak",
        [
            pytest.param("5", id="one-char"),
            pytest.param("5b515c", id="six-chars-under-documented-min"),
            pytest.param("5b515cf1bcaa0", id="thirteen-chars-over-max"),
            pytest.param("unknown", id="non-hex-build-string"),
            pytest.param("dirty", id="non-hex-short"),
            pytest.param("5b515cf-", id="trailing-punctuation"),
            pytest.param("v5b515cf", id="leading-letter"),
        ],
    )
    def test_an_implausible_tweak_is_not_comparable(self, tweak):
        """A boolean here has to mean something (#387, round 9).

        The header regex accepts any ``[A-Za-z0-9_.+-]`` token, so two distinct
        lies were reachable. A one-character tweak prefix-matches about one SHA
        in sixteen and returned ``True`` off almost no evidence; a non-hex build
        string returned ``False``, asserting a *mismatch* between two things
        that were never comparable. Both are now ``None``.
        """
        assert env_mod._gemm_provenance_matches(tweak, self.MANIFEST_SHA) is None

    def test_a_one_char_tweak_no_longer_matches_by_luck(self):
        """The concrete false positive, stated as such."""
        assert self.MANIFEST_SHA.startswith("5")
        assert env_mod._gemm_provenance_matches("5", self.MANIFEST_SHA) is None

    def test_both_sides_must_be_valid_for_a_verdict(self):
        """An invalid pin and an invalid tweak are both "not comparable".

        Pairs with the manifest-side validation: neither side alone can produce
        a verdict, so no combination of junk yields True/False.
        """
        short_pin = env_mod._capture_therock(
            {"submodules": [{"submodule_name": "rocm-libraries", "pin_sha": "5b515cf1bc"}]},
            [],
        )["gemm_libraries_commit"]
        assert env_mod._gemm_provenance_matches("5b515cf1bc", short_pin) is None
        assert env_mod._gemm_provenance_matches("unknown", self.MANIFEST_SHA) is None

    def test_comparison_is_case_insensitive(self):
        assert (
            env_mod._gemm_provenance_matches("5B515CF1BC", self.MANIFEST_SHA.upper())
            is True
        )

    @pytest.mark.parametrize(
        "tweak,manifest_sha",
        [
            (None, MANIFEST_SHA),
            ("5b515cf1bc", None),
            (None, None),
            ("", MANIFEST_SHA),
            ("5b515cf1bc", ""),
        ],
    )
    def test_absent_side_yields_none_not_false(self, tweak, manifest_sha):
        """``None`` means "could not check", ``False`` means "checked, mismatched".

        Collapsing them would report every real image today as a provenance
        mismatch.
        """
        assert env_mod._gemm_provenance_matches(tweak, manifest_sha) is None


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


class TestKernelDbFingerprintPerArch:
    """The wheel layout nests the kernel DB per target (#387, 8th pass).

    ``library/`` on a TheRock install contains *only* ``gfx*`` directories, so a
    flat-only scan returned ``None`` and added a partial reason for a kernel
    database that was entirely present. These cover the per-arch loop directly:
    the nested-directory tests elsewhere exercise ``_enumerate_catalog_dir``,
    which is a separate implementation of the same idea, so deleting this loop
    left ``hipblaslt.kernel_db_revision`` null with the suite still green.
    """

    def test_a_wheel_layout_kernel_db_is_fingerprinted(self, tmp_path: Path):
        library = tmp_path / "library"
        (library / "gfx950").mkdir(parents=True)
        (library / "gfx950" / "TensileLibrary_A.dat").write_bytes(b"x")
        fp = env_mod._kernel_db_filename_fingerprint(library)
        assert fp is not None and fp.startswith("filenames-sha256:")

    def test_the_arch_qualifies_the_name(self, tmp_path: Path):
        """The same kernel under two targets must not collapse to one entry."""
        one = tmp_path / "one" / "library"
        (one / "gfx950").mkdir(parents=True)
        (one / "gfx950" / "TensileLibrary_A.dat").write_bytes(b"x")

        two = tmp_path / "two" / "library"
        (two / "gfx950").mkdir(parents=True)
        (two / "gfx950" / "TensileLibrary_A.dat").write_bytes(b"x")
        (two / "gfx942").mkdir(parents=True)
        (two / "gfx942" / "TensileLibrary_A.dat").write_bytes(b"x")

        assert env_mod._kernel_db_filename_fingerprint(
            one
        ) != env_mod._kernel_db_filename_fingerprint(two)

    def test_a_flat_layout_digest_is_unchanged_by_the_nested_branch(
        self, tmp_path: Path
    ):
        """Classic installs must see no fingerprint churn from this feature.

        Pinned by construction rather than by a recorded literal: a flat
        directory and the same directory with an *empty* gfx subdirectory must
        agree, because the nested branch contributes no names.
        """
        flat = tmp_path / "flat" / "library"
        flat.mkdir(parents=True)
        (flat / "TensileLibrary_A.dat").write_bytes(b"x")

        with_empty_arch = tmp_path / "mixed" / "library"
        with_empty_arch.mkdir(parents=True)
        (with_empty_arch / "TensileLibrary_A.dat").write_bytes(b"x")
        (with_empty_arch / "gfx950").mkdir()

        assert env_mod._kernel_db_filename_fingerprint(
            flat
        ) == env_mod._kernel_db_filename_fingerprint(with_empty_arch)

    def test_non_gfx_subdirectories_are_ignored(self, tmp_path: Path):
        """A packager's cache dir alongside the DB must not enter the digest."""
        library = tmp_path / "library"
        library.mkdir()
        (library / "TensileLibrary_A.dat").write_bytes(b"x")
        baseline = env_mod._kernel_db_filename_fingerprint(library)

        noise = library / "__pycache__"
        noise.mkdir()
        (noise / "TensileLibrary_Z.dat").write_bytes(b"z")
        assert env_mod._kernel_db_filename_fingerprint(library) == baseline

    def test_recursion_stops_at_one_level(self, tmp_path: Path):
        library = tmp_path / "library"
        deep = library / "gfx950" / "nested"
        deep.mkdir(parents=True)
        (deep / "TensileLibrary_A.dat").write_bytes(b"x")
        assert env_mod._kernel_db_filename_fingerprint(library) is None


class TestHipblasltBlockShape:
    def test_applied_prs_is_empty_dict_initially(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_hipblaslt(reasons)
        assert block["applied_prs"] == {}

    def test_block_keys_stable(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_hipblaslt(reasons)
        assert set(block.keys()) == {
            "rocm_release_tweak",
            "package_version",
            "lib_hash",
            "kernel_db_revision",
            "upstream_commit",
            "upstream_commit_matches_tweak",
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
        tweak_reason = next(
            r for r in reasons if r.startswith("hipblaslt.rocm_release_tweak")
        )
        assert "not readable" in tweak_reason

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
        # rocm_release_tweak failed but package_version succeeded
        assert block["rocm_release_tweak"] is None
        assert block["package_version"] == "1.2.0"
        tweak_reason = next(
            r for r in reasons if r.startswith("hipblaslt.rocm_release_tweak")
        )
        assert "not readable" not in tweak_reason
        assert "HIPBLASLT_VERSION_TWEAK" in tweak_reason


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
# probe_namespace (schema 1.12)
# ---------------------------------------------------------------------------


class TestCaptureProbeNamespace:
    @staticmethod
    def _set_boot_id(monkeypatch, tmp_path, value="boot-aaaa-bbbb"):
        boot_file = tmp_path / "boot_id"
        boot_file.write_text(f"{value}\n")
        monkeypatch.setattr(env_mod, "BOOT_ID_FILE", boot_file)

    def _ns_link(self, tmp_path, target, name):
        # os.readlink() returns the target string even for a dangling symlink,
        # so the target need not exist -- that is intentional here.
        ns_link = tmp_path / name
        ns_link.symlink_to(target)
        return ns_link

    def test_returns_boot_salted_mnt_digest_when_proc_ns_readable(
        self, monkeypatch, tmp_path
    ):
        self._set_boot_id(monkeypatch, tmp_path)
        mount_link = self._ns_link(
            tmp_path, "mnt:[4026531840]", "self_mnt_ns"
        )
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", mount_link)
        reasons = []
        result = env_mod._capture_probe_namespace(reasons)
        # Salted + hashed, NOT the raw inode token.
        assert result is not None
        assert result.startswith("mnt:")
        assert len(result) == len("mnt:") + 16
        assert "4026531840" not in result
        assert reasons == []

    def test_same_ns_token_different_boot_ids_differ(self, monkeypatch, tmp_path):
        """Boot scope prevents cross-boot collisions for the same inode token."""
        link = self._ns_link(tmp_path, "mnt:[4026531840]", "self_mnt_ns")
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", link)

        self._set_boot_id(monkeypatch, tmp_path, "boot-host-A")
        result_a = env_mod._capture_probe_namespace([])
        self._set_boot_id(monkeypatch, tmp_path, "boot-host-B")
        result_b = env_mod._capture_probe_namespace([])

        assert result_a != result_b
        # Same token + boot is a stable observation, but equality remains
        # advisory because Linux can recycle namespace inode numbers.
        self._set_boot_id(monkeypatch, tmp_path, "boot-host-A")
        assert env_mod._capture_probe_namespace([]) == result_a

    def test_local_only_marker_hashes_token_and_records_partial_reason(
        self, monkeypatch, tmp_path
    ):
        """No boot_id emits a hashed local-only value, never the raw token."""
        monkeypatch.setattr(env_mod, "BOOT_ID_FILE", tmp_path / "no_boot_id")
        mount_link = self._ns_link(
            tmp_path, "mnt:[4026531840]", "self_mnt_ns"
        )
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", mount_link)
        reasons = []
        result = env_mod._capture_probe_namespace(reasons)
        assert result is not None
        assert result.startswith("mnt-local:")
        assert len(result) == len("mnt-local:") + 16
        assert "4026531840" not in result
        assert any(r.startswith("probe_namespace.boot_id:") for r in reasons)

    def test_falls_back_to_cgroup_namespace_handle_when_mnt_unreadable(
        self, monkeypatch, tmp_path
    ):
        self._set_boot_id(monkeypatch, tmp_path)
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", tmp_path / "nonexistent_ns")
        cgroup_link = self._ns_link(
            tmp_path, "cgroup:[4026533001]", "self_cgroup_ns"
        )
        monkeypatch.setattr(env_mod, "SELF_CGROUP_NS", cgroup_link)
        reasons = []
        result = env_mod._capture_probe_namespace(reasons)
        assert result is not None
        assert result.startswith("cgroup-ns:")
        assert len(result) == len("cgroup-ns:") + 16
        assert "4026533001" not in result
        assert any(
            r.startswith("probe_namespace.mount_namespace:") for r in reasons
        )

    def test_distinct_cgroup_namespace_handles_do_not_collapse(
        self, monkeypatch, tmp_path
    ):
        """Unlike /proc/self/cgroup='0::/', namespace handles distinguish peers."""
        self._set_boot_id(monkeypatch, tmp_path)
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", tmp_path / "nonexistent_ns")
        cgroup_a = self._ns_link(
            tmp_path, "cgroup:[4026533001]", "self_cgroup_ns_a"
        )
        cgroup_b = self._ns_link(
            tmp_path, "cgroup:[4026533002]", "self_cgroup_ns_b"
        )
        monkeypatch.setattr(env_mod, "SELF_CGROUP_NS", cgroup_a)
        result_a = env_mod._capture_probe_namespace([])
        monkeypatch.setattr(env_mod, "SELF_CGROUP_NS", cgroup_b)
        result_b = env_mod._capture_probe_namespace([])
        assert result_a != result_b

    def test_cgroup_fallback_without_boot_id_is_hashed_and_local(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(env_mod, "BOOT_ID_FILE", tmp_path / "no_boot_id")
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", tmp_path / "nonexistent_ns")
        cgroup_link = self._ns_link(
            tmp_path, "cgroup:[4026533001]", "self_cgroup_ns"
        )
        monkeypatch.setattr(env_mod, "SELF_CGROUP_NS", cgroup_link)
        reasons = []
        result = env_mod._capture_probe_namespace(reasons)
        assert result is not None
        assert result.startswith("cgroup-ns-local:")
        assert len(result) == len("cgroup-ns-local:") + 16
        assert "4026533001" not in result
        assert any(r.startswith("probe_namespace.boot_id:") for r in reasons)
        assert any(
            r.startswith("probe_namespace.mount_namespace:") for r in reasons
        )

    def test_returns_none_when_all_sources_fail(self, monkeypatch, tmp_path):
        self._set_boot_id(monkeypatch, tmp_path)
        monkeypatch.setattr(env_mod, "SELF_MNT_NS", tmp_path / "nonexistent_ns")
        monkeypatch.setattr(
            env_mod, "SELF_CGROUP_NS", tmp_path / "nonexistent_cgroup_ns"
        )
        reasons = []
        result = env_mod._capture_probe_namespace(reasons)
        assert result is None
        assert any(
            r.startswith("probe_namespace.mount_namespace:") for r in reasons
        )
        assert any(
            r.startswith("probe_namespace.cgroup_namespace:") for r in reasons
        )

    def test_safe_wrapper_contains_unexpected_failure(self, monkeypatch):
        def boom(reasons):
            raise RuntimeError("forced namespace failure")

        monkeypatch.setattr(env_mod, "_capture_probe_namespace", boom)
        reasons = []
        assert env_mod._capture_probe_namespace_safe(reasons) is None
        assert any("unexpected capture failure" in r for r in reasons)

    def test_probe_namespace_in_collect_env_output(self, all_disabled):
        snap = collect_env()
        d = snap.to_dict()
        assert "probe_namespace" in d
        assert d["probe_namespace"] is None
        assert any(
            r.startswith("probe_namespace.mount_namespace:")
            for r in snap.partial_reasons
        )
        assert any(
            r.startswith("probe_namespace.cgroup_namespace:")
            for r in snap.partial_reasons
        )


# ---------------------------------------------------------------------------
# container_detected + execution_context (schema 1.11)
# ---------------------------------------------------------------------------


class TestContainerDetected:
    """_detect_container_detected(): runtime-agnostic isolation smoke test."""

    def test_baremetal_no_signal_is_false(self, all_disabled, monkeypatch):
        # No named runtime, no markers, and mount-ns check disabled ->
        # the honest "no isolation observed" answer.
        monkeypatch.setattr(
            env_mod, "_mount_namespace_differs_from_init", lambda: False
        )
        assert env_mod._detect_container_detected() is False

    def test_named_runtime_implies_true(self, all_disabled, tmp_path, monkeypatch):
        # docker marker -> _detect_container_type() != baremetal -> True.
        marker = tmp_path / ".dockerenv"
        marker.write_text("")
        monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", marker)
        monkeypatch.setattr(
            env_mod, "_mount_namespace_differs_from_init", lambda: False
        )
        assert env_mod._detect_container_detected() is True

    def test_private_mount_namespace_implies_true(
        self, all_disabled, monkeypatch
    ):
        # No named runtime, but a private mount namespace -> True. This is
        # the RE-worker / stripped-sandbox case runtime_context.type misses.
        monkeypatch.setattr(
            env_mod, "_mount_namespace_differs_from_init", lambda: True
        )
        assert env_mod._detect_container_detected() is True

    def test_kubepods_cgroup_token_implies_true(
        self, all_disabled, tmp_path, monkeypatch
    ):
        # A k8s pod: runtime_context.type falls through to baremetal (no
        # docker/podman/singularity), but the cgroup carries a kubepods
        # token -> container_detected True. The core false-negative fix.
        self_cgroup = tmp_path / "self_cgroup"
        self_cgroup.write_text("0::/kubepods.slice/kubepods-besteffort-pod123.slice\n")
        monkeypatch.setattr(env_mod, "SELF_CGROUP_FILE", self_cgroup)
        monkeypatch.setattr(
            env_mod, "_mount_namespace_differs_from_init", lambda: False
        )
        # type still reads baremetal (no named runtime)...
        assert env_mod._detect_container_type() == "baremetal"
        # ...but the isolation smoke test correctly fires.
        assert env_mod._detect_container_detected() is True

    def test_safe_wrapper_swallows_exceptions(self, monkeypatch):
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(env_mod, "_detect_container_detected", boom)
        # The disaster-path wrapper must never raise.
        assert env_mod._detect_container_detected_safe() is False


class TestExecutionContext:
    """execution_context.probe_invocation stamping via collect_env()."""

    def test_default_is_direct(self, all_disabled):
        snap = collect_env()
        assert snap.execution_context["probe_invocation"] == "direct"
        assert snap.execution_context["likely_execution_platform"] is None

    def test_valid_label_is_stamped(self, all_disabled):
        snap = collect_env(probe_invocation="buck2_action")
        assert snap.execution_context["probe_invocation"] == "buck2_action"

    def test_unknown_label_falls_back_to_direct_with_reason(self, all_disabled):
        snap = collect_env(probe_invocation="nonsense")
        assert snap.execution_context["probe_invocation"] == "direct"
        assert snap.partial is True
        assert any(
            r.startswith("execution_context.probe_invocation")
            for r in snap.partial_reasons
        )

    def test_warning_predicate_direct_never_warns(self, all_disabled):
        # Shared helper used by BOTH the CLI and _probe_main.
        assert env_mod.execution_context_warning("direct", False) is None
        assert env_mod.execution_context_warning("direct", True) is None

    def test_warning_predicate_claim_without_isolation_warns(self, all_disabled):
        msg = env_mod.execution_context_warning("buck2_action", False)
        assert msg is not None and msg.startswith("WARNING:")

    def test_buck2_run_without_isolation_is_labeled_client_host(
        self, all_disabled
    ):
        msg = env_mod.execution_context_warning("buck2_run", False)
        assert msg is not None and msg.startswith("NOTICE:")
        assert "client-host snapshot" in msg
        assert "remote-worker" in msg

    def test_warning_predicate_suppressed_by_container_detected(
        self, all_disabled
    ):
        assert env_mod.execution_context_warning("buck2_action", True) is None

    def test_warning_predicate_suppressed_by_image_env(
        self, all_disabled, monkeypatch
    ):
        monkeypatch.setenv("AORTA_DOCKER_IMAGE", "img:tag")
        assert env_mod.execution_context_warning("buck2_action", False) is None
        monkeypatch.delenv("AORTA_DOCKER_IMAGE", raising=False)
        monkeypatch.setenv("AORTA_RE_IMAGE", "re:tag")
        assert env_mod.execution_context_warning("buck2_action", False) is None

    def test_disaster_snapshot_preserves_probe_invocation(
        self, all_disabled, monkeypatch
    ):
        # If an internal probe crashes, collect_env falls back to the
        # disaster snapshot -- which must still record the caller's
        # probe_invocation rather than silently relabelling it "direct".
        def boom(*args, **kwargs):
            raise RuntimeError("forced probe crash")

        # _run_rdhc runs early in the probe body; make it explode so the
        # top-level guard builds a disaster snapshot.
        monkeypatch.setattr(env_mod, "_run_rdhc", boom)
        snap = collect_env(probe_invocation="buck2_action")
        assert snap.partial is True
        assert snap.execution_context["probe_invocation"] == "buck2_action"
        # And a garbage label in the crash path still normalizes to direct.
        snap2 = collect_env(probe_invocation="nonsense")
        assert snap2.execution_context["probe_invocation"] == "direct"


class TestProbeMainExecutionContext:
    """_probe_main mirrors the CLI --execution-context flag + warning."""

    @staticmethod
    def _probe_main_mod():
        # Load by file path (like _cli_mod above) so these tests pass from a
        # clean checkout without aorta installed / PYTHONPATH set -- the rest
        # of this module deliberately loads environment.py by path too.
        probe_main_path = Path(env_mod.__file__).parent / "_probe_main.py"
        spec = importlib.util.spec_from_file_location(
            "aorta.instrumentation._probe_main", probe_main_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def _write(self, tmp_path, argv_extra, monkeypatch, container=False):
        monkeypatch.setattr(
            env_mod, "_detect_container_detected", lambda: container
        )
        probe_main = self._probe_main_mod()
        out = str(tmp_path / "env.json")
        rc = probe_main.main(["_probe_main", *argv_extra, out])
        return rc, out

    def test_stamps_probe_invocation(self, all_disabled, tmp_path, monkeypatch):
        rc, out = self._write(
            tmp_path, ["--execution-context", "buck2_action"], monkeypatch
        )
        assert rc == 0
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
        assert doc["execution_context"]["probe_invocation"] == "buck2_action"

    def test_warns_on_claim_without_isolation(
        self, all_disabled, tmp_path, monkeypatch, capsys
    ):
        # The dependency-free entry point must emit the same claim-vs-reality
        # warning as the Click CLI -- it is the one most likely used inside a
        # Buck2 action / container.
        rc, _ = self._write(
            tmp_path, ["--execution-context", "buck2_action"], monkeypatch,
            container=False,
        )
        assert rc == 0
        assert "WARNING" in capsys.readouterr().err

    def test_no_warning_when_container_detected(
        self, all_disabled, tmp_path, monkeypatch, capsys
    ):
        rc, _ = self._write(
            tmp_path, ["--execution-context", "buck2_action"], monkeypatch,
            container=True,
        )
        assert rc == 0
        assert "WARNING" not in capsys.readouterr().err

    def test_no_warning_for_direct(
        self, all_disabled, tmp_path, monkeypatch, capsys
    ):
        rc, _ = self._write(tmp_path, [], monkeypatch, container=False)
        assert rc == 0
        assert "WARNING" not in capsys.readouterr().err

    def test_no_warning_when_re_image_set(
        self, all_disabled, tmp_path, monkeypatch, capsys
    ):
        # $AORTA_RE_IMAGE (the phase-2 launcher convention) suppresses the
        # warning the same way $AORTA_DOCKER_IMAGE does.
        monkeypatch.setenv("AORTA_RE_IMAGE", "re-worker-image:tag")
        rc, _ = self._write(
            tmp_path, ["--execution-context", "buck2_action"], monkeypatch,
            container=False,
        )
        assert rc == 0
        assert "WARNING" not in capsys.readouterr().err

    def test_missing_flag_value_is_hard_error_not_silent_stdout(
        self, all_disabled, tmp_path, monkeypatch, capsys
    ):
        # Forgotten value: `--execution-context <outpath>` must NOT eat the
        # output path as the label and dump JSON to stdout while exiting 0.
        monkeypatch.setattr(
            env_mod, "_detect_container_detected", lambda: False
        )
        probe_main = self._probe_main_mod()
        out = tmp_path / "env.json"
        # Only the flag + the intended output path; the path is not a valid
        # label, so the parser must reject rather than consume it.
        rc = probe_main.main(["_probe_main", "--execution-context", str(out)])
        captured = capsys.readouterr()
        assert rc == 2
        assert "--execution-context requires one of" in captured.err
        # The artifact must NOT have been written, and JSON must NOT have
        # been dumped to stdout under the guise of success.
        assert not out.exists()
        assert "schema_version" not in captured.out

    def test_invalid_flag_value_is_hard_error(
        self, all_disabled, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            env_mod, "_detect_container_detected", lambda: False
        )
        probe_main = self._probe_main_mod()
        out = tmp_path / "env.json"
        rc = probe_main.main(
            ["_probe_main", "--execution-context", "nonsense", str(out)]
        )
        assert rc == 2
        assert "--execution-context requires one of" in capsys.readouterr().err
        assert not out.exists()


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
        # Change detector, and only that: changing the captured set is a schema
        # change, so a future PR has to acknowledge it here.
        #
        # What this test canNOT do -- and must not be read as doing -- is show
        # that the pinned upstream libraries are fully covered. It compares the
        # generated list against a second hand-written list, so it only catches
        # drift between the two. Coverage of the installed libraries is measured
        # by scripts/audit_env_knobs.py, whose uncovered-knob direction is the
        # one no hand-written list can check.
        assert set(CANONICAL_ENV_VARS) == {
            # GPU scoping
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            # HSA / runtime
            "HSA_XNACK",
            "HSA_KERNARG_POOL_SIZE",
            "HSA_NO_SCRATCH_RECLAIM",
            "HSA_OVERRIDE_GFX_VERSION",
            "HSA_TOOLS_DISABLE_REGISTER",
            # GPU queue / codegen / build target
            "GPU_MAX_HW_QUEUES",
            "AMDGCN_USE_BUFFER_OPS",
            "DISABLE_TF32",
            "PYTORCH_ROCM_ARCH",
            "HIP_LAUNCH_BLOCKING",
            # RCCL / NCCL
            "NCCL_MAX_NCHANNELS",
            "NCCL_P2P_LEVEL",
            "NCCL_IB_HCA",
            "NCCL_SOCKET_IFNAME",
            "RCCL_MSCCL_ENABLE",
            # AINIC (AMD-Pensando RoCE NIC) net-plugin + fabric tuning
            "RCCL_AINIC_ROCE",
            "NCCL_NET_PLUGIN",
            "NCCL_NET",
            "RCCL_CTS_OFFLOAD_ENABLED",
            "NCCL_IB_GID_INDEX",
            "NCCL_IB_ROCE_VERSION_NUM",
            "NCCL_IB_TC",
            "NCCL_IB_FIFO_TC",
            "NCCL_GDR_FLUSH_DISABLE",
            "NCCL_GDRCOPY_ENABLE",
            "NCCL_IB_USE_INLINE",
            "NCCL_IB_PCI_RELAXED_ORDERING",
            "NCCL_IB_QPS_PER_CONNECTION",
            "NCCL_PXN_DISABLE",
            "NCCL_IGNORE_CPU_AFFINITY",
            "NCCL_NET_OPTIONAL_RECV_COMPLETION",
            "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING",
            "NCCL_IB_TIMEOUT",
            "NCCL_IB_SL",
            "NCCL_IB_SPLIT_DATA_ON_QPS",
            "NCCL_DMABUF_ENABLE",
            "NCCL_CUMEM_ENABLE",
            "IONIC_LOCKFREE",
            "RCCL_DISABLE_RAIL_TREES",
            "RCCL_LL128_FORCE_ENABLE",
            "NCCL_WORK_FIFO_BYTES",
            # gfx950 fence-ordering debug knob (SDC investigation)
            "RCCL_GFX9_CHEAP_FENCE_OFF",
            # FBGEMM
            "FBGEMM_NO_JK",
            "FBGEMM_TBE_V2",
            "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL",
            "FBGEMM_BOUNDS_CHECK_INDICES_V2",
            # MIOpen
            "MIOPEN_SYSTEM_DB_PATH",
            "MIOPEN_USER_DB_PATH",
            "MIOPEN_DEBUG_DISABLE_FIND_DB",
            "MIOPEN_FIND_MODE",
            # SDPA / Flash Attention backend selection
            # Note: USE_ROCM_CK_SDPA / USE_ROCM_CK_GEMM are NOT here --
            # they're build-time cmake flags, captured under
            # composable_kernel.{pytorch_use_ck_sdpa,pytorch_use_ck_gemm}
            "TORCH_ROCM_FA_PREFER_CK",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
            # GEMM backend preference + autotune pinning
            "TORCH_BLAS_PREFER_HIPBLASLT",
            "TORCH_HIPBLASLT_TUNING_FILE",
            "TORCH_HIPBLASLT_TUNING_OVERRIDE_FILE",
            # PyTorch / inductor
            "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
            "PYTORCH_CUDA_ALLOC_CONF",
            # Dynamic loader
            "LD_LIBRARY_PATH",
            # hipBLASLt / rocBLAS / Tensile GEMM numeric + kernel-path selection
            # (MI350X recom-repro NaN escalation). Capture is comprehensive:
            # behavior-changing and diagnostic/report-only knobs are both
            # retained, with the distinction recorded in the registry.
            # Library / ExtOp loading
            "HIPBLASLT_TENSILE_LIBPATH",
            "HIPBLASLT_EXT_OP_LIBRARY_PATH",
            "HIPBLASLT_PRELOAD_KERNELS",
            "TENSILE_MAX_DECOMPRESSED_BYTES",
            "ROCBLAS_TENSILE_LIBPATH",
            "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH",
            # Backend routing + generator choice
            "ROCBLAS_USE_HIPBLASLT",
            "ROCBLAS_USE_HIPBLASLT_BATCHED",
            "HIPBLASLT_USE_ROCROLLER",
            "HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL",
            "HIPBLASLT_TUNING_OVERRIDE_FILE",
            # Numeric path
            "HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32",
            "ROCBLAS_DEFAULT_ATOMICS_MODE",
            "ROCBLAS_INTERNAL_FP16_ALT_IMPL",
            "ROCBLAS_INTERNAL_FP16_ALT_IMPL_RNZ",
            "ROCBLAS_INTERNAL_FORCE_VALU_FOR_DGEMM",
            # Allocator + workspace sizing
            "ROCBLAS_STREAM_ORDER_ALLOC",
            "ROCBLAS_DEVICE_MEMORY_SIZE",
            "ROCBLAS_INTERNAL_TRSM_REG_KERNEL_MEM_LIMIT",
            # Solution selection
            "TENSILE_SOLUTION_INDEX",
            "TENSILE_SOLUTION_SELECTION_METHOD",
            "TENSILE_EXPERIMENTAL_SELECTION",
            "TENSILE_TAM_SELECTION_ENABLE",
            "TENSILE_NAIVE_SEARCH",
            "TENSILE_METRIC",
            "TENSILE_PREDICTION_LIB",
            "TENSILE_GRIDBASED_KDTREE",
            "TENSILE_GRIDBASED_BATCH_EXP",
            "ANALYTICAL_GEMM_HEURISTICS",
            "ANALYTICAL_GEMM_HEURISTICS_VARIANCE",
            "GRIDBASED_TOPSOLS",
            # Stream-K launch geometry
            "TENSILE_STREAMK_DYNAMIC_GRID",
            "TENSILE_STREAMK_FIXED_GRID",
            "TENSILE_STREAMK_MAX_CUS",
            "TENSILE_STREAMK_DATA_PARALLEL",
            "TENSILE_STREAMK_DYNAMIC_WGM",
            "TENSILE_STREAMK_FULL_TILES",
            "TENSILE_STREAMK_GRID_MULTIPLIER",
            # Workgroup mapping + StaggerU
            "TENSILE_FIXED_WGM",
            "TENSILE_FIXED_WGMXCC",
            "TENSILE_FIXED_WGMXCCCHUNK",
            "TENSILE_DISABLE_STAGGERU",
            "TENSILE_FIXED_STAGGERU",
            "TENSILE_FIXED_STAGGERU_MAPPING",
            "TENSILE_FIXED_STAGGERU_STRIDE_SHIFT",
            # Debug bits that skip work
            "TENSILE_DB2",
            # Forward-compat (absent from the 1.4.0 .so)
            "TENSILE_STREAMK5_FORCE_MODE",
            "TENSILE_STREAMK_TILES",
            "TENSILE_ADAPTIVE_GEMM_NTAB_ALGO",
            "TENSILE_STREAMK_SPLIT",
            # In-library numeric checking
            "HIPBLASLT_CHECK_NUMERICS",
            "HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY",
            "HIPBLASLT_CHECK_NUMERICS_SCAN_FROM",
            "HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL",
            "HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST",
            "ROCBLAS_CHECK_NUMERICS",
            # Diagnostics / reporting. Captured with their consumer recorded
            # (category="gemm_diagnostics"), not excluded: recording a value
            # never claims it changed execution.
            "ANALYTICAL_GEMM_DEBUG",
            "ORIGAMI_LOG_FILE",
            "HIPBLASLT_LOG_FILE",
            "HIPBLASLT_LOG_LEVEL",
            "HIPBLASLT_LOG_MASK",
            "HIPBLASLT_BENCH_PERF",
            "HIPBLASLT_BENCH_PERF_ALL",
            "HIPBLASLT_BENCH_PRINT_COMMAND",
            "HIPBLASLT_ENABLE_MARKER",
            "TENSILE_ENABLE_MARKER",
            "ROCBLAS_LAYER",
            "ROCBLAS_LOG_PATH",
            "ROCBLAS_LOG_TRACE_PATH",
            "ROCBLAS_LOG_BENCH_PATH",
            "ROCBLAS_LOG_PROFILE_PATH",
            "ROCBLAS_VERBOSE_HIPBLASLT_ERROR",
            "ROCBLAS_VERBOSE_TENSILE_ERROR",
            "TENSILE_DB",
            "TENSILE_ADAPTIVE_GEMM_LOG",
            "TENSILE_AUTO_GSU_ALGO",
            "TENSILE_SOLUTION_SELECTION_TRACE",
            "TENSILE_BENCHMARK",
        }

    def test_report_only_gemm_knobs_are_captured_and_classified(self):
        """Report-only knobs are CAPTURED, with the classification recorded.

        Earlier 1.15 drafts excluded these because their only consumer is a print
        or a client-side report. That made the snapshot's contents depend on our
        classification being correct -- and the 2026-08-02 audit overturned two
        of its own name-based verdicts, so the judgement was in the wrong place.
        Capture is now comprehensive and the judgement lives in ``category`` /
        ``consumer``, where a triager can see it without it deciding what the
        snapshot preserves."""
        for name in (
            "ANALYTICAL_GEMM_DEBUG",
            "ORIGAMI_LOG_FILE",
            "HIPBLASLT_LOG_FILE",
            "HIPBLASLT_LOG_LEVEL",
            "HIPBLASLT_LOG_MASK",
            "HIPBLASLT_BENCH_PERF",
            "HIPBLASLT_BENCH_PERF_ALL",
            "HIPBLASLT_BENCH_PRINT_COMMAND",
            "HIPBLASLT_ENABLE_MARKER",
            "TENSILE_ENABLE_MARKER",
            "ROCBLAS_LAYER",
            "ROCBLAS_LOG_PATH",
            "ROCBLAS_LOG_TRACE_PATH",
            "ROCBLAS_LOG_BENCH_PATH",
            "ROCBLAS_LOG_PROFILE_PATH",
            "ROCBLAS_VERBOSE_HIPBLASLT_ERROR",
            "ROCBLAS_VERBOSE_TENSILE_ERROR",
            "TENSILE_DB",
            "TENSILE_ADAPTIVE_GEMM_LOG",
            "TENSILE_AUTO_GSU_ALGO",
            "TENSILE_SOLUTION_SELECTION_TRACE",
            "TENSILE_BENCHMARK",
        ):
            assert name in CANONICAL_ENV_VARS, f"{name} must be captured"
            knob = env_mod.ENV_KNOBS_BY_NAME[name]
            assert knob.category == "gemm_diagnostics", (name, knob.category)

    def test_tensile_db2_is_not_classified_as_diagnostics(self):
        """The sharpest classification case, kept as a test now that it no longer
        decides capture: ``TENSILE_DB``'s bits are all prints, while
        ``TENSILE_DB2``'s low bits gate skipKernelLaunch /
        skipInitKernelLaunch -- so DB2 skips work and is categorised as such."""
        assert env_mod.ENV_KNOBS_BY_NAME["TENSILE_DB"].category == "gemm_diagnostics"
        assert env_mod.ENV_KNOBS_BY_NAME["TENSILE_DB2"].category == "gemm_skip_work"


class TestEnvKnobRegistry:
    """The manifest that ``CANONICAL_ENV_VARS`` is generated from."""

    def test_canonical_env_vars_is_generated_from_the_registry(self):
        assert CANONICAL_ENV_VARS == tuple(k.name for k in env_mod.ENV_KNOB_REGISTRY)
        assert len(set(CANONICAL_ENV_VARS)) == len(CANONICAL_ENV_VARS), "duplicate knob name"
        assert set(env_mod.ENV_KNOBS_BY_NAME) == set(CANONICAL_ENV_VARS)

    def test_every_registered_value_is_preserved_verbatim(self, isolated_env):
        """Review's suggested test: drive the capture from the registry itself.

        Every registered knob is exported with a distinct sentinel and must come
        back byte-identical -- so a knob can never be listed in the manifest yet
        silently dropped, mangled, or deduplicated on the way into a snapshot."""
        expected = {}
        for i, knob in enumerate(env_mod.ENV_KNOB_REGISTRY):
            value = f"sentinel-{i}-{knob.category}"
            isolated_env.setenv(knob.name, value)
            expected[knob.name] = value

        captured = env_mod._capture_env_vars()

        assert captured == expected

    def test_every_knob_carries_a_classification_and_a_provenance(self):
        for knob in env_mod.ENV_KNOB_REGISTRY:
            assert knob.library.strip(), knob.name
            assert knob.consumer.strip(), knob.name
            assert knob.source_reference.strip(), knob.name
            assert knob.reference_build.strip(), knob.name
            assert knob.category in env_knobs.CATEGORIES, (knob.name, knob.category)

    def test_gemm_knob_library_attribution_matches_its_prefix(self):
        """The GEMM knobs' ``library`` is measured from the shipped libraries, so
        it must at least agree with the owning project implied by the name -- a
        TENSILE_ knob can live in either or both, but a HIPBLASLT_ knob claiming
        rocblas-only would mean the manifest and the audit disagree."""
        for knob in env_mod.ENV_KNOB_REGISTRY:
            if knob.name.startswith("HIPBLASLT_"):
                assert "hipblaslt" in knob.library, (knob.name, knob.library)
            elif knob.name.startswith("ROCBLAS_"):
                assert "rocblas" in knob.library, (knob.name, knob.library)

    def test_forward_compat_knobs_are_marked_absent_from_the_reference_build(self):
        """A knob the reference build does not ship is recorded as such rather
        than implying it was verified there. This is the honesty half of review
        2: capture does not claim library support."""
        for name in (
            "TENSILE_STREAMK5_FORCE_MODE",
            "TENSILE_STREAMK_TILES",
            "TENSILE_STREAMK_SPLIT",
            "TENSILE_ADAPTIVE_GEMM_NTAB_ALGO",
            "HIPBLASLT_BENCH_PERF_ALL",
        ):
            knob = env_mod.ENV_KNOBS_BY_NAME[name]
            assert knob.source_reference == env_knobs.ABSENT_FROM_REFERENCE_BUILD, name

    def test_inherited_knobs_are_marked_unaudited_not_given_a_false_source(self):
        """Knobs inherited from schema <= 1.14 were never traced to a call site.
        The manifest says so instead of inventing a source location -- the audit
        debt is visible and countable."""
        inherited = [
            k
            for k in env_mod.ENV_KNOB_REGISTRY
            if k.source_reference == env_knobs.INHERITED_UNAUDITED
        ]
        assert inherited, "expected the pre-1.15 knobs to be marked unaudited"
        for knob in inherited:
            assert not knob.name.startswith(("HIPBLASLT_", "ROCBLAS_", "TENSILE_")), (
                f"{knob.name} is a GEMM knob and should carry a measured provenance"
            )

    def test_unset_knob_is_null_and_does_not_make_the_snapshot_partial(self, isolated_env):
        """Review 2's distinction, asserted: ``null`` means UNSET in this
        process. It is not a statement about library support, and it never
        contributes a partial reason."""
        captured = env_mod._capture_env_vars()
        assert all(v is None for v in captured.values())

    def test_a_knob_absent_from_the_reference_library_is_still_serialized(
        self, all_disabled, tmp_path
    ):
        """Review 2's suggested test: export a forward-compatible variable that
        the reference build does not ship, and it is still recorded. Capture
        reflects the declared environment, not what the library supports."""
        knob = env_mod.ENV_KNOBS_BY_NAME["TENSILE_STREAMK5_FORCE_MODE"]
        assert knob.source_reference == env_knobs.ABSENT_FROM_REFERENCE_BUILD
        all_disabled.setenv("TENSILE_STREAMK5_FORCE_MODE", "1")
        output = tmp_path / "env.json"

        snapshot = env_mod.capture_to(output)
        as_json = json.loads(output.read_text())

        assert snapshot.env_vars["TENSILE_STREAMK5_FORCE_MODE"] == "1"
        assert as_json["env_vars"]["TENSILE_STREAMK5_FORCE_MODE"] == "1"
        assert EnvSnapshot.from_dict(as_json).env_vars["TENSILE_STREAMK5_FORCE_MODE"] == "1"


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
            "HIPBLASLT_TENSILE_DIR",
            "ROCBLAS_VERSION_HEADER", "ROCBLAS_LIB_DIR", "ROCBLAS_TENSILE_DIR",
            "CK_VERSION_HEADER", "CK_TILE_CONFIG_HEADER",
            "MIOPEN_VERSION_HEADER", "MIOPEN_LIB_DIR", "MIOPEN_KERNEL_DB_DIR",
            "RCCL_VERSION_HEADER", "RCCL_LIB_DIR",
            "DOCKERENV_MARKER",
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


# ---------------------------------------------------------------------------
# rocBLAS introspection -- mirrors the hipBLASLt block 1:1
# ---------------------------------------------------------------------------


class TestRocblasHeaderParsing:
    def test_parse_full_header(self):
        text = """
        #ifndef _ROCBLAS_VERSION_H_
        #define _ROCBLAS_VERSION_H_
        #define ROCBLAS_VERSION_MAJOR     5
        #define ROCBLAS_VERSION_MINOR     2
        #define ROCBLAS_VERSION_PATCH     0
        #define ROCBLAS_VERSION_TWEAK     dabb6df2b9
        #endif
        """
        commit, version = env_mod._parse_version_header(
            text, env_mod._ROCBLAS_TWEAK_RE, env_mod._ROCBLAS_VERSION_RE
        )
        assert commit == "dabb6df2b9"
        assert version == "5.2.0"

    def test_parse_missing_tweak_returns_none_commit(self):
        text = """
        #define ROCBLAS_VERSION_MAJOR 5
        #define ROCBLAS_VERSION_MINOR 2
        #define ROCBLAS_VERSION_PATCH 0
        """
        commit, version = env_mod._parse_version_header(
            text, env_mod._ROCBLAS_TWEAK_RE, env_mod._ROCBLAS_VERSION_RE
        )
        assert commit is None
        assert version == "5.2.0"

    def test_parse_empty_returns_none_pair(self):
        commit, version = env_mod._parse_version_header(
            "", env_mod._ROCBLAS_TWEAK_RE, env_mod._ROCBLAS_VERSION_RE
        )
        assert (commit, version) == (None, None)


class TestRocblasLibHash:
    def test_hash_resolved_through_symlink(self, tmp_path: Path, monkeypatch):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        real = lib_dir / "librocblas.so.5.2.70201"
        real.write_bytes(b"hello rocblas")
        symlink_a = lib_dir / "librocblas.so.5"
        symlink_b = lib_dir / "librocblas.so"
        symlink_a.symlink_to(real.name)
        symlink_b.symlink_to(symlink_a.name)
        monkeypatch.setattr(env_mod, "ROCBLAS_LIB_DIR", lib_dir)

        digest = env_mod._hash_shared_library(env_mod.ROCBLAS_LIB_DIR, "librocblas.so")
        expected = "sha256:" + hashlib.sha256(b"hello rocblas").hexdigest()
        assert digest == expected

    def test_no_library_returns_none(self, tmp_path: Path):
        digest = env_mod._hash_shared_library(tmp_path / "empty", "librocblas.so")
        assert digest is None

    def test_stripped_image_falls_back_to_versioned_filename(
        self, tmp_path: Path
    ):
        """Regression guard: stripped runtime images ship only the
        versioned ``libfoo.so.MAJOR.MINOR.PATCH`` (the SONAME-versioned
        ``.so.1`` symlink is created by ldconfig, the unversioned ``.so``
        is ``-dev``-only). The probe must still hash the actual file.
        """
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        # Only the versioned filename exists -- no unversioned `.so`,
        # no SONAME `.so.1`. This is what `dpkg -L librocblas0` ships
        # before ldconfig runs in a stripped image.
        real = lib_dir / "librocblas.so.5.2.70201"
        real.write_bytes(b"stripped-image bytes")

        digest = env_mod._hash_shared_library(lib_dir, "librocblas.so")
        expected = "sha256:" + hashlib.sha256(b"stripped-image bytes").hexdigest()
        assert digest == expected, (
            "stripped-image fallback failed: probe should hash "
            "librocblas.so.5.2.70201 when the unversioned .so symlink is "
            "missing (a -dev-only artifact)"
        )

    def test_picks_highest_versioned_when_multiple_present(
        self, tmp_path: Path
    ):
        """When several versioned files exist (e.g. mid-upgrade state or
        sideloaded debug build), pick the highest -- that's the file the
        SONAME would normally point at after ldconfig runs.
        """
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "librocblas.so.5.1.00000").write_bytes(b"old")
        (lib_dir / "librocblas.so.5.2.70201").write_bytes(b"new")

        digest = env_mod._hash_shared_library(lib_dir, "librocblas.so")
        expected = "sha256:" + hashlib.sha256(b"new").hexdigest()
        assert digest == expected

    def test_picks_numerically_highest_across_digit_boundary(
        self, tmp_path: Path
    ):
        """Regression guard: the version-suffix sort must be by integer
        tuple, not lexicographic. ``5.10.0`` is newer than ``5.9.0`` but
        sorts *before* it as a string (``"1" < "9"``), so a lex-sorted
        fallback would record the older file's hash on a multi-version
        install.
        """
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "librocblas.so.5.9.0").write_bytes(b"old-five-nine")
        (lib_dir / "librocblas.so.5.10.0").write_bytes(b"new-five-ten")

        digest = env_mod._hash_shared_library(lib_dir, "librocblas.so")
        expected = "sha256:" + hashlib.sha256(b"new-five-ten").hexdigest()
        wrong = "sha256:" + hashlib.sha256(b"old-five-nine").hexdigest()
        assert digest == expected, (
            "lib_hash describes the wrong file -- the integer-tuple vs "
            f"string-sort regression has reappeared. Expected {expected!r}, "
            f"got {digest!r}. If this is the wrong-side hash {wrong!r}, "
            "_hash_shared_library has reverted to lex-sorting its glob."
        )


class TestRocblasBlockShape:
    def test_block_keys_stable(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_rocblas(reasons)
        assert set(block.keys()) == {
            "rocm_release_tweak",
            "package_version",
            "lib_hash",
            "kernel_db_revision",
            "upstream_commit",
            "upstream_commit_matches_tweak",
            "applied_prs",
        }
        assert block["applied_prs"] == {}

    def test_partial_reasons_use_rocblas_prefix(self, all_disabled):
        reasons: list[str] = []
        env_mod._capture_rocblas(reasons)
        assert reasons, "expected reasons under all_disabled"
        assert all(r.startswith("rocblas.") for r in reasons), reasons

    def test_reason_distinguishes_unreadable_from_missing_define(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        header = tmp_path / "rocblas-version.h"
        header.write_text(
            "#define ROCBLAS_VERSION_MAJOR 5\n"
            "#define ROCBLAS_VERSION_MINOR 2\n"
            "#define ROCBLAS_VERSION_PATCH 0\n"
            # No TWEAK on purpose
        )
        monkeypatch.setattr(env_mod, "ROCBLAS_VERSION_HEADER", header)
        monkeypatch.setattr(env_mod, "ROCBLAS_LIB_DIR", tmp_path / "no_libs")
        monkeypatch.setattr(env_mod, "ROCBLAS_TENSILE_DIR", tmp_path / "no_tensile")
        reasons: list[str] = []
        block = env_mod._capture_rocblas(reasons)
        assert block["rocm_release_tweak"] is None
        assert block["package_version"] == "5.2.0"
        tweak_reason = next(
            r for r in reasons if r.startswith("rocblas.rocm_release_tweak")
        )
        assert "not readable" not in tweak_reason
        assert "ROCBLAS_VERSION_TWEAK" in tweak_reason


# ---------------------------------------------------------------------------
# Composable Kernel
# ---------------------------------------------------------------------------


class TestCKHeaderParsing:
    def test_parse_full_header(self):
        text = """
        #define CK_VERSION 1.2.0
        #define CK_VERSION_MAJOR 1
        #define CK_VERSION_MINOR 2
        #define CK_VERSION_PATCH 0
        #define CK_COMMIT_ID 23d531c8ae9721ac990116751542ab63e11d27c8
        """
        version, commit = env_mod._parse_ck_header(text)
        assert version == "1.2.0"
        # Full 40-char SHA preserved (CK uses long form, unlike hipblaslt's 7-12 short)
        assert commit == "23d531c8ae9721ac990116751542ab63e11d27c8"

    def test_parse_short_commit_still_accepted(self):
        text = "#define CK_COMMIT_ID abc1234\n"
        version, commit = env_mod._parse_ck_header(text)
        assert version is None
        assert commit == "abc1234"

    def test_parse_empty_returns_none_pair(self):
        assert env_mod._parse_ck_header("") == (None, None)
        assert env_mod._parse_ck_header(None) == (None, None)


class TestCKBlockShape:
    def test_block_has_two_subsections_plus_build_flags(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_composable_kernel(reasons)
        assert set(block.keys()) == {
            "system",
            "pytorch_bundled",
            "pytorch_use_ck_sdpa",
            "pytorch_use_ck_gemm",
        }
        assert set(block["system"].keys()) == {"version", "commit", "ck_tile_present"}
        assert set(block["pytorch_bundled"].keys()) == {"present", "symbol_count"}


class TestCKPytorchBuildFlags:
    """USE_ROCM_CK_SDPA / USE_ROCM_CK_GEMM are build-time cmake flags
    consumed when the wheel is built -- NOT runtime env vars. The
    composable_kernel block surfaces them via ``__config__.show()``
    parsing, exactly like the FBGEMM flags. Setting them at runtime
    in the workload's env does nothing.
    """

    def test_torch_absent_returns_null_pair_no_reason(
        self, isolated_env, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        sdpa, gemm = env_mod._read_pytorch_ck_flags(reasons)
        assert sdpa is None
        assert gemm is None
        # No reason -- pytorch_version captures torch absence elsewhere.
        assert reasons == []

    def test_both_flags_on(self, isolated_env, monkeypatch):
        import builtins
        import types

        config = types.SimpleNamespace(
            show=lambda: "CXX_FLAGS=-DUSE_ROCM_CK_SDPA -DUSE_ROCM_CK_GEMM -O2"
        )
        fake_torch = types.SimpleNamespace(__config__=config)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        sdpa, gemm = env_mod._read_pytorch_ck_flags(reasons)
        assert sdpa is True
        assert gemm is True

    def test_one_flag_off_distinguishes_from_null(
        self, isolated_env, monkeypatch
    ):
        """``False`` must be a meaningful answer (a wheel built without
        the CK SDPA path is dispatching to AOTriton -- a real and
        important state to surface), distinct from ``None`` (couldn't
        ask).
        """
        import builtins
        import types

        config = types.SimpleNamespace(
            show=lambda: "CXX_FLAGS=-DUSE_ROCM_CK_GEMM -O2"  # no SDPA
        )
        fake_torch = types.SimpleNamespace(__config__=config)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        sdpa, gemm = env_mod._read_pytorch_ck_flags(reasons)
        assert sdpa is False
        assert gemm is True

    def test_use_rocm_ck_sdpa_not_in_canonical_env_vars(self):
        """Regression guard: schema 1.1 deliberately removed
        USE_ROCM_CK_SDPA from CANONICAL_ENV_VARS because it's a
        build-time cmake flag, not a runtime env var. If a future PR
        re-adds it, this test catches the regression and forces a
        deliberate review.
        """
        assert "USE_ROCM_CK_SDPA" not in CANONICAL_ENV_VARS
        assert "USE_ROCM_CK_GEMM" not in CANONICAL_ENV_VARS

    def test_partial_reasons_use_composable_kernel_prefix(self, all_disabled):
        reasons: list[str] = []
        env_mod._capture_composable_kernel(reasons)
        # Only system.* should appear -- pytorch_bundled is silent when
        # torch absence is already captured by pytorch_version.
        prefixes = {r.split(":", 1)[0] for r in reasons}
        assert prefixes <= {
            "composable_kernel.system.version",
            "composable_kernel.system.commit",
        }, reasons

    def test_ck_tile_present_when_header_exists(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        ck_tile_header = tmp_path / "ck_tile_config.hpp"
        ck_tile_header.write_text("// header")
        monkeypatch.setattr(env_mod, "CK_VERSION_HEADER", tmp_path / "no_ck.h")
        monkeypatch.setattr(env_mod, "CK_TILE_CONFIG_HEADER", ck_tile_header)
        reasons: list[str] = []
        block = env_mod._capture_composable_kernel(reasons)
        assert block["system"]["ck_tile_present"] is True


class TestCKPytorchBundledProbe:
    def test_torch_absent_returns_default_no_reason(
        self, isolated_env, monkeypatch
    ):
        """Common case: torch not importable. We record no reason because
        pytorch_version already captures the absence."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._probe_pytorch_bundled_ck(reasons)
        assert block == {"present": False, "symbol_count": None}
        # Critical: no reason added (avoids duplicating pytorch_version's
        # already-recorded absence)
        assert reasons == []

    def test_cpu_only_torch_does_not_flip_partial(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """A wheel with ``torch.version.hip is None`` is CPU-only by
        design -- the absence of libtorch_hip.so is documented, not a
        fallback. The probe should return the default block WITHOUT
        appending a partial reason, mirroring the docker-on-baremetal
        contract.
        """
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        torch_dir.mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        # No lib/libtorch_hip.so on purpose -- this is CPU-only torch.
        # Critically, version.hip is None.
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip=None, cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._probe_pytorch_bundled_ck(reasons)
        assert block == {"present": False, "symbol_count": None}
        # Documented absence -- NO reason appended (consumer can read
        # the CPU-only state from torch.version.hip themselves).
        assert reasons == [], (
            f"CPU-only torch should not trigger partial; got reasons: {reasons}"
        )

    def test_hip_torch_with_missing_lib_does_flip_partial(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Inverse case: torch.version.hip claims HIP support but
        libtorch_hip.so is gone. That's a broken/incomplete install --
        partial=True with a clear reason is correct.
        """
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        torch_dir.mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip="7.2.0", cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Faking torch.__file__ only removes the PRIMARY lookup path. The
        # /proc/self/maps fallback would still find a real libtorch_hip.so
        # mapped into this process, which is exactly what happens when the
        # suite runs inside a ROCm-torch CI image -- the probe then reports
        # present=True and the "lib is gone" premise never gets tested.
        monkeypatch.setattr(env_mod, "_loaded_lib_path_from_maps", lambda sonames: None)
        reasons: list[str] = []
        block = env_mod._probe_pytorch_bundled_ck(reasons)
        assert block == {"present": False, "symbol_count": None}
        # This IS a partial -- claims HIP, lib gone. Reason should
        # name the situation so an operator can act on it.
        assert any("not found" in r and "claims HIP" in r for r in reasons), reasons

    def test_nm_missing_records_reason(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Stripped container (binutils not installed)."""
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        (torch_dir / "lib" / "libtorch_hip.so").write_bytes(b"x")
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(__file__=str(torch_init))

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        reasons: list[str] = []
        block = env_mod._probe_pytorch_bundled_ck(reasons)
        assert block == {"present": False, "symbol_count": None}
        assert any("nm/c++filt" in r for r in reasons), reasons

    def test_happy_path_counts_ck_symbols(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        (torch_dir / "lib" / "libtorch_hip.so").write_bytes(b"x")
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(__file__=str(torch_init))

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/" + name)

        def fake_run(cmd, **kwargs):
            if cmd[0].endswith("nm"):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="raw\n", stderr=""
                )
            if cmd[0].endswith("c++filt"):
                # Three demangled lines, two contain ck:: namespace
                stdout = (
                    "ck::tensor_operation::Foo\n"
                    "std::vector<int>\n"
                    "ck::Block::Bar\n"
                )
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=stdout, stderr=""
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        block = env_mod._probe_pytorch_bundled_ck(reasons)
        assert block == {"present": True, "symbol_count": 2}
        assert reasons == []


# ---------------------------------------------------------------------------
# Tensile
# ---------------------------------------------------------------------------


class TestCombinedKernelDbFingerprint:
    def test_both_dirs_present_combines_filenames(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "Kernels.so-000-gfx942.hsaco").write_bytes(b"x")
        (b / "TensileLibrary_gfx942.dat").write_bytes(b"y")
        fp = env_mod._combined_kernel_db_fingerprint([a, b])
        assert fp is not None and fp.startswith("filenames-sha256:")

    def test_one_dir_missing_still_fingerprints(self, tmp_path: Path):
        a = tmp_path / "a"
        a.mkdir()
        (a / "TensileLibrary_X.dat").write_bytes(b"x")
        fp = env_mod._combined_kernel_db_fingerprint([a, tmp_path / "missing"])
        assert fp is not None

    def test_both_dirs_missing_returns_none(self, tmp_path: Path):
        assert env_mod._combined_kernel_db_fingerprint(
            [tmp_path / "a", tmp_path / "b"]
        ) is None

    def test_a_wheel_layout_kernel_db_is_not_null(self, tmp_path: Path):
        """``tensile.kernel_db_combined_hash`` on a TheRock install (#387, 8th pass).

        ``library/`` there holds nothing but ``gfx*`` directories, so the
        flat-only scan produced ``null`` *and* a partial reason for a kernel
        database that was fully present -- which reads as a damaged install and
        points triage at the wrong thing.
        """
        library = tmp_path / "hipblaslt" / "library"
        (library / "gfx950").mkdir(parents=True)
        (library / "gfx950" / "TensileLibrary_A.dat").write_bytes(b"x")
        fp = env_mod._combined_kernel_db_fingerprint([library])
        assert fp is not None and fp.startswith("filenames-sha256:")

    def test_nested_entries_are_tagged_by_library_and_arch(self, tmp_path: Path):
        """One kernel name, two libraries, two targets -- four distinct entries.

        The tag is ``<library>/<arch>/<file>``, so neither the library nor the
        architecture may collapse. Asserted as inequality against each partial
        layout rather than a recorded digest, so the test survives a hash change.
        """

        def build(root: Path, spec: dict[str, list[str]]) -> Path:
            for arch, libs in spec.items():
                for lib in libs:
                    d = root / lib / "library" / arch
                    d.mkdir(parents=True)
                    (d / "TensileLibrary_A.dat").write_bytes(b"x")
            return root

        both = build(tmp_path / "both", {"gfx950": ["hipblaslt", "rocblas"]})
        one_lib = build(tmp_path / "one_lib", {"gfx950": ["hipblaslt"]})
        two_arch = build(
            tmp_path / "two_arch", {"gfx950": ["hipblaslt"], "gfx942": ["hipblaslt"]}
        )

        def fp(root: Path) -> str | None:
            return env_mod._combined_kernel_db_fingerprint(
                sorted(root.glob("*/library"))
            )

        assert fp(both) != fp(one_lib)
        assert fp(two_arch) != fp(one_lib)
        assert fp(both) != fp(two_arch)

    def test_a_flat_layout_digest_is_unchanged_by_the_nested_branch(
        self, tmp_path: Path
    ):
        """Classic installs must not see churn: the nested branch adds nothing."""
        flat = tmp_path / "flat" / "library"
        flat.mkdir(parents=True)
        (flat / "TensileLibrary_A.dat").write_bytes(b"x")

        with_empty_arch = tmp_path / "flat" / "library2"
        with_empty_arch.mkdir(parents=True)
        (with_empty_arch / "TensileLibrary_A.dat").write_bytes(b"x")
        (with_empty_arch / "gfx950").mkdir()

        # Same parent (`flat`), so the library tag matches and only the nested
        # branch can differ.
        assert env_mod._combined_kernel_db_fingerprint(
            [flat]
        ) == env_mod._combined_kernel_db_fingerprint([with_empty_arch])

    def test_an_unlistable_arch_dir_invalidates_the_whole_digest(
        self, tmp_path: Path, monkeypatch
    ):
        """A partial union must not be published as a complete fingerprint.

        This value exists only to be compared, so hashing whatever happened to
        be readable makes two identical installs differ because one had a
        permissions problem -- and being non-null, it carries no partial reason
        to explain it (#387, 9th pass).
        """
        library = tmp_path / "hipblaslt" / "library"
        (library / "gfx950").mkdir(parents=True)
        (library / "gfx950" / "TensileLibrary_A.dat").write_bytes(b"x")
        # A second, readable library so there is a non-empty subset to publish.
        other = tmp_path / "rocblas" / "library"
        other.mkdir(parents=True)
        (other / "TensileLibrary_B.dat").write_bytes(b"y")
        assert env_mod._combined_kernel_db_fingerprint([library, other]) is not None

        real_iterdir = Path.iterdir

        def boom(self):
            if self.name == "gfx950":
                raise PermissionError(13, "Permission denied")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", boom)
        assert env_mod._combined_kernel_db_fingerprint([library, other]) is None

    def test_an_unlistable_library_dir_invalidates_the_whole_digest(
        self, tmp_path: Path, monkeypatch
    ):
        a = tmp_path / "hipblaslt" / "library"
        a.mkdir(parents=True)
        (a / "TensileLibrary_A.dat").write_bytes(b"x")
        b = tmp_path / "rocblas" / "library"
        b.mkdir(parents=True)
        (b / "TensileLibrary_B.dat").write_bytes(b"y")

        real_iterdir = Path.iterdir

        def boom(self):
            if self.parent.name == "rocblas":
                raise PermissionError(13, "Permission denied")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", boom)
        assert env_mod._combined_kernel_db_fingerprint([a, b]) is None

    def test_a_missing_dir_is_still_not_a_failure(self, tmp_path: Path):
        """The distinction that keeps the above from being over-strict.

        Only hipBLASLt being installed is a normal layout, not a failure to read
        anything, so an absent directory is skipped as before.
        """
        a = tmp_path / "hipblaslt" / "library"
        a.mkdir(parents=True)
        (a / "TensileLibrary_A.dat").write_bytes(b"x")
        assert (
            env_mod._combined_kernel_db_fingerprint([a, tmp_path / "rocblas" / "library"])
            is not None
        )


    def test_dir_basename_namespaces_collisions(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        # Same filename in both dirs -- the (dir, file) tagging should
        # produce a different fingerprint than putting both in one dir.
        (a / "Kernels.dat").write_bytes(b"x")
        (b / "Kernels.dat").write_bytes(b"x")
        fp_separated = env_mod._combined_kernel_db_fingerprint([a, b])

        c = tmp_path / "c"
        c.mkdir()
        (c / "Kernels.dat").write_bytes(b"x")
        fp_alone = env_mod._combined_kernel_db_fingerprint([c])
        assert fp_separated != fp_alone

    def test_real_world_library_basename_does_not_collapse(
        self, tmp_path: Path
    ):
        """Regression guard for the production layout:
        /opt/rocm/lib/{hipblaslt,rocblas}/library/.

        Both directories' immediate basename is ``library`` -- using
        ``d.name`` directly would key every entry as ``library/<file>``
        and make hipblaslt's and rocblas's same-named kernel files
        indistinguishable. The fingerprint must use the parent
        directory's name (``hipblaslt`` vs ``rocblas``) for tagging.
        """
        # Mirror the real prod layout exactly.
        hipblaslt_dir = tmp_path / "hipblaslt" / "library"
        rocblas_dir = tmp_path / "rocblas" / "library"
        hipblaslt_dir.mkdir(parents=True)
        rocblas_dir.mkdir(parents=True)
        # A same-named kernel file in BOTH directories.
        (hipblaslt_dir / "Kernels.dat").write_bytes(b"x")
        (rocblas_dir / "Kernels.dat").write_bytes(b"x")

        fp_combined = env_mod._combined_kernel_db_fingerprint(
            [hipblaslt_dir, rocblas_dir]
        )
        # Compare against a hypothetical "I only saw the rocblas dir"
        # fingerprint -- if the namespacing collapsed, both would be
        # equal since "library/Kernels.dat" + "library/Kernels.dat"
        # dedupes after sort().
        fp_rocblas_only = env_mod._combined_kernel_db_fingerprint(
            [rocblas_dir]
        )
        assert fp_combined != fp_rocblas_only, (
            "Combined fingerprint collapsed when both directories share "
            "the basename 'library'. The probe must tag by the library "
            "name (parent dir), not the immediate basename."
        )


@pytest.fixture
def python314_path_semantics(monkeypatch):
    """Make ``Path.is_file`` / ``is_dir`` / ``exists`` suppress every OSError.

    That is what Python 3.14 does (CPython gh-101357): they return ``False``
    for ANY OSError instead of raising for some kinds. This branch is what adds
    3.14 to the support matrix, and the CPU matrix does run it -- but the local
    interpreter here is 3.13, where these still raise, so without this fixture
    the 3.14 behaviour is simply not exercised and every assertion below passes
    for the wrong reason.

    Simulating the semantics rather than gating on ``sys.version_info`` keeps
    the guarantee under test on every interpreter, which matters because the
    bug is invisible on the one most of us run locally.
    """

    def suppressed(real):
        def wrapper(self, *args, **kwargs):
            try:
                return real(self, *args, **kwargs)
            except OSError:
                return False

        return wrapper

    for name in ("is_file", "is_dir", "exists"):
        monkeypatch.setattr(Path, name, suppressed(getattr(Path, name)))
    return monkeypatch


class TestKernelDbFingerprintUnderPython314PathSemantics:
    """An unreadable entry must invalidate the digest on 3.14 too (#387).

    The enclosing functions already catch ``OSError`` and return ``None`` so an
    incomplete enumeration cannot ship as a confident hash. On 3.14 the
    predicates they relied on stopped raising, so the exception never arrived
    and the entry was silently OMITTED instead -- reinstating exactly the
    "two identical hosts compare unequal, with no partial reason" failure the
    invalidation was added to remove, on the newest supported interpreter.

    These pass trivially if the code uses ``stat()``; they fail if it goes back
    to ``is_file()`` / ``is_dir()``.
    """

    @staticmethod
    def _unreadable(monkeypatch, name: str):
        real_stat = Path.stat

        def boom(self, *args, **kwargs):
            if self.name == name:
                raise PermissionError(13, "Permission denied")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", boom)

    def test_unreadable_flat_entry_invalidates_the_combined_digest(
        self, tmp_path: Path, python314_path_semantics
    ):
        library = tmp_path / "hipblaslt" / "library"
        library.mkdir(parents=True)
        (library / "TensileLibrary_A.dat").write_bytes(b"x")
        (library / "locked.dat").write_bytes(b"y")
        assert env_mod._combined_kernel_db_fingerprint([library]) is not None

        self._unreadable(python314_path_semantics, "locked.dat")
        assert env_mod._combined_kernel_db_fingerprint([library]) is None

    def test_unreadable_arch_dir_invalidates_the_combined_digest(
        self, tmp_path: Path, python314_path_semantics
    ):
        library = tmp_path / "hipblaslt" / "library"
        (library / "gfx950").mkdir(parents=True)
        (library / "gfx950" / "TensileLibrary_A.dat").write_bytes(b"x")
        assert env_mod._combined_kernel_db_fingerprint([library]) is not None

        self._unreadable(python314_path_semantics, "gfx950")
        assert env_mod._combined_kernel_db_fingerprint([library]) is None

    def test_unreadable_entry_invalidates_the_per_library_digest(
        self, tmp_path: Path, python314_path_semantics
    ):
        library = tmp_path / "library"
        library.mkdir()
        (library / "TensileLibrary_A.dat").write_bytes(b"x")
        (library / "locked.dat").write_bytes(b"y")
        assert env_mod._kernel_db_filename_fingerprint(library) is not None

        self._unreadable(python314_path_semantics, "locked.dat")
        assert env_mod._kernel_db_filename_fingerprint(library) is None

    def test_an_absent_root_is_still_skipped_not_invalidating(
        self, tmp_path: Path, python314_path_semantics
    ):
        """The over-strictness guard, re-checked under the new semantics.

        Absent is a layout (only hipBLASLt installed); unreadable is a failure.
        Routing the root probe through stat() must not collapse the two.
        """
        a = tmp_path / "hipblaslt" / "library"
        a.mkdir(parents=True)
        (a / "TensileLibrary_A.dat").write_bytes(b"x")
        assert (
            env_mod._combined_kernel_db_fingerprint([a, tmp_path / "rocblas" / "library"])
            is not None
        )

    def test_an_unreadable_root_invalidates_rather_than_being_skipped(
        self, tmp_path: Path, python314_path_semantics
    ):
        """The other half of that distinction, which safe_is_dir could not make."""
        a = tmp_path / "hipblaslt" / "library"
        a.mkdir(parents=True)
        (a / "TensileLibrary_A.dat").write_bytes(b"x")
        b = tmp_path / "rocblas" / "library"
        b.mkdir(parents=True)
        (b / "TensileLibrary_B.dat").write_bytes(b"y")

        self._unreadable(python314_path_semantics, "library")
        assert env_mod._combined_kernel_db_fingerprint([a, b]) is None


class TestTensileBlock:
    def test_block_keys_stable(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_tensile(reasons)
        assert set(block.keys()) == {"package_version", "kernel_db_combined_hash"}

    def test_tensile_pip_absence_does_not_record_reason(self, all_disabled):
        """Tensile is rarely on production hosts; suppress the import-miss reason."""
        reasons: list[str] = []
        env_mod._capture_tensile(reasons)
        assert all("Tensile not importable" not in r for r in reasons)

    def test_kernel_db_absent_records_reason(self, all_disabled):
        reasons: list[str] = []
        env_mod._capture_tensile(reasons)
        assert any(
            r.startswith("tensile.kernel_db_combined_hash:") for r in reasons
        )


# ---------------------------------------------------------------------------
# Static Tensile catalog (issue #54)
# ---------------------------------------------------------------------------


def _make_tensile_install(directory: Path, *, archs=("gfx942",), content=b"menu"):
    """Stand up a synthetic Tensile library dir and return it.

    Mirrors a modern hipBLASLt/rocBLAS layout: per-arch ``.dat`` logic
    files, a ``TensileManifest.txt``, and a ``.hsaco`` code object.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        (directory / f"TensileLibrary_lazy_{arch}.dat").write_bytes(
            content + arch.encode()
        )
        (directory / f"Kernels.so-000-{arch}.hsaco").write_bytes(b"obj-" + arch.encode())
    (directory / "TensileManifest.txt").write_text("\n".join(archs) + "\n")
    return directory


class TestTensileMenuEnumeration:
    def test_missing_dir_is_partial_not_silent(self, tmp_path: Path):
        menu = env_mod._enumerate_tensile_menu(tmp_path / "does_not_exist")
        assert menu["status"] == "partial"
        assert menu["reason"] is not None
        assert menu["files"] is None
        # A "couldn't read" probe must be distinguishable from an empty menu.
        assert menu["logic_file_count"] is None

    def test_present_but_empty_is_ok_with_zero_count(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        menu = env_mod._enumerate_tensile_menu(empty)
        assert menu["status"] == "ok"
        assert menu["logic_file_count"] == 0
        assert menu["file_count"] == 0
        assert menu["gfx_arch_coverage"] == []

    def test_enumerates_count_archs_and_per_file_hashes(self, tmp_path: Path):
        d = _make_tensile_install(tmp_path / "lib", archs=("gfx942", "gfx90a"))
        menu = env_mod._enumerate_tensile_menu(d)
        assert menu["status"] == "ok"
        # Two .dat logic files (one per arch); .hsaco + .txt are not "logic".
        assert menu["logic_file_count"] == 2
        assert menu["file_count"] == 5  # 2 dat + 2 hsaco + 1 manifest
        assert menu["gfx_arch_coverage"] == ["gfx90a", "gfx942"]
        # Every file carries a content hash + size; logic flag set right.
        for f in menu["files"]:
            assert f["sha256"].startswith("sha256:")
            assert isinstance(f["size"], int)
        logic = [f for f in menu["files"] if f["is_logic"]]
        assert {f["suffix"] for f in logic} == {".dat"}

    def test_hashes_are_stable_across_calls(self, tmp_path: Path):
        d = _make_tensile_install(tmp_path / "lib")
        first = env_mod._enumerate_tensile_menu(d)
        second = env_mod._enumerate_tensile_menu(d)
        assert first == second
        assert first["combined_content_hash"] == second["combined_content_hash"]

    def test_two_install_diff_surfaces_changed_logic_file(self, tmp_path: Path):
        """The core acceptance scenario: same filenames, changed bytes."""
        host_a = _make_tensile_install(tmp_path / "a", content=b"menuA")
        host_b = _make_tensile_install(tmp_path / "b", content=b"menuB")
        menu_a = env_mod._enumerate_tensile_menu(host_a)
        menu_b = env_mod._enumerate_tensile_menu(host_b)

        # The shallow filename fingerprint can't tell these apart...
        assert env_mod._kernel_db_filename_fingerprint(
            host_a
        ) == env_mod._kernel_db_filename_fingerprint(host_b)
        # ...but the deepened per-file content enumeration does.
        assert menu_a["combined_content_hash"] != menu_b["combined_content_hash"]

        by_name_a = {f["name"]: f["sha256"] for f in menu_a["files"]}
        by_name_b = {f["name"]: f["sha256"] for f in menu_b["files"]}
        changed = [n for n in by_name_a if by_name_a[n] != by_name_b[n]]
        # Diff localizes exactly the logic + code-object files (manifest
        # content is identical), not just "something differs".
        assert "TensileLibrary_lazy_gfx942.dat" in changed
        assert "TensileManifest.txt" not in changed

    def test_extract_gfx_archs_dedups_and_lowercases(self):
        names = ["TensileLibrary_GFX90A.dat", "x_gfx90a.dat", "y_gfx942.co", "none.dat"]
        assert env_mod._extract_gfx_archs(names) == ["gfx90a", "gfx942"]

    def test_combined_hash_is_none_when_a_file_hash_fails(self, tmp_path, monkeypatch):
        """A combined hash over a missing per-file hash isn't a true
        content fingerprint -- it must be None, not a content-sha256:
        value (Copilot review on PR #228).
        """
        d = _make_tensile_install(tmp_path / "lib", archs=("gfx942",))
        victim = "TensileLibrary_lazy_gfx942.dat"
        real_hash = env_mod._hash_file_path

        def flaky_hash(path):
            if path.name == victim:
                return None  # simulate an unreadable file
            return real_hash(path)

        monkeypatch.setattr(env_mod, "_hash_file_path", flaky_hash)
        menu = env_mod._enumerate_tensile_menu(d)
        assert menu["status"] == "partial"
        assert menu["combined_content_hash"] is None
        # The unreadable file is still listed (not dropped), with sha256 None.
        victim_entry = [f for f in menu["files"] if f["name"] == victim][0]
        assert victim_entry["sha256"] is None

    def test_broken_symlink_matching_catalog_suffix_is_not_silently_dropped(
        self, tmp_path: Path
    ):
        """A broken symlink whose name matches a catalog suffix must be
        listed in ``files`` with ``sha256: None`` and downgrade the menu
        to ``partial`` -- not silently vanish from the enumeration.

        The original implementation skipped any entry whose ``is_file()``
        raised ``OSError`` (which a broken symlink does) before it ever
        reached the hashing step that would otherwise catch this, so a
        host with one broken/dangling logic file could report a clean
        ``status: "ok"`` with a ``combined_content_hash`` that silently
        omitted it.
        """
        d = _make_tensile_install(tmp_path / "lib", archs=("gfx942",))
        broken = d / "TensileLibrary_lazy_gfx90a.dat"
        broken.symlink_to(d / "does_not_exist.dat")

        menu = env_mod._enumerate_tensile_menu(d)
        assert menu["status"] == "partial"
        assert menu["reason"] is not None
        names = [f["name"] for f in menu["files"]]
        assert broken.name in names, "matched filename must not be silently dropped"
        broken_entry = [f for f in menu["files"] if f["name"] == broken.name][0]
        assert broken_entry["sha256"] is None
        assert menu["combined_content_hash"] is None


class TestCatalogCompactDetail:
    """Compact mode drops the per-file lists but keeps every fingerprint."""

    def test_enumerate_compact_drops_files_keeps_summary(self, tmp_path: Path):
        d = _make_tensile_install(tmp_path / "lib", archs=("gfx942", "gfx90a"))
        full = env_mod._enumerate_tensile_menu(d, include_files=True)
        compact = env_mod._enumerate_tensile_menu(d, include_files=False)
        # The heavy per-file list is gone in compact...
        assert full["files"] is not None and len(full["files"]) > 0
        assert compact["files"] is None
        # ...but every summary + fingerprint field is preserved and equal.
        for key in (
            "status",
            "dir",
            "file_count",
            "logic_file_count",
            "gfx_arch_coverage",
            "combined_content_hash",
        ):
            assert compact[key] == full[key], key
        # The whole point: the menu-level content hash is unchanged, so a
        # two-host diff still detects THAT the catalog changed.
        assert compact["combined_content_hash"] is not None

    def test_tensile_catalog_compact_drops_both_menus(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_TENSILE_DIR", _make_tensile_install(tmp_path / "hb")
        )
        monkeypatch.setattr(
            env_mod, "ROCBLAS_TENSILE_DIR", _make_tensile_install(tmp_path / "rb")
        )
        empty_lib = {
            "package_version": None,
            "lib_hash": None,
            "kernel_db_revision": None,
        }
        block = env_mod._build_tensile_catalog(
            dict(empty_lib),
            dict(empty_lib),
            {"package_version": None, "kernel_db_combined_hash": None},
            [],
            include_files=False,
        )
        assert block["hipblaslt"]["menu"]["files"] is None
        assert block["rocblas"]["menu"]["files"] is None
        # Fingerprints survive so the compact block still diffs.
        assert block["hipblaslt"]["menu"]["combined_content_hash"] is not None
        assert block["hipblaslt"]["menu"]["file_count"] > 0

    def test_miopen_catalog_compact_drops_menu_files(
        self, tmp_path: Path, monkeypatch
    ):
        db = _make_miopen_db(tmp_path / "db")
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", db)
        monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
        block = env_mod._build_miopen_catalog({}, [], include_files=False)
        assert block["menu"]["files"] is None
        assert block["menu"]["combined_content_hash"] is not None
        assert block["menu"]["file_count"] > 0

    def test_collect_env_defaults_to_compact(
        self, tmp_path: Path, monkeypatch
    ):
        # The default probe (no detail arg) must not carry per-file lists.
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_TENSILE_DIR", _make_tensile_install(tmp_path / "hb")
        )
        monkeypatch.setattr(
            env_mod, "ROCBLAS_TENSILE_DIR", _make_tensile_install(tmp_path / "rb")
        )
        monkeypatch.setattr(
            env_mod, "MIOPEN_KERNEL_DB_DIR", _make_miopen_db(tmp_path / "db")
        )
        monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
        snap = env_mod.collect_env()
        assert snap.tensile_catalog["hipblaslt"]["menu"]["files"] is None
        assert snap.tensile_catalog["rocblas"]["menu"]["files"] is None
        assert snap.miopen_catalog["menu"]["files"] is None
        # detail="full" restores them.
        snap_full = env_mod.collect_env(detail="full")
        assert snap_full.tensile_catalog["hipblaslt"]["menu"]["files"] is not None
        assert snap_full.miopen_catalog["menu"]["files"] is not None


class TestTensileCatalogBlock:
    def test_default_and_disaster_shapes_match_built_block(self, tmp_path: Path):
        built = env_mod._build_tensile_catalog(
            {"package_version": None, "lib_hash": None, "kernel_db_revision": None},
            {"package_version": None, "lib_hash": None, "kernel_db_revision": None},
            {"package_version": None, "kernel_db_combined_hash": None},
            [],
        )
        empty = env_mod._empty_tensile_catalog()
        assert set(built.keys()) == set(empty.keys())
        assert set(built["hipblaslt"].keys()) == set(empty["hipblaslt"].keys())
        assert set(built["hipblaslt"]["menu"].keys()) == set(
            empty["hipblaslt"]["menu"].keys()
        )

    def test_preserves_existing_identity_fields_no_regression(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_TENSILE_DIR", _make_tensile_install(tmp_path / "hb")
        )
        monkeypatch.setattr(
            env_mod, "ROCBLAS_TENSILE_DIR", _make_tensile_install(tmp_path / "rb")
        )
        hipblaslt = {
            "package_version": "1.2.0",
            "lib_hash": "sha256:aa",
            "kernel_db_revision": "filenames-sha256:bb",
        }
        rocblas = {
            "package_version": "5.0.2",
            "lib_hash": "sha256:cc",
            "kernel_db_revision": "filenames-sha256:dd",
        }
        tensile = {
            "package_version": None,
            "kernel_db_combined_hash": "filenames-sha256:ee",
        }
        block = env_mod._build_tensile_catalog(hipblaslt, rocblas, tensile, [])
        # Existing hashes flow through verbatim -- no regression.
        assert block["hipblaslt"]["lib_hash"] == "sha256:aa"
        assert block["hipblaslt"]["kernel_db_revision"] == "filenames-sha256:bb"
        assert block["rocblas"]["lib_hash"] == "sha256:cc"
        assert block["combined"]["kernel_db_combined_hash"] == "filenames-sha256:ee"
        assert block["status"] == "ok"

    def test_partial_threads_reason_into_partial_reasons(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_TENSILE_DIR", _make_tensile_install(tmp_path / "hb")
        )
        monkeypatch.setattr(
            env_mod, "ROCBLAS_TENSILE_DIR", tmp_path / "missing_rocblas"
        )
        reasons: list[str] = []
        block = env_mod._build_tensile_catalog(
            {"package_version": None, "lib_hash": None, "kernel_db_revision": None},
            {"package_version": None, "lib_hash": None, "kernel_db_revision": None},
            {"package_version": None, "kernel_db_combined_hash": None},
            reasons,
        )
        assert block["status"] == "partial"
        assert block["rocblas"]["menu"]["status"] == "partial"
        assert block["hipblaslt"]["menu"]["status"] == "ok"
        assert any(
            r.startswith("tensile_catalog.rocblas.menu:") for r in reasons
        )

    def test_doc_is_labeled_as_installed_identity_not_runtime(self):
        doc = env_mod._empty_tensile_catalog()["doc"]
        assert "recipe book" in doc.lower()
        assert "381881" in doc or "368xxx" in doc  # explicitly disclaims the runtime pick

    def test_block_present_in_collect_env_snapshot(self, all_disabled):
        snap = collect_env()
        d = snap.to_dict()
        assert "tensile_catalog" in d
        assert set(d["tensile_catalog"].keys()) == {
            "doc",
            "status",
            "hipblaslt",
            "rocblas",
            "combined",
        }


# ---------------------------------------------------------------------------
# Static MIOpen catalog (issue #54 follow-up)
# ---------------------------------------------------------------------------


def _make_miopen_db(directory: Path, *, archs=("gfx942_120",), content=b"db"):
    """Stand up a synthetic MIOpen db dir (find-db, perf-db, model, kdb)."""
    directory.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        (directory / f"{arch}.HIP.fdb.txt").write_bytes(content + b"-fdb")
        (directory / f"{arch}.db.txt").write_bytes(content + b"-perf")
        (directory / f"{arch}.kdb").write_bytes(content + b"-kdb")
    (directory / "gfx908.tn.model").write_bytes(b"heuristic-model")
    return directory


class TestMiopenCatalogBlock:
    def test_enumerates_dbs_archs_and_logic_count(self, tmp_path, monkeypatch):
        d = _make_miopen_db(tmp_path / "db", archs=("gfx942_120", "gfx90a_104"))
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", d)
        monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
        block = env_mod._build_miopen_catalog(
            {"package_version": "3.5.0", "lib_hash": "sha256:aa",
             "kernel_db_revision": "filenames-sha256:bb"},
            [],
        )
        assert block["status"] == "ok"
        # 2 archs x (fdb + perf + kdb) = 6 logic dbs; +1 model = 7 files.
        assert block["menu"]["logic_file_count"] == 6
        assert block["menu"]["file_count"] == 7
        assert block["menu"]["gfx_arch_coverage"] == ["gfx908", "gfx90a", "gfx942"]
        # No regression: identity fields pass through verbatim.
        assert block["lib_hash"] == "sha256:aa"
        assert block["kernel_db_revision"] == "filenames-sha256:bb"
        assert block["db_dir_source"] == "default"
        # The .model file is catalog-but-not-logic.
        model = [f for f in block["menu"]["files"] if f["name"].endswith(".model")][0]
        assert model["is_logic"] is False
        assert model["suffix"] == ".tn.model"

    def test_multipart_suffix_labeled_correctly(self, tmp_path, monkeypatch):
        d = _make_miopen_db(tmp_path / "db")
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", d)
        monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
        block = env_mod._build_miopen_catalog({}, [])
        suffixes = {f["name"]: f["suffix"] for f in block["menu"]["files"]}
        # find-db keeps its full multi-part suffix, not pathlib's bare .txt
        assert suffixes["gfx942_120.HIP.fdb.txt"] == ".fdb.txt"
        assert suffixes["gfx942_120.db.txt"] == ".db.txt"
        assert suffixes["gfx942_120.kdb"] == ".kdb"

    def test_system_db_path_override_is_honored_and_recorded(self, tmp_path, monkeypatch):
        override = _make_miopen_db(tmp_path / "override_db")
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", tmp_path / "default_unused")
        monkeypatch.setenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, str(override))
        block = env_mod._build_miopen_catalog({}, [])
        assert block["db_dir"] == str(override)
        assert block["db_dir_source"] == env_mod.MIOPEN_SYSTEM_DB_PATH_ENV
        assert block["status"] == "ok"
        assert block["env_overrides"][env_mod.MIOPEN_SYSTEM_DB_PATH_ENV] == str(override)

    def test_kernel_db_revision_follows_system_db_path_override(
        self, tmp_path, monkeypatch
    ):
        """Regression: ``kernel_db_revision`` must describe the SAME
        directory as ``db_dir``/``menu`` when ``MIOPEN_SYSTEM_DB_PATH``
        is set. Copying the legacy top-level ``miopen`` block's value
        verbatim would silently describe the packaged default dir while
        ``db_dir``/``menu`` describe the override -- one block, two
        different directories.
        """
        default_dir = _make_miopen_db(tmp_path / "default_db", archs=("gfx908_60",))
        override_dir = _make_miopen_db(
            tmp_path / "override_db", archs=("gfx942_120", "gfx90a_104")
        )
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", default_dir)
        monkeypatch.setenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, str(override_dir))

        # The stale value a naive "copy the top-level field verbatim"
        # implementation would have used.
        stale_kernel_db_revision = env_mod._kernel_db_filename_fingerprint(
            default_dir, suffixes=env_mod.MIOPEN_KERNEL_DB_SUFFIXES
        )
        block = env_mod._build_miopen_catalog(
            {"kernel_db_revision": stale_kernel_db_revision}, []
        )

        expected = env_mod._kernel_db_filename_fingerprint(
            override_dir, suffixes=env_mod.MIOPEN_KERNEL_DB_SUFFIXES
        )
        assert block["db_dir"] == str(override_dir)
        assert block["kernel_db_revision"] == expected
        assert block["kernel_db_revision"] != stale_kernel_db_revision

    def test_kernel_db_revision_reuses_top_level_value_when_no_override(
        self, tmp_path, monkeypatch
    ):
        """No-regression case: without an override, reuse the already-
        computed top-level ``miopen.kernel_db_revision`` verbatim rather
        than fingerprinting the directory a second time.
        """
        d = _make_miopen_db(tmp_path / "db")
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", d)
        monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
        sentinel = "filenames-sha256:sentinel-from-top-level-miopen-block"
        block = env_mod._build_miopen_catalog(
            {"kernel_db_revision": sentinel}, []
        )
        assert block["db_dir_source"] == "default"
        assert block["kernel_db_revision"] == sentinel

    def test_missing_dir_is_partial_and_threads_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", tmp_path / "nope")
        monkeypatch.delenv(env_mod.MIOPEN_SYSTEM_DB_PATH_ENV, raising=False)
        reasons: list[str] = []
        block = env_mod._build_miopen_catalog({}, reasons)
        assert block["status"] == "partial"
        assert any(r.startswith("miopen_catalog.menu:") for r in reasons)

    def test_two_install_diff_localizes_changed_db(self, tmp_path, monkeypatch):
        a = _make_miopen_db(tmp_path / "a", content=b"hostA")
        b = _make_miopen_db(tmp_path / "b", content=b"hostB")
        ma = env_mod._enumerate_catalog_dir(
            a, env_mod._suffix_classifier(
                env_mod.MIOPEN_CATALOG_SUFFIXES, env_mod.MIOPEN_LOGIC_SUFFIXES),
            kind="MIOpen db")
        mb = env_mod._enumerate_catalog_dir(
            b, env_mod._suffix_classifier(
                env_mod.MIOPEN_CATALOG_SUFFIXES, env_mod.MIOPEN_LOGIC_SUFFIXES),
            kind="MIOpen db")
        assert ma["combined_content_hash"] != mb["combined_content_hash"]

    def test_block_present_and_shaped_in_snapshot(self, all_disabled):
        d = collect_env().to_dict()
        assert set(d["miopen_catalog"].keys()) == {
            "doc", "status", "package_version", "lib_hash", "kernel_db_revision",
            "db_dir", "db_dir_source", "env_overrides", "menu",
        }


class TestCatalogNestedArchDirs:
    """The wheel layout's ``gfx*`` subdirectories (#381), and #387's fix."""

    @staticmethod
    def _enumerate(directory):
        return env_mod._enumerate_catalog_dir(
            directory,
            env_mod._suffix_classifier(
                env_mod.MIOPEN_CATALOG_SUFFIXES, env_mod.MIOPEN_LOGIC_SUFFIXES
            ),
            kind="MIOpen db",
        )

    def _tree(self, tmp_path):
        root = tmp_path / "db"
        (root / "gfx950").mkdir(parents=True)
        (root / "gfx950" / "a.kdb").write_bytes(b"nested")
        (root / "flat.kdb").write_bytes(b"flat")
        return root

    def test_nested_arch_files_are_recorded_with_an_arch_prefix(self, tmp_path):
        menu = self._enumerate(self._tree(tmp_path))
        assert menu["status"] == "ok"
        names = [f["name"] for f in menu["files"]]
        assert "gfx950/a.kdb" in names and "flat.kdb" in names
        assert menu["combined_content_hash"] is not None

    def test_an_unlistable_arch_dir_forces_partial(self, tmp_path, monkeypatch):
        """It hides EVERY kernel for that target, so it cannot read as "ok".

        Silently dropping it left ``status: "ok"`` plus a combined hash computed
        over the archs that did list -- a clean-looking fingerprint of a
        different catalog than the one on disk (#387).
        """
        root = self._tree(tmp_path)
        real_iterdir = Path.iterdir

        def selective(self):
            if self.name == "gfx950":
                raise OSError("permission denied")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", selective)
        menu = self._enumerate(root)

        assert menu["status"] == "partial"
        assert "gfx950" in menu["reason"]
        assert "not listable" in menu["reason"]
        # A hash over the surviving entries would misrepresent the catalog.
        assert menu["combined_content_hash"] is None
        # ...but whatever DID list is still reported, not thrown away.
        assert [f["name"] for f in menu["files"]] == ["flat.kdb"]

    def test_an_unconfirmable_entry_is_kept_and_not_blamed_on_an_arch_dir(
        self, tmp_path, monkeypatch
    ):
        """``is_dir()`` raising is already handled by the flat path.

        Such an entry stays in ``candidates`` under its flat name and is hashed
        there, so it must not ALSO be reported as an unlistable arch dir --
        that reason is reserved for a confirmed directory whose contents are
        genuinely missing from the catalog.
        """
        root = tmp_path / "db"
        root.mkdir()
        (root / "gfx942.kdb").write_bytes(b"x")
        real_is_dir = Path.is_dir

        def selective(self):
            if self.name == "gfx942.kdb":
                raise OSError("permission denied")
            return real_is_dir(self)

        monkeypatch.setattr(Path, "is_dir", selective)
        menu = self._enumerate(root)
        # It hashed fine, so nothing is missing and "ok" is the honest reading.
        assert menu["status"] == "ok"
        assert [f["name"] for f in menu["files"]] == ["gfx942.kdb"]
        assert menu["reason"] is None

    def test_both_degradations_are_reported_together(self, tmp_path, monkeypatch):
        """A hash failure and an unlistable arch dir are different problems."""
        root = self._tree(tmp_path)
        real_iterdir = Path.iterdir
        real_open = Path.open

        def selective_iterdir(self):
            if self.name == "gfx950":
                raise OSError("permission denied")
            return real_iterdir(self)

        def selective_open(self, *args, **kwargs):
            if self.name == "flat.kdb":
                raise OSError("permission denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "iterdir", selective_iterdir)
        monkeypatch.setattr(Path, "open", selective_open)
        menu = self._enumerate(root)
        assert menu["status"] == "partial"
        assert "unreadable" in menu["reason"]
        assert "not listable" in menu["reason"]


# ---------------------------------------------------------------------------
# Static rocFFT catalog (issue #54 follow-up)
# ---------------------------------------------------------------------------


class TestRocfftCatalogBlock:
    def test_absent_when_no_cache_is_silent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path / "nope")
        monkeypatch.delenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, raising=False)
        monkeypatch.delenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, raising=False)
        reasons: list[str] = []
        block = env_mod._build_rocfft_catalog(reasons)
        assert block["status"] == "absent"
        assert block["kernel_cache"]["present"] is False
        # Absence is the documented common case -- NOT partial.
        assert reasons == []

    def test_present_cache_is_fingerprinted(self, tmp_path, monkeypatch):
        (tmp_path / env_mod.ROCFFT_KERNEL_CACHE_NAME).write_bytes(b"sqlite-bytes")
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path)
        monkeypatch.delenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, raising=False)
        monkeypatch.delenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, raising=False)
        block = env_mod._build_rocfft_catalog([])
        assert block["status"] == "ok"
        assert block["kernel_cache"]["present"] is True
        assert block["kernel_cache"]["sha256"].startswith("sha256:")
        assert block["kernel_cache"]["source"] == "rocm_lib"
        assert block["kernel_cache"]["size"] == len(b"sqlite-bytes")

    def test_nested_rocfft_subdir_layout_resolves(self, tmp_path, monkeypatch):
        """Some installs ship the cache under ``lib/rocfft/`` rather than
        directly in ``lib/`` -- both are real default layouts per
        rocFFT's ``rtc_cache.cpp`` search order.
        """
        nested = tmp_path / "rocfft" / env_mod.ROCFFT_KERNEL_CACHE_NAME
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"nested-cache")
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path)
        monkeypatch.delenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, raising=False)
        monkeypatch.delenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, raising=False)
        block = env_mod._build_rocfft_catalog([])
        assert block["status"] == "ok"
        assert block["kernel_cache"]["path"] == str(nested)
        assert block["kernel_cache"]["source"] == "rocm_lib"

    def test_rtc_sys_cache_path_env_override_wins(self, tmp_path, monkeypatch):
        """ROCFFT_RTC_SYS_CACHE_PATH is the read-only system-cache override
        -- the one that matters for installed-library identity.
        """
        cache = tmp_path / "custom" / env_mod.ROCFFT_KERNEL_CACHE_NAME
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"override-cache")
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path / "rocm_unused")
        monkeypatch.setenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, str(cache.parent))
        monkeypatch.delenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, raising=False)
        block = env_mod._build_rocfft_catalog([])
        assert block["status"] == "ok"
        assert block["kernel_cache"]["path"] == str(cache)
        assert block["kernel_cache"]["source"] == env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV

    def test_rtc_cache_path_env_alone_is_not_used_for_resolution(
        self, tmp_path, monkeypatch
    ):
        """ROCFFT_RTC_CACHE_PATH is the read-write, per-process USER cache
        -- mutable and workload-dependent, not installed-library identity.
        A cache file sitting there must NOT be fingerprinted as the
        static catalog, even though it is recorded for visibility.
        """
        user_cache_dir = tmp_path / "user_cache"
        user_cache_dir.mkdir()
        (user_cache_dir / env_mod.ROCFFT_KERNEL_CACHE_NAME).write_bytes(b"mutable")
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path / "no_system_cache")
        monkeypatch.delenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, raising=False)
        monkeypatch.setenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, str(user_cache_dir))
        block = env_mod._build_rocfft_catalog([])
        # No system cache found -- must stay "absent", NOT resolve the
        # mutable user cache as installed identity.
        assert block["status"] == "absent"
        assert block["kernel_cache"]["present"] is False
        # Still recorded for visibility.
        assert block["env_overrides"][env_mod.ROCFFT_RTC_CACHE_PATH_ENV] == str(
            user_cache_dir
        )

    def test_override_recorded_even_when_cache_absent(self, tmp_path, monkeypatch):
        """Configured RTC-cache paths are captured even with no cache file.

        Distinguishes "override set, but no cache shipped" from "override
        never set" -- the case Copilot flagged on PR #228. Covers both
        the system-cache and user-cache override variables.
        """
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path / "nope")
        monkeypatch.setenv(
            env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, "/some/configured/sys/dir"
        )
        monkeypatch.setenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, "/some/configured/dir")
        block = env_mod._build_rocfft_catalog([])
        assert block["status"] == "absent"
        assert block["kernel_cache"]["present"] is False
        assert block["env_overrides"][env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV] == (
            "/some/configured/sys/dir"
        )
        assert block["env_overrides"][env_mod.ROCFFT_RTC_CACHE_PATH_ENV] == (
            "/some/configured/dir"
        )

    def test_not_captured_default_distinct_from_probed_absent(self, tmp_path, monkeypatch):
        """The default/backfill shape must be distinguishable from a probe
        that ran and found no cache (Copilot review on PR #228).
        """
        # Default / disaster / from_dict backfill shape: "not captured".
        default = env_mod._empty_rocfft_catalog()
        assert default["status"] == "partial"
        assert default["kernel_cache"]["reason"] == "rocfft_catalog not captured"
        # A real probe that finds nothing: clean "absent", no reason.
        monkeypatch.setattr(env_mod, "ROCFFT_LIB_DIR", tmp_path / "nope")
        monkeypatch.delenv(env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV, raising=False)
        monkeypatch.delenv(env_mod.ROCFFT_RTC_CACHE_PATH_ENV, raising=False)
        probed = env_mod._build_rocfft_catalog([])
        assert probed["status"] == "absent"
        assert probed["kernel_cache"]["reason"] is None

    def test_block_present_and_shaped_in_snapshot(self, all_disabled):
        d = collect_env().to_dict()
        assert set(d["rocfft_catalog"].keys()) == {
            "doc", "status", "env_overrides", "kernel_cache",
        }
        assert set(d["rocfft_catalog"]["env_overrides"].keys()) == {
            env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV,
            env_mod.ROCFFT_RTC_CACHE_PATH_ENV,
        }

    def test_summary_renders_present_unreadable_not_present_none(self, all_disabled):
        """A present-but-unhashable cache must not render as "present None"
        in the CLI brief (Copilot review on PR #228).
        """
        d = collect_env().to_dict()
        d["rocfft_catalog"] = {
            "doc": "x",
            "status": "partial",
            "env_overrides": {
                env_mod.ROCFFT_RTC_SYS_CACHE_PATH_ENV: None,
                env_mod.ROCFFT_RTC_CACHE_PATH_ENV: None,
            },
            "kernel_cache": {
                "present": True, "path": "/x", "source": "rocm_lib",
                "size": 10, "sha256": None,
                "reason": "cache file present but could not be hashed",
            },
        }
        snap = env_mod.EnvSnapshot.from_dict(d)
        catalog_line = [l for l in snap.summary().splitlines() if "catalog:" in l][0]
        assert "rocfft=present (unreadable)" in catalog_line
        assert "present None" not in catalog_line

    def test_summary_renders_dash_not_none_for_not_captured_hashes(self):
        """A not-captured catalog (e.g. a pre-1.9 snapshot backfilled via
        ``from_dict``) must render its missing per-menu hashes as ``-``,
        not the literal string ``"None"`` -- the same ambiguity
        ``pkg_state()`` avoids elsewhere.
        """
        d = _example_snapshot().to_dict()
        del d["tensile_catalog"]
        del d["miopen_catalog"]
        del d["rocfft_catalog"]
        snap = env_mod.EnvSnapshot.from_dict(d)
        catalog_line = [l for l in snap.summary().splitlines() if "catalog:" in l][0]
        assert "None" not in catalog_line
        assert "tensile[hb=- rb=-]" in catalog_line
        assert "miopen=-" in catalog_line


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------


class TestTritonBlock:
    def test_triton_unavailable_returns_none_with_reason(
        self, isolated_env, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "triton":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_triton(reasons)
        assert block == {"package_version": None, "commit": None}
        assert any("triton" in r for r in reasons)

    def test_triton_with_version_returns_string(self, isolated_env, monkeypatch):
        import builtins
        import types

        real_import = builtins.__import__
        fake_triton = types.SimpleNamespace(__version__="3.5.1+rocm7.2.1.gita272dfa8")

        def fake_import(name, *args, **kwargs):
            if name == "triton":
                return fake_triton
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_triton(reasons)
        assert block == {
            "package_version": "3.5.1+rocm7.2.1.gita272dfa8",
            "commit": "a272dfa8",
        }
        assert reasons == []


# ---------------------------------------------------------------------------
# TorchRec package identity (schema 1.14)
# ---------------------------------------------------------------------------


# A full 40-char build SHA, the shape TorchRec's setup.py writes into
# torchrec/version.py -- deliberately longer than any version-string local
# segment can carry (setup.py truncates those to sha[:7]).
_TORCHREC_FULL_SHA = "0123abcdef0123abcdef0123abcdef0123abcdef"

# Byte-for-byte the format of TorchRec's setup.py `_export_version`.
_TORCHREC_VERSION_PY = "__version__ = '{version}'\ngit_version = '{commit}'\n"


def _torchrec_pkg_spec(tmp_path: Path, version_py: str | None):
    """A REAL ModuleSpec over a synthetic torchrec package directory.

    Built with the genuine machinery rather than a stand-in object so the
    ``loader`` / ``submodule_search_locations`` attributes that
    ``_capture_torchrec`` inspects behave like the real thing.
    """
    pkg = tmp_path / "torchrec"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    if version_py is not None:
        (pkg / "version.py").write_text(version_py)
    return importlib.util.spec_from_file_location(
        "torchrec", pkg / "__init__.py", submodule_search_locations=[str(pkg)]
    )


def _torchrec_namespace_spec(tmp_path: Path):
    """The spec CPython returns for a PEP 420 namespace package.

    A bare directory named ``torchrec`` on ``sys.path`` (no ``__init__.py``)
    resolves to a LOADER-LESS spec -- verified on 3.12::

        ModuleSpec(name='torchrec', loader=None,
                   submodule_search_locations=_NamespacePath(['.../torchrec']))

    Constructed here rather than resolved for real because a genuinely
    installed torchrec in the test environment would out-rank the namespace
    portion and mask the case under test.
    """
    import importlib.machinery

    pkg = tmp_path / "torchrec"
    pkg.mkdir(exist_ok=True)
    spec = importlib.machinery.ModuleSpec("torchrec", None)
    spec.submodule_search_locations = [str(pkg)]
    return spec


def _expect_torchrec(**overrides):
    """The full ``torchrec`` block shape, with ``None`` for anything not named.

    Written out rather than compared key-by-key so a test also fails when a NEW
    key appears -- the block is customer-facing identity, and a silently added
    field is exactly what a reviewer would want flagged.
    """
    block = {
        "package_version": None,
        "commit": None,
        "source_version": None,
        "source_commit": None,
        "distribution_version": None,
    }
    assert set(overrides) <= set(block), f"unknown torchrec keys: {set(overrides) - set(block)}"
    block.update(overrides)
    return block


class TestTorchrec:
    @pytest.fixture
    def no_torchrec_spec(self, monkeypatch):
        """Pin ``find_spec`` to "not importable".

        ``_capture_torchrec`` now reads the installed ``torchrec/version.py``,
        so on a machine that genuinely has torchrec its real SHA would leak
        into the version-string parser cases below and mask what they assert.
        """
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def test_torchrec_absent_is_suppressed_no_reason(self, monkeypatch, no_torchrec_spec):
        """No dist-info AND not importable (find_spec None) -> genuinely
        absent -> no partial reason, mirroring fbgemm_gpu/aiter."""
        import importlib.metadata as md

        def fake_version(name):
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr(md, "version", fake_version)
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec()
        assert reasons == []

    def test_torchrec_namespace_package_counts_as_absent(self, monkeypatch, tmp_path):
        """A bare ``torchrec/`` directory on sys.path is a PEP 420 namespace
        portion, not an install. Without the loader guard it would inject a
        phantom partial reason -- and since ``partial=bool(reasons)`` that
        would mark an otherwise clean snapshot incomplete."""
        import importlib.metadata as md

        def fake_version(name):
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr(md, "version", fake_version)
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name: _torchrec_namespace_spec(tmp_path)
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec()
        assert reasons == []

    def test_torchrec_importable_without_version_py_is_partial(self, monkeypatch, tmp_path):
        """Importable, no dist-info, no version.py -> present with unknown
        identity. Must NOT be reported as absent."""
        import importlib.metadata as md

        def fake_version(name):
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr(md, "version", fake_version)
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name: _torchrec_pkg_spec(tmp_path, None)
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec()
        assert any("torchrec" in r and "no readable source version" in r for r in reasons)

    def test_torchrec_unknown_source_never_promotes_unrelated_metadata(
        self, monkeypatch, tmp_path
    ):
        """An import target without version.py cannot be bound to a dist-info
        found elsewhere. Keep the latter separate and mark the primary identity
        unknown instead of cleanly describing code that will not execute."""
        import importlib.metadata as md

        monkeypatch.setattr(md, "version", lambda name: "2.0.0")
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name: _torchrec_pkg_spec(tmp_path, None)
        )
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="2.0.0")
        assert any("not promoted" in r for r in reasons), reasons

    def test_torchrec_source_commit_survives_when_source_version_is_missing(
        self, monkeypatch, tmp_path
    ):
        import importlib.metadata as md

        version_py = f"git_version = '{_TORCHREC_FULL_SHA}'\n"
        monkeypatch.setattr(md, "version", lambda name: "2.0.0")
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            source_commit=_TORCHREC_FULL_SHA,
            distribution_version="2.0.0",
        )
        assert any("not promoted" in r for r in reasons), reasons
        torchrec_line = next(
            line
            for line in _example_snapshot(torchrec=block).summary().splitlines()
            if line.lstrip().startswith("torchrec:")
        )
        assert "present (version unknown)" in torchrec_line
        assert _TORCHREC_FULL_SHA in torchrec_line

    def test_torchrec_buck_link_tree_resolved_from_version_py(self, monkeypatch, tmp_path):
        """Buck / PYTHONPATH link-tree: no dist-info, but the package ships
        version.py -> report that identity, no partial reason."""
        import importlib.metadata as md

        def fake_version(name):
            raise md.PackageNotFoundError(name)

        version_py = _TORCHREC_VERSION_PY.format(version="1.8.0a0", commit=_TORCHREC_FULL_SHA)
        monkeypatch.setattr(md, "version", fake_version)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec(
            package_version="1.8.0a0",
            commit=_TORCHREC_FULL_SHA,
            source_version="1.8.0a0",
            source_commit=_TORCHREC_FULL_SHA,
        )
        assert reasons == []

    def test_torchrec_release_wheel_commit_from_version_py(self, monkeypatch, tmp_path):
        """The headline case: a release wheel's version string carries no SHA,
        but torchrec/version.py holds the FULL 40-char one. Reporting null
        here (schema 1.14) made two different builds of "1.4.0" indistinguishable."""
        import importlib.metadata as md

        version_py = _TORCHREC_VERSION_PY.format(version="1.4.0", commit=_TORCHREC_FULL_SHA)
        monkeypatch.setattr(md, "version", lambda name: "1.4.0")
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec(
            package_version="1.4.0",
            commit=_TORCHREC_FULL_SHA,
            source_version="1.4.0",
            source_commit=_TORCHREC_FULL_SHA,
            distribution_version="1.4.0",
        )
        assert reasons == []

    def test_torchrec_version_py_unknown_sha_is_rejected(self, monkeypatch, tmp_path):
        """setup.py writes ``git_version = 'Unknown'`` when the build tree is
        not a git checkout -- that must never surface as a commit."""
        import importlib.metadata as md

        version_py = _TORCHREC_VERSION_PY.format(version="1.4.0", commit="Unknown")
        monkeypatch.setattr(md, "version", lambda name: "1.4.0")
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec(
            package_version="1.4.0",
            source_version="1.4.0",
            distribution_version="1.4.0",
        )
        assert reasons == []

    def test_torchrec_version_py_unknown_version_is_rejected(self, monkeypatch, tmp_path):
        """A non-git build tree also gets ``__version__ = 'Unknown'``.

        Publishing it verbatim made ``package_version: "Unknown"`` the primary
        identity of a clean, non-partial snapshot, which a snapshot diff then
        reads as a version.
        """
        import importlib.metadata as md

        version_py = _TORCHREC_VERSION_PY.format(version="Unknown", commit="Unknown")
        monkeypatch.setattr(
            md, "version", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError())
        )
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec()
        assert reasons == [
            "torchrec.package_version: import target has no readable source "
            "version and no distribution metadata; primary identity remains unknown"
        ]

    def test_torchrec_conflicting_copies_report_the_import_target_and_say_so(
        self, monkeypatch, tmp_path
    ):
        """Two copies on sys.path: the importable one's version.py does not
        describe the distribution the metadata came from.

        Schema 1.15 reports BOTH identities and prefers the import target -- the
        code that would actually run -- instead of publishing the other
        install's version with the SHA quietly dropped. The disagreement is a
        fact about the process, so it belongs in partial_reasons rather than
        being smoothed away."""
        import importlib.metadata as md

        version_py = _TORCHREC_VERSION_PY.format(version="9.9.9", commit=_TORCHREC_FULL_SHA)
        monkeypatch.setattr(md, "version", lambda name: "1.4.0")
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec(
            package_version="9.9.9",
            commit=_TORCHREC_FULL_SHA,
            source_version="9.9.9",
            source_commit=_TORCHREC_FULL_SHA,
            distribution_version="1.4.0",
        )
        assert any("9.9.9" in r and "1.4.0" in r for r in reasons), reasons

    def test_torchrec_pep440_equivalent_versions_are_not_a_conflict(self, monkeypatch, tmp_path):
        """``1.4.0-1`` and ``1.4.0.post1`` are the same PEP 440 version, so the
        two identities agree and no conflict is reported. A raw string compare
        called this a conflict and dropped the SHA."""
        import importlib.metadata as md

        version_py = _TORCHREC_VERSION_PY.format(version="1.4.0-1", commit=_TORCHREC_FULL_SHA)
        monkeypatch.setattr(md, "version", lambda name: "1.4.0.post1")
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec(
            package_version="1.4.0-1",
            commit=_TORCHREC_FULL_SHA,
            source_version="1.4.0-1",
            source_commit=_TORCHREC_FULL_SHA,
            distribution_version="1.4.0.post1",
        )
        assert reasons == []

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("1.0+01", "1.0+1"),
            ("1.0+abc.01", "1.0+abc.1"),
            ("1.0_dev", "1.0.dev0"),
        ],
    )
    def test_torchrec_pep440_equivalent_spellings(self, left: str, right: str):
        assert env_mod._same_distribution_version(left, right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("1.0+dev", "1.0.dev0+dev"),
            ("1.0+abc.dev", "1.0.dev0+abc.dev"),
        ],
    )
    def test_torchrec_pep440_non_equivalent_versions_stay_distinct(
        self, left: str, right: str
    ):
        assert not env_mod._same_distribution_version(left, right)

    def test_torchrec_malformed_loader_is_fail_soft(self, monkeypatch, tmp_path):
        import importlib.machinery
        import importlib.metadata as md

        class BrokenLoader:
            @property
            def get_data(self):
                raise RuntimeError("broken loader property")

        pkg = tmp_path / "torchrec"
        pkg.mkdir()
        spec = importlib.machinery.ModuleSpec(
            "torchrec", BrokenLoader(), origin=str(pkg / "__init__.py")
        )
        spec.submodule_search_locations = [str(pkg)]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(md, "version", lambda name: "2.0.0")
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="2.0.0")
        assert any("no readable source version" in r for r in reasons), reasons

    def test_torchrec_malformed_loader_still_uses_filesystem_fallback(
        self, monkeypatch, tmp_path
    ):
        import importlib.machinery
        import importlib.metadata as md

        class BrokenLoader:
            @property
            def get_data(self):
                raise RuntimeError("broken loader property")

        pkg = tmp_path / "torchrec"
        pkg.mkdir()
        (pkg / "version.py").write_text(
            _TORCHREC_VERSION_PY.format(
                version="1.4.0", commit=_TORCHREC_FULL_SHA
            )
        )
        spec = importlib.machinery.ModuleSpec(
            "torchrec", BrokenLoader(), origin=str(pkg / "__init__.py")
        )
        spec.submodule_search_locations = [str(pkg)]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(md, "version", lambda name: "1.4.0")
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="1.4.0",
            commit=_TORCHREC_FULL_SHA,
            source_version="1.4.0",
            source_commit=_TORCHREC_FULL_SHA,
            distribution_version="1.4.0",
        )
        assert reasons == []

    def test_torchrec_originless_custom_loader_is_not_a_namespace(self, monkeypatch):
        import importlib.machinery
        import importlib.metadata as md

        class VirtualPackageLoader:
            def get_data(self, path):
                return _TORCHREC_VERSION_PY.format(
                    version="7.7.7", commit=_TORCHREC_FULL_SHA
                ).encode()

        spec = importlib.machinery.ModuleSpec(
            "torchrec", VirtualPackageLoader(), origin=None, is_package=True
        )
        spec.submodule_search_locations = ["/virtual/torchrec"]

        def no_distribution(name):
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(md, "version", no_distribution)
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="7.7.7",
            commit=_TORCHREC_FULL_SHA,
            source_version="7.7.7",
            source_commit=_TORCHREC_FULL_SHA,
        )
        assert reasons == []

    def test_torchrec_unusable_spec_origin_does_not_cost_the_snapshot(self, monkeypatch):
        """A spec whose ``origin`` is not a str/PathLike must stay fail-soft.

        The ownership check builds ``Path(origin)`` before its per-candidate
        guard, so a custom loader reporting a non-path origin raised TypeError
        out of ``_capture_torchrec`` and collect_env's never-raises gate
        answered with a disaster snapshot -- the whole environment capture
        traded for one unverifiable TorchRec install.
        """
        import importlib.machinery
        import importlib.metadata as md

        spec = importlib.machinery.ModuleSpec("torchrec", object(), origin=object())
        spec.submodule_search_locations = ["/virtual/torchrec"]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(md, "version", lambda name: "2.0.0")
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="2.0.0")
        assert any("ownership cannot be verified" in r for r in reasons), reasons

    def test_torchrec_ownership_is_fail_soft_when_cwd_is_gone(self, monkeypatch, tmp_path):
        """``absolute()`` on a RELATIVE origin calls os.getcwd().

        A long-lived trainer whose scratch directory is cleaned up underneath
        it then hits FileNotFoundError inside the ownership check, which must
        degrade to "cannot verify" rather than discard the snapshot.
        """
        import importlib.machinery
        import importlib.metadata as md
        import shutil as _shutil

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        spec = importlib.machinery.ModuleSpec("torchrec", object(), origin="torchrec/__init__.py")
        spec.submodule_search_locations = ["torchrec"]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(md, "version", lambda name: "2.0.0")
        monkeypatch.chdir(scratch)
        _shutil.rmtree(scratch)
        reasons: list[str] = []

        block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="2.0.0")
        assert any("ownership cannot be verified" in r for r in reasons), reasons

    def test_torchrec_release_wheel_has_no_commit(self):
        assert env_mod._torchrec_commit("1.4.0") is None

    def test_torchrec_setuptools_scm_commit(self):
        assert env_mod._torchrec_commit("1.5.0.dev0+g0123abc") == "0123abc"

    def test_torchrec_source_build_bare_hex_commit(self):
        """TorchRec source builds use a BARE hex local segment (no 'g' lead-in),
        e.g. 1.8.0a0+0123abc -> 0123abc (P1 review)."""
        assert env_mod._torchrec_commit("1.8.0a0+0123abc") == "0123abc"

    def test_torchrec_bare_hex_ignores_non_sha_local(self):
        """A non-SHA local segment (+cpu / +fb / too-short) must NOT be
        mistaken for a commit."""
        for ver in ("1.4.0+cpu", "1.4.0+fb", "1.4.0+abcd", "2026.7.2"):
            assert env_mod._torchrec_commit(ver) is None, ver

    def test_torchrec_scm_dirty_date_is_not_a_commit(self):
        """Regression: setuptools_scm's dirty-tree marker ``+d<YYYYMMDD>`` is
        entirely valid hex, so a naive bare-hex match would publish the build
        DATE as a commit SHA. A wrong SHA is worse than none (phantom
        'commit changed' diffs), so these must resolve to None."""
        for ver in ("1.4.0+d20240101", "1.4.0+20240101"):
            assert env_mod._torchrec_commit(ver) is None, ver

    def test_torchrec_real_sha_survives_dirty_date_guard(self):
        """The dirty-date guard must not swallow genuine SHAs, including a
        bare SHA followed by the dirty tag, or an uppercase SHA."""
        for ver, expected in (
            ("1.4.0+deadbeef", "deadbeef"),
            ("1.8.0a0+0123abc.d20240101", "0123abc"),
            ("1.8.0a0+0123ABC", "0123abc"),
        ):
            assert env_mod._torchrec_commit(ver) == expected, ver

    def test_torchrec_metadata_error_records_reason(self, monkeypatch, no_torchrec_spec):
        import importlib.metadata as md

        def boom(name):
            raise RuntimeError("broken dist-info")

        monkeypatch.setattr(md, "version", boom)
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec()
        assert any("torchrec" in r for r in reasons)

    def test_torchrec_metadata_error_keeps_the_readable_source_identity(
        self, monkeypatch, tmp_path
    ):
        """A metadata read that RAISES must not discard an identity we already
        have. Returning all-null here threw away a version.py that had just been
        read successfully."""
        import importlib.metadata as md

        def boom(name):
            raise OSError("unreadable dist-info")

        version_py = _TORCHREC_VERSION_PY.format(version="1.4.0", commit=_TORCHREC_FULL_SHA)
        monkeypatch.setattr(md, "version", boom)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _torchrec_pkg_spec(tmp_path, version_py),
        )
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec(
            package_version="1.4.0",
            commit=_TORCHREC_FULL_SHA,
            source_version="1.4.0",
            source_commit=_TORCHREC_FULL_SHA,
        )
        assert any("metadata lookup raised (OSError)" in r for r in reasons), reasons

    def test_torchrec_find_spec_error_is_not_reported_as_absent(self, monkeypatch):
        """A broken ``sys.path`` entry makes ``find_spec`` raise. "We could not
        look" is not the same fact as "torchrec is not installed", and only the
        latter is allowed to be silent."""
        import importlib.metadata as md

        def boom_spec(name):
            raise ImportError("broken meta-path finder")

        monkeypatch.setattr(md, "version", lambda name: (_ for _ in ()).throw(
            md.PackageNotFoundError(name)
        ))
        monkeypatch.setattr(importlib.util, "find_spec", boom_spec)
        reasons: list[str] = []
        block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec()
        assert any("find_spec raised (ImportError)" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# TorchRec identity against REAL on-disk layouts (schema 1.15)
#
# Everything above monkeypatches ``find_spec`` / ``importlib.metadata``. These
# tests instead build the layout on disk and let the real machinery resolve it,
# because the two defects this class exists to pin -- a ``.par`` whose
# ``version.py`` lives inside a zip, and a single-module ``torchrec.py`` next to
# a foreign ``version.py`` -- are invisible to a patched spec. (The PEP 420 bug
# fixed earlier survived the suite for the same reason: the spec was an
# ``object()`` stand-in.)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _only_sys_path(*entries: Path):
    """Run with ``sys.path`` REPLACED by *entries*.

    Replaced, not prepended: a host with a real torchrec installed would answer
    every lookup and the scenario under test would silently not be tested. Both
    ``find_spec`` and ``importlib.metadata`` resolve through ``sys.path``, so
    emptying it is what makes "not installed" simulatable at all.
    """
    saved_path = sys.path[:]
    saved_modules = {k: v for k, v in sys.modules.items() if k.startswith("torchrec")}
    sys.path[:] = [str(e) for e in entries]
    for name in saved_modules:
        del sys.modules[name]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "torchrec" or name.startswith("torchrec."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def _make_torchrec_package(root: Path, *, version: str | None, commit: str | None) -> Path:
    """A real importable ``torchrec`` package. Its ``__init__`` RAISES, which
    doubles as the proof that the probe never executes it."""
    pkg = root / "torchrec"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('torchrec must not be imported')\n")
    if version is not None:
        (pkg / "version.py").write_text(_TORCHREC_VERSION_PY.format(version=version, commit=commit))
    return pkg


def _make_dist_info(
    root: Path, version: str, *, owned_files: tuple[str, ...] = ()
) -> Path:
    d = root / f"torchrec-{version}.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: torchrec\nVersion: {version}\n")
    (d / "RECORD").write_text(
        "".join(f"{path},,\n" for path in owned_files)
    )
    return d


class TestTorchrecRealLayouts:
    def test_par_zipimport_resolves_identity_through_the_loader(self, tmp_path):
        """The deployment this feature exists for: a Buck ``.par``.

        ``submodule_search_locations`` points INSIDE the archive, so a
        filesystem read finds nothing and the probe used to report null while
        also claiming "no readable version.py" -- with the file sitting in the
        zip. Reading through ``loader.get_data`` is what fixes it."""
        stage = tmp_path / "stage"
        _make_torchrec_package(stage, version="1.4.0", commit=_TORCHREC_FULL_SHA)
        par = tmp_path / "app.par"
        with zipfile.ZipFile(par, "w") as zf:
            zf.write(stage / "torchrec" / "__init__.py", "torchrec/__init__.py")
            zf.write(stage / "torchrec" / "version.py", "torchrec/version.py")

        with _only_sys_path(par):
            spec = importlib.util.find_spec("torchrec")
            assert type(spec.loader).__name__ == "zipimporter", "layout is not a real zipimport"
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="1.4.0",
            commit=_TORCHREC_FULL_SHA,
            source_version="1.4.0",
            source_commit=_TORCHREC_FULL_SHA,
        )
        assert reasons == []
        assert "torchrec" not in sys.modules

    def test_par_metadata_owns_the_archived_package(self, tmp_path):
        """RECORD ownership must work inside a ``.par``.

        ``locate_file`` yields a ``zipfile._path.Path``, which has no
        ``__fspath__``; building a ``pathlib.Path`` from it raised ``TypeError``
        into a broad handler, so an archive that demonstrably holds both the
        metadata and the code reported "ownership cannot be verified".
        """
        stage = tmp_path / "stage"
        _make_torchrec_package(stage, version=None, commit=None)
        par = tmp_path / "app.par"
        with zipfile.ZipFile(par, "w") as zf:
            zf.write(stage / "torchrec" / "__init__.py", "torchrec/__init__.py")
            zf.writestr(
                "torchrec-1.4.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: torchrec\nVersion: 1.4.0\n",
            )
            zf.writestr("torchrec-1.4.0.dist-info/RECORD", "torchrec/__init__.py,,\n")

        with _only_sys_path(par):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="1.4.0",
            distribution_version="1.4.0",
        )
        assert reasons == []
        assert "torchrec" not in sys.modules

    def test_symlinked_par_metadata_still_owns_the_archived_package(self, tmp_path):
        """Lexical archive paths must stay aligned on both sides."""
        stage = tmp_path / "stage"
        _make_torchrec_package(stage, version=None, commit=None)
        par = tmp_path / "real.par"
        with zipfile.ZipFile(par, "w") as zf:
            zf.write(stage / "torchrec" / "__init__.py", "torchrec/__init__.py")
            zf.writestr(
                "torchrec-1.4.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: torchrec\nVersion: 1.4.0\n",
            )
            zf.writestr(
                "torchrec-1.4.0.dist-info/RECORD",
                "torchrec/__init__.py,,\n",
            )
        alias = tmp_path / "alias.par"
        alias.symlink_to(par.name)

        with _only_sys_path(alias):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="1.4.0",
            distribution_version="1.4.0",
        )
        assert reasons == []

    def test_editable_install_ownership_comes_from_direct_url(self, tmp_path):
        """A PEP 660 RECORD lists the ``.pth`` and finder, never the sources.

        RECORD ownership can therefore never succeed for an editable install,
        even though ``direct_url.json`` -- itself in the RECORD -- names the
        exact source tree the import target resolves into.
        """
        src = tmp_path / "src"
        _make_torchrec_package(src, version=None, commit=None)
        site = tmp_path / "site"
        dist_info = _make_dist_info(
            site,
            "1.4.0",
            owned_files=(
                "__editable__.torchrec-1.4.0.pth",
                "__editable___torchrec_1_4_0_finder.py",
                "torchrec-1.4.0.dist-info/direct_url.json",
            ),
        )
        (dist_info / "direct_url.json").write_text(
            json.dumps({"url": src.as_uri(), "dir_info": {"editable": True}})
        )

        with _only_sys_path(src, site):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="1.4.0",
            distribution_version="1.4.0",
        )
        assert reasons == []

    def test_editable_remote_file_url_cannot_claim_a_local_source(self, tmp_path):
        src = tmp_path / "src"
        _make_torchrec_package(src, version=None, commit=None)
        site = tmp_path / "site"
        dist_info = _make_dist_info(
            site,
            "1.4.0",
            owned_files=("torchrec-1.4.0.dist-info/direct_url.json",),
        )
        (dist_info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": f"file://remote.example{src.as_posix()}",
                    "dir_info": {"editable": True},
                }
            )
        )

        with _only_sys_path(src, site):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="1.4.0")
        assert any("ownership cannot be verified" in reason for reason in reasons)

    def test_record_external_leaf_symlink_cannot_claim_the_import_target(self, tmp_path):
        src = tmp_path / "src"
        package = _make_torchrec_package(src, version=None, commit=None)
        metadata = tmp_path / "metadata"
        metadata.mkdir()
        link = metadata / "external-link.py"
        link.symlink_to(package / "__init__.py")
        _make_dist_info(
            metadata,
            "8.8.8",
            owned_files=("external-link.py",),
        )

        with _only_sys_path(metadata, src):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="8.8.8")
        assert any("ownership cannot be verified" in reason for reason in reasons)

    def test_disagreeing_duplicate_metadata_is_reported_not_reconciled(self, tmp_path):
        """``version()`` returns the first match in directory order.

        Two torchrec dist-infos claiming different versions were reconciled
        silently, so the same install could yield a different answer on a
        different filesystem.
        """
        site = tmp_path / "site"
        _make_torchrec_package(site, version=None, commit=None)
        _make_dist_info(site, "2.0.0", owned_files=("torchrec/__init__.py",))
        _make_dist_info(site, "10.0.0", owned_files=("torchrec/__init__.py",))

        with _only_sys_path(site):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert len(reasons) == 1
        assert "metadata disagrees across 2 distinct versions (10.0.0, 2.0.0)" in reasons[0]
        assert block["package_version"] is None
        assert block["distribution_version"] in ("2.0.0", "10.0.0")

    def test_agreeing_duplicate_metadata_is_not_a_conflict(self, tmp_path):
        """An editable install routinely leaves a ``*.egg-info`` in the source
        tree beside the ``*.dist-info`` in site-packages, and both are on
        ``sys.path``. Two metadata dirs naming the SAME version leave the answer
        unambiguous, so this must not fire a conflict."""
        site = tmp_path / "site"
        _make_torchrec_package(site, version=None, commit=None)
        _make_dist_info(site, "1.4.0", owned_files=("torchrec/__init__.py",))
        egg = site / "torchrec.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text(
            "Metadata-Version: 2.1\nName: torchrec\nVersion: 1.4.0\n"
        )

        with _only_sys_path(site):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert reasons == []
        assert block["distribution_version"] == "1.4.0"

    def test_equivalent_later_metadata_can_establish_ownership(self, tmp_path):
        """The first metadata candidate need not be the owning one."""
        metadata = tmp_path / "metadata"
        metadata.mkdir()
        egg = metadata / "torchrec.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text(
            "Metadata-Version: 2.1\nName: torchrec\nVersion: 1.4\n"
        )
        package_root = tmp_path / "package"
        _make_torchrec_package(package_root, version=None, commit=None)
        _make_dist_info(
            package_root,
            "1.4.0",
            owned_files=("torchrec/__init__.py",),
        )

        with _only_sys_path(metadata, package_root):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block["package_version"] == "1.4"
        assert block["distribution_version"] == "1.4"
        assert reasons == []

    def test_invalid_metadata_version_is_retained_but_never_promoted(self, tmp_path):
        site = tmp_path / "site"
        _make_torchrec_package(site, version=None, commit=None)
        _make_dist_info(
            site,
            "Unknown",
            owned_files=("torchrec/__init__.py",),
        )

        with _only_sys_path(site):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="Unknown")
        assert len(reasons) == 1
        assert "metadata reports an invalid version 'Unknown'" in reasons[0]

    def test_pathological_numeric_metadata_version_is_fail_soft(self, tmp_path):
        site = tmp_path / "site"
        _make_torchrec_package(site, version="1.0", commit=None)
        dist_info = site / "torchrec-x.dist-info"
        dist_info.mkdir()
        huge = "9" * 5000
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: torchrec\n"
            f"Version: {huge}\n"
        )
        (dist_info / "RECORD").write_text("torchrec/__init__.py,,\n")

        with _only_sys_path(site):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block["package_version"] == "1.0"
        assert block["source_version"] == "1.0"
        assert block["distribution_version"] == huge
        assert any("metadata reports an invalid version" in reason for reason in reasons)

    def test_single_module_does_not_adopt_a_foreign_version_py(self, tmp_path):
        """``torchrec.py`` (a module, not a package) has no search locations.
        Falling back to ``dirname(origin)`` made it publish an unrelated
        project's version and SHA as torchrec's -- a wrong identity, which is
        worse than none."""
        (tmp_path / "torchrec.py").write_text("raise RuntimeError('must not import')\n")
        (tmp_path / "version.py").write_text(
            _TORCHREC_VERSION_PY.format(version="9.9.9", commit="d" * 40)
        )

        with _only_sys_path(tmp_path):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block["source_version"] is None
        assert block["source_commit"] is None
        assert block["package_version"] != "9.9.9"
        assert block["commit"] != "d" * 40
        assert any("no readable source version" in r for r in reasons), reasons

    def test_source_first_metadata_second_never_reports_the_other_sha(self, tmp_path):
        """The review's required case: a real two-entry ``sys.path`` with a
        Buck/source TorchRec first and an unrelated ``torchrec-*.dist-info``
        second. The snapshot must never present the unrelated distribution's
        identity as the thing that will run, and must surface the conflict."""
        source_root = tmp_path / "buck-link-tree"
        _make_torchrec_package(source_root, version="1.8.0a0", commit=_TORCHREC_FULL_SHA)
        pip_root = tmp_path / "site-packages"
        _make_dist_info(pip_root, "2.0.0")
        _make_torchrec_package(pip_root, version="2.0.0", commit="b" * 40)

        with _only_sys_path(source_root, pip_root):
            assert importlib.metadata.version("torchrec") == "2.0.0", "layout not resolving"
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        # The import target wins, and its SHA travels with its own version.
        assert block["source_version"] == "1.8.0a0"
        assert block["source_commit"] == _TORCHREC_FULL_SHA
        assert block["package_version"] == "1.8.0a0"
        assert block["commit"] == _TORCHREC_FULL_SHA
        # The other install is still recorded -- separately, and never as `commit`.
        assert block["distribution_version"] == "2.0.0"
        assert block["commit"] != "b" * 40
        assert any("1.8.0a0" in r and "2.0.0" in r for r in reasons), reasons

    def test_source_without_version_py_does_not_promote_second_metadata(self, tmp_path):
        source_root = tmp_path / "buck-link-tree"
        _make_torchrec_package(source_root, version=None, commit=None)
        pip_root = tmp_path / "site-packages"
        _make_dist_info(pip_root, "2.0.0")
        _make_torchrec_package(pip_root, version="2.0.0", commit="b" * 40)

        with _only_sys_path(source_root, pip_root):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="2.0.0")
        assert any("not promoted" in r for r in reasons), reasons

    def test_same_distribution_record_can_supply_missing_source_version(
        self, tmp_path
    ):
        root = tmp_path / "site-packages"
        _make_torchrec_package(root, version=None, commit=None)
        _make_dist_info(
            root,
            "3.2.1",
            owned_files=("torchrec/__init__.py",),
        )

        with _only_sys_path(root):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="3.2.1",
            distribution_version="3.2.1",
        )
        assert reasons == []
        torchrec_line = next(
            line
            for line in _example_snapshot(torchrec=block).summary().splitlines()
            if line.lstrip().startswith("torchrec:")
        )
        assert "3.2.1" in torchrec_line
        assert "(not installed)" not in torchrec_line

    def test_owned_metadata_pairs_with_source_only_commit(self, tmp_path):
        root = tmp_path / "site-packages"
        pkg = _make_torchrec_package(root, version=None, commit=None)
        (pkg / "version.py").write_text(
            f"git_version = '{_TORCHREC_FULL_SHA}'\n"
        )
        _make_dist_info(
            root,
            "3.2.1",
            owned_files=("torchrec/__init__.py", "torchrec/version.py"),
        )

        with _only_sys_path(root):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(
            package_version="3.2.1",
            commit=_TORCHREC_FULL_SHA,
            source_commit=_TORCHREC_FULL_SHA,
            distribution_version="3.2.1",
        )
        assert reasons == []

    def test_namespace_portion_loses_to_a_real_install_either_order(self, tmp_path):
        """A bare ``torchrec/`` directory is a PEP 420 portion. CPython's path
        finder only yields one when no real package exists anywhere on the path,
        so detection does not depend on ``sys.path`` order -- asserted rather
        than assumed, since the review raised order as a risk."""
        ns_root = tmp_path / "ns"
        (ns_root / "torchrec").mkdir(parents=True)
        real_root = tmp_path / "real"
        _make_torchrec_package(real_root, version="1.4.0", commit=_TORCHREC_FULL_SHA)
        _make_dist_info(real_root, "1.4.0")

        for entries in ((ns_root, real_root), (real_root, ns_root)):
            with _only_sys_path(*entries):
                reasons: list[str] = []
                block = env_mod._capture_torchrec(reasons)
            assert block["source_commit"] == _TORCHREC_FULL_SHA, entries
            assert block["package_version"] == "1.4.0", entries
            assert reasons == [], entries

    def test_imported_namespace_still_counts_as_absent(self, tmp_path):
        ns_root = tmp_path / "ns"
        (ns_root / "torchrec").mkdir(parents=True)

        with _only_sys_path(ns_root):
            imported = importlib.import_module("torchrec")
            # CPython renamed this class in 3.11 and kept the old name as an
            # alias, so pinning only the new spelling would pass on 3.12 while
            # the guard under test is dead on 3.10 (the declared minimum, the
            # GPU CI container's interpreter, and a cpu-tests matrix entry).
            assert type(imported.__spec__.loader).__name__ in (
                "NamespaceLoader",
                "_NamespaceLoader",
            )
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec()
        assert reasons == []

    def test_namespace_with_unrelated_metadata_never_becomes_primary(self, tmp_path):
        root = tmp_path / "mixed"
        (root / "torchrec").mkdir(parents=True)
        _make_dist_info(root, "2.0.0")

        with _only_sys_path(root):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)

        assert block == _expect_torchrec(distribution_version="2.0.0")
        assert any("namespace" in r for r in reasons), reasons

    def test_absent_everywhere_is_silent(self, tmp_path):
        """No package, no dist-info, nothing raising -> suppressed, like
        fbgemm_gpu / aiter. Requires an EMPTY sys.path to be a real test."""
        with _only_sys_path(tmp_path):
            reasons: list[str] = []
            block = env_mod._capture_torchrec(reasons)
        assert block == _expect_torchrec()
        assert reasons == []


# ---------------------------------------------------------------------------
# Package commit extraction (schema 1.8)
# ---------------------------------------------------------------------------


class TestPackageCommitExtraction:
    def test_setuptools_scm_plus_g_sha(self):
        # +g<sha> and +<distance>.g<sha> setuptools_scm local segments.
        assert env_mod._extract_commit_from_version("0.1.11.dev32+g9a469a608") == (
            "9a469a608"
        )
        assert env_mod._extract_commit_from_version("1.2.0+5.g0123abc") == "0123abc"

    def test_rocm_fork_dot_git_sha(self):
        # ROCm triton: "3.5.1+rocm7.2.1.gita272dfa8" -> a272dfa8.
        assert env_mod._extract_commit_from_version(
            "3.5.1+rocm7.2.1.gita272dfa8"
        ) == "a272dfa8"

    def test_plain_local_segment_is_not_a_commit(self):
        # +fb / +cpu / +rocm7.2.1 carry no SHA -> None (must not false-match).
        for v in ("2.13.0a0+fb", "2.12.0+cpu", "2.10.0.dev+rocm7.0", None, ""):
            assert env_mod._extract_commit_from_version(v) is None

    def test_uppercase_sha_is_matched_and_lowercased(self):
        # Hex is case-insensitive; an uppercase SHA must be captured and
        # normalized to lowercase to match the docstring contract.
        assert env_mod._extract_commit_from_version("0.1.11+gA469A608") == "a469a608"
        assert env_mod._extract_commit_from_version(
            "3.5.1+rocm7.2.1.gitA272DFA8"
        ) == "a272dfa8"

    def test_commit_from_version_string_wins(self):
        assert env_mod._capture_python_package_commit(
            "definitely_not_a_real_pkg_xyz", "1.0+g0123abcd"
        ) == "0123abcd"

    def test_commit_from_module_git_version_attr(self, monkeypatch):
        import builtins
        import types

        real_import = builtins.__import__
        fake = types.SimpleNamespace(
            version=types.SimpleNamespace(git_version="deadbeef1234")
        )

        def fake_import(name, *args, **kwargs):
            if name == "fbgemm_gpu":
                return fake
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # version string has no SHA -> falls back to module attr.
        assert env_mod._capture_python_package_commit(
            "fbgemm_gpu", "1.4.0"
        ) == "deadbeef1234"

    def test_absent_package_yields_none(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fbgemm_gpu":
                raise ImportError("absent")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert env_mod._capture_python_package_commit("fbgemm_gpu", None) is None

    def test_non_sha_module_attr_is_rejected(self, monkeypatch):
        # A git_version-style attr that is NOT a hex SHA ("unknown",
        # "dirty", a tag) must not leak into the commit field.
        import builtins
        import types

        real_import = builtins.__import__
        for bad in ("unknown", "dirty", "v2.13.0-release", "N/A"):
            fake = types.SimpleNamespace(
                version=types.SimpleNamespace(git_version=bad)
            )

            def fake_import(name, *args, _fake=fake, **kwargs):
                if name == "fbgemm_gpu":
                    return _fake
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", fake_import)
            assert (
                env_mod._capture_python_package_commit("fbgemm_gpu", "1.4.0") is None
            ), bad

    def test_bare_full_sha_attr_is_accepted_and_lowercased(self, monkeypatch):
        import builtins
        import types

        real_import = builtins.__import__
        sha = "FF65F5BC672795C5E5033900EA0A0C4F8566C8CF"
        fake = types.SimpleNamespace(__git_version__=sha)

        def fake_import(name, *args, **kwargs):
            if name == "fbgemm_gpu":
                return fake
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert env_mod._capture_python_package_commit(
            "fbgemm_gpu", "1.4.0"
        ) == sha.lower()

    def test_commit_from_attr_value_unit(self):
        assert env_mod._commit_from_attr_value("ff65f5bc") == "ff65f5bc"
        assert env_mod._commit_from_attr_value("0.1+g9a469a6") == "9a469a6"
        assert env_mod._commit_from_attr_value("unknown") is None
        assert env_mod._commit_from_attr_value("") is None
        assert env_mod._commit_from_attr_value(None) is None
        assert env_mod._commit_from_attr_value(12345) is None


# ---------------------------------------------------------------------------
# Torch native-lib location (/proc/self/maps fallback for Buck torch)
# ---------------------------------------------------------------------------


class TestTorchNativeLibDir:
    def test_prefers_wheel_layout_when_present(self, tmp_path, monkeypatch):
        import types

        lib = tmp_path / "torch" / "lib"
        lib.mkdir(parents=True)
        fake_torch = types.SimpleNamespace(__file__=str(tmp_path / "torch" / "__init__.py"))
        # Even if maps would resolve, the on-disk wheel layout wins.
        monkeypatch.setattr(
            env_mod, "_loaded_lib_path_from_maps", lambda sonames: Path("/should/not/win")
        )
        assert env_mod._torch_native_lib_dir(fake_torch) == lib

    def test_falls_back_to_maps_when_no_wheel_lib_dir(self, tmp_path, monkeypatch):
        import types

        # Buck layout: torch.__file__ exists but there is no sibling lib/.
        fake_torch = types.SimpleNamespace(
            __file__=str(tmp_path / "linktree" / "torch" / "__init__.py")
        )
        buck_lib = tmp_path / "buck-out" / "lib" / "libtorch_hip.so"
        buck_lib.parent.mkdir(parents=True)
        buck_lib.write_bytes(b"x")
        monkeypatch.setattr(
            env_mod, "_loaded_lib_path_from_maps", lambda sonames: buck_lib
        )
        assert env_mod._torch_native_lib_dir(fake_torch) == buck_lib.parent

    def test_returns_none_when_nothing_locatable(self, tmp_path, monkeypatch):
        import types

        fake_torch = types.SimpleNamespace(
            __file__=str(tmp_path / "torch" / "__init__.py")
        )
        monkeypatch.setattr(
            env_mod, "_loaded_lib_path_from_maps", lambda sonames: None
        )
        assert env_mod._torch_native_lib_dir(fake_torch) is None

    def test_symbol_dump_recovers_via_maps_for_buck_torch(
        self, isolated_env, tmp_path, monkeypatch
    ):
        """The key Buck recovery: no <torch>/lib/libtorch_hip.so on disk,
        but the lib is mapped into the process -> the symbol dump finds it
        via /proc/self/maps instead of recording a 'not found' reason.
        """
        import types

        # Fake torch with HIP claimed but NO sibling lib/ dir.
        torch_dir = tmp_path / "linktree" / "torch"
        torch_dir.mkdir(parents=True)
        (torch_dir / "__init__.py").write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_dir / "__init__.py"),
            version=types.SimpleNamespace(hip="7.0.0", cuda=None),
        )
        # The "real" lib lives in a Buck artifact dir, mapped into proc.
        buck_lib = tmp_path / "buck-out" / "lib" / env_mod.PYTORCH_HIP_LIB_NAME
        buck_lib.parent.mkdir(parents=True)
        buck_lib.write_bytes(b"\x7fELF")
        monkeypatch.setattr(
            env_mod,
            "_loaded_lib_path_from_maps",
            lambda sonames: buck_lib if env_mod.PYTORCH_HIP_LIB_NAME in sonames else None,
        )
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/" + name)

        def fake_run(cmd, **kwargs):
            if cmd[0].endswith("nm"):
                # nm must have been pointed at the maps-recovered path.
                assert str(buck_lib) in cmd
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="0 T mangled\n", stderr=""
                )
            if cmd[0].endswith("c++filt"):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout="ck::tensor_operation::Foo\n", stderr="",
                )
            raise AssertionError(f"unexpected cmd {cmd}")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        out = env_mod._dump_pytorch_hip_demangled_symbols(
            reasons, "composable_kernel.pytorch_bundled", torch_mod=fake_torch
        )
        assert out == "ck::tensor_operation::Foo\n"
        assert not any("not found" in r for r in reasons), reasons

    def test_maps_parse_handles_spaced_pathname(self, tmp_path, monkeypatch):
        """A mapped pathname containing spaces must be recovered intact;
        the maps pathname is the trailing field, so a bounded split keeps
        it whole instead of truncating at the first space.
        """
        soname = env_mod.PYTORCH_HIP_LIB_NAME
        spaced = f"/opt/my libs/{soname}"
        maps = (
            "555555554000-555555555000 r--p 00000000 08:01 100 /usr/bin/python3\n"
            f"7ffff7a00000-7ffff7b00000 r-xp 00000000 08:01 200 {spaced}\n"
        )
        fake_maps = tmp_path / "maps"
        fake_maps.write_text(maps)
        monkeypatch.setattr(env_mod, "_PROC_SELF_MAPS", fake_maps)
        assert env_mod._loaded_lib_path_from_maps((soname,)) == Path(spaced)

    def test_maps_parse_skips_deleted_mapping(self, tmp_path, monkeypatch):
        """A mapping whose backing file was unlinked after dlopen is
        rendered by the kernel as '<pathname> (deleted)'. That literal
        string is never a real, scannable path, so it must be skipped
        rather than returned as a false 'hit'.
        """
        soname = env_mod.PYTORCH_HIP_LIB_NAME
        real_lib = tmp_path / "buck-out" / "lib" / soname
        real_lib.parent.mkdir(parents=True)
        real_lib.write_bytes(b"x")
        maps = (
            "555555554000-555555555000 r--p 00000000 08:01 100 /usr/bin/python3\n"
            f"7ffff7a00000-7ffff7b00000 r-xp 00000000 08:01 200 {real_lib} (deleted)\n"
        )
        fake_maps = tmp_path / "maps"
        fake_maps.write_text(maps)
        monkeypatch.setattr(env_mod, "_PROC_SELF_MAPS", fake_maps)
        assert env_mod._loaded_lib_path_from_maps((soname,)) is None

    def test_native_lib_dir_ignores_stale_maps_hit(self, tmp_path, monkeypatch):
        """Even if _loaded_lib_path_from_maps somehow returns a path that
        no longer exists on disk (e.g. a build-artifact dir that was
        cleaned up post-load), _torch_native_lib_dir must not trust it
        and scan a torn-down directory silently.
        """
        import types

        fake_torch = types.SimpleNamespace(
            __file__=str(tmp_path / "linktree" / "torch" / "__init__.py")
        )
        stale_lib = tmp_path / "buck-out" / "lib" / env_mod.PYTORCH_HIP_LIB_NAME
        # Note: parent dir intentionally not created -- nothing exists on disk.
        monkeypatch.setattr(
            env_mod, "_loaded_lib_path_from_maps", lambda sonames: stale_lib
        )
        assert env_mod._torch_native_lib_dir(fake_torch) is None


# ---------------------------------------------------------------------------
# FBGEMM
# ---------------------------------------------------------------------------


class TestFbgemmBlock:
    def test_block_keys_stable(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_fbgemm(reasons)
        assert set(block.keys()) == {
            "package_version",
            "commit",
            "pytorch_use_fbgemm",
            "pytorch_use_fbgemm_genai",
        }

    def test_fbgemm_gpu_absence_does_not_record_reason(self, all_disabled):
        reasons: list[str] = []
        env_mod._capture_fbgemm(reasons)
        assert all("fbgemm_gpu not importable" not in r for r in reasons)

    def test_torch_absent_returns_null_flags_no_extra_reason(
        self, isolated_env, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("torch", "fbgemm_gpu"):
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_fbgemm(reasons)
        assert block["pytorch_use_fbgemm"] is None
        assert block["pytorch_use_fbgemm_genai"] is None
        # No reason added -- pytorch_version captures torch absence elsewhere
        assert reasons == []

    def test_pytorch_use_fbgemm_parsed_from_config(
        self, isolated_env, monkeypatch
    ):
        import builtins
        import types

        config = types.SimpleNamespace(
            show=lambda: "BLAS_INFO=mkl, CXX_FLAGS=-DUSE_FBGEMM -DUSE_FBGEMM_GENAI -O2"
        )
        fake_torch = types.SimpleNamespace(__config__=config)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            if name == "fbgemm_gpu":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_fbgemm(reasons)
        assert block["pytorch_use_fbgemm"] is True
        assert block["pytorch_use_fbgemm_genai"] is True

    def test_pytorch_use_fbgemm_off_when_not_in_flags(
        self, isolated_env, monkeypatch
    ):
        """A ROCm wheel built without FBGEMM should yield False (not None)."""
        import builtins
        import types

        config = types.SimpleNamespace(
            show=lambda: "BLAS_INFO=mkl, CXX_FLAGS=-O2 -DOTHER_FLAG"
        )
        fake_torch = types.SimpleNamespace(__config__=config)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            if name == "fbgemm_gpu":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_fbgemm(reasons)
        assert block["pytorch_use_fbgemm"] is False
        assert block["pytorch_use_fbgemm_genai"] is False

    def test_use_fbgemm_regex_does_not_match_genai_substring(self):
        """Regression guard: -DUSE_FBGEMM (plain) must not false-positive on
        -DUSE_FBGEMM_GENAI alone.
        """
        text = "CXX_FLAGS=-DUSE_FBGEMM_GENAI -O2"
        assert env_mod._FBGEMM_DEFINE_RE.search(text) is None
        assert env_mod._FBGEMM_GENAI_DEFINE_RE.search(text) is not None


# ---------------------------------------------------------------------------
# AITER
# ---------------------------------------------------------------------------


class TestAiterBlock:
    @staticmethod
    def _force_no_aiter_dist(monkeypatch):
        """Make `importlib.metadata.version("amd_aiter" | "aiter")` raise
        PackageNotFoundError so tests are deterministic regardless of
        whether a developer / CI host happens to have the dist installed.
        """
        import importlib.metadata as _md
        real_version = _md.version

        def fake_version(name):
            if name in ("amd_aiter", "aiter"):
                raise _md.PackageNotFoundError(name)
            return real_version(name)

        monkeypatch.setattr(_md, "version", fake_version)

    def test_aiter_absence_does_not_record_reason(self, all_disabled):
        """Most production hosts don't have aiter; suppress the noise."""
        self._force_no_aiter_dist(all_disabled)
        reasons: list[str] = []
        block = env_mod._capture_aiter(reasons)
        assert block == {
            "package_version": None,
            "package_dist_name": None,
            "commit": None,
            "hsa_tree": None,
        }
        assert all("aiter not importable" not in r for r in reasons)

    def test_aiter_with_version_returns_string(self, isolated_env, monkeypatch):
        import builtins
        import types

        self._force_no_aiter_dist(monkeypatch)
        real_import = builtins.__import__
        fake_aiter = types.SimpleNamespace(__version__="0.1.4+rocm7.2.gitabc")

        def fake_import(name, *args, **kwargs):
            if name == "aiter":
                return fake_aiter
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aiter(reasons)
        # +rocm... local segment carries no `+g<sha>` -> commit stays None.
        # No amd_aiter dist (forced) -> dist_name None.
        assert block == {
            "package_version": "0.1.4+rocm7.2.gitabc",
            "package_dist_name": None,
            "commit": None,
            "hsa_tree": None,
        }
        assert reasons == []

    def test_aiter_setuptools_scm_commit_extracted(
        self, isolated_env, monkeypatch
    ):
        """`+g<sha>` setuptools_scm local-version segment -> commit field.

        Matches the AMD-internal ROCm/PyTorch image-tag convention where
        `aiter-9a469a6` in the tag mirrors the `+g9a469a608` segment in
        amd_aiter's version.
        """
        import builtins
        import types

        self._force_no_aiter_dist(monkeypatch)
        fake_aiter = types.SimpleNamespace(
            __version__="0.1.11.dev32+g9a469a608"
        )
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (
                fake_aiter if name == "aiter" else real_import(name, *a, **kw)
            ),
        )
        block = env_mod._capture_aiter([])
        assert block["package_version"] == "0.1.11.dev32+g9a469a608"
        assert block["commit"] == "9a469a608"

    def test_aiter_dist_metadata_fallback_populates_dist_name(
        self, isolated_env, monkeypatch
    ):
        """Path 3: aiter import succeeds but lacks __version__ AND
        aiter._version; importlib.metadata.version("amd_aiter") provides
        both the version string and the dist_name signal.
        """
        import builtins
        import importlib.metadata as _md
        import types

        # aiter module without __version__ and no _version submodule.
        fake_aiter = types.SimpleNamespace()
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (
                fake_aiter if name == "aiter" else real_import(name, *a, **kw)
            ),
        )
        # amd_aiter dist resolves; aiter dist does not.
        real_version = _md.version

        def fake_version(name):
            if name == "amd_aiter":
                return "0.1.11.dev32+g9a469a608"
            if name == "aiter":
                raise _md.PackageNotFoundError(name)
            return real_version(name)

        monkeypatch.setattr(_md, "version", fake_version)
        block = env_mod._capture_aiter([])
        assert block["package_version"] == "0.1.11.dev32+g9a469a608"
        assert block["package_dist_name"] == "amd_aiter"
        assert block["commit"] == "9a469a608"


# ---------------------------------------------------------------------------
# Real-torch integration -- complements TestPytorchVersion's monkeypatched
# unit tests by exercising _capture_pytorch_version against the actual
# torch wheel installed in the venv. Skipped when torch isn't importable.
# ---------------------------------------------------------------------------


class TestPytorchVersionRealTorch:
    """Integration test: real `import torch`, real ``__version__``.

    The unit tests in ``TestPytorchVersion`` use ``SimpleNamespace`` fakes
    -- they cover the contract but cannot catch the class of bug where
    real torch's ``__version__`` is some unusual type or where the real
    install path masks an attribute the fake doesn't model. This class
    runs the probe against the actual installed torch.

    Tagged ``@pytest.mark.rocm`` so it can be deselected on hosts that
    aren't validating ROCm builds (``pytest -m 'not rocm'``); on hosts
    without torch installed it self-skips via ``pytest.importorskip``.
    """

    @pytest.mark.rocm
    def test_capture_matches_real_torch_version(self):
        torch = pytest.importorskip(
            "torch",
            reason="real-torch integration test requires torch in the venv",
        )
        reasons: list[str] = []
        captured = env_mod._capture_pytorch_version(reasons)
        assert captured is not None, (
            f"probe returned None against real torch; reasons: {reasons}"
        )
        # The probe stringifies, but it should still equal the source
        # ``__version__`` exactly -- never the literal "None", never
        # truncated, never a repr().
        assert captured == str(torch.__version__)
        assert captured != "None"
        assert reasons == []

    @pytest.mark.rocm
    def test_pytorch_build_git_commit_matches_real_torch(self):
        """``pytorch_build.git_commit`` must equal ``torch.version.git_version``.
        That field is the linchpin -- it deterministically pins every
        third_party submodule for GitHub-tree lookup.
        """
        torch = pytest.importorskip("torch")
        snapshot = collect_env()
        expected = getattr(torch.version, "git_version", None) or None
        assert snapshot.pytorch_build["git_commit"] == expected

    @pytest.mark.rocm
    def test_full_collect_env_against_real_torch(self):
        """End-to-end: real collect_env() returns a snapshot whose
        pytorch_version matches torch.__version__ exactly. Catches any
        wiring break between _capture_pytorch_version and the
        EnvSnapshot constructor that the unit tests miss.
        """
        torch = pytest.importorskip(
            "torch",
            reason="real-torch integration test requires torch in the venv",
        )
        snapshot = collect_env()
        assert snapshot.pytorch_version == str(torch.__version__)
        assert "pytorch_version" not in " ".join(snapshot.partial_reasons), (
            "pytorch_version probe recorded a partial reason against the "
            f"real torch install: {snapshot.partial_reasons}"
        )


# ---------------------------------------------------------------------------
# Generic helpers extracted during the refactor
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Host system (kernel + glibc + machine arch)
# ---------------------------------------------------------------------------


class TestHostBlock:
    def test_block_keys_stable(self, all_disabled):
        snapshot = collect_env()
        assert set(snapshot.host.keys()) == {
            "kernel_release",
            "kernel_version",
            "machine",
            "glibc_version",
        }

    def test_real_host_populates_fields(self, all_disabled):
        """Smoke test against the real host -- on any Linux/macOS test
        runner ``os.uname()`` and ``os.confstr`` work, so all four
        fields should be non-null. The all_disabled fixture doesn't
        sabotage stdlib syscalls.
        """
        snapshot = collect_env()
        host = snapshot.host
        assert host["kernel_release"] is not None
        assert host["machine"] is not None
        # glibc may legitimately be empty on non-glibc systems (musl,
        # macOS) -- assert only the field is present, not its value.
        assert "glibc_version" in host

    def test_uname_failure_records_reason(self, all_disabled, monkeypatch):
        def boom():
            raise OSError("no uname for you")

        monkeypatch.setattr(env_mod.os, "uname", boom)
        reasons: list[str] = []
        block = env_mod._capture_host(reasons)
        assert block["kernel_release"] is None
        assert any(r.startswith("host.kernel_release") for r in reasons)

    def test_glibc_version_strips_redundant_prefix(
        self, all_disabled, monkeypatch
    ):
        """``os.confstr`` returns ``"glibc 2.35"`` on Linux. The
        ``glibc `` prefix duplicates the field name -- store the bare
        version string so consumers comparing across hosts do
        ``"2.35" == "2.35"`` rather than dealing with a stray prefix.
        """
        monkeypatch.setattr(env_mod.os, "confstr", lambda name: "glibc 2.35")
        reasons: list[str] = []
        block = env_mod._capture_host(reasons)
        assert block["glibc_version"] == "2.35"

    def test_glibc_version_unprefixed_value_passes_through(
        self, all_disabled, monkeypatch
    ):
        """Defensive: if confstr ever returns a bare version string
        (some libcs / future Python versions might), don't munge it.
        """
        monkeypatch.setattr(env_mod.os, "confstr", lambda name: "2.42")
        reasons: list[str] = []
        block = env_mod._capture_host(reasons)
        assert block["glibc_version"] == "2.42"


# ---------------------------------------------------------------------------
# amdgpu_driver (host-kernel scope KFD/AMDGPU driver identity, schema 1.10)
# ---------------------------------------------------------------------------


class TestQueryAmdgpuPackage:
    """_query_amdgpu_package(): glob-capable dpkg-then-rpm, no shell pipe."""

    @staticmethod
    def _tools(present, outputs):
        """(fake_which, fake_run) for the given present tools + argv->(_rc, out) map."""

        def fake_which(name):
            return f"/usr/bin/{name}" if name in present else None

        def fake_run(cmd, **kwargs):
            rc, out = outputs.get(tuple(cmd), (1, ""))
            return subprocess.CompletedProcess(
                args=cmd, returncode=rc, stdout=out, stderr=""
            )

        return fake_which, fake_run

    def test_dpkg_hit_wins_and_reports_full_name(self, isolated_env, monkeypatch):
        fw, fr = self._tools(
            {"dpkg-query", "rpm"},
            {
                ("dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                 "amdgpu-dkms*"): (
                    0,
                    "amdgpu-dkms\t1:6.14.14-2212064.24.04\n",
                ),
            },
        )
        monkeypatch.setattr(env_mod.shutil, "which", fw)
        monkeypatch.setattr(env_mod.subprocess, "run", fr)
        assert env_mod._query_amdgpu_package() == {
            "name": "amdgpu-dkms",
            "version": "1:6.14.14-2212064.24.04",
            "manager": "dpkg",
            "full_name": "amdgpu-dkms=1:6.14.14-2212064.24.04",
        }

    def test_falls_back_to_rpm_when_dpkg_absent(self, isolated_env, monkeypatch):
        # No dpkg-query on PATH (RHEL); rpm answers with NAME/VER-REL/ARCH.
        fw, fr = self._tools(
            {"rpm"},
            {
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-dkms*"): (0, ""),
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-kmod*"): (
                    0,
                    "amdgpu-kmod\t6.14.14-2212064.el9\tx86_64\n",
                ),
            },
        )
        monkeypatch.setattr(env_mod.shutil, "which", fw)
        monkeypatch.setattr(env_mod.subprocess, "run", fr)
        assert env_mod._query_amdgpu_package() == {
            "name": "amdgpu-kmod",
            "version": "6.14.14-2212064.el9",
            "manager": "rpm",
            "full_name": "amdgpu-kmod-6.14.14-2212064.el9.x86_64",
        }

    def test_kernel_suffixed_rpm_name_is_matched(self, isolated_env, monkeypatch):
        # The reported real-world case: the fbk kernel bakes the release
        # into the RPM package NAME, so an exact `rpm -q amdgpu-kmod` finds
        # nothing. The glob `amdgpu-kmod*` catches it, and full_name
        # reconstructs the complete NVRA an operator would diff on.
        real_name = (
            "amdgpu-kmod-6.9.0-0_fbk10_brcmrdma13_141_g9b20106afb70"
        )
        fw, fr = self._tools(
            {"rpm"},
            {
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-dkms*"): (0, ""),
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-kmod*"): (
                    0,
                    f"{real_name}\t6.14.14.000000-2226257.1\tx86_64\n",
                ),
            },
        )
        monkeypatch.setattr(env_mod.shutil, "which", fw)
        monkeypatch.setattr(env_mod.subprocess, "run", fr)
        result = env_mod._query_amdgpu_package()
        assert result["name"] == "amdgpu-kmod"  # stable family label
        assert result["manager"] == "rpm"
        assert result["version"] == "6.14.14.000000-2226257.1"
        assert result["full_name"] == (
            "amdgpu-kmod-6.9.0-0_fbk10_brcmrdma13_141_g9b20106afb70"
            "-6.14.14.000000-2226257.1.x86_64"
        )

    def test_dpkg_present_but_package_missing_falls_to_rpm(
        self, isolated_env, monkeypatch
    ):
        # dpkg-query runs but returns no matching line (package not
        # installed under dpkg); rpm then finds it. Exercises the "manager
        # present, package absent" branch rather than "manager absent".
        fw, fr = self._tools(
            {"dpkg-query", "rpm"},
            {
                ("dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                 "amdgpu-dkms*"): (1, ""),
                ("dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                 "amdgpu-kmod*"): (1, ""),
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-dkms*"): (
                    0,
                    "amdgpu-dkms\t6.14.14-2212064.el9\tnoarch\n",
                ),
            },
        )
        monkeypatch.setattr(env_mod.shutil, "which", fw)
        monkeypatch.setattr(env_mod.subprocess, "run", fr)
        assert env_mod._query_amdgpu_package() == {
            "name": "amdgpu-dkms",
            "version": "6.14.14-2212064.el9",
            "manager": "rpm",
            "full_name": "amdgpu-dkms-6.14.14-2212064.el9.noarch",
        }

    def test_firmware_package_is_ignored(self, isolated_env, monkeypatch):
        # The glob amdgpu-dkms* / amdgpu-kmod* would never match firmware,
        # but a defensive "firmware" filter guards against a candidate list
        # change; assert a firmware line in the output is skipped.
        fw, fr = self._tools(
            {"rpm"},
            {
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-dkms*"): (
                    0,
                    "amdgpu-dkms-firmware\t6.14.14-1\tnoarch\n",
                ),
                ("rpm", "-qa", "--qf",
                 "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n",
                 "amdgpu-kmod*"): (0, ""),
            },
        )
        monkeypatch.setattr(env_mod.shutil, "which", fw)
        monkeypatch.setattr(env_mod.subprocess, "run", fr)
        assert env_mod._query_amdgpu_package() is None

    def test_no_package_manager_returns_none(self, isolated_env, monkeypatch):
        fw, fr = self._tools(set(), {})
        monkeypatch.setattr(env_mod.shutil, "which", fw)
        monkeypatch.setattr(env_mod.subprocess, "run", fr)
        assert env_mod._query_amdgpu_package() is None

    def test_timeout_is_fail_soft(self, isolated_env, monkeypatch):
        def boom(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(
            env_mod.shutil, "which", lambda n: f"/usr/bin/{n}"
        )
        monkeypatch.setattr(env_mod.subprocess, "run", boom)
        # Every query times out -> None, no raise.
        assert env_mod._query_amdgpu_package() is None


_MODINFO_AMDGPU = (
    "filename:       /lib/modules/6.8.0/updates/dkms/amdgpu.ko\n"
    "version:        6.14.14\n"
    "srcversion:     A1B2C3D4E5F6A7B8C9D0\n"
    "license:        GPL and additional rights\n"
)


class TestModinfoField:
    def test_parses_named_field(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda n: "/sbin/modinfo")
        monkeypatch.setattr(
            env_mod.subprocess,
            "run",
            lambda cmd, **k: subprocess.CompletedProcess(
                cmd, 0, stdout=_MODINFO_AMDGPU, stderr=""
            ),
        )
        assert env_mod._modinfo_field("amdgpu", "version") == "6.14.14"
        assert (
            env_mod._modinfo_field("amdgpu", "srcversion")
            == "A1B2C3D4E5F6A7B8C9D0"
        )

    def test_absent_modinfo_returns_none(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda n: None)
        assert env_mod._modinfo_field("amdgpu", "version") is None

    def test_nonzero_exit_returns_none(self, isolated_env, monkeypatch):
        # modinfo exits non-zero when the module isn't found (GPU-less box).
        monkeypatch.setattr(env_mod.shutil, "which", lambda n: "/sbin/modinfo")
        monkeypatch.setattr(
            env_mod.subprocess,
            "run",
            lambda cmd, **k: subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="modinfo: ERROR: Module amdgpu not found.\n"
            ),
        )
        assert env_mod._modinfo_field("amdgpu", "version") is None


class TestAmdgpuDriver:
    """_capture_amdgpu_driver(): host-scope, degrades cleanly with no GPU."""

    def test_block_keys_stable(self, all_disabled):
        snapshot = collect_env()
        assert set(snapshot.amdgpu_driver.keys()) == {
            "scope",
            "status",
            "package_name",
            "package_version",
            "package_full_name",
            "package_manager",
            "module_version",
            "module_srcversion",
            "kmd_version",
            "kfd_device_present",
            "kfd_sysfs_present",
        }
        assert snapshot.amdgpu_driver["scope"] == "host_kernel"

    def test_no_gpu_is_documented_absence_not_partial(
        self, isolated_env, monkeypatch
    ):
        # Nothing resolvable: no package manager, no modinfo, no KFD nodes,
        # no kmd_version. This is the dev-machine / CPU-runner case.
        monkeypatch.setattr(env_mod.shutil, "which", lambda n: None)
        monkeypatch.setattr(env_mod, "_path_exists", lambda p: False)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": None}, reasons)
        assert block["status"] == "absent"
        assert block["package_version"] is None
        assert block["kfd_device_present"] is False
        # The whole point: a GPU-less host must NOT pollute partial_reasons.
        assert reasons == []

    def test_present_full_capture(self, isolated_env, monkeypatch):
        monkeypatch.setattr(
            env_mod,
            "_query_amdgpu_package",
            lambda: {
                "name": "amdgpu-dkms",
                "version": "1:6.14.14-2212064.24.04",
                "manager": "dpkg",
                "full_name": "amdgpu-dkms=1:6.14.14-2212064.24.04",
            },
        )
        monkeypatch.setattr(
            env_mod,
            "_modinfo_field",
            lambda mod, field: {
                "version": "6.14.14",
                "srcversion": "A1B2C3D4E5F6A7B8C9D0",
            }.get(field),
        )
        monkeypatch.setattr(env_mod, "_path_exists", lambda p: True)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": "6.16.13"}, reasons)
        assert block["status"] == "present"
        assert block["package_name"] == "amdgpu-dkms"
        assert block["package_version"] == "1:6.14.14-2212064.24.04"
        assert block["package_full_name"] == "amdgpu-dkms=1:6.14.14-2212064.24.04"
        assert block["package_manager"] == "dpkg"
        assert block["module_version"] == "6.14.14"
        assert block["module_srcversion"] == "A1B2C3D4E5F6A7B8C9D0"
        # kmd_version is REUSED from the passed rocm block (no second read).
        assert block["kmd_version"] == "6.16.13"
        assert block["kfd_device_present"] is True
        assert reasons == []

    def test_kernel_suffixed_rpm_full_name_flows_into_block(
        self, isolated_env, monkeypatch
    ):
        # End-to-end at the capture layer: the resolver returns a
        # kernel-suffixed rpm identity and the block preserves the full
        # NVRA in package_full_name while package_name stays the family.
        full = (
            "amdgpu-kmod-6.9.0-0_fbk10_brcmrdma13_141_g9b20106afb70"
            "-6.14.14.000000-2226257.1.x86_64"
        )
        monkeypatch.setattr(
            env_mod,
            "_query_amdgpu_package",
            lambda: {
                "name": "amdgpu-kmod",
                "version": "6.14.14.000000-2226257.1",
                "manager": "rpm",
                "full_name": full,
            },
        )
        monkeypatch.setattr(env_mod, "_modinfo_field", lambda mod, field: None)
        monkeypatch.setattr(env_mod, "_path_exists", lambda p: False)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": None}, reasons)
        assert block["package_name"] == "amdgpu-kmod"
        assert block["package_full_name"] == full
        assert block["status"] == "present"
        # A resolvable package means NO conflict reason.
        assert reasons == []

    def test_kmd_version_reused_from_rocm_block(self, isolated_env, monkeypatch):
        # Even with everything else absent, a kmd_version in the rocm block
        # flows through and makes the block "present".
        monkeypatch.setattr(env_mod, "_query_amdgpu_package", lambda: None)
        monkeypatch.setattr(env_mod, "_modinfo_field", lambda mod, field: None)
        monkeypatch.setattr(env_mod, "_path_exists", lambda p: False)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": "6.16.13"}, reasons)
        assert block["kmd_version"] == "6.16.13"
        assert block["status"] == "present"

    def test_loaded_but_unpackaged_records_conflict_reason(
        self, isolated_env, monkeypatch
    ):
        # KFD node + modinfo present, but no dpkg/rpm package resolvable:
        # an unusual, diagnosable state that DOES earn one partial reason.
        monkeypatch.setattr(env_mod, "_query_amdgpu_package", lambda: None)
        monkeypatch.setattr(
            env_mod,
            "_modinfo_field",
            lambda mod, field: "6.14.14" if field == "version" else None,
        )
        monkeypatch.setattr(env_mod, "_path_exists", lambda p: True)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": None}, reasons)
        assert block["status"] == "present"
        assert block["package_version"] is None
        assert any(
            r.startswith("amdgpu_driver.package_version") for r in reasons
        )

    def test_kfd_passthrough_container_is_not_a_conflict(
        self, isolated_env, monkeypatch
    ):
        # The normal ROCm-container case: /dev/kfd is mounted from the host
        # kernel, but the container filesystem has no amdgpu package and no
        # /lib/modules entry (modinfo returns None). This is NOT a conflict
        # -- /dev/kfd is host state while dpkg/rpm + modinfo read the
        # container filesystem -- so it must NOT pollute partial_reasons.
        monkeypatch.setattr(env_mod, "_query_amdgpu_package", lambda: None)
        monkeypatch.setattr(env_mod, "_modinfo_field", lambda mod, field: None)
        monkeypatch.setattr(
            env_mod,
            "_path_exists",
            lambda p: p == env_mod.KFD_DEVICE_NODE,
        )
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": None}, reasons)
        assert block["kfd_device_present"] is True
        assert block["package_version"] is None
        assert block["status"] == "present"
        assert reasons == []

    def test_kfd_presence_uses_module_constants(self, isolated_env, monkeypatch):
        # kfd_device_present / kfd_sysfs_present must key off the module
        # path constants so tests (and future path moves) stay in sync.
        seen: list = []

        def fake_exists(p):
            seen.append(p)
            return p == env_mod.KFD_DEVICE_NODE

        monkeypatch.setattr(env_mod, "_query_amdgpu_package", lambda: None)
        monkeypatch.setattr(env_mod, "_modinfo_field", lambda mod, field: None)
        monkeypatch.setattr(env_mod, "_path_exists", fake_exists)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": None}, reasons)
        assert block["kfd_device_present"] is True
        assert block["kfd_sysfs_present"] is False
        assert env_mod.KFD_DEVICE_NODE in seen
        assert env_mod.KFD_SYSFS_DIR in seen

    def test_empty_matches_capture_all_absent(self, isolated_env, monkeypatch):
        # _empty_amdgpu_driver() must be byte-identical to a real all-absent
        # capture, so the dataclass default never diverges from reality.
        monkeypatch.setattr(env_mod, "_query_amdgpu_package", lambda: None)
        monkeypatch.setattr(env_mod, "_modinfo_field", lambda mod, field: None)
        monkeypatch.setattr(env_mod, "_path_exists", lambda p: False)
        reasons: list[str] = []
        block = env_mod._capture_amdgpu_driver({"kmd_version": None}, reasons)
        assert block == env_mod._empty_amdgpu_driver()


# ---------------------------------------------------------------------------
# MIOpen (deep-learning primitives -- conv kernels)
# ---------------------------------------------------------------------------


class TestMiopen:
    def test_block_keys_stable(self, all_disabled):
        snapshot = collect_env()
        assert set(snapshot.miopen.keys()) == {
            "rocm_release_tweak",
            "package_version",
            "lib_hash",
            "kernel_db_revision",
        }

    def test_partial_reasons_use_miopen_prefix(self, all_disabled):
        reasons: list[str] = []
        env_mod._capture_miopen(reasons)
        assert reasons
        assert all(r.startswith("miopen.") for r in reasons), reasons

    def test_full_capture_against_real_files(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        header_dir = tmp_path / "include"
        header_dir.mkdir()
        (header_dir / "version.h").write_text(
            "#define MIOPEN_VERSION_MAJOR 3\n"
            "#define MIOPEN_VERSION_MINOR 5\n"
            "#define MIOPEN_VERSION_PATCH 1\n"
            "#define MIOPEN_VERSION_TWEAK dabb6df2b9\n"
        )
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "libMIOpen.so").write_bytes(b"miopen-bytes")
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        (db_dir / "gfx942_64.db.txt").write_text("k1")
        (db_dir / "gfx942_64.HIP.fdb.txt").write_text("k2")

        monkeypatch.setattr(env_mod, "MIOPEN_VERSION_HEADER", header_dir / "version.h")
        monkeypatch.setattr(env_mod, "MIOPEN_LIB_DIR", lib_dir)
        monkeypatch.setattr(env_mod, "MIOPEN_KERNEL_DB_DIR", db_dir)
        reasons: list[str] = []
        block = env_mod._capture_miopen(reasons)
        assert block["rocm_release_tweak"] == "dabb6df2b9"
        assert block["package_version"] == "3.5.1"
        assert block["lib_hash"] is not None
        assert block["kernel_db_revision"] is not None
        assert block["kernel_db_revision"].startswith("filenames-sha256:")
        assert reasons == []


# ---------------------------------------------------------------------------
# RCCL (collectives, NCCL-compatible API)
# ---------------------------------------------------------------------------


class TestRccl:
    def test_block_keys_stable(self, all_disabled):
        snapshot = collect_env()
        assert set(snapshot.rccl.keys()) == {
            "version_code",
            "version",
            "lib_hash",
            "net_plugin_mode",
            "plugin_path",
            "plugin_lib_hash",
            "anp_lib_hash",
            "net_lib_hash",
        }

    def test_decode_modern_version_code(self):
        # 22707 = 2*10000 + 27*100 + 7  (modern scheme; X=2, Y>=9)
        code, version = env_mod._parse_rccl_header(
            "#define NCCL_VERSION_CODE 22707\n"
        )
        assert code == 22707
        assert version == "2.27.7"

    def test_decode_legacy_version_code(self):
        # 2807 = 2*1000 + 8*100 + 7  (legacy scheme; X<=2, Y<=8)
        code, version = env_mod._parse_rccl_header(
            "#define NCCL_VERSION_CODE 2807\n"
        )
        assert code == 2807
        assert version == "2.8.7"

    def test_empty_header_returns_none_pair(self):
        assert env_mod._parse_rccl_header("") == (None, None)
        assert env_mod._parse_rccl_header(None) == (None, None)

    def test_full_capture(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        header_dir = tmp_path / "include"
        header_dir.mkdir()
        (header_dir / "rccl.h").write_text(
            "// rccl header\n"
            "#define NCCL_VERSION_CODE 22707\n"
        )
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "librccl.so").write_bytes(b"rccl-bytes")

        monkeypatch.setattr(env_mod, "RCCL_VERSION_HEADER", header_dir / "rccl.h")
        monkeypatch.setattr(env_mod, "RCCL_LIB_DIR", lib_dir)
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["version_code"] == 22707
        assert block["version"] == "2.27.7"
        assert block["lib_hash"] is not None
        # NCCL_NET_PLUGIN unset -> internal mode, no plugin path/hash.
        assert block["net_plugin_mode"] == "internal"
        assert block["plugin_path"] is None
        assert block["plugin_lib_hash"] is None
        assert block["anp_lib_hash"] is None
        assert block["net_lib_hash"] is None
        assert reasons == []

    def _rccl_dirs(self, tmp_path: Path, monkeypatch):
        """Set up a librccl.so + header so the install reads as present.

        Also clears LD_LIBRARY_PATH so plugin resolution is deterministic
        (only the dirs the test sets up are searched).
        """
        header_dir = tmp_path / "include"
        header_dir.mkdir()
        (header_dir / "rccl.h").write_text("#define NCCL_VERSION_CODE 22707\n")
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "librccl.so").write_bytes(b"rccl-bytes")
        monkeypatch.setattr(env_mod, "RCCL_VERSION_HEADER", header_dir / "rccl.h")
        monkeypatch.setattr(env_mod, "RCCL_LIB_DIR", lib_dir)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        return lib_dir

    def test_net_plugin_external_via_absolute_path(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # Real ANP deployment: NCCL_NET_PLUGIN points at an absolute path
        # to librccl-net.so in a user-build tree (NOT /opt/rocm/lib).
        self._rccl_dirs(tmp_path, monkeypatch)
        anp_build = tmp_path / "apps" / "amd-anp" / "build"
        anp_build.mkdir(parents=True)
        plugin = anp_build / "librccl-net.so"
        plugin.write_bytes(b"anp-net-bytes")
        isolated_env.setenv("NCCL_NET_PLUGIN", str(plugin))
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "external"
        assert block["plugin_path"] == str(plugin)
        assert block["plugin_lib_hash"] is not None
        assert reasons == []

    def test_plugin_lib_hash_is_resolved_file_not_sibling(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # plugin_lib_hash must hash exactly the resolved plugin file, not
        # a higher-versioned sibling in the same dir (the _hash_shared_library
        # fallback risk). Place a sibling with different bytes alongside.
        self._rccl_dirs(tmp_path, monkeypatch)
        anp_build = tmp_path / "apps" / "amd-anp" / "build"
        anp_build.mkdir(parents=True)
        plugin = anp_build / "librccl-net.so"
        plugin.write_bytes(b"the-real-resolved-plugin")
        sibling = anp_build / "librccl-net.so.9.9.9"
        sibling.write_bytes(b"a-different-sibling-build")
        isolated_env.setenv("NCCL_NET_PLUGIN", str(plugin))
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "external"
        assert block["plugin_lib_hash"] == env_mod._hash_file_path(plugin)
        assert block["plugin_lib_hash"] != env_mod._hash_file_path(sibling)
        assert reasons == []

    def test_net_plugin_external_via_ld_library_path(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # Bare-name form: NCCL_NET_PLUGIN=librccl-net.so resolved through
        # an LD_LIBRARY_PATH entry.
        self._rccl_dirs(tmp_path, monkeypatch)
        anp_build = tmp_path / "apps" / "amd-anp" / "build"
        anp_build.mkdir(parents=True)
        plugin = anp_build / "librccl-net.so"
        plugin.write_bytes(b"anp-net-bytes")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(anp_build))
        isolated_env.setenv("NCCL_NET_PLUGIN", "librccl-net.so")
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "external"
        assert block["plugin_path"] == str(plugin)
        assert block["plugin_lib_hash"] is not None
        assert reasons == []

    def test_net_plugin_external_bare_name_normalised(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # NCCL normalises a bare "ncclnet"-style name; we accept the
        # common forms. Here the env value omits the lib prefix / .so.
        self._rccl_dirs(tmp_path, monkeypatch)
        anp_build = tmp_path / "anp"
        anp_build.mkdir()
        plugin = anp_build / "librccl-net.so"
        plugin.write_bytes(b"anp-net-bytes")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(anp_build))
        isolated_env.setenv("NCCL_NET_PLUGIN", "rccl-net")
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "external"
        assert block["plugin_path"] == str(plugin)
        assert reasons == []

    def test_net_plugin_internal_when_env_unset(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # An ANP .so present in the rccl lib dir is still "internal" when
        # the operator did not opt in via NCCL_NET_PLUGIN.
        lib_dir = self._rccl_dirs(tmp_path, monkeypatch)
        (lib_dir / "librccl-net.so").write_bytes(b"anp-bytes")
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "internal"
        assert block["plugin_path"] is None
        assert block["plugin_lib_hash"] is None
        # Packaged-install best-effort scan still records the lib-dir hash.
        assert block["net_lib_hash"] is not None
        assert reasons == []

    def test_net_plugin_internal_when_env_whitespace_only(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # NCCL_NET_PLUGIN exported as whitespace is equivalent to unset:
        # the documented "internal" case, NOT an unresolvable "unknown".
        self._rccl_dirs(tmp_path, monkeypatch)
        isolated_env.setenv("NCCL_NET_PLUGIN", "   ")
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "internal"
        assert block["plugin_path"] is None
        assert block["plugin_lib_hash"] is None
        assert not any(r.startswith("rccl.net_plugin_mode") for r in reasons)

    def test_net_plugin_unknown_when_env_set_but_unresolvable(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # Launcher asked for a plugin but it can't be found anywhere ->
        # expected-but-failed capture: unknown + a partial reason.
        self._rccl_dirs(tmp_path, monkeypatch)
        isolated_env.setenv("NCCL_NET_PLUGIN", "/nonexistent/librccl-net.so")
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "unknown"
        assert block["plugin_path"] is None
        assert block["plugin_lib_hash"] is None
        assert any(r.startswith("rccl.net_plugin_mode") for r in reasons)

    def test_resolve_net_plugin_empty_returns_none(self):
        assert env_mod._resolve_net_plugin("") is None
        assert env_mod._resolve_net_plugin("   ") is None

    def test_net_plugin_directory_is_not_external(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # NCCL_NET_PLUGIN pointing at a directory (or any non-regular
        # file) must NOT resolve -- the runtime can't dlopen a directory.
        self._rccl_dirs(tmp_path, monkeypatch)
        a_dir = tmp_path / "not-a-plugin-dir"
        a_dir.mkdir()
        isolated_env.setenv("NCCL_NET_PLUGIN", str(a_dir))
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "unknown"
        assert block["plugin_path"] is None
        assert block["plugin_lib_hash"] is None
        assert any(r.startswith("rccl.net_plugin_mode") for r in reasons)

    def test_resolve_net_plugin_oserror_on_explicit_path_is_failsoft(
        self, monkeypatch
    ):
        # Path.is_file() can raise (e.g. PermissionError when a parent dir
        # is not traversable -- and on CPython < 3.12 it does NOT swallow
        # it). The resolver must treat that as "does not resolve" and
        # return None rather than letting the exception escape and trip
        # the disaster snapshot. Force the raise so the contract is
        # verified on every Python version, not just <3.12.
        def _boom(self):  # noqa: ANN001
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(env_mod.Path, "is_file", _boom)
        assert env_mod._resolve_net_plugin("/restricted/dir/librccl-net.so") is None

    def test_net_plugin_unknown_does_not_break_capture_on_oserror(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # End-to-end: an explicit NCCL_NET_PLUGIN whose existence check
        # raises OSError must degrade to net_plugin_mode="unknown" + a
        # reason, and _capture_rccl must NOT raise.
        self._rccl_dirs(tmp_path, monkeypatch)
        isolated_env.setenv("NCCL_NET_PLUGIN", "/restricted/dir/librccl-net.so")
        real_is_file = env_mod.Path.is_file

        def _maybe_boom(self):  # noqa: ANN001
            if str(self) == "/restricted/dir/librccl-net.so":
                raise PermissionError(13, "Permission denied")
            return real_is_file(self)

        monkeypatch.setattr(env_mod.Path, "is_file", _maybe_boom)
        reasons: list[str] = []
        block = env_mod._capture_rccl(reasons)
        assert block["net_plugin_mode"] == "unknown"
        assert block["plugin_path"] is None
        assert block["plugin_lib_hash"] is None
        assert any(r.startswith("rccl.net_plugin_mode") for r in reasons)


# ---------------------------------------------------------------------------
# Multi-vendor NIC / RoCE fabric capture (issue #202, schema 1.7)
# ---------------------------------------------------------------------------


# Real output captured from a live node: 8x BCM57608 (Broadcom) + 2x
# ConnectX-7 (CX7), no AINIC. Used verbatim as parser fixtures.
_REAL_LSPCI_CX7 = (
    "31:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]\n"
    "31:00.1 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]\n"
)
_REAL_LSPCI_BROADCOM = (
    "03:00.0 Ethernet controller: Broadcom Inc. and subsidiaries BCM57608 "
    "25Gb/50Gb/100Gb/200Gb/400Gb Ethernet (rev 11)\n"
)
_REAL_IBV_DEVICES = (
    "    device                 node GUID\n"
    "    ------              ----------------\n"
    "    bnxt_re0            d604e6fffe3e3890\n"
    "    bnxt_re1            d604e6fffe3e4348\n"
)
_REAL_RDMA_LINK = (
    "link bnxt_re0/1 state ACTIVE physical_state LINK_UP netdev benic7p1 \n"
    "link bnxt_re1/1 state ACTIVE physical_state LINK_UP netdev benic8p1 \n"
)
# ethtool -i output captured verbatim from the real node (benic1p1 /
# fenic0). Note: driver "version:" is a kernel-ish build string (same for
# both NICs); firmware-version carries the meaningful per-NIC value, and
# the CX7 form has a parenthesised suffix that must survive parsing.
_REAL_ETHTOOL_BNXT = (
    "driver: bnxt_en\n"
    "version: 6.9.0-0_fbk10_brcmrdma13_141_g9\n"
    "firmware-version: 232.0.219.16/pkg 232.1.196.16\n"
    "expansion-rom-version: \n"
    "bus-info: 0000:f3:00.0\n"
)
_REAL_ETHTOOL_CX7 = (
    "driver: mlx5_core\n"
    "version: 6.9.0-0_fbk10_brcmrdma13_141_g9\n"
    "firmware-version: 28.36.1010 (FB_0000000038)\n"
    "expansion-rom-version: \n"
    "bus-info: 0000:31:00.0\n"
)
# Real output captured from a live AINIC host (8x gfx950, PR #208 review):
# the RoCE devices are named rdma0..rdma7 (NOT ionic_*) and the netdevs
# tw-eth0..7 -- the case that broke name-prefix matching.
_AINIC_IBV_DEVICES = (
    "    device                 node GUID\n"
    "    ------              ----------------\n"
    "    rdma0            ba0a90fffe1c0a00\n"
    "    rdma1            ba0a90fffe1c0a01\n"
    "    rdma2            ba0a90fffe1c0a02\n"
    "    rdma3            ba0a90fffe1c0a03\n"
    "    rdma4            ba0a90fffe1c0a04\n"
    "    rdma5            ba0a90fffe1c0a05\n"
    "    rdma6            ba0a90fffe1c0a06\n"
    "    rdma7            ba0a90fffe1c0a07\n"
)
_AINIC_RDMA_LINK = "".join(
    f"link rdma{i}/1 state ACTIVE physical_state LINK_UP netdev tw-eth{i} \n"
    for i in range(8)
)
_AINIC_DCQCN = (
    "dcqcn-profile 1\n"
    "  enabled true\n"
    "  token-bucket-size 800000\n"
    "  ai-rate 160\n"
    "  hai-rate 300\n"
    "  cnp-dscp 48\n"
)


class TestNicsParsers:
    """Pure-parser unit tests against real captured output shapes."""

    def test_parse_ibv_devices_all_names(self):
        # Returns ALL device names (column 0); vendor binding is the
        # caller's job via sysfs driver, not a name-prefix filter here.
        assert env_mod._parse_ibv_devices(_REAL_IBV_DEVICES) == [
            "bnxt_re0",
            "bnxt_re1",
        ]
        # Generic rdma<N> naming is parsed identically (the AINIC case).
        assert env_mod._parse_ibv_devices(_AINIC_IBV_DEVICES) == [
            f"rdma{i}" for i in range(8)
        ]

    def test_parse_ibv_devices_empty(self):
        assert env_mod._parse_ibv_devices("") == []
        assert env_mod._parse_ibv_devices(None) == []

    def test_parse_rdma_link_shape(self):
        links = env_mod._parse_rdma_link(_REAL_RDMA_LINK)
        assert links == [
            {"device": "bnxt_re0", "state": "ACTIVE", "netdev": "benic7p1"},
            {"device": "bnxt_re1", "state": "ACTIVE", "netdev": "benic8p1"},
        ]

    def test_parse_rdma_link_down_state(self):
        text = (
            "link ionic_2/1 state DOWN physical_state DISABLED netdev enp137s0\n"
        )
        links = env_mod._parse_rdma_link(text)
        assert links == [
            {"device": "ionic_2", "state": "DOWN", "netdev": "enp137s0"}
        ]

    def test_parse_rdma_link_truncated_line_is_failsoft(self):
        # A malformed/truncated line where a key token is last (no value)
        # must NOT raise IndexError -- the field degrades to None.
        text = "link bnxt_re0/1 state\n"
        links = env_mod._parse_rdma_link(text)
        assert links == [
            {"device": "bnxt_re0", "state": None, "netdev": None}
        ]

    def test_sysfs_device_driver_resolves_symlink(self, tmp_path):
        # The authoritative vendor binding: read <name>/device/driver and
        # return its basename. Naming-independent (rdma0 -> ionic).
        root = tmp_path / "class"
        dev = root / "rdma0" / "device"
        dev.mkdir(parents=True)
        os.symlink("../../../bus/pci/drivers/ionic", dev / "driver")
        assert env_mod._sysfs_device_driver(root, "rdma0") == "ionic"
        assert env_mod._sysfs_device_driver(root, "missing") is None
        assert env_mod._sysfs_device_driver(root, "") is None

    def test_split_firmware_version(self):
        # Broadcom glued form -> split; de-dup when the halves are equal.
        assert env_mod._split_firmware_version(
            "232.0.219.16/pkg 232.1.196.16"
        ) == ("232.0.219.16", "232.1.196.16")
        assert env_mod._split_firmware_version(
            "232.0.219.16/pkg 232.0.219.16"
        ) == ("232.0.219.16", None)
        # CX7 parenthesised form (no /pkg) passes through unchanged.
        assert env_mod._split_firmware_version(
            "28.36.1010 (FB_0000000038)"
        ) == ("28.36.1010 (FB_0000000038)", None)
        assert env_mod._split_firmware_version(None) == (None, None)

    def test_parse_ethtool_fields(self):
        assert (
            env_mod._parse_ethtool_field(_REAL_ETHTOOL_BNXT, "firmware-version")
            == "232.0.219.16/pkg 232.1.196.16"
        )
        assert (
            env_mod._parse_ethtool_field(_REAL_ETHTOOL_BNXT, "version")
            == "6.9.0-0_fbk10_brcmrdma13_141_g9"
        )
        assert env_mod._parse_ethtool_field(_REAL_ETHTOOL_BNXT, "driver") == "bnxt_en"
        assert env_mod._parse_ethtool_field(None, "version") is None
        # CX7 firmware carries a parenthesised suffix that must survive.
        assert (
            env_mod._parse_ethtool_field(_REAL_ETHTOOL_CX7, "firmware-version")
            == "28.36.1010 (FB_0000000038)"
        )
        # Empty field value -> None (not "").
        assert (
            env_mod._parse_ethtool_field(_REAL_ETHTOOL_CX7, "expansion-rom-version")
            is None
        )

    def test_parse_nicctl_version_json_and_text(self):
        assert (
            env_mod._parse_nicctl_version('{"firmware": {"version": "1.117.5-a-56"}}')
            == "1.117.5-a-56"
        )
        assert (
            env_mod._parse_nicctl_version("Firmware version 1.1.1-salinaainicbase-1")
            == "1.1.1-salinaainicbase-1"
        )
        assert env_mod._parse_nicctl_version("") is None


class TestNicsCapture:
    """_capture_nics() fail-soft + documented-absence matrix."""

    def _fake_tools(self, present_tools, outputs):
        """Build (fake_which, fake_run) for the given tools + command map.

        outputs maps a tuple-of-argv -> (returncode, stdout). A missing
        key returns ("", rc=0) i.e. empty success (documented absence).
        """
        def fake_which(name):
            return f"/usr/bin/{name}" if name in present_tools else None

        def fake_run(cmd, **kwargs):
            # Strip a leading sudo -n -E wrapper for lookup.
            argv = cmd
            if argv[:3] == ["sudo", "-n", "-E"]:
                argv = argv[3:]
            rc, out = outputs.get(tuple(argv), (0, ""))
            return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=out, stderr="")

        return fake_which, fake_run

    def _build_sysfs(
        self, tmp_path, monkeypatch, *, netdevs=None, ib_devs=None, net_vendor=None
    ):
        """Build fake /sys/class/{net,infiniband} trees + monkeypatch roots.

        netdevs:    {netdev_name: driver}  -> device/driver symlink
        ib_devs:    {ib_name: driver}      -> device/driver symlink
        net_vendor: {netdev_name: "0x14e4"} -> device/vendor file
        """
        net_root = tmp_path / "class_net"
        ib_root = tmp_path / "class_ib"
        net_root.mkdir(parents=True, exist_ok=True)
        ib_root.mkdir(parents=True, exist_ok=True)

        def _mk(root, name, driver=None, vendor=None):
            dev = root / name / "device"
            dev.mkdir(parents=True, exist_ok=True)
            if driver is not None:
                os.symlink(f"../../../bus/pci/drivers/{driver}", dev / "driver")
            if vendor is not None:
                (dev / "vendor").write_text(vendor + "\n")

        netdevs = netdevs or {}
        ib_devs = ib_devs or {}
        net_vendor = net_vendor or {}
        for name in set(netdevs) | set(net_vendor):
            _mk(net_root, name, driver=netdevs.get(name), vendor=net_vendor.get(name))
        for name, driver in ib_devs.items():
            _mk(ib_root, name, driver=driver)
        monkeypatch.setattr(env_mod, "SYS_CLASS_NET", net_root)
        monkeypatch.setattr(env_mod, "SYS_CLASS_INFINIBAND", ib_root)

    def test_no_lspci_records_one_reason_all_unknown(
        self, isolated_env, monkeypatch
    ):
        fake_which, fake_run = self._fake_tools(set(), {})
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        assert set(nics.keys()) == {"ainic", "broadcom", "cx7"}
        assert all(nics[v] == {"present": None} for v in nics)
        assert any("lspci not on PATH" in r for r in reasons)

    def test_vendor_absent_is_documented_absence(
        self, isolated_env, monkeypatch
    ):
        # Only lspci present, returns empty for every vendor id.
        fake_which, fake_run = self._fake_tools({"lspci"}, {})
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        assert nics["ainic"] == {"present": False}
        assert nics["broadcom"] == {"present": False}
        assert nics["cx7"] == {"present": False}
        # No vendor was present -> nothing fell back -> NO partial.
        assert reasons == []

    def test_lspci_failure_is_undeterminable_not_absent(
        self, isolated_env, monkeypatch
    ):
        # lspci runs but FAILS (non-zero) for one vendor: presence is
        # undeterminable -> present=None + a reason, NOT a documented
        # absence. Other vendors (clean empty exit) stay present=False.
        fake_which, fake_run = self._fake_tools(
            {"lspci"}, {("lspci", "-d", "1dd8:1002"): (1, "")}
        )
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        assert nics["ainic"] == {"present": None}
        assert nics["broadcom"] == {"present": False}
        assert nics["cx7"] == {"present": False}
        assert any(r.startswith("nics.ainic") for r in reasons)

    def test_broadcom_tier1_full_capture(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # lspci present for broadcom; Tier-1 tools return real shapes.
        outputs = {
            ("lspci", "-d", "14e4:1760"): (0, _REAL_LSPCI_BROADCOM),
            ("lspci", "-d", "1dd8:1002"): (0, ""),
            ("lspci", "-d", "15b3:1021"): (0, ""),
            ("ethtool", "-i", "benic1p1"): (0, _REAL_ETHTOOL_BNXT),
            ("ibv_devices",): (0, _REAL_IBV_DEVICES),
            ("rdma", "link"): (0, _REAL_RDMA_LINK),
        }
        fake_which, fake_run = self._fake_tools(
            {"lspci", "ethtool", "ibv_devices", "rdma"}, outputs
        )
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        # sysfs: benic1p1 carries vendor 0x14e4 (for netdev->vendor iface
        # discovery); all netdevs + ib devices bind to driver bnxt_en (for
        # the driver-based vendor binding).
        self._build_sysfs(
            tmp_path, monkeypatch,
            netdevs={"benic1p1": "bnxt_en", "benic7p1": "bnxt_en", "benic8p1": "bnxt_en"},
            ib_devs={"bnxt_re0": "bnxt_en", "bnxt_re1": "bnxt_en"},
            net_vendor={"benic1p1": "0x14e4"},
        )

        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        b = nics["broadcom"]
        assert b["present"] is True
        # Broadcom glued "<fw>/pkg <pkg>" is split into firmware + pkg_version.
        assert b["firmware"] == "232.0.219.16"
        assert b["pkg_version"] == "232.1.196.16"
        # sysfs /sys/module/bnxt_en/version absent on the real node ->
        # driver_version falls back to ethtool -i version:.
        assert b["driver_version"] == "6.9.0-0_fbk10_brcmrdma13_141_g9"
        assert b["rdma_devices"] == ["bnxt_re0", "bnxt_re1"]
        assert b["links"][0] == {
            "device": "bnxt_re0",
            "state": "ACTIVE",
            "netdev": "benic7p1",
        }
        assert reasons == []

    def test_cx7_present_with_zero_rdma_is_not_partial(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # CX7 present in lspci but no mlx5_ devices in ibv/rdma (observed
        # on the real node). Must be present:true with empty lists, NO
        # partial reason.
        outputs = {
            ("lspci", "-d", "15b3:1021"): (0, _REAL_LSPCI_CX7),
            ("lspci", "-d", "1dd8:1002"): (0, ""),
            ("lspci", "-d", "14e4:1760"): (0, ""),
            ("ibv_devices",): (0, _REAL_IBV_DEVICES),  # only bnxt_re*, no mlx5_
            ("rdma", "link"): (0, _REAL_RDMA_LINK),
        }
        fake_which, fake_run = self._fake_tools(
            {"lspci", "ibv_devices", "rdma"}, outputs
        )
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        # bnxt_re* devices/netdevs bind to bnxt_en, so the driver-based
        # filter correctly excludes them from cx7 (mlx5_core).
        self._build_sysfs(
            tmp_path, monkeypatch,
            netdevs={"benic7p1": "bnxt_en", "benic8p1": "bnxt_en"},
            ib_devs={"bnxt_re0": "bnxt_en", "bnxt_re1": "bnxt_en"},
        )

        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        c = nics["cx7"]
        assert c["present"] is True
        assert c["rdma_devices"] == []
        assert c["links"] == []
        assert reasons == []

    def test_ainic_tier2_skipped_when_nicctl_absent(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # AINIC present but nicctl not installed -> Tier-2 is a documented
        # absence: no nicctl_version/card keys, no partial.
        outputs = {
            ("lspci", "-d", "1dd8:1002"): (0, "c1:00.0 Ethernet controller: Pensando\n"),
            ("lspci", "-d", "14e4:1760"): (0, ""),
            ("lspci", "-d", "15b3:1021"): (0, ""),
            ("ibv_devices",): (0, ""),
            ("rdma", "link"): (0, ""),
        }
        fake_which, fake_run = self._fake_tools(
            {"lspci", "ibv_devices", "rdma"}, outputs  # no nicctl
        )
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        self._build_sysfs(tmp_path, monkeypatch)  # empty sysfs

        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        a = nics["ainic"]
        assert a["present"] is True
        assert "nicctl_version" not in a  # Tier-2 not attempted
        assert reasons == []

    def test_ainic_tier2_sudo_denied_records_partial(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # AINIC + nicctl present, but sudo denied (rc!=0) -> fields None
        # + partial reasons.
        outputs = {
            ("lspci", "-d", "1dd8:1002"): (0, "c1:00.0 Pensando\n"),
            ("lspci", "-d", "14e4:1760"): (0, ""),
            ("lspci", "-d", "15b3:1021"): (0, ""),
            ("ibv_devices",): (0, ""),
            ("rdma", "link"): (0, ""),
            ("nicctl", "--version"): (0, "1.117.5-a-74"),
            # all sudo nicctl calls fail (rc=1)
            ("nicctl", "show", "version", "firmware", "--json"): (1, ""),
            ("nicctl", "show", "version", "host-software", "--json"): (1, ""),
            ("nicctl", "show", "card", "--detail"): (1, ""),
            ("nicctl", "show", "dcqcn", "--roce-device", "ionic_0", "--profile-id", "1"): (1, ""),
        }
        fake_which, fake_run = self._fake_tools(
            {"lspci", "ibv_devices", "rdma", "nicctl"}, outputs
        )
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        self._build_sysfs(tmp_path, monkeypatch)  # empty sysfs

        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        a = nics["ainic"]
        assert a["nicctl_version"] == "1.117.5-a-74"  # --version succeeded
        assert a["card"]["firmware"] is None
        assert a["card"]["uuid"] is None
        assert any(r.startswith("nics.ainic") for r in reasons)

    def test_ainic_rdma_naming_recovered_via_sysfs_driver(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # Regression for PR #208 review (AINIC host): RoCE devices named
        # rdma0..rdma7 (not ionic_*) must still be bound to ainic via their
        # kernel driver (ionic), and DCQCN must target the resolved device
        # (rdma0), not a hardcoded ionic_0.
        outputs = {
            ("lspci", "-d", "1dd8:1002"): (0, "c1:00.0 Ethernet controller: Pensando\n"),
            ("lspci", "-d", "14e4:1760"): (0, ""),
            ("lspci", "-d", "15b3:1021"): (0, ""),
            ("ibv_devices",): (0, _AINIC_IBV_DEVICES),
            ("rdma", "link"): (0, _AINIC_RDMA_LINK),
            ("nicctl", "--version"): (0, "nicctl version 1.117.5-a-74"),
            # DCQCN keyed on the RESOLVED device (rdma0). If the code still
            # hardcoded ionic_0 this key would miss and dcqcn would be None.
            ("nicctl", "show", "dcqcn", "--roce-device", "rdma0", "--profile-id", "1"):
                (0, _AINIC_DCQCN),
        }
        fake_which, fake_run = self._fake_tools(
            {"lspci", "ibv_devices", "rdma", "nicctl"}, outputs
        )
        monkeypatch.setattr(env_mod.shutil, "which", fake_which)
        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        # All 8 rdma devices + netdevs bind to the ionic driver.
        self._build_sysfs(
            tmp_path, monkeypatch,
            netdevs={f"tw-eth{i}": "ionic" for i in range(8)},
            ib_devs={f"rdma{i}": "ionic" for i in range(8)},
        )

        reasons: list[str] = []
        nics = env_mod._capture_nics(reasons)
        a = nics["ainic"]
        assert a["present"] is True
        # All 8 RoCE devices recovered despite the rdma<N> naming.
        assert a["rdma_devices"] == [f"rdma{i}" for i in range(8)]
        assert len(a["links"]) == 8
        assert all(ln["state"] == "ACTIVE" for ln in a["links"])
        assert a["links"][0] == {
            "device": "rdma0", "state": "ACTIVE", "netdev": "tw-eth0"
        }
        # DCQCN targeted the resolved device (rdma0) and parsed.
        assert a["dcqcn"]["token_bucket_size"] == 800000
        assert a["dcqcn"]["enabled"] is True
        # No "no AINIC RDMA device resolved" reason (devices were found).
        assert not any("no AINIC RDMA device" in r for r in reasons)

    def test_run_nic_cmd_sudo_exec_failure_names_sudo(self, monkeypatch):
        # When sudo=True and the exec itself raises (e.g. sudo missing),
        # the reason must name the actual runner (sudo), not the wrapped
        # tool, so partial_reasons are accurate.
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/nicctl")

        def boom_run(cmd, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "sudo")

        monkeypatch.setattr(env_mod.subprocess, "run", boom_run)
        reasons: list[str] = []
        out = env_mod._run_nic_cmd(
            ["nicctl", "--version"], reasons, "nics.ainic.x", sudo=True
        )
        assert out is None
        assert any("failed to invoke sudo" in r for r in reasons)
        assert not any("failed to invoke nicctl" in r for r in reasons)


# ---------------------------------------------------------------------------
# GPU architecture detection (rocm_agent_enumerator)
# ---------------------------------------------------------------------------


class TestGpuArch:
    def test_block_keys_stable(self, all_disabled):
        snapshot = collect_env()
        assert set(snapshot.gpu_arch.keys()) == {
            "agent_count",
            "gfx_targets",
            "agent_arch_counts",
        }

    def test_binary_missing_records_reason(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        # Also stub the /opt/rocm/bin fallback by patching Path.exists
        # to return False for the canonical path. Easiest: monkeypatch
        # ROCM_AGENT_ENUMERATOR_BIN to a name that won't exist anywhere.
        monkeypatch.setattr(
            env_mod, "ROCM_AGENT_ENUMERATOR_BIN", "definitely_not_a_real_binary_xyz"
        )
        reasons: list[str] = []
        block = env_mod._capture_gpu_arch(reasons)
        assert block == {
            "agent_count": None,
            "gfx_targets": None,
            "agent_arch_counts": None,
        }
        assert any("not on PATH" in r for r in reasons)

    def test_happy_path_parses_one_per_line(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setattr(
            env_mod.shutil, "which",
            lambda name: "/usr/bin/" + name,
        )

        def fake_run(cmd, **kwargs):
            assert cmd[0].endswith("rocm_agent_enumerator")
            assert kwargs["timeout"] == env_mod.GPU_ARCH_TIMEOUT_SEC
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout="gfx942\ngfx942\ngfx942\ngfx942\n", stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        block = env_mod._capture_gpu_arch(reasons)
        assert block["agent_count"] == 4
        assert block["gfx_targets"] == ["gfx942"]
        assert block["agent_arch_counts"] == {"gfx942": 4}
        assert reasons == []

    def test_timeout_uses_gpu_specific_budget_and_records_reason(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setattr(
            env_mod.shutil, "which",
            lambda name: "/usr/bin/" + name,
        )

        def fake_run(cmd, **kwargs):
            assert kwargs["timeout"] == env_mod.GPU_ARCH_TIMEOUT_SEC
            raise subprocess.TimeoutExpired(
                cmd=cmd,
                timeout=kwargs["timeout"],
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        block = env_mod._capture_gpu_arch(reasons)
        assert block["agent_count"] is None
        assert any("exceeded 30s timeout" in reason for reason in reasons)

    def test_filters_gfx000_placeholder(self, isolated_env, monkeypatch):
        """Some hosts include a gfx000 placeholder for the host CPU
        agent. It's not a GPU; drop it from the targets list.
        """
        monkeypatch.setattr(
            env_mod.shutil, "which",
            lambda name: "/usr/bin/" + name,
        )

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout="gfx000\ngfx942\ngfx942\n", stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        block = env_mod._capture_gpu_arch(reasons)
        assert block["agent_count"] == 2  # gfx000 dropped
        assert "gfx000" not in (block["agent_arch_counts"] or {})

    def test_mixed_arch_host_captures_distribution(
        self, isolated_env, monkeypatch
    ):
        """A host with mixed-arch GPUs (e.g. mi300x + rx7900) should
        show both in `gfx_targets` AND the per-arch count in
        `agent_arch_counts`. Catches the cross-environment confound
        where two trials look identical except one ran on a different
        second card.
        """
        monkeypatch.setattr(
            env_mod.shutil, "which",
            lambda name: "/usr/bin/" + name,
        )

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout="gfx1100\ngfx942\ngfx942\n", stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        block = env_mod._capture_gpu_arch(reasons)
        assert block["gfx_targets"] == ["gfx1100", "gfx942"]
        assert block["agent_arch_counts"] == {"gfx1100": 1, "gfx942": 2}

    def test_no_kfd_access_records_stderr_tail(
        self, isolated_env, monkeypatch
    ):
        """The most common failure: user not in render group, the
        binary exits non-zero with a stderr message about /dev/kfd.
        We surface that stderr so the operator knows what to fix.
        """
        monkeypatch.setattr(
            env_mod.shutil, "which",
            lambda name: "/usr/bin/" + name,
        )

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1,
                stdout="",
                stderr="cannot open /dev/kfd: Permission denied\n",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        block = env_mod._capture_gpu_arch(reasons)
        assert block["agent_count"] is None
        assert any("Permission denied" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# AOTriton (default ROCm Flash Attention backend)
# ---------------------------------------------------------------------------


class TestAotritonBlockShape:
    def test_block_keys_stable(self, all_disabled):
        """Schema-shape guard. The aotriton block always has these keys
        regardless of presence/absence of the bundled lib.
        """
        snapshot = collect_env()
        assert set(snapshot.aotriton.keys()) == {
            "bundled_present",
            "bundled_version",
            "bundled_lib_hash",
            "bundled_images_dir_present",
            "installed_prefix",
        }


class TestAotritonProbe:
    def test_torch_absent_returns_default_no_reason(
        self, isolated_env, monkeypatch
    ):
        """torch missing -> documented absence (already captured by
        pytorch_version), aotriton probe stays silent.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        assert block["bundled_present"] is False
        assert block["bundled_version"] is None
        assert reasons == []

    def test_cpu_only_torch_skips_silently(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """torch.version.hip is None -> CPU-only wheel, no AOTriton by
        design. Mirrors the bundled-CK probe's CPU-only handling.
        """
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        torch_dir.mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip=None, cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        assert block["bundled_present"] is False
        assert reasons == []

    def test_happy_path_parses_version_and_hashes(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        (torch_dir / "lib" / "libaotriton_v2.so.0.11.1").write_bytes(b"aot")
        (torch_dir / "lib" / "aotriton.images").mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip="7.2.5", cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        assert block["bundled_present"] is True
        assert block["bundled_version"] == "0.11.1"
        assert block["bundled_lib_hash"] is not None
        assert block["bundled_lib_hash"].startswith("sha256:")
        assert block["bundled_images_dir_present"] is True
        assert block["installed_prefix"] is None
        assert reasons == []

    def test_picks_highest_version_when_multiple_present(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        (torch_dir / "lib" / "libaotriton_v2.so.0.10.0").write_bytes(b"old")
        (torch_dir / "lib" / "libaotriton_v2.so.0.11.1").write_bytes(b"new")
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip="7.2.5", cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        # Numeric-tuple sort: 0.11.1 > 0.10.0 even though string sort
        # would say "0.11.1" < "0.10.0" (since '1' < '0' in second slot).
        assert block["bundled_version"] == "0.11.1"

    def test_version_and_hash_describe_the_same_file_under_minor_crossover(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Regression guard for the round-4 bug:

        bundled_version was version-tuple-sorted (correct), but
        bundled_lib_hash was string-sort-sorted (wrong). For any pair
        crossing a digit boundary, e.g. 0.9.0 vs 0.10.0:
          - tuple sort picks 0.10.0
          - string sort picks 0.9.0 (lexically '9' > '1')

        That left bundled_version="0.10.0" while
        bundled_lib_hash hashed the bytes of 0.9.0 -- the two fields
        described different files for the same record. This test pins
        the fix: both must point at 0.10.0.
        """
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        (torch_dir / "lib" / "libaotriton_v2.so.0.9.0").write_bytes(b"NINE-zero")
        (torch_dir / "lib" / "libaotriton_v2.so.0.10.0").write_bytes(b"TEN-zero")
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip="7.2.5", cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        assert block["bundled_version"] == "0.10.0"
        # The hash MUST be of the 0.10.0 bytes, NOT the 0.9.0 bytes
        # that string-sort would have chosen.
        expected_hash = "sha256:" + hashlib.sha256(b"TEN-zero").hexdigest()
        wrong_hash = "sha256:" + hashlib.sha256(b"NINE-zero").hexdigest()
        assert block["bundled_lib_hash"] == expected_hash, (
            f"bundled_lib_hash describes the wrong file -- the "
            f"version-tuple-sort vs string-sort regression has reappeared. "
            f"Expected hash of '0.10.0' bytes ({expected_hash!r}), got "
            f"{block['bundled_lib_hash']!r}. If this is the wrong-side "
            f"hash {wrong_hash!r}, _capture_aotriton fell back to "
            f"_hash_shared_library's glob+string-sort instead of using "
            f"_hash_file_path(best_path)."
        )

    def test_no_aotriton_in_lib_dir_records_reason(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """HIP torch but no libaotriton_v2.so* -> custom build with
        AOTriton disabled. Worth flagging.
        """
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        # No aotriton lib on purpose.
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip="7.2.5", cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        assert block["bundled_present"] is False
        assert any("no libaotriton_v2.so" in r for r in reasons)

    def test_installed_prefix_env_var_recorded(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """AOTRITON_INSTALLED_PREFIX is the operator's override pointing
        PyTorch at a system AOTriton install. Capturing it is critical
        for cross-env diffs (a host with the override set behaves
        differently from one without).
        """
        import builtins
        import types

        torch_dir = tmp_path / "torch"
        (torch_dir / "lib").mkdir(parents=True)
        (torch_dir / "lib" / "libaotriton_v2.so.0.11.1").write_bytes(b"x")
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            version=types.SimpleNamespace(hip="7.2.5", cuda=None),
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setenv("AOTRITON_INSTALLED_PREFIX", "/opt/aotriton-0.12")
        reasons: list[str] = []
        block = env_mod._capture_aotriton(reasons)
        assert block["installed_prefix"] == "/opt/aotriton-0.12"


# ---------------------------------------------------------------------------
# pytorch_build block (structured PyTorch identity + submodule SHAs)
# ---------------------------------------------------------------------------


class TestPytorchBuildBlockShape:
    """Schema-shape and disaster-defaults guards."""

    def test_block_keys_stable(self, all_disabled):
        snapshot = collect_env()
        assert set(snapshot.pytorch_build.keys()) == {
            "git_commit",
            "hip_version",
            "cuda_version",
            "debug",
            "install_kind",
            "source_path",
            "submodule_commits",
            "flags",
            "build_flags",
            "binary_introspection",
            "cmake_cache",
            "ninja_hipcc",
        }

    def test_build_flags_keys_stable(self, all_disabled):
        """The 17-key parsed build_flags subset is the schema contract.

        Bumping it is a deliberate change -- mirrors test_canonical_var_names_stable.
        Order intentionally not asserted (dict iteration order is the
        insertion order from PYTORCH_BUILD_FLAG_NAMES, but consumers
        should treat the dict as a set-of-keys mapping).
        """
        snapshot = collect_env()
        bf = snapshot.pytorch_build["build_flags"]
        assert set(bf.keys()) == set(env_mod.PYTORCH_BUILD_FLAG_NAMES)
        # all_disabled fakes torch import absence -> every flag is None.
        assert all(v is None for v in bf.values())

    def test_submodule_commits_keys_stable(self, all_disabled):
        snapshot = collect_env()
        subs = snapshot.pytorch_build["submodule_commits"]
        # The canonical submodule list IS schema; bumping it is a
        # deliberate change. Mirrors test_canonical_var_names_stable.
        assert set(subs.keys()) == {
            "_source",
            "composable_kernel",
            "aiter",
            "fbgemm",
        }

    def test_canonical_submodules_constant_is_stable(self):
        # If you add a third_party submodule to track, update both this
        # set and TestPytorchBuildBlockShape.test_submodule_commits_keys_stable.
        assert set(env_mod.CANONICAL_PYTORCH_SUBMODULES) == {
            "composable_kernel",
            "aiter",
            "fbgemm",
        }


class TestDetectPytorchInstallKind:
    def test_explicit_env_var_wins(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        src = tmp_path / "my_pytorch_src"
        (src / "third_party").mkdir(parents=True)
        monkeypatch.setenv("AORTA_PYTORCH_SRC", str(src))
        kind, path = env_mod._detect_pytorch_install_kind()
        assert kind == "source"
        assert path == src.resolve()

    def test_env_var_pointing_at_invalid_path_falls_through(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # AORTA_PYTORCH_SRC set, but the dir has no third_party/ -- we
        # honour the operator's intent only when the structure is valid.
        bogus = tmp_path / "not_a_pytorch_tree"
        bogus.mkdir()
        monkeypatch.setenv("AORTA_PYTORCH_SRC", str(bogus))
        # No torch in this environment -> falls to "unknown"; the point
        # is we don't return ("source", bogus).
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        kind, path = env_mod._detect_pytorch_install_kind()
        assert kind == "unknown"

    def test_torch_absent_returns_unknown(self, isolated_env, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        kind, path = env_mod._detect_pytorch_install_kind()
        assert kind == "unknown"
        assert path is None

    def test_walk_up_finds_source_tree(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """`import torch` from inside a source checkout: torch.__file__
        sits under a directory that has .git + third_party siblings.
        """
        import builtins
        import types

        # Layout: <tmp>/.git, <tmp>/third_party/, <tmp>/torch/__init__.py
        (tmp_path / ".git").mkdir()
        (tmp_path / "third_party").mkdir()
        torch_dir = tmp_path / "torch"
        torch_dir.mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            __version__="2.99.0",
        )

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        kind, path = env_mod._detect_pytorch_install_kind()
        assert kind == "source"
        assert path == tmp_path.resolve()

    def test_wheel_install_default(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """No env var, no .git, no third_party -- a stock wheel install."""
        import builtins
        import types

        site = tmp_path / "site-packages"
        site.mkdir()
        torch_dir = site / "torch"
        torch_dir.mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            __version__="2.99.0",
        )

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        kind, path = env_mod._detect_pytorch_install_kind()
        assert kind == "wheel"
        assert path is None


class TestGitRevParseHead:
    def test_happy_path_returns_full_sha(self, tmp_path: Path, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout="ff65f5bc672795c5e5033900ea0a0c4f8566c8cf\n",
                stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        sha = env_mod._git_rev_parse_head(tmp_path)
        assert sha == "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf"

    def test_non_hex_output_rejected(self, tmp_path: Path, monkeypatch):
        """Defensive: a misconfigured git aliasing rev-parse to something
        else (unlikely but possible) must not poison the snapshot.
        """
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="not-a-sha\n", stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        assert env_mod._git_rev_parse_head(tmp_path) is None

    def test_git_missing_returns_none(self, tmp_path: Path, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        assert env_mod._git_rev_parse_head(tmp_path) is None

    def test_nonzero_exit_returns_none(self, tmp_path: Path, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=128, stdout="", stderr="not a git repo",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        assert env_mod._git_rev_parse_head(tmp_path) is None


class TestCapturePytorchSubmodules:
    def test_source_tree_populates_via_git(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        third_party = tmp_path / "third_party"
        third_party.mkdir()
        for name in env_mod.CANONICAL_PYTORCH_SUBMODULES:
            (third_party / name).mkdir()

        sha_map = {
            "composable_kernel": "1" * 40,
            "aiter": "2" * 40,
            "fbgemm": "3" * 40,
        }

        def fake_run(cmd, **kwargs):
            sub_name = Path(cmd[2]).name
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=sha_map.get(sub_name, "") + "\n",
                stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        result = env_mod._capture_pytorch_submodules(
            "source", tmp_path, "abc1234", reasons
        )
        assert result["_source"] == "git"
        assert result["composable_kernel"] == "1" * 40
        assert result["aiter"] == "2" * 40
        assert result["fbgemm"] == "3" * 40
        assert reasons == []

    def test_wheel_install_emits_url_template(self, isolated_env):
        """No source tree -- partial reason must contain the GitHub URL
        template with the captured commit substituted in. Operators
        reading env.json get a copy-pasteable recovery URL.
        """
        reasons: list[str] = []
        result = env_mod._capture_pytorch_submodules(
            "wheel", None, "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf", reasons
        )
        assert result["_source"] is None
        for name in env_mod.CANONICAL_PYTORCH_SUBMODULES:
            assert result[name] is None
        assert len(reasons) == 1
        reason = reasons[0]
        assert "github.com/pytorch/pytorch/tree/" in reason
        assert "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf" in reason
        assert "AORTA_PYTORCH_SRC" in reason

    def test_wheel_install_unknown_commit_uses_placeholder(self, isolated_env):
        """If git_commit is null too, the URL template still appears with
        the literal `<git_commit>` placeholder so the operator at least
        sees the recovery shape.
        """
        reasons: list[str] = []
        env_mod._capture_pytorch_submodules("wheel", None, None, reasons)
        assert any("<git_commit>" in r for r in reasons)

    def test_unknown_install_kind_records_reason(self, isolated_env):
        reasons: list[str] = []
        env_mod._capture_pytorch_submodules("unknown", None, None, reasons)
        assert any("torch import failed" in r for r in reasons)

    def test_partial_submodule_set_records_missing(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Source tree exists but only some submodules are checked out.
        The missing ones land in a single-line reason, not N reasons.
        """
        third_party = tmp_path / "third_party"
        third_party.mkdir()
        # Only composable_kernel exists; aiter + fbgemm don't.
        (third_party / "composable_kernel").mkdir()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="a" * 40 + "\n", stderr="",
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        reasons: list[str] = []
        result = env_mod._capture_pytorch_submodules(
            "source", tmp_path, "abc1234", reasons
        )
        assert result["composable_kernel"] == "a" * 40
        assert result["aiter"] is None
        assert result["fbgemm"] is None
        assert result["_source"] == "git"
        assert len(reasons) == 1
        assert "aiter" in reasons[0]
        assert "fbgemm" in reasons[0]


class TestCapturePytorchBuildIntegration:
    """The full block, exercised against fake torch."""

    def test_torch_absent_returns_default_block(
        self, isolated_env, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_pytorch_build(reasons)
        assert block["git_commit"] is None
        assert block["install_kind"] == "unknown"
        # The block-level probe must NOT add a generic "torch import
        # raised" reason -- pytorch_version already records the absence
        # and double-counting would noise up partial_reasons. The single
        # reason that DOES fire is the submodule-commits one (a
        # consumer-facing affordance saying SHAs are unrecoverable),
        # which is a separate field-level signal.
        assert not any(
            r.startswith("pytorch_build: torch import raised") for r in reasons
        )
        sub_reasons = [
            r for r in reasons if r.startswith("pytorch_build.submodule_commits")
        ]
        assert len(sub_reasons) == 1
        assert "torch import failed" in sub_reasons[0]

    def test_real_torch_version_fields_captured(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        import builtins
        import types

        site = tmp_path / "site"
        site.mkdir()
        torch_dir = site / "torch"
        torch_dir.mkdir()
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        fake_torch = types.SimpleNamespace(
            __file__=str(torch_init),
            __version__="2.99.0",
            version=types.SimpleNamespace(
                git_version="ff65f5bc672795c5e5033900ea0a0c4f8566c8cf",
                hip="7.2.5",
                cuda=None,
                debug=False,
            ),
        )

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_pytorch_build(reasons)
        assert block["git_commit"] == "ff65f5bc672795c5e5033900ea0a0c4f8566c8cf"
        assert block["hip_version"] == "7.2.5"
        assert block["cuda_version"] is None
        assert block["debug"] is False
        assert block["install_kind"] == "wheel"
        # Wheel install -> single recovery-URL reason
        assert any(
            "github.com/pytorch/pytorch/tree/" in r for r in reasons
        )


class TestHipSymbolDumpCache:
    """Per-collect_env() cache that dedupes the nm|c++filt subprocess
    across the CK probe and the binary-introspection probe.
    """

    def test_first_get_invokes_dump_subsequent_reuse(self, monkeypatch):
        calls: list[str] = []

        def fake_dump(reasons, prefix, *, torch_mod=None):
            calls.append(prefix)
            return "ck::foo\nck::bar\n"

        monkeypatch.setattr(env_mod, "_dump_pytorch_hip_demangled_symbols", fake_dump)
        cache = env_mod._HipSymbolDumpCache()
        reasons: list[str] = []
        a = cache.get(reasons, "first")
        b = cache.get(reasons, "second")
        assert a == b == "ck::foo\nck::bar\n"
        assert calls == ["first"]

    def test_failed_dump_cached_no_duplicate_reasons(self, monkeypatch):
        def fake_dump(reasons, prefix, *, torch_mod=None):
            reasons.append(f"{prefix}: nm/c++filt not on PATH")
            return None

        monkeypatch.setattr(env_mod, "_dump_pytorch_hip_demangled_symbols", fake_dump)
        cache = env_mod._HipSymbolDumpCache()
        reasons: list[str] = []
        assert cache.get(reasons, "first") is None
        assert cache.get(reasons, "second") is None
        assert reasons == ["first: nm/c++filt not on PATH"]

    def test_cache_shared_across_probes_in_collect_env(
        self, all_disabled, monkeypatch
    ):
        calls: list[str] = []

        def fake_dump(reasons, prefix, *, torch_mod=None):
            calls.append(prefix)
            return None

        monkeypatch.setattr(env_mod, "_dump_pytorch_hip_demangled_symbols", fake_dump)
        collect_env()
        # CK probe and binary_introspection probe both use the cache;
        # only the first prefix actually invokes the dump.
        assert len(calls) <= 1


class TestCapturePytorchBinaryIntrospection:
    """Direct facts about the compiled PyTorch wheel -- no inference."""

    @staticmethod
    def _fake_torch(tmp_path: Path, *, with_aotriton: bool, cfg_text: str | None):
        import types
        torch_dir = tmp_path / "site" / "torch"
        lib_dir = torch_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        if with_aotriton:
            (lib_dir / "libaotriton_v2.so.0.11.2").write_text("")
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        config_obj = (
            types.SimpleNamespace(show=lambda: cfg_text)
            if cfg_text is not None
            else None
        )
        return types.SimpleNamespace(
            __file__=str(torch_init),
            __version__="2.99.0",
            __config__=config_obj,
            version=types.SimpleNamespace(
                git_version="abc1234", hip="7.2.5", cuda=None, debug=False,
            ),
        )

    def test_torch_lib_bundled_detects_versioned_soname(self, tmp_path):
        torch_mod = self._fake_torch(tmp_path, with_aotriton=True, cfg_text=None)
        block = env_mod._capture_pytorch_binary_introspection([], torch_mod=torch_mod)
        assert block["torch_lib_bundled"] == {"libaotriton_v2.so": True}

    def test_torch_lib_bundled_absent_renders_false(self, tmp_path):
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)
        block = env_mod._capture_pytorch_binary_introspection([], torch_mod=torch_mod)
        assert block["torch_lib_bundled"] == {"libaotriton_v2.so": False}

    def test_cxx_define_presence_parsed_from_config_show(self, tmp_path):
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)
        flags = {"cxx_flags_raw": "-DUSE_ROCM_CK_SDPA -O3"}
        block = env_mod._capture_pytorch_binary_introspection(
            [], torch_mod=torch_mod, flags=flags,
        )
        assert block["cxx_flags_use_defines"] == {
            "USE_ROCM_CK_SDPA": True,
            "USE_ROCM_CK_GEMM": False,
        }

    def test_cxx_define_regex_does_not_false_match_substring(self, tmp_path):
        # `USE_ROCM_CK_SDPA_FOO` must not match `USE_ROCM_CK_SDPA`.
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)
        flags = {"cxx_flags_raw": "-DUSE_ROCM_CK_SDPA_FOO -O3"}
        block = env_mod._capture_pytorch_binary_introspection(
            [], torch_mod=torch_mod, flags=flags,
        )
        assert block["cxx_flags_use_defines"]["USE_ROCM_CK_SDPA"] is False

    def test_cxx_define_in_cuda_flags_only_does_not_leak(self, tmp_path):
        """A `-DUSE_ROCM_CK_SDPA` token that lives in CUDA_FLAGS must NOT
        appear in cxx_flags_use_defines -- the field name is the
        contract.
        """
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)
        flags = {
            "cxx_flags_raw": "-O3 -fPIC",
            "cuda_flags_raw": "-DUSE_ROCM_CK_SDPA -arch=gfx942",
        }
        block = env_mod._capture_pytorch_binary_introspection(
            [], torch_mod=torch_mod, flags=flags,
        )
        assert block["cxx_flags_use_defines"]["USE_ROCM_CK_SDPA"] is False

    def test_cxx_flags_raw_none_yields_none_dict(self, tmp_path):
        """No CXX_FLAGS source -> the whole cxx_flags_use_defines dict
        stays None; we don't fabricate False entries.
        """
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)
        block = env_mod._capture_pytorch_binary_introspection(
            [], torch_mod=torch_mod, flags={"cxx_flags_raw": None},
        )
        assert block["cxx_flags_use_defines"] is None

    def test_torch_none_returns_full_default_shape(self):
        block = env_mod._capture_pytorch_binary_introspection([], torch_mod=None)
        assert block["torch_lib_bundled"] is None
        assert block["cxx_flags_use_defines"] is None
        assert all(
            v is None for v in block["libtorch_hip_symbol_counts"].values()
        )

    def test_torch_lib_scan_oserror_yields_none_with_partial_reason(
        self, tmp_path, monkeypatch
    ):
        """A failed torch/lib scan (missing dir, permission denied, ...)
        must NOT report False per lib -- False is the definitive
        "scanned, lib absent" signal. Whole dict stays None and a
        partial reason is recorded so the operator knows the probe
        failed rather than the libs being missing.
        """
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)

        def boom(self):
            raise PermissionError("denied")

        monkeypatch.setattr(env_mod.Path, "iterdir", boom)
        reasons: list[str] = []
        block = env_mod._capture_pytorch_binary_introspection(
            reasons, torch_mod=torch_mod
        )
        assert block["torch_lib_bundled"] is None
        assert any(
            r.startswith(
                "pytorch_build.binary_introspection.torch_lib_bundled:"
            )
            and "PermissionError" in r
            for r in reasons
        )

    def test_dump_uses_provided_torch_mod_not_ambient(
        self, tmp_path, monkeypatch
    ):
        """Standalone-call path: when torch_mod is passed but no cache,
        the freshly-created cache must use the passed torch_mod (not
        re-import ambient torch). Otherwise `torch_lib_bundled` (uses
        passed) and `libtorch_hip_symbol_counts` (would use ambient)
        would describe different torch installations.
        """
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)
        # Set hip so the dump helper proceeds past the CPU-only guard,
        # and create the lib so the early lib_path.exists() check passes.
        torch_mod.version = type(torch_mod.version)(
            git_version="abc1234", hip="7.2.5", cuda=None, debug=False,
        )
        lib_dir = Path(torch_mod.__file__).parent / "lib"
        (lib_dir / "libtorch_hip.so").write_text("")

        # Trip if _safe_import_torch is called for the dump's prefix --
        # that would mean the helper re-imported ambient torch instead
        # of using the passed module.
        called: list[str] = []
        real_safe = env_mod._safe_import_torch

        def trip(reasons, prefix):
            if prefix == "pytorch_build.binary_introspection":
                called.append(prefix)
            return real_safe(reasons, prefix)

        monkeypatch.setattr(env_mod, "_safe_import_torch", trip)
        # Stub the subprocess work (we're checking the import path,
        # not the nm/c++filt dump itself).
        monkeypatch.setattr(env_mod.shutil, "which", lambda _name: None)

        env_mod._capture_pytorch_binary_introspection([], torch_mod=torch_mod)
        assert called == [], (
            "binary_introspection re-imported ambient torch despite "
            "being given an explicit torch_mod"
        )

    def test_torch_none_skips_symbol_cache_lookup(self, monkeypatch):
        """Caller signal `torch_mod=None` -> skip the cache entirely;
        otherwise on a real-torch host the cache would still dump
        symbols and contradict the default-shape contract.
        """
        called: list[str] = []

        class TripwireCache:
            def get(self, reasons, prefix):
                called.append(prefix)
                return "pytorch_flash::mha_fwd\n"

        block = env_mod._capture_pytorch_binary_introspection(
            [], torch_mod=None, hip_symbol_cache=TripwireCache(),
        )
        assert called == []
        assert all(
            v is None for v in block["libtorch_hip_symbol_counts"].values()
        )

    def test_symbol_counts_use_provided_cache(self, tmp_path, monkeypatch):
        torch_mod = self._fake_torch(tmp_path, with_aotriton=False, cfg_text=None)

        class FixedCache:
            def get(self, reasons, prefix, *, torch_mod=None):
                return (
                    "void pytorch_flash::mha_fwd()\n"
                    "void pytorch_flash::mha_bwd()\n"
                    "void aotriton::TensorView()\n"
                    "void unrelated::symbol()\n"
                )

        block = env_mod._capture_pytorch_binary_introspection(
            [], torch_mod=torch_mod, hip_symbol_cache=FixedCache()
        )
        counts = block["libtorch_hip_symbol_counts"]
        assert counts["pytorch_flash::"] == 2
        assert counts["aotriton::"] == 1
        assert counts["ck_tile::FmhaFwd"] == 0


class TestSummaryPytorchBuildFlagsLineUnavailable:
    """Regression guards: distinguish "unavailable" from "all off" in the
    `torch flags:` brief at two granularities -- whole-block and
    per-cell.
    """

    @staticmethod
    def _snap_with_flags(flags_block):
        return _example_snapshot(
            pytorch_build={
                "git_commit": None, "hip_version": None, "cuda_version": None,
                "debug": None, "install_kind": "wheel", "source_path": None,
                "submodule_commits": {"_source": None},
                "flags": flags_block,
                "binary_introspection": {
                    "libtorch_hip_symbol_counts": {
                        m: None for m in env_mod._LIBTORCH_HIP_SYMBOL_MARKERS
                    },
                    "torch_lib_bundled": None,
                    "cxx_flags_use_defines": None,
                },
                "build_flags": {
                    name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES
                },
            },
        )

    def test_archs_present_but_settings_defines_none_renders_question_marks(self):
        snap = self._snap_with_flags({
            "build_settings": None, "cxx_defines": None,
            "cxx_flags_raw": None, "cuda_flags_raw": None,
            "gpu_arch_list": ["gfx942"],
        })
        line = snap._summary_pytorch_build_flags_line()
        assert "gpu_archs=[gfx942]" in line
        assert "USE_ROCM=?" in line
        assert "USE_ROCM=no" not in line
        assert "USE_FLASH_ATTENTION=?" in line

    def test_settings_populated_but_cxx_defines_none_renders_cxx_only_flags_unknown(self):
        """When CXX_FLAGS line is missing from __config__.show(),
        cxx_defines is None. CXX-only flags (e.g. USE_ROCM_CK_SDPA)
        must render `=?` not `=no` -- absence-of-source != absence-of-flag.
        """
        snap = self._snap_with_flags({
            "build_settings": {"USE_ROCM": "ON", "USE_CUDA": "OFF"},
            "cxx_defines": None,
            "cxx_flags_raw": None, "cuda_flags_raw": None,
            "gpu_arch_list": None,
        })
        line = snap._summary_pytorch_build_flags_line()
        assert "USE_ROCM=ON" in line
        assert "USE_CUDA=OFF" in line
        # CXX-only flags couldn't be read -> unknown, NOT off.
        assert "USE_ROCM_CK_SDPA=?" in line
        assert "USE_FLASH_ATTENTION=?" in line
        assert "USE_ROCM_CK_SDPA=no" not in line

    def test_empty_cxx_defines_dict_renders_cxx_only_flags_no(self):
        """An empty dict (we read CXX_FLAGS, no -D defines present) is
        a definitive "feature off" signal, distinct from None.
        """
        snap = self._snap_with_flags({
            "build_settings": {"USE_ROCM": "ON"},
            "cxx_defines": {},
            "cxx_flags_raw": "-O3 -fPIC",
            "cuda_flags_raw": None,
            "gpu_arch_list": None,
        })
        line = snap._summary_pytorch_build_flags_line()
        assert "USE_ROCM_CK_SDPA=no" in line
        assert "USE_FLASH_ATTENTION=no" in line

    def test_gpu_arch_list_empty_renders_none_not_question_mark(self):
        """CPU-only wheel: torch.cuda.get_arch_list() returns []. That's
        a successful, definitive result -- distinct from None
        (probe failed). Render `(none)` not `?`.
        """
        snap = self._snap_with_flags({
            "build_settings": {"USE_ROCM": "ON"},
            "cxx_defines": {},
            "cxx_flags_raw": None, "cuda_flags_raw": None,
            "gpu_arch_list": [],
        })
        line = snap._summary_pytorch_build_flags_line()
        assert "gpu_archs=[(none)]" in line
        assert "gpu_archs=[?]" not in line

    def test_gpu_arch_list_none_renders_question_mark(self):
        snap = self._snap_with_flags({
            "build_settings": {"USE_ROCM": "ON"},
            "cxx_defines": {},
            "cxx_flags_raw": None, "cuda_flags_raw": None,
            "gpu_arch_list": None,
        })
        line = snap._summary_pytorch_build_flags_line()
        assert "gpu_archs=[?]" in line


class TestSummaryStableBuildFlagsLineAotritonCombined:
    """Issue: brief AOTRITON cell must honor DISABLE_AOTRITON, otherwise
    a build that reports only `-DDISABLE_AOTRITON` (no USE_AOTRITON
    setting) renders `AOTRITON=?` despite a definitive disable signal.
    """

    @staticmethod
    def _snap(use_aotriton, disable_aotriton):
        bf = {name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES}
        bf["USE_AOTRITON"] = use_aotriton
        bf["DISABLE_AOTRITON"] = disable_aotriton
        # Anchor: keep at least one other flag populated so the
        # "all-None -> early unavailable return" guard doesn't fire
        # when both AOTRITON inputs are None.
        bf["USE_ROCM"] = True
        base = _example_snapshot()
        return _example_snapshot(
            pytorch_build={**base.pytorch_build, "build_flags": bf}
        )

    def test_disable_only_true_renders_off(self):
        line = self._snap(None, True)._summary_stable_build_flags_line()
        assert "AOTRITON=off" in line

    def test_use_only_true_renders_on(self):
        line = self._snap(True, None)._summary_stable_build_flags_line()
        assert "AOTRITON=on" in line

    def test_disable_false_renders_on(self):
        line = self._snap(None, False)._summary_stable_build_flags_line()
        assert "AOTRITON=on" in line

    def test_disable_wins_over_use_on_conflict(self):
        line = self._snap(True, True)._summary_stable_build_flags_line()
        assert "AOTRITON=off" in line

    def test_both_none_renders_question_mark(self):
        line = self._snap(None, None)._summary_stable_build_flags_line()
        assert "AOTRITON=?" in line


class TestProjectPytorchBuildFlags:
    """Issue #170: stable parsed subset of compile-time PyTorch flags."""

    def test_boolean_on_off_coerced(self):
        flags = {
            "build_settings": {"USE_ROCM": "ON", "USE_CUDA": "OFF"},
            "cxx_defines": None,
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_ROCM"] is True
        assert out["USE_CUDA"] is False

    def test_boolean_true_false_one_zero_coerced(self):
        flags = {
            "build_settings": {
                "USE_NCCL": "TRUE",
                "USE_MKL": "FALSE",
                "USE_OPENMP": "1",
                "USE_KINETO": "0",
            },
            "cxx_defines": None,
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_NCCL"] is True
        assert out["USE_MKL"] is False
        assert out["USE_OPENMP"] is True
        assert out["USE_KINETO"] is False

    def test_non_boolean_value_kept_as_string(self):
        """BUILD_TYPE=Release is not boolean -- preserve the original casing."""
        flags = {
            "build_settings": {"BUILD_TYPE": "Release"},
            "cxx_defines": None,
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["BUILD_TYPE"] == "Release"

    def test_missing_keys_present_as_none(self):
        """Every key in PYTORCH_BUILD_FLAG_NAMES must be in the output;
        missing ones are None (distinguishable from False).
        """
        out = env_mod._project_pytorch_build_flags(
            {"build_settings": {"USE_ROCM": "ON"}, "cxx_defines": None}
        )
        assert set(out.keys()) == set(env_mod.PYTORCH_BUILD_FLAG_NAMES)
        assert out["DISABLE_AOTRITON"] is None
        assert out["USE_FLASH_ATTENTION"] is None

    def test_cxx_define_without_value_is_true(self):
        """Bare ``-DUSE_FLASH_ATTENTION`` (no =value) means "on" by cmake convention."""
        flags = {
            "build_settings": None,
            "cxx_defines": {"USE_FLASH_ATTENTION": None, "USE_ROCM_CK_SDPA": None},
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_FLASH_ATTENTION"] is True
        assert out["USE_ROCM_CK_SDPA"] is True

    def test_cxx_define_with_value_coerced(self):
        flags = {
            "build_settings": None,
            "cxx_defines": {"USE_AOTRITON": "ON"},
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_AOTRITON"] is True

    def test_build_settings_wins_over_cxx_defines(self):
        """Cmake-canonical settings beat per-target define injection."""
        flags = {
            "build_settings": {"USE_ROCM": "ON"},
            "cxx_defines": {"USE_ROCM": "OFF"},
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_ROCM"] is True

    def test_caffe2_use_miopen_alias_maps_to_use_miopen(self):
        """Issue #170: CAFFE2_USE_MIOPEN is an alias for USE_MIOPEN."""
        flags = {
            "build_settings": {"CAFFE2_USE_MIOPEN": "ON"},
            "cxx_defines": None,
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_MIOPEN"] is True

    def test_canonical_use_miopen_wins_over_caffe2_alias(self):
        """When both spellings appear, the canonical name takes precedence
        (alias tuple is ordered USE_MIOPEN first).
        """
        flags = {
            "build_settings": {"USE_MIOPEN": "ON", "CAFFE2_USE_MIOPEN": "OFF"},
            "cxx_defines": None,
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_MIOPEN"] is True

    def test_absent_flag_stays_none_even_when_both_sources_parsed(self):
        """Issue #170 mock: keys not present in __config__.show() are
        null, not False (DISABLE_AOTRITON: null on a build with
        USE_AOTRITON: true). The brief line in `pytorch_build.flags`
        carries the cmake-convention "no" rendering for operators who
        want it; `pytorch_build.build_flags` preserves the
        "set vs unset" distinction.
        """
        flags = {
            "build_settings": {"USE_ROCM": "ON"},
            "cxx_defines": {},  # parsed, empty
        }
        out = env_mod._project_pytorch_build_flags(flags)
        assert out["USE_ROCM_CK_SDPA"] is None
        assert out["DISABLE_AOTRITON"] is None
        assert out["BUILD_TYPE"] is None

    def test_settings_alias_wins_even_when_canonical_in_defines(self):
        """Documented precedence: every alias in build_settings beats
        every alias in cxx_defines. A `-DUSE_MIOPEN` in defines must
        not override `CAFFE2_USE_MIOPEN=ON` in settings just because
        USE_MIOPEN comes earlier in the alias tuple.
        """
        flags = {
            "build_settings": {"CAFFE2_USE_MIOPEN": "OFF"},
            "cxx_defines": {"USE_MIOPEN": None},  # bare -DUSE_MIOPEN
        }
        out = env_mod._project_pytorch_build_flags(flags)
        # Settings says OFF -> False wins, not the True from -D define.
        assert out["USE_MIOPEN"] is False

    def test_none_flags_block_yields_all_none(self):
        """Torch import failed upstream -> patch returns None block;
        projection still produces the full schema, all None.
        """
        out = env_mod._project_pytorch_build_flags(None)
        assert set(out.keys()) == set(env_mod.PYTORCH_BUILD_FLAG_NAMES)
        assert all(v is None for v in out.values())


class TestCapturePytorchBuildFlagsRawSchema:
    """Direct coverage for `_capture_pytorch_build_flags()` raw output.

    The other tests cover `build_flags` (the projected stable subset)
    and the brief lines, but the raw structured block (cxx_defines,
    cxx_flags_raw, cuda_flags_raw, gpu_arch_list) feeds env.json and
    callers reading the raw schema would silently regress without
    direct coverage.
    """

    def _patch_torch(self, monkeypatch, fake_torch):
        import builtins
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (
                fake_torch if name == "torch" else real_import(name, *a, **kw)
            ),
        )

    def test_raw_fields_populated_from_full_config_show(self, monkeypatch):
        import types
        cfg = (
            "Build settings: BUILD_TYPE=Release, USE_ROCM=ON, "
            "CXX_FLAGS=-DUSE_ROCM_CK_SDPA -DFLASH_NAMESPACE=pytorch_flash -O3, "
            "CUDA_FLAGS=-arch=gfx942 -DCUDA_ONLY"
        )
        fake_torch = types.SimpleNamespace(
            __config__=types.SimpleNamespace(show=lambda: cfg),
            cuda=types.SimpleNamespace(get_arch_list=lambda: ["gfx942", "gfx950"]),
        )
        self._patch_torch(monkeypatch, fake_torch)
        out = env_mod._capture_pytorch_build_flags([])
        assert out["build_settings"]["USE_ROCM"] == "ON"
        assert out["build_settings"]["BUILD_TYPE"] == "Release"
        assert out["cxx_defines"] == {
            "FLASH_NAMESPACE": "pytorch_flash",
            "USE_ROCM_CK_SDPA": None,
        }
        assert out["cxx_flags_raw"].startswith("-DUSE_ROCM_CK_SDPA")
        assert out["cuda_flags_raw"].startswith("-arch=gfx942")
        assert out["gpu_arch_list"] == ["gfx942", "gfx950"]

    def test_arch_list_captured_when_config_show_unavailable(self, monkeypatch):
        """gpu_arch_list source is independent of __config__.show()."""
        import types
        fake_torch = types.SimpleNamespace(
            __config__=None,
            cuda=types.SimpleNamespace(get_arch_list=lambda: ["gfx942"]),
        )
        self._patch_torch(monkeypatch, fake_torch)
        reasons: list[str] = []
        out = env_mod._capture_pytorch_build_flags(reasons)
        assert out["gpu_arch_list"] == ["gfx942"]
        assert out["build_settings"] is None
        assert out["cxx_defines"] is None
        # __config__.show unavailable adds a partial reason
        assert any(r.startswith("pytorch_build.flags") for r in reasons)


class TestCapturePytorchBuildFlagsFromConfigShow:
    """End-to-end: torch.__config__.show() text -> build_flags dict."""

    @staticmethod
    def _fake_torch(tmp_path: Path, config_show_text: str):
        import types
        torch_dir = tmp_path / "site" / "torch"
        torch_dir.mkdir(parents=True, exist_ok=True)
        torch_init = torch_dir / "__init__.py"
        torch_init.write_text("")
        return types.SimpleNamespace(
            __file__=str(torch_init),
            __version__="2.99.0",
            version=types.SimpleNamespace(
                git_version="abc1234", hip=None, cuda=None, debug=False,
            ),
            __config__=types.SimpleNamespace(show=lambda: config_show_text),
            cuda=types.SimpleNamespace(get_arch_list=lambda: ["gfx942"]),
        )

    def _patch_torch(self, monkeypatch, fake_torch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_ck_sdpa_build_yields_true_for_attention_flags(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        cfg = (
            "PyTorch built with:\n"
            "  - GCC 11.4\n"
            "Build settings: BUILD_TYPE=Release, USE_ROCM=ON, USE_CUDA=OFF, "
            "USE_NCCL=ON, USE_MKLDNN=ON, USE_FLASH_ATTENTION=ON, "
            "USE_MEM_EFF_ATTENTION=ON, USE_FBGEMM=ON, USE_FBGEMM_GENAI=OFF, "
            "USE_AOTRITON=ON, "
            "CXX_FLAGS=-DUSE_ROCM_CK_SDPA -DUSE_FLASH_ATTENTION -O3"
        )
        self._patch_torch(monkeypatch, self._fake_torch(tmp_path, cfg))
        block = env_mod._capture_pytorch_build([])
        bf = block["build_flags"]
        assert bf["USE_ROCM_CK_SDPA"] is True
        assert bf["USE_FLASH_ATTENTION"] is True
        assert bf["USE_AOTRITON"] is True
        assert bf["USE_MEM_EFF_ATTENTION"] is True
        assert bf["USE_CUDA"] is False
        assert bf["BUILD_TYPE"] == "Release"

    def test_absent_keys_render_as_none_not_omitted(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        """Per issue acceptance: DISABLE_AOTRITON on a stock upstream
        build is not in __config__.show() -- must render as null, not
        be absent from the dict.
        """
        cfg = "Build settings: USE_ROCM=ON, BUILD_TYPE=Release"
        self._patch_torch(monkeypatch, self._fake_torch(tmp_path, cfg))
        block = env_mod._capture_pytorch_build([])
        bf = block["build_flags"]
        assert "DISABLE_AOTRITON" in bf
        assert bf["DISABLE_AOTRITON"] is None
        assert "USE_FLASH_ATTENTION" in bf
        assert bf["USE_FLASH_ATTENTION"] is None

    def test_torch_import_fails_yields_all_none_no_extra_reason(
        self, isolated_env, monkeypatch
    ):
        """Torch import failure: pytorch_version probe records it
        elsewhere; build_flags must NOT add a duplicate reason and the
        full schema must still appear (all None).
        """
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        block = env_mod._capture_pytorch_build(reasons)
        bf = block["build_flags"]
        assert set(bf.keys()) == set(env_mod.PYTORCH_BUILD_FLAG_NAMES)
        assert all(v is None for v in bf.values())
        assert not any(r.startswith("pytorch_build.build_flags") for r in reasons)


class TestSummaryStableBuildFlagsLine:
    """Issue #170: brief one-liner format."""

    def _snap(self, build_flags):
        base = _example_snapshot()
        return _example_snapshot(
            pytorch_build={**base.pytorch_build, "build_flags": build_flags}
        )

    def test_all_on_renders_compact_form(self):
        bf = {name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES}
        bf.update({
            "USE_FLASH_ATTENTION": True,
            "USE_ROCM_CK_SDPA": True,
            "USE_AOTRITON": True,
            "USE_MEM_EFF_ATTENTION": True,
        })
        snap = self._snap(bf)
        line = snap._summary_stable_build_flags_line()
        assert line == "FLASH_ATTN=on CK_SDPA=on AOTRITON=on MEM_EFF=on"

    def test_off_and_unknown_render_distinctly(self):
        bf = {name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES}
        bf.update({
            "USE_FLASH_ATTENTION": False,
            "USE_ROCM_CK_SDPA": True,
            # USE_AOTRITON intentionally absent (None)
            "USE_MEM_EFF_ATTENTION": False,
        })
        snap = self._snap(bf)
        line = snap._summary_stable_build_flags_line()
        assert line == "FLASH_ATTN=off CK_SDPA=on AOTRITON=? MEM_EFF=off"

    def test_all_none_renders_unavailable(self):
        bf = {name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES}
        snap = self._snap(bf)
        line = snap._summary_stable_build_flags_line()
        assert "unavailable" in line

    def test_summary_includes_flags_line(self):
        """Brief output must include the issue's `flags:` one-liner."""
        bf = {name: None for name in env_mod.PYTORCH_BUILD_FLAG_NAMES}
        bf.update({"USE_FLASH_ATTENTION": True, "USE_ROCM_CK_SDPA": True})
        snap = self._snap(bf)
        body = snap.summary()
        flags_line = next(
            (ln for ln in body.splitlines() if ln.lstrip().startswith("flags:")),
            None,
        )
        assert flags_line is not None
        assert "FLASH_ATTN=on" in flags_line
        assert "CK_SDPA=on" in flags_line


class TestCapturePytorchCmakeCache:
    """Issue #176: parsed CMakeCache.txt for source/editable installs."""

    @staticmethod
    def _make_cache(tmp_path: Path, body: str) -> Path:
        build = tmp_path / "build"
        build.mkdir()
        cache = build / "CMakeCache.txt"
        cache.write_text(body, encoding="utf-8")
        return cache

    def test_wheel_install_returns_null_no_partial(self, tmp_path):
        reasons: list[str] = []
        block = env_mod._capture_pytorch_cmake_cache("wheel", tmp_path, reasons)
        assert block == {"_source_file": None, "entries": None}
        assert reasons == []

    def test_no_build_dir_returns_null_no_partial(self, tmp_path):
        reasons: list[str] = []
        block = env_mod._capture_pytorch_cmake_cache("source", tmp_path, reasons)
        assert block == {"_source_file": None, "entries": None}
        assert reasons == []

    def test_parses_filtered_entries_sorted(self, tmp_path):
        cache_body = (
            "// header comment\n"
            "# unrelated comment\n"
            "USE_FLASH_ATTENTION:BOOL=ON\n"
            "USE_ROCM_CK_SDPA:BOOL=ON\n"
            "BUILD_TYPE:STRING=Release\n"
            "FLASH_NAMESPACE:STRING=pytorch_flash\n"
            "BORING_VAR_NOT_ALLOWLISTED:STRING=keep-out\n"
            "USE_NUMA:BOOL=OFF\n"
        )
        self._make_cache(tmp_path, cache_body)
        block = env_mod._capture_pytorch_cmake_cache("source", tmp_path, [])
        assert block["entries"] == {
            "BUILD_TYPE": {"type": "STRING", "value": "Release"},
            "FLASH_NAMESPACE": {"type": "STRING", "value": "pytorch_flash"},
            "USE_FLASH_ATTENTION": {"type": "BOOL", "value": "ON"},
            "USE_NUMA": {"type": "BOOL", "value": "OFF"},
            "USE_ROCM_CK_SDPA": {"type": "BOOL", "value": "ON"},
        }
        assert "BORING_VAR_NOT_ALLOWLISTED" not in block["entries"]
        assert block["_source_file"].endswith("CMakeCache.txt")

    def test_unreadable_cache_records_partial_reason(self, tmp_path, monkeypatch):
        self._make_cache(tmp_path, "USE_ROCM:BOOL=ON\n")

        def boom(self, *args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(env_mod.Path, "read_text", boom)
        reasons: list[str] = []
        block = env_mod._capture_pytorch_cmake_cache("source", tmp_path, reasons)
        assert block["entries"] is None
        assert any(
            r.startswith("pytorch_build.cmake_cache: read failed")
            and "PermissionError" in r
            for r in reasons
        )


class TestCapturePytorchNinjaHipcc:
    """Issue #176: streamed build.ninja per-target HIPCC introspection."""

    @staticmethod
    def _make_ninja(tmp_path: Path, body: str) -> Path:
        build = tmp_path / "build"
        build.mkdir()
        ninja = build / "build.ninja"
        ninja.write_text(body, encoding="utf-8")
        return ninja

    def test_wheel_install_returns_null(self, tmp_path):
        reasons: list[str] = []
        block = env_mod._capture_pytorch_ninja_hipcc("wheel", tmp_path, reasons)
        # 1.4 additive keys (_parser, _legacy_scripts_scanned) are all
        # None on the wheel path -- no parse attempted.
        assert block == {
            "_source_file": None,
            "_parser": None,
            "_legacy_scripts_scanned": None,
            "targets": None,
        }
        assert reasons == []

    def test_targets_of_interest_captured(self, tmp_path):
        # Two build statements: torch_hip (target of interest) + an
        # unrelated target that must be filtered out.
        body = (
            "build foo.o: HIP_COMPILER__torch_hip_unscanned src/foo.hip\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DUSE_ROCM_CK_SDPA "
            "-DCK_TILE_FMHA_FWD_FAST_EXP2 -DFLASH_NAMESPACE=pytorch_flash\n"
            "  FLAGS = -fgpu-flush-denormals-to-zero --offload-arch=gfx942 "
            "--offload-arch=gfx950 -O3\n"
            "\n"
            "build bar.o: HIP_COMPILER__unrelated_target src/bar.hip\n"
            "  DEFINES = -Dunrelated_target_EXPORTS -DSHOULD_BE_IGNORED\n"
            "  FLAGS = -O2\n"
        )
        self._make_ninja(tmp_path, body)
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert set(block["targets"]) == {"torch_hip"}
        t = block["targets"]["torch_hip"]
        assert t["defines"]["USE_ROCM_CK_SDPA"] is None
        assert t["defines"]["FLASH_NAMESPACE"] == "pytorch_flash"
        assert t["use_defines_present"]["USE_ROCM_CK_SDPA"] is True
        assert t["use_defines_present"]["DISABLE_AOTRITON"] is False
        assert t["codegen_flags_present"]["-fgpu-flush-denormals-to-zero"] is True
        assert t["codegen_flags_present"]["-ffast-math"] is False
        assert t["offload_archs"] == ["gfx942", "gfx950"]

    def test_scanned_no_matches_returns_source_file_and_empty_targets(
        self, tmp_path
    ):
        """Distinguishable from `targets: None` (wheel / no file): file
        existed, parser ran, just nothing matched _NINJA_HIPCC_TARGETS_OF_INTEREST.
        """
        body = (
            "build foo.o: HIP_COMPILER__unknown src/foo.hip\n"
            "  DEFINES = -Dunknown_target_EXPORTS\n"
            "  FLAGS = -O3\n"
        )
        self._make_ninja(tmp_path, body)
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert block["targets"] == {}
        assert block["_source_file"] is not None
        assert block["_source_file"].endswith("build.ninja")

    def test_dollar_continuation_folded_for_target_marker(self, tmp_path):
        """Ninja `$\\n` line continuation: the `-D<target>_EXPORTS`
        token can land on a continuation line. Without folding the
        whole DEFINES block would be misclassified.
        """
        body = (
            "build x.o: HIP_COMPILER__torch_hip_unscanned\n"
            "  DEFINES = -DA -DB $\n"
            "    -Dtorch_hip_EXPORTS -DUSE_ROCM_CK_SDPA $\n"
            "    -DC\n"
            "  FLAGS = -O3 --offload-arch=gfx942\n"
        )
        self._make_ninja(tmp_path, body)
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert "torch_hip" in block["targets"]
        defines = block["targets"]["torch_hip"]["defines"]
        # Tokens from every physical line must be captured.
        assert "A" in defines
        assert "USE_ROCM_CK_SDPA" in defines
        assert "C" in defines

    def test_dollar_continuation_in_flags_captures_offload_arch(
        self, tmp_path
    ):
        """The same continuation handling must apply to FLAGS so an
        --offload-arch=... token on a continuation line is captured.
        """
        body = (
            "build x.o: HIP_COMPILER__torch_hip_unscanned\n"
            "  DEFINES = -Dtorch_hip_EXPORTS\n"
            "  FLAGS = -O3 $\n"
            "    --offload-arch=gfx942 $\n"
            "    --offload-arch=gfx950\n"
        )
        self._make_ninja(tmp_path, body)
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert block["targets"]["torch_hip"]["offload_archs"] == [
            "gfx942", "gfx950",
        ]

    def test_cxx_rule_with_same_target_exports_does_not_pollute_hip_data(
        self, tmp_path
    ):
        """cmake propagates target-level defines to all sources, so a
        CXX rule for .cpp files in the torch_hip target ALSO carries
        `-Dtorch_hip_EXPORTS`. Without per-rule filtering the parser
        would merge that CXX rule's data into ninja_hipcc.targets[
        torch_hip] -- polluting the defines and producing empty
        offload_archs (CXX rules don't carry --offload-arch).
        """
        body = (
            "build foo.cpp.o: CXX_COMPILER__torch_hip_unscanned src/foo.cpp\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DCXX_ONLY_DEFINE\n"
            "  FLAGS = -O3 -fPIC\n"
            "\n"
            "build bar.hip.o: HIP_COMPILER__torch_hip_unscanned src/bar.hip\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DUSE_ROCM_CK_SDPA\n"
            "  FLAGS = -O3 --offload-arch=gfx942\n"
        )
        self._make_ninja(tmp_path, body)
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        defines = block["targets"]["torch_hip"]["defines"]
        # HIP-rule defines made it through.
        assert "USE_ROCM_CK_SDPA" in defines
        # CXX-only defines must NOT appear in the HIP target's defines.
        assert "CXX_ONLY_DEFINE" not in defines
        # offload_archs from the HIP rule is preserved.
        assert block["targets"]["torch_hip"]["offload_archs"] == ["gfx942"]

    def test_conflicting_define_values_resolve_deterministically(self, tmp_path):
        """Two HIP build statements in the same target set the same
        macro to different values. The merge must produce a stable
        result across runs (set iteration is hash-order; PYTHONHASHSEED
        randomization would otherwise flip which value wins). Sorted
        block iteration -> "lexicographically-largest block wins".
        """
        body = (
            "build a.hip.o: HIP_COMPILER__torch_hip_unscanned src/a.hip\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DCK_TILE_FLAVOR=fast\n"
            "  FLAGS = -O3\n"
            "\n"
            "build b.hip.o: HIP_COMPILER__torch_hip_unscanned src/b.hip\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DCK_TILE_FLAVOR=safe\n"
            "  FLAGS = -O3\n"
        )
        self._make_ninja(tmp_path, body)
        # Run twice; both runs must agree.
        out_a = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        out_b = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert (
            out_a["targets"]["torch_hip"]["defines"]["CK_TILE_FLAVOR"]
            == out_b["targets"]["torch_hip"]["defines"]["CK_TILE_FLAVOR"]
        )
        # Sorted-block-wins: lexicographic order of the two whole
        # DEFINES strings -- "...=fast" sorts before "...=safe", so
        # the safe-flavor block wins on the merge.
        assert (
            out_a["targets"]["torch_hip"]["defines"]["CK_TILE_FLAVOR"] == "safe"
        )

    def test_streaming_does_not_slurp_giant_files(self, tmp_path, monkeypatch):
        """Sanity: parser uses iterator-style read, not .read() / .readlines().

        Catches a future "let's just slurp" regression on a 350+ MB
        build.ninja. Wraps the real file in a small proxy that
        intercepts .read() / .readlines() (the slurp methods) while
        delegating context-management and iteration. The proxy
        approach avoids mutating attributes on the real
        ``TextIOWrapper`` instance, which is portability-fragile
        across CPython versions.
        """
        self._make_ninja(
            tmp_path,
            "build x.o: HIP_COMPILER__torch_hip_unscanned\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DA\n  FLAGS = -O3\n",
        )

        slurp_calls: list[str] = []

        class _NoSlurpFileProxy:
            def __init__(self, real_fh, name):
                self._fh = real_fh
                self._name = name

            def __enter__(self):
                self._fh.__enter__()
                return self

            def __exit__(self, *exc):
                return self._fh.__exit__(*exc)

            def __iter__(self):
                return iter(self._fh)

            def read(self, *a, **kw):
                slurp_calls.append(self._name)
                return self._fh.read(*a, **kw)

            def readlines(self, *a, **kw):
                slurp_calls.append(self._name)
                return self._fh.readlines(*a, **kw)

        real_open = env_mod.Path.open

        def tracking_open(self, *a, **kw):
            return _NoSlurpFileProxy(real_open(self, *a, **kw), self.name)

        monkeypatch.setattr(env_mod.Path, "open", tracking_open)
        env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        # iter(fh) doesn't invoke .read() / .readlines(); slurp would.
        assert slurp_calls == []


class TestCapturePytorchLegacyFindhipFallback:
    """1.4: per-source *.hip.o.cmake fallback for legacy FindHIP builds.

    When build.ninja has only CUSTOM_COMMAND rules for .hip compiles
    (no HIP_COMPILER), HIPCC flags live in per-source
    *.hip.o.cmake scripts instead. Exercised against ROCm 7.2
    rocm/pytorch-private:* Jenkins image shape.
    """

    @staticmethod
    def _make_build_tree(
        tmp_path: Path,
        cmake_scripts: dict[str, str],
        ninja_body: str | None = None,
    ) -> None:
        """Synthesize <tmp>/build/ with a build.ninja + per-source scripts.

        cmake_scripts maps relative path under build/ to file contents.
        ninja_body defaults to a CUSTOM_COMMAND-only stub (no
        HIP_COMPILER rule), which is what triggers the fallback.
        """
        build = tmp_path / "build"
        build.mkdir()
        if ninja_body is None:
            ninja_body = (
                "build foo.hip.o : CUSTOM_COMMAND src/foo.hip\n"
                "  COMMAND = cmake -P foo.hip.o.cmake\n"
            )
        (build / "build.ninja").write_text(ninja_body, encoding="utf-8")
        for relpath, body in cmake_scripts.items():
            target = build / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")

    def test_ck_sdpa_script_yields_full_target_block(self, tmp_path):
        """Real-shape fixture mirroring the repro image's
        ck_sdpa_generated_fmha_bwd_api.hip.o.cmake. Both HIP_HIPCC_FLAGS
        and HIP_CLANG_FLAGS contribute -D defines + flags that surface
        under the ck_sdpa target.
        """
        self._make_build_tree(tmp_path, {
            "caffe2/aten/src/ATen/CMakeFiles/ck_sdpa.dir/native/transformers/"
            "hip/flash_attn/ck/ck_sdpa_generated_fmha_bwd_api.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS --offload-compress;-std=c++17;"
                "-fgpu-flush-denormals-to-zero;-DCK_USE_FNUZ_FP8 -DCK_USE_GFX94 "
                "-DCK_USE_XDL -DUSE_ROCM_CK_SDPA -DROCM_VERSION=70200 "
                "-DCK_TILE_FMHA_FWD_FAST_EXP2=1 "
                "-DUSE_LAYERNORM_FAST_RECIPROCAL)\n"
                "set(HIP_CLANG_FLAGS -fPIC;-DUSE_ROCM;-DHIPBLAS_V2;"
                "-DHIPBLASLT_OUTER_VEC;-DUSE_ROCM_CK_GEMM;"
                "--offload-arch=gfx950;--offload-arch=gfx942)\n"
                "set(HIP_HIPCC_FLAGS_RELEASE )\n"
                "set(HIP_NVCC_FLAGS )\n"
            ),
        })
        reasons: list[str] = []
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, reasons)
        assert block["_parser"] == "legacy_findhip_per_source"
        assert block["_legacy_scripts_scanned"] == 1
        assert reasons == []
        assert set(block["targets"]) == {"ck_sdpa"}
        ck = block["targets"]["ck_sdpa"]
        # SDPA-critical defines from HIP_HIPCC_FLAGS
        assert ck["use_defines_present"]["USE_ROCM_CK_SDPA"] is True
        assert ck["use_defines_present"]["CK_TILE_FMHA_FWD_FAST_EXP2"] is True
        assert ck["use_defines_present"]["CK_USE_FNUZ_FP8"] is True
        assert ck["use_defines_present"]["USE_LAYERNORM_FAST_RECIPROCAL"] is True
        # Defines from HIP_CLANG_FLAGS must also be unioned in
        assert ck["use_defines_present"]["HIPBLAS_V2"] is True
        assert ck["use_defines_present"]["HIPBLASLT_OUTER_VEC"] is True
        assert ck["use_defines_present"]["USE_ROCM_CK_GEMM"] is True
        # Codegen flag picked up via substring scan
        assert ck["codegen_flags_present"]["-fgpu-flush-denormals-to-zero"] is True
        # --offload-arch values from HIP_CLANG_FLAGS, sorted
        assert ck["offload_archs"] == ["gfx942", "gfx950"]
        # Value parsing (not just presence) for ROCM_VERSION
        assert ck["defines"]["ROCM_VERSION"] == "70200"

    def test_scripts_for_multiple_targets_attributed_correctly(self, tmp_path):
        """Each script's parent path-segment maps to one target. A
        script under c10_hip.dir must not bleed into torch_hip.dir.
        """
        self._make_build_tree(tmp_path, {
            "caffe2/CMakeFiles/torch_hip.dir/aten/torch_hip_generated_a.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DTORCH_HIP_ONLY;--offload-arch=gfx942)\n"
            ),
            "c10/hip/CMakeFiles/c10_hip.dir/c10_hip_generated_b.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DC10_HIP_ONLY;--offload-arch=gfx950)\n"
            ),
        })
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert block["_parser"] == "legacy_findhip_per_source"
        assert block["_legacy_scripts_scanned"] == 2
        assert set(block["targets"]) == {"torch_hip", "c10_hip"}
        assert "TORCH_HIP_ONLY" in block["targets"]["torch_hip"]["defines"]
        assert "C10_HIP_ONLY" not in block["targets"]["torch_hip"]["defines"]
        assert "C10_HIP_ONLY" in block["targets"]["c10_hip"]["defines"]
        assert "TORCH_HIP_ONLY" not in block["targets"]["c10_hip"]["defines"]
        assert block["targets"]["torch_hip"]["offload_archs"] == ["gfx942"]
        assert block["targets"]["c10_hip"]["offload_archs"] == ["gfx950"]

    def test_scripts_under_non_interest_target_dirs_ignored(self, tmp_path):
        """gloo_hip / caffe2_nvrtc / HIP test binaries don't appear in
        _NINJA_HIPCC_TARGETS_OF_INTEREST. Scripts under their dirs
        must be silently skipped (not error, not appear in targets).
        """
        self._make_build_tree(tmp_path, {
            "third_party/gloo/gloo/CMakeFiles/gloo_hip.dir/x_generated.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DGLOO_HIP_DEFINE;--offload-arch=gfx900)\n"
            ),
            "caffe2/aten/src/ATen/CMakeFiles/ck_sdpa.dir/x_generated.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DCK_DEFINE;--offload-arch=gfx950)\n"
            ),
        })
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert set(block["targets"]) == {"ck_sdpa"}
        # Only the ck_sdpa script counts -- gloo_hip script wasn't read.
        assert block["_legacy_scripts_scanned"] == 1
        assert "GLOO_HIP_DEFINE" not in block["targets"]["ck_sdpa"]["defines"]

    def test_no_scripts_under_build_returns_empty_targets_with_reason(
        self, tmp_path,
    ):
        """build.ninja exists, ninja-only scan finds zero HIP_COMPILER
        rules, fallback walks build/ and finds zero *.hip.o.cmake
        either. Must leave targets: {} (not None) and append a
        partial reason explaining what was tried.
        """
        self._make_build_tree(tmp_path, {})  # only build.ninja, no scripts
        reasons: list[str] = []
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, reasons)
        assert block["targets"] == {}
        assert block["_parser"] is None
        assert block["_legacy_scripts_scanned"] is None
        assert any(
            "legacy FindHIP fallback found no *.hip.o.cmake scripts" in r
            for r in reasons
        )

    def test_all_scripts_unreadable_emits_distinct_partial_reason(
        self, tmp_path, monkeypatch,
    ):
        """When *.hip.o.cmake scripts exist under a target dir but
        every read raises OSError (e.g. permissions stripped),
        targets MUST end up `{}` AND a partial_reason MUST explain
        the permissions case -- not be silently dropped as if no
        scripts existed (Copilot round-7 review).
        """
        self._make_build_tree(tmp_path, {
            "caffe2/aten/src/ATen/CMakeFiles/ck_sdpa.dir/x.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DUSE_ROCM_CK_SDPA)\n"
            ),
        })
        real_open = env_mod.Path.open

        def deny_cmake_open(self, *args, **kwargs):
            if str(self).endswith(".hip.o.cmake"):
                raise PermissionError("denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(env_mod.Path, "open", deny_cmake_open)
        reasons: list[str] = []
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, reasons)
        assert block["targets"] == {}
        assert block["_parser"] is None
        # The new reason must explicitly mention unreadable scripts so
        # the operator knows to check permissions, not file presence.
        assert any(
            "unreadable" in r and "permissions" in r
            for r in reasons
        ), f"reasons did not mention unreadable / permissions: {reasons}"

    def test_oversized_script_emits_truncation_partial_reason(
        self, tmp_path,
    ):
        """A pathologically large *.hip.o.cmake (> 1 MiB) gets
        truncated to keep the read bounded, but tail defines/flags
        past the cap are silently dropped from the target. A partial
        reason MUST surface this so the operator can tell the
        snapshot is incomplete -- otherwise the data loss is invisible
        (Copilot round-9 review).
        """
        # Build a >1 MiB script by padding the value section. The
        # leading set(...) is still parseable so the target gets
        # populated; the truncation reason proves we noticed.
        pad = "x" * (env_mod._LEGACY_FINDHIP_MAX_FILE_BYTES + 64)
        self._make_build_tree(tmp_path, {
            "caffe2/aten/src/ATen/CMakeFiles/ck_sdpa.dir/big.hip.o.cmake": (
                f"set(HIP_HIPCC_FLAGS -DUSE_ROCM_CK_SDPA;{pad})\n"
            ),
        })
        reasons: list[str] = []
        block = env_mod._capture_pytorch_ninja_hipcc(
            "source", tmp_path, reasons,
        )
        # Defines from the leading portion still surface.
        assert "ck_sdpa" in block["targets"]
        # Reason must explicitly call out truncation so the operator
        # knows the snapshot is partial.
        assert any(
            "truncated" in r and "bytes" in r
            for r in reasons
        ), f"reasons missing truncation notice: {reasons}"

    def test_per_target_traversal_skips_non_interest_directories(
        self, tmp_path,
    ):
        """The parser must NOT iterate per-source scripts under target
        dirs we don't surface (gloo_hip / caffe2_nvrtc / HIP test
        binaries). The previous global rglob would walk every
        ``*.hip.o.cmake`` under build/ and discard non-interest paths
        post-hoc; the per-target form should never even open them
        (Copilot round-9 review).
        """
        self._make_build_tree(tmp_path, {
            # Interest target
            "caffe2/aten/src/ATen/CMakeFiles/ck_sdpa.dir/keep.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DUSE_ROCM_CK_SDPA)\n"
            ),
            # Non-interest target -- must not be opened
            "third_party/gloo/gloo/CMakeFiles/gloo_hip.dir/skip.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -DGLOO_HIP_DEFINE)\n"
            ),
        })
        opened: list[str] = []
        real_open = env_mod.Path.open

        def tracking_open(self, *args, **kwargs):
            if str(self).endswith(".hip.o.cmake"):
                opened.append(self.name)
            return real_open(self, *args, **kwargs)

        # Patch is on the env_mod re-export so the parser's calls hit it.
        import unittest.mock as _mock
        with _mock.patch.object(env_mod.Path, "open", tracking_open):
            block = env_mod._capture_pytorch_ninja_hipcc(
                "source", tmp_path, [],
            )
        assert "ck_sdpa" in block["targets"]
        assert "keep.hip.o.cmake" in opened
        # The gloo_hip script must never have been opened.
        assert "skip.hip.o.cmake" not in opened, (
            f"per-target traversal still walked non-interest scripts: {opened}"
        )

    def test_multiline_set_packed_with_d_defines_tokenized(self, tmp_path):
        """The packed-defines case: a single ;-element contains
        multiple space-separated -D defines (cmake variable-inheritance
        quirk). The tokenizer must split on BOTH `;` AND whitespace.
        """
        self._make_build_tree(tmp_path, {
            "caffe2/aten/src/ATen/CMakeFiles/ck_sdpa.dir/x.hip.o.cmake": (
                "set(HIP_HIPCC_FLAGS -std=c++17;"
                "-DA=1 -DB=2 -DC=3 -DUSE_ROCM_CK_SDPA;-Wall)\n"
            ),
        })
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        defines = block["targets"]["ck_sdpa"]["defines"]
        assert defines["A"] == "1"
        assert defines["B"] == "2"
        assert defines["C"] == "3"
        assert "USE_ROCM_CK_SDPA" in defines
        assert block["targets"]["ck_sdpa"]["use_defines_present"][
            "USE_ROCM_CK_SDPA"
        ] is True

    def test_modern_ninja_path_still_sets_parser_marker(self, tmp_path):
        """Schema-stability: when the modern ninja parser succeeds, the
        new _parser key must be set to "ninja_defines" -- consumers
        that check _parser to disambiguate strategies must get a clear
        signal, not None.
        """
        build = tmp_path / "build"
        build.mkdir()
        (build / "build.ninja").write_text(
            "build x.o: HIP_COMPILER__torch_hip_unscanned src/x.hip\n"
            "  DEFINES = -Dtorch_hip_EXPORTS -DUSE_ROCM_CK_SDPA\n"
            "  FLAGS = -O3 --offload-arch=gfx942\n",
            encoding="utf-8",
        )
        block = env_mod._capture_pytorch_ninja_hipcc("source", tmp_path, [])
        assert block["_parser"] == "ninja_defines"
        assert block["_legacy_scripts_scanned"] is None
        assert "torch_hip" in block["targets"]


class TestCaptureAiterHsaTree:
    """Issue #176: per-arch fingerprint of aiter's HSA code-object tree."""

    @staticmethod
    def _make_hsa(tmp_path: Path, layout: dict[str, dict[str, bytes]]) -> Path:
        """Create hsa/<gfx>/ tree from {gfx: {relpath: bytes}} mapping."""
        hsa = tmp_path / "hsa"
        for gfx, files in layout.items():
            arch_dir = hsa / gfx
            arch_dir.mkdir(parents=True)
            for relpath, data in files.items():
                target = arch_dir / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        return hsa

    def test_no_roots_returns_none(self, monkeypatch):
        # No aiter_meta, no AORTA_PYTORCH_SRC, no aiter module passed.
        monkeypatch.delenv(env_mod.AORTA_PYTORCH_SRC_ENV, raising=False)
        monkeypatch.setattr(
            "importlib.util.find_spec", lambda name: None,
        )
        assert env_mod._capture_aiter_hsa_tree(None, []) is None

    def test_aiter_meta_find_spec_root_picked_up(self, tmp_path, monkeypatch):
        """Primary documented HSA discovery path: a pip-installed
        `aiter_meta` whose ModuleSpec.submodule_search_locations
        points at the dist's site-packages dir. The hsa/ tree lives
        directly under that dir.
        """
        import importlib.util as _iutil
        import types

        site = tmp_path / "site-packages" / "aiter_meta"
        site.mkdir(parents=True)
        # Top-level marker file so the dir looks like a real pkg.
        (site / "__init__.py").write_text("")
        self._make_hsa(site, {"gfx942": {"k.co": b"abc", "meta.json": b"{}"}})

        fake_spec = types.SimpleNamespace(
            origin=str(site / "__init__.py"),
            submodule_search_locations=[str(site)],
        )
        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, "")
        monkeypatch.delenv(env_mod.AORTA_PYTORCH_SRC_ENV, raising=False)
        monkeypatch.setattr(
            _iutil, "find_spec",
            lambda name: fake_spec if name == "aiter_meta" else None,
        )
        out = env_mod._capture_aiter_hsa_tree(None, [])
        assert out is not None
        # Root was the find_spec location -- not AORTA_PYTORCH_SRC.
        roots = list(out.keys())
        assert len(roots) == 1
        assert "aiter_meta" in roots[0]
        assert out[roots[0]]["gfx942"]["co_count"] == 1
        assert out[roots[0]]["gfx942"]["file_count"] == 2

    def test_aorta_pytorch_src_root_picked_up(self, tmp_path, monkeypatch):
        third_party = tmp_path / "third_party" / "aiter"
        third_party.mkdir(parents=True)
        self._make_hsa(third_party, {"gfx942": {"k.co": b"abc", "meta.json": b"{}"}})
        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, str(tmp_path))
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        out = env_mod._capture_aiter_hsa_tree(None, [])
        assert out is not None and len(out) == 1
        per_arch = next(iter(out.values()))
        assert per_arch["gfx942"]["file_count"] == 2
        assert per_arch["gfx942"]["co_count"] == 1
        assert isinstance(per_arch["gfx942"]["combined_sha256"], str)

    def test_combined_sha256_deterministic_across_runs(self, tmp_path, monkeypatch):
        """Two runs over identical bytes produce identical hashes
        regardless of mtime / iteration order.
        """
        a_root = tmp_path / "tree_a" / "third_party" / "aiter"
        b_root = tmp_path / "tree_b" / "third_party" / "aiter"
        a_root.mkdir(parents=True)
        b_root.mkdir(parents=True)
        layout = {"gfx942": {"a.co": b"\x01\x02", "sub/b.co": b"\x03\x04"}}
        self._make_hsa(a_root, layout)
        self._make_hsa(b_root, layout)

        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, str(tmp_path / "tree_a"))
        out_a = env_mod._capture_aiter_hsa_tree(None, [])
        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, str(tmp_path / "tree_b"))
        out_b = env_mod._capture_aiter_hsa_tree(None, [])

        sha_a = next(iter(out_a.values()))["gfx942"]["combined_sha256"]
        sha_b = next(iter(out_b.values()))["gfx942"]["combined_sha256"]
        assert sha_a == sha_b

    def test_per_file_read_failure_nulls_combined_sha256_keeps_counts(
        self, tmp_path, monkeypatch
    ):
        """A partial-tree hash silently compares-equal to another
        partial-tree hash with the same readable subset, leading
        consumers to conclude two trees match when they may not.
        On any read failure the whole arch's combined_sha256 must
        be None; counts stay (the listing is still valid).
        """
        third_party = tmp_path / "third_party" / "aiter"
        third_party.mkdir(parents=True)
        self._make_hsa(third_party, {
            "gfx942": {"a.co": b"abc", "b.co": b"def"},
        })
        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, str(tmp_path))
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

        # Make `b.co` raise on read; `a.co` reads fine.
        real_open = env_mod.Path.open

        def selective_open(self, *a, **kw):
            if self.name == "b.co":
                raise PermissionError("denied")
            return real_open(self, *a, **kw)

        monkeypatch.setattr(env_mod.Path, "open", selective_open)
        reasons: list[str] = []
        out = env_mod._capture_aiter_hsa_tree(None, reasons)
        per_arch = next(iter(out.values()))["gfx942"]
        assert per_arch["combined_sha256"] is None
        assert per_arch["file_count"] == 2
        assert per_arch["co_count"] == 2
        assert any(
            r.startswith("aiter.hsa_tree: read failed") and "PermissionError" in r
            for r in reasons
        )

    def test_combined_sha256_changes_when_byte_changes(
        self, tmp_path, monkeypatch
    ):
        """Single-byte change in any .co produces a different hash --
        guards against an accidental regression to e.g. counting only
        file paths.
        """
        root_a = tmp_path / "a" / "third_party" / "aiter"
        root_b = tmp_path / "b" / "third_party" / "aiter"
        root_a.mkdir(parents=True)
        root_b.mkdir(parents=True)
        self._make_hsa(root_a, {"gfx942": {"k.co": b"\x00"}})
        self._make_hsa(root_b, {"gfx942": {"k.co": b"\x01"}})
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, str(tmp_path / "a"))
        sha_a = next(iter(
            env_mod._capture_aiter_hsa_tree(None, []).values()
        ))["gfx942"]["combined_sha256"]
        monkeypatch.setenv(env_mod.AORTA_PYTORCH_SRC_ENV, str(tmp_path / "b"))
        sha_b = next(iter(
            env_mod._capture_aiter_hsa_tree(None, []).values()
        ))["gfx942"]["combined_sha256"]
        assert sha_a != sha_b


class TestCapturePytorchSdpa:
    """Issue #176: runtime SDPA backend state."""

    def test_torch_absent_yields_all_none(self, isolated_env, monkeypatch):
        import builtins
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("simulated"))
                if name == "torch"
                else real_import(name, *a, **kw)
            ),
        )
        block = env_mod._capture_pytorch_sdpa([])
        assert block["backends_enabled"] == {
            name: None for name in env_mod._PYTORCH_SDPA_GETTERS
        }

    def test_all_getters_present_returns_bools(self, isolated_env, monkeypatch):
        import builtins
        import types
        cuda_ns = types.SimpleNamespace(
            flash_sdp_enabled=lambda: True,
            mem_efficient_sdp_enabled=lambda: True,
            math_sdp_enabled=lambda: True,
            cudnn_sdp_enabled=lambda: False,
        )
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(cuda=cuda_ns),
        )
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (
                fake_torch if name == "torch" else real_import(name, *a, **kw)
            ),
        )
        block = env_mod._capture_pytorch_sdpa([])
        assert block["backends_enabled"] == {
            "flash_sdp_enabled": True,
            "mem_efficient_sdp_enabled": True,
            "math_sdp_enabled": True,
            "cudnn_sdp_enabled": False,
        }

    def test_missing_getter_renders_none_not_false(
        self, isolated_env, monkeypatch
    ):
        """Older torch wheels lack one or more getters; that's
        distinguishable from `False` (which means "we asked, the
        backend is disabled").
        """
        import builtins
        import types
        cuda_ns = types.SimpleNamespace(
            flash_sdp_enabled=lambda: True,
            # mem_efficient_sdp_enabled / math_sdp_enabled / cudnn_sdp_enabled
            # intentionally absent.
        )
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(cuda=cuda_ns),
        )
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (
                fake_torch if name == "torch" else real_import(name, *a, **kw)
            ),
        )
        block = env_mod._capture_pytorch_sdpa([])
        assert block["backends_enabled"]["flash_sdp_enabled"] is True
        assert block["backends_enabled"]["mem_efficient_sdp_enabled"] is None
        assert block["backends_enabled"]["math_sdp_enabled"] is None
        assert block["backends_enabled"]["cudnn_sdp_enabled"] is None


class TestPytorchSdpaSnapshotRoundTrip:
    """Schema regression: 1.2 snapshots without `pytorch_sdpa` must
    round-trip through 1.3 `from_dict` and emerge with the dataclass-
    default backends_enabled shape.
    """

    def test_legacy_snapshot_without_pytorch_sdpa_loads(self):
        d = _example_snapshot().to_dict()
        del d["pytorch_sdpa"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.pytorch_sdpa == {
            "backends_enabled": {
                name: None for name in env_mod._PYTORCH_SDPA_GETTERS
            }
        }


class TestSummaryNewBriefLines:
    """Issue #176 brief one-liners: cmake cache, ninja hipcc,
    aiter HSA tree, SDPA. Each tests both the available and
    unavailable rendering so wording / shape regressions surface.
    """

    @staticmethod
    def _snap_with_pytorch_build(pytorch_build_overrides):
        base = _example_snapshot()
        return _example_snapshot(
            pytorch_build={**base.pytorch_build, **pytorch_build_overrides}
        )

    def _line(self, snap, prefix):
        for ln in snap.summary().splitlines():
            if ln.lstrip().startswith(prefix):
                return ln
        raise AssertionError(f"no `{prefix}` line in summary")

    # ---- nics ----
    def test_nics_undeterminable_presence_renders_question_mark(self):
        # present=None (e.g. lspci missing/failed) must surface as
        # "<vendor>(?)", never be silently dropped into "(none present)".
        snap = _example_snapshot(nics={
            "ainic": {"present": None},
            "broadcom": {"present": False},
            "cx7": {"present": None},
        })
        line = self._line(snap, "nics:")
        assert "ainic(?)" in line
        assert "cx7(?)" in line
        assert "(none present)" not in line

    def test_nics_all_absent_renders_none_present(self):
        snap = _example_snapshot(nics={
            "ainic": {"present": False},
            "broadcom": {"present": False},
            "cx7": {"present": False},
        })
        line = self._line(snap, "nics:")
        assert "(none present)" in line

    def test_nics_present_vendor_renders_fw_and_links(self):
        snap = _example_snapshot(nics={
            "ainic": {"present": False},
            "broadcom": {
                "present": True,
                "firmware": "232.0.219.16",
                "links": [
                    {"device": "bnxt_re0", "state": "ACTIVE"},
                    {"device": "bnxt_re1", "state": "DOWN"},
                ],
            },
            "cx7": {"present": False},
        })
        line = self._line(snap, "nics:")
        assert "broadcom(fw=232.0.219.16 links=1/2)" in line

    # ---- cmake cache ----
    def test_cmake_cache_unavailable_renders_explicit_message(self):
        snap = self._snap_with_pytorch_build({
            "cmake_cache": {"_source_file": None, "entries": None},
        })
        line = self._line(snap, "cmake cache:")
        assert "unavailable" in line

    def test_cmake_cache_available_renders_count_and_path(self):
        snap = self._snap_with_pytorch_build({
            "cmake_cache": {
                "_source_file": "/work/build/CMakeCache.txt",
                "entries": {
                    "USE_ROCM": {"type": "BOOL", "value": "ON"},
                    "BUILD_TYPE": {"type": "STRING", "value": "Release"},
                },
            },
        })
        line = self._line(snap, "cmake cache:")
        assert "2 allowlisted entries" in line
        assert "/work/build/CMakeCache.txt" in line

    # ---- ninja hipcc ----
    def test_ninja_hipcc_unavailable_renders_explicit_message(self):
        snap = self._snap_with_pytorch_build({
            "ninja_hipcc": {"_source_file": None, "targets": None},
        })
        line = self._line(snap, "ninja hipcc:")
        assert "unavailable" in line

    def test_ninja_hipcc_available_renders_per_target_define_count(self):
        snap = self._snap_with_pytorch_build({
            "ninja_hipcc": {
                "_source_file": "/work/build/build.ninja",
                "targets": {
                    "torch_hip": {
                        "defines": {"A": None, "B": "1", "C": None},
                        "use_defines_present": {},
                        "codegen_flags_present": {},
                        "offload_archs": ["gfx942"],
                    },
                },
            },
        })
        line = self._line(snap, "ninja hipcc:")
        assert "torch_hip=3D" in line
        assert "archs=[gfx942]" in line

    def test_ninja_hipcc_scanned_no_matches_renders_no_targets(self):
        snap = self._snap_with_pytorch_build({
            "ninja_hipcc": {
                "_source_file": "/work/build/build.ninja",
                "targets": {},
            },
        })
        line = self._line(snap, "ninja hipcc:")
        assert "no targets of interest matched" in line

    # ---- aiter HSA tree ----
    def test_aiter_hsa_tree_absent_renders_not_present(self):
        snap = _example_snapshot(
            aiter={**_example_snapshot().aiter, "hsa_tree": None}
        )
        line = self._line(snap, "aiter hsa:")
        assert line.endswith("(not present)")

    def test_aiter_hsa_tree_brief_disambiguates_roots(self):
        """Two roots both shipping gfx942 must NOT collapse to two
        unlabelled `gfx942=...` cells.
        """
        snap = _example_snapshot(
            aiter={
                **_example_snapshot().aiter,
                "hsa_tree": {
                    "/usr/lib/python/site-packages/aiter_meta/hsa": {
                        "gfx942": {
                            "file_count": 100, "co_count": 90,
                            "combined_sha256": "aaaaaaaa" + "0" * 56,
                        },
                    },
                    "/work/pytorch/third_party/aiter/hsa": {
                        "gfx942": {
                            "file_count": 99, "co_count": 89,
                            "combined_sha256": "bbbbbbbb" + "0" * 56,
                        },
                    },
                },
            },
        )
        line = self._line(snap, "aiter hsa:")
        # Each root identified by its last 2 path components -> distinct.
        assert "aiter_meta/hsa:gfx942=" in line
        assert "aiter/hsa:gfx942=" in line
        # Distinct hashes also surface so a diff is visible.
        assert "aaaaaaaa" in line
        assert "bbbbbbbb" in line

    # ---- SDPA ----
    def test_sdpa_unavailable_renders_explicit_message(self):
        snap = _example_snapshot(
            pytorch_sdpa={"backends_enabled": {
                name: None for name in env_mod._PYTORCH_SDPA_GETTERS
            }}
        )
        line = self._line(snap, "sdpa:")
        assert "unavailable" in line

    def test_sdpa_mixed_state_renders_compact_form(self):
        snap = _example_snapshot(
            pytorch_sdpa={"backends_enabled": {
                "flash_sdp_enabled": True,
                "mem_efficient_sdp_enabled": False,
                "math_sdp_enabled": True,
                "cudnn_sdp_enabled": None,
            }}
        )
        line = self._line(snap, "sdpa:")
        assert "flash=on" in line
        assert "mem_eff=off" in line
        assert "math=on" in line
        assert "cudnn=?" in line


class TestSafeImportTorch:
    """Centralised torch-import helper used by every probe that needs
    torch (composable_kernel, aotriton, fbgemm-flag scan, CK-flag scan,
    pytorch_build).
    """

    def test_import_error_silent(self, isolated_env, monkeypatch):
        """ImportError -> None, no reason added (pytorch_version probe
        records the absence elsewhere, this helper must not double-count).
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        result = env_mod._safe_import_torch(reasons, "test_probe")
        assert result is None
        assert reasons == []

    def test_unexpected_exception_records_with_probe_name(
        self, isolated_env, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise RuntimeError("C ext load failed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        result = env_mod._safe_import_torch(reasons, "composable_kernel.foo")
        assert result is None
        assert len(reasons) == 1
        assert reasons[0].startswith("composable_kernel.foo: torch import raised")
        assert "RuntimeError" in reasons[0]

    def test_success_returns_module(self, isolated_env, monkeypatch):
        import builtins
        import types

        fake_torch = types.SimpleNamespace(__version__="2.99.0")
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        result = env_mod._safe_import_torch(reasons, "test_probe")
        assert result is fake_torch
        assert reasons == []


class TestHashFilePath:
    """Helper that hashes a caller-supplied Path (vs
    ``_hash_shared_library`` which globs and string-sorts internally).
    """

    def test_hash_specific_file(self, tmp_path: Path):
        f = tmp_path / "libfoo.so.0.10.0"
        f.write_bytes(b"specific bytes")
        digest = env_mod._hash_file_path(f)
        expected = "sha256:" + hashlib.sha256(b"specific bytes").hexdigest()
        assert digest == expected

    def test_resolves_symlink(self, tmp_path: Path):
        real = tmp_path / "real.so"
        real.write_bytes(b"real bytes")
        link = tmp_path / "link.so"
        link.symlink_to(real)
        digest = env_mod._hash_file_path(link)
        expected = "sha256:" + hashlib.sha256(b"real bytes").hexdigest()
        assert digest == expected

    def test_missing_returns_none(self, tmp_path: Path):
        assert env_mod._hash_file_path(tmp_path / "nonexistent") is None

    def test_directory_returns_none(self, tmp_path: Path):
        d = tmp_path / "adir"
        d.mkdir()
        assert env_mod._hash_file_path(d) is None


class TestPythonPackageVersionHelper:
    def test_suppress_missing_skips_import_reason(self, isolated_env, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fakepkg":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        result = env_mod._capture_python_package_version(
            "fakepkg", reasons, suppress_missing=True
        )
        assert result is None
        assert reasons == []

    def test_suppress_missing_still_reports_other_failures(
        self, isolated_env, monkeypatch
    ):
        """suppress_missing only swallows ImportError, not broken __version__."""
        import builtins
        import types

        real_import = builtins.__import__
        fake_mod = types.SimpleNamespace()  # no __version__

        def fake_import(name, *args, **kwargs):
            if name == "brokenpkg":
                return fake_mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        result = env_mod._capture_python_package_version(
            "brokenpkg", reasons, suppress_missing=True
        )
        assert result is None
        assert any("__version__" in r for r in reasons)

    def test_custom_reason_prefix_used_in_output(
        self, isolated_env, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fakepkg":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasons: list[str] = []
        env_mod._capture_python_package_version(
            "fakepkg", reasons, reason_prefix="custom.thing"
        )
        assert any(r.startswith("custom.thing:") for r in reasons)


# ---------------------------------------------------------------------------
# library_linkage: DT_RPATH vs DT_RUNPATH (schema 1.17, issue #413)
# ---------------------------------------------------------------------------

DT_NULL = 0
DT_SONAME = 14
DT_RPATH = 15
DT_RUNPATH = 29


def _write_elf(
    path: Path,
    tags: tuple[int, ...],
    *,
    bits: int = 64,
    endian: str = "<",
    terminate: bool = True,
) -> Path:
    """Write a minimal but structurally valid ELF carrying *tags* in .dynamic.

    Synthesised rather than compiled: the probe's whole job is to read the
    dynamic section of libraries built by someone else's toolchain, so the
    test must be able to produce an RPATH object on a host whose linker
    defaults to RUNPATH (and vice versa) without needing a compiler at all.
    Only the fields the parser reads are meaningful; everything else is
    zero-filled. ``terminate=False`` omits the trailing ``DT_NULL`` the ABI
    requires, producing the malformed shape the parser must reject rather
    than read as "carries neither tag".
    """
    is_64 = bits == 64
    ehdr_size = 64 if is_64 else 52
    phentsize = 56 if is_64 else 32
    dyn_entry = 16 if is_64 else 8
    phoff = ehdr_size
    dynoff = phoff + phentsize

    ident = b"\x7fELF" + bytes([2 if is_64 else 1, 1 if endian == "<" else 2, 1]) + bytes(9)
    if is_64:
        ehdr = ident + struct.pack(
            f"{endian}HHIQQQIHHHHHH",
            3, 62, 1, 0, phoff, 0, 0, ehdr_size, phentsize, 1, 0, 0, 0,
        )
    else:
        ehdr = ident + struct.pack(
            f"{endian}HHIIIIIHHHHHH",
            3, 3, 1, 0, phoff, 0, 0, ehdr_size, phentsize, 1, 0, 0, 0,
        )

    dyn_entries = [*tags, DT_NULL] if terminate else list(tags)
    dyn_size = len(dyn_entries) * dyn_entry
    if is_64:
        phdr = struct.pack(
            f"{endian}IIQQQQQQ", 2, 0, dynoff, 0, 0, dyn_size, dyn_size, 8
        )
        dynamic = b"".join(
            struct.pack(f"{endian}qQ", tag, 0) for tag in dyn_entries
        )
    else:
        phdr = struct.pack(
            f"{endian}IIIIIIII", 2, dynoff, 0, 0, dyn_size, dyn_size, 0, 4
        )
        dynamic = b"".join(
            struct.pack(f"{endian}iI", tag, 0) for tag in dyn_entries
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ehdr + phdr + dynamic)
    return path


class TestReadElfSearchPathTags:
    """The ELF parse itself: what the file says, nothing inferred."""

    def test_reads_rpath(self, tmp_path: Path):
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME, DT_RPATH))
        assert env_mod._read_elf_search_path_tags(lib) == (["rpath"], None)

    def test_reads_runpath(self, tmp_path: Path):
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME, DT_RUNPATH))
        assert env_mod._read_elf_search_path_tags(lib) == (["runpath"], None)

    def test_neither_tag_is_a_clean_read_not_an_error(self, tmp_path: Path):
        """``([], None)`` -- "carries neither" is an answer, not a failure.

        The distinction from ``([], reason)`` is the whole point: a
        statically-pathed library and an unreadable one must not collapse
        into the same reading.
        """
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME,))
        assert env_mod._read_elf_search_path_tags(lib) == ([], None)

    def test_both_tags_are_both_reported(self, tmp_path: Path):
        """The raw reading stays lossless; the loader's preference is derived."""
        lib = _write_elf(tmp_path / "libfoo.so", (DT_RPATH, DT_RUNPATH))
        tags, error = env_mod._read_elf_search_path_tags(lib)
        assert (tags, error) == (["rpath", "runpath"], None)
        # glibc ignores DT_RPATH entirely once DT_RUNPATH is present.
        assert env_mod._effective_link_tag(tags) == "runpath"

    def test_32bit_big_endian_parses(self, tmp_path: Path):
        """Not an x86-64 parser. DT_RPATH is 15 on every ABI."""
        lib = _write_elf(
            tmp_path / "libfoo.so", (DT_RPATH,), bits=32, endian=">"
        )
        assert env_mod._read_elf_search_path_tags(lib) == (["rpath"], None)

    def test_non_elf_file_reports_a_reason(self, tmp_path: Path):
        plain = tmp_path / "libfoo.so"
        plain.write_text("I am a linker script, not an object\n")
        tags, error = env_mod._read_elf_search_path_tags(plain)
        assert tags == []
        assert error == "not an ELF file"

    def test_truncated_elf_reports_a_reason(self, tmp_path: Path):
        stub = tmp_path / "libfoo.so"
        stub.write_bytes(b"\x7fELF\x02\x01\x01" + bytes(9))
        tags, error = env_mod._read_elf_search_path_tags(stub)
        assert tags == []
        assert error is not None

    def test_missing_file_never_raises(self, tmp_path: Path):
        tags, error = env_mod._read_elf_search_path_tags(tmp_path / "nope.so")
        assert tags == []
        assert error is not None

    def test_absurd_program_header_count_is_bounded(self, tmp_path: Path):
        """A corrupt header must not make the probe allocate gigabytes."""
        lib = _write_elf(tmp_path / "libfoo.so", (DT_RPATH,))
        raw = bytearray(lib.read_bytes())
        struct.pack_into("<H", raw, 0x38, 60000)  # e_phnum
        lib.write_bytes(bytes(raw))
        tags, error = env_mod._read_elf_search_path_tags(lib)
        assert tags == []
        assert error is not None


# Byte offset of ``p_filesz`` in the single 64-bit program header ``_write_elf``
# emits: 64-byte ELF header, then p_type(4) p_flags(4) p_offset(8) p_vaddr(8)
# p_paddr(8) before it.
_P_FILESZ_OFFSET = 64 + 32


def _set_declared_dynamic_size(path: Path, size: int) -> None:
    """Rewrite ``PT_DYNAMIC``'s ``p_filesz`` without touching the bytes it covers."""
    raw = bytearray(path.read_bytes())
    struct.pack_into("<Q", raw, _P_FILESZ_OFFSET, size)
    path.write_bytes(bytes(raw))


class TestMalformedDynamicSectionIsUnreadableNotEmpty:
    """A malformed ``PT_DYNAMIC`` must never read as "carries neither tag".

    ``([], None)`` is a positive claim -- "we read the whole section and no
    search-path tag was in it" -- which flows into ``rocm_rpath=False`` and
    silences the triage runner's substitution warning. Reporting it for
    input we did not actually manage to read whole is a silent false
    negative inside the one feature whose entire purpose is to stop a
    silent failure, so every case below must come back as a reason string.
    """

    def test_truncated_section_is_rejected_even_with_a_complete_entry(
        self, tmp_path: Path
    ):
        """One intact entry before the cut is not the section.

        The file holds a single readable ``DT_SONAME`` entry while
        ``p_filesz`` declares three, so a length-unchecked parse finds a
        well-formed entry, no tag, and reports "carries neither" for a
        library whose ``DT_RPATH`` is in the part that is missing.
        """
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME, DT_RPATH))
        raw = lib.read_bytes()
        declared = 3 * 16
        lib.write_bytes(raw[: len(raw) - 2 * 16])  # keep only DT_SONAME
        _set_declared_dynamic_size(lib, declared)
        assert env_mod._read_elf_search_path_tags(lib) == (
            [],
            "truncated dynamic section",
        )

    def test_misaligned_section_is_rejected(self, tmp_path: Path):
        """``p_filesz`` that is not a whole number of entries is not a section."""
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME, DT_RPATH))
        _set_declared_dynamic_size(lib, 30)
        tags, error = env_mod._read_elf_search_path_tags(lib)
        assert tags == []
        assert error is not None
        assert "misaligned" in error

    def test_oversized_section_is_rejected_not_clamped(self, tmp_path: Path):
        """The regression that motivated the check.

        Clamping the read to the byte bound and parsing what fits is the
        tempting cheap fix, and it is wrong precisely when it matters: here
        the ``DT_RPATH`` sits past the clamp, so a clamped parse returns
        ``([], None)`` -- a confident "this stack is substitution-safe"
        derived from the half of the section it never looked at.
        """
        padding = (env_mod._MAX_DYNAMIC_BYTES // 16 + 8) * (DT_SONAME,)
        lib = _write_elf(tmp_path / "libfoo.so", (*padding, DT_RPATH))
        tags, error = env_mod._read_elf_search_path_tags(lib)
        assert tags == []
        assert error is not None
        assert "implausibly large" in error

    def test_section_without_dt_null_is_rejected(self, tmp_path: Path):
        """The ABI terminator is how we know we saw the whole array."""
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME,), terminate=False)
        assert env_mod._read_elf_search_path_tags(lib) == (
            [],
            "dynamic section not DT_NULL-terminated",
        )

    def test_empty_section_is_rejected(self, tmp_path: Path):
        lib = _write_elf(tmp_path / "libfoo.so", ())
        _set_declared_dynamic_size(lib, 0)
        assert env_mod._read_elf_search_path_tags(lib) == (
            [],
            "empty dynamic section",
        )

    def test_a_well_formed_object_is_still_a_clean_read(self, tmp_path: Path):
        """The strictness must not turn real libraries into unreadable ones.

        Measured alongside this: with these checks in place the census reads
        57/57 objects on the classic ROCm 7.0.2 host and 61/61 in the ROCm 10
        wheel image with zero unreadable, so the bar is "malformed", not
        "unusual".
        """
        lib = _write_elf(tmp_path / "libfoo.so", (DT_SONAME,))
        assert env_mod._read_elf_search_path_tags(lib) == ([], None)


class TestCaptureLibraryLinkage:
    """The block: per-library records plus the ANY aggregate Layer 2 uses."""

    @pytest.fixture
    def rocm_libs(self, tmp_path: Path, monkeypatch):
        """Point every ROCm lib-dir constant at one empty temp directory."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        for name in (
            "ROCM_CORE_LIB_DIR",
            "HIPBLASLT_LIB_DIR",
            "ROCBLAS_LIB_DIR",
            "MIOPEN_LIB_DIR",
            "RCCL_LIB_DIR",
        ):
            monkeypatch.setattr(env_mod, name, lib_dir)
        monkeypatch.setattr(env_mod, "_torch_native_lib_dir", lambda _mod: None)
        monkeypatch.setattr(
            env_mod, "_safe_import_torch", lambda _reasons, _name: None
        )
        return lib_dir

    def test_rpath_stack_sets_the_trigger(self, rocm_libs):
        _write_elf(rocm_libs / "libhipblaslt.so.1", (DT_RPATH,))
        block = env_mod._capture_library_linkage()
        assert block["status"] == "ok"
        assert block["rocm_rpath"] is True
        assert block["tags_observed"] == ["rpath"]

    def test_runpath_stack_leaves_the_trigger_off(self, rocm_libs):
        for soname in ("libhipblaslt.so.1", "librocblas.so.5", "libamdhip64.so.7"):
            _write_elf(rocm_libs / soname, (DT_RUNPATH,))
        block = env_mod._capture_library_linkage()
        assert block["status"] == "ok"
        assert block["rocm_rpath"] is False
        assert block["tags_observed"] == ["runpath"]

    def test_a_single_rpath_among_runpaths_sets_the_trigger(self, rocm_libs):
        """ANY, not all -- and this is the case that made it ANY.

        #413 measured an RPATH on a NEIGHBOURING library defeating a
        hipBLASLt substitution. One tagged object anywhere in the chain is
        sufficient, because DT_RPATH is inherited by everything loaded
        beneath it, so an "all" or "majority" rule would under-report the
        hazard on exactly the mixed stacks where it bites.
        """
        _write_elf(rocm_libs / "libhipblaslt.so.1", (DT_RUNPATH,))
        _write_elf(rocm_libs / "librocblas.so.5", (DT_RUNPATH,))
        _write_elf(rocm_libs / "libamdhip64.so.7", (DT_RPATH,))
        block = env_mod._capture_library_linkage()
        assert block["rocm_rpath"] is True
        assert block["tags_observed"] == ["rpath", "runpath"]

    def test_mixed_tags_are_not_collapsed_to_one_verdict(
        self, rocm_libs, tmp_path, monkeypatch
    ):
        """The real ROCm 10 shape: ROCm on RPATH, torch's own lib on RUNPATH.

        Both readings must survive in the record. A reader who saw only a
        single per-image verdict would conclude torch is on RPATH too, and
        #413's end-to-end failure is precisely one where torch's DT_RUNPATH
        did not rescue the substitution.
        """
        _write_elf(rocm_libs / "libhipblaslt.so.1", (DT_RPATH,))
        torch_lib = tmp_path / "torch" / "lib"
        _write_elf(torch_lib / "libtorch_hip.so", (DT_RUNPATH,))
        monkeypatch.setattr(
            env_mod, "_torch_native_lib_dir", lambda _mod: torch_lib
        )

        block = env_mod._capture_library_linkage()
        by_name = {e["name"]: e for e in block["libraries"]}
        assert by_name["libhipblaslt.so"]["effective_tag"] == "rpath"
        assert by_name["libtorch_hip.so"]["effective_tag"] == "runpath"
        assert by_name["libtorch_hip.so"]["scope"] == "pytorch"
        assert sorted(block["tags_observed"]) == ["rpath", "runpath"]
        # torch's RUNPATH must not dilute the ROCm-scope aggregate.
        assert block["rocm_rpath"] is True

    def test_absent_install_claims_nothing(self, rocm_libs):
        """No libraries found is not evidence the stack is substitution-safe."""
        block = env_mod._capture_library_linkage()
        assert block["status"] == "absent"
        assert block["rocm_rpath"] is None
        assert block["tags_observed"] == []
        assert all(e["path"] is None for e in block["libraries"])
        assert all(e["effective_tag"] == "unknown" for e in block["libraries"])

    def test_present_but_unreadable_is_distinct_from_absent(self, rocm_libs):
        """The rocm_paths ``source="none"`` principle, applied here.

        "There is no hipBLASLt on this host" and "there it is, and we could
        not read it" are different operator problems and must not render
        identically.
        """
        (rocm_libs / "libhipblaslt.so.1").write_text("not an object file")
        block = env_mod._capture_library_linkage()
        assert block["status"] == "unreadable"
        assert block["rocm_rpath"] is None
        entry = next(
            e for e in block["libraries"] if e["name"] == "libhipblaslt.so"
        )
        assert entry["path"] is not None
        assert entry["effective_tag"] == "unknown"
        assert entry["reason"] == "not an ELF file"

    def test_never_raises_on_a_hostile_tree(self, tmp_path, monkeypatch):
        """Fail-soft contract: a lib dir that is a FILE must not explode."""
        broken = tmp_path / "not-a-dir"
        broken.write_text("x")
        for name in (
            "ROCM_CORE_LIB_DIR",
            "HIPBLASLT_LIB_DIR",
            "ROCBLAS_LIB_DIR",
            "MIOPEN_LIB_DIR",
            "RCCL_LIB_DIR",
        ):
            monkeypatch.setattr(env_mod, name, broken)
        monkeypatch.setattr(env_mod, "_torch_native_lib_dir", lambda _mod: None)
        monkeypatch.setattr(
            env_mod, "_safe_import_torch", lambda _reasons, _name: None
        )
        block = env_mod._capture_library_linkage()
        assert block["status"] == "absent"
        assert block["rocm_rpath"] is None

    def test_reads_the_same_file_the_identity_block_hashes(self, rocm_libs):
        """Shared candidate ordering, so path and lib_hash cannot disagree.

        Two versioned siblings present: the linkage record must name the
        same one ``_hash_shared_library`` picks, and that choice is by
        integer tuple (5.10.0 > 5.9.0), not lexicographic.
        """
        _write_elf(rocm_libs / "librocblas.so.5.9.0", (DT_RUNPATH,))
        _write_elf(rocm_libs / "librocblas.so.5.10.0", (DT_RPATH,))
        entry = next(
            e
            for e in env_mod._capture_library_linkage()["libraries"]
            if e["name"] == "librocblas.so"
        )
        assert entry["path"].endswith("librocblas.so.5.10.0")
        assert entry["effective_tag"] == "rpath"

    def test_appends_no_partial_reasons(self, rocm_libs):
        """This block reports its failures in-band, not via partial_reasons.

        Nothing in it is ever None -- an unreadable file is the positive
        value ``effective_tag="unknown"`` plus a reason -- so it has no
        "expected to populate, fell back" case for ``partial`` to describe,
        and a second copy of a broken library would only duplicate what the
        identity block already recorded.
        """
        (rocm_libs / "libhipblaslt.so.1").write_text("junk")
        # There is no reasons list to append to -- the contract is structural,
        # not a promise the body happens to keep. docs/env-probe.md states it
        # in the same terms ("never raises a partial_reason"); this is what
        # stops that row from drifting back into describing one.
        assert not inspect.signature(env_mod._capture_library_linkage).parameters

        block = env_mod._capture_library_linkage()
        # And the failure is reported positively instead, in-band.
        entry = next(
            e for e in block["libraries"] if e["name"] == "libhipblaslt.so"
        )
        assert entry["effective_tag"] == "unknown"
        assert entry["reason"] == "not an ELF file"
        assert block["census"]["unreadable"] == 1


class TestLinkageCensusNotACuratedSample:
    """``rocm_rpath`` must describe the install, not the five names we picked.

    The aggregate claims "some object in the loading chain carries
    ``DT_RPATH``". A hand-maintained sample can only support "some object I
    listed does", and the two coincide only while the stack is uniform. On
    the pinned ROCm 10 image every object is RPATH, so the sample answered
    correctly by correlation -- while the object #413 actually measured
    defeating a ``libhipblaslt`` substitution, ``libhipblas.so.3``, was
    never in it. These tests pin the aggregate to the directory census so
    the claim is true by construction instead.
    """

    @pytest.fixture
    def rocm_libs(self, tmp_path: Path, monkeypatch):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        for name in (
            "ROCM_CORE_LIB_DIR",
            "HIPBLASLT_LIB_DIR",
            "ROCBLAS_LIB_DIR",
            "MIOPEN_LIB_DIR",
            "RCCL_LIB_DIR",
        ):
            monkeypatch.setattr(env_mod, name, lib_dir)
        monkeypatch.setattr(env_mod, "_torch_native_lib_dir", lambda _mod: None)
        monkeypatch.setattr(
            env_mod, "_safe_import_torch", lambda _reasons, _name: None
        )
        return lib_dir

    def _mixed_stack(self, lib_dir: Path) -> None:
        """Every sampled library on RUNPATH; only libhipblas on RPATH.

        The stack the curated list silently got wrong. Nothing here is
        hypothetical: hipBLAS is the neighbour whose inherited ``DT_RPATH``
        ``LD_DEBUG=libs`` named in #413's end-to-end reproduction, and it is
        not one of the libraries the identity blocks hash.
        """
        for soname in (
            "libamdhip64.so.7",
            "libhipblaslt.so.1",
            "librocblas.so.5",
            "libMIOpen.so.1",
            "librccl.so.1",
        ):
            _write_elf(lib_dir / soname, (DT_RUNPATH,))
        _write_elf(lib_dir / "libhipblas.so.3", (DT_RPATH,))

    def test_unsampled_rpath_library_still_sets_the_trigger(self, rocm_libs):
        """The case the sampled list would have answered False on."""
        self._mixed_stack(rocm_libs)
        block = env_mod._capture_library_linkage()

        # Every named entry reads RUNPATH -- so the verdict cannot be coming
        # from them, which is exactly the point.
        assert {e["effective_tag"] for e in block["libraries"]} == {"runpath"}
        assert block["rocm_rpath"] is True

    def test_the_census_names_the_object_the_sample_omits(self, rocm_libs):
        """The evidence ships with the verdict, so it is checkable, not trusted."""
        self._mixed_stack(rocm_libs)
        census = env_mod._capture_library_linkage()["census"]
        assert census["rpath_examples"] == ["libhipblas.so.3"]
        assert census["rpath"] == 1
        assert census["runpath"] == 5
        assert census["scanned"] == 6
        assert census["truncated"] is False

    def test_a_uniformly_runpath_install_still_reads_false(self, rocm_libs):
        """The census must not manufacture the hazard it is looking for.

        Measured on the classic ROCm 7.0.2 host this repo runs on: 57 of 57
        objects under ``/opt/rocm/lib`` carry ``DT_RUNPATH`` and none carry
        ``DT_RPATH``, so widening the aggregate from five names to the whole
        directory keeps the triage warning silent on every ROCm 7 run.
        """
        for soname in ("libamdhip64.so.7", "libhipblas.so.3", "libfoo.so.1"):
            _write_elf(rocm_libs / soname, (DT_RUNPATH,))
        block = env_mod._capture_library_linkage()
        assert block["rocm_rpath"] is False
        assert block["census"]["rpath"] == 0

    def test_kernel_objects_are_not_counted_as_libraries(self, rocm_libs):
        """``Kernels.so-000-gfx950.hsaco`` is a code object, not a library.

        It has no ``PT_DYNAMIC``, so counting it would add a permanent
        ``unreadable`` to every wheel-layout census and make a real
        unreadable library harder to notice.
        """
        _write_elf(rocm_libs / "libamdhip64.so.7", (DT_RUNPATH,))
        (rocm_libs / "Kernels.so-000-gfx950.hsaco").write_bytes(b"\x7fELFjunk")
        census = env_mod._capture_library_linkage()["census"]
        assert census["scanned"] == 1
        assert census["unreadable"] == 0

    def test_an_unreadable_object_does_not_read_as_no_rpath(self, rocm_libs):
        """Counted separately, and it cannot be what makes the answer False."""
        (rocm_libs / "libbroken.so.1").write_text("not an object file")
        block = env_mod._capture_library_linkage()
        assert block["census"]["unreadable"] == 1
        assert block["census"]["rpath"] == 0
        assert block["rocm_rpath"] is None
        assert block["status"] == "unreadable"

    def test_an_unreadable_lib_dir_is_not_an_empty_one(
        self, tmp_path, monkeypatch
    ):
        """A directory we cannot list is not a directory with no RPATH in it.

        The census would otherwise scan zero files and hand back the same
        shape a genuinely clean install produces, which is the "unreadable
        read as empty" confusion the per-file parse already refuses.
        """
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        _write_elf(lib_dir / "libamdhip64.so.7", (DT_RPATH,))
        lib_dir.chmod(0o000)
        for name in (
            "ROCM_CORE_LIB_DIR",
            "HIPBLASLT_LIB_DIR",
            "ROCBLAS_LIB_DIR",
            "MIOPEN_LIB_DIR",
            "RCCL_LIB_DIR",
        ):
            monkeypatch.setattr(env_mod, name, lib_dir)
        monkeypatch.setattr(env_mod, "_torch_native_lib_dir", lambda _mod: None)
        monkeypatch.setattr(
            env_mod, "_safe_import_torch", lambda _reasons, _name: None
        )
        try:
            block = env_mod._capture_library_linkage()
        finally:
            lib_dir.chmod(0o755)

        assert block["census"]["unlistable_dirs"] == [str(lib_dir)]
        assert block["status"] == "unreadable"
        assert block["rocm_rpath"] is None

    def test_a_missing_lib_dir_is_absent_not_unreadable(self, tmp_path, monkeypatch):
        """The other half of the same distinction: nothing here to fail to read."""
        missing = tmp_path / "nowhere"
        for name in (
            "ROCM_CORE_LIB_DIR",
            "HIPBLASLT_LIB_DIR",
            "ROCBLAS_LIB_DIR",
            "MIOPEN_LIB_DIR",
            "RCCL_LIB_DIR",
        ):
            monkeypatch.setattr(env_mod, name, missing)
        monkeypatch.setattr(env_mod, "_torch_native_lib_dir", lambda _mod: None)
        monkeypatch.setattr(
            env_mod, "_safe_import_torch", lambda _reasons, _name: None
        )
        block = env_mod._capture_library_linkage()
        assert block["census"]["unlistable_dirs"] == []
        assert block["status"] == "absent"

    def test_a_truncated_census_is_recorded_as_such(self, rocm_libs, monkeypatch):
        """A capped scan can only support a positive answer, so say when it capped."""
        monkeypatch.setattr(env_mod, "_MAX_CENSUS_FILES", 2)
        for index in range(5):
            _write_elf(rocm_libs / f"lib{index}.so.1", (DT_RUNPATH,))
        census = env_mod._capture_library_linkage()["census"]
        assert census["truncated"] is True
        assert census["scanned"] == 2

    def test_both_wheel_layout_component_dirs_are_scanned(
        self, tmp_path, monkeypatch
    ):
        """The wheel layout splits ROCm across two site-packages components.

        ``_rocm_sdk_core`` holds the HIP runtime and ``_rocm_sdk_libraries``
        holds hipBLAS(Lt) / rocBLAS / MIOpen / RCCL, so a census that read
        only one of them would miss half the install -- including, in the
        measured image, ``libhipblas.so.3``.
        """
        core = tmp_path / "_rocm_sdk_core" / "lib"
        libs = tmp_path / "_rocm_sdk_libraries" / "lib"
        core.mkdir(parents=True)
        libs.mkdir(parents=True)
        _write_elf(core / "libamdhip64.so.7", (DT_RUNPATH,))
        _write_elf(libs / "libhipblas.so.3", (DT_RPATH,))
        monkeypatch.setattr(env_mod, "ROCM_CORE_LIB_DIR", core)
        for name in (
            "HIPBLASLT_LIB_DIR",
            "ROCBLAS_LIB_DIR",
            "MIOPEN_LIB_DIR",
            "RCCL_LIB_DIR",
        ):
            monkeypatch.setattr(env_mod, name, libs)
        monkeypatch.setattr(env_mod, "_torch_native_lib_dir", lambda _mod: None)
        monkeypatch.setattr(
            env_mod, "_safe_import_torch", lambda _reasons, _name: None
        )

        block = env_mod._capture_library_linkage()
        assert len(block["census"]["dirs"]) == 2
        assert block["census"]["scanned"] == 2
        assert block["rocm_rpath"] is True


class TestLibraryLinkageSchemaSurface:
    """Schema 1.17 wiring: ordering, back-fill, disaster path, brief."""

    def test_emitted_next_to_the_rocm_block(self):
        keys = list(_example_snapshot().to_dict())
        assert keys[keys.index("therock") + 1] == "library_linkage"

    def test_pre_117_snapshot_backfills_as_unknown_not_absent(self):
        """A 1.16 producer never looked at any dynamic section.

        ``absent`` would assert "this host had no readable ROCm libraries"
        on behalf of a capture that made no such claim -- the same trap
        ``therock`` fell into.
        """
        d = _example_snapshot().to_dict()
        del d["library_linkage"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.library_linkage == env_mod._null_library_linkage()
        assert rebuilt.library_linkage["status"] == "unknown"
        assert rebuilt.library_linkage["rocm_rpath"] is None

    def test_short_block_is_merged_over_the_null_shape(self):
        """A 1.17 consumer can index the documented keys on any artifact."""
        d = _example_snapshot().to_dict()
        d["library_linkage"] = {"status": "ok"}
        rebuilt = EnvSnapshot.from_dict(d)
        assert set(rebuilt.library_linkage) == {
            "status",
            "rocm_rpath",
            "tags_observed",
            "census",
            "libraries",
        }
        assert rebuilt.library_linkage["rocm_rpath"] is None

    def test_census_null_shape_matches_the_emitted_census(self):
        """The assertion above stops at the top level; ``census`` needs its own.

        ``from_dict`` merges the null shape one level deep, so a key added
        inside ``census`` is back-filled by nothing: a consumer indexing it on a
        short artifact gets ``KeyError`` while the top-level key set still
        matches. Not reachable today -- ``census`` shipped in the same 1.17 that
        introduced the block, so no artifact predates it -- but the next key
        added inside it is free to catch here.
        """
        emitted = env_mod._capture_library_linkage()["census"]
        backfilled = env_mod._empty_library_linkage()["census"]
        assert set(backfilled) == set(emitted)

    def test_disaster_snapshot_carries_the_shape_claiming_nothing(self):
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="boom"
        )
        assert snap.library_linkage["status"] == "unknown"
        assert snap.library_linkage["rocm_rpath"] is None

    def test_summary_renders_the_tristate_not_a_bare_none(self):
        """``rocm_rpath=None`` must not print as ``None`` in the brief.

        A reader cannot tell "read them, none carried RPATH" from "could
        not read them" out of a bare ``None``.
        """
        snap = _example_snapshot(
            library_linkage=env_mod._empty_library_linkage()
        )
        line = next(
            ln for ln in snap.summary().splitlines()
            if ln.lstrip().startswith("linkage:")
        )
        assert "rocm_rpath=unknown" in line

    def test_summary_names_the_hazard_when_present(self):
        line = next(
            ln for ln in _example_snapshot().summary().splitlines()
            if ln.lstrip().startswith("linkage:")
        )
        assert "rocm_rpath=yes" in line
        assert "rpath,runpath" in line
