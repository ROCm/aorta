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
import logging
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
            "ROCM_BIN_DIR",
            "ROCM_VERSION_FILE",
            "ROCM_VERSION_DEV_FILE",
            "KMD_VERSION_FILE",
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
            "RCCL_VERSION_HEADER",
            "RCCL_LIB_DIR",
            "SYS_CLASS_NET",
            "SYS_CLASS_INFINIBAND",
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
            "ROCM_BIN_DIR",
            "ROCM_VERSION_FILE",
            "ROCM_VERSION_DEV_FILE",
            "KMD_VERSION_FILE",
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
            "RCCL_VERSION_HEADER",
            "RCCL_LIB_DIR",
            "SYS_CLASS_NET",
            "SYS_CLASS_INFINIBAND",
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
    "rocblas",
    "composable_kernel",
    "tensile",
    "triton",
    "fbgemm",
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
    "library_introspection",
    "library_introspection_alternates",
    "pytorch_sdpa",
    "nics",
}


class TestSchemaCompleteness:
    def test_all_top_level_keys_present_when_everything_unavailable(
        self, all_disabled
    ):
        snapshot = collect_env()
        assert set(snapshot.to_dict().keys()) == REQUIRED_TOP_KEYS
        assert snapshot.schema_version == "1.7"
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
            "rocm_release_tweak": "dabb6df2b9",
            "package_version": "1.2.2",
            "lib_hash": "sha256:abc",
            "kernel_db_revision": "filenames-sha256:def",
            "applied_prs": {},
        },
        "rocblas": {
            "rocm_release_tweak": "dabb6df2b9",
            "package_version": "5.2.0",
            "lib_hash": "sha256:bbb",
            "kernel_db_revision": "filenames-sha256:ccc",
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
        "triton": {"package_version": "3.5.1+rocm7.2.1.gita272dfa8"},
        "fbgemm": {
            "package_version": None,
            "pytorch_use_fbgemm": True,
            "pytorch_use_fbgemm_genai": True,
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
        assert rebuilt.schema_version == SCHEMA_VERSION

    def test_from_dict_defaults_partial_reasons_when_missing(self):
        """Older env.json without partial_reasons still loads (defaults to [])."""
        d = _example_snapshot().to_dict()
        del d["partial_reasons"]
        rebuilt = EnvSnapshot.from_dict(d)
        assert rebuilt.partial_reasons == []

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

    def test_disaster_snapshot_nics_block_is_fully_shaped(self):
        """The disaster path shapes nics like every other block (all
        vendor keys present, undeterminable presence) -- not an empty
        dict -- so downstream diffs/parsers stay predictable."""
        snap = env_mod._disaster_snapshot(
            preceding_reasons=[], unexpected_reason="collect_env: boom"
        )
        assert set(snap.nics.keys()) == {"ainic", "broadcom", "cx7"}
        assert all(snap.nics[v] == {"present": None} for v in snap.nics)

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
        #
        # The real "no-probing-in-CLI" guard is
        # `test_cli_does_no_probing_imports` below -- this one is a
        # soft canary against the file ballooning beyond pure wiring.
        line_count = sum(1 for _ in cli_path.read_text().splitlines())
        assert line_count <= 350, (
            f"cli/env.py is {line_count} lines; soft budget is 350. "
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
            "rocm_release_tweak",
            "package_version",
            "lib_hash",
            "kernel_db_revision",
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
            # GPU scoping
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            # HSA / runtime
            "HSA_XNACK",
            "HSA_KERNARG_POOL_SIZE",
            "HSA_NO_SCRATCH_RECLAIM",
            "HSA_OVERRIDE_GFX_VERSION",
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
        assert block == {"package_version": None}
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
        assert block == {"package_version": "3.5.1+rocm7.2.1.gita272dfa8"}
        assert reasons == []


# ---------------------------------------------------------------------------
# FBGEMM
# ---------------------------------------------------------------------------


class TestFbgemmBlock:
    def test_block_keys_stable(self, all_disabled):
        reasons: list[str] = []
        block = env_mod._capture_fbgemm(reasons)
        assert set(block.keys()) == {
            "package_version",
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
