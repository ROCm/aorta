"""Tests for ``aorta.instrumentation.environment`` (issue #147 acceptance).

Strategy: load the module by file path so the test does not pull in the
torch-dependent ``aorta.utils`` package. Every subprocess and filesystem
touchpoint is monkeypatched, so tests run on any host without ROCm,
RDHC, hipconfig, hipblaslt, or torch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


capture_environment = env_mod.capture_environment
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

    Result: ``capture_environment`` exercises only pure-Python paths and
    every block returns its null-shaped form.
    """
    monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "no_rocm")
    monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "no_rocm_dev")
    monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "no_kmd")
    monkeypatch.setattr(
        env_mod, "HIPBLASLT_VERSION_HEADER", tmp_path / "no_header.h"
    )
    monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", tmp_path / "no_libs")
    monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tmp_path / "no_tensile")
    monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
    monkeypatch.setattr(
        env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv"
    )
    monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")
    monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
    return monkeypatch


# ---------------------------------------------------------------------------
# Schema completeness + versioning
# ---------------------------------------------------------------------------


REQUIRED_TOP_KEYS = {
    "schema_version",
    "captured_at",
    "system_health",
    "rocm",
    "hip",
    "hipblaslt",
    "runtime_context",
    "docker",
    "env_vars",
    "python_version",
    "pytorch_version",
}


class TestSchemaCompleteness:
    def test_all_top_level_keys_present_when_everything_unavailable(
        self, all_disabled, tmp_path: Path
    ):
        out = tmp_path / "env.json"
        snapshot = capture_environment(out)
        # All keys must be present regardless of availability
        assert set(snapshot.keys()) == REQUIRED_TOP_KEYS
        # The blocks themselves either have content or are explicitly null
        assert snapshot["schema_version"] == "1.0"
        assert snapshot["system_health"] is None
        assert snapshot["rocm"] == {
            "version": None,
            "version_dev": None,
            "kmd_version": None,
        }
        assert snapshot["hip"] == {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

    def test_snapshot_is_written_to_disk(self, all_disabled, tmp_path: Path):
        out = tmp_path / "env.json"
        snapshot = capture_environment(out)
        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert on_disk == snapshot

    def test_schema_version_constant_is_emitted(self, all_disabled, tmp_path: Path):
        snapshot = capture_environment(tmp_path / "env.json")
        assert snapshot["schema_version"] == SCHEMA_VERSION

    def test_captured_at_is_iso8601_utc(self, all_disabled, tmp_path: Path):
        snapshot = capture_environment(tmp_path / "env.json")
        ts = snapshot["captured_at"]
        # Issue's example uses trailing Z; validate shape rather than exact value
        assert ts.endswith("Z")
        assert "T" in ts


# ---------------------------------------------------------------------------
# RDHC wrapper
# ---------------------------------------------------------------------------


class TestRdhcWrapper:
    def test_rdhc_unavailable_returns_none(self, all_disabled, tmp_path: Path):
        # all_disabled already stubs shutil.which to return None
        result = env_mod._run_rdhc(tmp_path / "rdhc_out.json")
        assert result is None

    def test_rdhc_present_but_sudo_n_fails_returns_none(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            # sudo -n returns non-zero when password would be required
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="sudo: a password is required"
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        assert env_mod._run_rdhc(tmp_path / "rdhc_out.json") is None

    def test_rdhc_timeout_returns_none(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        assert env_mod._run_rdhc(tmp_path / "rdhc_out.json") is None

    def test_rdhc_happy_path_returns_parsed_json(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")
        out = tmp_path / "rdhc_out.json"

        rdhc_payload = {
            "rdhc_version": "1.4.0",
            "tests": {"gpu_present": "PASS"},
            "general_info": {"hostname": "test-host"},
            "gpu_info": [{"name": "MI300X"}],
            "firmware": [],
        }

        def fake_run(cmd, **kwargs):
            assert cmd[0] == "sudo"
            assert "-n" in cmd
            assert "--quick" in cmd
            assert "--json" in cmd
            out.write_text(json.dumps(rdhc_payload))
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        result = env_mod._run_rdhc(out)
        assert result == rdhc_payload

    def test_rdhc_malformed_json_returns_none(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/rdhc")
        out = tmp_path / "rdhc_out.json"

        def fake_run(cmd, **kwargs):
            out.write_text("not valid json {{{")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
        assert env_mod._run_rdhc(out) is None


# ---------------------------------------------------------------------------
# ROCm version files
# ---------------------------------------------------------------------------


class TestRocmVersionFiles:
    def test_all_present(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "version"
        v.write_text("7.2.1\n")
        vdev = tmp_path / "version-dev"
        vdev.write_text("7.2.1.50311-abc1234\n")
        kmd = tmp_path / "kmd_version"
        kmd.write_text("6.16.13\n")

        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", v)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", vdev)
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", kmd)

        result = env_mod._capture_rocm_version_files()
        assert result == {
            "version": "7.2.1",
            "version_dev": "7.2.1.50311-abc1234",
            "kmd_version": "6.16.13",
        }

    def test_all_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "nope1")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "nope2")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "nope3")
        result = env_mod._capture_rocm_version_files()
        assert result == {"version": None, "version_dev": None, "kmd_version": None}

    def test_partial_missing(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "version"
        v.write_text("7.2.1\n")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", v)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", tmp_path / "nope")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "also_nope")
        result = env_mod._capture_rocm_version_files()
        assert result == {
            "version": "7.2.1",
            "version_dev": None,
            "kmd_version": None,
        }

    def test_empty_file_treated_as_none(self, tmp_path: Path, monkeypatch):
        # /opt/rocm/.info/version-dev is sometimes installed but empty
        empty = tmp_path / "version-dev"
        empty.write_text("")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", empty)
        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", tmp_path / "nope")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", tmp_path / "nope")
        result = env_mod._capture_rocm_version_files()
        assert result["version_dev"] is None


# ---------------------------------------------------------------------------
# HIP toolchain
# ---------------------------------------------------------------------------


class TestHipToolchain:
    def test_hipconfig_missing_returns_all_none(self, isolated_env, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        result = env_mod._capture_hip_toolchain()
        assert result == {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

    def test_hipconfig_happy_path(self, isolated_env, monkeypatch):
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
        result = env_mod._capture_hip_toolchain()
        assert result == {
            "version": "7.2.53211-e1a6bc5663",
            "platform": "amd",
            "compiler": "clang",
            "runtime": "rocclr",
            "cpp_config": "-D__HIP_PLATFORM_AMD__",
        }

    def test_hipconfig_one_field_fails(self, isolated_env, monkeypatch):
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
        result = env_mod._capture_hip_toolchain()
        assert result["version"] == "ok"
        assert result["cpp_config"] is None


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

        # Add another file -> fingerprint changes
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
        block = env_mod._capture_hipblaslt()
        assert block["applied_prs"] == {}

    def test_block_keys_stable(self, all_disabled):
        block = env_mod._capture_hipblaslt()
        assert set(block.keys()) == {
            "commit",
            "package_version",
            "lib_hash",
            "tensile_yaml_revision",
            "applied_prs",
        }


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

    def test_runtime_context_venv_path_populated(
        self, all_disabled, monkeypatch
    ):
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
    def test_baremetal_returns_none(self):
        rt = {"type": "baremetal", "python_env": "system"}
        assert env_mod._capture_docker_metadata(rt) is None

    def test_docker_picks_up_aorta_env_vars(self, isolated_env):
        isolated_env.setenv("AORTA_DOCKER_IMAGE", "rocm/pytorch:7.2")
        isolated_env.setenv("AORTA_DOCKER_DIGEST", "sha256:deadbeef")
        rt = {"type": "docker"}
        block = env_mod._capture_docker_metadata(rt)
        assert block["image"] == "rocm/pytorch:7.2"
        assert block["digest"] == "sha256:deadbeef"

    def test_docker_block_emits_keys_even_without_env(self, isolated_env):
        rt = {"type": "docker"}
        block = env_mod._capture_docker_metadata(rt)
        assert set(block.keys()) == {"image", "digest", "container_id"}
        assert block["image"] is None
        assert block["digest"] is None

    def test_container_id_extracted_from_cgroup(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        cgroup = tmp_path / "cgroup"
        cid = "abc123def456789012345678901234567890abcd"
        cgroup.write_text(f"12:freezer:/docker/{cid}\n")
        # _read_container_id() reads /proc/self/cgroup, not /proc/1/cgroup;
        # patch the Path constructor it uses inside.
        monkeypatch.setattr(env_mod, "_read_text_file", lambda p: cgroup.read_text())
        rt = {"type": "docker"}
        block = env_mod._capture_docker_metadata(rt)
        assert block["container_id"] == cid


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

    def test_workload_config_vars_NOT_captured(self, isolated_env):
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
            "HSA_XNACK",
            "HSA_KERNARG_POOL_SIZE",
            "HSA_NO_SCRATCH_RECLAIM",
            "GPU_MAX_HW_QUEUES",
            "AMDGCN_USE_BUFFER_OPS",
            "DISABLE_TF32",
            "NCCL_MAX_NCHANNELS",
            "FBGEMM_NO_JK",
            "FBGEMM_TBE_V2",
            "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL",
            "FBGEMM_BOUNDS_CHECK_INDICES_V2",
            "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
            "PYTORCH_CUDA_ALLOC_CONF",
        }


# ---------------------------------------------------------------------------
# PyTorch version + 'no GPU compute' guard
# ---------------------------------------------------------------------------


class TestPytorchVersion:
    def test_torch_unavailable_returns_none(self, isolated_env):
        # Force an ImportError at probe-time even if torch happens to be
        # installed in the test env, by monkeypatching builtins.__import__.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            assert env_mod._capture_pytorch_version() is None


class TestNoGpuCompute:
    """Guard against introducing GPU work into the env probe.

    True GPU-zero verification is via rocprofv3 in CI; here we assert
    that the orchestrator never reaches into ``torch.cuda`` (which would
    initialise a HIP context).
    """

    def test_torch_cuda_never_called(self, all_disabled, tmp_path: Path):
        if "torch" not in sys.modules:
            pytest.skip("torch not available; trivially passes")
        torch = sys.modules["torch"]
        with patch.object(torch.cuda, "is_available") as is_avail, patch.object(
            torch.cuda, "device_count"
        ) as dev_count:
            capture_environment(tmp_path / "env.json")
            is_avail.assert_not_called()
            dev_count.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


class TestCaptureEnvironmentEndToEnd:
    def test_full_snapshot_with_realistic_files(
        self, isolated_env, tmp_path: Path, monkeypatch
    ):
        # Stand up a realistic-looking ROCm root in tmp.
        rocm = tmp_path / "rocm"
        info = rocm / ".info"
        info.mkdir(parents=True)
        (info / "version").write_text("7.2.1\n")
        (info / "version-dev").write_text("7.2.1.50311-abc1234\n")

        kmd = tmp_path / "amdgpu_version"
        kmd.write_text("6.16.13\n")

        hipblaslt_inc = rocm / "include" / "hipblaslt"
        hipblaslt_inc.mkdir(parents=True)
        (hipblaslt_inc / "hipblaslt-version.h").write_text(
            "#define HIPBLASLT_VERSION_MAJOR 1\n"
            "#define HIPBLASLT_VERSION_MINOR 2\n"
            "#define HIPBLASLT_VERSION_PATCH 2\n"
            "#define HIPBLASLT_VERSION_TWEAK dabb6df2b9\n"
        )

        lib = rocm / "lib"
        lib.mkdir()
        (lib / "libhipblaslt.so").write_bytes(b"fake binary")

        tensile = rocm / "lib" / "hipblaslt" / "library"
        tensile.mkdir(parents=True)
        (tensile / "TensileLibrary_X.dat").write_bytes(b"x")
        (tensile / "TensileLibrary_Y.dat").write_bytes(b"y")

        monkeypatch.setattr(env_mod, "ROCM_VERSION_FILE", info / "version")
        monkeypatch.setattr(env_mod, "ROCM_VERSION_DEV_FILE", info / "version-dev")
        monkeypatch.setattr(env_mod, "KMD_VERSION_FILE", kmd)
        monkeypatch.setattr(
            env_mod, "HIPBLASLT_VERSION_HEADER", hipblaslt_inc / "hipblaslt-version.h"
        )
        monkeypatch.setattr(env_mod, "HIPBLASLT_LIB_DIR", lib)
        monkeypatch.setattr(env_mod, "HIPBLASLT_TENSILE_DIR", tensile)
        monkeypatch.setattr(env_mod, "DOCKERENV_MARKER", tmp_path / "no_dockerenv")
        monkeypatch.setattr(
            env_mod, "PODMAN_CONTAINERENV_MARKER", tmp_path / "no_podmanenv"
        )
        monkeypatch.setattr(env_mod, "CGROUP_FILE", tmp_path / "no_cgroup")
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)

        out = tmp_path / "env.json"
        snapshot = capture_environment(out)

        # Schema completeness
        assert set(snapshot.keys()) == REQUIRED_TOP_KEYS

        # ROCm
        assert snapshot["rocm"]["version"] == "7.2.1"
        assert snapshot["rocm"]["version_dev"] == "7.2.1.50311-abc1234"
        assert snapshot["rocm"]["kmd_version"] == "6.16.13"

        # HIP (no hipconfig available -> all None)
        assert snapshot["hip"] == {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

        # hipBLASLt
        assert snapshot["hipblaslt"]["commit"] == "dabb6df2b9"
        assert snapshot["hipblaslt"]["package_version"] == "1.2.2"
        assert snapshot["hipblaslt"]["lib_hash"] == (
            "sha256:" + hashlib.sha256(b"fake binary").hexdigest()
        )
        assert snapshot["hipblaslt"]["tensile_yaml_revision"].startswith(
            "filenames-sha256:"
        )
        assert snapshot["hipblaslt"]["applied_prs"] == {}

        # Runtime
        assert snapshot["runtime_context"]["type"] == "baremetal"
        assert snapshot["docker"] is None  # baremetal -> docker is null

        # RDHC unavailable
        assert snapshot["system_health"] is None

        # File on disk matches return value
        assert json.loads(out.read_text()) == snapshot
