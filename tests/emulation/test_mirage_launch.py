"""Tests for the mirage emulation launch backend + the Environment axis.

Covers, without needing mirage/rocjitsu/torch installed:
* `Environment` carries `emulator` / `mirage_profile` and round-trips via asdict
  (the shape the dispatcher threads into `_aorta_environment`).
* The built-in `emulated-rocjitsu` environment resolves with the new keys.
* JSON sidecars accept the new keys.
* The launch backend detects emulated environments and builds the correct
  `mirage run --profile … -- <argv>` wrapping, and leaves non-emulated argv
  untouched.
"""

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from aorta.emulation.mirage_launch import (
    CONFIG_KEY_ENVIRONMENT,
    ENV_MIRAGE_BIN,
    EmulationError,
    is_emulated_environment,
    resolve_emulation,
    wrap_argv_for_environment,
)
from aorta.registry.environments import get_environment
from aorta.registry.sidecar import load_sidecar_environments
from aorta.registry.types import Environment


def _emulated_config(profile="cdna4-2gpu", emulator="rocjitsu"):
    """A trial config carrying an emulated environment descriptor."""
    env = Environment(
        name="emu", emulator=emulator, mirage_profile=profile
    )
    return {CONFIG_KEY_ENVIRONMENT: asdict(env)}


@pytest.fixture
def mirage_on_path(tmp_path, monkeypatch):
    """Put a fake executable `mirage` on $PATH so resolution succeeds."""
    fake = tmp_path / "mirage"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(ENV_MIRAGE_BIN, raising=False)
    return fake


class TestEnvironmentSchema:
    def test_fields_default_none(self):
        env = Environment(name="local")
        assert env.emulator is None
        assert env.mirage_profile is None

    def test_asdict_includes_new_keys(self):
        env = Environment(name="emu", emulator="rocjitsu", mirage_profile="cdna4-2gpu")
        d = asdict(env)
        assert d["emulator"] == "rocjitsu"
        assert d["mirage_profile"] == "cdna4-2gpu"

    def test_builtin_emulated_environment(self):
        env = get_environment("emulated-rocjitsu")
        assert env.emulator == "rocjitsu"
        assert env.mirage_profile == "mi350x"

    def test_builtin_emulated_dbt_environment(self):
        env = get_environment("emulated-rocjitsu-dbt")
        assert env.emulator == "rocjitsu-dbt"
        assert env.mirage_profile == "dbt-mi350x"


class TestSidecar:
    def test_sidecar_accepts_emulator_keys(self, tmp_path: Path):
        sidecar = tmp_path / "envs.json"
        sidecar.write_text(
            '{"version": 1, "environments": '
            '{"my-emu": {"emulator": "rocjitsu", "mirage_profile": "cdna3"}}}'
        )
        envs = load_sidecar_environments(sidecar)
        assert envs["my-emu"].emulator == "rocjitsu"
        assert envs["my-emu"].mirage_profile == "cdna3"


class TestDetection:
    def test_emulated_by_profile(self):
        assert is_emulated_environment(_emulated_config()) is True

    def test_emulated_by_emulator_only(self):
        cfg = {CONFIG_KEY_ENVIRONMENT: {"emulator": "rocjitsu"}}
        assert is_emulated_environment(cfg) is True

    def test_noop_is_not_emulated(self):
        cfg = {CONFIG_KEY_ENVIRONMENT: {"emulator": "noop"}}
        assert is_emulated_environment(cfg) is False

    def test_local_is_not_emulated(self):
        assert is_emulated_environment({CONFIG_KEY_ENVIRONMENT: {}}) is False

    def test_missing_descriptor_is_not_emulated(self):
        assert is_emulated_environment({}) is False


class TestWrapArgv:
    def test_wraps_with_mirage_run(self, mirage_on_path):
        argv = wrap_argv_for_environment(
            _emulated_config(), ["torchrun", "--nproc_per_node=2", "script.py"]
        )
        assert argv[0] == str(mirage_on_path)
        assert argv[1:4] == ["run", "--profile", "cdna4-2gpu"]
        assert "--" in argv
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["torchrun", "--nproc_per_node=2", "script.py"]

    def test_passthrough_when_not_emulated(self):
        inner = ["python", "-c", "print(1)"]
        assert wrap_argv_for_environment({CONFIG_KEY_ENVIRONMENT: {}}, inner) == inner

    def test_workdir_and_session_flags(self, mirage_on_path):
        spec = resolve_emulation(
            _emulated_config(),
            ["python", "x.py"],
            workdir="/work",
            reuse_session="s-123",
            keep_session=True,
        )
        assert spec is not None
        assert "--workdir" in spec.argv and "/work" in spec.argv
        assert "--session" in spec.argv and "s-123" in spec.argv
        assert "--keep-session" in spec.argv

    def test_emulator_without_profile_errors(self, mirage_on_path):
        cfg = {CONFIG_KEY_ENVIRONMENT: {"emulator": "rocjitsu"}}
        with pytest.raises(EmulationError, match="requires 'mirage_profile'"):
            wrap_argv_for_environment(cfg, ["python", "x.py"])

    def test_empty_argv_errors(self, mirage_on_path):
        with pytest.raises(EmulationError, match="empty argv"):
            resolve_emulation(_emulated_config(), [])

    def test_missing_mirage_binary_errors(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, no mirage
        monkeypatch.delenv(ENV_MIRAGE_BIN, raising=False)
        with pytest.raises(EmulationError, match="mirage CLI not found"):
            wrap_argv_for_environment(_emulated_config(), ["python", "x.py"])

    def test_mirage_bin_env_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "my-mirage"
        custom.write_text("#!/bin/sh\nexit 0\n")
        custom.chmod(0o755)
        monkeypatch.setenv(ENV_MIRAGE_BIN, str(custom))
        argv = wrap_argv_for_environment(_emulated_config(), ["python", "x.py"])
        assert argv[0] == str(custom)

    def test_mirage_bin_expands_tilde(self, monkeypatch, tmp_path):
        custom = tmp_path / "my-mirage"
        custom.write_text("#!/bin/sh\nexit 0\n")
        custom.chmod(0o755)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(ENV_MIRAGE_BIN, "~/my-mirage")
        argv = wrap_argv_for_environment(_emulated_config(), ["python", "x.py"])
        assert argv[0] == str(custom)

    def test_mirage_bin_relative_path(self, monkeypatch, tmp_path):
        # A relative path with a directory component (as used by the repo's
        # docs/scripts, e.g. ./build/manylinux/bin/mirage) must resolve to the
        # absolute executable rather than falling through to a $PATH lookup.
        bindir = tmp_path / "build" / "bin"
        bindir.mkdir(parents=True)
        custom = bindir / "mirage"
        custom.write_text("#!/bin/sh\nexit 0\n")
        custom.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(ENV_MIRAGE_BIN, "./build/bin/mirage")
        argv = wrap_argv_for_environment(_emulated_config(), ["python", "x.py"])
        assert argv[0] == str(custom)

    def test_mirage_bin_relative_path_not_executable_errors(
        self, monkeypatch, tmp_path
    ):
        # A path-shaped override that doesn't point to an executable must fail
        # loudly (not silently fall back to $PATH).
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(ENV_MIRAGE_BIN, "./does/not/exist/mirage")
        with pytest.raises(EmulationError, match="does not point to an executable"):
            wrap_argv_for_environment(_emulated_config(), ["python", "x.py"])

    def test_mirage_bin_directory_rejected(self, monkeypatch, tmp_path):
        # A directory typically carries the +x (traversal) bit; it must not be
        # accepted just because os.access(..., X_OK) is true -- require a file.
        d = tmp_path / "mirage"  # a directory named like the binary
        d.mkdir()
        assert os.access(d, os.X_OK)  # sanity: dir is "executable"
        monkeypatch.setenv(ENV_MIRAGE_BIN, str(d))
        with pytest.raises(EmulationError, match="does not point to an executable"):
            wrap_argv_for_environment(_emulated_config(), ["python", "x.py"])


class TestSubprocessWorkloadEmulation:
    """The aorta probe SubprocessWorkload wraps its argv when emulated."""

    def _make(self, env_descriptor):
        from aorta.workloads._subprocess import (
            CONFIG_KEY_LOG_PREFIX,
            CONFIG_KEY_SUBPROCESS_ARGV,
            SubprocessWorkload,
        )

        return SubprocessWorkload, {
            CONFIG_KEY_SUBPROCESS_ARGV: ["/bin/echo", "hi"],
            CONFIG_KEY_LOG_PREFIX: "results/_subprocess/trial_d0_m0_t0",
            CONFIG_KEY_ENVIRONMENT: env_descriptor,
        }

    def test_argv_wrapped_when_emulated(self, mirage_on_path, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cls, cfg = self._make(
            {"name": "emu", "emulator": "rocjitsu", "mirage_profile": "cdna4-2gpu"}
        )
        wl = cls(cfg)
        wl.setup()
        assert wl._argv[0] == str(mirage_on_path)
        assert "--profile" in wl._argv and "cdna4-2gpu" in wl._argv
        assert wl._argv[-2:] == ("/bin/echo", "hi")

    def test_argv_unchanged_when_not_emulated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cls, cfg = self._make({"name": "local"})
        wl = cls(cfg)
        wl.setup()
        assert wl._argv == ("/bin/echo", "hi")
