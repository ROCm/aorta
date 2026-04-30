"""Tests for temporary stub implementations."""

import os
import pytest
from unittest.mock import patch

from aorta.run._stubs import (
    EnvSnapshot,
    collect_env,
    Environment,
    Mitigation,
    get_environment,
    get_mitigation,
)


class TestEnvSnapshot:
    """Tests for EnvSnapshot stub."""

    def test_to_dict(self):
        """EnvSnapshot converts to dict correctly."""
        snapshot = EnvSnapshot(
            hostname="testhost",
            python_version="3.10.0",
            pytorch_version="2.0.0",
            rocm_version="6.0.0",
            env_vars={"KEY": "value"},
        )

        data = snapshot.to_dict()

        assert data["hostname"] == "testhost"
        assert data["python_version"] == "3.10.0"
        assert data["pytorch_version"] == "2.0.0"
        assert data["rocm_version"] == "6.0.0"
        assert data["env_vars"] == {"KEY": "value"}

    def test_to_dict_with_none_values(self):
        """EnvSnapshot handles None values."""
        snapshot = EnvSnapshot(
            hostname="testhost",
            python_version="3.10.0",
            pytorch_version=None,
            rocm_version=None,
            env_vars={},
        )

        data = snapshot.to_dict()

        assert data["pytorch_version"] is None
        assert data["rocm_version"] is None


class TestCollectEnv:
    """Tests for collect_env stub function."""

    def test_returns_env_snapshot(self):
        """collect_env returns an EnvSnapshot."""
        snapshot = collect_env()
        assert isinstance(snapshot, EnvSnapshot)

    def test_captures_hostname(self):
        """collect_env captures hostname."""
        snapshot = collect_env()
        assert snapshot.hostname is not None
        assert len(snapshot.hostname) > 0

    def test_captures_python_version(self):
        """collect_env captures Python version."""
        snapshot = collect_env()
        assert snapshot.python_version is not None
        assert "3." in snapshot.python_version  # Python 3.x

    def test_captures_relevant_env_vars(self):
        """collect_env captures relevant environment variables."""
        with patch.dict(os.environ, {
            "ROCM_PATH": "/opt/rocm",
            "WORLD_SIZE": "4",
            "IRRELEVANT_VAR": "ignored",
        }, clear=False):
            snapshot = collect_env()

        assert "ROCM_PATH" in snapshot.env_vars
        assert "WORLD_SIZE" in snapshot.env_vars
        # Irrelevant vars should not be captured
        assert "IRRELEVANT_VAR" not in snapshot.env_vars


class TestEnvironment:
    """Tests for Environment stub."""

    def test_default_values(self):
        """Environment has correct defaults."""
        env = Environment(name="test")
        assert env.name == "test"
        assert env.kind == "local"
        assert env.docker is None
        assert env.venv is None
        assert env.rocm is None
        assert env.source_package == "aorta"

    def test_custom_values(self):
        """Environment accepts custom values."""
        env = Environment(
            name="ci",
            kind="docker",
            docker="aorta:latest",
            venv="/opt/venv",
            rocm="6.0.0",
            source_package="aorta-internal",
        )
        assert env.name == "ci"
        assert env.kind == "docker"
        assert env.docker == "aorta:latest"


class TestMitigation:
    """Tests for Mitigation stub."""

    def test_default_values(self):
        """Mitigation has correct defaults."""
        mit = Mitigation(name="test")
        assert mit.name == "test"
        assert mit.env_vars == {}

    def test_custom_env_vars(self):
        """Mitigation accepts custom env vars."""
        mit = Mitigation(
            name="tf32_off",
            env_vars={"DISABLE_TF32": "1"},
        )
        assert mit.name == "tf32_off"
        assert mit.env_vars == {"DISABLE_TF32": "1"}


class TestGetEnvironment:
    """Tests for get_environment stub function."""

    def test_local_environment(self):
        """get_environment returns local environment."""
        env = get_environment("local")
        assert env.name == "local"
        assert env.kind == "local"

    def test_unknown_environment_raises(self):
        """get_environment raises for unknown environment."""
        with pytest.raises(ValueError) as exc_info:
            get_environment("nonexistent_env")

        error_msg = str(exc_info.value)
        assert "Unknown environment" in error_msg
        assert "nonexistent_env" in error_msg
        assert "Available" in error_msg

    def test_error_lists_available_environments(self):
        """Error message includes available environments."""
        with pytest.raises(ValueError) as exc_info:
            get_environment("bad")

        assert "local" in str(exc_info.value)


class TestGetMitigation:
    """Tests for get_mitigation stub function."""

    def test_none_mitigation(self):
        """get_mitigation returns 'none' mitigation."""
        mit = get_mitigation("none")
        assert mit.name == "none"
        assert mit.env_vars == {}

    def test_tf32_off_mitigation(self):
        """get_mitigation returns tf32_off mitigation."""
        mit = get_mitigation("tf32_off")
        assert mit.name == "tf32_off"
        assert mit.env_vars == {"DISABLE_TF32": "1"}

    def test_unknown_mitigation_raises(self):
        """get_mitigation raises for unknown mitigation."""
        with pytest.raises(ValueError) as exc_info:
            get_mitigation("nonexistent_mitigation")

        error_msg = str(exc_info.value)
        assert "Unknown mitigation" in error_msg
        assert "nonexistent_mitigation" in error_msg
        assert "Available" in error_msg

    def test_error_lists_available_mitigations(self):
        """Error message includes available mitigations."""
        with pytest.raises(ValueError) as exc_info:
            get_mitigation("bad")

        error_msg = str(exc_info.value)
        assert "none" in error_msg
        assert "tf32_off" in error_msg
