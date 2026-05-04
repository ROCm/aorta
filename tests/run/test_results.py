"""Tests for TrialResult dataclass."""

import pytest

from aorta.run.results import TrialResult


class TestTrialResult:
    """Tests for TrialResult serialization and deserialization."""

    def test_trial_result_roundtrip(self):
        """TrialResult serializes/deserializes losslessly."""
        result = TrialResult(
            trial_id="test_0",
            workload="fsdp",
            execution_env={"kind": "local", "name": "local"},
            mitigations_applied=("none",),
            config={},
            env={},
            result={"passed": True},
            wall_clock_sec=10.5,
            exit_status="ok",
        )
        data = result.to_dict()
        restored = TrialResult.from_dict(data)
        assert restored == result

    def test_trial_result_roundtrip_with_all_fields(self):
        """TrialResult handles complex nested data."""
        result = TrialResult(
            trial_id="complex_t2",
            workload="custom_workload",
            execution_env={
                "kind": "docker",
                "name": "ci_env",
                "image": "aorta:latest",
                "digest": "sha256:abc123",
                "venv": "/opt/venv",
                "rocm": "6.0.0",
                "source_package": "aorta-internal",
            },
            mitigations_applied=("tf32_off", "custom_mitigation"),
            config={"steps": 100, "batch_size": 32, "nested": {"key": "value"}},
            env={
                "hostname": "testhost",
                "python_version": "3.10.0",
                "pytorch_version": "2.0.0",
                "rocm_version": "6.0.0",
                "env_vars": {"ROCM_PATH": "/opt/rocm"},
            },
            result={
                "passed": False,
                "failure_count": 2,
                "failure_details": [{"iter": 50, "error": "NaN detected"}],
            },
            wall_clock_sec=123.456,
            exit_status="workload_failed",
        )
        data = result.to_dict()
        restored = TrialResult.from_dict(data)
        assert restored == result

    def test_to_dict_converts_tuple_to_list(self):
        """Mitigations tuple is converted to list for JSON compatibility."""
        result = TrialResult(
            trial_id="test_0",
            workload="fsdp",
            execution_env={},
            mitigations_applied=("none", "tf32_off"),
            config={},
            env={},
            result={},
            wall_clock_sec=1.0,
            exit_status="ok",
        )
        data = result.to_dict()
        assert data["mitigations_applied"] == ["none", "tf32_off"]
        assert isinstance(data["mitigations_applied"], list)

    def test_from_dict_handles_default_schema_version(self):
        """Missing schema_version defaults to 0.1."""
        data = {
            "trial_id": "test_0",
            "workload": "fsdp",
            "execution_env": {},
            "mitigations_applied": [],
            "config": {},
            "env": {},
            "result": {},
            "wall_clock_sec": 1.0,
            "exit_status": "ok",
        }
        result = TrialResult.from_dict(data)
        assert result.schema_version == "0.1"

    def test_trial_result_is_frozen(self):
        """TrialResult is immutable."""
        from dataclasses import FrozenInstanceError

        result = TrialResult(
            trial_id="test_0",
            workload="fsdp",
            execution_env={},
            mitigations_applied=(),
            config={},
            env={},
            result={},
            wall_clock_sec=1.0,
            exit_status="ok",
        )
        with pytest.raises(FrozenInstanceError):
            result.trial_id = "modified"  # type: ignore[misc]

    def test_exit_status_values(self):
        """All valid exit_status values are accepted."""
        for status in ["ok", "workload_failed", "infrastructure_failed", "timeout"]:
            result = TrialResult(
                trial_id="test",
                workload="test",
                execution_env={},
                mitigations_applied=(),
                config={},
                env={},
                result={},
                wall_clock_sec=0.0,
                exit_status=status,  # type: ignore[arg-type]
            )
            assert result.exit_status == status

    def test_mutable_fields_are_defensively_copied(self):
        """Mutating the dict passed in must not affect the stored value."""
        config = {"steps": 10, "nested": {"k": "v"}}
        env = {"HOST": "h"}
        result = TrialResult(
            trial_id="t",
            workload="w",
            execution_env={"kind": "local"},
            mitigations_applied=(),
            config=config,
            env=env,
            result={"passed": True},
            wall_clock_sec=1.0,
            exit_status="ok",
        )

        # Outer-level mutation
        config["steps"] = 999
        # Nested mutation
        config["nested"]["k"] = "modified"
        env["HOST"] = "mutated"

        assert result.config["steps"] == 10
        assert result.config["nested"]["k"] == "v"
        assert result.env["HOST"] == "h"

    def test_to_dict_returns_independent_copies(self):
        """Mutating to_dict() output must not affect the TrialResult."""
        result = TrialResult(
            trial_id="t",
            workload="w",
            execution_env={"kind": "local"},
            mitigations_applied=(),
            config={"steps": 10},
            env={"HOST": "h"},
            result={"passed": True},
            wall_clock_sec=1.0,
            exit_status="ok",
        )
        data = result.to_dict()
        data["config"]["steps"] = 999
        data["env"]["HOST"] = "mutated"

        assert result.config["steps"] == 10
        assert result.env["HOST"] == "h"
