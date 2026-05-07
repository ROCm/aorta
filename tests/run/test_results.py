"""Tests for TrialResult dataclass."""

import pytest

from aorta.run.results import TrialResult


class TestTrialResult:
    """Tests for TrialResult serialization and deserialization."""

    def test_trial_result_roundtrip(self):
        """TrialResult serializes/deserializes losslessly.

        ``execution_env`` mirrors :class:`aorta.registry.Environment`
        (``name``/``docker``/``venv``/``source_package``); ROCm
        version, runtime kind, and image digest live inside ``env``
        (A1's ``EnvSnapshot``) -- see the ``TrialResult`` docstring.
        """
        result = TrialResult(
            trial_id="fsdp_d0_m0_t0",
            workload="fsdp",
            execution_env={
                "name": "local",
                "docker": None,
                "venv": None,
                "source_package": "aorta",
            },
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
        """TrialResult handles complex nested data.

        Schema discipline: the static descriptor (``execution_env``)
        carries only the registry-level recipe
        (``name``/``docker``/``venv``/``source_package``).  Runtime
        observations -- ROCm version, image digest, hostname,
        env_vars -- live in ``env`` (the A1 ``EnvSnapshot`` dict).
        Mixing them was a stub-ism that round 3 cleaned up in the
        production code; this fixture now reflects the same split.
        """
        result = TrialResult(
            trial_id="custom_workload_d2_m4_t6",
            workload="custom_workload",
            execution_env={
                "name": "ci_env",
                "docker": "aorta:latest",
                "venv": "/opt/venv",
                "source_package": "aorta-internal",
            },
            mitigations_applied=("tf32_off", "custom_mitigation"),
            config={"steps": 100, "batch_size": 32, "nested": {"key": "value"}},
            env={
                "schema_version": "1.1",
                "python_version": "3.10.0",
                "pytorch_version": "2.0.0",
                "rocm": {"version": "6.0.0"},
                "docker": {"digest": "sha256:abc123"},
                "env_vars": {"ROCM_PATH": "/opt/rocm"},
                "partial": False,
                "partial_reasons": [],
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
        """All valid exit_status values are accepted.

        ``"timeout"`` is deliberately NOT in the literal: B1 ships no
        ``--timeout`` flag and no watchdog so no code path can produce
        it.  Re-add it (and this test entry) in the same commit that
        adds a producer.
        """
        for status in ["ok", "workload_failed", "infrastructure_failed"]:
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
            execution_env={"name": "local"},
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
            execution_env={"name": "local"},
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
