"""Tests for launch mode validation."""

import os
from unittest.mock import patch

import pytest

from aorta.run.validation import (
    PROCESS_OPTIONAL_POLICY,
    PROCESS_REQUIRED_POLICY,
    resolve_trial_isolation,
    resolve_trial_isolation_policy,
    validate_launch_mode,
)
from aorta.workloads import Workload, WorkloadResult


class SingleProcessWorkload(Workload):
    """Mock single_process workload for testing."""

    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=True)


class DistributedWorkload(Workload):
    """Mock distributed workload for testing."""

    launch_mode = "distributed"
    min_world_size = 2

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=True)


class DistributedWorkload4(Workload):
    """Mock distributed workload requiring 4 ranks."""

    launch_mode = "distributed"
    min_world_size = 4

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=True)


class RequiredProcessWorkload(SingleProcessWorkload):
    trial_isolation_default = "process"
    trial_isolation_required = True
    trial_isolation_supported = frozenset({"process"})


class ProcessCapableWorkload(SingleProcessWorkload):
    trial_isolation_supported = frozenset({"in_process", "process"})


class TestSingleProcessValidation:
    """Tests for single_process workload validation."""

    def test_single_process_without_torchrun_passes(self):
        """single_process workload with WORLD_SIZE=1 is valid."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            # Should not raise
            validate_launch_mode(SingleProcessWorkload)

    def test_single_process_no_world_size_passes(self):
        """single_process workload without WORLD_SIZE (default 1) is valid."""
        env = os.environ.copy()
        env.pop("WORLD_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            # Should not raise
            validate_launch_mode(SingleProcessWorkload)

    def test_single_process_under_torchrun_raises(self):
        """single_process workload with WORLD_SIZE>1 raises."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(SingleProcessWorkload)

            error_msg = str(exc_info.value)
            assert "single_process" in error_msg
            assert "torchrun" in error_msg
            assert "WORLD_SIZE=4" in error_msg

    def test_single_process_world_size_2_raises(self):
        """single_process with WORLD_SIZE=2 also raises."""
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(SingleProcessWorkload)

            assert "WORLD_SIZE=2" in str(exc_info.value)


class TestDistributedValidation:
    """Tests for distributed workload validation."""

    def test_distributed_with_sufficient_ranks_passes(self):
        """distributed workload with WORLD_SIZE >= min_world_size is valid."""
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=False):
            # Should not raise
            validate_launch_mode(DistributedWorkload)

    def test_distributed_with_excess_ranks_passes(self):
        """distributed workload with WORLD_SIZE > min_world_size is valid."""
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False):
            # Should not raise
            validate_launch_mode(DistributedWorkload)

    def test_distributed_without_torchrun_raises(self):
        """distributed workload with WORLD_SIZE < min_world_size raises."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(DistributedWorkload)

            error_msg = str(exc_info.value)
            assert "requires WORLD_SIZE >= 2" in error_msg
            assert "got 1" in error_msg
            assert "torchrun" in error_msg

    def test_distributed_no_world_size_raises(self):
        """distributed workload without WORLD_SIZE (default 1) raises."""
        env = os.environ.copy()
        env.pop("WORLD_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(DistributedWorkload)

            assert "requires WORLD_SIZE >= 2" in str(exc_info.value)

    def test_distributed_min_4_with_2_ranks_raises(self):
        """distributed with min_world_size=4 but only 2 ranks raises."""
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(DistributedWorkload4)

            error_msg = str(exc_info.value)
            assert "requires WORLD_SIZE >= 4" in error_msg
            assert "got 2" in error_msg

    def test_distributed_min_4_with_4_ranks_passes(self):
        """distributed with min_world_size=4 and exactly 4 ranks passes."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False):
            # Should not raise
            validate_launch_mode(DistributedWorkload4)


class TestInvalidWorldSize:
    """Tests for structurally-invalid WORLD_SIZE values.

    WORLD_SIZE is the rank count, so values < 1 are nonsensical
    regardless of what the workload declares -- reject them up-front
    with a clear message instead of silently treating ``WORLD_SIZE=0``
    like the ``> 1`` / ``< min`` branches don't fire.
    """

    def test_world_size_zero_raises(self):
        """WORLD_SIZE=0 is structurally invalid."""
        with patch.dict(os.environ, {"WORLD_SIZE": "0"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(SingleProcessWorkload)
            assert "WORLD_SIZE=0" in str(exc_info.value)
            assert "must be >= 1" in str(exc_info.value)

    def test_world_size_negative_raises(self):
        """Negative WORLD_SIZE is structurally invalid."""
        with patch.dict(os.environ, {"WORLD_SIZE": "-1"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(DistributedWorkload)
            assert "WORLD_SIZE=-1" in str(exc_info.value)
            assert "must be >= 1" in str(exc_info.value)


class TestErrorMessages:
    """Tests for error message quality."""

    def test_single_process_error_mentions_workload_name(self):
        """Error message includes workload class name."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(SingleProcessWorkload)

            assert "SingleProcessWorkload" in str(exc_info.value)

    def test_distributed_error_suggests_torchrun(self):
        """Error message suggests torchrun command."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                validate_launch_mode(DistributedWorkload)

            error_msg = str(exc_info.value)
            assert "torchrun --standalone --nproc_per_node=2" in error_msg
            # Must steer users to the console script, not the non-runnable
            # `-m aorta run` form the message previously suggested.
            assert "$(which aorta) run" in error_msg


class TestTrialIsolationValidation:
    def test_auto_uses_workload_default(self):
        assert resolve_trial_isolation(SingleProcessWorkload, "auto") == "in_process"
        assert resolve_trial_isolation(RequiredProcessWorkload, "auto") == "process"

    def test_recipe_can_strengthen_to_process(self):
        assert resolve_trial_isolation(ProcessCapableWorkload, "process") == "process"

    def test_workload_must_opt_in_to_process(self):
        with pytest.raises(ValueError, match="does not support.*process"):
            resolve_trial_isolation(SingleProcessWorkload, "process")

    def test_required_workload_cannot_be_downgraded(self):
        with pytest.raises(ValueError, match="requires.*process"):
            resolve_trial_isolation(RequiredProcessWorkload, "in_process")

    def test_side_effect_free_policy_resolves_auto(self):
        assert (
            resolve_trial_isolation_policy(
                "required",
                PROCESS_REQUIRED_POLICY,
                "auto",
            )
            == "process"
        )
        assert (
            resolve_trial_isolation_policy(
                "optional",
                PROCESS_OPTIONAL_POLICY,
                "auto",
            )
            == "in_process"
        )

    @pytest.mark.parametrize("value", ["thread", "", 1])
    def test_invalid_request_rejected(self, value):
        with pytest.raises(ValueError, match="trial_isolation"):
            resolve_trial_isolation(SingleProcessWorkload, value)

    def test_malformed_plugin_metadata_rejected(self):
        class BadWorkload(SingleProcessWorkload):
            trial_isolation_default = "thread"

        with pytest.raises(ValueError, match="invalid trial_isolation_default"):
            resolve_trial_isolation(BadWorkload, "auto")

    def test_invalid_distributed_result_scope_rejected(self):
        class BadScopeWorkload(SingleProcessWorkload):
            distributed_result_scope = "localish"

        with pytest.raises(ValueError, match="distributed_result_scope"):
            resolve_trial_isolation(BadScopeWorkload, "auto")
