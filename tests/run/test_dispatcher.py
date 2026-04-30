"""Tests for the dispatcher module."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from aorta.run.dispatcher import RunRequest, run_trials, _run_single_trial
from aorta.run.results import TrialResult
from aorta.run._stubs import Environment
from aorta.workloads import Workload, WorkloadResult


class PassingWorkload(Workload):
    """Mock workload that always passes."""

    launch_mode = "single_process"
    min_world_size = 1
    setup_called = False
    run_called = False
    cleanup_called = False

    def setup(self) -> None:
        PassingWorkload.setup_called = True

    def run(self) -> WorkloadResult:
        PassingWorkload.run_called = True
        return WorkloadResult(
            passed=True,
            total_iterations=100,
            elapsed_sec=1.5,
        )

    def cleanup(self) -> None:
        PassingWorkload.cleanup_called = True


class FailingWorkload(Workload):
    """Mock workload that always fails."""

    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(
            passed=False,
            failure_count=3,
            failure_details=[{"iter": 50, "error": "NaN detected"}],
        )

    def cleanup(self) -> None:
        pass


class CrashingWorkload(Workload):
    """Mock workload that crashes during run."""

    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        raise RuntimeError("Workload crashed!")

    def cleanup(self) -> None:
        pass


class TestRunRequest:
    """Tests for RunRequest dataclass."""

    def test_default_values(self):
        """RunRequest has sensible defaults."""
        req = RunRequest(workload="test", trials=1)
        assert req.environment == "local"
        assert req.mitigations == ("none",)
        assert req.extra_env == {}
        assert req.steps is None
        assert req.config_overrides == {}
        assert req.results_dir == Path("results")
        assert req.collect == ()

    def test_custom_values(self):
        """RunRequest accepts custom values."""
        req = RunRequest(
            workload="fsdp",
            trials=3,
            environment="ci",
            mitigations=("tf32_off",),
            extra_env={"DEBUG": "1"},
            steps=100,
            config_overrides={"batch_size": 32},
            results_dir=Path("/tmp/results"),
            collect=("rocprof",),
        )
        assert req.workload == "fsdp"
        assert req.trials == 3
        assert req.environment == "ci"
        assert req.mitigations == ("tf32_off",)
        assert req.extra_env == {"DEBUG": "1"}
        assert req.steps == 100
        assert req.config_overrides == {"batch_size": 32}
        assert req.results_dir == Path("/tmp/results")
        assert req.collect == ("rocprof",)

    def test_is_frozen(self):
        """RunRequest is immutable."""
        req = RunRequest(workload="test", trials=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            req.workload = "modified"  # type: ignore[misc]


class TestRunTrials:
    """Tests for run_trials function."""

    @pytest.fixture(autouse=True)
    def reset_workload_state(self):
        """Reset workload state before each test."""
        PassingWorkload.setup_called = False
        PassingWorkload.run_called = False
        PassingWorkload.cleanup_called = False
        yield

    def test_runs_workload_lifecycle(self, tmp_path):
        """Dispatcher calls setup, run, cleanup in order."""
        mock_ep = MagicMock()
        mock_ep.name = "passing"
        mock_ep.load.return_value = PassingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="passing",
                trials=1,
                results_dir=tmp_path,
            )
            results = run_trials(req)

        assert PassingWorkload.setup_called
        assert PassingWorkload.run_called
        assert PassingWorkload.cleanup_called
        assert len(results) == 1
        assert results[0].exit_status == "ok"

    def test_multiple_trials(self, tmp_path):
        """Dispatcher runs correct number of trials."""
        mock_ep = MagicMock()
        mock_ep.name = "passing"
        mock_ep.load.return_value = PassingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="passing",
                trials=3,
                results_dir=tmp_path,
            )
            results = run_trials(req)

        assert len(results) == 3
        assert all(r.exit_status == "ok" for r in results)
        assert [r.trial_id for r in results] == ["passing_t0", "passing_t1", "passing_t2"]

    def test_failing_workload_sets_exit_status(self, tmp_path):
        """Failed workload sets exit_status to workload_failed."""
        mock_ep = MagicMock()
        mock_ep.name = "failing"
        mock_ep.load.return_value = FailingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="failing",
                trials=1,
                results_dir=tmp_path,
            )
            results = run_trials(req)

        assert len(results) == 1
        assert results[0].exit_status == "workload_failed"

    def test_crashing_workload_sets_exit_status(self, tmp_path):
        """Crashing workload sets exit_status to infrastructure_failed."""
        mock_ep = MagicMock()
        mock_ep.name = "crashing"
        mock_ep.load.return_value = CrashingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="crashing",
                trials=1,
                results_dir=tmp_path,
            )
            results = run_trials(req)

        assert len(results) == 1
        assert results[0].exit_status == "infrastructure_failed"
        assert "RuntimeError" in str(results[0].result["failure_details"])

    def test_one_failing_trial_doesnt_stop_others(self, tmp_path):
        """One trial failing doesn't prevent other trials from running."""
        # Create a workload that fails on first trial
        call_count = [0]

        class AlternatingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                pass

            def run(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("First trial fails")
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "alternating"
        mock_ep.load.return_value = AlternatingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="alternating",
                trials=3,
                results_dir=tmp_path,
            )
            results = run_trials(req)

        assert len(results) == 3
        assert call_count[0] == 3  # All 3 trials ran
        assert results[0].exit_status == "infrastructure_failed"
        assert results[1].exit_status == "ok"
        assert results[2].exit_status == "ok"

    def test_writes_json_files(self, tmp_path):
        """Dispatcher writes JSON files to results_dir."""
        mock_ep = MagicMock()
        mock_ep.name = "passing"
        mock_ep.load.return_value = PassingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="passing",
                trials=2,
                results_dir=tmp_path,
            )
            run_trials(req)

        # Check JSON files exist
        json_0 = tmp_path / "passing" / "trial_0.json"
        json_1 = tmp_path / "passing" / "trial_1.json"
        assert json_0.exists()
        assert json_1.exists()

        # Check JSON content is valid
        with open(json_0) as f:
            data = json.load(f)
        assert data["trial_id"] == "passing_t0"
        assert data["workload"] == "passing"
        assert data["exit_status"] == "ok"

    def test_rank_aware_writing(self, tmp_path):
        """Only RANK=0 writes JSON files."""
        mock_ep = MagicMock()
        mock_ep.name = "passing"
        mock_ep.load.return_value = PassingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        # Simulate rank 1 (not rank 0)
        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with patch.dict(os.environ, {"RANK": "1"}):
                req = RunRequest(
                    workload="passing",
                    trials=1,
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        # Should still return results
        assert len(results) == 1
        assert results[0].exit_status == "ok"

        # But should not write JSON
        json_0 = tmp_path / "passing" / "trial_0.json"
        assert not json_0.exists()


class TestMitigationUnion:
    """Tests for mitigation environment variable handling."""

    def test_mitigation_env_applied(self, tmp_path):
        """Mitigation env vars are applied during trial."""
        captured_env = {}

        class EnvCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                captured_env.update(dict(os.environ))

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "capture"
        mock_ep.load.return_value = EnvCapturingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="capture",
                trials=1,
                mitigations=("tf32_off",),
                results_dir=tmp_path,
            )
            run_trials(req)

        assert captured_env.get("DISABLE_TF32") == "1"

    def test_extra_env_overrides_mitigations(self, tmp_path):
        """extra_env overrides mitigation env vars."""
        captured_env = {}

        class EnvCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                captured_env.update(dict(os.environ))

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "capture"
        mock_ep.load.return_value = EnvCapturingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="capture",
                trials=1,
                mitigations=("tf32_off",),
                extra_env={"DISABLE_TF32": "0", "CUSTOM_VAR": "custom"},
                results_dir=tmp_path,
            )
            run_trials(req)

        # extra_env should override mitigation
        assert captured_env.get("DISABLE_TF32") == "0"
        assert captured_env.get("CUSTOM_VAR") == "custom"


class TestConfigOverrides:
    """Tests for workload configuration."""

    def test_steps_passed_to_workload(self, tmp_path):
        """Steps are passed to workload config."""
        captured_config = {}

        class ConfigCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def __init__(self, config):
                super().__init__(config)
                captured_config.update(config)

            def setup(self):
                pass

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "config"
        mock_ep.load.return_value = ConfigCapturingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="config",
                trials=1,
                steps=100,
                results_dir=tmp_path,
            )
            run_trials(req)

        assert captured_config.get("steps") == 100

    def test_config_overrides_passed_to_workload(self, tmp_path):
        """Config overrides are passed to workload."""
        captured_config = {}

        class ConfigCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def __init__(self, config):
                super().__init__(config)
                captured_config.update(config)

            def setup(self):
                pass

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "config"
        mock_ep.load.return_value = ConfigCapturingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="config",
                trials=1,
                config_overrides={"batch_size": 32, "lr": 0.001},
                results_dir=tmp_path,
            )
            run_trials(req)

        assert captured_config.get("batch_size") == 32
        assert captured_config.get("lr") == 0.001


class TestEnvironmentRestoration:
    """Tests for environment variable restoration after trials."""

    def test_environment_restored_after_trial(self, tmp_path):
        """Environment is restored after each trial."""
        original_value = os.environ.get("TEST_RESTORE_VAR")

        class EnvModifyingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                os.environ["TEST_RESTORE_VAR"] = "modified"

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "env_modify"
        mock_ep.load.return_value = EnvModifyingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        try:
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                req = RunRequest(
                    workload="env_modify",
                    trials=1,
                    results_dir=tmp_path,
                )
                run_trials(req)

            # Environment should be restored
            assert os.environ.get("TEST_RESTORE_VAR") == original_value
        finally:
            # Cleanup
            if original_value is None:
                os.environ.pop("TEST_RESTORE_VAR", None)
            else:
                os.environ["TEST_RESTORE_VAR"] = original_value
