"""End-to-end integration tests for aorta run."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aorta.cli.run import run as run_cmd
from aorta.run.dispatcher import RunRequest, run_trials
from aorta.workloads import Workload, WorkloadResult


class IntegrationTestWorkload(Workload):
    """Integration test workload with configurable behavior."""

    launch_mode = "single_process"
    min_world_size = 1

    def __init__(self, config):
        super().__init__(config)
        self.steps = config.get("steps", 10)

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(
            passed=True,
            total_iterations=self.steps,
            elapsed_sec=0.1 * self.steps,
            metrics={"throughput": 100.0 / self.steps},
        )

    def cleanup(self) -> None:
        pass


@pytest.fixture
def mock_workload():
    """Fixture to mock workload discovery."""
    mock_ep = MagicMock()
    mock_ep.name = "integration_test"
    mock_ep.load.return_value = IntegrationTestWorkload

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        yield


class TestEndToEndDispatcher:
    """End-to-end tests using the dispatcher API."""

    def test_single_trial_writes_valid_json(self, tmp_path, mock_workload):
        """Single trial produces valid JSON output."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            environment="local",
            mitigations=("none",),
            config_overrides={"steps": 50},
            results_dir=tmp_path,
        )

        results = run_trials(req)

        assert len(results) == 1
        result = results[0]
        assert result.workload == "integration_test"
        assert result.exit_status == "ok"

        # Verify JSON file (filename encodes cell coordinates per spec).
        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert data["trial_id"] == "integration_test_d0_m0_t0"
        assert data["workload"] == "integration_test"
        assert data["exit_status"] == "ok"
        assert data["config"]["steps"] == 50
        assert data["result"]["total_iterations"] == 50

    def test_multiple_trials_all_write_json(self, tmp_path, mock_workload):
        """Multiple trials each write their own JSON file."""
        req = RunRequest(
            workload="integration_test",
            trials=3,
            results_dir=tmp_path,
        )

        results = run_trials(req)

        assert len(results) == 3

        for i in range(3):
            json_path = tmp_path / "integration_test" / f"trial_d0_m0_t{i}.json"
            assert json_path.exists(), f"Missing trial_d0_m0_t{i}.json"

            with open(json_path) as f:
                data = json.load(f)

            assert data["trial_id"] == f"integration_test_d0_m0_t{i}"

    def test_mitigations_recorded_in_result(self, tmp_path, mock_workload):
        """Mitigations are recorded in trial result."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            mitigations=("tf32_off",),
            results_dir=tmp_path,
        )

        results = run_trials(req)

        assert results[0].mitigations_applied == ("tf32_off",)

        # Also check JSON
        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        with open(json_path) as f:
            data = json.load(f)
        assert data["mitigations_applied"] == ["tf32_off"]

    def test_execution_env_populated(self, tmp_path, mock_workload):
        """Execution environment metadata mirrors the registry Environment."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            environment="local",
            results_dir=tmp_path,
        )

        results = run_trials(req)

        exec_env = results[0].execution_env
        # Schema mirrors aorta.registry.Environment: name + docker +
        # venv + source_package.  The built-in "local" has no docker /
        # venv (current process) and is sourced from the aorta package.
        assert exec_env["name"] == "local"
        assert exec_env["docker"] is None
        assert exec_env["venv"] is None
        assert exec_env["source_package"] == "aorta"

    def test_env_snapshot_captured(self, tmp_path, mock_workload):
        """Environment snapshot from aorta.instrumentation is captured."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            results_dir=tmp_path,
        )

        results = run_trials(req)

        # A1 EnvSnapshot schema -- spot-check stable, always-present fields.
        env = results[0].env
        assert "schema_version" in env
        assert "python_version" in env
        assert "env_vars" in env
        # ``partial`` is always populated (True/False); ``partial_reasons``
        # is always a list.  Together they form A1's fail-soft contract.
        assert isinstance(env.get("partial"), bool)
        assert isinstance(env.get("partial_reasons"), list)

    def test_wall_clock_measured(self, tmp_path, mock_workload):
        """Wall clock time is measured for each trial."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            results_dir=tmp_path,
        )

        results = run_trials(req)

        assert results[0].wall_clock_sec > 0


class TestEndToEndCli:
    """End-to-end tests using the CLI."""

    def test_cli_success_message(self, tmp_path, mock_workload):
        """CLI shows success message on passing trials."""
        runner = CliRunner()
        result = runner.invoke(
            run_cmd,
            [
                "--workload",
                "integration_test",
                "--trials",
                "1",
                "--results-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "passed" in result.output.lower()
        assert "Results in:" in result.output

    def test_cli_writes_json(self, tmp_path, mock_workload):
        """CLI writes JSON files to results directory."""
        runner = CliRunner()
        runner.invoke(
            run_cmd,
            [
                "--workload",
                "integration_test",
                "--trials",
                "2",
                "--results-dir",
                str(tmp_path),
            ],
        )

        assert (tmp_path / "integration_test" / "trial_d0_m0_t0.json").exists()
        assert (tmp_path / "integration_test" / "trial_d0_m0_t1.json").exists()

    def test_cli_with_steps(self, tmp_path, mock_workload):
        """CLI --steps option is passed to workload."""
        runner = CliRunner()
        runner.invoke(
            run_cmd,
            [
                "--workload",
                "integration_test",
                "--trials",
                "1",
                "--steps",
                "75",
                "--results-dir",
                str(tmp_path),
            ],
        )

        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["config"]["steps"] == 75

    def test_cli_with_mitigations(self, tmp_path, mock_workload):
        """CLI --mitigations option is recorded."""
        runner = CliRunner()
        runner.invoke(
            run_cmd,
            [
                "--workload",
                "integration_test",
                "--trials",
                "1",
                "--mitigations",
                "tf32_off",
                "--results-dir",
                str(tmp_path),
            ],
        )

        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "tf32_off" in data["mitigations_applied"]


class TestJsonSchema:
    """Tests for JSON output schema compliance."""

    def test_schema_version_present(self, tmp_path, mock_workload):
        """JSON includes schema_version field."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            results_dir=tmp_path,
        )

        run_trials(req)

        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "schema_version" in data
        assert data["schema_version"] == "0.1"

    def test_required_fields_present(self, tmp_path, mock_workload):
        """All required fields are present in JSON output."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            results_dir=tmp_path,
        )

        run_trials(req)

        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        with open(json_path) as f:
            data = json.load(f)

        required_fields = [
            "schema_version",
            "trial_id",
            "workload",
            "execution_env",
            "mitigations_applied",
            "config",
            "env",
            "result",
            "wall_clock_sec",
            "exit_status",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_result_contains_workload_result_fields(self, tmp_path, mock_workload):
        """Result field contains WorkloadResult fields."""
        req = RunRequest(
            workload="integration_test",
            trials=1,
            results_dir=tmp_path,
        )

        run_trials(req)

        json_path = tmp_path / "integration_test" / "trial_d0_m0_t0.json"
        with open(json_path) as f:
            data = json.load(f)

        result = data["result"]
        assert "passed" in result
        assert "total_iterations" in result
        assert "elapsed_sec" in result


class TestErrorScenarios:
    """Tests for error handling in integration scenarios."""

    def test_unknown_workload_error_cli(self):
        """CLI reports clear error for unknown workload."""
        runner = CliRunner()
        result = runner.invoke(
            run_cmd,
            [
                "--workload",
                "definitely_nonexistent_workload_xyz",
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_unknown_environment_error(self, mock_workload):
        """Unknown environment raises a clear registry error."""
        from aorta.registry import UnknownEnvironmentError

        req = RunRequest(
            workload="integration_test",
            trials=1,
            environment="nonexistent_env",
        )

        with pytest.raises(UnknownEnvironmentError) as exc_info:
            run_trials(req)

        assert "unknown environment" in str(exc_info.value).lower()
        assert "nonexistent_env" in str(exc_info.value)

    def test_unknown_mitigation_error(self, mock_workload):
        """Unknown mitigation raises a clear registry error."""
        from aorta.registry import UnknownMitigationError

        req = RunRequest(
            workload="integration_test",
            trials=1,
            mitigations=("nonexistent_mitigation",),
        )

        with pytest.raises(UnknownMitigationError) as exc_info:
            run_trials(req)

        assert "unknown mitigation" in str(exc_info.value).lower()
        assert "nonexistent_mitigation" in str(exc_info.value)
