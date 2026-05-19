"""Tests for the dispatcher module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aorta.run.dispatcher import RunRequest, run_trials
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
        from dataclasses import FrozenInstanceError

        req = RunRequest(workload="test", trials=1)
        with pytest.raises(FrozenInstanceError):
            req.workload = "modified"  # type: ignore[misc]

    def test_mutable_fields_are_defensively_copied(self):
        """Mutating the dicts passed in must not affect the stored request.

        ``frozen=True`` only blocks attribute reassignment.  The dict
        fields would otherwise still be mutable through the original
        reference, letting a caller change an in-flight request after
        ``run_trials`` has read its config.
        """
        extra_env_in = {"FOO": "1"}
        config_in = {"steps": 10, "nested": {"k": "v"}}

        req = RunRequest(
            workload="w",
            trials=1,
            extra_env=extra_env_in,
            config_overrides=config_in,
        )

        extra_env_in["FOO"] = "999"
        extra_env_in["NEW"] = "added"
        config_in["steps"] = 999
        config_in["nested"]["k"] = "modified"

        assert req.extra_env == {"FOO": "1"}
        assert req.config_overrides["steps"] == 10
        assert req.config_overrides["nested"]["k"] == "v"


class TestRunTrials:
    """Tests for run_trials function."""

    @pytest.fixture(autouse=True)
    def reset_workload_state(self):
        """Reset workload state before each test."""
        PassingWorkload.setup_called = False
        PassingWorkload.run_called = False
        PassingWorkload.cleanup_called = False
        yield

    def test_rejects_non_positive_trials(self, tmp_path):
        """run_trials raises ValueError when trials < 1."""
        for bad in (0, -1, -100):
            req = RunRequest(workload="anything", trials=bad, results_dir=tmp_path)
            with pytest.raises(ValueError, match="trials must be >= 1"):
                run_trials(req)

    def test_rejects_invalid_extra_env_keys(self, tmp_path):
        """``extra_env`` validation must mirror the CLI's parse-time check.

        Library callers (B2 triage matrix, programmatic users) pass
        ``extra_env`` directly without going through CLI parsing.
        Without this check, a bad key would only surface mid-trial
        inside ``os.environ.update`` with a much less friendly error.
        """
        for bad in ({"": "v"}, {"1BAD": "v"}, {"with space": "v"}, {"=foo": "v"}):
            req = RunRequest(
                workload="anything",
                trials=1,
                extra_env=bad,
                results_dir=tmp_path,
            )
            with pytest.raises(ValueError, match="Invalid extra_env keys"):
                run_trials(req)

    def test_rejects_reserved_aorta_prefix_in_config_overrides(self, tmp_path):
        """``_aorta_*`` keys are reserved for platform-supplied values
        (currently ``_aorta_environment``).  A caller passing one in
        ``config_overrides`` would be silently clobbered by the
        dispatcher; failing loudly surfaces typos and prevents callers
        from depending on a slot that isn't theirs.
        """
        req = RunRequest(
            workload="anything",
            trials=1,
            config_overrides={"_aorta_environment": "hijack"},
            results_dir=tmp_path,
        )
        with pytest.raises(ValueError, match=r"reserved '_aorta_' prefix"):
            run_trials(req)

    def test_cleanup_error_is_logged_not_swallowed(self, tmp_path, caplog):
        """A failing ``cleanup()`` is logged so leaked resources are visible."""
        import logging

        class CleanupExplodes(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                pass

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                raise RuntimeError("simulated GPU teardown failure")

        mock_ep = MagicMock()
        mock_ep.name = "leaky"
        mock_ep.load.return_value = CleanupExplodes

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with caplog.at_level(logging.WARNING, logger="aorta.run.dispatcher"):
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                req = RunRequest(
                    workload="leaky",
                    trials=1,
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        # The trial itself still passed -- cleanup failure must NOT
        # mask the original outcome.
        assert results[0].exit_status == "ok"
        # But the cleanup failure was logged with type + trial_id.
        # trial_id format is ``<workload>_d<n>_m<n>_t<n>`` (B1 spec).
        msgs = [r.getMessage() for r in caplog.records]
        assert any("cleanup()" in m and "RuntimeError" in m and "leaky_d0_m0_t0" in m for m in msgs)

    def test_non_integer_rank_falls_back_to_zero(self, tmp_path, caplog):
        """A non-integer RANK env var must not crash the run."""
        import logging

        mock_ep = MagicMock()
        mock_ep.name = "passing"
        mock_ep.load.return_value = PassingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with caplog.at_level(logging.WARNING, logger="aorta.run.dispatcher"):
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                with patch.dict(os.environ, {"RANK": "not-a-number"}):
                    req = RunRequest(
                        workload="passing",
                        trials=1,
                        results_dir=tmp_path,
                    )
                    results = run_trials(req)

        assert len(results) == 1
        assert results[0].exit_status == "ok"
        # Rank parsing fell back to 0, so the JSON file *was* written.
        # Filename mirrors the cell-coordinate trial_id (d/m/t).
        assert (tmp_path / "passing" / "trial_d0_m0_t0.json").exists()
        # And we logged a warning about the bad value.
        assert any(
            "RANK" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )

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
        # Spec: trial_id encodes <workload>_d<dataset>_m<mitigation>_t<trial>.
        assert [r.trial_id for r in results] == [
            "passing_d0_m0_t0",
            "passing_d0_m0_t1",
            "passing_d0_m0_t2",
        ]

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

        # Check JSON files exist (filename mirrors trial_id's d/m/t).
        json_0 = tmp_path / "passing" / "trial_d0_m0_t0.json"
        json_1 = tmp_path / "passing" / "trial_d0_m0_t1.json"
        assert json_0.exists()
        assert json_1.exists()

        # Check JSON content is valid
        with open(json_0) as f:
            data = json.load(f)
        assert data["trial_id"] == "passing_d0_m0_t0"
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
        json_0 = tmp_path / "passing" / "trial_d0_m0_t0.json"
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

    def test_sidecar_files_are_forwarded_to_registry(self, tmp_path):
        """``--mitigations-file`` (sidecar_files) must reach the registry.

        Pin the wiring with a real B3.1 sidecar JSON: declare a custom
        mitigation, drive a trial through it, and confirm the env-var
        actually appears in the live ``os.environ`` during ``setup()``.
        Regression guard: previously the CLI parsed the option and
        dropped it on the floor.
        """
        captured: dict[str, str] = {}

        class CaptureWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                captured.update(os.environ)

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        sidecar = tmp_path / "site_mitigations.json"
        sidecar.write_text(
            json.dumps(
                {
                    "version": 1,
                    "mitigations": {
                        "sidecar_only_mit": {"AORTA_TEST_SIDECAR": "yes"},
                    },
                }
            )
        )

        mock_ep = MagicMock()
        mock_ep.name = "capture"
        mock_ep.load.return_value = CaptureWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="capture",
                trials=1,
                mitigations=("sidecar_only_mit",),
                results_dir=tmp_path,
                sidecar_files=(sidecar,),
            )
            results = run_trials(req)

        assert results[0].exit_status == "ok"
        assert captured.get("AORTA_TEST_SIDECAR") == "yes"

    def test_env_snapshot_reflects_mitigation_env(self, tmp_path):
        """``env_snapshot`` must be captured AFTER mitigations apply.

        The persisted snapshot is supposed to describe the *actual*
        environment the workload ran under.  ``tf32_off`` sets
        ``DISABLE_TF32=1``, which is in A1's canonical env-var list, so
        it should appear in the trial result's ``env.env_vars``.
        Capturing pre-application would silently lose this signal.
        """

        class TrivialWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                pass

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "trivial"
        mock_ep.load.return_value = TrivialWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        original = os.environ.pop("DISABLE_TF32", None)
        try:
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                req = RunRequest(
                    workload="trivial",
                    trials=1,
                    mitigations=("tf32_off",),
                    results_dir=tmp_path,
                )
                results = run_trials(req)
        finally:
            if original is not None:
                os.environ["DISABLE_TF32"] = original

        env_vars = results[0].env.get("env_vars", {})
        assert env_vars.get("DISABLE_TF32") == "1", (
            "snapshot must reflect the mitigation env -- order bug if missing"
        )

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

    def test_resolved_environment_threaded_into_workload_config(self, tmp_path):
        """The dispatcher threads the resolved ``Environment`` descriptor
        into ``config["_aorta_environment"]`` so workloads that can
        isolate themselves (e.g., a wrapper that invokes ``docker run``
        instead of ``python``) know *which* environment was selected for
        this cell.

        ``aorta triage`` varies the environment axis across cells; the
        in-process dispatcher API has no other way to communicate that
        per-cell selection to the workload (the platform deliberately
        does not pull or run docker images itself).
        """
        captured_config = {}

        class EnvCapturingWorkload(Workload):
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
        mock_ep.name = "envcapture"
        mock_ep.load.return_value = EnvCapturingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        # Built-in ``local`` env has docker=None, venv=None -- pick that
        # to keep the test free of registry / sidecar setup.
        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="envcapture",
                trials=1,
                environment="local",
                results_dir=tmp_path,
            )
            run_trials(req)

        env_block = captured_config.get("_aorta_environment")
        assert env_block is not None, (
            "dispatcher must thread the resolved Environment descriptor into "
            "config['_aorta_environment']; otherwise workloads cannot tell "
            "which environment was selected for the cell."
        )
        assert env_block["name"] == "local"
        # Built-in ``local`` has no docker / venv.
        assert env_block["docker"] is None
        assert env_block["venv"] is None
        assert env_block["source_package"] == "aorta"

    def test_non_none_docker_env_round_trips_to_both_sites(self, tmp_path):
        """Built-in ``local`` env has ``docker=venv=buck_target=None``, so the
        other env tests don't exercise the non-``None`` branch of the
        ``asdict(env_descriptor)`` serialization at the two call sites
        (``config["_aorta_environment"]`` and ``TrialResult.execution_env``).
        A regression that drops a field from either dict would slip
        through.  Pin all five fields at both sites with a docker-set env.
        """
        from aorta.registry import Environment

        captured_config: dict = {}

        class EnvCapturingWorkload(Workload):
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

        env = Environment(
            name="docker-test",
            docker="img@sha256:deadbeef",
            venv=None,
            source_package="test",
        )

        mock_ep = MagicMock()
        mock_ep.name = "envcapture"
        mock_ep.load.return_value = EnvCapturingWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="docker-test",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        expected = {
            "name": "docker-test",
            "docker": "img@sha256:deadbeef",
            "venv": None,
            "buck_target": None,
            "source_package": "test",
        }
        assert captured_config["_aorta_environment"] == expected
        assert results[0].execution_env == expected

    def test_non_none_buck_target_env_round_trips_to_both_sites(self, tmp_path):
        """#182: peer of the docker-set round-trip test above, but for the
        ``buck_target`` field. Pins that the dispatcher threads a Buck-tier
        environment all the way through to ``config["_aorta_environment"]``
        and ``TrialResult.execution_env`` without losing the field.

        This is the contract aorta-internal#36 (recom_repro Buck pilot) and
        aorta-internal#37 (regression-gate Buck tier) will rely on -- a
        regression here breaks both downstream workstreams silently.
        """
        from aorta.registry import Environment

        captured_config: dict = {}

        class EnvCapturingWorkload(Workload):
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

        env = Environment(
            name="buck-test",
            docker=None,
            venv=None,
            buck_target="//workloads/recom_repro:recom_repro",
            source_package="test",
        )

        mock_ep = MagicMock()
        mock_ep.name = "envcapture"
        mock_ep.load.return_value = EnvCapturingWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="buck-test",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        expected = {
            "name": "buck-test",
            "docker": None,
            "venv": None,
            "buck_target": "//workloads/recom_repro:recom_repro",
            "source_package": "test",
        }
        assert captured_config["_aorta_environment"] == expected
        assert results[0].execution_env == expected


class TestEnvironmentRestoration:
    """Tests for environment variable restoration after trials."""

    def test_environment_restore_does_not_use_global_clear(self, tmp_path):
        """``run_trials`` must not call ``os.environ.clear()``.

        ``run_trials`` is a public library API, so a global wipe-and-
        repopulate would, for an instant, blank the entire environment
        for every other thread reading ``os.environ`` in the process.
        Use a diff-based restore instead.
        """
        clear_calls = [0]
        original_clear = os.environ.clear

        def tracking_clear():
            clear_calls[0] += 1
            original_clear()

        class Trivial(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                pass

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "trivial"
        mock_ep.load.return_value = Trivial

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with patch.object(os.environ, "clear", tracking_clear):
                req = RunRequest(
                    workload="trivial",
                    trials=2,
                    mitigations=("tf32_off",),
                    extra_env={"AORTA_TEST_TS_GUARD": "1"},
                    results_dir=tmp_path,
                )
                run_trials(req)

        assert clear_calls[0] == 0, (
            "run_trials must restore the environment by diff, "
            "not via os.environ.clear() + repopulate"
        )

    def test_dataset_and_mitigation_index_in_trial_id_and_filename(self, tmp_path):
        """``trial_id`` and the JSON filename must encode cell coordinates.

        Spec format: ``<workload>_d<dataset>_m<mitigation>_t<trial>``
        (issue #148 schema example: ``"fsdp_d0_m0_t0"``).  ``aorta
        run`` is one cell, but ``aorta triage`` (B2) calls
        ``run_trials`` once per cell with distinct
        ``dataset_index`` / ``mitigation_index`` values, so artifacts
        from different cells must not collide on disk.

        Pin both the in-memory ``trial_id`` and the on-disk filename
        for a non-zero (d, m) so a regression to ``trial_<idx>.json``
        would be caught.
        """
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
                dataset_index=3,
                mitigation_index=7,
            )
            results = run_trials(req)

        assert [r.trial_id for r in results] == [
            "passing_d3_m7_t0",
            "passing_d3_m7_t1",
        ]
        assert (tmp_path / "passing" / "trial_d3_m7_t0.json").exists()
        assert (tmp_path / "passing" / "trial_d3_m7_t1.json").exists()
        # Old format must not appear.
        assert not (tmp_path / "passing" / "trial_0.json").exists()
        assert not (tmp_path / "passing" / "trial_1.json").exists()

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
