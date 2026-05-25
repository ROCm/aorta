"""Tests for the dispatcher module."""

import json
import os
import sys
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


class SetupCrashingWorkload(Workload):
    """Mock workload whose setup() raises (e.g. missing dependency at import)."""

    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        raise ImportError("simulated missing dependency in setup()")

    def run(self) -> WorkloadResult:
        # Should never be reached; assert just in case so a regression where
        # the dispatcher swallows setup() failures and proceeds to run()
        # surfaces loudly instead of silently passing the test.
        raise AssertionError("run() must not be called when setup() raised")

    def cleanup(self) -> None:
        pass


class TestRunRequest:
    """Tests for RunRequest dataclass."""

    def test_default_values(self):
        """RunRequest has sensible defaults."""
        req = RunRequest(workload="test", trials=1)
        assert req.environment == "local"
        # ``image`` and ``buck_target`` both default to ``None`` so
        # omitting either new CLI flag keeps the resolved
        # environment's corresponding field untouched -- backward-
        # compat with every pre-existing run.
        assert req.image is None
        assert req.buck_target is None
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

    def test_user_supplied_aorta_subprocess_argv_rejected(self, tmp_path):
        """``RunRequest.subprocess_argv`` is the only legal channel.

        Per issue #188 FR 1.11: users must not be able to smuggle argv
        via ``config_overrides`` -- the reserved-prefix guard at the
        top of ``run_trials`` blocks ``_aorta_subprocess_argv`` exactly
        like every other ``_aorta_*`` key.
        """
        req = RunRequest(
            workload="anything",
            trials=1,
            config_overrides={"_aorta_subprocess_argv": ["echo", "pwn"]},
            results_dir=tmp_path,
        )
        with pytest.raises(ValueError, match=r"reserved '_aorta_' prefix"):
            run_trials(req)

    def test_subprocess_argv_injected_into_config(self, tmp_path):
        """``RunRequest.subprocess_argv`` lands at ``config['_aorta_subprocess_argv']``.

        Verifies the dispatcher injects the typed field AFTER
        ``config_overrides`` is merged (per issue #188 FR 1.11), so a
        user-supplied ``config_overrides`` cannot clobber the slot.
        """
        captured_config: dict = {}

        class ConfigCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                captured_config.update(self.config)

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "argv_capturing"
        mock_ep.load.return_value = ConfigCapturingWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="argv_capturing",
                trials=1,
                results_dir=tmp_path,
                subprocess_argv=("echo", "hi"),
            )
            run_trials(req)
        assert captured_config["_aorta_subprocess_argv"] == ["echo", "hi"]

    def test_subprocess_argv_absent_when_none(self, tmp_path):
        """No ``_aorta_subprocess_argv`` key when ``RunRequest.subprocess_argv`` is None.

        Otherwise every existing workload's ``config`` would gain a
        spurious ``None`` slot and the JSON-serialised trial result
        would diverge from today's shape.
        """
        captured_config: dict = {}

        class ConfigCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                captured_config.update(self.config)

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "noargv"
        mock_ep.load.return_value = ConfigCapturingWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(workload="noargv", trials=1, results_dir=tmp_path)
            run_trials(req)
        assert "_aorta_subprocess_argv" not in captured_config

    def test_probe_extras_injected_into_config(self, tmp_path):
        """``RunRequest.probe_extras`` lands at ``config['_aorta_probe_extras']``.

        The probe-mode runner builds this dict per cell so
        ``SubprocessWorkload`` can pick up cell name / env-passthrough
        mode / timeout / cell-env-bundle without parsing the recipe
        itself.
        """
        captured_config: dict = {}

        class ConfigCapturingWorkload(Workload):
            launch_mode = "single_process"
            min_world_size = 1

            def setup(self):
                captured_config.update(self.config)

            def run(self):
                return WorkloadResult(passed=True)

            def cleanup(self):
                pass

        mock_ep = MagicMock()
        mock_ep.name = "extras_capturing"
        mock_ep.load.return_value = ConfigCapturingWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="extras_capturing",
                trials=1,
                results_dir=tmp_path,
                probe_extras={"cell_name": "none-none", "timeout_per_trial": None},
            )
            run_trials(req)
        assert captured_config["_aorta_probe_extras"] == {
            "cell_name": "none-none",
            "timeout_per_trial": None,
        }

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

    def test_setup_crashing_workload_sets_workload_setup_failed(self, tmp_path):
        """setup() exception gets its own bucket, not infrastructure_failed.

        A row of all-setup-failures must read as "workload never got off
        the ground", not "100% reproduction of the bug under test" -- so
        the dispatcher splits the setup() try-block out and attributes
        its exception to workload_setup_failed. The failure_details
        record carries a ``phase: "setup"`` marker so trial_*.json is
        self-describing.
        """
        mock_ep = MagicMock()
        mock_ep.name = "setup_crashing"
        mock_ep.load.return_value = SetupCrashingWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            req = RunRequest(
                workload="setup_crashing",
                trials=1,
                results_dir=tmp_path,
            )
            results = run_trials(req)

        assert len(results) == 1
        assert results[0].exit_status == "workload_setup_failed"
        details = results[0].result["failure_details"]
        assert len(details) == 1
        assert details[0]["type"] == "ImportError"
        assert details[0]["phase"] == "setup"
        assert "missing dependency" in details[0]["error"]
        assert results[0].result["main_work_started"] is False

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

        This is the contract downstream Buck-aware workloads and
        regression-gate consumers rely on -- a regression here breaks
        both downstream workstreams silently.
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


class TestBuckTargetIsKeywordOnly:
    """``RunRequest.buck_target`` MUST be keyword-only.

    Pins the backward-compat guarantee that adding ``buck_target``
    to ``RunRequest`` does not shift the positional ``__init__``
    signature. The field is declared BEFORE ``mitigations`` in the
    source so the docstring "Attributes:" order matches the
    conceptual grouping (env-tier overlay first, then mitigation
    set); without ``kw_only=True`` that ordering would silently
    move ``mitigations`` from positional slot 4 to slot 5, breaking
    every external caller that constructed a ``RunRequest``
    positionally as ``RunRequest("wl", 1, "env", ("mit",))``.

    Single-purpose class -- if this trips, the regression is one
    bit (the ``kw_only=True`` got dropped from the field()) and
    the fix is one bit too.
    """

    def test_buck_target_signature_is_keyword_only(self):
        """Inspect the dataclass's __init__ signature directly and
        assert ``buck_target``'s ``kind`` is ``KEYWORD_ONLY``.

        Direct signature inspection (rather than a "construct with
        positional and expect TypeError" probe) is the right shape
        of assertion here because ``RunRequest`` has many positional
        fields AFTER ``buck_target``'s source position
        (``extra_env``, ``steps``, ``config_overrides``, ...).
        Removing ``buck_target`` from the positional list just
        slides those down -- a string passed as the 5th positional
        would land in ``extra_env`` and silently typecheck
        (``deepcopy('a string')`` is fine), so a "construct +
        TypeError" probe would false-pass.

        Inspecting ``Parameter.kind`` is the single bit of truth:
        if ``kw_only=True`` is dropped from the field(), this
        assertion trips with a clear "POSITIONAL_OR_KEYWORD vs
        KEYWORD_ONLY" diff in the failure message.
        """
        import inspect
        sig = inspect.signature(RunRequest)
        assert sig.parameters["buck_target"].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"buck_target must be KEYWORD_ONLY (so adding it before "
            f"existing positional fields like ``mitigations`` doesn't "
            f"shift those fields' positional slots and break external "
            f"positional callers of ``RunRequest``). Got "
            f"{sig.parameters['buck_target'].kind}."
        )

    def test_mitigations_is_still_positional_at_slot_4(self):
        """Sanity: ``mitigations`` remains positionally addressable
        as the 4th argument despite ``buck_target`` being declared
        before it in the source. The whole point of ``kw_only=True``
        on ``buck_target`` is to keep this true.

        Behavior before the ``buck_target`` field was added:
        ``RunRequest("wl", 1, "env", ("none",))`` constructed with
        ``mitigations=("none",)``. Behavior after this field was
        added (this PR, the #182 follow-up that introduced the
        ``--buck-target`` overlay) MUST be identical; that's the
        backward-compat contract ``kw_only=True`` exists to
        enforce.
        """
        req = RunRequest("wl", 1, "env", ("none",))
        assert req.mitigations == ("none",)
        assert req.buck_target is None

    def test_buck_target_works_as_keyword(self):
        """Companion positive case: kwarg form is the only accepted
        spelling, and it sets the field correctly. Cheap belt-and-
        suspenders pin so a future ``kw_only=True`` typo can't pass
        the failure-path test by simply rejecting both spellings.
        """
        req = RunRequest("wl", 1, "env", buck_target="//foo:bar")
        assert req.buck_target == "//foo:bar"
        assert req.mitigations == ("none",)


class TestBuckTargetOverride:
    """RunRequest.buck_target overlays the resolved environment's Buck pin.

    These tests pin the four behavioral guarantees that
    downstream regression-gate dispatchers rely on, all
    asserted at the place the value is actually consumed
    (``config["_aorta_environment"]["buck_target"]`` and
    ``TrialResult.execution_env["buck_target"]``):

    1. Override sets the value on a named env that had no buck_target.
       (BUCK_ONLY gates pointing at ``--environment local
       --buck-target //foo:bar``.)
    2. Override replaces a value the named env already declared.
       (A buck-aware named env can be overridden per-run.)
    3. ``buck_target=None`` is a no-op: a named env's existing pin
       survives. This is the backward-compat guarantee.
    4. Override preserves the named env's other fields
       (``docker``, ``venv``, ``source_package``). The override is a
       single-axis pin, not a wholesale env replacement.

    Together these pin the contract symmetric to
    ``aorta env probe --buck-target`` -- the CLI flag overlays one
    field, the rest of the recipe is preserved.
    """

    @staticmethod
    def _build_capture_workload(captured_config):
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
        return EnvCapturingWorkload

    @staticmethod
    def _mock_workload_discovery(workload_cls):
        mock_ep = MagicMock()
        mock_ep.name = "envcapture"
        mock_ep.load.return_value = workload_cls
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        return mock_eps

    def test_override_sets_value_on_named_env_with_no_buck_target(self, tmp_path):
        """Override populates a previously-empty Buck axis.

        Scenario: an operator uses the built-in ``local`` env (which
        has ``buck_target=None``) and adds ``--buck-target //:aorta``
        to pin a buck-built binary for this run. The dispatcher must
        thread ``//:aorta`` into ``config["_aorta_environment"]``;
        otherwise a Buck-aware workload wrapper has no way to know
        which target to ``buck2 run``.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="local",
            docker=None, venv=None, buck_target=None,
            source_package="aorta",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="local",
                    buck_target="//:aorta",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert captured_config["_aorta_environment"]["buck_target"] == "//:aorta"
        assert results[0].execution_env["buck_target"] == "//:aorta"

    def test_override_replaces_named_env_buck_target(self, tmp_path):
        """Override wins when the named env ALSO declared buck_target.

        Scenario: a registered buck-aware env declares
        ``//workloads/recom_repro:recom_repro`` as its default
        target; the operator overrides per-run to point at a custom
        Buck-built CLI for an A/B test. Without the runtime
        override, the operator would have to register a one-shot
        named env per variant -- exactly the friction this flag is
        meant to remove.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="recom-buck",
            docker=None, venv=None,
            buck_target="//workloads/recom_repro:recom_repro",
            source_package="aorta",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="recom-buck",
                    buck_target="//:aorta",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert captured_config["_aorta_environment"]["buck_target"] == "//:aorta"
        assert results[0].execution_env["buck_target"] == "//:aorta"

    def test_none_override_preserves_named_env_buck_target(self, tmp_path):
        """``buck_target=None`` (the default) is a no-op.

        Backward-compat guarantee: a pre-existing invocation that
        relied on a named env's declared ``buck_target`` must
        continue to see that value flow into
        ``_aorta_environment``. Otherwise this CLI addition would
        silently break every existing buck-aware named env.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="recom-buck",
            docker=None, venv=None,
            buck_target="//workloads/recom_repro:recom_repro",
            source_package="aorta",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="recom-buck",
                    # buck_target omitted -- default is None
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert (captured_config["_aorta_environment"]["buck_target"]
                == "//workloads/recom_repro:recom_repro")
        assert (results[0].execution_env["buck_target"]
                == "//workloads/recom_repro:recom_repro")

    def test_empty_string_override_preserves_named_env_buck_target(self, tmp_path):
        """``buck_target=""`` is treated as "no override" -- same as ``None``.

        Empty string is never a valid Buck2 label, so an explicit
        ``--buck-target ""`` (or a library caller passing ``""``)
        should NOT silently overlay ``buck_target=""`` onto the
        resolved env. The dispatcher uses a truthy check so both
        ``None`` (the default) and ``""`` flow through without
        touching the named env's existing pin -- otherwise an
        operator who accidentally produced an empty value (shell
        variable expansion, dispatcher emitting an unset
        environment override) would land in a ``buck2 run ""``
        style failure that's hard to attribute back to the flag.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="recom-buck",
            docker=None, venv=None,
            buck_target="//workloads/recom_repro:recom_repro",
            source_package="aorta",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="recom-buck",
                    buck_target="",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert (captured_config["_aorta_environment"]["buck_target"]
                == "//workloads/recom_repro:recom_repro")
        assert (results[0].execution_env["buck_target"]
                == "//workloads/recom_repro:recom_repro")

    def test_override_preserves_other_env_fields(self, tmp_path):
        """Override is a single-axis pin, not a wholesale env replacement.

        Scenario: the named env declares ``docker=img@sha256:...``
        (the gate's BUCK_IN_DOCKER tier needs the docker pin) AND a
        default ``buck_target``. The operator overrides only the Buck
        axis -- the docker pin, venv, source_package must survive.
        Otherwise a BUCK_IN_DOCKER gate that overlays the Buck axis
        would silently lose its docker pin and dispatch as
        BUCK_ONLY -- exactly the mis-classification the gate
        validator is meant to make impossible.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="buck-in-docker",
            docker="img@sha256:" + "d" * 64,
            venv="/opt/venv",
            buck_target="//default:target",
            source_package="custom_pkg",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="buck-in-docker",
                    buck_target="//:aorta",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        expected = {
            "name": "buck-in-docker",
            # docker survived the buck_target overlay
            "docker": "img@sha256:" + "d" * 64,
            # venv survived
            "venv": "/opt/venv",
            # buck_target was overridden
            "buck_target": "//:aorta",
            # source_package survived
            "source_package": "custom_pkg",
        }
        assert captured_config["_aorta_environment"] == expected
        assert results[0].execution_env == expected


class TestImageIsKeywordOnly:
    """``RunRequest.image`` MUST be keyword-only.

    Mirror of :class:`TestBuckTargetIsKeywordOnly` on the docker
    axis. Same backward-compat reasoning: declaring ``image``
    BEFORE ``mitigations`` in the source (to keep the docstring
    "Attributes:" grouping of env-tier overlays together) would
    otherwise shift ``mitigations`` from positional slot 4, silently
    breaking any external caller that constructed a ``RunRequest``
    positionally.

    Three-test structure mirrors ``TestBuckTargetIsKeywordOnly``
    exactly so a regression in one axis is locatable from the
    test-class name alone.
    """

    def test_image_signature_is_keyword_only(self):
        """Direct ``inspect.signature`` check on the ``image``
        parameter's ``kind``. Single bit of truth: if
        ``kw_only=True`` is dropped from the field() in a future
        refactor, this trips with a clear "POSITIONAL_OR_KEYWORD vs
        KEYWORD_ONLY" diff. See
        ``TestBuckTargetIsKeywordOnly::test_buck_target_signature_is_keyword_only``
        for why signature inspection beats a positional+TypeError
        probe here (same reasoning: many positional fields after
        ``image``'s source position).
        """
        import inspect
        sig = inspect.signature(RunRequest)
        assert sig.parameters["image"].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"image must be KEYWORD_ONLY (so adding it before "
            f"existing positional fields like ``mitigations`` doesn't "
            f"shift those fields' positional slots and break external "
            f"positional callers of ``RunRequest``). Got "
            f"{sig.parameters['image'].kind}."
        )

    def test_mitigations_is_still_positional_with_both_overlays_declared(self):
        """Sanity: with BOTH ``image`` and ``buck_target`` declared
        before ``mitigations`` (the post-#193 source order), the
        pre-PR-#191 invocation ``RunRequest("wl", 1, "env",
        ("none",))`` STILL binds ``mitigations`` correctly.

        Stronger version of the matching test in
        ``TestBuckTargetIsKeywordOnly`` because here we're checking
        the composite: two kw_only fields declared before
        ``mitigations``. If either field accidentally loses its
        ``kw_only=True``, this test trips.
        """
        req = RunRequest("wl", 1, "env", ("none",))
        assert req.mitigations == ("none",)
        assert req.image is None
        assert req.buck_target is None

    def test_image_works_as_keyword(self):
        """Belt-and-suspenders positive case (matches the structure
        of ``TestBuckTargetIsKeywordOnly``). Catches a typo on
        ``kw_only=True`` that would make the negative test
        false-pass by rejecting both spellings."""
        req = RunRequest("wl", 1, "env", image="sha256:" + "a" * 64)
        assert req.image == "sha256:" + "a" * 64
        assert req.mitigations == ("none",)


class TestImageOverride:
    """RunRequest.image overlays the resolved environment's docker pin.

    Symmetric peer of :class:`TestBuckTargetOverride`. Same four
    guarantees, same call sites checked
    (``config["_aorta_environment"]`` and
    ``TrialResult.execution_env``), same workload-capturing fixture
    pattern. The mirroring is deliberate -- a regression that drops
    one of the two override paths trips the corresponding test
    class, so the fix is locatable from the test name alone.

    Together these tests + :class:`TestBuckTargetOverride` pin the
    full dispatch matrix that downstream regression-gate runners
    rely on: DOCKER_ONLY (image only), BUCK_ONLY (buck only),
    BUCK_IN_DOCKER (both, exercising the independence guarantee).
    """

    @staticmethod
    def _build_capture_workload(captured_config):
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
        return EnvCapturingWorkload

    @staticmethod
    def _mock_workload_discovery(workload_cls):
        mock_ep = MagicMock()
        mock_ep.name = "envcapture"
        mock_ep.load.return_value = workload_cls
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        return mock_eps

    def test_override_sets_value_on_named_env_with_no_docker(self, tmp_path):
        """Override populates a previously-empty docker axis.

        Scenario: an operator uses the built-in ``local`` env (which
        has ``docker=None``) and adds ``--image sha256:...`` to pin
        a docker image for this run. The dispatcher must thread the
        image digest into ``config["_aorta_environment"]["docker"]``;
        otherwise a docker-aware workload wrapper has no way to know
        which image to ``docker run``.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="local",
            docker=None, venv=None, buck_target=None,
            source_package="aorta",
        )
        digest = "sha256:" + "a" * 64
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="local",
                    image=digest,
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert captured_config["_aorta_environment"]["docker"] == digest
        assert results[0].execution_env["docker"] == digest

    def test_override_replaces_named_env_docker(self, tmp_path):
        """Override wins when the named env ALSO declared docker.

        Scenario: a registered docker-aware env declares
        ``rocm/pytorch@sha256:OLD`` as its default image; the
        operator overrides per-run to test a candidate fix image.
        Without the runtime override the operator would have to
        register a one-shot named env per image variant.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="docker-base",
            docker="rocm/pytorch@sha256:" + "0" * 64,
            venv=None,
            buck_target=None,
            source_package="aorta",
        )
        candidate = "rocm/pytorch@sha256:" + "f" * 64
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="docker-base",
                    image=candidate,
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert captured_config["_aorta_environment"]["docker"] == candidate
        assert results[0].execution_env["docker"] == candidate

    def test_none_image_preserves_named_env_docker(self, tmp_path):
        """``image=None`` (the default) is a no-op.

        Backward-compat guarantee: pre-existing invocations that
        relied on a named env's declared ``docker`` continue to see
        that value flow into ``_aorta_environment``. Same role as
        the equivalent ``test_none_override_preserves...`` test for
        ``buck_target``.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        existing = "rocm/pytorch@sha256:" + "0" * 64
        env = Environment(
            name="docker-base",
            docker=existing, venv=None, buck_target=None,
            source_package="aorta",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="docker-base",
                    # image omitted -- default is None
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert captured_config["_aorta_environment"]["docker"] == existing
        assert results[0].execution_env["docker"] == existing

    def test_empty_string_image_preserves_named_env_docker(self, tmp_path):
        """``image=""`` is treated as "no override" -- same as ``None``.

        Empty string is never a valid OCI image reference, so an
        explicit ``--image ""`` (or a library caller passing ``""``)
        should NOT silently overlay ``docker=""`` onto the resolved
        env. The dispatcher uses a truthy check so both ``None``
        (the default) and ``""`` flow through without touching the
        named env's existing docker -- otherwise an operator who
        accidentally produced an empty value (shell variable
        expansion, dispatcher emitting an unset environment override)
        would land in a ``docker run ""``-style failure that's hard
        to attribute back to the flag. Mirrors the equivalent test
        on the buck axis (``test_empty_string_override_preserves_
        named_env_buck_target``).
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        existing = "rocm/pytorch@sha256:" + "0" * 64
        env = Environment(
            name="docker-base",
            docker=existing, venv=None, buck_target=None,
            source_package="aorta",
        )
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="docker-base",
                    image="",
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        assert captured_config["_aorta_environment"]["docker"] == existing
        assert results[0].execution_env["docker"] == existing

    def test_override_preserves_other_env_fields(self, tmp_path):
        """Override is a single-axis pin, not a wholesale env
        replacement.

        Scenario: the named env declares a default buck_target AND
        the operator overrides only the docker axis (e.g. to A/B-test
        a candidate image while keeping the same buck-built binary).
        The buck_target, venv, source_package fields must survive.
        Otherwise an A/B docker test would silently drop the buck pin
        and dispatch as DOCKER_ONLY instead of BUCK_IN_DOCKER --
        exactly the mis-classification the gate validator prevents.
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="buck-in-docker",
            docker="rocm/pytorch@sha256:" + "0" * 64,
            venv="/opt/venv",
            buck_target="//workloads/recom_repro:recom_repro",
            source_package="custom_pkg",
        )
        candidate = "rocm/pytorch@sha256:" + "f" * 64
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="buck-in-docker",
                    image=candidate,
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        expected = {
            "name": "buck-in-docker",
            # docker was overridden
            "docker": candidate,
            # venv survived
            "venv": "/opt/venv",
            # buck_target survived the docker overlay
            "buck_target": "//workloads/recom_repro:recom_repro",
            # source_package survived
            "source_package": "custom_pkg",
        }
        assert captured_config["_aorta_environment"] == expected
        assert results[0].execution_env == expected

    def test_image_and_buck_target_overrides_are_independent(self, tmp_path):
        """Both axes overridden simultaneously (BUCK_IN_DOCKER gate).

        Asserts the two ``replace(...)`` calls in the dispatcher
        compose: neither overlay clobbers the other. This is the
        exact runtime shape a downstream BUCK_IN_DOCKER regression-
        gate tier produces (an operator pin on BOTH the docker
        image AND the buck target, for highest-assurance gates).
        """
        from aorta.registry import Environment

        captured_config: dict = {}
        workload_cls = self._build_capture_workload(captured_config)
        env = Environment(
            name="local",
            docker=None, venv=None, buck_target=None,
            source_package="aorta",
        )
        digest = "sha256:" + "e" * 64
        target = "//:aorta"
        with patch("importlib.metadata.entry_points",
                   return_value=self._mock_workload_discovery(workload_cls)):
            with patch("aorta.run.dispatcher.get_environment", return_value=env):
                req = RunRequest(
                    workload="envcapture",
                    trials=1,
                    environment="local",
                    image=digest,
                    buck_target=target,
                    results_dir=tmp_path,
                )
                results = run_trials(req)

        env_block = captured_config["_aorta_environment"]
        assert env_block["docker"] == digest
        assert env_block["buck_target"] == target
        # No regression of the original 'local' env fields.
        assert env_block["name"] == "local"
        assert env_block["venv"] is None
        assert env_block["source_package"] == "aorta"
        assert results[0].execution_env == env_block


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


class NoisyWorkload(Workload):
    """Workload that writes a marker to stdout + stderr and records the
    ``_aorta_log_*`` keys the dispatcher injected, for save_logs tests."""

    launch_mode = "single_process"
    min_world_size = 1
    seen_config: dict = {}

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        NoisyWorkload.seen_config = dict(self.config)
        print("STDOUT-FROM-WORKLOAD")
        print("STDERR-FROM-WORKLOAD", file=sys.stderr)
        return WorkloadResult(passed=True)

    def cleanup(self) -> None:
        pass


class NoisyCrashingWorkload(Workload):
    """Prints, then raises -- exercises ExitStack flush on exception."""

    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        print("BEFORE-CRASH-STDOUT")
        print("BEFORE-CRASH-STDERR", file=sys.stderr)
        raise RuntimeError("boom")

    def cleanup(self) -> None:
        pass


class TestSaveLogs:
    """Per-trial stdout/stderr capture knob (default off)."""

    def _run(self, tmp_path, **kw) -> Path:
        mock_ep = MagicMock(name="noisy")
        mock_ep.name = "noisy"
        mock_ep.load.return_value = NoisyWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            run_trials(RunRequest(workload="noisy", trials=1, results_dir=tmp_path, **kw))
        return tmp_path / "noisy"

    def test_default_off_writes_no_log_files(self, tmp_path):
        cell_dir = self._run(tmp_path)
        assert not (cell_dir / "trial_d0_m0_t0.stdout.log").exists()
        assert not (cell_dir / "trial_d0_m0_t0.stderr.log").exists()

    def test_on_captures_stdout_and_stderr(self, tmp_path):
        cell_dir = self._run(tmp_path, save_logs=True)
        assert (cell_dir / "trial_d0_m0_t0.stdout.log").read_text().strip() == "STDOUT-FROM-WORKLOAD"
        assert (cell_dir / "trial_d0_m0_t0.stderr.log").read_text().strip() == "STDERR-FROM-WORKLOAD"

    def test_on_injects_aorta_log_keys_for_subprocess_wrappers(self, tmp_path):
        cell_dir = self._run(tmp_path, save_logs=True)
        cfg = NoisyWorkload.seen_config
        assert cfg["_aorta_save_logs"] is True
        # Absolute path-with-stem so wrappers can build sibling files
        # (``<prefix>.subprocess.{stdout,stderr}.log``) without needing
        # to know ``results_dir``. ``tmp_path`` is already absolute, so
        # the dispatcher's ``.absolute()`` call is a no-op here -- but
        # the relative-input path is covered by
        # ``test_on_injects_absolute_prefix_even_with_relative_results_dir``.
        assert cfg["_aorta_log_prefix"] == str((cell_dir / "trial_d0_m0_t0").absolute())
        assert Path(cfg["_aorta_log_prefix"]).is_absolute()

    def test_on_injects_absolute_prefix_even_with_relative_results_dir(self, tmp_path, monkeypatch):
        """Default ``RunRequest(results_dir=Path("results"))`` is relative;
        wrappers whose subprocesses run with a different cwd (docker bind
        mounts, torchrun-launched workers) would otherwise be unable to
        locate the sibling-log directory. Pin that the dispatcher
        anchors the prefix against cwd before injection."""
        monkeypatch.chdir(tmp_path)
        mock_ep = MagicMock(name="noisy")
        mock_ep.name = "noisy"
        mock_ep.load.return_value = NoisyWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            run_trials(RunRequest(workload="noisy", trials=1, save_logs=True))
        assert Path(NoisyWorkload.seen_config["_aorta_log_prefix"]).is_absolute()

    def test_on_non_rank_zero_writes_no_log_files(self, tmp_path):
        with patch.dict(os.environ, {"RANK": "1"}):
            cell_dir = self._run(tmp_path, save_logs=True)
        assert not (cell_dir / "trial_d0_m0_t0.stdout.log").exists()
        assert not (cell_dir / "trial_d0_m0_t0.stderr.log").exists()

    def test_on_captures_output_even_when_workload_raises(self, tmp_path):
        mock_ep = MagicMock(name="crasher")
        mock_ep.name = "crasher"
        mock_ep.load.return_value = NoisyCrashingWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            results = run_trials(
                RunRequest(workload="crasher", trials=1, results_dir=tmp_path, save_logs=True)
            )
        cell_dir = tmp_path / "crasher"
        assert results[0].exit_status == "infrastructure_failed"
        assert (cell_dir / "trial_d0_m0_t0.stdout.log").read_text().strip() == "BEFORE-CRASH-STDOUT"
        assert (cell_dir / "trial_d0_m0_t0.stderr.log").read_text().strip() == "BEFORE-CRASH-STDERR"

    def test_log_open_failure_degrades_gracefully_and_restores_env(self, tmp_path):
        """If log-file open() raises, the trial must still run, the env
        overlay must still be restored (otherwise mitigation vars leak
        across cells), and the _aorta_* keys must NOT be injected."""
        real_open = open

        def raising_open(path, *args, **kwargs):
            if str(path).endswith((".stdout.log", ".stderr.log")):
                raise PermissionError("simulated log-open failure")
            return real_open(path, *args, **kwargs)

        os.environ.pop("AORTA_LEAK_PROBE", None)
        NoisyWorkload.seen_config = {}
        mock_ep = MagicMock(name="noisy")
        mock_ep.name = "noisy"
        mock_ep.load.return_value = NoisyWorkload
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        with patch("importlib.metadata.entry_points", return_value=mock_eps), patch(
            "builtins.open", side_effect=raising_open
        ):
            results = run_trials(
                RunRequest(
                    workload="noisy",
                    trials=1,
                    results_dir=tmp_path,
                    save_logs=True,
                    extra_env={"AORTA_LEAK_PROBE": "leaked"},
                )
            )
        assert results[0].exit_status == "ok"
        assert not (tmp_path / "noisy" / "trial_d0_m0_t0.stdout.log").exists()
        assert "AORTA_LEAK_PROBE" not in os.environ, "env overlay leaked when log-open failed"
        assert "_aorta_save_logs" not in NoisyWorkload.seen_config
        assert "_aorta_log_prefix" not in NoisyWorkload.seen_config
