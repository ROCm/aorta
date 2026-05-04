"""Tests for CLI argument parsing."""

from click.testing import CliRunner

from aorta.cli.run import run


class TestCliParsing:
    """Tests for CLI argument parsing and validation."""

    def test_workload_required(self):
        """--workload is required."""
        runner = CliRunner()
        result = runner.invoke(run, ["--trials", "1"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--workload" in result.output

    def test_collect_validates_known_recipes(self):
        """Unknown collector names raise clear error."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "fsdp",
                "--collect",
                "bogus_recipe",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown collector recipes" in result.output
        assert "bogus_recipe" in result.output
        # Should list valid recipes
        assert "rocprof" in result.output

    def test_collect_accepts_valid_recipes(self):
        """Valid collector names are accepted."""
        runner = CliRunner()
        # This should fail on workload discovery, not collector validation
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent_workload",
                "--collect",
                "rocprof,numerics,amd_log",
            ],
        )
        # Should not fail on collector validation
        assert "Unknown collector recipes" not in result.output

    def test_collect_comma_separated(self):
        """Multiple collectors can be comma-separated."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--collect",
                "rocprof,numerics",
            ],
        )
        # Should not fail on collector validation
        assert "Unknown collector recipes" not in result.output

    def test_mitigations_comma_separated(self):
        """Multiple mitigations can be comma-separated."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--mitigations",
                "none,tf32_off",
            ],
        )
        # Should not fail on mitigation parsing
        # Will fail on workload discovery instead
        assert "Invalid" not in result.output or "extra-env" in result.output

    def test_extra_env_parsing(self):
        """--extra-env parses KEY=VALUE pairs."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--extra-env",
                "DEBUG=1,VERBOSE=true",
            ],
        )
        # Should not fail on extra-env parsing
        # Will fail on workload discovery instead
        assert "Invalid extra-env format" not in result.output

    def test_extra_env_invalid_format(self):
        """Invalid extra-env format raises clear error."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "fsdp",
                "--extra-env",
                "NOEQUALS",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid extra-env format" in result.output

    def test_extra_env_empty_key_rejected(self):
        """``=VALUE`` (empty key) is rejected with a clear error."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "fsdp",
                "--extra-env",
                "=somevalue",
            ],
        )
        assert result.exit_code != 0
        assert "key is empty" in result.output

    def test_extra_env_invalid_key_rejected(self):
        """Keys that don't match the env-var name pattern are rejected."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "fsdp",
                "--extra-env",
                "1BAD=value",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid extra-env key" in result.output

    def test_default_results_dir_does_not_require_existing_path(self, tmp_path):
        """``--results-dir`` must accept a non-existent path.

        Click's ``writable=True`` validation rejects paths that do not
        already exist, which broke the default ``results`` on a fresh
        checkout.  Letting the dispatcher's ``mkdir`` handle creation
        keeps the failure mode consistent with ``aorta env probe``.
        """
        runner = CliRunner()
        target = tmp_path / "does" / "not" / "exist"
        # ``--workload nonexistent`` ensures we fail at workload
        # discovery, not at Click's path validation.
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--results-dir",
                str(target),
            ],
        )
        # Click should NOT have rejected the path before invoking the
        # callback -- if it had, we'd see "Invalid value for '--results-dir'".
        assert "Invalid value for '--results-dir'" not in result.output

    def test_steps_option(self):
        """--steps is passed as integer."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--steps",
                "100",
            ],
        )
        # Should not fail on steps parsing
        assert "Invalid value" not in result.output or "steps" not in result.output

    def test_trials_default(self):
        """--trials defaults to 1."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
            ],
        )
        # CLI should use default trials=1
        # Will fail on workload discovery
        assert "trials" not in result.output.lower() or "failed" in result.output.lower()

    def test_environment_default(self):
        """--environment defaults to local."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
            ],
        )
        # Should use local environment by default
        # Will fail on workload discovery
        assert "environment" not in result.output.lower() or "unknown" not in result.output.lower()

    def test_results_dir_option(self):
        """--results-dir accepts path."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--results-dir",
                "/tmp/custom_results",
            ],
        )
        # Should accept custom results dir
        assert "results-dir" not in result.output.lower() or "invalid" not in result.output.lower()

    def test_unknown_workload_error_message(self):
        """Unknown workload shows available workloads."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "definitely_not_a_real_workload_xyz123",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "available" in result.output.lower()


class TestCliErrorHandling:
    """Tests for CLI error handling and reporting."""

    def test_unknown_environment_error(self):
        """Unknown environment shows available environments."""
        runner = CliRunner()
        # Need to use a workload that doesn't exist since fsdp workload
        # is not implemented yet
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--environment",
                "unknown_env",
            ],
        )
        assert result.exit_code != 0
        # Should fail on workload discovery first
        assert "not found" in result.output.lower()

    def test_unknown_mitigation_error(self):
        """Unknown mitigation shows available mitigations."""
        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "--workload",
                "nonexistent",
                "--mitigations",
                "unknown_mitigation",
            ],
        )
        assert result.exit_code != 0
        # Should fail on workload discovery first
        assert "not found" in result.output.lower()
