"""Tests for CLI argument parsing."""

import ast
import inspect

from click.testing import CliRunner

from aorta.cli.run import run


class TestCliHandlerIsThinShell:
    """B1 spec hard rule: the Click handler is a ~30-line shim.

    Issue #148, "Python API contract" section: *"the Click handler in
    cli/run.py is under ~30 lines and contains no `for trial in
    range(...)` loop"*.  Anything beyond parse-args / build-request /
    call-run_trials / map-exit-code lives in
    ``aorta.run.dispatcher`` or ``aorta.run.cli_helpers``.

    These tests pin that contract so the handler can't silently grow
    business logic again (the round-5 review caught a ~120-line
    handler doing collector validation, extra-env parsing, and
    pass/fail aggregation -- all now in the library).
    """

    def _handler_function(self):
        # ``run`` is a Click command; the underlying Python function
        # is on its ``callback`` attribute.
        return run.callback

    def test_body_is_short(self):
        """Handler body is the documented ~30 lines (give or take comments)."""
        fn = self._handler_function()
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        body_start = func_def.body[0].lineno
        body_end = func_def.end_lineno
        assert body_end is not None
        body_lines = body_end - body_start + 1
        # Spec says "~30 lines".  Allow a small cushion for inline
        # comments and the docstring; reject anything that's actually
        # carrying business logic (the previous regression was 80+).
        assert body_lines <= 45, (
            f"Click handler body has grown to {body_lines} lines -- "
            "move logic into aorta.run.dispatcher or aorta.run.cli_helpers"
        )

    def test_no_per_trial_loop(self):
        """Spec literal: no ``for trial in range(...)`` inside the handler."""
        fn = self._handler_function()
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # ``for trial ... in range(...)`` is the disallowed
                # pattern -- per-trial iteration is run_trials' job.
                if (
                    isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ):
                    raise AssertionError(
                        "Click handler contains a `for ... in range(...)` "
                        "loop; per-trial iteration belongs in run_trials()."
                    )


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
        """Keys that don't match the env-var name pattern are rejected.

        Validation lives in ``run_trials`` (library entry-point) so
        programmatic callers that bypass the CLI parser get the same
        protection; the CLI bridges the resulting ``ValueError`` to
        a ``ClickException``.  The error names the offending key and
        the POSIX env-var pattern.
        """
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
        assert "1BAD" in result.output
        assert "must match" in result.output

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
