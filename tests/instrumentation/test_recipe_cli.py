"""CLI smoke tests for ``aorta env recipe`` (issue #163, A1.2c).

These run the Click command via ``CliRunner``, exercising the
``--format buck`` happy path, the ``--format dockerfile`` placeholder,
and the input-validation paths (missing file, invalid JSON, JSON that
isn't an object). Unit tests for the emitter itself live in
``test_buck_recipe.py``; this module only proves the CLI wiring
threads through cleanly.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from aorta.cli.env import env


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fixture_env_json(tmp_path):
    """A minimal env.json with one buck-introspected library."""
    env_dict = {
        "schema_version": "1.4",
        "build_system": {
            "kind": "buck2",
            "buck2_version": "buck2 2026-04-15",
            "repo_root": "/data/users/me/monorepo",
            "revision": "abc1234",
        },
        "library_introspection": [
            {
                "name": "hipblaslt",
                "source": "buck",
                "revision": "abc1234",
                "target": "//third-party/rocm:hipblaslt",
            },
        ],
        "library_introspection_alternates": [],
    }
    path = tmp_path / "env.json"
    path.write_text(json.dumps(env_dict))
    return path


class TestFormatBuck:
    def test_emits_to_stdout_with_header(self, runner, fixture_env_json):
        """Acceptance: `aorta env recipe --format buck <env.json>` emits
        text to stdout starting with the loud BEST-EFFORT header.
        """
        result = runner.invoke(
            env, ["recipe", str(fixture_env_json), "--format", "buck"]
        )
        assert result.exit_code == 0, result.output
        assert result.output.startswith("#")
        assert "BEST-EFFORT, NOT EXACT" in result.output

    def test_emits_one_rule_per_buck_entry(self, runner, fixture_env_json):
        result = runner.invoke(
            env, ["recipe", str(fixture_env_json), "--format", "buck"]
        )
        assert result.exit_code == 0, result.output
        assert result.output.count("prebuilt_cxx_library(") == 1
        assert 'name = "hipblaslt"' in result.output
        assert "//third-party/rocm:hipblaslt" in result.output
        assert 'version = "abc1234"' in result.output


class TestFormatDockerfilePlaceholder:
    def test_dockerfile_exits_with_not_implemented_message(
        self, runner, fixture_env_json
    ):
        """Acceptance: ``--format dockerfile`` exits with a clear "not
        yet implemented" error -- the surface is reserved but the
        emitter is out of scope for A1.2c.
        """
        result = runner.invoke(
            env, ["recipe", str(fixture_env_json), "--format", "dockerfile"]
        )
        assert result.exit_code != 0
        # The message should be obvious enough that the operator knows
        # to file a follow-up rather than think they typed something wrong.
        assert "not yet implemented" in result.output.lower()
        assert "dockerfile" in result.output.lower()


class TestInputValidation:
    def test_missing_file_yields_clean_click_error(self, runner, tmp_path):
        """A bad path is a Click-validated path arg, so the error
        message comes from Click itself (no Python traceback).
        """
        missing = tmp_path / "does-not-exist.json"
        result = runner.invoke(
            env, ["recipe", str(missing), "--format", "buck"]
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or \
               "no such" in result.output.lower()

    def test_invalid_json_surfaces_as_click_error(self, runner, tmp_path):
        """A file that exists but isn't valid JSON should not produce
        a Python traceback. The CLI wraps the JSON parse error in a
        ClickException.
        """
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not valid json")
        result = runner.invoke(env, ["recipe", str(bad), "--format", "buck"])
        assert result.exit_code != 0
        assert "failed to read" in result.output.lower()
        # No Python traceback should leak.
        assert "Traceback" not in result.output

    def test_json_array_rejected_with_clear_error(self, runner, tmp_path):
        """env.json is always a JSON object; an array, scalar, or null
        is rejected with a one-line error rather than crashing
        deeper in the emitter.
        """
        bad = tmp_path / "array.json"
        bad.write_text("[1, 2, 3]")
        result = runner.invoke(env, ["recipe", str(bad), "--format", "buck"])
        assert result.exit_code != 0
        assert "not a json object" in result.output.lower()

    def test_unknown_format_rejected_by_click(self, runner, fixture_env_json):
        """``--format`` is a ``click.Choice``; an unknown value should
        be rejected during arg parsing, before any of our code runs.
        """
        result = runner.invoke(
            env, ["recipe", str(fixture_env_json), "--format", "ninja"]
        )
        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or \
               "ninja" in result.output.lower()


class TestHelpSurface:
    def test_recipe_help_advertises_supported_formats(self, runner):
        """The CLI's --help output for `recipe` must mention `buck` and
        `dockerfile` so operators discover the surface without
        reading the docs.
        """
        result = runner.invoke(env, ["recipe", "--help"])
        assert result.exit_code == 0, result.output
        assert "buck" in result.output
        assert "dockerfile" in result.output

    def test_env_group_help_lists_recipe_subcommand(self, runner):
        """The top-level `aorta env --help` must list the new
        `recipe` subcommand alongside `probe`.
        """
        result = runner.invoke(env, ["--help"])
        assert result.exit_code == 0, result.output
        assert "recipe" in result.output
        assert "probe" in result.output
