"""CLI tests for the unified ``aorta sweep`` command (issue #248).

``aorta sweep`` merges ``aorta triage`` and ``aorta probe`` into one
front door. These tests pin:

* automatic flow dispatch (workload vs. subprocess) and the engine knobs
  each flow hands to ``run_recipe``;
* the consistency guards that reject mismatched inputs up-front;
* the ``--`` separator safety inherited from ``aorta probe``;
* the discovery subcommands; and
* that the deprecated ``triage`` / ``probe`` aliases still work but emit a
  stderr deprecation notice while reaching the same engine.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import aorta.cli.probe as probe_cli
import aorta.cli.triage as triage_cli
from aorta.cli import main
from aorta.cli.probe import probe
from aorta.cli.triage import triage
from aorta.triage.recipe import Recipe

PROBE_FIXTURES = Path(__file__).parent.parent / "probe" / "fixtures"
PROBE_MINIMAL = PROBE_FIXTURES / "probe_minimal.yaml"

# Click 8.2 removed ``CliRunner(mix_stderr=...)`` and now always captures stdout
# and stderr as separate streams (``result.stdout`` / ``result.stderr``, with
# ``result.output`` holding the combined text). On the ``click>=8.0`` floor we
# still need ``mix_stderr=False`` to get that same separation. Feature-detect the
# argument -- ``click.__version__`` is itself deprecated in 8.4+ -- so the tests
# that assert on stderr independently run identically on both.
_CLIRUNNER_ACCEPTS_MIX_STDERR = "mix_stderr" in inspect.signature(CliRunner.__init__).parameters


def _split_stream_runner() -> CliRunner:
    """A ``CliRunner`` that keeps stdout and stderr separate across click versions.

    Use this only when a test asserts on ``result.stderr`` / ``result.stdout``
    independently. Tests that assert on the combined ``result.output`` should use
    a plain ``CliRunner()``: on the ``click>=8.0`` floor a stream-separated runner
    drops stderr out of ``result.output``.
    """
    if _CLIRUNNER_ACCEPTS_MIX_STDERR:
        return CliRunner(mix_stderr=False)
    return CliRunner()


_TRIAGE_RECIPE_TEXT = (
    "schema_version: 1\n"
    "workload: fsdp\n"
    "trials: 1\n"
    "steps: 1\n"
    "cells:\n"
    "  - name: baseline-local\n"
    "    mitigations: [none]\n"
    "    environment: local\n"
)


@pytest.fixture
def mock_run_recipe(monkeypatch):
    """Patch ``run_recipe`` at both binding sites the flows reach.

    ``execute_triage_run`` and ``execute_probe`` live in the ``triage`` /
    ``probe`` modules and reference each module's own ``run_recipe`` global;
    ``aorta sweep`` calls those functions, so patching both modules covers
    every dispatch path.
    """
    mock = MagicMock(return_value=Path("/tmp/sweep-mock-run-dir"))
    monkeypatch.setattr(probe_cli, "run_recipe", mock)
    monkeypatch.setattr(triage_cli, "run_recipe", mock)
    return mock


def _write_triage_recipe(tmp_path: Path) -> Path:
    path = tmp_path / "triage.yaml"
    path.write_text(_TRIAGE_RECIPE_TEXT, encoding="utf-8")
    return path


# --- flow dispatch -------------------------------------------------------


def test_trailing_command_routes_to_probe_flow(mock_run_recipe, tmp_path):
    """`aorta sweep run --recipe <probe> -- echo hi` runs the subprocess flow."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sweep",
            "run",
            "--recipe",
            str(PROBE_MINIMAL),
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_run_recipe.assert_called_once()
    args, kwargs = mock_run_recipe.call_args
    recipe_arg = args[0] if args else kwargs.get("recipe")
    assert isinstance(recipe_arg, Recipe)
    assert recipe_arg.probe_extras is not None
    assert kwargs.get("layout") == "flat_resume"
    assert kwargs.get("resume_existing") is True
    assert kwargs.get("subprocess_argv") == ("echo", "hi")


def test_empty_collect_csv_is_ignored_in_probe_flow(mock_run_recipe, tmp_path):
    """An effectively empty ``--collect`` value should not block subprocess flow."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sweep",
            "run",
            "--recipe",
            str(PROBE_MINIMAL),
            "--collect",
            " , , ",
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_run_recipe.assert_called_once()


def test_probe_recipe_without_command_routes_to_probe_flow_guard(mock_run_recipe):
    """A probe-mode recipe with no trailing command is a clear usage error."""
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "run", "--recipe", str(PROBE_MINIMAL)])
    assert result.exit_code != 0
    assert "requires a user command after '--'" in result.output
    mock_run_recipe.assert_not_called()


def test_triage_recipe_routes_to_workload_flow(mock_run_recipe, tmp_path):
    """`aorta sweep run --recipe <triage>` runs the in-process workload flow."""
    recipe = _write_triage_recipe(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["sweep", "run", "--recipe", str(recipe), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    mock_run_recipe.assert_called_once()
    args, kwargs = mock_run_recipe.call_args
    recipe_arg = args[0] if args else kwargs.get("recipe")
    assert isinstance(recipe_arg, Recipe)
    assert recipe_arg.probe_extras is None  # workload flow never sets this
    assert kwargs.get("layout", "timestamped") == "timestamped"
    assert kwargs.get("subprocess_argv") is None


def test_strict_threads_to_workload_flow(mock_run_recipe, tmp_path):
    """`--strict` reaches run_recipe as strict=True on the workload flow."""
    recipe = _write_triage_recipe(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["sweep", "run", "--recipe", str(recipe), "--output", str(tmp_path / "out"), "--strict"],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = mock_run_recipe.call_args
    assert kwargs.get("strict") is True


def test_strict_defaults_false_in_workload_flow(mock_run_recipe, tmp_path):
    """Without `--strict`, run_recipe is called with strict=False (opt-in)."""
    recipe = _write_triage_recipe(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["sweep", "run", "--recipe", str(recipe), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    _, kwargs = mock_run_recipe.call_args
    assert kwargs.get("strict") is False


def test_strict_rejected_in_probe_flow(mock_run_recipe, tmp_path):
    """`--strict` is a workload-flow flag; combining it with a probe run errors."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sweep",
            "run",
            "--recipe",
            str(PROBE_MINIMAL),
            "--output",
            str(tmp_path / "out"),
            "--strict",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "--strict applies to the workload flow only" in result.output
    mock_run_recipe.assert_not_called()


def test_flag_mode_routes_to_workload_flow(mock_run_recipe, tmp_path):
    """Flag-shim (no recipe, no command) runs the workload flow."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sweep",
            "run",
            "--workload",
            "fsdp",
            "--mitigation-axis",
            "none",
            "--environment-axis",
            "local",
            "--trials",
            "1",
            "--steps",
            "1",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    mock_run_recipe.assert_called_once()
    args, kwargs = mock_run_recipe.call_args
    recipe_arg = args[0] if args else kwargs.get("recipe")
    assert recipe_arg.probe_extras is None


def test_default_output_dir_per_flow(mock_run_recipe, tmp_path):
    """`--output` default resolves to probe_results / triage_results per flow."""
    runner = CliRunner()
    r1 = runner.invoke(main, ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--", "echo", "hi"])
    assert r1.exit_code == 0, r1.output
    _, kwargs = mock_run_recipe.call_args
    assert kwargs.get("output_dir") == Path("probe_results")

    mock_run_recipe.reset_mock()
    recipe = _write_triage_recipe(tmp_path)
    r2 = runner.invoke(main, ["sweep", "run", "--recipe", str(recipe)])
    assert r2.exit_code == 0, r2.output
    _, kwargs = mock_run_recipe.call_args
    assert kwargs.get("output_dir") == Path("triage_results")


# --- consistency guards --------------------------------------------------


def test_command_with_triage_recipe_rejected(mock_run_recipe, tmp_path):
    """A trailing command with a workload recipe is rejected."""
    recipe = _write_triage_recipe(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "run", "--recipe", str(recipe), "--", "echo", "hi"])
    assert result.exit_code != 0
    assert "only valid for a probe-mode run" in result.output
    mock_run_recipe.assert_not_called()


def test_env_passthrough_rejected_in_workload_flow(mock_run_recipe, tmp_path):
    """`--env-passthrough-mode` is meaningless without a subprocess command."""
    recipe = _write_triage_recipe(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["sweep", "run", "--recipe", str(recipe), "--env-passthrough-mode", "file"],
    )
    assert result.exit_code != 0
    assert "probe flow only" in result.output
    mock_run_recipe.assert_not_called()


def test_triage_only_flag_rejected_in_probe_flow(mock_run_recipe):
    """Workload-only knobs cannot be combined with a subprocess command."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--trials", "3", "--", "echo", "hi"],
    )
    assert result.exit_code != 0
    assert "only apply to the workload flow" in result.output
    mock_run_recipe.assert_not_called()


@pytest.mark.parametrize(
    "extra_flags",
    [
        ["--stop-after-events", "3"],
        ["--max-trials", "20"],
        ["--disable-detector", "tier3"],
    ],
)
def test_probe_only_flag_rejected_in_workload_flow(mock_run_recipe, tmp_path, extra_flags):
    """Probe-only knobs have no effect on a workload run and are rejected up-front."""
    recipe = _write_triage_recipe(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "run", "--recipe", str(recipe), *extra_flags])
    assert result.exit_code != 0
    assert "only apply to the probe/subprocess flow" in result.output
    assert extra_flags[0] in result.output
    mock_run_recipe.assert_not_called()


def test_bare_positional_without_separator_rejected(mock_run_recipe):
    """A bare positional with no `--` is refused (inherited probe safety)."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "SMOKE-1", "echo", "hi"]
    )
    assert result.exit_code != 0
    assert "missing '--' separator" in result.output
    mock_run_recipe.assert_not_called()


def test_empty_argv_after_separator_rejected(mock_run_recipe):
    """`-- ` with nothing after it exits non-zero."""
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--"])
    assert result.exit_code != 0
    mock_run_recipe.assert_not_called()


def test_leaked_dash_flag_without_separator_names_sweep(mock_run_recipe):
    """A dash-prefixed leaked token (no `--`) errors naming `aorta sweep run`, not `aorta probe`.

    `_bare_positional_before_separator` skips dash tokens, so a forgotten
    `--` with a flag-shaped first token (`-c`) slips past the sweep guard
    into `execute_probe`'s `validate_trailing_argv`. That usage error must
    point the user at the front door they actually invoked.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "-c"])
    assert result.exit_code != 0
    assert "looks like a flag" in result.output
    assert "aorta sweep run" in result.output
    assert "aorta probe" not in result.output
    mock_run_recipe.assert_not_called()


def test_dash_command_after_separator_names_sweep(mock_run_recipe):
    """Same class with `--` present: a dash-shaped user command names the sweep front door."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--", "-c"]
    )
    assert result.exit_code != 0
    assert "looks like a flag" in result.output
    assert "aorta sweep run" in result.output
    assert "aorta probe" not in result.output
    mock_run_recipe.assert_not_called()


# --- discovery subcommands -----------------------------------------------


def test_list_mitigations_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "list-mitigations"])
    assert result.exit_code == 0, result.output
    assert "NAME" in result.output and "SOURCE" in result.output


def test_list_environments_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "list-environments"])
    assert result.exit_code == 0, result.output
    assert "NAME" in result.output and "DOCKER" in result.output


def test_list_patterns_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "list-patterns"])
    assert result.exit_code == 0, result.output
    # Banner names the command the user actually ran, not "aorta probe".
    assert "aorta sweep built-in pattern library" in result.output
    assert "aorta probe" not in result.output


def test_list_patterns_version_banner():
    runner = CliRunner()
    result = runner.invoke(main, ["sweep", "list-patterns", "--version"])
    assert result.exit_code == 0, result.output
    assert "pattern library v" in result.output
    assert result.output.startswith("aorta sweep pattern library v")


# --- _peek_recipe_mode dispatch helper -----------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("mode: probe\n", "probe"),
        ("mode: triage\n", "triage"),
        ("workload: fsdp\n", "triage"),  # mode absent -> triage default
        ("mode: prboe\n", None),  # typo: defer to the real loader, don't guess
        ("mode: null\n", None),
        ("- just\n- a\n- list\n", None),  # not a mapping
    ],
)
def test_peek_recipe_mode(tmp_path, body, expected):
    from aorta.cli.sweep import _peek_recipe_mode

    p = tmp_path / "r.yaml"
    p.write_text("schema_version: 1\n" + body, encoding="utf-8")
    assert _peek_recipe_mode(p) == expected


# --- deprecation aliases -------------------------------------------------


def test_triage_run_warns_and_delegates(mock_run_recipe, tmp_path):
    """`aorta triage run` still works but prints a stderr deprecation notice."""
    recipe = _write_triage_recipe(tmp_path)
    runner = _split_stream_runner()
    result = runner.invoke(
        triage, ["run", "--recipe", str(recipe), "--output-dir", str(tmp_path / "o")]
    )
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.stderr
    assert "aorta sweep run" in result.stderr
    assert "deprecated" not in result.stdout  # notice must not pollute stdout
    mock_run_recipe.assert_called_once()


def test_probe_warns_and_delegates(mock_run_recipe, tmp_path):
    """`aorta probe` still works but prints a stderr deprecation notice."""
    runner = _split_stream_runner()
    result = runner.invoke(
        probe,
        ["--recipe", str(PROBE_MINIMAL), "--output", str(tmp_path / "o"), "--", "echo", "hi"],
    )
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.stderr
    assert "aorta sweep run" in result.stderr
    mock_run_recipe.assert_called_once()


def test_triage_list_mitigations_warns():
    runner = _split_stream_runner()
    result = runner.invoke(triage, ["list-mitigations"])
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.stderr
    assert "aorta sweep list-mitigations" in result.stderr


def test_probe_list_patterns_points_at_sweep_list_patterns():
    """`aorta probe --list-patterns` warns toward `aorta sweep list-patterns`, not `run`."""
    runner = _split_stream_runner()
    result = runner.invoke(probe, ["--list-patterns"])
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.stderr
    assert "aorta sweep list-patterns" in result.stderr
    # The generic `aorta sweep run` target is wrong for this surface.
    assert "aorta sweep run" not in result.stderr
    # Notice stays on stderr; stdout remains the catalogue.
    assert "deprecated" not in result.stdout
    assert "pattern" in result.stdout.lower()


def test_probe_list_patterns_version_points_at_sweep_list_patterns():
    """`aorta probe --list-patterns --version` keeps the notice on stderr, stdout parseable."""
    runner = _split_stream_runner()
    result = runner.invoke(probe, ["--list-patterns", "--version"])
    assert result.exit_code == 0, result.output
    assert "aorta sweep list-patterns" in result.stderr
    assert "aorta sweep run" not in result.stderr
    assert "deprecated" not in result.stdout
    assert result.stdout.startswith("aorta probe pattern library v")


# --- cross-front-door parity --------------------------------------------
#
# The PR's headline guarantee is that ``aorta sweep run`` is byte-identical
# to the legacy ``aorta triage run`` / ``aorta probe`` aliases because every
# front door delegates to the same shared engine. The tests above pin each
# front door's ``run_recipe`` call against hard-coded expectations *in
# isolation*; that lets the two paths drift apart while both still pass (a
# future edit could change one dispatcher's engine kwargs without the other).
# These tests pin the calls to be *equal* for identical input, so a
# divergence between front doors fails loudly instead of silently. They are
# the ``aorta sweep`` extension of ``tests/probe/test_shared_engine.py``,
# which only ever covered the original probe<->triage pair.


def _capture_engine_call(mock, invoke):
    """Run one CLI invocation; return ``(recipe, kwargs_without_recipe)``.

    Asserts the invocation succeeded and reached ``run_recipe`` exactly
    once. ``recipe`` is normalised out of ``args``/``kwargs`` so the
    comparison doesn't depend on whether a flow passes it positionally.
    """
    mock.reset_mock()
    result = invoke()
    assert result.exit_code == 0, result.output
    mock.assert_called_once()
    args, kwargs = mock.call_args
    kwargs = dict(kwargs)
    recipe_arg = args[0] if args else kwargs.pop("recipe", None)
    return recipe_arg, kwargs


def test_workload_flow_parity_sweep_vs_triage(mock_run_recipe, tmp_path):
    """`aorta sweep run` reaches run_recipe identically to `aorta triage run`."""
    recipe = _write_triage_recipe(tmp_path)
    out = str(tmp_path / "out")
    runner = CliRunner()

    sweep_recipe, sweep_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(main, ["sweep", "run", "--recipe", str(recipe), "--output", out]),
    )
    triage_recipe, triage_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(triage, ["run", "--recipe", str(recipe), "--output-dir", out]),
    )

    assert sweep_recipe == triage_recipe
    assert sweep_kwargs == triage_kwargs


def test_subprocess_flow_parity_sweep_vs_probe(mock_run_recipe, tmp_path):
    """`aorta sweep run -- <cmd>` reaches run_recipe identically to `aorta probe -- <cmd>`."""
    out = str(tmp_path / "out")
    runner = CliRunner()

    sweep_recipe, sweep_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(
            main,
            ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--output", out, "--", "echo", "hi"],
        ),
    )
    probe_recipe, probe_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(
            probe,
            ["--recipe", str(PROBE_MINIMAL), "--output", out, "--", "echo", "hi"],
        ),
    )

    assert sweep_recipe == probe_recipe
    assert sweep_kwargs == probe_kwargs


def test_subprocess_flow_parity_sweep_vs_probe_with_stop_after(mock_run_recipe, tmp_path):
    """`--stop-after-events`/`--max-trials` reach run_recipe identically across front doors."""
    out = str(tmp_path / "out")
    flags = ["--stop-after-events", "3", "--max-trials", "20"]
    runner = CliRunner()

    sweep_recipe, sweep_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(
            main,
            ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--output", out, *flags, "--", "echo", "hi"],
        ),
    )
    probe_recipe, probe_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(
            probe,
            ["--recipe", str(PROBE_MINIMAL), "--output", out, *flags, "--", "echo", "hi"],
        ),
    )

    assert sweep_recipe == probe_recipe
    assert sweep_kwargs == probe_kwargs


def test_subprocess_flow_parity_sweep_vs_probe_with_disable_detector(mock_run_recipe, tmp_path):
    """`--disable-detector` (repeatable) reaches run_recipe identically across front doors."""
    out = str(tmp_path / "out")
    flags = ["--disable-detector", "tier3", "--disable-detector", "tier2:hang"]
    runner = CliRunner()

    sweep_recipe, sweep_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(
            main,
            ["sweep", "run", "--recipe", str(PROBE_MINIMAL), "--output", out, *flags, "--", "echo", "hi"],
        ),
    )
    probe_recipe, probe_kwargs = _capture_engine_call(
        mock_run_recipe,
        lambda: runner.invoke(
            probe,
            ["--recipe", str(PROBE_MINIMAL), "--output", out, *flags, "--", "echo", "hi"],
        ),
    )

    assert sweep_recipe == probe_recipe
    assert sweep_kwargs == probe_kwargs
