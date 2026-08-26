"""The probe-mode ``collect:`` channel and the ``--collect`` CLI overlay.

Collectors are cross-cutting rather than a matrix axis, so ``collect`` is valid
in both recipe modes (like ``stop_after``). Probe mode synthesises its own
cells, so the recipe-level names/options apply to every ``(mitigation,
diagnostic)`` pair and there is no cell scope to override at.

``--collect`` REPLACES the recipe's list, matching the workload flow's
precedence -- these tests pin that it is a replacement and not a merge, and
that the surviving option set is re-validated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aorta.cli.sweep import sweep_run
from aorta.probe.cli_helpers import ProbeUsageError, apply_recipe_overrides
from aorta.triage.recipe import RecipeSchemaError, load_recipe

_PROBE_HEAD = (
    "schema_version: 1\n"
    "mode: probe\n"
    "trials: 1\n"
    "mitigation_axis: [none]\n"
    "diagnostic_axis: [none]\n"
)


def _write(tmp_path: Path, body: str = "") -> Path:
    path = tmp_path / "r.yaml"
    path.write_text(_PROBE_HEAD + body, encoding="utf-8")
    return path


def _probe_recipe(tmp_path: Path, body: str = ""):
    return load_recipe(_write(tmp_path, body))


# ---- Recipe-scope acceptance in probe mode ------------------------------


def test_probe_recipe_defaults_to_no_collector(tmp_path):
    recipe = _probe_recipe(tmp_path)
    assert recipe.collect == ()
    assert recipe.collect_options == {}


def test_probe_recipe_accepts_list_form(tmp_path):
    recipe = _probe_recipe(tmp_path, "collect: [rocprof]\n")
    assert recipe.collect == ("rocprof",)
    assert recipe.collect_options == {}


def test_probe_recipe_accepts_mapping_form_with_options(tmp_path):
    recipe = _probe_recipe(
        tmp_path,
        'collect:\n  rocprof:\n    trace: "kernel,hip"\n    summary_units: "msec"\n',
    )
    assert recipe.collect == ("rocprof",)
    assert recipe.collect_options == {"rocprof": {"trace": "kernel,hip", "summary_units": "msec"}}


def test_probe_recipe_accepts_proton(tmp_path):
    recipe = _probe_recipe(tmp_path, 'collect:\n  proton:\n    backend: "roctracer"\n')
    assert recipe.collect == ("proton",)
    assert recipe.collect_options == {"proton": {"backend": "roctracer"}}


def test_probe_recipe_rejects_unknown_collector(tmp_path):
    with pytest.raises(RecipeSchemaError, match="unknown collector recipe"):
        _probe_recipe(tmp_path, "collect: [not_a_collector]\n")


def test_probe_recipe_rejects_a_bad_option(tmp_path):
    with pytest.raises(RecipeSchemaError, match="rocprof: unknown option"):
        _probe_recipe(tmp_path, 'collect:\n  rocprof:\n    traces: "kernel"\n')


def test_probe_recipe_rejects_the_queue_interception_conflict(tmp_path):
    with pytest.raises(RecipeSchemaError, match="queue interceptor"):
        _probe_recipe(tmp_path, "collect: [rocprof, proton]\n")


def test_probe_recipe_allows_the_instrumentation_backend_alongside_rocprof(tmp_path):
    recipe = _probe_recipe(
        tmp_path,
        'collect:\n  rocprof:\n  proton:\n    backend: "instrumentation"\n',
    )
    assert recipe.collect == ("rocprof", "proton")


def test_probe_recipe_error_is_labelled_recipe_collect(tmp_path):
    with pytest.raises(RecipeSchemaError, match=r"^recipe\.collect:"):
        _probe_recipe(tmp_path, "collect: [rocprof, proton]\n")


def test_probe_cells_inherit_the_recipe_collect(tmp_path):
    """Probe mode synthesises its cells without a cell-scope ``collect``, so
    replacing the recipe-level tuple is enough to reach every cell."""
    recipe = _probe_recipe(tmp_path, "collect: [rocprof]\n")
    for cell in recipe.cells:
        assert cell.collect is None
        assert cell.effective_collect(recipe.collect) == ("rocprof",)


# ---- --collect overlay: replace, not merge -----------------------------


def test_cli_collect_replaces_the_recipe_list(tmp_path):
    recipe = _probe_recipe(tmp_path, "collect: [rocprof]\n")
    out = apply_recipe_overrides(
        recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("layer_numerics",)
    )
    assert out.collect == ("layer_numerics",)
    assert "rocprof" not in out.collect


def test_cli_collect_absent_leaves_the_recipe_list_intact(tmp_path):
    recipe = _probe_recipe(tmp_path, "collect: [rocprof]\n")
    out = apply_recipe_overrides(recipe, ticket=None, cli_passthrough_mode=None)
    assert out.collect == ("rocprof",)


def test_cli_collect_adds_a_collector_to_a_recipe_with_none(tmp_path):
    recipe = _probe_recipe(tmp_path)
    out = apply_recipe_overrides(
        recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("rocprof",)
    )
    assert out.collect == ("rocprof",)


def test_cli_collect_drops_options_for_collectors_it_replaced(tmp_path):
    """Per-collector options are recipe-file-only, so options belonging to a
    collector the CLI dropped must not survive as orphans."""
    recipe = _probe_recipe(tmp_path, 'collect:\n  rocprof:\n    trace: "kernel,hip"\n')
    out = apply_recipe_overrides(
        recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("layer_numerics",)
    )
    assert out.collect_options == {}


def test_cli_collect_keeps_options_for_collectors_it_kept(tmp_path):
    recipe = _probe_recipe(
        tmp_path,
        'collect:\n  rocprof:\n    trace: "kernel,hip"\n  layer_numerics:\n    NANLOG_SPEC: "{}"\n',
    )
    out = apply_recipe_overrides(
        recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("rocprof",)
    )
    assert out.collect == ("rocprof",)
    assert out.collect_options == {"rocprof": {"trace": "kernel,hip"}}


def test_cli_collect_dedups_preserving_order(tmp_path):
    recipe = _probe_recipe(tmp_path)
    out = apply_recipe_overrides(
        recipe,
        ticket=None,
        cli_passthrough_mode=None,
        cli_collect=("layer_numerics", "rocprof", "layer_numerics"),
    )
    assert out.collect == ("layer_numerics", "rocprof")


def test_cli_collect_rejects_an_unknown_collector(tmp_path):
    recipe = _probe_recipe(tmp_path)
    with pytest.raises(ProbeUsageError, match="unknown collector recipe"):
        apply_recipe_overrides(
            recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("not_a_collector",)
        )


def test_cli_collect_error_is_labelled_with_the_flag(tmp_path):
    recipe = _probe_recipe(tmp_path)
    with pytest.raises(ProbeUsageError, match=r"^--collect:"):
        apply_recipe_overrides(
            recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("not_a_collector",)
        )


def test_cli_collect_rejects_a_conflicting_pair(tmp_path):
    recipe = _probe_recipe(tmp_path)
    with pytest.raises(ProbeUsageError, match="queue interceptor"):
        apply_recipe_overrides(
            recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("rocprof", "proton")
        )


def test_cli_collect_revalidates_the_surviving_recipe_options(tmp_path):
    """Narrowing the list can resolve a conflict -- but it can also leave a kept
    collector paired with a partner the recipe author never validated against.

    Here the recipe pins a Proton backend that is fine on its own; adding
    ``rocprof`` from the CLI makes the surviving pair unrunnable, and that has
    to be caught even though the recipe itself loaded clean.
    """
    recipe = _probe_recipe(tmp_path, 'collect:\n  proton:\n    backend: "roctracer"\n')
    with pytest.raises(ProbeUsageError, match="queue interceptor"):
        apply_recipe_overrides(
            recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("rocprof", "proton")
        )


def test_cli_collect_restating_a_legal_recipe_pair_is_accepted(tmp_path):
    """Validating the bare name list would invent a conflict the recipe resolves.

    ``rocprof`` + ``proton`` is only unrunnable on a queue-intercepting Proton
    backend. This recipe pins the instrumentation backend, so restating both
    names on the CLI is a no-op and must not be rejected against Proton's
    *default* backend.
    """
    recipe = _probe_recipe(
        tmp_path,
        'collect:\n  rocprof:\n  proton:\n    backend: "instrumentation"\n',
    )
    out = apply_recipe_overrides(
        recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("rocprof", "proton")
    )
    assert out.collect == ("rocprof", "proton")
    assert out.collect_options == {"proton": {"backend": "instrumentation"}}


def test_cli_collect_narrowing_can_resolve_a_conflict(tmp_path):
    recipe = _probe_recipe(
        tmp_path,
        'collect:\n  rocprof:\n  proton:\n    backend: "instrumentation"\n',
    )
    out = apply_recipe_overrides(
        recipe, ticket=None, cli_passthrough_mode=None, cli_collect=("proton",)
    )
    assert out.collect == ("proton",)
    assert out.collect_options == {"proton": {"backend": "instrumentation"}}


# ---- CLI surface -------------------------------------------------------


def test_sweep_run_accepts_collect_on_the_probe_flow(tmp_path):
    """``--collect`` used to be rejected outright on the probe flow. It is now
    the whole point: a collector attaches by wrapping the user command's argv,
    so profiling an opaque ``-- <command>`` is exactly what it is for."""
    result = CliRunner().invoke(
        sweep_run,
        [
            "--recipe",
            str(_write(tmp_path)),
            "--output",
            str(tmp_path / "out"),
            "--collect",
            "rocprof",
            "--dry-run",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "applies to the workload flow only" not in result.output


def test_sweep_run_rejects_an_unknown_collector_on_the_probe_flow(tmp_path):
    result = CliRunner().invoke(
        sweep_run,
        [
            "--recipe",
            str(_write(tmp_path)),
            "--output",
            str(tmp_path / "out"),
            "--collect",
            "not_a_collector",
            "--dry-run",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "unknown collector recipe" in result.output


def test_sweep_run_rejects_the_conflicting_pair_on_the_probe_flow(tmp_path):
    result = CliRunner().invoke(
        sweep_run,
        [
            "--recipe",
            str(_write(tmp_path)),
            "--output",
            str(tmp_path / "out"),
            "--collect",
            "rocprof,proton",
            "--dry-run",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "queue interceptor" in result.output
