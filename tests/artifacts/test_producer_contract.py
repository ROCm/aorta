"""Contract tests binding the readers to aorta's own producers.

Two things are pinned here that the unit tests cannot see:

* the readers parse what ``write_matrix_json`` / the env probe actually
  emit today, rather than what a hand-written fixture claims they emit;
* ``aorta.artifacts`` stays importable from a base install -- it is placed
  in core precisely so a consumer that installs no extras can use it, and
  a drive-by third-party import would quietly undo that.

This is deliberately *not* a golden key-set guard over ``matrix.json``.
That is a separate, wider piece of work tracked on its own.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import aorta.artifacts as artifacts_pkg
from aorta.artifacts import (
    ENV_SCHEMA_MAJOR,
    MATRIX_SCHEMA_VERSION,
    read_matrix,
)
from aorta.artifacts._common import classify_dotted_schema
from aorta.instrumentation.environment import SCHEMA_VERSION as ENV_PRODUCER_SCHEMA_VERSION
from aorta.triage.matrix import aggregate_cell
from aorta.triage.output import write_matrix_json
from aorta.triage.recipe import build_recipe_from_flags


@dataclass
class _Trial:
    exit_status: str
    wall_clock_sec: float
    result: dict


def _trial(passed: bool, hint: str | None = None) -> _Trial:
    result: dict = {"passed": passed, "step_times_ms": [100.0]}
    if hint is not None:
        result["failure_details"] = [{"hint": hint}]
    return _Trial(
        exit_status="ok" if passed else "workload_failed",
        wall_clock_sec=1.0,
        result=result,
    )


def _write_real_matrix(path: Path) -> None:
    recipe = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local",
        trials=4,
        steps=10,
        ticket="PROJ-1",
    )
    hint = "residual NaN in Shampoo preconditioner"
    repro = aggregate_cell(
        name="none-local",
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=[_trial(True), _trial(False, hint), _trial(False, hint), _trial(False, hint)],
        effective_steps=10,
    )
    clean = aggregate_cell(
        name="tf32_off-local",
        mitigations=("tf32_off",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=[_trial(True) for _ in range(4)],
        effective_steps=10,
    )
    write_matrix_json(
        path,
        recipe,
        [repro, clean],
        baseline_name="none-local",
        confound_tags={},
        run_timestamp="2026-01-01T00-00-00",
        warnings=[],
    )


def test_reader_parses_what_write_matrix_json_emits(tmp_path: Path):
    path = tmp_path / "matrix.json"
    _write_real_matrix(path)

    matrix = read_matrix(path)

    assert matrix.schema_status == "supported"
    assert matrix.missing_fields == ()
    assert matrix.missing_cell_fields() == {}
    assert matrix.workload == "fsdp"
    assert matrix.ticket == "PROJ-1"
    assert matrix.steps_per_trial == 10
    assert matrix.trials_per_cell == 4

    repro = matrix.cell("none-local")
    assert repro is not None
    assert repro.mitigations == ("none",)
    assert repro.trials == 4
    assert repro.passed_count == 1
    assert repro.failed_count == 3
    assert repro.failure_rate == 0.75
    assert repro.workload_failed_count == 3
    assert repro.error is None
    # The producer emits ``[text, trials]`` pairs, not bare strings.
    assert [(h.text, h.trials) for h in repro.failure_hints] == [
        ("residual NaN in Shampoo preconditioner", 3)
    ]

    clean = matrix.cell("tf32_off-local")
    assert clean is not None
    assert clean.failure_rate == 0.0
    assert clean.failed_count == 0
    assert clean.failure_hints == ()


def test_matrix_schema_constant_tracks_the_producer(tmp_path: Path):
    """A bump in ``write_matrix_json`` should land here as a failing test.

    ``matrix.json`` has no versioning policy behind its hardcoded integer, so
    unlike env.json a change cannot be assumed additive -- someone has to look.
    """
    path = tmp_path / "matrix.json"
    _write_real_matrix(path)

    emitted = json.loads(path.read_text(encoding="utf-8"))["schema_version"]

    assert emitted == MATRIX_SCHEMA_VERSION


def test_env_reader_accepts_the_probes_current_schema_version():
    """Minor bumps stay ``supported``; only a major bump needs attention."""
    status, note = classify_dotted_schema(ENV_PRODUCER_SCHEMA_VERSION, ENV_SCHEMA_MAJOR)

    assert (status, note) == ("supported", None)


# ---- dependency discipline ------------------------------------------------


def _artifact_sources() -> list[Path]:
    pkg_dir = Path(artifacts_pkg.__file__).parent
    return [p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts]


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or not node.module:
                continue
            names.add(node.module.split(".", 1)[0])
    return names


def test_artifacts_package_imports_only_the_stdlib():
    """No third-party dependency, ever.

    The external consumer this package is placed in core for installs
    ``amd-aorta`` with no extras; anything beyond the stdlib here would make
    ``import aorta.artifacts`` a dependency negotiation.
    """
    assert _artifact_sources(), "expected to find aorta.artifacts source files"
    allowed = set(sys.stdlib_module_names) | {"aorta"}
    for source in _artifact_sources():
        offenders = _top_level_imports(source) - allowed
        assert not offenders, f"{source.name} imports non-stdlib {sorted(offenders)}"


def test_artifacts_package_imports_nothing_else_from_aorta():
    """Readers describe the on-disk shape; they must not pull in the producers.

    Importing ``aorta.triage`` or ``aorta.instrumentation`` here would drag a
    large module graph into every consumer and couple the reader to code that
    only the writing side needs.
    """
    for source in _artifact_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = next(
                    (a.name for a in node.names if a.name.split(".", 1)[0] == "aorta"), None
                )
            if module and module.split(".", 1)[0] == "aorta":
                assert module.startswith("aorta.artifacts"), (
                    f"{source.name} imports {module}; aorta.artifacts must stay "
                    "self-contained within the package"
                )
