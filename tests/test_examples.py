"""Integrity of the shipped ``examples/`` tree.

The examples double as end-to-end tests of the collectors, so they have to
stay loadable and self-consistent: every example directory carries a README, a
payload, and a recipe that loads under the real loader with a real collector
attached. Broken example recipes are worse than no examples -- they are the
first thing a new user runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aorta.instrumentation.proton import ENV_PREFIX, build_env
from aorta.triage.recipe import load_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
PROFILING = EXAMPLES / "profiling"

#: Sidecar templates that predate the categorised tree; both are referenced
#: from ``src/aorta/registry/README.md`` and ``docs/buck2-build-reference.md``.
_SIDECAR_TEMPLATES = ("mitigations-sidecar.json", "probe-flag-sidecar.json")


def _example_dirs() -> list[Path]:
    """Every leaf example directory: ``profiling/<collector>/<example>/``."""
    if not PROFILING.is_dir():
        return []
    return sorted(
        child
        for category in PROFILING.iterdir()
        if category.is_dir()
        for child in category.iterdir()
        if child.is_dir()
    )


def _category_dirs() -> list[Path]:
    return sorted(child for child in PROFILING.iterdir() if child.is_dir())


_EXAMPLE_DIRS = _example_dirs()
_EXAMPLE_IDS = [str(path.relative_to(EXAMPLES)) for path in _EXAMPLE_DIRS]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# ---- Tree shape ---------------------------------------------------------


def test_examples_tree_exists():
    assert EXAMPLES.is_dir()
    assert PROFILING.is_dir()


def test_at_least_one_example_per_collector_category():
    categories = {path.name for path in _category_dirs()}
    assert {"rocprof", "proton"} <= categories
    for category in _category_dirs():
        assert [c for c in category.iterdir() if c.is_dir()], f"{_rel(category)} has no examples"


def test_sidecar_templates_are_still_in_place():
    """Both are linked from docs that would break if they moved."""
    for name in _SIDECAR_TEMPLATES:
        assert (EXAMPLES / name).is_file()


def test_index_readmes_exist():
    assert (EXAMPLES / "README.md").is_file()
    assert (PROFILING / "README.md").is_file()


@pytest.mark.parametrize("category", _category_dirs(), ids=lambda p: p.name)
def test_every_category_is_listed_in_the_profiling_index(category):
    index = (PROFILING / "README.md").read_text(encoding="utf-8")
    assert f"{category.name}/" in index, f"{category.name} missing from profiling/README.md"


# ---- Per-example contents ----------------------------------------------


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_has_a_readme(example):
    readme = example / "README.md"
    assert readme.is_file(), f"{_rel(example)} has no README.md"
    assert readme.read_text(encoding="utf-8").strip(), f"{_rel(readme)} is empty"


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_has_a_payload(example):
    payloads = [
        path
        for path in example.iterdir()
        if path.is_file() and path.suffix in {".py", ".hip", ".sh", ".cpp"}
    ]
    assert payloads, f"{_rel(example)} has no payload file"


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_is_listed_in_its_category_index(example):
    index = (example.parent.parent / "README.md").read_text(encoding="utf-8")
    assert example.name in index, f"{_rel(example)} missing from profiling/README.md"


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_readme_names_the_collector_and_the_payload(example):
    readme = (example / "README.md").read_text(encoding="utf-8")
    assert example.parent.name in readme, f"{_rel(example)}/README.md does not name its collector"


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_carries_no_absolute_host_paths(example):
    """No host specifics: an example must be runnable on someone else's machine."""
    forbidden = ("/apps/", "/home/")
    for path in example.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in forbidden:
            assert needle not in text, f"{_rel(path)} contains a host path ({needle})"


# ---- Recipes load -----------------------------------------------------


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_recipe_loads(example):
    recipe_path = example / "recipe.yaml"
    assert recipe_path.is_file(), f"{_rel(example)} has no recipe.yaml"
    recipe = load_recipe(recipe_path)
    assert recipe.cells, f"{_rel(recipe_path)} produced no cells"


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_recipe_activates_its_categorys_collector(example):
    """The recipe's ``collect:`` block must be ACTIVE (not commented out) and
    must name the collector whose directory it lives in -- otherwise the
    example silently demonstrates nothing."""
    recipe = load_recipe(example / "recipe.yaml")
    assert recipe.collect == (example.parent.name,), (
        f"{_rel(example)}/recipe.yaml collects {recipe.collect}, "
        f"expected exactly ('{example.parent.name}',)"
    )


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_recipe_is_probe_mode(example):
    """Every profiling example drives an opaque command after ``--``, which is
    the probe flow."""
    recipe = load_recipe(example / "recipe.yaml")
    assert recipe.probe_extras is not None, f"{_rel(example)}/recipe.yaml is not mode: probe"


@pytest.mark.parametrize("example", _EXAMPLE_DIRS, ids=_EXAMPLE_IDS)
def test_example_recipe_collector_options_validate(example):
    """``load_recipe`` already validates them; assert the block is non-trivial
    so an example is a usable template rather than an empty gesture."""
    recipe = load_recipe(example / "recipe.yaml")
    name = example.parent.name
    assert recipe.collect_options.get(name), (
        f"{_rel(example)}/recipe.yaml sets no {name} options; the examples are "
        "the copy-paste templates for the option schema"
    )


# ---- Payload exit convention --------------------------------------------
#
# Proton runs the payload through ``execute_as_main``, which on Triton 3.6.0
# catches ``Exception`` -- not ``BaseException``. A ``SystemExit`` on the
# success path therefore escapes Proton's own CLI before ``finalize()`` writes
# the ``.hatchet``, and the run reports a clean exit 0 with no profile at all.
# Payloads must return normally on success and exit non-zero only on failure.


def _python_payloads() -> list[Path]:
    return sorted(path for example in _EXAMPLE_DIRS for path in example.glob("*.py"))


_PAYLOADS = _python_payloads()
_PAYLOAD_IDS = [str(path.relative_to(EXAMPLES)) for path in _PAYLOADS]


def _main_guard_body(payload: Path) -> list[ast.stmt]:
    """Statements under ``if __name__ == "__main__":``, or ``[]`` if absent."""
    for node in ast.parse(payload.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.If) and "__name__" in ast.dump(node.test):
            return node.body
    return []


def _exit_code_of_main_guard(payload: Path, returned: int) -> int | None:
    """Run *only* the ``__main__`` block with ``main()`` stubbed to return
    ``returned``, and report how it terminated.

    Executing the guard block alone keeps this a CPU test: the payload's
    module-level ``import torch`` never runs. Returns the ``SystemExit`` code,
    or ``None`` when the block fell off the end without raising.
    """
    module = ast.Module(body=_main_guard_body(payload), type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        exec(compile(module, str(payload), "exec"), {"main": lambda: returned})  # noqa: S102
    except SystemExit as exc:
        return 1 if exc.code is None else int(exc.code)
    return None


@pytest.mark.parametrize("payload", _PAYLOADS, ids=_PAYLOAD_IDS)
def test_payload_has_a_main_guard(payload):
    assert _main_guard_body(payload), f'{_rel(payload)} has no if __name__ == "__main__" block'


@pytest.mark.parametrize("payload", _PAYLOADS, ids=_PAYLOAD_IDS)
def test_payload_success_path_does_not_raise_systemexit(payload):
    """A ``SystemExit(0)`` here costs the whole Proton capture on Triton 3.6.0."""
    assert _exit_code_of_main_guard(payload, 0) is None, (
        f"{_rel(payload)} raises SystemExit when main() returns 0; Proton's "
        "execute_as_main on Triton 3.6.0 lets that escape and skips finalize(), "
        "so the payload exits 0 having written no profile"
    )


@pytest.mark.parametrize("payload", _PAYLOADS, ids=_PAYLOAD_IDS)
@pytest.mark.parametrize("code", [1, 2])
def test_payload_failure_path_still_exits_nonzero(payload, code):
    """The success-path fix must not swallow the self-check's failure exit:
    payloads signal a bad result with 1 and a missing GPU with 2, and the
    platform needs those to mark the trial failed."""
    assert (
        _exit_code_of_main_guard(payload, code) == code
    ), f"{_rel(payload)} does not propagate main() == {code} as an exit status"


# ---- The mode: env contract has two halves ------------------------------
#
# Under ``mode: env`` the collector only exports variables; the payload is what
# calls ``proton.start()``. So a knob the recipe sets and the payload never
# reads is accepted at recipe load, exported, and then silently missing from
# the capture -- no validation error anywhere. Holding the payloads to the
# bundle ``build_env`` actually produces, rather than to a list written out
# here, is what makes a future option fail this test instead of shipping as a
# no-op.

#: Exported for a payload that wants to write artifacts alongside the profile.
#: It configures nothing about the session -- and ``NAME`` already carries the
#: directory -- so a payload that ignores it loses no measurement.
_OPTIONAL_ENV_SUFFIXES = frozenset({"DIR"})


def _interpolated_env_suffixes(payload: Path) -> set[str]:
    """Names the payload interpolates after ``_ENV_PREFIX`` in executable code.

    Read out of the f-string nodes rather than searched for in the text. Every
    one of these names also appears in prose -- the docstrings that explain
    which variable carries which knob name them -- so a substring check over the
    file would be satisfied by the explanation alone, and a payload that stopped
    reading a variable would still pass. An f-string cannot appear in a
    docstring, so what this collects is what the code actually looks up.
    """
    suffixes: set[str] = set()
    for node in ast.walk(ast.parse(payload.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.JoinedStr):
            continue
        suffixes.update(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return suffixes


def _env_mode_proton_examples() -> list[Path]:
    proton = [example for example in _EXAMPLE_DIRS if example.parent.name == "proton"]
    return [
        example
        for example in proton
        if (load_recipe(example / "recipe.yaml").collect_options.get("proton") or {}).get("mode")
        == "env"
    ]


_ENV_MODE_EXAMPLES = _env_mode_proton_examples()
_ENV_MODE_IDS = [str(path.relative_to(EXAMPLES)) for path in _ENV_MODE_EXAMPLES]


@pytest.mark.skipif(not _ENV_MODE_EXAMPLES, reason="no mode: env proton example in the tree")
@pytest.mark.parametrize("example", _ENV_MODE_EXAMPLES, ids=_ENV_MODE_IDS)
def test_env_mode_payload_reads_every_session_variable(example, tmp_path):
    """Options chosen to make ``build_env`` emit every key it can: the optional
    ones are omitted unless asked for, so a narrower call would assert less."""
    exported = build_env(
        tmp_path,
        {
            "mode": "env",
            "backend": "rocprofiler",
            "backend_mode": "pcsampling",
            "hook": "triton",
        },
    )
    required = sorted(
        suffix
        for suffix in (name[len(ENV_PREFIX) :] for name in exported)
        if suffix not in _OPTIONAL_ENV_SUFFIXES
    )
    assert required, "build_env exported nothing to check; the bundle or the prefix moved"
    looked_up: set[str] = set()
    for payload in sorted(example.glob("*.py")):
        looked_up |= _interpolated_env_suffixes(payload)
    missing = [suffix for suffix in required if suffix not in looked_up]
    assert not missing, (
        f"{_rel(example)} never reads {[ENV_PREFIX + s for s in missing]}; a recipe can "
        "set those knobs and the payload would start Proton without them, so the trial "
        "would report a capture that is missing what it asked for"
    )
