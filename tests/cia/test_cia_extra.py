"""The [cia] extra installs something on every Python this package supports.

``requires-python`` is >=3.10 and the extra pinned ``dspy-ai`` behind a
``python_version >= '3.11'`` marker. On 3.10 that made
``pip install amd-aorta[cia]`` succeed having installed nothing, and the first
``import aorta.cia.launch.planner`` fail on a missing module -- an install that
reported success and a runtime error that named the wrong problem.

The marker was copied from the chat extras, where it is real: chat/config.py
reads a profile with stdlib tomllib. Nothing under aorta/cia needs 3.11, and
dspy declares >=3.10 itself.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
CIA_EXTRA = PYPROJECT["project"]["optional-dependencies"]["cia"]


class TestTheExtraResolvesOnEverySupportedPython:
    def test_nothing_in_it_is_gated_on_a_python_version(self):
        gated = [dep for dep in CIA_EXTRA if "python_version" in dep]
        assert not gated, f"these install nothing on 3.10: {gated}"

    def test_it_still_brings_dspy(self):
        assert any(d.split(";")[0].strip().startswith("dspy") for d in CIA_EXTRA)

    def test_the_floor_is_the_one_it_claims(self):
        assert PYPROJECT["project"]["requires-python"] == ">=3.10"


class TestNothingHereActuallyNeeds311:
    """If it did, the marker would have been right and the fix would be a floor."""

    def test_every_module_parses_as_310(self):
        offenders = []
        for path in sorted((ROOT / "src" / "aorta" / "cia").rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))
            except SyntaxError as exc:
                offenders.append(f"{path.relative_to(ROOT)}: {exc.msg}")
        assert not offenders, offenders

    @pytest.mark.parametrize(
        "construct", ["import tomllib", "from typing import Self", "except*"]
    )
    def test_no_module_uses_a_311_only_construct(self, construct):
        hits = [
            str(p.relative_to(ROOT))
            for p in (ROOT / "src" / "aorta" / "cia").rglob("*.py")
            if construct in p.read_text(encoding="utf-8")
        ]
        assert not hits, f"{construct} found in {hits}"


class TestTheChatExtrasKeepTheirMarkers:
    """Theirs encode a real constraint; this fix must not sweep them up."""

    @pytest.mark.parametrize("extra", ["chat-cli", "chat-ui"])
    def test_they_are_still_gated(self, extra):
        deps = PYPROJECT["project"]["optional-dependencies"].get(extra, [])
        assert any("python_version" in d for d in deps), extra


def test_a_missing_dspy_names_the_extra_rather_than_the_module():
    """Without [cia] installed, the error should say what to install."""
    import inspect

    from aorta.cia import llm

    source = inspect.getsource(llm)
    assert "amd-aorta[cia]" in source
    assert "except ImportError" in source
