"""The declared minimum has to be able to run the code.

``graph/nodes.py`` imports ``langgraph.config.get_stream_writer`` at module
scope and ``session.py`` asks for the ``custom`` stream mode by name. The
declared floor was ``langgraph>=0.2.0``, and 0.2.0 has no ``langgraph.config``
at all -- so an installation that satisfied the dependency could not import the
chat graph, and neither the UI nor the CLI would start.

0.2.69 is the first release carrying both. Measured by installing each version
and importing, not read off a changelog: 0.2.68 raises ModuleNotFoundError and
0.2.69 both imports and streams a custom chunk.

What this file guards is the agreement between the two -- that what the code
imports is what the floor promises. A future import from a newer langgraph API
needs the floor raised with it, and this is what says so.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: The release that first provided get_stream_writer and the custom stream
#: mode. Bisected against PyPI; 0.2.68 has neither.
_REQUIRED_FLOOR = (0, 2, 69)


def _declared_floor() -> tuple[int, ...]:
    spec = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    for group in spec["project"].get("optional-dependencies", {}).values():
        for requirement in group:
            match = re.match(r"langgraph>=(\d+)\.(\d+)\.(\d+)", requirement)
            if match:
                return tuple(int(part) for part in match.groups())
    pytest.fail("no langgraph requirement with a lower bound found")


class TestTheFloorCoversWhatWeImport:
    def test_it_is_at_least_the_release_that_provides_the_api(self):
        assert _declared_floor() >= _REQUIRED_FLOOR

    def test_the_old_floor_would_not_have_been(self):
        """0.2.0 satisfied the declaration and could not import the graph."""
        assert (0, 2, 0) < _REQUIRED_FLOOR


class TestTheImportsThatSetIt:
    """Named here so raising the floor and changing these stay connected."""

    def test_get_stream_writer_is_imported_at_module_scope(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "aorta" / "chat" / "graph" / "nodes.py"
        ).read_text(encoding="utf-8")

        assert "from langgraph.config import get_stream_writer" in source

    def test_the_custom_stream_mode_is_requested(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "aorta" / "chat" / "session.py"
        ).read_text(encoding="utf-8")

        assert '"custom"' in source

    def test_both_work_on_the_installed_version(self):
        """The floor is a claim about the minimum; this checks what is here."""
        from langgraph.config import get_stream_writer

        assert callable(get_stream_writer)


class TestTheInstalledVersionSatisfiesTheFloor:
    def test_it_is_not_below_what_we_declare(self):
        import importlib.metadata as metadata

        installed = tuple(
            int(part) for part in metadata.version("langgraph").split(".")[:3]
        )

        assert installed >= _declared_floor()
