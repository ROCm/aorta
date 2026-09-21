"""A contributed tool has to be able to say what it needs.

The catalogue is generated from the live registry precisely so that a tool
contributed through the ``aorta.chat_tools`` entry point is rankable without
being named in this package. That held for ranking and not for the one rule
enforced in code: the requirement was a dict here, keyed by hardcoded name, so
a third-party tool that reads pasted source had no way to say so. It would be
proposed for a question with nothing to read, and then be handed nothing.

The requirement is read off the tool now, the same way its description is. The
built-ins stay in the table -- they are the reason the rule exists and naming
them keeps that visible -- but the table is a fallback rather than the register
of what is allowed to have a requirement.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.tools import BaseTool, tool

from aorta.chat.tools.capabilities import (
    KNOWN_REQUIREMENTS,
    enforce_requirements,
    requirements,
)


@tool
def reads_a_paste(source: str) -> str:
    """A contributed tool that analyses source the user supplied."""
    return source


reads_a_paste.metadata = {"requires": ["pasted_source"]}


@tool(extras={"requires": ["pasted_source"]})
def declared_in_the_decorator(source: str) -> str:
    """The same, said through the decorator instead."""
    return source


class HandRolled(BaseTool):
    """A tool shipped as a class rather than a decorated function."""

    name: str = "hand_rolled"
    description: str = "Analyses source the user supplied."
    requires: tuple = ("pasted_source",)

    def _run(self, *args, **kwargs):
        return ""


@tool
def needs_nothing(question: str) -> str:
    """A contributed tool that works on any question."""
    return question


_REGISTRY = {
    "reads_a_paste": reads_a_paste,
    "declared_in_the_decorator": declared_in_the_decorator,
    "hand_rolled": HandRolled(),
    "needs_nothing": needs_nothing,
}


class TestAContributedToolCanDeclareIt:
    """The finding: this was the one thing a plugin could not say."""

    @pytest.mark.parametrize(
        "name",
        ["reads_a_paste", "declared_in_the_decorator", "hand_rolled"],
        ids=["metadata", "extras", "attribute"],
    )
    def test_the_declaration_is_read_off_the_tool(self, name):
        assert requirements(name, _REGISTRY) == frozenset({"pasted_source"})

    @pytest.mark.parametrize(
        "name",
        ["reads_a_paste", "declared_in_the_decorator", "hand_rolled"],
        ids=["metadata", "extras", "attribute"],
    )
    def test_and_it_is_dropped_when_nothing_was_pasted(self, name):
        kept, dropped = enforce_requirements(
            [name], has_pasted_source=False, registry=_REGISTRY
        )

        assert kept == [] and dropped == [name]

    def test_it_survives_when_something_was_pasted(self):
        kept, _ = enforce_requirements(
            list(_REGISTRY), has_pasted_source=True, registry=_REGISTRY
        )

        assert kept == list(_REGISTRY)

    def test_a_tool_that_declares_nothing_is_left_alone(self):
        kept, dropped = enforce_requirements(
            ["needs_nothing"], has_pasted_source=False, registry=_REGISTRY
        )

        assert kept == ["needs_nothing"] and dropped == []


class TestTheBuiltInsStillWork:
    """They predate the attribute; the table is their fallback."""

    @pytest.mark.parametrize(
        "name", ["triage_kernel_source", "triage_assembly_source"]
    )
    def test_the_table_still_speaks_for_them(self, name):
        assert requirements(name, {}) == frozenset({"pasted_source"})

    def test_an_unknown_name_requires_nothing(self):
        assert requirements("no_such_tool", {}) == frozenset()

    def test_a_declaration_on_the_tool_wins_over_the_table(self):
        """The tool is the better authority on what the tool needs."""

        @tool
        def triage_kernel_source(source: str) -> str:
            """A replacement that needs nothing."""
            return source

        triage_kernel_source.metadata = {"requires": []}

        assert requirements(
            "triage_kernel_source", {"triage_kernel_source": triage_kernel_source}
        ) == frozenset()


class TestDeclarationsThatDoNotMakeSense:
    def test_a_bare_string_is_read_as_one_requirement(self):
        """``requires = "pasted_source"`` is the obvious thing to write."""

        @tool
        def stringly(source: str) -> str:
            """Declared with a string rather than a list."""
            return source

        stringly.metadata = {"requires": "pasted_source"}

        assert requirements("stringly", {"stringly": stringly}) == frozenset({"pasted_source"})

    def test_a_requirement_nothing_can_check_is_not_enforced(self):
        """Enforcing it would drop the tool on a condition never evaluated."""

        @tool
        def futuristic(question: str) -> str:
            """Declares something this version has never heard of."""
            return question

        futuristic.metadata = {"requires": ["gpu_access"]}

        assert requirements("futuristic", {"futuristic": futuristic}) == frozenset()

    def test_but_it_is_reported(self, caplog):
        """Silently ignoring what a contributor asked for is how they find out late."""

        @tool
        def futuristic(question: str) -> str:
            """Declares something this version has never heard of."""
            return question

        futuristic.metadata = {"requires": ["gpu_access"]}

        with caplog.at_level(logging.WARNING, logger="aorta.chat.tools.capabilities"):
            requirements("futuristic", {"futuristic": futuristic})

        assert "gpu_access" in caplog.text
        assert "pasted_source" in caplog.text, "the message should say what is known"

    def test_a_known_requirement_alongside_an_unknown_one_still_holds(self):
        @tool
        def mixed(source: str) -> str:
            """Declares one checkable requirement and one not."""
            return source

        mixed.metadata = {"requires": ["gpu_access", "pasted_source"]}

        assert requirements("mixed", {"mixed": mixed}) == frozenset({"pasted_source"})

    def test_a_declaration_that_is_not_iterable_is_ignored(self):
        @tool
        def nonsense(question: str) -> str:
            """Declared with a number."""
            return question

        nonsense.metadata = {"requires": 3}

        assert requirements("nonsense", {"nonsense": nonsense}) == frozenset()


class TestTheRegistryIsReadOnce:
    def test_enforcing_does_not_reload_it_per_candidate(self, monkeypatch):
        """load_chat_tools rescans the entry points on every call."""
        import aorta.chat.plugins as plugins

        calls = []
        real = plugins.load_chat_tools

        def counted():
            calls.append(1)
            return real()

        monkeypatch.setattr(plugins, "load_chat_tools", counted)
        enforce_requirements(
            ["triage_kernel_source", "triage_assembly_source", "search_code"],
            has_pasted_source=False,
        )

        assert len(calls) == 1, f"loaded the registry {len(calls)} times for 3 candidates"


class TestTheKnownSetIsClosedOnPurpose:
    def test_pasted_source_is_the_one_that_exists(self):
        assert KNOWN_REQUIREMENTS == frozenset({"pasted_source"})
