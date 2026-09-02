"""The catalogue the selector ranks against.

Two properties keep tool choice from decaying back into a routing table.

A description says what a tool *does* and what evidence it returns, never which
symptom it belongs to. The moment one says "use this for NaN", the coupling is
back: it works for the phrasings someone thought of and fails for the rest. A
matcher keyed on "inf" fires on a question about "inference throughput", which
is not a hypothetical -- it happened.

And the catalogue is generated from the registry rather than written beside it.
A second list is a list that drifts, and the descriptions already exist as
docstrings.
"""

from __future__ import annotations

import re

import pytest

from aorta.chat.plugins import BUILTIN_CHAT_TOOLS, load_chat_tools
from aorta.chat.tools.capabilities import (
    MAX_CANDIDATES,
    _summary,
    catalogue,
    requirements,
)

#: Names of failure modes. A description carrying one is claiming a symptom
#: rather than describing an instrument, which is the coupling to avoid.
#:
#: Note what is *not* here: "use this for exact searches" is capability
#: guidance and belongs in a tool description -- the act node reads the same
#: string when it decides how to call a bound tool. The hazard is symptom
#: vocabulary, not the imperative mood.
_SYMPTOM_WORDS = re.compile(r"\b(race|nan|hazard|deadlock|corrupt|crash)\b", re.I)


def _tools() -> dict:
    return {name: entry.tool for name, entry in load_chat_tools().items()}


def test_every_tool_describes_itself():
    """A tool with no description cannot be chosen for a good reason."""
    missing = [name for name, tool in _tools().items() if not _summary(tool.description)]
    assert not missing, f"no description: {missing}"


@pytest.mark.parametrize("name", sorted(BUILTIN_CHAT_TOOLS))
def test_no_description_names_a_symptom(name: str):
    found = _SYMPTOM_WORDS.findall(_summary(BUILTIN_CHAT_TOOLS[name].description))
    assert not found, (
        f"{name}'s description names {found}, which coupIes it to a phrasing. "
        "Say what evidence the tool returns and let the model decide whether "
        "that answers the question."
    )


def test_the_catalogue_is_generated_from_the_registry():
    """Not written beside it, which is how two lists drift apart."""
    listed = {line.split(":", 1)[0].removeprefix("- ") for line in catalogue().splitlines()}
    assert listed == set(_tools())


def test_the_catalogue_omits_the_generated_argument_list():
    """``Args:`` tells the model how to call a tool, not whether to."""
    assert "Args:" not in catalogue()


def test_a_tool_outside_the_builtins_is_rankable():
    """The extensibility claim: a contributed tool needs no edit here.

    ``catalogue`` takes the registry it is given, so anything reaching
    ``load_chat_tools`` -- including an ``aorta.chat_tools`` entry point --
    appears without being named in this package.
    """

    class _Contributed:
        description = "Reads a vendor trace and reports the queue depth it recorded."

    text = catalogue({"vendor_trace": _Contributed()})
    assert "vendor_trace" in text
    assert "queue depth" in text


def test_the_shortlist_is_short_enough_to_read_as_a_recommendation():
    assert 2 <= MAX_CANDIDATES <= 4


def test_only_source_reading_tools_declare_a_requirement():
    """The filter must stay narrow: it overrides the model's judgement.

    Requiring something of a tool that does not need it silently removes a
    correct answer, which is worse than the wrong answer it was meant to stop.
    """
    declared = {name for name in _tools() if requirements(name)}
    assert declared == {"triage_kernel_source", "triage_assembly_source"}
