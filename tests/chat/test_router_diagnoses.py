"""Pasted code plus a symptom routes to the tools, not to retrieval.

The router's "action" category was written when the only tools searched the
codebase, so it describes finding, listing and reading files. The diagnostic
tools arrived later and it was never told. A user who pastes a failing training
script and says it dies partway through matches none of the examples, routes to
retrieval, and gets "I could not answer that from the retrieved context" -- or,
worse, a fluent guess from the model's own knowledge with no GPU behind it.

These assert on the prompt rather than on a live model: the prompt is the thing
that regressed, and it is what every provider reads.
"""

from __future__ import annotations

import pytest

from aorta.chat.graph.nodes import ROUTER_PROMPT

_PROMPT = ROUTER_PROMPT.lower()


def test_router_still_offers_exactly_the_two_categories_the_graph_branches_on():
    assert '"question"' in ROUTER_PROMPT
    assert '"action"' in ROUTER_PROMPT
    assert ROUTER_PROMPT.rstrip().endswith("question or action")


def test_router_knows_a_pasted_artefact_is_not_a_retrieval_question():
    assert "pasted" in _PROMPT


@pytest.mark.parametrize("artefact", ["source", "assembly", "log", "stack trace"])
def test_router_names_the_kinds_of_artefact_users_paste(artefact):
    assert artefact in _PROMPT


@pytest.mark.parametrize(
    "symptom",
    ["wrong", "crash", "hang", "loss that stops being a number"],
)
def test_router_names_the_symptoms_that_warrant_a_gpu(symptom):
    """The NaN case reads as none of the old examples -- no verb, just a symptom."""
    assert symptom in _PROMPT


def test_router_does_not_require_the_user_to_name_a_tool():
    """ConSan only routed correctly because the prompt said 'run the sanitizer'."""
    assert "whether or not it names a tool" in _PROMPT


def test_router_still_prefers_action_when_uncertain():
    assert "in doubt" in _PROMPT and "classify as action" in _PROMPT


def test_router_keeps_the_codebase_navigation_examples():
    """The diagnostic wording is an addition, not a replacement."""
    for example in ("find all functions", "list all", "search for"):
        assert example in _PROMPT
