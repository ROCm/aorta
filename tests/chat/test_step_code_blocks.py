"""Tool output must stay inside the code block it is rendered in.

Tool output is arbitrary text. ``read_file`` over this repository's own
Markdown returns fenced blocks, and a fixed ``` opener ends at the first
fence inside the content -- the remainder then renders as Markdown, so a
file's headings and prose escape into the step.

These assert against rendered HTML rather than the returned string, because
the bug is only visible once a Markdown renderer has had it.
"""

from __future__ import annotations

import pytest

cl = pytest.importorskip("chainlit", reason="requires the chat-ui extra")
markdown_it = pytest.importorskip("markdown_it", reason="requires a CommonMark renderer")

from aorta.chat.ui.app import _as_code_block, _node_reasoning


def render(markdown: str) -> str:
    return markdown_it.MarkdownIt("commonmark").render(markdown)


#: Output that is itself Markdown, which is what read_file returns here.
MARKDOWN_OUTPUT = "read_file(README.md) ->\n# Title\n\n```python\nx = 1\n```\n\nDone."


def test_markdown_output_cannot_escape():
    """The prose after an inner fence stays code, rather than becoming a paragraph."""
    html = render(_as_code_block(MARKDOWN_OUTPUT))
    assert "<p>Done.</p>" not in html, "tool output escaped into the step's Markdown"


def test_a_heading_in_output_is_not_rendered_as_one():
    """A file's headings must not restyle the step."""
    assert "<h1>" not in render(_as_code_block(MARKDOWN_OUTPUT))


@pytest.mark.parametrize(
    "payload",
    [
        "```",
        "````````\nnot escaping\n````````",
        "trailing ```",
        "``` opener with no close",
        "~~~\ntilde fence\n~~~",
        "",
    ],
)
def test_adversarial_payloads_stay_contained(payload):
    """Whatever run of backticks the content holds, the fence outruns it."""
    html = render(_as_code_block(payload))
    assert html.count("<pre>") == 1, f"fence did not contain: {payload!r}"


def test_each_tool_call_is_its_own_block():
    """Distinct calls stay visually separate.

    An indented code block would also contain the output, but consecutive
    indented chunks separated by a blank line are one block in CommonMark,
    which would run each tool's output into the next.
    """
    delta = {"tool_trace": [MARKDOWN_OUTPUT, "list_jobs() ->\njob-1  ok"]}
    html = render(_node_reasoning("act", delta))
    assert html.count("<pre>") == 2


def test_the_output_survives_verbatim():
    """Containment must not come at the cost of altering what the tool said."""
    html = render(_as_code_block(MARKDOWN_OUTPUT))
    assert "x = 1" in html and "# Title" in html


def test_truncation_still_applies():
    """The 1500-character cap on each entry is unchanged."""
    delta = {"tool_trace": ["x" * 5000]}
    assert "x" * 1500 in _node_reasoning("act", delta)
    assert "x" * 1501 not in _node_reasoning("act", delta)


def test_an_empty_trace_renders_nothing():
    assert _node_reasoning("act", {"tool_trace": []}) == ""
