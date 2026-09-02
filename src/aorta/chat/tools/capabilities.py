"""What each tool produces, so the model can work out which one applies.

The obvious way to choose a tool is a mapping someone writes down: rules in the
system prompt ("if the question mentions NaN, run the NaN tool"), or symptom
phrases matched against the text. Both encode a guess about how a user will
describe a problem, and both are wrong the moment somebody describes it
differently -- a matcher keyed on "inf" fires on a question about "inference
throughput".

So there is no mapping here. The catalogue is generated from the registry: a
tool's description is its docstring, which it needs regardless, and the selector
asks a model whether the evidence that tool returns would answer the question in
front of it. A new tool needs a docstring. A new way of describing an old
problem needs nothing at all.

The one thing decided in code is :data:`_REQUIRES`, which is structural rather
than semantic: a tool that analyses pasted source cannot run when nothing was
pasted, and that is not a judgement call.
"""

from __future__ import annotations

#: Tools that read something the user supplied in the message. Proposing one of
#: these with nothing pasted is impossible, not merely unlikely, so it is the
#: single filter applied to the model's ranking.
_REQUIRES: dict[str, frozenset[str]] = {
    "triage_kernel_source": frozenset({"pasted_source"}),
    "triage_assembly_source": frozenset({"pasted_source"}),
}

#: Enough for the model to offer an alternative, few enough that the first
#: entry still reads as the recommendation.
MAX_CANDIDATES = 3


def _summary(description: str) -> str:
    """A tool's description without the generated argument list.

    ``@tool`` appends an ``Args:`` block from the signature. It tells the model
    how to call the tool, which matters at call time and is noise when the
    question is whether to call it at all.
    """
    text = " ".join((description or "").split())
    for marker in (" Args:", " Returns:", " Raises:"):
        head, sep, _ = text.partition(marker)
        if sep:
            text = head
    return text.strip()


def catalogue(tools: dict[str, object] | None = None) -> str:
    """The tool descriptions the selector ranks, one per line.

    Generated from the live registry, so a tool contributed through the
    ``aorta.chat_tools`` entry point is rankable without being named here.
    """
    if tools is None:
        from aorta.chat.plugins import load_chat_tools

        tools = {name: entry.tool for name, entry in load_chat_tools().items()}
    lines = []
    for name, tool in sorted(tools.items()):
        summary = _summary(getattr(tool, "description", ""))
        if summary:
            lines.append(f"- {name}: {summary}")
    return "\n".join(lines)


def requirements(tool_name: str) -> frozenset[str]:
    """What the request must contain for *tool_name* to be able to run."""
    return _REQUIRES.get(tool_name, frozenset())


def enforce_requirements(
    tools: list[str], *, has_pasted_source: bool
) -> tuple[list[str], list[str]]:
    """Drop proposals whose preconditions the request cannot satisfy.

    The one thing not left to the model, because it is not a judgement: a tool
    that analyses supplied source has nothing to analyse when the user supplied
    none. Returns the surviving tools and the ones removed, so a caller can say
    what it dropped rather than silently shortening the list.
    """
    kept, dropped = [], []
    for name in tools:
        if "pasted_source" in requirements(name) and not has_pasted_source:
            dropped.append(name)
        else:
            kept.append(name)
    return kept, dropped
