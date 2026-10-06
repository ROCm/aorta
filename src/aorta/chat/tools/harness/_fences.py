"""Finding the code in a message that also explains the problem.

Engineers paste their symptoms around their code, and both harnesses have to
get the code out before anything else looks at it. The assembly side learned
this first -- prose reaching the assembler produces a wall of syntax errors
about English -- and the kernel side inherited the same messages without the
same extraction, so a fenced paste reached hipcc with the backticks still on
it.

The rule is here rather than in both, because a fence is a fence whatever is
inside it. What differs is only which block each harness recognises as its own,
which is the predicate the caller passes.
"""

from __future__ import annotations

import re
from typing import Callable

#: ```lang ... ``` or a bare fence. Non-greedy, so several blocks in one
#: message stay separate rather than merging into one with the fences in the
#: middle of it.
#:
#: The tag allows more than letters because the tags people write do: ```c++
#: is what an editor inserts for this language, and a letters-only class left
#: the fence unmatched, so the whole message went to the compiler with the
#: backticks still on it -- the failure this is here to prevent, for the most
#: likely tag of all.
_FENCED = re.compile(r"```[\w+#.-]*\n(.*?)```", re.S)


def fenced_blocks(text: str) -> list[str]:
    """Every fenced block in *text*, outermost first, fences removed."""
    return [block.strip("\n") for block in _FENCED.findall(text)]


def extract_fenced(text: str, recognises: Callable[[str], bool]) -> tuple[str, bool]:
    """The first fenced block *recognises* accepts, else *text* unchanged.

    Returns the code and whether a fence produced it. The caller needs the
    second value: a fence is the user pointing at the code, and without one the
    whole message is only a guess at where the code was.

    First-match rather than last or longest, and that is a policy rather than
    an accident. A message with several blocks is usually the code followed by
    the error it produced, or by what the author tried next; the first block
    that looks like the thing being asked about is the thing being asked about.
    Blocks the predicate rejects are skipped, so a log or a stack trace fenced
    ahead of the code does not win.
    """
    for block in fenced_blocks(text):
        if recognises(block):
            return block, True
    return text, False


__all__ = ["extract_fenced", "fenced_blocks"]
