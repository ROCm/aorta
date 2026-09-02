"""One category vocabulary, shared by both front doors.

``aorta.agent`` reaches a category by trying mitigations; ``aorta.cia`` reaches
one by reading instrument evidence. They answer different questions, so both
loops survive -- but a verdict has to mean the same thing whichever produced
it, which it cannot if each ships its own list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.agent.llm import AUTOPSY_CATEGORIES

_SRC = Path(__file__).resolve().parents[2] / "src"
_CIA = _SRC / "aorta" / "cia"

#: Reachable only from instrument evidence. The probe agent never proposes
#: these, and before the port they lived in a second list under ``aorta.cia``.
_EVIDENCE_ONLY = frozenset({"gpu_race", "numeric_silent", "tooling_gap"})


def test_the_evidence_only_categories_joined_the_shared_set():
    missing = _EVIDENCE_ONLY - AUTOPSY_CATEGORIES
    assert not missing, (
        f"{sorted(missing)} are produced by aorta.cia but absent from "
        "AUTOPSY_CATEGORIES, so a verdict carrying one would not validate."
    )


def test_the_probe_agents_own_categories_survived_the_merge():
    """Adding to a shared set must not quietly remove from it."""
    for category in ("rccl_hang", "thermal_throttle", "illegal_mem", "oom_fragment",
                     "checkpoint_race", "launch_error", "perf_regression", "unknown"):
        assert category in AUTOPSY_CATEGORIES


def test_cia_ships_no_second_category_list():
    """The failure this guards is duplication, not absence.

    A literal list of category names anywhere under ``aorta.cia`` is a second
    source of truth that will drift from the first. The router's prompt text is
    generated from the shared set for exactly this reason.
    """
    quorum = {"rccl_hang", "thermal_throttle", "oom_fragment", "checkpoint_race"}
    offenders: list[str] = []
    for path in sorted(_CIA.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # Four or more of the agent-only names together is a list, not prose.
        if len({name for name in quorum if f'"{name}"' in text}) >= 4:
            offenders.append(str(path.relative_to(_SRC)))
    assert not offenders, (
        "these look like a second category list:\n  " + "\n  ".join(offenders)
        + "\n\nImport AUTOPSY_CATEGORIES instead."
    )


@pytest.mark.parametrize("supplied", ["gpu_race", "numeric_silent", "unknown"])
def test_a_known_category_passes_through(supplied: str):
    from aorta.cia.autopsy.router import coerce_category

    assert coerce_category(supplied) == supplied


@pytest.mark.parametrize("supplied", ["", "   ", "GPU_RACE", "probably_a_race", None])
def test_anything_outside_the_vocabulary_reports_as_unknown(supplied):
    """The field is free text, so a model can return a plausible-looking label.

    Passing it through would put a string nobody downstream handles into the
    report; 'unknown' is the honest answer and one the schema knows.
    """
    from aorta.cia.autopsy.router import coerce_category

    assert coerce_category(supplied) == "unknown"


def test_the_prompt_lists_exactly_the_shared_vocabulary():
    """Generated, not typed -- so the prompt cannot drift from the validator."""
    from aorta.cia.autopsy.router import _CATEGORY_DESC

    for category in AUTOPSY_CATEGORIES:
        assert category in _CATEGORY_DESC
