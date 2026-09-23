"""Turning what someone pasted into something a compiler will accept.

People paste a fragment, or a fragment wrapped in an explanation, far more often
than they paste a compilable unit. Both have already broken this: a bare
instruction sequence assembled to an object with no kernel symbol, which the
analyser then refused as unparseable; and prose around a fenced block reached
the assembler as source and produced errors about the English.

Neither is the user's mistake, and neither should reach a tool as a failure.
"""

from __future__ import annotations

import pytest

from aorta.chat.tools.harness.assembly import AsmHarnessError, prepare_asm

_FRAGMENT = """\
s_load_dword s4, s[0:1], 0x10
s_load_dwordx4 s[4:7], s[0:1], 0x0
v_mov_b32 v0, s4
s_endpgm
"""


def test_a_bare_instruction_sequence_becomes_assemblable():
    """A fragment has no symbol, no directives and no size."""
    prepared = prepare_asm(_FRAGMENT)
    assert prepared.wrapped
    for required in (".amdhsa_kernel", ".size", "s_endpgm"):
        assert required in prepared.program, f"{required} missing from the wrapped unit"


def test_the_wrapped_unit_declares_the_registers_the_fragment_uses():
    """Understating the budget is a launch failure, not a compile error.

    The fragment writes s[4:7] and v0, so the descriptor has to claim at least
    that many, and nothing in the pasted text says so explicitly.
    """
    prepared = prepare_asm(_FRAGMENT)
    assert prepared.sgpr_count >= 8
    assert prepared.vgpr_count >= 1


def test_a_complete_unit_is_left_alone():
    """Wrapping something already whole would produce two kernel symbols."""
    prepared = prepare_asm(prepare_asm(_FRAGMENT).program)
    assert not prepared.wrapped


def test_prose_around_a_fenced_block_is_discarded():
    """The model passes the message through, explanation and all."""
    pasted = (
        "Our hand-tuned prologue gives the wrong answer, here it is:\n\n"
        "```asm\n" + _FRAGMENT + "```\n\n"
        "We think the second load clobbers the first. What do you reckon?"
    )
    prepared = prepare_asm(pasted)
    assert "We think" not in prepared.program
    assert "reckon" not in prepared.program
    assert "s_load_dword" in prepared.program


def test_the_kernel_name_reaches_the_symbol():
    prepared = prepare_asm(_FRAGMENT, kernel_name="tile_reverse")
    assert "tile_reverse" in prepared.program


@pytest.mark.parametrize("empty", ["", "   \n\n  ", "```\n```"], ids=["empty", "blank", "fence"])
def test_nothing_to_assemble_is_refused_with_a_reason(empty: str):
    """Better a message the user can act on than an assembler error."""
    with pytest.raises(AsmHarnessError):
        prepare_asm(empty)
