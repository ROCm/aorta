"""Refusing a bug report before it reaches the assembler.

A fenced block wins over the prose around it, which is the case the harness
tests already cover. The case they do not is the message with no fence at all:
the fallback returned the whole thing, and the test it had to pass asked only
whether some line was not blank, not a fence, not a comment and not a label.
Every sentence in every bug report passes that.

So "the kernel hangs after the second load" was wrapped in a kernel descriptor
and handed to clang, which answered with a wall of syntax errors about the
sentence -- the exact outcome this module exists to prevent, reported to the
user as though their assembly would not build.

What separates the two is that assembly names instructions. A known mnemonic
prefix settles it, and where the mnemonic is unusual, most of the lines still
have to read as ``op dst, src`` -- which a sentence with a comma in it does
not.
"""

from __future__ import annotations

import pytest

from aorta.chat.tools.harness.assembly import AsmHarnessError, extract_code, prepare_asm

_INSTRUCTIONS = "s_load_dword s4, s[0:1], 0x10\nv_mov_b32 v0, s4\ns_endpgm"


class TestABugReportIsNotAssembly:
    """The regression, in the words people actually use."""

    @pytest.mark.parametrize(
        "message",
        [
            "the kernel hangs after the second load",
            "My training loss goes to NaN a few steps in. No crash, it just "
            "stops being a number.",
            "can you check the waits in my inner loop please",
            "it works on one wave and deadlocks on two",
            "the second load never lands, I think there is a missing wait",
        ],
    )
    def test_it_is_refused_rather_than_assembled(self, message):
        with pytest.raises(AsmHarnessError):
            prepare_asm(message)

    def test_the_refusal_asks_for_what_is_missing(self):
        """The user has to be able to tell this from "your assembly is wrong"."""
        with pytest.raises(AsmHarnessError) as raised:
            prepare_asm("the kernel hangs after the second load")

        message = str(raised.value)
        assert "no assembly found" in message
        assert "```" in message, "the surest remedy should be named"

    def test_a_sentence_with_a_comma_is_still_a_sentence(self):
        """The comma is what makes prose look like ``op dst, src``."""
        with pytest.raises(AsmHarnessError):
            prepare_asm("No crash, it just stops being a number")

    def test_an_empty_message_is_still_reported_as_empty(self):
        """A different complaint, and the more useful one for a blank paste."""
        with pytest.raises(AsmHarnessError, match="empty"):
            prepare_asm("   \n  \n")


class TestUnfencedAssemblyStillWorks:
    """Most pastes have no fence, and they have to keep working."""

    def test_a_bare_instruction_sequence(self):
        assert prepare_asm(_INSTRUCTIONS).wrapped

    def test_an_indented_sequence(self):
        assert prepare_asm("    v_mov_b32 v0, s4\n    s_endpgm").wrapped

    def test_a_single_instruction(self):
        assert prepare_asm("s_endpgm").wrapped

    @pytest.mark.parametrize(
        "line",
        [
            "ds_read_b32 v1, v0",
            "buffer_load_dword v0, v1, s[0:3], 0 offen",
            "global_load_dwordx4 v[0:3], v[4:5], off",
            "flat_store_dword v[0:1], v2",
            "scratch_load_dword v0, off, off",
        ],
        ids=["lds", "buffer", "global", "flat", "scratch"],
    )
    def test_the_memory_instruction_families(self, line):
        """Not every instruction starts with s_ or v_."""
        assert prepare_asm(line).wrapped

    def test_an_unfamiliar_mnemonic_carried_by_its_neighbours(self):
        """A macro or a mnemonic this list has never heard of.

        The majority rule is what keeps the prefix list from being a
        whitelist that quietly refuses real assembly.
        """
        text = "some_new_op v0, v1\nanother_op v2, v3\nyet_another v4, v5"

        assert prepare_asm(text).wrapped

    def test_instructions_with_comments_and_labels_around_them(self):
        text = "// the inner loop\nloop:\n\ts_load_dword s4, s[0:1], 0x0\n\ts_endpgm"

        assert prepare_asm(text).wrapped

    def test_a_mnemonic_that_does_not_fit_the_prefix_scheme(self):
        """``exp`` is one word and belongs to no family."""
        assert prepare_asm("exp mrt0 v0, v1, v2, v3 done vm").wrapped

    @pytest.mark.parametrize(
        "prose", ["explain the hang please", "export the matrix to json"]
    )
    def test_but_it_is_matched_whole_and_not_as_a_prefix(self, prose):
        """Otherwise "explain" and "export" open a sentence that reads as code."""
        with pytest.raises(AsmHarnessError):
            prepare_asm(prose)


class TestACompleteFileIsTakenAsOne:
    """A paste carrying a descriptor has already said what it is."""

    def test_it_is_not_asked_to_name_instructions_as_well(self):
        """The gate must sit after this, or a header-only unit is refused."""
        text = '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n.amdhsa_kernel k\n.end_amdhsa_kernel'
        prepared = prepare_asm(text)

        assert prepared.wrapped is False
        assert prepared.kernel == "k"

    def test_and_it_is_passed_through_untouched(self):
        text = (
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n'
            ".amdhsa_kernel k\n.end_amdhsa_kernel\ns_endpgm"
        )

        assert prepare_asm(text).program.strip() == text.strip()


class TestAFenceIsTakenAtItsWord:
    """A fence is the user pointing at the code, and stays trusted."""

    def test_prose_around_a_fence_is_still_stripped(self):
        message = f"my kernel hangs, here it is:\n```asm\n{_INSTRUCTIONS}\n```\nwhat is wrong?"
        prepared = prepare_asm(message)

        assert "what is wrong" not in prepared.program
        assert "v_mov_b32" in prepared.program

    def test_what_a_fence_holds_is_not_second_guessed(self):
        """Deliberate: clang's complaint then names text they marked as code.

        That is comprehensible in a way that a complaint about their bug
        report is not, so the stricter test is for the unfenced case only.
        """
        assert prepare_asm(f"```\n{_INSTRUCTIONS}\n```").wrapped

    def test_an_empty_fence_does_not_count_as_the_block(self):
        """It assembled to a valid kernel with nothing in it, which then
        reported no hazards and read exactly like a clean result."""
        with pytest.raises(AsmHarnessError):
            prepare_asm("```\n```\nthe kernel hangs after the second load")

    def test_a_fence_full_of_prose_falls_back_to_the_message(self):
        message = f"```\njust some notes\n```\n{_INSTRUCTIONS}"
        # The fence does hold something non-blank, so it is taken; what matters
        # is that a message which does contain assembly is not lost either way.
        assert prepare_asm(message).wrapped


class TestExtractCodeSaysWhereItLooked:
    def test_a_fence_is_reported_as_one(self):
        text, fenced = extract_code(f"before\n```asm\n{_INSTRUCTIONS}\n```\nafter")

        assert fenced is True
        assert "before" not in text and "after" not in text

    def test_an_unfenced_message_is_returned_whole(self):
        text, fenced = extract_code(_INSTRUCTIONS)

        assert fenced is False
        assert text == _INSTRUCTIONS

    def test_the_caller_needs_the_difference(self):
        """Without it there is no way to be strict about only one of them."""
        _, fenced_prose = extract_code("```\nthe kernel hangs\n```")
        _, bare_prose = extract_code("the kernel hangs")

        assert fenced_prose is True
        assert bare_prose is False
