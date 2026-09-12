"""Turn pasted AMD GPU assembly into something the assembler will accept.

Engineers paste the part they are worried about -- a prologue, an inner loop --
not a whole translation unit. That fragment has no target directive, no kernel
descriptor and no symbol size, so the assembler rejects it, and even when it does
not, a kernel whose ELF symbol has size zero gives the analyser nothing to read.

This wraps a fragment in the smallest valid kernel around it, sizing the register
budget from the registers the fragment actually names. A paste that is already a
complete translation unit is passed through untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aorta.chat.tools.harness._fences import extract_fenced

WAVEFRONT = 64

# s5, s[10:11], v2, v[0:3] -- capture the highest index each one implies.
_SGPR = re.compile(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b")
_VGPR = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")
# A fragment that already carries a descriptor is a whole file, not a fragment.
_COMPLETE = ("amdhsa_kernel", ".amdgcn_target")
# Lines that are directives or labels rather than instructions.
_NOT_AN_INSTRUCTION = re.compile(r"^\s*(\.|//|;|#|\w+:)")
# What AMD GCN/CDNA instruction names begin with. A line starting with one of
# these is assembly and nothing else; no English sentence opens with "v_mov_b32".
_MNEMONIC_PREFIXES = (
    "s_", "v_", "ds_", "buffer_", "flat_", "global_", "scratch_", "image_", "tbuffer_",
)
# Instruction names that do not fit the prefix scheme. Matched whole, never as
# a prefix: "exp" as a prefix would read "explain the hang" as assembly.
_MNEMONICS = frozenset({"exp"})
# A mnemonic is lower-case and unpunctuated, which is most of what separates
# "v_mov_b32 v0, s4" from "No crash, it just stops being a number."
_MNEMONIC = re.compile(r"^[a-z][a-z0-9_.]*$")


class AsmHarnessError(ValueError):
    """Raised when the paste cannot be made into a kernel."""


@dataclass(frozen=True)
class PreparedAsm:
    program: str
    kernel: str
    wrapped: bool
    sgpr_count: int
    vgpr_count: int


def _highest(pattern: re.Pattern[str], text: str) -> int:
    """Highest register index the text names, or -1 if it names none."""
    best = -1
    for match in pattern.finditer(text):
        lo, hi, single = match.groups()
        for value in (hi, lo, single):
            if value is not None:
                best = max(best, int(value))
    return best


def _looks_like_instructions(text: str) -> bool:
    """Whether any line is a mnemonic rather than a label, comment or fence.

    A fence marker counted as an instruction here, so an empty code block
    assembled into a valid kernel with nothing in it -- which then reports no
    hazards, and reads exactly like a clean result.
    """
    return any(
        line.strip()
        and not line.lstrip().startswith("```")
        and not _NOT_AN_INSTRUCTION.match(line)
        for line in text.splitlines()
    )


def _instruction_lines(text: str) -> list[str]:
    """The lines that are neither blank, fence, directive, label nor comment."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("```")
        and not _NOT_AN_INSTRUCTION.match(line)
    ]


def _parses_as_operation(line: str) -> bool:
    """Whether *line* reads as ``op dst, src`` rather than as a sentence.

    The comma is what a sentence also has, so it is the operands that decide:
    a register or an immediate is one token, and "it just stops being a number"
    is not.
    """
    head, _, rest = line.partition(" ")
    if not _MNEMONIC.match(head) or "," not in rest:
        return False
    operands = [operand.strip() for operand in rest.split(",")]
    return all(operand and " " not in operand for operand in operands)


def _looks_like_assembly(text: str) -> bool:
    """Whether unfenced text is assembly, rather than a description of a bug.

    Stricter than :func:`_looks_like_instructions`, which asks only whether a
    line is *not* a comment or a label -- a question every sentence in every
    bug report also answers yes to. Something has to name an instruction: a
    known mnemonic prefix settles it outright, and failing that most of the
    lines have to read as operations, which is how an unusual mnemonic or a
    macro still gets through.
    """
    lines = _instruction_lines(text)
    if not lines:
        return False
    heads = [line.split()[0] for line in lines]
    if any(head.startswith(_MNEMONIC_PREFIXES) or head in _MNEMONICS for head in heads):
        return True
    return sum(_parses_as_operation(line) for line in lines) * 2 >= len(lines)


def extract_code(text: str) -> tuple[str, bool]:
    """Pull the assembly out of a message that also explains the problem.

    Engineers paste their symptoms around the code, and the model deciding what
    to pass along is a judgement it sometimes gets wrong. Prose reaching the
    assembler produces a wall of syntax errors that says nothing about the bug,
    so a fenced block, when present, wins over everything around it.

    Returns the text and whether a fence produced it. The caller needs to know:
    a fence is the user pointing at the code, and without one the whole message
    is only a guess at where the code was.
    """
    return extract_fenced(text, _looks_like_instructions)


def prepare_asm(source: str, *, kernel_name: str = "pasted_kernel", arch: str = "gfx950") -> PreparedAsm:
    """Wrap a fragment into an assemblable kernel, or pass a whole file through."""
    text, fenced = extract_code(source)
    text = text.strip("\n")
    if not text.strip():
        raise AsmHarnessError("the pasted assembly is empty.")

    if any(marker in text for marker in _COMPLETE):
        name = kernel_name
        found = re.search(r"\.amdhsa_kernel\s+(\S+)", text)
        if found:
            name = found.group(1)
        return PreparedAsm(program=text + "\n", kernel=name, wrapped=False,
                           sgpr_count=0, vgpr_count=0)

    # A fence is the user pointing at the code, so what is inside it is taken
    # as offered. Without one the whole message is here, and most messages are
    # mostly prose -- which wraps into a descriptor and reaches clang as a wall
    # of syntax errors about the sentence, saying nothing about the bug.
    if not fenced and not _looks_like_assembly(text):
        raise AsmHarnessError(
            "no assembly found in the message. Paste the instructions you want "
            "checked -- a fenced ``` block is surest, and a bare sequence like "
            "'v_mov_b32 v0, s4' works too."
        )

    if not _looks_like_instructions(text):
        raise AsmHarnessError(
            "no instructions found in the pasted assembly; paste the instruction "
            "sequence you want checked."
        )

    # The register budget has to cover every register the fragment names, or the
    # assembler rejects the descriptor. Leave headroom for the exec/vcc pair.
    sgpr = max(_highest(_SGPR, text) + 1, 16)
    vgpr = max(_highest(_VGPR, text) + 1, 4)

    body = "\n".join(line.rstrip() for line in text.splitlines())
    if "s_endpgm" not in body:
        body += "\n\ts_endpgm"

    program = f"""\
\t.amdgcn_target "amdgcn-amd-amdhsa--{arch}"
\t.amdhsa_code_object_version 6
\t.text
\t.protected\t{kernel_name}
\t.globl\t{kernel_name}
\t.p2align\t8
\t.type\t{kernel_name},@function
{kernel_name}:
{body}
.Lfunc_end0:
\t.size\t{kernel_name}, .Lfunc_end0-{kernel_name}
\t.section\t.rodata,"a",@progbits
\t.p2align\t6, 0x0
\t.amdhsa_kernel {kernel_name}
\t\t.amdhsa_group_segment_fixed_size 65536
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 64
\t\t.amdhsa_user_sgpr_count 2
\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1
\t\t.amdhsa_system_sgpr_workgroup_id_x 1
\t\t.amdhsa_system_vgpr_workitem_id 0
\t\t.amdhsa_next_free_vgpr {vgpr}
\t\t.amdhsa_next_free_sgpr {sgpr}
\t\t.amdhsa_accum_offset {max(4, (vgpr + 3) // 4 * 4)}
\t\t.amdhsa_reserve_vcc 0
\t\t.amdhsa_float_denorm_mode_32 3
\t\t.amdhsa_float_denorm_mode_16_64 3
\t\t.amdhsa_dx10_clamp 1
\t\t.amdhsa_ieee_mode 1
\t.end_amdhsa_kernel
"""
    return PreparedAsm(program=program, kernel=kernel_name, wrapped=True,
                       sgpr_count=sgpr, vgpr_count=vgpr)
