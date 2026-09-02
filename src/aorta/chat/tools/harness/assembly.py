"""Turn pasted gfx950 assembly into something the assembler will accept.

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

WAVEFRONT = 64

# s5, s[10:11], v2, v[0:3] -- capture the highest index each one implies.
_SGPR = re.compile(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b")
_VGPR = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")
# A fragment that already carries a descriptor is a whole file, not a fragment.
_COMPLETE = ("amdhsa_kernel", ".amdgcn_target")
# Lines that are directives or labels rather than instructions.
_NOT_AN_INSTRUCTION = re.compile(r"^\s*(\.|//|;|#|\w+:)")
# ```asm ... ``` or a bare fence, non-greedy so several blocks stay separate.
_FENCED = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)


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


def extract_code(text: str) -> str:
    """Pull the assembly out of a message that also explains the problem.

    Engineers paste their symptoms around the code, and the model deciding what
    to pass along is a judgement it sometimes gets wrong. Prose reaching the
    assembler produces a wall of syntax errors that says nothing about the bug,
    so a fenced block, when present, wins over everything around it.
    """
    blocks = _FENCED.findall(text)
    for block in blocks:
        if _looks_like_instructions(block):
            return block.strip("\n")
    return text


def prepare_asm(source: str, *, kernel_name: str = "pasted_kernel", arch: str = "gfx950") -> PreparedAsm:
    """Wrap a fragment into an assemblable kernel, or pass a whole file through."""
    text = extract_code(source).strip("\n")
    if not text.strip():
        raise AsmHarnessError("the pasted assembly is empty.")

    if any(marker in text for marker in _COMPLETE):
        name = kernel_name
        found = re.search(r"\.amdhsa_kernel\s+(\S+)", text)
        if found:
            name = found.group(1)
        return PreparedAsm(program=text + "\n", kernel=name, wrapped=False,
                           sgpr_count=0, vgpr_count=0)

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
