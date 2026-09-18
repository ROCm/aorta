"""Getting the kernel out of the message it arrived in.

The tool contract says the source is passed exactly as the user pasted it, and
what people paste is a kernel with their symptoms around it. The assembly
harness has extracted the fenced block since it was written -- prose reaching
the assembler produces a wall of errors about English -- and the kernel harness
inherited the same wording without the same extraction.

So ``kernel_name`` found the function inside the backticks, the harness was
built around it, and the .hip written to the node still had "any idea what's
wrong?" in it. hipcc failed on the prose, and the user was told their kernel
did not compile.
"""

from __future__ import annotations

import pytest

from aorta.chat.tools.harness.kernel import (
    HarnessError,
    extract_source,
    looks_like_hip,
    prepare_source,
)

KERNEL = """__global__ void reduce(float* out, const float* in) {
    __shared__ float s[256];
    s[threadIdx.x] = in[threadIdx.x];
    out[threadIdx.x] = s[threadIdx.x];
}"""

FENCED = f"""my reduction gives wrong results sometimes, here it is:

```cpp
{KERNEL}
```

any idea what's wrong?"""


class TestTheProseDoesNotReachTheCompiler:
    def test_a_fenced_kernel_is_extracted(self):
        assert extract_source(FENCED).strip() == KERNEL

    def test_the_generated_program_has_no_prose_in_it(self):
        program = prepare_source(FENCED).program

        assert "any idea" not in program
        assert "```" not in program

    def test_the_kernel_is_still_found(self):
        assert prepare_source(FENCED).kernel == "reduce"

    @pytest.mark.parametrize("tag", ["", "cpp", "c++", "hip", "cuda"])
    def test_whatever_the_author_tagged_the_fence(self, tag):
        assert extract_source(f"look:\n```{tag}\n{KERNEL}\n```\n").strip() == KERNEL

    def test_an_unfenced_kernel_is_unchanged(self):
        """Most pastes are bare, and they were already working."""
        assert extract_source(KERNEL).strip() == KERNEL

    def test_a_whole_program_still_passes_through(self):
        program = f"#include <hip/hip_runtime.h>\n{KERNEL}\nint main() {{ return 0; }}"

        assert extract_source(f"```\n{program}\n```").strip() == program


class TestWhichBlockWhenThereAreSeveral:
    """The policy, stated: the first block that is a kernel."""

    def test_the_kernel_wins_over_a_log_pasted_before_it(self):
        message = f"I get this:\n```\nmemory access fault at 0x7f\n```\nfrom:\n```\n{KERNEL}\n```"

        assert extract_source(message).strip() == KERNEL

    def test_the_first_kernel_wins_when_there_are_two(self):
        second = KERNEL.replace("reduce", "reduce_v2")
        message = f"before:\n```\n{KERNEL}\n```\nafter:\n```\n{second}\n```"

        assert extract_source(message).strip() == KERNEL

    def test_a_block_that_is_not_code_is_skipped(self):
        message = f"```\njust some notes about the bug\n```\n```\n{KERNEL}\n```"

        assert extract_source(message).strip() == KERNEL

    def test_with_no_recognisable_block_the_message_is_used_whole(self):
        """So an unfenced paste still works, and a bad paste fails as it did."""
        message = "```\nnothing here is code\n```"

        assert extract_source(message) == message


class TestRecognisingAKernel:
    @pytest.mark.parametrize(
        "block,expected",
        [
            (KERNEL, True),
            ("int main() { return 0; }", True),
            ("__device__ float f(float x) { return x; }", False),
            ("memory access fault at 0x7f0000", False),
            ("Traceback (most recent call last):", False),
            ("", False),
        ],
    )
    def test_what_counts(self, block, expected):
        assert looks_like_hip(block) is expected


class TestTheFailuresThatShouldStillFail:
    def test_an_empty_message_is_still_refused(self):
        with pytest.raises(HarnessError, match="no source"):
            prepare_source("   \n  ")

    def test_a_message_with_no_kernel_is_still_refused(self):
        with pytest.raises(HarnessError):
            prepare_source("my loss goes to NaN after a few steps")


class TestTheAssemblyHarnessIsUnaffected:
    """It had this behaviour first; sharing the rule must not change it."""

    def test_it_still_extracts_its_own_block(self):
        from aorta.chat.tools.harness.assembly import extract_code

        text, fenced = extract_code("here:\n```asm\ns_load_dword s4, s[0:1], 0x0\n```\nwhy?")

        assert fenced is True
        assert text.strip() == "s_load_dword s4, s[0:1], 0x0"

    def test_an_unfenced_paste_still_reports_no_fence(self):
        from aorta.chat.tools.harness.assembly import extract_code

        text, fenced = extract_code("s_endpgm")

        assert (text, fenced) == ("s_endpgm", False)

    def test_a_kernel_fence_is_not_mistaken_for_assembly(self):
        """Each harness keeps its own idea of what its block looks like."""
        from aorta.chat.tools.harness.assembly import extract_code

        _text, fenced = extract_code(f"```cpp\n{KERNEL}\n```")

        assert fenced is True, "a fence is still a fence"
