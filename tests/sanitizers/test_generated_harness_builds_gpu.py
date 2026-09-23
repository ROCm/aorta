"""The harness this package writes has to compile, not just look right.

Every other test of :mod:`aorta.chat.tools.harness.kernel` reads the generated
text: that a fill reaches the memset, that a guarded kernel is caught, that the
geometry is what was asked for. None of them compiles it. A change to the
generated code that produced invalid HIP would pass all of them and fail in
front of a user, on a GPU node, several minutes into a triage.

That is not hypothetical for the code this guards. The fill byte is rendered
directly into ``hipMemset(ptr, <fill>, ...)`` and the launch geometry into
``kernel<<<grid, dim3(...)>>>``; both are string interpolation into C++ that
nothing on a CPU runner ever parses.

Marked ``gpu`` and ``rocm``, so it runs in the MI350 job and nowhere else. It
skips rather than fails when hipcc is absent, because a missing toolchain is a
fact about the runner and not about this package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# Here rather than under tests/chat/, which is where the code it covers lives.
# That directory's conftest ignores itself when langchain_core is absent, while
# the GPU image intentionally installs [tests,hw-queue] without chat extras.
# In tests/chat this lane went green having collected none of these tests.
#
# The harness is stdlib string generation and needs no chat dependency,
# asserted below. tests/sanitizers/ is ungated and already activates GPU CI.
from aorta.chat.tools.harness.kernel import prepare_source

pytestmark = [pytest.mark.gpu, pytest.mark.rocm]

#: The architecture the cluster tools build for. Overridable because the runner
#: is not necessarily the cluster: a gfx942 box should test its own target
#: rather than fail on one it cannot produce.
ARCH = os.environ.get("CIA_GPU_ARCH", "gfx950")

#: Reads an input inside a branch, so the fill is load-bearing: with zeros the
#: guarded store never runs, which is the case `fill` exists for.
GUARDED = (
    "__global__ void scale(const float* in, float* out) {\n"
    "  int i = threadIdx.x;\n"
    "  if (in[i] > 0) { out[i] = in[i] * 2.0f; }\n"
    "}"
)

#: Two dimensions, so the generated launch geometry is exercised rather than
#: the one-dimensional default.
TILED = (
    "__global__ void transpose(const float* in, float* out) {\n"
    "  __shared__ float tile[16][16];\n"
    "  tile[threadIdx.y][threadIdx.x] = in[threadIdx.y * 16 + threadIdx.x];\n"
    "  __syncthreads();\n"
    "  out[threadIdx.x * 16 + threadIdx.y] = tile[threadIdx.y][threadIdx.x];\n"
    "}"
)


def _hipcc() -> str:
    found = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    if not Path(found).is_file():
        pytest.skip("hipcc is not on this runner")
    return found


def _build_and_run(program: str) -> subprocess.CompletedProcess:
    """Compile a generated harness for ARCH and run it."""
    compiler = _hipcc()
    with tempfile.TemporaryDirectory(prefix="aorta-harness-") as work:
        source = Path(work) / "harness.hip"
        binary = Path(work) / "harness.bin"
        source.write_text(program, encoding="utf-8")
        built = subprocess.run(
            [compiler, f"--offload-arch={ARCH}", "-o", str(binary), str(source)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert built.returncode == 0, (
            f"the generated harness did not compile:\n{built.stderr[-2000:]}"
        )
        return subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=120
        )


class TestTheGeneratedHarnessCompiles:
    def test_a_plain_kernel_builds_and_runs(self):
        prepared = prepare_source(
            "__global__ void bump(float* o) { o[threadIdx.x] += 1.0f; }",
            block=64,
        )
        done = _build_and_run(prepared.program)

        assert done.returncode == 0, done.stderr[-2000:]

    def test_a_guarded_kernel_builds_and_runs(self):
        prepared = prepare_source(GUARDED, block=64)
        done = _build_and_run(prepared.program)

        assert done.returncode == 0, done.stderr[-2000:]

    def test_a_non_zero_fill_still_builds(self):
        """The fill is interpolated into hipMemset; only a compiler checks it."""
        prepared = prepare_source(GUARDED, block=64, fill_byte=1)

        assert "hipMemset(h_in, 1," in prepared.program
        done = _build_and_run(prepared.program)

        assert done.returncode == 0, done.stderr[-2000:]

    def test_a_two_dimensional_launch_builds(self):
        """The geometry is interpolated too, and dim3 is easy to get wrong."""
        prepared = prepare_source(TILED, block=16, block_y=16)
        done = _build_and_run(prepared.program)

        assert done.returncode == 0, done.stderr[-2000:]


class TestItCanRunWhereItIsMeantTo:
    def test_the_harness_needs_none_of_the_chat_extras(self):
        """The GPU image has [tests,hw-queue] and nothing else.

        A module-level skip on an extra this module does not use is invisible:
        the job goes green having run nothing. Asserted rather than commented,
        because that is exactly how it was wrong.
        """
        import aorta.chat.tools.harness.kernel as harness

        for absent in ("dspy", "langchain_core", "chainlit"):
            assert absent not in getattr(harness, "__dict__", {}), absent

    def test_it_generates_without_them(self):
        prepared = prepare_source("__global__ void k(float* o){ o[0] = 1.0f; }")

        assert "hipMalloc" in prepared.program


class TestItRunsOnTheHardwareItClaims:
    def test_the_kernel_actually_launched(self):
        """A harness that compiles and silently launches nothing proves little.

        The generated main reports a failed launch on stderr, so an empty
        stderr with a zero exit is the evidence that it ran.
        """
        prepared = prepare_source(GUARDED, block=64, fill_byte=1)
        done = _build_and_run(prepared.program)

        assert done.returncode == 0
        assert "error" not in done.stderr.lower(), done.stderr[-2000:]
