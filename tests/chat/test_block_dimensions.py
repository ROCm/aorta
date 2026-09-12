"""A launch that does not match the kernel proves nothing about it.

The generated harness was always one-dimensional. A kernel written for a 32x32
tile -- the transpose in this PR's own examples -- was launched as
``<<<1, 32>>>``, which does not fail: it runs with ``threadIdx.y`` pinned at
zero, so every thread touches row zero of the tile and the rows that were
supposed to race with each other never both run.

The result is a clean sanitizer report about a kernel nobody asked to check,
which is worse than an error. Nothing in the source says how wide the second
dimension should be, so the harness asks instead of guessing.
"""

from __future__ import annotations

import pytest

from aorta.chat.tools.harness.kernel import (
    HarnessError,
    Prepared,
    dims_used,
    prepare_source,
)

TRANSPOSE = """
__global__ void transpose(float* out, const float* in, int n) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    out[x * n + y] = tile[threadIdx.x][threadIdx.y];
}
"""

ONE_D = """
__global__ void bump(float* o) {
    __shared__ float s[256];
    s[threadIdx.x] = o[threadIdx.x];
    o[threadIdx.x] = s[threadIdx.x];
}
"""

THREE_D = """
__global__ void stencil(float* o) {
    int i = threadIdx.x + threadIdx.y + threadIdx.z;
    o[i] = i;
}
"""


def _launch(program: str, kernel: str) -> str:
    return next(line.strip() for line in program.splitlines() if f"{kernel}<<<" in line)


class TestAKernelIsNotLaunchedFlat:
    def test_a_two_dimensional_kernel_is_refused_without_its_second_dimension(self):
        with pytest.raises(HarnessError, match="threadIdx.y"):
            prepare_source(TRANSPOSE)

    def test_the_refusal_says_what_to_pass(self):
        """A refusal the model cannot act on is a dead end of its own."""
        with pytest.raises(HarnessError) as raised:
            prepare_source(TRANSPOSE)

        message = str(raised.value)
        assert "block_y" in message
        assert "32" in message, "an example is what makes it actionable"

    def test_and_why_it_matters(self):
        with pytest.raises(HarnessError, match="zero"):
            prepare_source(TRANSPOSE)

    def test_a_three_dimensional_kernel_is_refused_too(self):
        with pytest.raises(HarnessError) as raised:
            prepare_source(THREE_D)

        assert "block_y" in str(raised.value)
        assert "block_z" in str(raised.value)

    def test_supplying_only_one_of_two_is_still_refused(self):
        with pytest.raises(HarnessError, match="threadIdx.z"):
            prepare_source(THREE_D, block=8, block_y=8)


class TestWithTheDimensionsGiven:
    def test_the_launch_is_two_dimensional(self):
        prepared = prepare_source(TRANSPOSE, block=32, block_y=32)

        assert "dim3(32, 32)" in _launch(prepared.program, "transpose")

    def test_three_dimensions_reach_the_launch(self):
        prepared = prepare_source(THREE_D, block=8, block_y=8, block_z=4)

        assert "dim3(8, 8, 4)" in _launch(prepared.program, "stencil")

    def test_a_block_past_the_hardware_limit_is_refused(self):
        """1024 threads is the ceiling; 32x64 would fail at launch instead."""
        with pytest.raises(HarnessError, match="1024"):
            prepare_source(TRANSPOSE, block=32, block_y=64)

    def test_the_geometry_is_recorded(self):
        prepared = prepare_source(TRANSPOSE, block=32, block_y=32)

        assert (prepared.block, prepared.block_y) == (32, 32)


class TestOneDimensionalKernelsAreUnchanged:
    """Most pastes are 1D and must not pay for this."""

    def test_no_dimensions_are_demanded(self):
        assert prepare_source(ONE_D).kernel == "bump"

    def test_the_launch_stays_a_plain_scalar(self):
        """dim3(256, 1, 1) would invite the question of what the ones are for."""
        prepared = prepare_source(ONE_D)

        assert "<<<1, 256>>>" in _launch(prepared.program, "bump")

    def test_the_inferred_block_size_still_applies(self):
        assert prepare_source(ONE_D).block == 256


class TestTheWavefrontWarningCountsTheWholeBlock:
    """It read the x extent, which for 32x32 is a sixteenth of the threads."""

    def test_a_two_dimensional_block_is_not_called_single_wave(self):
        prepared = prepare_source(TRANSPOSE, block=32, block_y=32)

        assert prepared.threads == 1024
        assert not prepared.single_wave

    def test_a_genuinely_small_block_still_warns(self):
        assert Prepared("", "k", True, 32, 1).single_wave

    def test_a_small_block_in_two_dimensions_warns(self):
        """8x8 is 64 threads: one wavefront, however it is spelled."""
        assert Prepared("", "k", True, 8, 1, 8).single_wave

    def test_just_past_a_wavefront_does_not(self):
        assert not Prepared("", "k", True, 8, 1, 9).single_wave


class TestSpottingTheDimensions:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("o[threadIdx.x] = 1;", []),
            ("o[threadIdx.y] = 1;", ["y"]),
            ("o[blockDim.y] = 1;", ["y"]),
            ("o[blockIdx.y] = 1;", ["y"]),
            ("o[threadIdx.z] = 1;", ["z"]),
            ("o[threadIdx.y + threadIdx.z] = 1;", ["y", "z"]),
        ],
    )
    def test_which_axes_a_kernel_indexes(self, source, expected):
        assert dims_used(source) == expected

    def test_a_variable_ending_in_y_is_not_an_axis(self):
        """``my.y`` is somebody's struct field, not a thread index."""
        assert dims_used("float v = point.y; o[threadIdx.x] = v;") == []
