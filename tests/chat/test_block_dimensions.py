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

import inspect

import pytest

from aorta.chat.tools.harness.kernel import (
    HarnessError,
    Prepared,
    block_dims_used,
    dims_used,
    grid_dims_used,
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

GRID_Y = """
__global__ void rows(float* o) {
    o[blockIdx.y * blockDim.x + threadIdx.x] = 1.0f;
}
"""

GRID_Z = """
__global__ void planes(float* o) {
    o[blockIdx.z * blockDim.x + threadIdx.x] = 1.0f;
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
        assert "grid_y" in message
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

    def test_block_geometry_does_not_satisfy_a_grid_axis(self):
        with pytest.raises(HarnessError, match="grid_y"):
            prepare_source(TRANSPOSE, block=32, block_y=32)

    def test_grid_geometry_does_not_satisfy_a_block_axis(self):
        with pytest.raises(HarnessError, match="block_y"):
            prepare_source(TRANSPOSE, grid=4, grid_y=4)

    def test_a_grid_y_kernel_names_the_right_argument(self):
        with pytest.raises(HarnessError) as raised:
            prepare_source(GRID_Y)

        message = str(raised.value)
        assert "blockIdx.y" in message
        assert "grid_y" in message
        assert "block_y" not in message

    def test_a_grid_z_kernel_names_the_right_argument(self):
        with pytest.raises(HarnessError, match="grid_z"):
            prepare_source(GRID_Z)


class TestWithTheDimensionsGiven:
    def test_the_launch_is_two_dimensional(self):
        prepared = prepare_source(
            TRANSPOSE, block=32, block_y=32, grid=4, grid_y=3
        )

        launch = _launch(prepared.program, "transpose")
        assert "<<<dim3(4, 3), dim3(32, 32)>>>" in launch

    def test_three_dimensions_reach_the_launch(self):
        prepared = prepare_source(THREE_D, block=8, block_y=8, block_z=4)

        assert "dim3(8, 8, 4)" in _launch(prepared.program, "stencil")

    def test_three_grid_dimensions_reach_the_launch(self):
        source = GRID_Y.replace(
            "blockIdx.y * blockDim.x",
            "blockIdx.y * gridDim.z + blockIdx.z",
        )
        prepared = prepare_source(source, grid=2, grid_y=3, grid_z=4)

        assert "<<<dim3(2, 3, 4), 256>>>" in _launch(
            prepared.program, "rows"
        )

    def test_a_z_only_grid_inserts_the_neutral_y_extent(self):
        prepared = prepare_source(GRID_Z, grid=2, grid_z=4)

        assert "<<<dim3(2, 1, 4), 256>>>" in _launch(
            prepared.program, "planes"
        )

    def test_a_block_past_the_hardware_limit_is_refused(self):
        """1024 threads is the ceiling; 32x64 would fail at launch instead."""
        with pytest.raises(HarnessError, match="1024"):
            prepare_source(
                TRANSPOSE, block=32, block_y=64, grid=1, grid_y=1
            )

    def test_the_geometry_is_recorded(self):
        prepared = prepare_source(
            TRANSPOSE, block=32, block_y=32, grid=4, grid_y=3
        )

        assert (prepared.block, prepared.block_y) == (32, 32)
        assert (prepared.grid, prepared.grid_y) == (4, 3)

    def test_generated_buffers_cover_the_full_grid(self):
        prepared = prepare_source(GRID_Y, block=256, grid=20, grid_y=20)

        # 256 threads * 400 blocks * four elements of padding.
        assert "constexpr size_t kElements = 409600;" in prepared.program

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"grid": -1},
            {"grid_y": -1},
            {"grid_z": -1},
        ],
    )
    def test_invalid_grid_extents_are_refused(self, kwargs):
        with pytest.raises(HarnessError, match="positive|at least"):
            prepare_source(ONE_D, **kwargs)


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
        prepared = prepare_source(
            TRANSPOSE, block=32, block_y=32, grid=1, grid_y=1
        )

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

    def test_block_axes_exclude_block_indices(self):
        assert block_dims_used(
            "o[threadIdx.y + blockDim.z + blockIdx.y] = 1;"
        ) == ["y", "z"]

    def test_grid_axes_exclude_thread_indices(self):
        assert grid_dims_used(
            "o[blockIdx.y + gridDim.z + threadIdx.y] = 1;"
        ) == ["y", "z"]


class TestTheToolExposesBothLaunchDomains:
    def test_grid_y_and_z_are_real_tool_arguments(self):
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        from aorta.chat.tools.cluster import triage_kernel_source

        parameters = inspect.signature(triage_kernel_source.func).parameters

        assert "block_y" in parameters and "block_z" in parameters
        assert "grid_y" in parameters and "grid_z" in parameters
