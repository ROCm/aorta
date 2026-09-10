"""A misrouted paste is pointed at a tool that exists.

Pasting a PyTorch model into ``triage_kernel_source`` fails to compile — ConSan
needs a ``__global__`` function — so the tool returns a redirect rather than an
error the turn dead-ends on. The redirect named ``run_nan_demo``, one of the
five canned tools this PR deliberately did not port. It is nowhere in the tree,
so the only thing the instruction could produce was a hallucinated tool call:
the hint caused the failure it was written to prevent.

``triage_workload`` is the general replacement — it runs pasted code on a node
and Watch reads the log, which is where a non-finite loss appears.
"""

from __future__ import annotations

import pathlib

import pytest

from aorta.chat.plugins import _DIAGNOSTIC_TOOL_NAMES
from aorta.chat.tools.cluster import _wrong_tool_hint

CLUSTER_SRC = pathlib.Path("src/aorta/chat/tools/cluster.py")

PYTORCH = """
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
"""

HIP_KERNEL = """
__global__ void racy(float* out){
  __shared__ float tile[256];
  tile[threadIdx.x] = threadIdx.x;
  out[threadIdx.x] = tile[(threadIdx.x+1)%256];
}
"""


class TestTheRedirect:
    def test_a_pasted_model_is_redirected(self):
        assert _wrong_tool_hint(PYTORCH)

    def test_it_names_triage_workload(self):
        assert "triage_workload" in _wrong_tool_hint(PYTORCH)

    def test_it_names_a_tool_that_is_registered(self):
        """The whole finding: the previous target did not exist."""
        hint = _wrong_tool_hint(PYTORCH)
        named = [name for name in _DIAGNOSTIC_TOOL_NAMES if name in hint]
        assert named, f"the hint names no registered tool: {hint!r}"

    def test_it_does_not_name_the_unported_tool(self):
        assert "run_nan_demo" not in _wrong_tool_hint(PYTORCH)

    def test_it_still_tells_the_model_not_to_ask_for_a_kernel(self):
        """The behaviour the hint exists for: do not dead-end on a request."""
        assert "Do not ask the user for kernel source" in _wrong_tool_hint(PYTORCH)


class TestItStaysQuietOtherwise:
    def test_a_hip_kernel_gets_no_redirect(self):
        assert _wrong_tool_hint(HIP_KERNEL) == ""

    def test_empty_source_gets_no_redirect(self):
        assert _wrong_tool_hint("") == ""


class TestTheRemovedToolsLeftNothingBehind:
    """Scaffolding from the five unported tools, kept only by nothing reading it."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "run_nan_demo",
            "KNOWN_RECIPES",
            "WAITCHECK_RECIPE",
            "_format_waitcheck",
            "_NAN_CACHE",
            "_WAITCHECK_CACHE",
            "_NAN_VALUE_RE",
            "_NAN_ASSERT_RE",
            "_NAN_WAVE_RE",
            "_NAN_KERNEL_RE",
        ],
    )
    def test_it_is_gone(self, symbol):
        assert symbol not in CLUSTER_SRC.read_text(encoding="utf-8")

    def test_the_tools_that_replaced_them_are_still_here(self):
        import aorta.chat.tools.cluster as cluster

        for name in _DIAGNOSTIC_TOOL_NAMES:
            assert hasattr(cluster, name), name
