"""Does the selector actually choose well, against a real model.

Everything else about tool choice can be tested offline: the parsing, the cap,
the structural filter, the widening on failure. None of it answers the only
question that matters -- given a problem described in a user's own words, does
the model reach for an instrument that will answer it.

That needs a model, so this is marked ``integration`` and skips when there is
none. The last case is the important one: a bug written after the catalogue
was, phrased in a way nobody anticipated. Passing the first three proves the
descriptions are memorable; passing the fourth proves they are meaningful.

It lives here rather than under ``tests/chat/`` because that suite deletes
every ``AORTA_CHAT_*`` variable before each test, so a developer's endpoint
cannot leak into a unit test. Correct for those, and the exact opposite of
what this one needs: real configuration is the point.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from aorta.chat.graph.nodes import selector_node

pytestmark = pytest.mark.integration

_ACTS_ON_PASTED_CODE = {"triage_kernel_source", "triage_assembly_source", "triage_workload"}

_CASES = {
    "pasted HIP kernel, nondeterministic": (
        "Different results run to run, never with one warp per block.\n"
        "```\n"
        "__global__ void reduce_sum(const float* in, float* out) {\n"
        "  __shared__ float p[256]; p[threadIdx.x] = in[threadIdx.x];\n"
        "  if (threadIdx.x == 0) { float s = 0; for (int i = 0; i < 256; ++i) s += p[i]; out[0] = s; }\n"
        "}\n"
        "```"
    ),
    "pasted assembly, consumed before the load landed": (
        "Our hand-tuned prologue returns the wrong value:\n"
        "```asm\n"
        "s_load_dword s4, s[0:1], 0x10\n"
        "s_load_dwordx4 s[4:7], s[0:1], 0x0\n"
        "v_mov_b32 v0, s4\n"
        "```"
    ),
    "pasted training code, value stops being finite": (
        "My loss stops being a number around step 5. No crash.\n"
        "```python\n"
        "class RMSNorm(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))\n"
        "```"
    ),
    # Written after the catalogue, and phrased without naming what is wrong.
    "unseen transpose, wrong for some blocks": (
        "Tile transpose gives garbage for some blocks but is fine with one warp:\n"
        "```\n"
        "__global__ void transpose(float* o, const float* i) {\n"
        "  __shared__ float t[32][32];\n"
        "  t[threadIdx.y][threadIdx.x] = i[blockIdx.x * 32 + threadIdx.x];\n"
        "  o[threadIdx.x * 32 + threadIdx.y] = t[threadIdx.x][threadIdx.y];\n"
        "}\n"
        "```"
    ),
}


@pytest.fixture(scope="module")
def a_model_is_reachable() -> None:
    """Skip rather than fail when there is nothing to ask."""
    from aorta.chat.inference.providers.factory import get_backend

    try:
        get_backend().get_chat_model(temperature=0.0, streaming=False)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no chat backend configured: {exc}")


@pytest.mark.parametrize("problem", _CASES.values(), ids=list(_CASES))
async def test_a_pasted_problem_selects_a_tool_that_reads_it(problem: str, a_model_is_reachable):
    """The top pick must act on what the user pasted, not on something else.

    Which of the three is arguable -- a kernel could reasonably go to the
    sanitizer or to a run. Choosing one that ignores the paste is not.
    """
    out = await selector_node({"messages": [HumanMessage(content=problem)]})
    candidates = out["candidate_tools"]
    assert candidates, f"selector proposed nothing. Rationale: {out['selection_rationale']!r}"
    assert candidates[0] in _ACTS_ON_PASTED_CODE, (
        f"top pick {candidates[0]!r} does not read the pasted code. "
        f"Ranked: {candidates}. Reason given: {out['selection_rationale']!r}"
    )


async def test_a_performance_question_does_not_start_a_diagnosis(a_model_is_reachable):
    """The counterpart, and the false positive a keyword matcher cannot avoid.

    'inf' is a substring of 'inference'. A throughput note is not a fault, and
    starting a cluster job over one costs an engineer real time.
    """
    out = await selector_node(
        {"messages": [HumanMessage(content="Inference throughput dropped on the new build.")]}
    )
    assert not (set(out["candidate_tools"]) & _ACTS_ON_PASTED_CODE), (
        f"a throughput question selected {out['candidate_tools']}"
    )
