"""The wrapper and the compiler have to name the same GPU.

Assembling a pasted fragment involves two statements of architecture, made in
different places: the `.amdgcn_target` directive in the unit we synthesise
around the paste, and the `-mcpu` we hand clang. Only the second read
CIA_GPU_ARCH, so on any cluster not running gfx950 they named different chips.

What that costs is diagnostic rather than cosmetic. The assemble fails on a
target mismatch, and the failure is about a directive the user never wrote --
so the reported error sends them looking for a mistake in a paste that was
fine. Where it does assemble, the register and hazard analysis that follows
describes an ISA nobody asked about, which is worse, because it reads as an
answer.

The tests that matter here drive the tool, not the harness. Asserting that
prepare_asm honours an arch it was handed passes just as well when the tool
never hands it one, which is the bug.
"""

from __future__ import annotations

import importlib

import pytest

# Every case here reaches aorta.chat.tools.cluster, which imports the agents
# and therefore DSPy. The chat lane installs [chat-cli] without [cia].
pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")

from aorta.chat.tools.harness.assembly import prepare_asm

_FRAGMENT = """\
s_load_dword s4, s[0:1], 0x10
v_mov_b32 v0, s4
s_endpgm
"""

_ARCHES = ["gfx950", "gfx942", "gfx90a"]


@pytest.fixture(autouse=True)
def _restore_the_arch():
    """Leave the setting as this session found it.

    The value is read per call now rather than captured at import, so what has
    to be put back is the cached settings object, not a reloaded module.
    """
    from aorta.chat.config import reset_settings

    yield
    reset_settings()


def _target_of(program: str) -> str:
    """The chip named by the unit's .amdgcn_target directive."""
    line = next(l for l in program.splitlines() if ".amdgcn_target" in l)
    return line.split('"')[1].rsplit("--", 1)[1]


def _reload_cluster(monkeypatch, arch: str | None):
    """Point the settings at *arch*, under the agents' own spelling.

    CIA_GPU_ARCH is one of the two names the setting answers to, and the one
    the agents use, so it is worth being the one exercised here.
    """
    from aorta.chat.config import reset_settings

    if arch is None:
        monkeypatch.delenv("CIA_GPU_ARCH", raising=False)
        monkeypatch.delenv("AORTA_CHAT_GPU_ARCH", raising=False)
    else:
        monkeypatch.setenv("CIA_GPU_ARCH", arch)
    reset_settings()
    return importlib.import_module("aorta.chat.tools.cluster")


def _arch_the_tool_asks_for(cluster, monkeypatch) -> str:
    """Run the tool far enough to see what it requests, and no further.

    Intercepting at the seam records the argument and then raises the harness's
    own error, which returns the tool before it stages a file or reaches a
    compute node -- so this stays a unit test while still being the real call.
    """
    seen: dict[str, str] = {}

    def spy(source, *, kernel_name="pasted_kernel", arch="gfx950"):
        seen["arch"] = arch
        raise cluster.AsmHarnessError("stop here")

    monkeypatch.setattr(cluster, "prepare_asm", spy)
    cluster.triage_assembly_source.func(_FRAGMENT)

    assert "arch" in seen, "the tool never called prepare_asm"
    return seen["arch"]


@pytest.mark.parametrize("arch", _ARCHES)
def test_the_tool_asks_for_the_arch_the_cluster_runs(monkeypatch, arch):
    """The regression: the call site passed nothing and took the default."""
    cluster = _reload_cluster(monkeypatch, arch)

    assert _arch_the_tool_asks_for(cluster, monkeypatch) == arch


@pytest.mark.parametrize("arch", _ARCHES)
def test_the_directive_and_the_compiler_flag_agree(monkeypatch, arch):
    """The two lines that disagreed, compared directly.

    The -mcpu side is the module's _ARCH; the directive side is built from the
    arch the tool actually requests. Any future edit that reintroduces a
    literal on either side separates them again.
    """
    cluster = _reload_cluster(monkeypatch, arch)
    requested = _arch_the_tool_asks_for(cluster, monkeypatch)

    assert cluster._arch() == arch
    assert _target_of(prepare_asm(_FRAGMENT, arch=requested).program) == cluster._arch()


def test_an_unset_environment_still_targets_the_default_fleet(monkeypatch):
    """Nothing changes for the common case of an MI355X node."""
    cluster = _reload_cluster(monkeypatch, None)

    assert cluster._arch() == "gfx950"
    assert _arch_the_tool_asks_for(cluster, monkeypatch) == "gfx950"


@pytest.mark.parametrize("arch", _ARCHES)
def test_the_wrapper_names_the_arch_it_was_given(arch):
    """The harness side of the contract the tool now relies on."""
    assert _target_of(prepare_asm(_FRAGMENT, arch=arch).program) == arch


def test_a_complete_paste_keeps_the_target_its_author_wrote(monkeypatch):
    """We supply an architecture, we do not overrule one.

    A pasted compilable unit carries its own .amdgcn_target. Rewriting it to
    match the cluster would silently change which ISA the user is asking
    about; leaving it means clang refuses the pair, which is the truth --
    gfx90a code does not run on a gfx950 node.
    """
    pasted = (
        '.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"\n'
        ".amdhsa_kernel my_kernel\n.end_amdhsa_kernel\ns_endpgm\n"
    )

    prepared = prepare_asm(pasted, arch="gfx950")

    assert not prepared.wrapped
    assert _target_of(prepared.program) == "gfx90a"


def test_the_tool_does_not_promise_a_chip_to_the_model():
    """The docstring is the tool description the model reads.

    It named gfx950 in prose while the code targeted whatever the environment
    said, so on a gfx942 cluster the model would report an assemble for the
    wrong chip in perfect confidence.
    """
    cluster = importlib.import_module("aorta.chat.tools.cluster")

    for tool in (cluster.triage_assembly_source, cluster.triage_kernel_source):
        assert "gfx" not in (tool.func.__doc__ or ""), f"{tool.name} names a chip"
