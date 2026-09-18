"""A clean sweep with generated inputs is not proof of a clean kernel.

The generated harness chooses every buffer and scalar value, so it exercises
one path. A direct ``if (input[i] > 0)`` is easy to notice; a scalar mode,
pointer-derived flag, ternary, switch, or helper condition is not. No regex can
prove that all data-dependent paths ran, so the caveat applies to every
generated main rather than only to source shapes the regex recognizes.

This is the same shape as the single-wave caveat already beside it: a geometry
that cannot show a cross-wave race returns clean for a reason that has nothing
to do with the kernel being safe. Both need saying, because the sanitizer
reports what it found and not what it could not have found.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="the harness needs the [cia] extra")

from aorta.chat.tools.harness.kernel import Param, branches_on_input, prepare_source

GUARDED = (
    "__global__ void k(const float* in, float* out) {\n"
    "  int i = threadIdx.x;\n"
    "  if (in[i] > 0) { out[i] = in[i] * 2.0f; }\n"
    "}"
)
WRITES_ONLY = (
    "__global__ void k(float* out) {\n"
    "  int i = threadIdx.x;\n"
    "  out[i] = 1.0f;\n"
    "}"
)
INDIRECT = (
    "__global__ void k(const float* in, float* out) {\n"
    "  int i = threadIdx.x;\n"
    "  bool active = in[i] > 0;\n"
    "  if (active) { out[i] = in[i] * 2.0f; }\n"
    "}"
)
SCALAR_GUARD = (
    "__global__ void k(float* out, int mode) {\n"
    "  int i = threadIdx.x;\n"
    "  if (mode == 1) { out[i] = 1.0f; }\n"
    "}"
)
TERNARY = (
    "__global__ void k(const float* in, float* out) {\n"
    "  int i = threadIdx.x;\n"
    "  out[i] = in[i] > 0 ? in[i] : 0.0f;\n"
    "}"
)
SWITCH = (
    "__global__ void k(float* out, int mode) {\n"
    "  switch (mode) { case 1: out[threadIdx.x] = 1.0f; break; }\n"
    "}"
)


class TestItNoticesADataDependentBranch:
    def test_a_guarded_read_is_found(self):
        assert prepare_source(GUARDED).input_guards == ("in",)

    def test_that_makes_the_result_qualified(self):
        assert prepare_source(GUARDED).data_dependent is True

    def test_a_while_loop_counts_too(self):
        source = (
            "__global__ void k(int* d) {\n"
            "  int i = threadIdx.x;\n"
            "  while (d[i] != 0) { d[i]--; }\n"
            "}"
        )

        assert prepare_source(source).input_guards == ("d",)

    def test_several_inputs_are_all_named(self):
        source = (
            "__global__ void k(const int* a, const int* b, int* out) {\n"
            "  int i = threadIdx.x;\n"
            "  if (a[i] > 0) out[i] = 1;\n"
            "  if (b[i] > 0) out[i] = 2;\n"
            "}"
        )

        assert set(prepare_source(source).input_guards) == {"a", "b"}


class TestItDoesNotCryWolf:
    def test_a_kernel_that_only_writes_is_not_flagged(self):
        assert prepare_source(WRITES_ONLY).data_dependent is False

    def test_branching_on_the_index_is_not_branching_on_input(self):
        """`if (i < n)` is a bounds check, not a data dependence."""
        source = (
            "__global__ void k(float* out) {\n"
            "  int i = threadIdx.x;\n"
            "  if (i < 10) out[i] = 1.0f;\n"
            "}"
        )

        assert prepare_source(source).data_dependent is False

    def test_a_kernel_with_no_pointers_is_not_flagged(self):
        source = "__global__ void k(int n) { if (n > 0) { } }"

        assert prepare_source(source).input_guards == ()

    def test_a_caller_supplied_main_is_not_flagged(self):
        """Their harness, their inputs; this caveat is about the generated one."""
        source = GUARDED + "\nint main() { return 0; }\n"
        prepared = prepare_source(source)

        assert prepared.wrapped is False
        assert prepared.data_dependent is False
        assert prepared.generated_inputs is False


class TestTheRegexDoesNotDecideHonesty:
    @pytest.mark.parametrize(
        "source",
        [INDIRECT, SCALAR_GUARD, TERNARY, SWITCH],
        ids=["pointer-derived", "scalar", "ternary", "switch"],
    )
    def test_common_indirect_paths_are_not_proven_independent(self, source):
        prepared = prepare_source(source)

        # The lightweight scan may miss these; generated_inputs must not.
        assert prepared.input_guards == ()
        assert prepared.generated_inputs is True

    def test_even_a_write_only_kernel_uses_generated_inputs(self):
        assert prepare_source(WRITES_ONLY).generated_inputs is True


class TestTheFillIsReachable:
    def test_zero_is_the_default(self):
        assert "hipMemset(h_in, 0," in prepare_source(GUARDED).program

    def test_a_caller_can_ask_for_non_zero(self):
        """The review's other branch: accept explicit initialisation."""
        assert "hipMemset(h_in, 1," in prepare_source(GUARDED, fill_byte=1).program

    def test_it_applies_to_every_buffer(self):
        source = "__global__ void k(const float* a, float* b) { b[0] = a[0]; }"
        program = prepare_source(source, fill_byte=1).program

        assert "hipMemset(h_a, 1," in program
        assert "hipMemset(h_b, 1," in program


class TestTheHelperItself:
    def test_it_takes_the_parsed_parameters(self):
        params = [Param(base_type="float", name="in", is_pointer=True)]

        assert branches_on_input("if (in[0] > 0) {}", params) == ["in"]

    def test_a_name_that_merely_appears_is_not_a_read(self):
        """`in` in a comment or an unrelated call is not a guarded read."""
        params = [Param(base_type="float", name="in", is_pointer=True)]

        assert branches_on_input("if (threadIdx.x < 4) { use(in); }", params) == []

    def test_no_pointers_means_no_work(self):
        assert branches_on_input("if (n > 0) {}", []) == []


@pytest.fixture()
def triage(tmp_path, monkeypatch):
    """Run the tool with the cluster stubbed, and put everything back after.

    monkeypatch rather than assignment: an earlier version of this set
    ``cluster._run_triage`` directly and never restored it, so every test that
    ran afterwards in the same process got this stub. Ten of them failed, in
    three unrelated files, and all of them passed on their own.
    """
    import aorta.chat.tools.cluster as cluster
    from aorta.chat.config import reset_settings

    monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
    monkeypatch.setenv("AORTA_CHAT_ALLOW_CLUSTER_JOBS", "true")
    reset_settings()
    monkeypatch.setattr(
        cluster, "_run_triage", lambda args, label: "Autopsy verdict:\n  category: clean"
    )

    def run(source: str, **kwargs) -> str:
        return cluster.triage_kernel_source.func(source, **kwargs)

    yield run
    reset_settings()


class TestTheCaveatReachesTheUser:
    def test_a_guarded_kernel_is_warned_about(self, triage):
        out = triage(GUARDED, block_size=256)

        assert "generated-input caveat" in out
        assert "INCONCLUSIVE" in out

    def test_it_names_the_input_and_the_way_out(self, triage):
        out = triage(GUARDED, block_size=256)

        assert "`in`" in out
        assert "main()" in out

    @pytest.mark.parametrize(
        "source",
        [INDIRECT, SCALAR_GUARD, TERNARY, SWITCH],
        ids=["pointer-derived", "scalar", "ternary", "switch"],
    )
    def test_every_generated_harness_is_qualified(self, triage, source):
        out = triage(source, block_size=256)

        assert "generated-input caveat" in out
        assert "INCONCLUSIVE" in out

    def test_a_write_only_kernel_is_still_qualified(self, triage):
        out = triage(WRITES_ONLY, block_size=256)

        assert "generated-input caveat" in out

    def test_a_nonzero_fill_explores_but_does_not_qualify_a_pass(self, triage):
        out = triage(GUARDED, block_size=256, fill=1)

        assert "generated-input caveat" in out
        assert "cannot make a clean result representative" in out

    def test_a_caller_supplied_main_needs_no_generated_input_caveat(self, triage):
        out = triage(GUARDED + "\nint main() { return 0; }\n")

        assert "generated-input caveat" not in out
