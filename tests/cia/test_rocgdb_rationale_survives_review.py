"""An LLM review must not erase ROCgDB's deterministic root cause and fix."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("dspy", reason="Autopsy routing needs the [cia] extra")

from aorta.cia.autopsy.orchestrator import run_autopsy

SESSION = (
    'Thread 7 "probe_trap" received signal SIGABRT, Aborted.\n'
    '__assert_fail (assertion=0x1 <str> "isfinite(y)", '
    'file=0x2 <str> "rmsnorm.hip", line=74, '
    "function=<rmsnorm_forward>) at hip_assert.h:85\n"
    "===== DEVICE ASSERT TRAPPED =====\n"
    '* 7  AMDGPU Wave 1:1:1:2 (5,0,0)/0 "probe_trap"\n'
    "#1  0x7fff in rmsnorm_forward(float const*, float*) at rmsnorm.hip:74\n"
    "nan-trap value row=5\n"
    "nan-trap value col_index=0\n"
    "nan-trap value mean_sq=0\n"
    "nan-trap value inv_rms=inf\n"
    "nan-trap value y=-nan\n"
)


class TestTheHardDebuggerConclusionSurvives:
    def test_router_review_cannot_drop_the_epsilon_fix(
        self, make_bundle, monkeypatch
    ):
        bundle = make_bundle(
            stderr="[train] step=5 pad_tokens=32 loss=nan\n",
            rocgdb_session=SESSION,
        )

        class Router:
            def __init__(self, _root):
                pass

            def __call__(self, **_kwargs):
                # Plausible but incomplete: this is what the live run returned.
                return SimpleNamespace(
                    category="numeric_silent",
                    confidence=0.95,
                    rationale=(
                        "ROCgDB captured mean_sq=0, inv_rms=inf and y=-nan "
                        "in the padding-row workgroup."
                    ),
                    next_probe="none",
                    next_probe_reason="",
                )

        monkeypatch.setattr("aorta.cia.autopsy.router.TriageRouter", Router)

        report = run_autopsy(bundle, use_llm=True)

        assert report["category"] == "numeric_silent"
        assert report["confidence"] >= 0.9
        assert "variance epsilon" in report["rationale"]
        assert "rsqrtf(mean_sq + eps)" in report["rationale"]
        assert "rsqrtf(col_index + eps)" not in report["rationale"]
        assert "LLM review:" in report["rationale"]

    def test_rule_based_path_already_contains_the_exact_fix(self, make_bundle):
        bundle = make_bundle(
            stderr="[train] step=5 loss=nan\n",
            rocgdb_session=SESSION,
        )

        report = run_autopsy(bundle, use_llm=False)

        assert "mean_sq is exactly zero" in report["rationale"]
        assert "inv_rms=inf" in report["rationale"]
        assert "rsqrtf(mean_sq + eps)" in report["rationale"]
        assert "rsqrtf(col_index + eps)" not in report["rationale"]
