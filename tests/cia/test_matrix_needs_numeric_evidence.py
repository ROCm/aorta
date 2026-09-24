"""A failed cell is not evidence of numeric corruption.

``classify_matrix`` called any cell with a non-zero failure a repro cell, and
a repro cell beside a clean mitigation cell returned ``numeric_silent`` at
0.88. An out-of-memory kill, a missing module and an unrelated non-zero exit
all satisfied that, so a workload that never computed anything was reported as
silently corrupting its arithmetic, confidently.

The check meant to prevent it was written and then not used::

    if is_repro and not NAN_HINTS.search(hints) and not NAN_HINTS.search(name):
        # Numeric silent often lacks explicit hint text; repro cell name is enough.
        pass

which reads like a guard and does nothing.

Being wrong loudly is the problem, not being unsure. A verdict at 0.88 ends an
investigation; "unknown at 0.3" starts one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="the adapters need the [cia] extra")

from aorta.cia.autopsy.adapters.aorta_matrix import classify_matrix

CLEAN_MITIGATION = {
    "name": "tf32_off",
    "failure_rate": 0.0,
    "failed_count": 0,
    "mitigations": ["tf32_off"],
}


def repro(name: str = "repro", **kw) -> dict:
    cell = {"name": name, "failure_rate": 1.0, "failed_count": 4}
    cell.update(kw)
    return cell


def matrix(*cells: dict) -> dict:
    return {"cells": list(cells)}


class TestAFailureThatIsNotNumeric:
    def test_an_oom_is_not_called_numeric_corruption(self):
        found = classify_matrix(
            matrix(repro(failure_hints=["CUDA error: out of memory"]), CLEAN_MITIGATION)
        )

        assert found.category != "numeric_silent"

    def test_an_oom_is_named_as_one(self):
        found = classify_matrix(
            matrix(repro(failure_hints=["torch: HIP out of memory"]), CLEAN_MITIGATION)
        )

        assert found.category == "oom_fragment"

    def test_a_launch_failure_is_named_as_one(self):
        found = classify_matrix(
            matrix(repro(failure_hints=["ModuleNotFoundError: torch"]), CLEAN_MITIGATION)
        )

        assert found.category == "launch_error"

    @pytest.mark.parametrize(
        "hint",
        ["No such file or directory", "command not found", "Permission denied"],
    )
    def test_other_launch_shapes_too(self, hint):
        found = classify_matrix(matrix(repro(failure_hints=[hint]), CLEAN_MITIGATION))

        assert found.category == "launch_error"

    def test_a_failed_inference_cell_is_not_numeric_corruption(self):
        """``inf`` was matching the first three letters of ``inference``."""
        found = classify_matrix(
            matrix(
                repro(
                    "inference_offline",
                    failure_hints=["inference request failed before producing output"],
                ),
                CLEAN_MITIGATION,
            )
        )

        assert found.category == "unknown"
        assert found.confidence <= 0.4

    def test_an_infrastructure_failure_is_not_numeric_corruption(self):
        """A clean mitigation beside it must not turn infrastructure into math."""
        found = classify_matrix(
            matrix(
                repro(
                    "infrastructure_probe",
                    failure_hints=["infrastructure unavailable on this worker"],
                ),
                CLEAN_MITIGATION,
            )
        )

        assert found.category == "unknown"
        assert found.confidence <= 0.4

    @pytest.mark.parametrize(
        "ordinary_word",
        [
            "information",
            "inflight request",
            "inference server",
            "infrastructure check",
        ],
    )
    def test_other_inf_prefixes_are_not_numeric_evidence(self, ordinary_word):
        found = classify_matrix(
            matrix(repro(failure_hints=[ordinary_word]), CLEAN_MITIGATION)
        )

        assert found.category != "numeric_silent"

    def test_residual_alone_is_a_workload_name_not_numeric_evidence(self):
        found = classify_matrix(
            matrix(repro("Residual-Repro", failure_hints=[]), CLEAN_MITIGATION)
        )

        assert found.category == "unknown"


class TestAFailureThatSaysNothing:
    def test_a_bare_non_zero_exit_is_unknown(self):
        """The case the predicate was really matching."""
        found = classify_matrix(matrix(repro(failure_hints=[]), CLEAN_MITIGATION))

        assert found.category == "unknown"

    def test_it_is_not_confident(self):
        found = classify_matrix(matrix(repro(failure_hints=[]), CLEAN_MITIGATION))

        assert found.confidence <= 0.4, found.confidence

    def test_it_says_why_it_cannot_tell(self):
        found = classify_matrix(matrix(repro(failure_hints=[]), CLEAN_MITIGATION))

        assert "does not identify a cause" in found.rationale

    def test_the_failure_is_still_reported(self):
        """Unknown must not mean silent: the cells did fail."""
        found = classify_matrix(matrix(repro(failure_hints=[]), CLEAN_MITIGATION))

        assert "AORTA_MATRIX_REPRO" in found.signals


class TestAFailureThatIsNumeric:
    def test_a_nan_hint_still_reaches_numeric_silent(self):
        found = classify_matrix(
            matrix(repro(failure_hints=["loss became nan at step 50"]), CLEAN_MITIGATION)
        )

        assert found.category == "numeric_silent"
        assert found.confidence == pytest.approx(0.88)

    def test_a_non_finite_hint_does_too(self):
        found = classify_matrix(
            matrix(repro(failure_hints=["non-finite gradient"]), CLEAN_MITIGATION)
        )

        assert found.category == "numeric_silent"

    @pytest.mark.parametrize(
        "hint",
        [
            "activation=inf",
            "gradient=-inf",
            "loss reached infinity",
            "nonfinite output",
            "non finite output",
            "non-finite output",
        ],
    )
    def test_complete_numeric_tokens_still_count(self, hint):
        found = classify_matrix(
            matrix(repro(failure_hints=[hint]), CLEAN_MITIGATION)
        )

        assert found.category == "numeric_silent"

    def test_the_cell_name_counts_as_evidence(self):
        """A cell called Residual-NaN-repro has said what it is."""
        found = classify_matrix(
            matrix(repro("Residual-NaN-repro", failure_hints=[]), CLEAN_MITIGATION)
        )

        assert found.category == "numeric_silent"

    def test_a_structured_count_counts_too(self):
        found = classify_matrix(
            matrix(
                repro(failure_hints=[], exit_status_counts={"numeric_nan": 3}),
                CLEAN_MITIGATION,
            )
        )

        assert found.category == "numeric_silent"

    def test_without_a_clean_mitigation_it_is_less_certain(self):
        """The mitigation column is what raises confidence, and it is absent."""
        found = classify_matrix(matrix(repro(failure_hints=["nan loss"])))

        assert found.category == "numeric_silent"
        assert found.confidence == pytest.approx(0.72)


class TestTheUnchangedCases:
    def test_all_cells_passing_is_still_infra_only(self):
        found = classify_matrix(
            matrix({"name": "a", "failure_rate": 0.0, "failed_count": 0})
        )

        assert found.category == "unknown"
        assert "AORTA_MATRIX_INFRA_OK" in found.signals

    def test_an_empty_matrix_matches_nothing(self):
        found = classify_matrix({"cells": []})

        assert found.category == "unknown"


class TestTheDeadGuardIsGone:
    def test_the_no_op_is_not_in_the_source(self):
        from pathlib import Path

        import aorta.cia.autopsy.adapters.aorta_matrix as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")

        assert "repro cell name is enough" not in source

    def test_the_hint_check_is_actually_used(self):
        from pathlib import Path

        import aorta.cia.autopsy.adapters.aorta_matrix as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")

        assert "def _numeric_evidence(" in source
        assert "numeric = [c for c in repro_failures if _numeric_evidence(c)]" in source
