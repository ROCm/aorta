"""Real-hardware (ROCm/gfx) RocJITsu sanitizer guardrail tests.

Selected by the Phase-2 GPU gate (``pytest -m "gpu or rocm"``) and run on the
self-hosted MI350 runner; the CPU gate deselects the whole module via the
``rocm`` marker (see ``tests/test_marker_partition.py`` and
``docs/ci-testing-plan.md``). These complement the pure-logic unit tests in this
directory, which cannot observe a real GPU.

Two layers, so the module contributes value on *any* ROCm runner while still
exercising the full backend when it is available:

1. **Fail-closed on real hardware** (always runs on a ROCm GPU): with no
   RocJITsu backend provisioned, the sanitizer pipeline must return
   ``not_checked`` -- never a false ``pass``. This is the safety-critical
   guarantee and needs no private build, so it runs in the ordinary PR GPU gate.

2. **Real ConSan repro cases** (opt-in): build and run the committed clean/racy
   LDS repros -- the same "our cases" the ``sanitizers-nightly.yml`` workflow
   exercises -- and assert ``clean => pass`` / ``racy => fail`` on real
   hardware. Self-skips unless ``ROCJITSU_BUILD`` (the DBI hook) and ``hipcc``
   are both present, so the standard PR container stays green.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import (
    ExecutionState,
    ExecutionSummary,
    SelectionRequirement,
    Verdict,
    run_consan,
    run_sanitizers,
    select_kernels,
)
from aorta.instrumentation.rocjitsu_sanitizers.backends import support
from aorta.instrumentation.rocjitsu_sanitizers.consan import resolve_consan_hook
from aorta.instrumentation.rocjitsu_sanitizers.models import KernelWorklist
from aorta.instrumentation.rocjitsu_sanitizers.selection import observations_from_consan_repro

pytestmark = pytest.mark.rocm

_REPO = Path(__file__).resolve().parents[3]
_REPRO_DIR = _REPO / "recipes" / "sanitizers" / "fixtures" / "repro"
_REPRO_SOURCE = {
    "clean": "consan_lds_race.hip",
    "racy": "consan_lds_race_2wave.hip",
}


def _detected_target() -> str | None:
    """The real gfx arch of the attached AMD GPU, or ``None`` off ROCm.

    Import and detection are both guarded so this module imports (and the CPU
    marker-partition collection succeeds) on a machine with no GPU at all.
    """
    try:
        from aorta.utils.gpu_control import GPUVendor, detect_gpu
    except Exception:
        return None
    try:
        vendor, arch = detect_gpu()
    except Exception:
        return None
    if vendor is not GPUVendor.AMD or not arch or not arch.startswith("gfx"):
        return None
    return arch


_TARGET = _detected_target()

skip_no_rocm = pytest.mark.skipif(
    _TARGET is None,
    reason="no AMD ROCm GPU detected (rocminfo reported no gfx target)",
)

# The real-repro layer needs the private DBI hook (via ROCJITSU_BUILD) *and* a
# compiler to build the fixture. Missing either => skip, never fail: the PR gate
# container has neither, so it must stay green.
_HAVE_HIPCC = shutil.which("hipcc") is not None
_HAVE_HOOK = resolve_consan_hook() is not None
skip_no_backend = pytest.mark.skipif(
    not (_HAVE_HIPCC and _HAVE_HOOK),
    reason="set ROCJITSU_BUILD (ConSan DBI hook) and install hipcc to run real repros",
)


def _worklist(variant: str, target: str) -> KernelWorklist:
    """Build the single-kernel worklist the daily ConSan recipes select."""
    return select_kernels(
        observations_from_consan_repro(variant, target=target),
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=1,
    )


@skip_no_rocm
def test_consan_fails_closed_without_backend(tmp_path: Path) -> None:
    """A supported GPU with no backend provisioned must NOT report a false pass.

    On the real detected target ConSan is a supported, runnable backend, yet
    with no repro command wired the pipeline must fail closed to
    ``not_checked`` -- the whole point of ``policy.on_missing_backend: fail``.
    """
    target = _TARGET
    assert target is not None  # guarded by skip_no_rocm

    # Sanity: the detected hardware really is a target where ConSan *could* run,
    # so a not_checked verdict below reflects a missing backend, not an
    # unsupported chip.
    assert support("consan", target)["runnable"] is True

    report = run_sanitizers(
        _worklist("racy", target),
        target=target,
        sanitizers=["consan"],
        output_dir=tmp_path,
        consan_command=None,
        on_missing_backend="fail",
    )

    assert report.overall_verdict is Verdict.NOT_CHECKED
    assert report.overall_verdict is not Verdict.PASS
    assert report.execution_status is ExecutionSummary.NOT_CHECKED
    (check,) = report.checks
    assert check.sanitizer == "consan"
    assert check.state is ExecutionState.NOT_CHECKED


@skip_no_rocm
@skip_no_backend
@pytest.mark.slow
@pytest.mark.parametrize(
    "variant,expected",
    [("clean", Verdict.PASS), ("racy", Verdict.FAIL)],
)
def test_consan_repro_case(variant: str, expected: Verdict, tmp_path: Path) -> None:
    """The committed ConSan repros reproduce their expected verdict on hardware.

    Non-strict so the verdict is driven by parsed race findings (deterministic:
    racy => findings => fail, clean => no findings => pass) rather than the
    strict-mode exit-92 load rejection the nightly baseline comparator handles.
    """
    target = _TARGET
    assert target is not None  # guarded by skip_no_rocm

    source = _REPRO_DIR / _REPRO_SOURCE[variant]
    binary = tmp_path / f"consan_{variant}"
    subprocess.run(
        ["hipcc", f"--offload-arch={target}", "-O1", "-g", str(source), "-o", str(binary)],
        check=True,
        timeout=600,
    )

    result = run_consan(
        _worklist(variant, target),
        command=binary,
        output_dir=tmp_path / "out",
        strict=False,
    )

    assert result.consan.verdict is expected
    if variant == "racy":
        # The guardrail's cardinal rule: a real data race is never a pass.
        assert result.consan.verdict is not Verdict.PASS
        assert result.consan.findings, "racy repro produced no ConSan findings"
