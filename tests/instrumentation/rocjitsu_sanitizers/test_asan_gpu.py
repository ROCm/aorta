"""Device ASan against the real toolchain, and on a real GPU when one is there.

The compile half needs only hipcc and proves the preflight on what the compiler
actually emits: the three builds below are the ones measured on gfx950, and two
of them link cleanly while carrying no device instrumentation. The run half also
needs an XNACK-capable GPU and the ASan-instrumented ROCm runtime, and runs the
two shipped recipes end to end; without the runtime it skips, which is the state
of the CI image today (it ships no lib/asan).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from aorta.instrumentation.rocjitsu_sanitizers import (
    Verdict,
    execute_sanitizer_run,
    inspect_device_instrumentation,
    read_report,
    resolve_asan_runtime,
)
from aorta.instrumentation.rocm_paths import resolve_rocm_roots

REPO = Path(__file__).resolve().parents[3]
SANITIZERS = REPO / "recipes" / "sanitizers"
REPRO = SANITIZERS / "fixtures" / "repro" / "asan_heap_overflow.hip"
ARCH = os.environ.get("CIA_GPU_ARCH", "gfx950")
ASAN_FLAGS = ("-fsanitize=address", "-shared-libsan", "-g", "-O1")


def _hipcc() -> str:
    found = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    if not Path(found).is_file():
        pytest.skip("hipcc is not on this runner")
    return found


def _build(output: Path, *flags: str) -> Path:
    env = dict(os.environ)
    # hipcc shells out to clang-offload-bundler, which is on PATH in neither
    # install layout.
    env["PATH"] = f"{resolve_rocm_roots().llvm_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    built = subprocess.run(
        [_hipcc(), *flags, str(REPRO), "-o", str(output)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    return output


@pytest.mark.rocm
@pytest.mark.parametrize(
    "flags,instrumented",
    [
        ((f"--offload-arch={ARCH}:xnack+", *ASAN_FLAGS), True),
        # Links cleanly; the compiler only warns that it ignored the sanitizer.
        ((f"--offload-arch={ARCH}", *ASAN_FLAGS), False),
        # xnack+ target and host ASan, device half switched off.
        ((f"--offload-arch={ARCH}:xnack+", *ASAN_FLAGS, "-fno-gpu-sanitize"), False),
    ],
    ids=["xnack-plus", "no-xnack-suffix", "no-gpu-sanitize"],
)
def test_preflight_matches_what_the_compiler_emitted(tmp_path, flags, instrumented):
    binary = _build(tmp_path / "repro", *flags)
    assert inspect_device_instrumentation(binary).instrumented_for(ARCH) is instrumented


def _staged_recipe(tmp_path: Path, name: str, binary: str, *defines: str) -> Path:
    """Copy a shipped recipe beside a freshly built fixture, as the nightly would."""

    bin_dir = tmp_path / "fixtures" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _build(bin_dir / binary, f"--offload-arch={ARCH}:xnack+", *ASAN_FLAGS, *defines)
    # The shipped recipe names gfx950; it has to name what the fixture was built for.
    data = yaml.safe_load((SANITIZERS / name).read_text())
    data["sanitizer_plan"]["target"] = ARCH
    recipe = tmp_path / name
    recipe.write_text(yaml.safe_dump(data, sort_keys=False))
    return recipe


requires_runtime = pytest.mark.skipif(
    resolve_asan_runtime() is None or not Path("/dev/kfd").exists(),
    reason=(
        "needs an XNACK-capable GPU and the ASan-instrumented ROCm runtime "
        "(AORTA_ASAN_ROCM_LIB or ROCM_ASAN_PATH)"
    ),
)


@pytest.mark.gpu
@pytest.mark.rocm
@requires_runtime
def test_the_overflow_recipe_fails_on_the_planted_store(tmp_path):
    recipe = _staged_recipe(
        tmp_path, "asan-heap-overflow.yaml", "asan_heap_overflow", "-DASAN_REPRO_OVERFLOW=1"
    )
    report = read_report(execute_sanitizer_run(recipe, output_dir=tmp_path / "out"))
    assert report.overall_verdict is Verdict.FAIL
    (check,) = report.checks
    (finding,) = check.findings
    assert finding.code == "heap-buffer-overflow"
    assert finding.kernel_name == "store_index"
    assert dict(finding.metadata)["side"] == "device"


@pytest.mark.gpu
@pytest.mark.rocm
@requires_runtime
def test_the_clean_recipe_passes(tmp_path):
    recipe = _staged_recipe(tmp_path, "asan-clean.yaml", "asan_clean")
    report = read_report(execute_sanitizer_run(recipe, output_dir=tmp_path / "out"))
    assert report.overall_verdict is Verdict.PASS, report.checks[0].reason
