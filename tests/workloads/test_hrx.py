"""Unit tests for the `hrx` HIP-launch probe workload.

Unit-level: no hipcc, no GPU. Build and run are monkeypatched so the verdict
parsing and result mapping are tested in isolation, plus the setup-time guard
when hipcc is absent.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from aorta.workloads import hrx as hrx_mod
from aorta.workloads.hrx import _PROBES, HrxWorkload


def test_vendored_sources_present():
    """Every registered probe's source (and code object source) is vendored."""
    for spec in _PROBES.values():
        assert (hrx_mod._KERNELS_DIR / spec.source).is_file()
        if spec.kernel_source:
            assert (hrx_mod._KERNELS_DIR / spec.kernel_source).is_file()


def test_unknown_probe_rejected():
    wl = HrxWorkload({"probe": "does_not_exist"})
    with pytest.raises(ValueError, match="unknown probe"):
        wl.setup()


def test_unknown_config_key_warns(caplog):
    wl = HrxWorkload({"probe": "module", "bogus": 1})
    with caplog.at_level(logging.WARNING, logger="aorta.workloads.hrx"):
        probe = wl._validated_probe()
    assert probe == "module"
    assert any("bogus" in r.message for r in caplog.records)


def test_setup_raises_without_hipcc(monkeypatch):
    """No hipcc -> clean setup failure (classified did_not_run, not a repro)."""
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: None)
    wl = HrxWorkload({"probe": "module"})
    with pytest.raises(RuntimeError, match="hipcc not found"):
        wl.setup()


def test_setup_rejects_non_bool_keep_build(monkeypatch):
    """A "false" string must not coerce to True (loose-typing guard)."""
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxWorkload, "_build", lambda self: self._build_dir / "x")
    wl = HrxWorkload({"probe": "module", "keep_build": "false"})
    with pytest.raises(ValueError, match="keep_build must be a bool"):
        wl.setup()


def test_build_env_strips_runtime_routing_vars(monkeypatch, tmp_path):
    """hipcc must build against the stock toolchain, not the HRX-on cell env.

    The dispatcher merges the cell env into os.environ before setup() runs, so
    a preloaded HRX libamdhip64.so would otherwise run inside the compiler.
    """
    # Use real paths so the setup-time LD_PRELOAD existence check passes; this
    # test is about build-env stripping, not routing validation.
    real_lib = tmp_path / "libamdhip64.so"
    real_lib.write_bytes(b"\x7fELF")
    monkeypatch.setenv("LD_PRELOAD", str(real_lib))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    monkeypatch.setenv("HRX_GPU_DRIVER", "amdgpu")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")  # unrelated var must survive

    captured: dict[str, dict[str, str] | None] = {}

    def _fake_run(cmd, *a, **k):
        captured["env"] = k.get("env")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    HrxWorkload({"probe": "static", "build_dir": str(tmp_path)}).setup()

    build_env = captured.get("env")
    assert build_env is not None, "build must pass an explicit sanitized env"
    for var in hrx_mod._RUNTIME_ROUTING_VARS:
        assert var not in build_env, f"{var} must be stripped from the build env"
    assert build_env.get("ROCM_PATH") == "/opt/rocm"


def test_relative_build_dir_is_resolved_absolute(monkeypatch, tmp_path):
    """A relative build_dir must be absolutized so hipcc -o / probe cwd agree.

    _run_hipcc runs with cwd=_KERNELS_DIR and run() with cwd=build_dir, so a
    relative build_dir would split the -o output from the probe exec.
    """
    monkeypatch.chdir(tmp_path)
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    wl = HrxWorkload({"probe": "static", "build_dir": "relbuild"})
    wl.setup()

    assert wl._build_dir.is_absolute()
    assert wl._binary.is_absolute()
    assert wl._build_dir == (tmp_path / "relbuild").resolve()
    # The hipcc -o target must be the absolute binary path, not a relative one.
    out_arg = captured["cmd"][captured["cmd"].index("-o") + 1]
    assert Path(out_arg).is_absolute()


def test_setup_rejects_nonexistent_ld_preload(monkeypatch, tmp_path):
    """A placeholder/typo'd LD_PRELOAD must fail setup, not silently fall back.

    ld.so only warns and runs against the default HIP runtime, so an hrx_on cell
    would otherwise report a misleading FULLY_WORKS.
    """
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxWorkload, "_build", lambda self: self._build_dir / "x")
    monkeypatch.setenv("LD_PRELOAD", "/path/to/hrx-root/lib/libamdhip64.so")
    wl = HrxWorkload({"probe": "module", "build_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="LD_PRELOAD names object"):
        wl.setup()


def test_setup_accepts_existing_ld_preload(monkeypatch, tmp_path):
    """An LD_PRELOAD that points at a real file must pass the existence check."""
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxWorkload, "_build", lambda self: self._build_dir / "x")
    real_lib = tmp_path / "libamdhip64.so"
    real_lib.write_bytes(b"\x7fELF")
    monkeypatch.setenv("LD_PRELOAD", str(real_lib))
    HrxWorkload({"probe": "module", "build_dir": str(tmp_path)}).setup()


def test_setup_skips_bare_soname_ld_preload(monkeypatch, tmp_path):
    """A bare soname (no '/') is resolved via LD_LIBRARY_PATH -- don't false-fail."""
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxWorkload, "_build", lambda self: self._build_dir / "x")
    monkeypatch.setenv("LD_PRELOAD", "libamdhip64.so")
    HrxWorkload({"probe": "module", "build_dir": str(tmp_path)}).setup()


def _prep_workload(monkeypatch, tmp_path: Path) -> HrxWorkload:
    """A workload whose setup() succeeds without a real toolchain."""
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxWorkload, "_build", lambda self: tmp_path / self._spec.binary)
    wl = HrxWorkload({"probe": "module", "build_dir": str(tmp_path)})
    wl.setup()
    return wl


def _fake_completed(
    stdout: str, returncode: int, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["probe"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_fully_works(monkeypatch, tmp_path):
    wl = _prep_workload(monkeypatch, tmp_path)
    stdout = "out[0]=107 (expect 107)\nVERDICT=FULLY_WORKS\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(stdout, 0))

    res = wl.run()

    assert res.passed is True
    assert res.failure_count == 0
    assert res.failure_details == []
    assert res.metrics.get("verdict") == "FULLY_WORKS"
    assert res.metrics.get("out0") == 107.0
    assert res.main_work_started is True
    assert res.executed_iterations == 1


def test_run_preload_ignored_is_failure(monkeypatch, tmp_path):
    """FULLY_WORKS + an ld.so 'ignored' warning is a false green -> fail it.

    Catches the exists-but-unloadable case (wrong arch / missing deps) that the
    setup existence check can't see.
    """
    wl = _prep_workload(monkeypatch, tmp_path)
    monkeypatch.setenv("LD_PRELOAD", "/some/hrx/libamdhip64.so")
    stdout = "out[0]=107 (expect 107)\nVERDICT=FULLY_WORKS\n"
    stderr = (
        "ERROR: ld.so: object '/some/hrx/libamdhip64.so' from LD_PRELOAD cannot "
        "be preloaded (cannot open shared object file): ignored.\n"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed(stdout, 0, stderr)
    )

    res = wl.run()

    assert res.passed is False
    assert res.metrics.get("preload_ignored") is True
    assert res.metrics.get("verdict") == "FULLY_WORKS"
    assert res.failure_details, "expected a failure detail"
    assert "ignored" in res.failure_details[0].get("hint", "").lower()


def test_run_no_preload_no_false_positive(monkeypatch, tmp_path):
    """With LD_PRELOAD unset, a passing verdict must stay green (no backstop)."""
    wl = _prep_workload(monkeypatch, tmp_path)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    stdout = "out[0]=107 (expect 107)\nVERDICT=FULLY_WORKS\n"
    # Stray stderr text must not be misread as an ld.so preload warning.
    stderr = "some unrelated warning: cannot be preloaded elsewhere\n"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed(stdout, 0, stderr)
    )

    res = wl.run()

    assert res.passed is True
    assert res.metrics.get("preload_ignored") is False


def test_run_output_not_written(monkeypatch, tmp_path):
    wl = _prep_workload(monkeypatch, tmp_path)
    stdout = "out[0]=0 (expect 107)\nVERDICT=OUTPUT_NOT_WRITTEN (module-launch out-arg broken)\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(stdout, 1))

    res = wl.run()

    assert res.passed is False
    assert res.failure_count == 1
    assert res.total_iterations == 1
    assert res.first_failure_iteration == 0
    assert res.metrics.get("verdict") == "OUTPUT_NOT_WRITTEN"
    assert res.metrics.get("out0") == 0.0
    assert res.failure_details, "expected a failure detail"
    detail = res.failure_details[0]
    assert detail.get("verdict") == "OUTPUT_NOT_WRITTEN"
    assert "OUTPUT_NOT_WRITTEN" in detail.get("hint", "")
    # It still dispatched and read back, so main work started.
    assert res.main_work_started is True


def test_run_timeout_is_failure(monkeypatch, tmp_path):
    wl = _prep_workload(monkeypatch, tmp_path)

    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=wl._timeout, output="", stderr="")

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    res = wl.run()

    assert res.passed is False
    assert res.metrics.get("timed_out") is True
    # Hung before printing anything -> did-not-run shape, no phantom iteration.
    assert res.total_iterations == 0
    assert res.failure_details, "expected a failure detail"
    assert "hung" in res.failure_details[0].get("hint", "")


def test_run_no_verdict_is_failure(monkeypatch, tmp_path):
    """A HIP init crash before any verdict must not pass."""
    wl = _prep_workload(monkeypatch, tmp_path)
    stdout = "HIP_ERR hipMalloc at 27: out of memory\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(stdout, 2))

    res = wl.run()

    assert res.passed is False
    assert res.metrics.get("verdict") is None
    assert res.main_work_started is False
    assert res.executed_iterations == 0
    # Never started -> 0 iterations + no failing index, so the matrix
    # elapsed_per_iter fallback can't mint a misleading step time.
    assert res.total_iterations == 0
    assert res.first_failure_iteration is None
    # The crash diagnostic is on stdout (probes printf there), so it must be
    # persisted in the failure detail, and the hint must point at stdout.
    assert res.failure_details, "expected a failure detail"
    detail = res.failure_details[0]
    assert "hipMalloc" in detail.get("stdout_tail", "")
    assert "stdout" in detail.get("hint", "")
