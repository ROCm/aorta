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
    for var in hrx_mod._RUNTIME_ROUTING_VARS:
        monkeypatch.setenv(var, "/some/hrx/value")
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


def _prep_workload(monkeypatch, tmp_path: Path) -> HrxWorkload:
    """A workload whose setup() succeeds without a real toolchain."""
    monkeypatch.setattr(hrx_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxWorkload, "_build", lambda self: tmp_path / self._spec.binary)
    wl = HrxWorkload({"probe": "module", "build_dir": str(tmp_path)})
    wl.setup()
    return wl


def _fake_completed(stdout: str, returncode: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["probe"], returncode=returncode, stdout=stdout, stderr="")


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


def test_run_output_not_written(monkeypatch, tmp_path):
    wl = _prep_workload(monkeypatch, tmp_path)
    stdout = "out[0]=0 (expect 107)\nVERDICT=OUTPUT_NOT_WRITTEN (module-launch out-arg broken)\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(stdout, 1))

    res = wl.run()

    assert res.passed is False
    assert res.failure_count == 1
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
