"""Unit tests for the `hrx_perf` HIP performance benchmark workload.

Unit-level: no hipcc, no GPU. Build and run are monkeypatched so per-step /
throughput parsing and result mapping are tested in isolation, plus the
setup-time guards (hipcc absent, bad LD_PRELOAD).
"""

from __future__ import annotations

import subprocess

import pytest

from aorta.workloads import hrx_perf as perf_mod
from aorta.workloads.hrx_perf import _BENCHES, HrxPerfWorkload


def test_vendored_bench_sources_present():
    for spec in _BENCHES.values():
        assert (perf_mod._KERNELS_DIR / spec.source).is_file()


def test_unknown_bench_rejected():
    wl = HrxPerfWorkload({"bench": "does_not_exist"})
    with pytest.raises(ValueError, match="unknown bench"):
        wl.setup()


def test_non_positive_size_rejected(monkeypatch):
    monkeypatch.setattr(perf_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxPerfWorkload, "_build", lambda self: self._build_dir / "x")
    wl = HrxPerfWorkload({"bench": "gemm", "size": 0})
    with pytest.raises(ValueError, match="must be > 0"):
        wl.setup()


def test_setup_raises_without_hipcc(monkeypatch):
    monkeypatch.setattr(perf_mod, "_resolve_hipcc", lambda _cfg: None)
    wl = HrxPerfWorkload({"bench": "gemm"})
    with pytest.raises(RuntimeError, match="hipcc not found"):
        wl.setup()


def test_setup_rejects_nonexistent_ld_preload(monkeypatch, tmp_path):
    monkeypatch.setattr(perf_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxPerfWorkload, "_build", lambda self: self._build_dir / "x")
    monkeypatch.setenv("LD_PRELOAD", "/path/to/hrx-root/lib/libamdhip64.so")
    wl = HrxPerfWorkload({"bench": "gemm", "build_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="LD_PRELOAD names object"):
        wl.setup()


def test_build_passes_sanitized_env_and_arch(monkeypatch, tmp_path):
    """The build must strip HRX routing vars and target the configured arch."""
    for var in ("LD_PRELOAD", "LD_LIBRARY_PATH", "HRX_GPU_DRIVER"):
        # Real path for LD_PRELOAD so the setup existence guard passes.
        if var == "LD_PRELOAD":
            lib = tmp_path / "libamdhip64.so"
            lib.write_bytes(b"\x7fELF")
            monkeypatch.setenv(var, str(lib))
        else:
            monkeypatch.setenv(var, str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        captured["env"] = k.get("env")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(perf_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    HrxPerfWorkload(
        {"bench": "triad", "gpu_arch": "gfx90a", "build_dir": str(tmp_path)}
    ).setup()

    for var in ("LD_PRELOAD", "LD_LIBRARY_PATH", "HRX_GPU_DRIVER"):
        assert var not in (captured["env"] or {})
    assert "--offload-arch=gfx90a" in captured["cmd"]
    assert "-O3" in captured["cmd"]


def _prep(monkeypatch, tmp_path, config=None):
    monkeypatch.setattr(perf_mod, "_resolve_hipcc", lambda _cfg: "/usr/bin/hipcc")
    monkeypatch.setattr(HrxPerfWorkload, "_build", lambda self: tmp_path / self._spec.binary)
    cfg = {"bench": "gemm", "build_dir": str(tmp_path), "iters": 3}
    if config:
        cfg.update(config)
    wl = HrxPerfWorkload(cfg)
    wl.setup()
    return wl


def _completed(stdout, returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=["bench"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_parses_step_times_and_throughput(monkeypatch, tmp_path):
    wl = _prep(monkeypatch, tmp_path)
    stdout = (
        "bench=gemm size=4096 iters=3 warmup=10\n"
        "step_ms=12.5\nstep_ms=12.0\nstep_ms=13.0\n"
        "GFLOPS=1234.5\nchecksum=4096.0 expected=4096\nRESULT=PERF_OK\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout, 0))

    res = wl.run()

    assert res.passed is True
    assert res.step_times_ms == [12.5, 12.0, 13.0]
    assert res.total_iterations == 3
    assert res.executed_iterations == 3
    assert res.configured_iterations == 3
    assert res.main_work_started is True
    assert res.metrics.get("gflops") == 1234.5
    assert res.metrics.get("mean_step_ms") == pytest.approx((12.5 + 12.0 + 13.0) / 3)
    assert res.metrics.get("result") == "PERF_OK"


def test_run_triad_throughput_metric_is_gbps(monkeypatch, tmp_path):
    wl = _prep(monkeypatch, tmp_path, {"bench": "triad", "size": 1000})
    stdout = (
        "bench=triad size=1000 iters=3 warmup=20\n"
        "step_ms=0.5\nstep_ms=0.6\nstep_ms=0.4\n"
        "GBPS=987.6\nchecksum=7.0 expected=7.0\nRESULT=PERF_OK\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout, 0))

    res = wl.run()

    assert res.passed is True
    assert res.metrics.get("gbps") == 987.6
    assert "gflops" not in res.metrics


def test_run_failed_result_is_failure(monkeypatch, tmp_path):
    wl = _prep(monkeypatch, tmp_path)
    stdout = "bench=gemm size=4096 iters=3 warmup=10\nchecksum=0.0 expected=4096\nRESULT=PERF_FAIL\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout, 1))

    res = wl.run()

    assert res.passed is False
    assert res.failure_count == 1
    assert res.failure_details, "expected a failure detail"
    assert res.metrics.get("result") == "PERF_FAIL"


def test_run_preload_ignored_fails_even_with_ok_result(monkeypatch, tmp_path):
    """A PERF_OK timing measured against the wrong runtime is not a valid A/B."""
    wl = _prep(monkeypatch, tmp_path)
    monkeypatch.setenv("LD_PRELOAD", "/some/hrx/libamdhip64.so")
    stdout = "step_ms=12.0\nGFLOPS=1000.0\nRESULT=PERF_OK\n"
    stderr = (
        "ERROR: ld.so: object '/some/hrx/libamdhip64.so' from LD_PRELOAD cannot "
        "be preloaded (cannot open shared object file): ignored.\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout, 0, stderr))

    res = wl.run()

    assert res.passed is False
    assert res.metrics.get("preload_ignored") is True
    assert "ignored" in res.failure_details[0].get("hint", "").lower()


def test_run_timeout_is_failure(monkeypatch, tmp_path):
    wl = _prep(monkeypatch, tmp_path)

    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="bench", timeout=wl._timeout, output="", stderr="")

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    res = wl.run()

    assert res.passed is False
    assert res.metrics.get("timed_out") is True
    assert res.total_iterations == 0
    assert res.main_work_started is False
    assert "timed out" in res.failure_details[0].get("hint", "")
