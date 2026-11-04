"""CPU tests for ``scripts/sanitizers/fpsan_probe.py``: its verdicts and exit codes.

The nightly FpSan job maps each exit code to its own ``::error::``, so a code
that means something else (a crash read as "FpSan unavailable") misleads the
operator. The children are stubbed; no Triton or GPU is needed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load():
    path = _REPO / "scripts" / "sanitizers" / "fpsan_probe.py"
    spec = importlib.util.spec_from_file_location("fpsan_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe_mod = _load()


def _stub_children(monkeypatch, *, digests=None, failed=(), instrumented=True, uninstrumented=(),
                   timeout=False):
    """Fake child runs: each writes the digest ``digests`` names for its run key."""
    digests = {
        "mix/a/plain": "p", "mix/a/fpsan": "f",
        "rowsum/a/fpsan": "r", "rowsum/b/fpsan": "r",
        "dot/a/fpsan": "d", "dot/b/fpsan": "d",
        **(digests or {}),
    }

    def run(argv, *, env, **_kwargs):
        kernel, variant = argv[argv.index("--kernel") + 1], argv[argv.index("--variant") + 1]
        key = f"{kernel}/{variant}/{env['TRITON_INSTRUMENTATION_MODE'] or 'plain'}"
        if timeout:
            raise subprocess.TimeoutExpired(argv, 900)
        if key in failed:
            return types.SimpleNamespace(returncode=1, stderr="boom\n")
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(json.dumps({"sha256": digests[key], "first_launch_seconds": 0.1,
                                   "device": "gfx950"}), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")

    skipped = {key.replace("/", "-") for key in uninstrumented}

    def check(cache: Path) -> dict:
        return {"instrumented": instrumented and cache.name.endswith("-fpsan")
                and cache.name not in skipped}

    monkeypatch.setattr(probe_mod.subprocess, "run", run)
    monkeypatch.setattr(probe_mod, "_instrumentation", lambda: check)
    monkeypatch.setattr(probe_mod, "triton", types.SimpleNamespace(__version__="3.8.0"))
    monkeypatch.setattr(probe_mod, "importlib", types.SimpleNamespace(
        util=types.SimpleNamespace(find_spec=lambda name: object())
    ))


@pytest.mark.parametrize(
    ("stub", "code", "verdict"),
    [
        ({}, 0, "fpsan_available"),
        ({"instrumented": False}, 1, "fpsan_unavailable"),
        # Instrumented, but the sanitized output is the plain one: nothing was rewritten.
        ({"digests": {"mix/a/fpsan": "p"}}, 1, "fpsan_unavailable"),
        # A tiling pair that ran uninstrumented matches on plain floats and proves nothing.
        ({"uninstrumented": ("rowsum/b/fpsan",)}, 1, "fpsan_unavailable"),
        ({"uninstrumented": ("dot/a/fpsan", "dot/b/fpsan")}, 1, "fpsan_unavailable"),
        ({"failed": ("mix/a/plain",)}, 2, "no_gpu"),
        ({"digests": {"dot/b/fpsan": "other"}}, 3, "fpsan_tiling_variant"),
        ({"failed": ("rowsum/b/fpsan",)}, 3, "fpsan_tiling_variant"),
    ],
)
def test_each_outcome_has_its_own_exit_code(tmp_path, monkeypatch, capsys, stub, code, verdict):
    _stub_children(monkeypatch, **stub)
    out = tmp_path / "probe.json"
    assert probe_mod.main(["--out", str(out), "--work", str(tmp_path / "work")]) == code
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary.get("verdict", "").startswith(verdict)
    assert json.loads(capsys.readouterr().out) == summary


def test_the_summary_names_each_fpsan_run_that_was_not_instrumented(tmp_path, monkeypatch, capsys):
    _stub_children(monkeypatch, uninstrumented=("dot/b/fpsan",))
    out = tmp_path / "probe.json"
    assert probe_mod.main(["--out", str(out), "--work", str(tmp_path / "work")]) == 1
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary.get("uninstrumented_runs") == ["dot/b/fpsan"]
    assert "tiling_invariant" not in summary


def test_a_probe_that_crashes_is_not_reported_as_fpsan_unavailable(tmp_path, monkeypatch):
    _stub_children(monkeypatch, timeout=True)
    out = tmp_path / "probe.json"
    assert probe_mod.main(["--out", str(out), "--work", str(tmp_path / "work")]) == 4
    assert json.loads(out.read_text(encoding="utf-8")).get("verdict", "").startswith(
        "probe_error: TimeoutExpired"
    )


def test_no_triton_checks_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_mod, "triton", None)
    monkeypatch.setattr(probe_mod, "_TRITON_IMPORT_ERROR", "OSError: libtriton.so: cannot open")
    out = tmp_path / "probe.json"
    assert probe_mod.main(["--out", str(out)]) == 2
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary.get("verdict", "").startswith("no_triton")
    assert summary.get("triton_import_error") == "OSError: libtriton.so: cannot open"
