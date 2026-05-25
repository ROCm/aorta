"""Tests for Tier 1 process-level detectors (FR 2.1).

Each detector is exercised against a synthetic command (no GPU, no
kernel state) so the suite runs under ``pytest -m "not gpu and not
rocm"`` on any Linux host.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.probe.classifier import tier1_process
from aorta.probe.classifier.tier1_process import Tier1Context


def _ctx(tmp_path: Path, exit_code: int, *, timed_out: bool = False) -> Tier1Context:
    return Tier1Context(exit_code=exit_code, timed_out=timed_out, trial_dir=tmp_path)


def test_pass_no_detectors(tmp_path):
    """Exit 0 + no timeout + no coredump fires nothing."""
    assert tier1_process.detect(_ctx(tmp_path, 0)) == []


def test_exit_nonzero_fires(tmp_path):
    """Non-zero exit fires ``tier1:exit_nonzero`` (and only it)."""
    assert tier1_process.detect(_ctx(tmp_path, 1)) == [tier1_process.DETECTOR_EXIT_NONZERO]


def test_timeout_fires_alone(tmp_path):
    """A timed-out trial fires ``tier1:timeout``, suppressing ``exit_nonzero``."""
    detectors = tier1_process.detect(_ctx(tmp_path, -1, timed_out=True))
    assert tier1_process.DETECTOR_TIMEOUT in detectors
    assert tier1_process.DETECTOR_EXIT_NONZERO not in detectors


@pytest.mark.parametrize(
    "sig,detector",
    [
        (signal.SIGSEGV, tier1_process.DETECTOR_SIGSEGV),
        (signal.SIGABRT, tier1_process.DETECTOR_SIGABRT),
        (signal.SIGBUS, tier1_process.DETECTOR_SIGBUS),
    ],
)
def test_signal_detectors(tmp_path, sig, detector):
    """``returncode == -SIG`` fires the corresponding ``tier1:*`` detector."""
    detectors = tier1_process.detect(_ctx(tmp_path, -int(sig)))
    assert detector in detectors
    # Signal classification suppresses the ``exit_nonzero`` detector
    # (per rubric FR 2.1: most-specific cause wins).
    assert tier1_process.DETECTOR_EXIT_NONZERO not in detectors


def test_coredump_fires(tmp_path):
    """Any ``core.*`` file in the trial dir flags the trial."""
    (tmp_path / "core.12345").write_bytes(b"")
    detectors = tier1_process.detect(_ctx(tmp_path, 1))
    assert tier1_process.DETECTOR_COREDUMP in detectors


def test_bare_core_filename_also_fires(tmp_path):
    """Distros with ``core_pattern = core`` produce a bare ``core``."""
    (tmp_path / "core").write_bytes(b"")
    detectors = tier1_process.detect(_ctx(tmp_path, 1))
    assert tier1_process.DETECTOR_COREDUMP in detectors


def test_no_coredump_when_missing(tmp_path):
    """Missing trial dir falls through cleanly (no FS-error explosion)."""
    detectors = tier1_process.detect(
        Tier1Context(exit_code=0, timed_out=False, trial_dir=tmp_path / "does-not-exist")
    )
    assert detectors == []


# ---- Synthetic-subprocess integration: real signals & exit codes --------


def test_synthetic_exit_nonzero(tmp_path):
    """``exit 1`` from bash flows through Tier 1 cleanly."""
    proc = subprocess.run(  # noqa: S607 -- audited test invocation
        ["bash", "-c", "exit 1"], check=False
    )
    detectors = tier1_process.detect(_ctx(tmp_path, proc.returncode))
    assert detectors == [tier1_process.DETECTOR_EXIT_NONZERO]


@pytest.mark.skipif(sys.platform != "linux", reason="signal semantics linux-only")
def test_synthetic_sigsegv(tmp_path):
    """A bash sub-shell self-kill produces ``returncode == -SIGSEGV``."""
    proc = subprocess.run(  # noqa: S607 -- audited test invocation
        ["bash", "-c", "kill -SEGV $$"], check=False
    )
    assert proc.returncode == -int(signal.SIGSEGV)
    detectors = tier1_process.detect(_ctx(tmp_path, proc.returncode))
    assert tier1_process.DETECTOR_SIGSEGV in detectors
