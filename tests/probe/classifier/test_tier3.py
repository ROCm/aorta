"""Tests for Tier 3 dmesg / amd-smi detectors (FR 2.3, 2.11)."""

from __future__ import annotations

import logging
import os

import pytest

from aorta.probe.classifier import tier3_kernel
from aorta.probe.classifier.tier3_kernel import (
    AmdSmiSnapshot,
    Tier3State,
    poll_amd_smi,
    scan_amd_smi,
    scan_dmesg,
    scan_dmesg_text,
)

# ---- Pure text-scan path -------------------------------------------------


@pytest.mark.parametrize(
    "text,detector",
    [
        ("amdgpu: GPU reset failed\nfoo", tier3_kernel.DETECTOR_AMDGPU_RESET),
        ("SDMA semaphore timeout", tier3_kernel.DETECTOR_SDMA_TIMEOUT),
        ("SDMA hang detected on engine 0", tier3_kernel.DETECTOR_SDMA_TIMEOUT),
        ("VM_L2_PROTECTION_FAULT_STATUS: ...", tier3_kernel.DETECTOR_VM_L2_FAULT),
        ("XGMI link down detected", tier3_kernel.DETECTOR_XGMI_LINK_ERROR),
        ("AER: Fatal error received", tier3_kernel.DETECTOR_PCIE_AER_FATAL),
    ],
)
def test_scan_dmesg_text_patterns(text, detector):
    fired = scan_dmesg_text(text)
    assert detector in fired


def test_scan_dmesg_text_no_match():
    assert scan_dmesg_text("nothing relevant here\nall green") == []


def test_scan_dmesg_text_empty():
    assert scan_dmesg_text("") == []


def test_xgmi_healthy_does_not_fire():
    """A healthy XGMI line ('XGMI initialized') stays silent."""
    assert scan_dmesg_text("XGMI initialized successfully") == []


# ---- Fail-soft missing-binary path (FR 2.11) -----------------------------


def test_dmesg_missing_logs_once(monkeypatch, caplog):
    """``dmesg`` missing -> single ``tier3 disabled:`` warning per invocation."""
    monkeypatch.setenv("PATH", "/nonexistent-dir-that-does-not-exist")
    state = Tier3State()
    with caplog.at_level(logging.WARNING):
        scan_dmesg(state)
        scan_dmesg(state)
        scan_dmesg(state)
    disabled_warnings = [
        record
        for record in caplog.records
        if "tier3 disabled" in record.getMessage() and "dmesg" in record.getMessage()
    ]
    assert len(disabled_warnings) == 1


def test_amdsmi_missing_logs_once(monkeypatch, caplog):
    """Same one-warning rule for ``amd-smi`` (FR 2.11 + R3)."""
    monkeypatch.setenv("PATH", "/nonexistent-dir-that-does-not-exist")
    monkeypatch.delenv("AORTA_PROBE_AMDSMI_FAKE", raising=False)
    state = Tier3State()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            poll_amd_smi(state)
    disabled_warnings = [
        record
        for record in caplog.records
        if "tier3 disabled" in record.getMessage() and "amd-smi" in record.getMessage()
    ]
    assert len(disabled_warnings) == 1


# ---- amd-smi diff logic via fake-shim env var ----------------------------


def test_amdsmi_fake_env_var_vram_growth(monkeypatch):
    """Fake-shim diff above VRAM threshold fires ``tier3:vram_growth``."""
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=100,throttle=0")
    state = Tier3State()
    pre = poll_amd_smi(state)
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=500,throttle=0")
    post = poll_amd_smi(state)
    fired = scan_amd_smi(state, pre, post)
    assert tier3_kernel.DETECTOR_VRAM_GROWTH in fired


def test_amdsmi_fake_env_var_thermal_throttle(monkeypatch):
    """Throttle counter incremented -> ``tier3:thermal_throttle`` fires."""
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=100,throttle=0")
    state = Tier3State()
    pre = poll_amd_smi(state)
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=100,throttle=5")
    post = poll_amd_smi(state)
    fired = scan_amd_smi(state, pre, post)
    assert tier3_kernel.DETECTOR_THERMAL_THROTTLE in fired


def test_amdsmi_below_threshold_does_not_fire(monkeypatch):
    """VRAM delta under the threshold stays silent (noise floor)."""
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=100,throttle=0")
    state = Tier3State()
    pre = poll_amd_smi(state)
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=110,throttle=0")
    post = poll_amd_smi(state)
    fired = scan_amd_smi(state, pre, post)
    assert tier3_kernel.DETECTOR_VRAM_GROWTH not in fired
    assert tier3_kernel.DETECTOR_THERMAL_THROTTLE not in fired


def test_amdsmi_missing_snapshot_returns_no_detectors():
    """None snapshot -> fail-soft (no fired detectors)."""
    state = Tier3State()
    pre = AmdSmiSnapshot(vram_used_mib=100, thermal_throttle_count=0)
    assert scan_amd_smi(state, pre, None) == []
    assert scan_amd_smi(state, None, pre) == []


# ---- dmesg shim via PATH (FR 2.3 happy path) -----------------------------


def test_scan_dmesg_via_shim(monkeypatch, tmp_path):
    """A fake ``dmesg`` script on PATH emits canned content the scanner sees."""
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "dmesg"
    shim.write_text(
        "#!/bin/sh\necho 'amdgpu: GPU reset triggered'\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    state = Tier3State()
    fired = scan_dmesg(state)
    assert tier3_kernel.DETECTOR_AMDGPU_RESET in fired
