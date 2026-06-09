"""Tests for Tier 3 dmesg / amd-smi detectors (FR 2.3, 2.11)."""

from __future__ import annotations

import logging
import os

import pytest

from aorta.probe.classifier import tier3_kernel
from aorta.probe.classifier.tier3_kernel import (
    GPU_IDLE_UTILIZATION_THRESHOLD_PCT,
    AmdSmiSnapshot,
    Tier3State,
    _parse_amd_smi_monitor_csv,
    gpu_idle_probe_from_state,
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


def test_scan_dmesg_text_truncates_to_tail_not_head(monkeypatch):
    """Regression for PR #197 review (Sonbol): the dmesg truncation
    must keep the tail, not the head.

    XGMI / HBM / MMU kernel signatures are almost always emitted
    in the seconds before a failing trial ends -- i.e. at the
    end of the dmesg ring. The previous head-slice (``text[:MAX]``)
    discarded exactly those lines on long-running trials.

    Verified by planting an XGMI error at the *tail* of an oversized
    blob and confirming the detector fires.
    """
    monkeypatch.setattr(tier3_kernel, "MAX_DMESG_BYTES", 100)
    head_noise = "boot: amdgpu probe ok\n" * 20  # well past 100 bytes
    tail_signature = "XGMI link down detected on dev 1\n"
    blob = head_noise + tail_signature
    assert len(blob) > 100, "fixture must exceed cap to force truncation"

    fired = scan_dmesg_text(blob)
    assert tier3_kernel.DETECTOR_XGMI_LINK_ERROR in fired, (
        "tail-side signature must survive truncation; the previous "
        "head-slice would have discarded it"
    )


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


def test_amdsmi_vram_growth_can_be_disabled(monkeypatch):
    """``check_vram_growth=False`` skips the pre/post VRAM delta leg."""
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=100,throttle=0")
    state = Tier3State()
    pre = poll_amd_smi(state)
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=500,throttle=0")
    post = poll_amd_smi(state)
    fired = scan_amd_smi(state, pre, post, check_vram_growth=False)
    assert tier3_kernel.DETECTOR_VRAM_GROWTH not in fired


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


# ---- Live amd-smi CSV parsing (PR #197 round-2 review) -------------------


def test_parse_amd_smi_monitor_csv_typical_shape():
    """The documented ``amd-smi monitor --csv --gfx --vram-usage``
    output: a single header row + one row per GPU. The parser
    sums VRAM_USED across GPUs and takes the max GFX%.
    """
    payload = (
        "GPU,GFX%,VRAM_USED\n"
        "0,5 %,14 MB\n"
        "1,87 %,1024 MB\n"
        "2,0 %,14 MB\n"
    )
    snap = _parse_amd_smi_monitor_csv(payload)
    assert snap is not None
    assert snap.vram_used_mib == 14 + 1024 + 14
    assert snap.gpu_utilization_pct == 87  # max across GPUs
    assert snap.thermal_throttle_count == 0  # live path leaves this at 0


def test_parse_amd_smi_monitor_csv_handles_units_and_na():
    """N/A cells contribute nothing; GB scales by 1024."""
    payload = (
        "GPU,GFX%,VRAM_USED\n"
        "0,N/A,4 GB\n"   # 4 * 1024 MiB
        "1,42 %,N/A\n"   # contributes util only
    )
    snap = _parse_amd_smi_monitor_csv(payload)
    assert snap is not None
    assert snap.vram_used_mib == 4 * 1024
    assert snap.gpu_utilization_pct == 42


def test_parse_amd_smi_monitor_csv_unknown_header_returns_none():
    """Unknown header (e.g. an older ROCm shipping different column
    names) returns None so the caller logs a single ``tier3
    disabled`` warning and the live path stays fail-soft."""
    payload = "FOO,BAR\n1,2\n"
    assert _parse_amd_smi_monitor_csv(payload) is None


def test_parse_amd_smi_monitor_csv_empty_returns_none():
    """An empty payload returns None (no header to inspect)."""
    assert _parse_amd_smi_monitor_csv("") is None


def test_poll_amd_smi_via_shim(monkeypatch, tmp_path):
    """End-to-end shim test: a fake ``amd-smi`` on PATH emits the
    monitor CSV the parser expects and ``poll_amd_smi`` returns a
    populated snapshot. Mirrors :func:`test_scan_dmesg_via_shim`
    for the new live polling path (PR #197 round-2 review on
    `tier3_kernel.py:277`).
    """
    monkeypatch.delenv("AORTA_PROBE_AMDSMI_FAKE", raising=False)
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "amd-smi"
    shim.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "GPU,GFX%,VRAM_USED\n"
        "0,12 %,256 MB\n"
        "1,80 %,512 MB\n"
        "EOF\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    state = Tier3State()
    snap = poll_amd_smi(state)
    assert snap is not None
    assert snap.vram_used_mib == 256 + 512
    assert snap.gpu_utilization_pct == 80


def test_poll_amd_smi_shim_failure_logs_once(monkeypatch, tmp_path, caplog):
    """A shim that exits non-zero -> ``poll_amd_smi`` returns None and
    logs a single ``tier3 disabled (amd-smi)`` warning (FR 2.11).
    """
    monkeypatch.delenv("AORTA_PROBE_AMDSMI_FAKE", raising=False)
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "amd-smi"
    shim.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    state = Tier3State()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert poll_amd_smi(state) is None
    disabled = [
        r for r in caplog.records
        if "tier3 disabled" in r.getMessage() and "amd-smi" in r.getMessage()
    ]
    assert len(disabled) == 1


# ---- gpu_idle_probe_from_state (PR #197 round-2 review) ------------------


def test_gpu_idle_probe_idle_when_utilization_below_threshold(monkeypatch):
    """``gpu_idle_probe_from_state`` returns True when amd-smi reports
    a max-GPU utilization strictly below
    :data:`GPU_IDLE_UTILIZATION_THRESHOLD_PCT`. Wires the third leg
    of the two-of-three Tier 2 hang predicate.
    """
    idle_pct = max(0, GPU_IDLE_UTILIZATION_THRESHOLD_PCT - 1)
    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", f"vram=100,throttle=0,util={idle_pct}")
    state = Tier3State()
    probe = gpu_idle_probe_from_state(state)
    assert probe() is True


def test_gpu_idle_probe_not_idle_when_utilization_at_or_above_threshold(monkeypatch):
    """Equal to the threshold is NOT idle (strict less-than)."""
    monkeypatch.setenv(
        "AORTA_PROBE_AMDSMI_FAKE",
        f"vram=100,throttle=0,util={GPU_IDLE_UTILIZATION_THRESHOLD_PCT}",
    )
    state = Tier3State()
    assert gpu_idle_probe_from_state(state)() is False


def test_gpu_idle_probe_returns_false_when_amd_smi_unavailable(monkeypatch):
    """No amd-smi on PATH, no fake env var -> probe returns False so
    the GPU leg can never single-handedly trip the two-of-three
    predicate (parity with the I/O leg's None-handling).
    """
    monkeypatch.delenv("AORTA_PROBE_AMDSMI_FAKE", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent-dir-that-does-not-exist")
    state = Tier3State()
    assert gpu_idle_probe_from_state(state)() is False


def test_gpu_idle_probe_returns_false_when_utilization_column_missing(monkeypatch):
    """Live snapshot with no ``gpu_utilization_pct`` (e.g. CSV had
    only VRAM_USED) -> probe returns False, NOT True.
    """
    monkeypatch.delenv("AORTA_PROBE_AMDSMI_FAKE", raising=False)
    state = Tier3State()

    def fake_poll(_state):
        return AmdSmiSnapshot(vram_used_mib=100, thermal_throttle_count=0)

    monkeypatch.setattr(tier3_kernel, "poll_amd_smi", fake_poll)
    assert gpu_idle_probe_from_state(state)() is False
