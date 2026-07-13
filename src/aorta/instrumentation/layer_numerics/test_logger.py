#!/usr/bin/env python3
"""Comprehensive test suite for instrument_nan_logger.py before customer delivery.

Run inside a Docker container with GPU:
    python test_logger.py

Tests:
  1. Backward compatibility: default config produces identical schema to original
  2. NANLOG_BAD_VALUES: various corruption patterns
  3. NANLOG_ALLOC_SNAPSHOT: full lifecycle
  4. Edge cases: empty tensor, scalar, 1D, inf-only, huge-only, all-NaN, large
  5. JSON compliance: every record parseable by json.loads
  6. Graceful degradation: bad env values don't crash
  7. NANLOG_DUMP_TENSOR: full lifecycle, content verification, one-shot guard
  8. NANLOG_DUMP_TENSOR + input channel: validates embed_proj aliasing workflow
  9. NANLOG_DUMP_TENSOR disabled by default: no files, no extra references
 10. NANLOG_DUMP_TENSOR with all flags combined: no interaction bugs
"""
from __future__ import annotations

import importlib
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import torch

PASS = 0
FAIL = 0
TESTS: list[str] = []


def _header(name: str) -> None:
    print(f"\n{'='*72}")
    print(f"  TEST: {name}")
    print(f"{'='*72}")


def _ok(msg: str) -> None:
    global PASS
    PASS += 1
    TESTS.append(f"  PASS: {msg}")
    print(f"  \u2705 {msg}")


def _fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    TESTS.append(f"  FAIL: {msg}")
    print(f"  \u274c {msg}")


def _assert(cond: bool, msg: str) -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg)


def _reload_logger(**env_overrides) -> "module":
    """Reload instrument_nan_logger with fresh env vars.

    Returns the reloaded module. Caller must set NANLOG_DIR to a temp dir."""
    for k in list(os.environ):
        if k.startswith("NANLOG"):
            del os.environ[k]
    for k, v in env_overrides.items():
        os.environ[k] = v

    mod_name = "instrument_nan_logger"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    import instrument_nan_logger as nl
    importlib.reload(nl)
    return nl


# ============================================================================
# Test 1: Backward compatibility
# ============================================================================
def test_backward_compat():
    _header("1. Backward Compatibility — default config schema unchanged")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test1_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir)

        _assert(nl._BAD_VALUES is False, "BAD_VALUES defaults to OFF")
        _assert(nl._ALLOC_SNAPSHOT is False, "ALLOC_SNAPSHOT defaults to OFF")
        _assert(nl._LOCATE is False, "LOCATE defaults to OFF")
        _assert(nl._CAPTURE_ADDR is True, "ADDR defaults to ON")
        _assert(nl._DUMP_TENSOR is False, "DUMP_TENSOR defaults to OFF")

        t = torch.randn(64, 128, device="cuda")
        stats = nl._device_stats(t)
        expected_keys = {"nan_count", "inf_count", "finite_count", "huge_count",
                         "finite_abs_max", "finite_max", "finite_min", "numel"}
        _assert(set(stats.keys()) == expected_keys,
                f"Default stats keys = {sorted(expected_keys)} (no locate/bad_values keys)")

        # Stash + drain produces a record with no extra fields
        nl._pending.clear()
        nl._stash("test.layer", "fwd", t, role="act")
        _assert(len(nl._pending) == 1, "One pending record after stash")
        rec, gpu_stats, held_ref = nl._pending[0]
        _assert("first_bad_flat" not in gpu_stats, "No first_bad_flat in GPU stats")
        _assert("bad_rows" not in gpu_stats, "No bad_rows in GPU stats")
        _assert(held_ref is None, "No tensor ref held when DUMP_TENSOR=0")
        nl._pending.clear()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 2: NANLOG_BAD_VALUES — various corruption patterns
# ============================================================================
def test_bad_values():
    _header("2. NANLOG_BAD_VALUES — corruption pattern identification")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test2_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_BAD_VALUES="1",
                            NANLOG_LOCATE="1")
        _assert(nl._BAD_VALUES is True, "BAD_VALUES is ON")

        drain_keys = ["nan_count", "inf_count", "finite_count", "huge_count",
                      "finite_abs_max", "finite_max", "finite_min", "numel",
                      "bad_rows",
                      "first_bad_flat", "first_bad_val", "first_bad_row", "first_bad_col"]

        def _run_stats(t: torch.Tensor) -> dict:
            stats = nl._device_stats(t)
            flat = [stats[k].to(torch.float64).reshape(()) for k in drain_keys]
            vals = torch.stack(flat).cpu().tolist()
            return dict(zip(drain_keys, vals))

        # Pattern A: single NaN at known position
        t_a = torch.randn(256, 8192, device="cuda")
        t_a[42, 100] = float("nan")
        d = _run_stats(t_a)
        _assert(int(d["nan_count"]) == 1, "PatternA: nan_count=1")
        _assert(int(d["first_bad_row"]) == 42, "PatternA: first_bad_row=42")
        _assert(int(d["first_bad_col"]) == 100, "PatternA: first_bad_col=100")
        _assert(math.isnan(d["first_bad_val"]), "PatternA: first_bad_value is NaN")
        _assert(int(d["bad_rows"]) == 1, "PatternA: bad_rows=1")

        # Pattern B: 1 NaN + 4 huge (Meta's production pattern)
        t_b = torch.randn(256, 8192, device="cuda")
        t_b[42, 100] = float("nan")
        t_b[42, 101] = 3.13e36
        t_b[42, 102] = -2.87e36
        t_b[42, 103] = 1.56e36
        t_b[42, 104] = -4.21e36
        d = _run_stats(t_b)
        _assert(int(d["nan_count"]) == 1 and int(d["huge_count"]) == 4,
                "PatternB: 1 NaN + 4 huge")
        _assert(int(d["bad_rows"]) == 1, "PatternB: all bad in 1 row")
        _assert(int(d["first_bad_row"]) == 42, "PatternB: first_bad at row 42")

        # Pattern C: bad spread across multiple rows (numeric blowup)
        t_c = torch.randn(256, 8192, device="cuda")
        t_c[10, :] = float("nan")
        t_c[20, :] = float("inf")
        t_c[30, :] = 1e20
        d = _run_stats(t_c)
        _assert(int(d["bad_rows"]) >= 3, f"PatternC: bad_rows={int(d['bad_rows'])} >= 3 (blowup)")
        _assert(int(d["first_bad_row"]) == 10, "PatternC: first bad at row 10 (first NaN row)")

        # Pattern D: inf-only, no NaN
        t_d = torch.randn(64, 64, device="cuda")
        t_d[5, 10] = float("inf")
        t_d[5, 11] = float("-inf")
        d = _run_stats(t_d)
        _assert(int(d["nan_count"]) == 0 and int(d["inf_count"]) == 2,
                "PatternD: 0 NaN, 2 Inf")
        _assert(int(d["first_bad_row"]) == 5, "PatternD: first bad row=5")
        _assert(int(d["first_bad_col"]) == 10, "PatternD: first bad col=10")
        _assert(math.isinf(d["first_bad_val"]), "PatternD: first_bad_val is Inf")

        # Pattern E: huge-only, no NaN/Inf
        t_e = torch.randn(32, 32, device="cuda")
        t_e[0, 0] = 2e15
        d = _run_stats(t_e)
        _assert(int(d["nan_count"]) == 0 and int(d["inf_count"]) == 0 and int(d["huge_count"]) == 1,
                "PatternE: 0 NaN, 0 Inf, 1 huge")
        _assert(int(d["first_bad_row"]) == 0 and int(d["first_bad_col"]) == 0,
                "PatternE: first bad at [0,0]")
        _assert(abs(d["first_bad_val"] - 2e15) < 1e10,
                f"PatternE: first_bad_val={d['first_bad_val']:.3e} ≈ 2e15")

        # Pattern F: 1D tensor
        t_f = torch.randn(4096, device="cuda")
        t_f[999] = float("nan")
        d = _run_stats(t_f)
        _assert(int(d["first_bad_col"]) == 999, "Pattern1D: first_bad_col=999 for 1D tensor")
        _assert(int(d["first_bad_row"]) == 0, "Pattern1D: first_bad_row=0 for 1D tensor")

        # Pattern G: all-NaN tensor
        t_g = torch.full((16, 16), float("nan"), device="cuda")
        d = _run_stats(t_g)
        _assert(int(d["nan_count"]) == 256, "AllNaN: nan_count=256")
        _assert(int(d["first_bad_flat"]) == 0, "AllNaN: first_bad_flat=0")

        # Pattern H: clean tensor — no bad elements
        t_h = torch.randn(128, 256, device="cuda")
        d = _run_stats(t_h)
        _assert(int(d["nan_count"]) == 0 and int(d["huge_count"]) == 0,
                "Clean: no bad elements")
        _assert(isinstance(d["first_bad_flat"], float), "Clean: first_bad_flat is a number (ignored)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 3: NANLOG_ALLOC_SNAPSHOT lifecycle
# ============================================================================
def test_alloc_snapshot():
    _header("3. NANLOG_ALLOC_SNAPSHOT — recording + dump + stop lifecycle")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test3_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_ALLOC_SNAPSHOT="1")
        _assert(nl._ALLOC_SNAPSHOT is True, "ALLOC_SNAPSHOT is ON")
        _assert(nl._snapshot_dumped is False, "Not dumped yet")

        # Start recording (in production this is called inside DMP __init__ hook;
        # in this unit test we call it explicitly since there's no DMP)
        nl._start_alloc_recording()
        _assert(nl._alloc_recording_started is True, "Recording started")

        # Do some GPU work
        for _ in range(10):
            x = torch.randn(512, 512, device="cuda")
            y = torch.mm(x, x)
            del x, y
        torch.cuda.synchronize()

        # Trigger dump
        nl._dump_alloc_snapshot(step=99)
        _assert(nl._snapshot_dumped is True, "Dump flag set after first dump")

        snap_files = list(Path(tmpdir).glob("alloc_snapshot_*.pickle"))
        _assert(len(snap_files) == 1, f"Exactly 1 snapshot file: {[f.name for f in snap_files]}")

        if snap_files:
            with open(snap_files[0], "rb") as f:
                snap = pickle.load(f)
            _assert("segments" in snap, "Snapshot contains 'segments'")
            _assert("device_traces" in snap, "Snapshot contains 'device_traces'")
            _assert("allocator_settings" in snap, "Snapshot contains 'allocator_settings'")

            trace = snap["device_traces"][0] if snap["device_traces"] else []
            _assert(len(trace) > 0, f"Device 0 has {len(trace)} trace events")
            if trace:
                ev0 = trace[0]
                _assert("action" in ev0, "Trace event has 'action'")
                _assert("addr" in ev0, "Trace event has 'addr'")
                _assert("stream" in ev0, "Trace event has 'stream'")
                _assert("time_us" in ev0, "Trace event has 'time_us'")
                _assert("frames" in ev0, "Trace event has 'frames' (call stack)")

        # Second dump should be skipped (guard)
        nl._dump_alloc_snapshot(step=100)
        snap_files2 = list(Path(tmpdir).glob("alloc_snapshot_*.pickle"))
        _assert(len(snap_files2) == 1, "Second dump is skipped (still 1 file)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 4: ALLOC_SNAPSHOT disabled by default
# ============================================================================
def test_alloc_snapshot_off():
    _header("4. NANLOG_ALLOC_SNAPSHOT=0 — no recording, no dump")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test4_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir)
        _assert(nl._ALLOC_SNAPSHOT is False, "ALLOC_SNAPSHOT defaults to OFF")

        nl._dump_alloc_snapshot(step=1)
        snap_files = list(Path(tmpdir).glob("alloc_snapshot_*.pickle"))
        _assert(len(snap_files) == 0, "No snapshot file when disabled")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 5: Edge cases — tensor shapes
# ============================================================================
def test_edge_cases():
    _header("5. Edge cases — unusual tensor shapes and dtypes")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test5_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_BAD_VALUES="1",
                            NANLOG_LOCATE="1")

        # Scalar tensor (0-dim)
        t_scalar = torch.tensor(float("nan"), device="cuda")
        stats = nl._device_stats(t_scalar)
        flat_idx = stats["first_bad_flat"].cpu().item()
        _assert(flat_idx == 0, f"Scalar NaN: first_bad_flat=0 (got {flat_idx})")

        # Empty tensor (numel=0) — should not crash _stash
        t_empty = torch.empty(0, 128, device="cuda")
        nl._pending.clear()
        nl._stash("test.empty", "fwd", t_empty, role="act")
        _assert(len(nl._pending) == 0, "Empty tensor: stash skips (numel=0)")

        # Very large tensor (ensure no OOM from extra reductions)
        try:
            t_large = torch.randn(4096, 4096, device="cuda")
            t_large[0, 0] = float("nan")
            stats = nl._device_stats(t_large)
            row = stats["first_bad_row"].cpu().item()
            _assert(row == 0, f"Large tensor: first_bad_row=0 (got {row})")
            del t_large
        except torch.cuda.OutOfMemoryError:
            _ok("Large tensor: skipped (OOM on this GPU)")

        # fp16 tensor
        t_fp16 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        t_fp16[3, 7] = float("nan")
        stats = nl._device_stats(t_fp16)
        row = stats["first_bad_row"].cpu().item()
        col = stats["first_bad_col"].cpu().item()
        _assert(row == 3 and col == 7, f"fp16: first_bad at [{row},{col}] (expected [3,7])")

        # bf16 tensor
        t_bf16 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        t_bf16[1, 2] = float("inf")
        stats = nl._device_stats(t_bf16)
        row = stats["first_bad_row"].cpu().item()
        col = stats["first_bad_col"].cpu().item()
        _assert(row == 1 and col == 2, f"bf16: first_bad at [{row},{col}] (expected [1,2])")

        # 3D tensor: row = dim0 index, col = flat offset within that dim0 slice
        t_3d = torch.randn(8, 32, 64, device="cuda")
        t_3d[2, 10, 30] = float("nan")
        stats = nl._device_stats(t_3d)
        flat_idx = stats["first_bad_flat"].cpu().item()
        expected_flat = 2 * 32 * 64 + 10 * 64 + 30
        _assert(int(flat_idx) == expected_flat,
                f"3D tensor: first_bad_flat={int(flat_idx)} (expected {expected_flat})")
        row = stats["first_bad_row"].cpu().item()
        col = stats["first_bad_col"].cpu().item()
        _assert(int(row) == 2, f"3D tensor: first_bad_row={int(row)} (expected 2, dim0 index)")
        expected_col = 10 * 64 + 30  # flat offset within dim0 slice
        _assert(int(col) == expected_col,
                f"3D tensor: first_bad_col={int(col)} (expected {expected_col})")

        # 4D tensor (batch, channel, H, W)
        t_4d = torch.randn(2, 3, 8, 8, device="cuda")
        t_4d[1, 2, 3, 4] = float("nan")
        stats = nl._device_stats(t_4d)
        row = stats["first_bad_row"].cpu().item()
        col = stats["first_bad_col"].cpu().item()
        _assert(int(row) == 1, f"4D tensor: first_bad_row={int(row)} (expected 1, dim0 index)")
        expected_col_4d = 2 * 8 * 8 + 3 * 8 + 4
        _assert(int(col) == expected_col_4d,
                f"4D tensor: first_bad_col={int(col)} (expected {expected_col_4d})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 6: Full drain + JSONL output + JSON compliance
# ============================================================================
def test_jsonl_output():
    _header("6. JSONL output — full drain cycle with JSON compliance")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test6_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_BAD_VALUES="1",
                            NANLOG_LOCATE="1", NANLOG_ADDR="1",
                            NANLOG_SAMPLE_EVERY="1")

        # Reset state
        nl._step = 0
        nl._records_written = 0
        nl._first_bad = None
        nl._pending.clear()

        # Step 1: clean tensor
        nl._step = 1
        t1 = torch.randn(64, 128, device="cuda")
        nl._stash("model.layer0", "fwd", t1, role="act")
        nl._drain_step()

        # Step 2: bad tensor (NaN + huge)
        nl._step = 2
        t2 = torch.randn(256, 8192, device="cuda")
        t2[42, 100] = float("nan")
        t2[42, 101] = 3.13e36
        nl._stash("model.proj10", "fwd", t2, role="act")
        nl._drain_step()

        # Step 3: inf tensor
        nl._step = 3
        t3 = torch.randn(32, 32, device="cuda")
        t3[0, 0] = float("inf")
        nl._stash("model.layer1", "bwd", t3, role="igrad")
        nl._drain_step()

        # Read and validate JSONL
        jsonl_files = list(Path(tmpdir).glob("layers_rank*.jsonl"))
        _assert(len(jsonl_files) == 1, f"Exactly 1 JSONL file: {[f.name for f in jsonl_files]}")

        if jsonl_files:
            records = []
            parse_errors = 0
            with open(jsonl_files[0]) as fh:
                for i, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        records.append(rec)
                    except json.JSONDecodeError as e:
                        parse_errors += 1
                        _fail(f"Line {i} JSON parse error: {e}")

            _assert(parse_errors == 0, f"All {len(records)} records are valid JSON")
            _assert(len(records) >= 3, f"At least 3 records written (got {len(records)})")

            # Check clean record has no bad_values fields
            clean_recs = [r for r in records if not r.get("bad")]
            if clean_recs:
                _assert("first_bad_value" not in clean_recs[0],
                        "Clean records have no first_bad_* fields")

            # Check bad record has bad_values fields
            bad_recs = [r for r in records if r.get("bad")]
            _assert(len(bad_recs) >= 2, f"{len(bad_recs)} bad records found")
            if bad_recs:
                br = bad_recs[0]
                _assert("first_bad_flat_idx" in br, "Bad record has first_bad_flat_idx")
                _assert("first_bad_row" in br, "Bad record has first_bad_row")
                _assert("first_bad_col" in br, "Bad record has first_bad_col")
                _assert("first_bad_value" in br, "Bad record has first_bad_value")
                _assert(br["first_bad_value"] in ("NaN", "Inf", "-Inf") or isinstance(br["first_bad_value"], (int, float)),
                        f"first_bad_value is JSON-safe: {br['first_bad_value']!r}")

                # Check address fields
                _assert("data_ptr" in br, "Bad record has data_ptr")
                _assert("storage_ptr" in br, "Bad record has storage_ptr")
                _assert("storage_nbytes" in br, "Bad record has storage_nbytes")

                # Verify the NaN record specifically
                nan_recs = [r for r in bad_recs if r.get("nan_count", 0) > 0]
                if nan_recs:
                    nr = nan_recs[0]
                    _assert(nr["first_bad_row"] == 42, f"NaN record: row={nr['first_bad_row']} (expected 42)")
                    _assert(nr["first_bad_col"] == 100, f"NaN record: col={nr['first_bad_col']} (expected 100)")
                    _assert(nr["first_bad_value"] == "NaN", f"NaN record: value={nr['first_bad_value']!r}")
                    _assert(nr["bad_rows"] == 1, f"NaN record: bad_rows={nr['bad_rows']} (expected 1)")

            # Validate JSON re-serialization roundtrip
            for rec in records:
                try:
                    s = json.dumps(rec)
                    json.loads(s)
                except Exception as e:
                    _fail(f"Roundtrip failed for record step={rec.get('step')}: {e}")
                    break
            else:
                _ok("All records pass JSON roundtrip (dumps->loads)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 7: Summary file
# ============================================================================
def test_summary():
    _header("7. Summary file — includes all settings")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test7_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_BAD_VALUES="1",
                            NANLOG_ALLOC_SNAPSHOT="1", NANLOG_LOCATE="1",
                            NANLOG_DUMP_TENSOR="1")

        nl._step = 5
        nl._first_bad = {"step": 3, "layer": "test", "direction": "fwd", "kind": "nan"}
        nl._write_summary()

        summary_files = list(Path(tmpdir).glob("summary_rank*.json"))
        _assert(len(summary_files) == 1, "Exactly 1 summary file")

        if summary_files:
            with open(summary_files[0]) as fh:
                summary = json.load(fh)
            _assert(summary.get("bad_values") is True, "Summary: bad_values=true")
            _assert(summary.get("alloc_snapshot") is True, "Summary: alloc_snapshot=true")
            _assert(summary.get("locate") is True, "Summary: locate=true")
            _assert(summary.get("capture_addr") is True, "Summary: capture_addr=true")
            _assert(summary.get("dump_tensor") is True, "Summary: dump_tensor=true")
            _assert("dump_tensor_dumped" in summary, "Summary: dump_tensor_dumped present")
            _assert("alloc_snapshot_dumped" in summary, "Summary: alloc_snapshot_dumped present")
            _assert(summary.get("first_bad") is not None, "Summary: first_bad recorded")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 8: Address capture cross-referencing
# ============================================================================
def test_addr_cross_ref():
    _header("8. NANLOG_ADDR — cross-reference aliasing between tensors")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test8_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_ADDR="1",
                            NANLOG_SAMPLE_EVERY="1")

        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._pending.clear()

        # Create two tensors that alias the same storage
        base = torch.randn(256, 8192, device="cuda")
        view = base[42:43, :]  # View of row 42

        nl._stash("model.proj10", "fwd", base, role="act")
        nl._stash("model.proj10.view", "fwd", view, role="input")
        nl._drain_step()

        jsonl_files = list(Path(tmpdir).glob("layers_rank*.jsonl"))
        if jsonl_files:
            records = []
            with open(jsonl_files[0]) as fh:
                for line in fh:
                    records.append(json.loads(line.strip()))

            _assert(len(records) == 2, f"2 records from aliased tensors")
            r0, r1 = records[0], records[1]
            _assert(r0.get("storage_ptr") == r1.get("storage_ptr"),
                    f"Aliased tensors share storage_ptr: {r0.get('storage_ptr')}")
            _assert(r0.get("storage_nbytes") == r1.get("storage_nbytes"),
                    f"Same storage_nbytes: {r0.get('storage_nbytes')}")
            _assert(r0.get("data_ptr") != r1.get("data_ptr"),
                    "Different data_ptr (view offset)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 9: BAD_VALUES byte-offset diagnostic
# ============================================================================
def test_byte_offset_diagnostic():
    _header("9. BAD_VALUES byte offset — precise cache-line identification")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test9_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_BAD_VALUES="1",
                            NANLOG_ADDR="1", NANLOG_SAMPLE_EVERY="1")

        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._pending.clear()

        t = torch.randn(256, 8192, device="cuda")
        t[42, 100] = float("nan")

        nl._stash("model.proj10", "fwd", t, role="act")
        nl._drain_step()

        jsonl_files = list(Path(tmpdir).glob("layers_rank*.jsonl"))
        if jsonl_files:
            with open(jsonl_files[0]) as fh:
                rec = json.loads(fh.readline().strip())

            if rec.get("bad"):
                flat_idx = rec["first_bad_flat_idx"]
                element_size = 4  # fp32
                storage_offset = rec.get("storage_offset_bytes", 0)
                byte_offset = storage_offset + flat_idx * element_size

                expected_flat = 42 * 8192 + 100
                expected_byte = expected_flat * 4
                _assert(flat_idx == expected_flat,
                        f"flat_idx={flat_idx} == {expected_flat}")
                _assert(byte_offset == expected_byte,
                        f"byte_offset={byte_offset} == {expected_byte} ({byte_offset/1024:.1f} KB into buffer)")

                storage_ptr = rec.get("storage_ptr")
                _assert(storage_ptr is not None,
                        f"storage_ptr={storage_ptr} + byte_offset={byte_offset} = exact corrupt address")
            else:
                _fail("Expected bad record")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 10: Graceful degradation
# ============================================================================
def test_graceful_degradation():
    _header("10. Graceful degradation — bad values don't crash")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test10_")

    # Test with bad NANLOG_DIR (read-only)
    try:
        ro_dir = tempfile.mkdtemp(prefix="nanlog_ro_")
        os.chmod(ro_dir, 0o444)
        nl = _reload_logger(NANLOG_DIR=ro_dir)
        _ok("Logger survives read-only NANLOG_DIR (falls back to temp)")
        os.chmod(ro_dir, 0o755)
        shutil.rmtree(ro_dir, ignore_errors=True)
    except Exception as e:
        _fail(f"Crashed on read-only dir: {e}")

    # Test _dump_alloc_snapshot when recording was never started
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_ALLOC_SNAPSHOT="0")
        nl._dump_alloc_snapshot(step=1)
        _ok("_dump_alloc_snapshot safe when ALLOC_SNAPSHOT=0")
    except Exception as e:
        _fail(f"_dump_alloc_snapshot crashed when disabled: {e}")

    # Test _dump_bad_tensor when DUMP_TENSOR is disabled
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="0")
        t = torch.randn(32, 32, device="cuda")
        t[0, 0] = float("nan")
        rec = {"step": 1, "layer_name": "test", "role": "act"}
        nl._dump_bad_tensor(rec, t)
        _ok("_dump_bad_tensor safe when DUMP_TENSOR=0")
        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 0, "No .pt file when DUMP_TENSOR=0")
    except Exception as e:
        _fail(f"_dump_bad_tensor crashed when disabled: {e}")

    # Test with non-tensor input to _stash
    try:
        nl._stash("test", "fwd", None, role="act")
        nl._stash("test", "fwd", "not_a_tensor", role="act")
        nl._stash("test", "fwd", 42, role="act")
        _ok("_stash handles non-tensor inputs gracefully")
    except Exception as e:
        _fail(f"_stash crashed on non-tensor input: {e}")

    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 11: NANLOG_DUMP_TENSOR — full lifecycle
# ============================================================================
def test_dump_tensor_lifecycle():
    _header("11. NANLOG_DUMP_TENSOR — full lifecycle: stash, detect, dump, one-shot guard")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test11_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1", NANLOG_ADDR="1")
        _assert(nl._DUMP_TENSOR is True, "DUMP_TENSOR is ON")
        _assert(nl._tensor_dumped is False, "tensor_dumped starts False")

        # Reset state
        nl._step = 0
        nl._records_written = 0
        nl._first_bad = None
        nl._pending.clear()

        # --- Step 1: clean tensor → no dump ---
        nl._step = 1
        t_clean = torch.randn(64, 128, device="cuda")
        nl._stash("model.layer0", "fwd", t_clean, role="act")

        # Verify tensor ref IS held (DUMP_TENSOR=1, not yet dumped)
        _assert(len(nl._pending) == 1, "Step1: 1 pending after stash")
        _, _, held = nl._pending[0]
        _assert(held is not None, "Step1: tensor ref IS held (DUMP_TENSOR=1)")
        _assert(held.data_ptr() == t_clean.data_ptr(), "Step1: held ref is the same tensor")

        nl._drain_step()
        _assert(nl._tensor_dumped is False, "Step1: no dump on clean tensor")
        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 0, "Step1: no .pt file for clean tensor")

        # --- Step 2: bad tensor → dump triggered ---
        nl._step = 2
        t_bad = torch.randn(256, 8192, device="cuda")
        t_bad[42, 100] = float("nan")
        t_bad[42, 101] = 3.13e36
        t_bad[42, 102] = -2.87e36

        nl._stash("model.emb_proj.projections.10.layers.0", "fwd", t_bad, role="input")

        # Verify ref held
        _, _, held = nl._pending[0]
        _assert(held is not None, "Step2: tensor ref held for bad tensor")

        nl._drain_step()
        _assert(nl._tensor_dumped is True, "Step2: dump triggered on first bad")
        _assert(nl._first_bad is not None, "Step2: first_bad set")

        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, f"Step2: exactly 1 .pt file (got {len(pt_files)})")

        if pt_files:
            # Verify filename format
            fname = pt_files[0].name
            _assert("step2" in fname, f"Filename contains step: {fname}")
            _assert("input" in fname, f"Filename contains role: {fname}")
            _assert("rank" in fname, f"Filename contains rank: {fname}")

            # Load and verify content
            loaded = torch.load(pt_files[0], weights_only=True)
            _assert(loaded.shape == torch.Size([256, 8192]),
                    f"Loaded shape={list(loaded.shape)} matches original")
            _assert(loaded.dtype == torch.float32, f"Loaded dtype={loaded.dtype}")
            _assert(loaded.device == torch.device("cpu"),
                    "Loaded tensor is on CPU (was copied from GPU)")

            # Verify the actual bad values are preserved
            _assert(torch.isnan(loaded[42, 100]).item(),
                    "Loaded tensor: NaN at [42,100] preserved")
            _assert(loaded[42, 101].item() > 3e36,
                    f"Loaded tensor: huge at [42,101]={loaded[42,101].item():.3e} preserved")
            _assert(loaded[42, 102].item() < -2e36,
                    f"Loaded tensor: huge at [42,102]={loaded[42,102].item():.3e} preserved")

            # Verify clean values are also preserved (not corrupted by the dump)
            _assert(torch.isfinite(loaded[0, 0]).item(),
                    "Loaded tensor: clean value at [0,0] is finite")
            _assert(torch.isfinite(loaded[100, 100]).item(),
                    "Loaded tensor: clean value at [100,100] is finite")

            # Verify we can do full post-hoc analysis on the dumped tensor
            bad_mask = torch.isnan(loaded) | torch.isinf(loaded) | (loaded.abs() > 1e10)
            bad_locs = torch.nonzero(bad_mask)
            _assert(len(bad_locs) == 3,
                    f"Post-hoc analysis: found {len(bad_locs)} bad elements (expected 3)")
            _assert(bad_locs[0].tolist() == [42, 100],
                    f"Post-hoc: first bad at {bad_locs[0].tolist()} (expected [42,100])")

        # --- Step 3: another bad tensor → should NOT produce a second dump ---
        nl._step = 3
        t_bad2 = torch.randn(64, 64, device="cuda")
        t_bad2[0, 0] = float("inf")
        nl._stash("model.layer1", "fwd", t_bad2, role="act")

        # After first dump, refs should NOT be held anymore
        _, _, held = nl._pending[0]
        _assert(held is None, "Step3: tensor ref NOT held after dump already done")

        nl._drain_step()
        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, "Step3: one-shot guard — still only 1 .pt file")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 12: DUMP_TENSOR with input channel — validates aliasing workflow
# ============================================================================
def test_dump_tensor_input_channel():
    _header("12. DUMP_TENSOR + input channel — embed_proj aliasing workflow")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test12_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_CHANNELS="act,input,igrad",
                            NANLOG_ADDR="1", NANLOG_WATCH_TYPES="Linear",
                            NANLOG_SAMPLE_EVERY="1")

        _assert("input" in nl._CHANNELS, "input channel is enabled")
        _assert("act" in nl._CHANNELS, "act channel is enabled")

        # Build a simple model simulating embed_proj.projections.10
        model = torch.nn.Sequential(
            torch.nn.Linear(8192, 256, bias=False),
        ).cuda()

        # Reset state and attach hooks
        nl._step = 0
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()
        nl._watched_names.clear()
        n = nl._attach(model)
        _assert(n == 1, f"Attached to {n} module (Linear)")

        # Create an input with NaN (simulating aliased corrupt memory)
        x = torch.randn(256, 8192, device="cuda")
        x[42, 100] = float("nan")
        input_data_ptr = hex(x.data_ptr())
        input_storage_ptr = hex(x.untyped_storage().data_ptr())

        # Forward pass — hooks should fire
        nl._step = 1
        out = model(x)
        _assert(len(nl._pending) >= 2,
                f"After forward: {len(nl._pending)} pending (expect >=2: act+input)")

        # Find the input record in pending
        input_entries = [(r, s, h) for r, s, h in nl._pending if r.get("role") == "input"]
        _assert(len(input_entries) >= 1, f"Found {len(input_entries)} input-role pending entries")
        if input_entries:
            rec, _, held = input_entries[0]
            _assert(rec.get("data_ptr") == input_data_ptr,
                    f"Input record data_ptr={rec.get('data_ptr')} matches x ({input_data_ptr})")
            _assert(rec.get("storage_ptr") == input_storage_ptr,
                    f"Input record storage_ptr={rec.get('storage_ptr')} matches x ({input_storage_ptr})")
            _assert(held is not None, "Input tensor ref IS held for dump")

        # Drain — should detect NaN in the input and dump it
        nl._drain_step()
        _assert(nl._tensor_dumped is True, "Dump triggered on NaN input")

        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, f"1 dump file produced (got {len(pt_files)})")

        if pt_files:
            loaded = torch.load(pt_files[0], weights_only=True)
            _assert(torch.isnan(loaded[42, 100]).item(),
                    "Dumped input tensor has NaN at [42,100]")
            _assert(loaded.shape == torch.Size([256, 8192]),
                    f"Dumped input shape={list(loaded.shape)} (the input, not output)")

        # Verify JSONL also has the input record with address
        jsonl_files = list(Path(tmpdir).glob("layers_rank*.jsonl"))
        if jsonl_files:
            records = []
            with open(jsonl_files[0]) as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            input_recs = [r for r in records if r.get("role") == "input"]
            _assert(len(input_recs) >= 1, f"JSONL has {len(input_recs)} input records")
            if input_recs:
                ir = input_recs[0]
                _assert(ir.get("bad") is True, "Input record marked bad=true")
                _assert(ir.get("data_ptr") == input_data_ptr,
                        "JSONL input record has correct data_ptr for cross-referencing")
                _assert(ir.get("storage_ptr") == input_storage_ptr,
                        "JSONL input record has correct storage_ptr for cross-referencing")

        del out

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 13: DUMP_TENSOR disabled — no refs held, no files
# ============================================================================
def test_dump_tensor_off():
    _header("13. NANLOG_DUMP_TENSOR=0 — no tensor refs held, no .pt files")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test13_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="0",
                            NANLOG_SAMPLE_EVERY="1")
        _assert(nl._DUMP_TENSOR is False, "DUMP_TENSOR is OFF")

        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._pending.clear()

        t = torch.randn(64, 128, device="cuda")
        t[0, 0] = float("nan")
        nl._stash("model.layer0", "fwd", t, role="act")

        # Verify no tensor ref held
        _, _, held = nl._pending[0]
        _assert(held is None, "No tensor ref held when DUMP_TENSOR=0")

        nl._drain_step()
        _assert(nl._tensor_dumped is False, "tensor_dumped stays False")
        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 0, "No .pt files when DUMP_TENSOR=0")

        # first_bad still detected normally
        _assert(nl._first_bad is not None, "first_bad detection still works without DUMP_TENSOR")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 14: DUMP_TENSOR with all flags combined — no interaction bugs
# ============================================================================
def test_dump_tensor_all_flags():
    _header("14. DUMP_TENSOR + all flags — no interaction or crash")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test14_")
    try:
        nl = _reload_logger(
            NANLOG_DIR=tmpdir,
            NANLOG_DUMP_TENSOR="1",
            NANLOG_BAD_VALUES="1",
            NANLOG_LOCATE="1",
            NANLOG_ADDR="1",
            NANLOG_CHANNELS="act,input,igrad",
            NANLOG_WATCH_TYPES="Linear",
            NANLOG_SAMPLE_EVERY="1",
            NANLOG_VERBOSE="1",
        )

        _assert(nl._DUMP_TENSOR is True, "DUMP_TENSOR ON")
        _assert(nl._BAD_VALUES is True, "BAD_VALUES ON")
        _assert(nl._LOCATE is True, "LOCATE ON")
        _assert(nl._CAPTURE_ADDR is True, "ADDR ON")
        _assert("input" in nl._CHANNELS, "input channel ON")

        # Build model
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, bias=True),
        ).cuda()

        nl._step = 0
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()
        nl._watched_names.clear()
        n = nl._attach(model)
        _assert(n == 2, f"Attached to {n} Linear modules")

        # Step 1: clean forward — all channels fire, no dump
        nl._step = 1
        x1 = torch.randn(16, 128, device="cuda")
        out1 = model(x1)
        nl._drain_step()
        _assert(nl._tensor_dumped is False, "No dump on clean forward")

        # Step 2: NaN input to second layer (inject between)
        nl._step = 2
        x2 = torch.randn(16, 128, device="cuda")
        # Hook the intermediate: corrupt the ReLU output which is Linear[1]'s input
        with torch.no_grad():
            mid = model[0](x2)
            mid = torch.relu(mid)
            mid[3, 10] = float("nan")
            # Manually stash as if the hook caught it (simulating the input channel)
            nl._stash("1", "fwd", mid, role="input")
            # Also stash the output (simulating act channel)
            out2 = model[2](mid)
            nl._stash("2", "fwd", out2, role="act")

        nl._drain_step()
        _assert(nl._tensor_dumped is True, "Dump triggered with all flags")
        _assert(nl._first_bad is not None, "first_bad set with all flags")

        # Verify files
        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, f"Exactly 1 .pt file with all flags (got {len(pt_files)})")

        jsonl_files = list(Path(tmpdir).glob("layers_rank*.jsonl"))
        if jsonl_files:
            records = []
            with open(jsonl_files[0]) as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line.strip()))

            bad_recs = [r for r in records if r.get("bad")]
            if bad_recs:
                br = bad_recs[0]
                # All enrichment fields present together
                _assert("first_bad_flat_idx" in br, "BAD_VALUES fields present alongside DUMP_TENSOR")
                _assert("bad_rows" in br, "LOCATE fields present alongside DUMP_TENSOR")
                _assert("data_ptr" in br, "ADDR fields present alongside DUMP_TENSOR")
                _assert("storage_ptr" in br, "storage_ptr present alongside DUMP_TENSOR")

        # Summary includes everything
        nl._write_summary()
        summary_files = list(Path(tmpdir).glob("summary_rank*.json"))
        if summary_files:
            with open(summary_files[0]) as fh:
                summary = json.load(fh)
            _assert(summary.get("dump_tensor") is True, "Summary: dump_tensor=true")
            _assert(summary.get("dump_tensor_dumped") is True, "Summary: dump_tensor_dumped=true")
            _assert(summary.get("bad_values") is True, "Summary: bad_values=true")
            _assert(summary.get("locate") is True, "Summary: locate=true")

        del out1, out2

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 15: DUMP_TENSOR — multiple bad tensors in same step, only first dumped
# ============================================================================
def test_dump_tensor_multiple_bad_same_step():
    _header("15. DUMP_TENSOR — multiple bad tensors same step, first one wins")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test15_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1")

        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        # Two bad tensors in the same step
        t1 = torch.randn(32, 64, device="cuda")
        t1[5, 10] = float("nan")
        t2 = torch.randn(64, 128, device="cuda")
        t2[10, 20] = float("inf")

        nl._stash("model.layer_A", "fwd", t1, role="act")
        nl._stash("model.layer_B", "fwd", t2, role="act")

        # Both should have refs held (not yet dumped)
        _, _, h1 = nl._pending[0]
        _, _, h2 = nl._pending[1]
        _assert(h1 is not None, "First bad tensor ref held")
        _assert(h2 is not None, "Second bad tensor ref held")

        nl._drain_step()
        _assert(nl._tensor_dumped is True, "Dump triggered")

        pt_files = sorted(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, f"Only 1 .pt file (one-shot), got {len(pt_files)}")

        if pt_files:
            loaded = torch.load(pt_files[0], weights_only=True)
            # First bad in drain order is t1 (layer_A)
            _assert(loaded.shape == torch.Size([32, 64]),
                    f"Dumped tensor is the FIRST bad one (shape={list(loaded.shape)})")
            _assert(torch.isnan(loaded[5, 10]).item(),
                    "Dumped tensor has NaN at [5,10] (from layer_A)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 16: DUMP_TENSOR — various dtypes (fp16, bf16, fp32)
# ============================================================================
def test_dump_tensor_dtypes():
    _header("16. DUMP_TENSOR — fp16, bf16, fp32 tensors all dump correctly")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test16_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1")

        dtypes = [
            (torch.float32, "fp32"),
            (torch.float16, "fp16"),
            (torch.bfloat16, "bf16"),
        ]

        for dt, name in dtypes:
            # Fresh state for each dtype test
            nl._step = 0
            nl._records_written = 0
            nl._first_bad = None
            nl._tensor_dumped = False
            nl._pending.clear()

            nl._step = 1
            t = torch.randn(64, 64, device="cuda", dtype=dt)
            t[7, 13] = float("nan")
            nl._stash(f"model.{name}_layer", "fwd", t, role="act")
            nl._drain_step()

            pt_files = sorted(Path(tmpdir).glob(f"bad_tensor_*{name}*"))
            _assert(len(pt_files) == 1, f"{name}: dump file created")
            if pt_files:
                loaded = torch.load(pt_files[0], weights_only=True)
                _assert(loaded.dtype == dt, f"{name}: dtype preserved ({loaded.dtype})")
                _assert(loaded.shape == torch.Size([64, 64]), f"{name}: shape preserved")
                _assert(torch.isnan(loaded[7, 13]).item(), f"{name}: NaN at [7,13] preserved")
                _assert(loaded.device == torch.device("cpu"), f"{name}: on CPU after dump")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 17: DUMP_TENSOR — large tensor (production-size embed_proj input)
# ============================================================================
def test_dump_tensor_large():
    _header("17. DUMP_TENSOR — production-size tensor (256x8192 fp32 = 8 MB)")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test17_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1")

        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        # Production-size: 256 x 8192 fp32 = 8 MiB
        t = torch.randn(256, 8192, device="cuda")
        # Meta's observed pattern: 1 NaN + 4 huge in one row
        t[42, 100] = float("nan")
        t[42, 101] = 3.13e36
        t[42, 102] = -2.87e36
        t[42, 103] = 1.56e36
        t[42, 104] = -4.21e36

        nl._stash("model.emb_proj.projections.10.layers.0", "fwd", t, role="input")
        nl._drain_step()

        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, "Large tensor dump produced")

        if pt_files:
            file_size = pt_files[0].stat().st_size
            _assert(file_size > 7 * 1024 * 1024,
                    f"File size={file_size/1024/1024:.1f} MB (expected ~8 MB for 256x8192 fp32)")

            loaded = torch.load(pt_files[0], weights_only=True)
            _assert(loaded.shape == torch.Size([256, 8192]), "Shape correct")

            # Verify ALL bad values preserved exactly
            _assert(torch.isnan(loaded[42, 100]).item(), "NaN preserved")
            _assert(abs(loaded[42, 101].item() - 3.13e36) < 1e33, "Huge[101] preserved")
            _assert(abs(loaded[42, 102].item() - (-2.87e36)) < 1e33, "Huge[102] preserved")
            _assert(abs(loaded[42, 103].item() - 1.56e36) < 1e33, "Huge[103] preserved")
            _assert(abs(loaded[42, 104].item() - (-4.21e36)) < 1e33, "Huge[104] preserved")

            # Full post-hoc analysis: find all bad, verify row pattern
            bad_mask = torch.isnan(loaded) | torch.isinf(loaded) | (loaded.abs() > 1e10)
            bad_locs = torch.nonzero(bad_mask)
            _assert(len(bad_locs) == 5, f"Post-hoc: 5 bad elements found (got {len(bad_locs)})")
            bad_rows = bad_locs[:, 0].unique()
            _assert(len(bad_rows) == 1 and bad_rows[0].item() == 42,
                    f"Post-hoc: all bad in row 42 (aliasing signature)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 18: DUMP_TENSOR — tensor that is a view (non-contiguous)
# ============================================================================
def test_dump_tensor_view():
    _header("18. DUMP_TENSOR — view/non-contiguous tensor dumps correctly")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test18_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1", NANLOG_ADDR="1")

        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        # Create a view (non-contiguous slice) — this is common in production
        # when the input to a layer is a slice of a larger buffer
        base = torch.randn(512, 8192, device="cuda")
        view = base[42:43, :]  # single row view, non-contiguous in storage
        view[0, 0] = float("nan")

        _assert(not view.is_contiguous() or view.storage_offset() > 0,
                "View is offset from base storage (realistic aliasing scenario)")

        nl._stash("model.proj10", "fwd", view, role="input")
        nl._drain_step()

        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, "View tensor dump produced")

        if pt_files:
            loaded = torch.load(pt_files[0], weights_only=True)
            _assert(loaded.shape == torch.Size([1, 8192]),
                    f"Dumped view shape={list(loaded.shape)} (just the view, not full base)")
            _assert(torch.isnan(loaded[0, 0]).item(), "NaN at [0,0] preserved in view dump")

            # The dump should contain ONLY the view data, not the full 512x8192 base
            file_size = pt_files[0].stat().st_size
            _assert(file_size < 100 * 1024,
                    f"File size={file_size/1024:.0f} KB (view is small, not full base)")

        # Verify JSONL has the STORAGE address (pointing to base), not just data_ptr
        jsonl_files = list(Path(tmpdir).glob("layers_rank*.jsonl"))
        if jsonl_files:
            with open(jsonl_files[0]) as fh:
                rec = json.loads(fh.readline().strip())
            _assert(rec.get("storage_ptr") == hex(base.untyped_storage().data_ptr()),
                    "JSONL storage_ptr points to base buffer (for aliasing cross-ref)")
            _assert(rec.get("storage_offset_bytes") > 0,
                    f"storage_offset_bytes={rec.get('storage_offset_bytes')} > 0 (view is offset)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 19: DUMP_TENSOR — inf-triggered and huge-triggered dumps
# ============================================================================
def test_dump_tensor_inf_and_huge():
    _header("19. DUMP_TENSOR — triggers on Inf and huge (not just NaN)")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test19_")
    try:
        # Test Inf trigger
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1")
        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        t_inf = torch.randn(32, 32, device="cuda")
        t_inf[2, 3] = float("inf")
        nl._stash("model.layer_inf", "fwd", t_inf, role="act")
        nl._drain_step()

        _assert(nl._tensor_dumped is True, "Dump triggers on Inf")
        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, "Inf-triggered dump file exists")
        if pt_files:
            loaded = torch.load(pt_files[0], weights_only=True)
            _assert(torch.isinf(loaded[2, 3]).item(), "Inf preserved in dump")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Test huge trigger (separate run to get clean state)
    tmpdir2 = tempfile.mkdtemp(prefix="nanlog_test19b_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir2, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1",
                            NANLOG_HUGE_THRESHOLD="1e5")
        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        t_huge = torch.randn(32, 32, device="cuda")
        t_huge[4, 5] = 2e6  # above threshold of 1e5
        nl._stash("model.layer_huge", "fwd", t_huge, role="act")
        nl._drain_step()

        _assert(nl._tensor_dumped is True, "Dump triggers on huge (above threshold)")
        pt_files = list(Path(tmpdir2).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, "Huge-triggered dump file exists")
        if pt_files:
            loaded = torch.load(pt_files[0], weights_only=True)
            _assert(abs(loaded[4, 5].item() - 2e6) < 100,
                    f"Huge value preserved: {loaded[4, 5].item():.3e}")

    finally:
        shutil.rmtree(tmpdir2, ignore_errors=True)


# ============================================================================
# Test 20: DUMP_TENSOR — file naming with special characters in layer name
# ============================================================================
def test_dump_tensor_filename_safety():
    _header("20. DUMP_TENSOR — safe filename from complex layer names")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test20_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1")
        nl._step = 42
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        t = torch.randn(16, 16, device="cuda")
        t[0, 0] = float("nan")

        # Realistic complex layer name with dots
        nl._stash("model.emb_proj.projections.10.layers.0.linear", "fwd", t, role="input")
        nl._drain_step()

        pt_files = list(Path(tmpdir).glob("bad_tensor_*.pt"))
        _assert(len(pt_files) == 1, "Dump file created for complex layer name")
        if pt_files:
            fname = pt_files[0].name
            # Should not contain dots (replaced with _) except .pt extension
            name_without_ext = fname.rsplit(".pt", 1)[0]
            _assert("." not in name_without_ext,
                    f"No dots in filename body (safe for filesystems): {fname}")
            _assert("step42" in fname, f"Step in filename: {fname}")
            _assert("input" in fname, f"Role in filename: {fname}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 21: DUMP_TENSOR — _dump_bad_tensor is exception-safe
# ============================================================================
def test_dump_tensor_exception_safety():
    _header("21. DUMP_TENSOR — exception in dump does not crash training")
    tmpdir = tempfile.mkdtemp(prefix="nanlog_test21_")
    try:
        nl = _reload_logger(NANLOG_DIR=tmpdir, NANLOG_DUMP_TENSOR="1",
                            NANLOG_SAMPLE_EVERY="1")
        nl._step = 1
        nl._records_written = 0
        nl._first_bad = None
        nl._tensor_dumped = False
        nl._pending.clear()

        # Create a read-only output dir AFTER logger init (to force save failure)
        ro_subdir = Path(tmpdir) / "locked"
        ro_subdir.mkdir()
        os.chmod(str(ro_subdir), 0o444)

        # Point logger to the locked dir
        nl._DIR = ro_subdir

        t = torch.randn(16, 16, device="cuda")
        t[0, 0] = float("nan")
        rec = {"step": 1, "layer_name": "test.layer", "role": "act"}

        # Should not raise — exception is caught internally
        try:
            nl._dump_bad_tensor(rec, t)
            _ok("_dump_bad_tensor does not crash on write failure")
        except Exception as e:
            _fail(f"_dump_bad_tensor raised: {e}")

        # Guard should still be set (don't retry on next bad)
        _assert(nl._tensor_dumped is True, "Guard set even on failure (no infinite retries)")

        os.chmod(str(ro_subdir), 0o755)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("  NaN Logger v2 — Comprehensive Pre-delivery Test Suite")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    if not torch.cuda.is_available():
        print("\nFATAL: No GPU available. Tests require CUDA/ROCm GPU.")
        sys.exit(1)

    # Ensure the logger module is importable
    logger_dir = Path(__file__).parent
    if str(logger_dir) not in sys.path:
        sys.path.insert(0, str(logger_dir))

    tests = [
        test_backward_compat,
        test_bad_values,
        test_alloc_snapshot,
        test_alloc_snapshot_off,
        test_edge_cases,
        test_jsonl_output,
        test_summary,
        test_addr_cross_ref,
        test_byte_offset_diagnostic,
        test_graceful_degradation,
        test_dump_tensor_lifecycle,
        test_dump_tensor_input_channel,
        test_dump_tensor_off,
        test_dump_tensor_all_flags,
        test_dump_tensor_multiple_bad_same_step,
        test_dump_tensor_dtypes,
        test_dump_tensor_large,
        test_dump_tensor_view,
        test_dump_tensor_inf_and_huge,
        test_dump_tensor_filename_safety,
        test_dump_tensor_exception_safety,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            _fail(f"TEST CRASHED: {test_fn.__name__}")
            traceback.print_exc()

    print(f"\n{'='*72}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*72}")
    for t in TESTS:
        print(t)
    print()

    if FAIL > 0:
        print(f"  *** {FAIL} FAILURE(S) — DO NOT DELIVER ***")
        sys.exit(1)
    else:
        print("  *** ALL TESTS PASSED — READY FOR DELIVERY ***")
        sys.exit(0)


if __name__ == "__main__":
    main()
