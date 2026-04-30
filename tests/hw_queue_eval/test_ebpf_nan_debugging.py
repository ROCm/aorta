"""
Tests for the eBPF NaN debugging modules (race detector, DMA tracer,
RCCL tracer, NaN correlator).

All subprocess and filesystem calls are mocked so tests run without
real GPUs, bpftrace, or root privileges.  Modules are loaded directly
by file path to avoid the torch-dependent aorta import chain.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Direct module loading (bypasses torch-dependent aorta.hw_queue_eval.__init__)
# ---------------------------------------------------------------------------

_CORE_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "src", "aorta", "hw_queue_eval", "core",
)

# Test-file-specific module name prefix.  Both this file and
# ``test_ebpf_tracer.py`` load the same source files via importlib;
# using bare names like ``ebpf_tracer`` would cause whichever test
# imports first to overwrite the other's module objects in
# ``sys.modules``, making the suite order-dependent.  Prefixing with
# ``_nandbg__`` keeps the two test files' module namespaces disjoint.
_NAME_PREFIX = "_nandbg__"


def _load_module(name: str, filename: str):
    """Load a core module under a test-file-specific sys.modules name.

    Note: any cross-module imports done via ``from X import ...`` inside
    the loaded files still resolve through normal package machinery, so
    there's no need to also register the bare ``name`` in ``sys.modules``
    here -- doing so is exactly what caused the collision with
    ``test_ebpf_tracer.py``.
    """
    qualified = _NAME_PREFIX + name
    filepath = os.path.join(_CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(qualified, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = mod
    spec.loader.exec_module(mod)
    return mod


_ebpf_tracer = _load_module("ebpf_tracer", "ebpf_tracer.py")
_ebpf_memory_tracer = _load_module("ebpf_memory_tracer", "ebpf_memory_tracer.py")
_race_detector = _load_module("ebpf_race_detector", "ebpf_race_detector.py")
_dma_tracer = _load_module("ebpf_dma_tracer", "ebpf_dma_tracer.py")
_rccl_tracer = _load_module("ebpf_rccl_tracer", "ebpf_rccl_tracer.py")
_nan_correlator = _load_module("ebpf_nan_correlator", "ebpf_nan_correlator.py")

BPFRaceDetector = _race_detector.BPFRaceDetector
RaceDetectionMetrics = _race_detector.RaceDetectionMetrics
RaceEvent = _race_detector.RaceEvent

BPFDMATracer = _dma_tracer.BPFDMATracer
DMATraceMetrics = _dma_tracer.DMATraceMetrics
DMAOverlapEvent = _dma_tracer.DMAOverlapEvent

BPFRCCLTracer = _rccl_tracer.BPFRCCLTracer
RCCLTraceMetrics = _rccl_tracer.RCCLTraceMetrics
CollectiveRaceEvent = _rccl_tracer.CollectiveRaceEvent

NaNCorrelator = _nan_correlator.NaNCorrelator
NaNDetection = _nan_correlator.NaNDetection
CorrelatedNaNReport = _nan_correlator.CorrelatedNaNReport

DriverQueueEvent = _ebpf_tracer.DriverQueueEvent
MemoryTraceEvent = _ebpf_memory_tracer.MemoryTraceEvent


# ===========================================================================
# BPFRaceDetector
# ===========================================================================

class TestRaceDetectionMetrics:

    def test_empty(self):
        m = RaceDetectionMetrics()
        assert m.races_detected == 0
        assert m.race_rate_per_sec == 0.0
        d = m.to_dict()
        assert d["races_detected"] == 0
        assert d["race_events"] == []

    def test_to_dict(self):
        m = RaceDetectionMetrics(
            total_submissions=100,
            races_detected=3,
            rings_with_races=[0, 2],
            trace_duration_ms=5000.0,
        )
        d = m.to_dict()
        assert d["total_submissions"] == 100
        assert d["races_detected"] == 3
        assert d["rings_with_races"] == [0, 2]
        assert d["race_rate_per_sec"] == pytest.approx(0.6)


class TestRaceEvent:

    def test_to_dict(self):
        ev = RaceEvent(
            timestamp_ns=2000000000,
            ring=1,
            submit_a_ts=1999900000,
            submit_a_pid=42,
            submit_a_comm="python",
            submit_a_fence=10,
            submit_b_ts=2000000000,
            submit_b_pid=42,
            submit_b_comm="python",
            submit_b_fence=15,
            gap_us=100.0,
            fence_gap=5,
        )
        d = ev.to_dict()
        assert d["ring"] == 1
        assert d["gap_us"] == 100.0
        assert d["fence_gap"] == 5
        assert d["submit_a"]["pid"] == 42
        assert d["submit_b"]["fence"] == 15

    def test_timestamp_ms(self):
        ev = RaceEvent(
            timestamp_ns=5_000_000, ring=0,
            submit_a_ts=0, submit_a_pid=1, submit_a_comm="a", submit_a_fence=0,
            submit_b_ts=0, submit_b_pid=1, submit_b_comm="b", submit_b_fence=0,
            gap_us=0, fence_gap=0,
        )
        assert ev.timestamp_ms == pytest.approx(5.0)


class TestBPFRaceDetector:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(target_pid=123, output_dir=Path(tmpdir))
            assert det._target_pid == 123
            assert det.is_running is False

    @patch.object(_race_detector, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script(self, _mock_probe):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(target_pid=99, output_dir=Path(tmpdir))
            path = det._generate_script()
            content = path.read_text()
            assert "RACE_SUBMIT" in content
            # RACE_DISPATCH was removed: the Python heuristic only uses
            # submissions, so emitting dispatch events from
            # ``amdgpu_sched_run_job`` was pure overhead.
            assert "RACE_DISPATCH" not in content
            assert "amdgpu_sched_run_job" not in content
            assert "pid == 99" in content

    @patch.object(_race_detector, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script_no_pid(self, _mock_probe):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(target_pid=None, output_dir=Path(tmpdir))
            path = det._generate_script()
            content = path.read_text()
            assert "pid ==" not in content

    def test_parse_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            det = BPFRaceDetector(output_dir=tmpdir)
            log = tmpdir / "race_detect.log"
            log.write_text(
                "RACE_SUBMIT|1000000000|42|100|python|0|10\n"
                "RACE_SUBMIT|1000050000|42|101|python|0|15\n"
                "RACE_DISPATCH|1000100000|42|100|python|0|10\n"
                "garbage line\n"
            )
            det._output_path = log
            events = det._parse_output()
            assert len(events) == 3
            assert events[0].event_type == "submit"
            assert events[0].tid == 100
            assert events[1].tid == 101

    def test_detect_race_same_ring_different_threads(self):
        """Two submits on same ring from different threads within window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(output_dir=Path(tmpdir), race_window_us=200.0)
            events = [
                _race_detector._RawEvent(1000000000, "submit", 42, 100, "python", 0, 10),
                _race_detector._RawEvent(1000050000, "submit", 42, 101, "python", 0, 15),
            ]
            metrics = det._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 1
            assert metrics.rings_with_races == [0]
            assert metrics.race_events[0].gap_us == pytest.approx(50.0)
            assert metrics.race_events[0].fence_gap == 5

    def test_no_race_same_thread(self):
        """Submits from same thread should not be flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(output_dir=Path(tmpdir), race_window_us=200.0)
            events = [
                _race_detector._RawEvent(1000000000, "submit", 42, 100, "python", 0, 10),
                _race_detector._RawEvent(1000050000, "submit", 42, 100, "python", 0, 11),
            ]
            metrics = det._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 0

    def test_no_race_consecutive_fences(self):
        """Consecutive fences indicate proper synchronization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(output_dir=Path(tmpdir), race_window_us=200.0)
            events = [
                _race_detector._RawEvent(1000000000, "submit", 42, 100, "python", 0, 10),
                _race_detector._RawEvent(1000050000, "submit", 42, 101, "python", 0, 11),
            ]
            metrics = det._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 0

    def test_no_race_outside_window(self):
        """Events outside the time window should not be flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(output_dir=Path(tmpdir), race_window_us=50.0)
            events = [
                _race_detector._RawEvent(1000000000, "submit", 42, 100, "python", 0, 10),
                _race_detector._RawEvent(1000100000, "submit", 42, 101, "python", 0, 15),
            ]
            metrics = det._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 0

    def test_stop_without_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(output_dir=Path(tmpdir))
            metrics = det.stop()
            assert isinstance(metrics, RaceDetectionMetrics)
            assert metrics.races_detected == 0

    @patch.object(_race_detector, "shutil")
    def test_start_raises_without_bpftrace(self, mock_shutil):
        mock_shutil.which.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            det = BPFRaceDetector(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace is not installed"):
                det.start()


# ===========================================================================
# BPFDMATracer
# ===========================================================================

class TestDMATraceMetrics:

    def test_empty(self):
        m = DMATraceMetrics()
        d = m.to_dict()
        assert d["overlaps_detected"] == 0
        assert d["overlap_events"] == []

    def test_to_dict(self):
        m = DMATraceMetrics(
            total_bo_moves=50,
            total_compute_submits=200,
            overlaps_detected=5,
            max_overlap_us=300.0,
            avg_overlap_us=150.0,
        )
        d = m.to_dict()
        assert d["total_bo_moves"] == 50
        assert d["overlaps_detected"] == 5
        assert d["max_overlap_us"] == 300.0


class TestBPFDMATracer:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(target_pid=456, output_dir=Path(tmpdir))
            assert tracer._target_pid == 456

    @patch.object(_dma_tracer, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script(self, _mock_probe):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(target_pid=42, output_dir=Path(tmpdir))
            path = tracer._generate_script()
            content = path.read_text()
            assert "DMA_MOVE" in content
            assert "DMA_CS" in content
            assert "pid == 42" in content

    def test_parse_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tracer = BPFDMATracer(output_dir=tmpdir)
            log = tmpdir / "dma_trace.log"
            log.write_text(
                "DMA_MOVE|1000000000|42|amdgpu|65536\n"
                "DMA_CS|1000200000|42|python|0\n"
                "garbage\n"
                "DMA_MOVE|1000300000|42|amdgpu|32768\n"
            )
            tracer._output_path = log
            events = tracer._parse_output()
            assert len(events) == 3
            assert events[0].event_type == "bo_move"
            assert events[0].value == 65536
            assert events[1].event_type == "compute"

    def test_detect_overlap(self):
        """Compute submit within overlap window of a BO move."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(output_dir=Path(tmpdir), overlap_window_us=500.0)
            events = [
                _dma_tracer._RawDMAEvent(1000000000, "bo_move", 42, "amdgpu", 65536),
                _dma_tracer._RawDMAEvent(1000200000, "compute", 42, "python", 0),
            ]
            metrics = tracer._detect_overlaps(events, 1_000_000_000)
            assert metrics.overlaps_detected == 1
            assert metrics.overlap_events[0].overlap_us == pytest.approx(200.0)
            assert metrics.total_bo_moves == 1
            assert metrics.total_compute_submits == 1

    def test_no_overlap_outside_window(self):
        """Compute submit beyond the overlap window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(output_dir=Path(tmpdir), overlap_window_us=100.0)
            events = [
                _dma_tracer._RawDMAEvent(1000000000, "bo_move", 42, "amdgpu", 65536),
                _dma_tracer._RawDMAEvent(1000200000, "compute", 42, "python", 0),
            ]
            metrics = tracer._detect_overlaps(events, 1_000_000_000)
            assert metrics.overlaps_detected == 0

    def test_max_and_avg_overlap(self):
        """Second compute sees both BO moves within window -> 3 overlaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(output_dir=Path(tmpdir), overlap_window_us=500.0)
            events = [
                _dma_tracer._RawDMAEvent(1000000000, "bo_move", 42, "a", 1024),
                _dma_tracer._RawDMAEvent(1000100000, "compute", 42, "p", 0),
                _dma_tracer._RawDMAEvent(1000200000, "bo_move", 42, "a", 2048),
                _dma_tracer._RawDMAEvent(1000500000, "compute", 42, "p", 0),
            ]
            metrics = tracer._detect_overlaps(events, 1_000_000_000)
            assert metrics.overlaps_detected == 3
            assert metrics.max_overlap_us == pytest.approx(500.0)
            assert metrics.avg_overlap_us == pytest.approx(300.0)

    def test_stop_without_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(output_dir=Path(tmpdir))
            metrics = tracer.stop()
            assert isinstance(metrics, DMATraceMetrics)

    @patch.object(_dma_tracer, "shutil")
    def test_start_raises_without_bpftrace(self, mock_shutil):
        mock_shutil.which.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFDMATracer(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace is not installed"):
                tracer.start()


# ===========================================================================
# BPFRCCLTracer
# ===========================================================================

class TestRCCLTraceMetrics:

    def test_empty(self):
        m = RCCLTraceMetrics()
        d = m.to_dict()
        assert d["races_detected"] == 0
        assert d["collective_submissions"] == 0

    def test_to_dict(self):
        m = RCCLTraceMetrics(
            total_submissions=300,
            collective_submissions=50,
            compute_submissions=250,
            races_detected=2,
        )
        d = m.to_dict()
        assert d["total_submissions"] == 300
        assert d["collective_submissions"] == 50
        assert d["races_detected"] == 2


class TestBPFRCCLTracer:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(target_pid=789, output_dir=Path(tmpdir))
            assert tracer._target_pid == 789

    @patch.object(_rccl_tracer, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script(self, _mock_probe):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(target_pid=42, output_dir=Path(tmpdir))
            path = tracer._generate_script()
            content = path.read_text()
            assert "RCCL_CS" in content
            assert "pid == 42" in content

    def test_parse_output_classifies_collective(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tracer = BPFRCCLTracer(output_dir=tmpdir)
            log = tmpdir / "rccl_trace.log"
            log.write_text(
                "RCCL_CS|1000000000|42|100|ncclKernAllReduce|0|1\n"
                "RCCL_CS|1000050000|42|101|python|1|2\n"
                "RCCL_CS|1000100000|42|102|rccl_worker|0|3\n"
            )
            tracer._output_path = log
            events = tracer._parse_output()
            assert len(events) == 3
            assert events[0].is_collective is True
            assert events[1].is_collective is False
            assert events[2].is_collective is True

    def test_detect_cross_ring_collective_observation(self):
        """Compute submit on a *different* ring inside the window is recorded
        as a cross-ring observation, not a confirmed race -- cross-ring WAR
        hazards depend on fence/barrier state that this tracer does not yet
        inspect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(output_dir=Path(tmpdir), race_window_us=200.0)
            events = [
                _rccl_tracer._RawRCCLEvent(
                    1000000000, 42, 100, "ncclAllReduce", 0, 1, True,
                ),
                _rccl_tracer._RawRCCLEvent(
                    1000100000, 42, 101, "python", 1, 2, False,
                ),
            ]
            metrics = tracer._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 0
            assert metrics.cross_ring_observations == 1
            assert metrics.race_events[0].same_ring is False
            assert metrics.race_events[0].gap_us == pytest.approx(100.0)

    def test_detect_same_ring_collective_race(self):
        """Same-ring collective-compute race is flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(output_dir=Path(tmpdir), race_window_us=200.0)
            events = [
                _rccl_tracer._RawRCCLEvent(
                    1000000000, 42, 100, "ncclAllReduce", 0, 1, True,
                ),
                _rccl_tracer._RawRCCLEvent(
                    1000050000, 42, 101, "python", 0, 2, False,
                ),
            ]
            metrics = tracer._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 1
            assert metrics.race_events[0].same_ring is True

    def test_no_race_outside_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(output_dir=Path(tmpdir), race_window_us=50.0)
            events = [
                _rccl_tracer._RawRCCLEvent(
                    1000000000, 42, 100, "ncclAllReduce", 0, 1, True,
                ),
                _rccl_tracer._RawRCCLEvent(
                    1000100000, 42, 101, "python", 1, 2, False,
                ),
            ]
            metrics = tracer._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 0

    def test_no_race_compute_only(self):
        """All compute, no collectives -- no races."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(output_dir=Path(tmpdir), race_window_us=200.0)
            events = [
                _rccl_tracer._RawRCCLEvent(
                    1000000000, 42, 100, "python", 0, 1, False,
                ),
                _rccl_tracer._RawRCCLEvent(
                    1000050000, 42, 101, "python", 0, 2, False,
                ),
            ]
            metrics = tracer._detect_races(events, 1_000_000_000)
            assert metrics.races_detected == 0

    def test_stop_without_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(output_dir=Path(tmpdir))
            metrics = tracer.stop()
            assert isinstance(metrics, RCCLTraceMetrics)

    @patch.object(_rccl_tracer, "shutil")
    def test_start_raises_without_bpftrace(self, mock_shutil):
        mock_shutil.which.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFRCCLTracer(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace is not installed"):
                tracer.start()


# ===========================================================================
# NaNCorrelator
# ===========================================================================

class TestNaNDetection:

    def test_timestamp_ms(self):
        ev = NaNDetection(timestamp_ns=5_000_000, step=1)
        assert ev.timestamp_ms == pytest.approx(5.0)


class TestNaNCorrelator:

    def _make_queue_events(self):
        return [
            DriverQueueEvent(1000000000, "submit", 42, "python", ring=0, fence=1),
            DriverQueueEvent(1000100000, "dispatch", 42, "python", ring=0, fence=1),
            DriverQueueEvent(1000200000, "submit", 42, "python", ring=1, fence=2),
            DriverQueueEvent(1000300000, "dispatch", 42, "python", ring=1, fence=2),
        ]

    def _make_memory_events(self):
        return [
            MemoryTraceEvent(1000050000, "bo_move", 42, "amdgpu", size_bytes=4096),
            MemoryTraceEvent(1000150000, "evict", 42, "kfd", size_bytes=0),
            MemoryTraceEvent(1000250000, "restore", 42, "kfd", size_bytes=0),
        ]

    def test_correlate_with_timestamp(self):
        correlator = NaNCorrelator(window_ms=1.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=1000150000, step=10, rank=0, source="post-all_reduce",
        ))
        correlator.set_queue_events(self._make_queue_events())
        correlator.set_memory_events(self._make_memory_events())

        reports = correlator.correlate()
        assert len(reports) == 1
        r = reports[0]
        assert r.queue_events_in_window > 0
        assert r.memory_events_in_window > 0
        assert r.evictions_in_window >= 1

    def test_correlate_without_timestamp(self):
        """When NaN timestamp is 0, all events are included."""
        correlator = NaNCorrelator(window_ms=1.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=0, step=5, rank=0, source="post-all_reduce",
        ))
        correlator.set_queue_events(self._make_queue_events())
        correlator.set_memory_events(self._make_memory_events())

        reports = correlator.correlate()
        assert len(reports) == 1
        r = reports[0]
        assert r.queue_events_in_window == 4
        assert r.memory_events_in_window == 3

    def test_correlate_with_race_events(self):
        correlator = NaNCorrelator(window_ms=100.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=0, step=1, rank=0, source="test",
        ))

        race_ev = RaceEvent(
            timestamp_ns=1000100000, ring=0,
            submit_a_ts=1000000000, submit_a_pid=42, submit_a_comm="p",
            submit_a_fence=10,
            submit_b_ts=1000100000, submit_b_pid=42, submit_b_comm="p",
            submit_b_fence=15,
            gap_us=100.0, fence_gap=5,
        )
        correlator.set_race_events([race_ev])

        reports = correlator.correlate()
        assert reports[0].race_events_in_window == 1
        assert "stream_race_detected" in reports[0].diagnosis
        assert reports[0].confidence in ("medium", "high")

    def test_correlate_with_dma_overlaps(self):
        correlator = NaNCorrelator(window_ms=100.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=0, step=1, rank=0, source="test",
        ))

        dma_ev = DMAOverlapEvent(
            timestamp_ns=1000200000,
            bo_move_start_ns=1000000000, compute_submit_ns=1000200000,
            overlap_us=200.0,
            bo_move_pid=42, bo_move_comm="a", bo_move_size=65536,
            compute_pid=42, compute_comm="p", compute_ring=0,
        )
        correlator.set_dma_overlaps([dma_ev])

        reports = correlator.correlate()
        assert reports[0].dma_overlaps_in_window == 1
        assert "h2d_dma_overlap" in reports[0].diagnosis

    def test_correlate_with_collective_races(self):
        correlator = NaNCorrelator(window_ms=100.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=0, step=1, rank=0, source="test",
        ))

        coll_ev = CollectiveRaceEvent(
            timestamp_ns=1000100000,
            collective_ring=0, collective_ts=1000000000,
            collective_pid=42, collective_comm="nccl",
            compute_ring=1, compute_ts=1000100000,
            compute_pid=42, compute_comm="python",
            gap_us=100.0, same_ring=False,
        )
        correlator.set_collective_races([coll_ev])

        reports = correlator.correlate()
        assert reports[0].collective_races_in_window == 1
        assert "collective_compute_race" in reports[0].diagnosis

    def test_dispatch_gap_spike_detection(self):
        correlator = NaNCorrelator(window_ms=100.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=0, step=1, rank=0, source="test",
        ))

        queue_events = [
            DriverQueueEvent(1000000000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1000010000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1000020000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1000030000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1000040000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1000050000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1000060000, "dispatch", 42, "p", ring=0),
            DriverQueueEvent(1005000000, "dispatch", 42, "p", ring=0),
        ]
        correlator.set_queue_events(queue_events)

        reports = correlator.correlate()
        assert reports[0].dispatch_gap_spike is True
        assert "dispatch_stall" in reports[0].diagnosis

    def test_no_kernel_events_diagnosis(self):
        correlator = NaNCorrelator(window_ms=0.001)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=9999999999999, step=1, rank=0, source="test",
        ))
        correlator.set_queue_events([
            DriverQueueEvent(1000000000, "dispatch", 42, "p", ring=0),
        ])

        reports = correlator.correlate()
        assert reports[0].diagnosis == "no_kernel_events_in_window"

    def test_multiple_nan_events(self):
        correlator = NaNCorrelator(window_ms=100.0)
        for step in range(5):
            correlator.add_nan_event(NaNDetection(
                timestamp_ns=0, step=step, rank=0, source="test",
            ))
        reports = correlator.correlate()
        assert len(reports) == 5

    def test_add_nan_events_from_log(self):
        log_content = (
            "INFO: Training step 41 complete\n"
            "[NaN POST-all_reduce] rank=0 step=42: nan=3\n"
            "[NaN DATADIST-embedding_lookup] rank=0 step=42: nan=1\n"
            "[NaN PRE-all_reduce] rank=0 step=43: nan=2\n"
            "INFO: Training step 43 complete\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            f.flush()

            correlator = NaNCorrelator()
            count = correlator.add_nan_events_from_log(f.name)
            assert count == 2
            assert len(correlator._nan_events) == 2
            assert correlator._nan_events[0].source == "POST-all_reduce"
            assert correlator._nan_events[1].source == "DATADIST-embedding_lookup"

        os.unlink(f.name)

    def test_add_nan_events_from_missing_log(self):
        correlator = NaNCorrelator()
        count = correlator.add_nan_events_from_log("/nonexistent/path.log")
        assert count == 0

    def test_export_reports(self):
        correlator = NaNCorrelator(window_ms=100.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=0, step=1, rank=0, source="test",
        ))
        reports = correlator.correlate()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            correlator.export_reports(reports, f.name)
            with open(f.name) as rf:
                data = json.load(rf)
            assert data["nan_count"] == 1
            assert data["window_ms"] == 100.0
            assert len(data["reports"]) == 1
            assert "diagnosis" in data["reports"][0]

        os.unlink(f.name)

    def test_confidence_levels(self):
        high = CorrelatedNaNReport(
            nan_event=NaNDetection(0, 1), window_ms=100.0,
            race_events_in_window=2, dma_overlaps_in_window=1,
        )
        assert NaNCorrelator._assess_confidence(high) == "high"

        medium = CorrelatedNaNReport(
            nan_event=NaNDetection(0, 1), window_ms=100.0,
            evictions_in_window=1, dispatch_gap_spike=True,
        )
        assert NaNCorrelator._assess_confidence(medium) == "medium"

        low = CorrelatedNaNReport(
            nan_event=NaNDetection(0, 1), window_ms=100.0,
        )
        assert NaNCorrelator._assess_confidence(low) == "low"
