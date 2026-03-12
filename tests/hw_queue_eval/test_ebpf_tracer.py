"""
Tests for eBPF tracer modules (queue tracer, memory tracer, policy evaluator).

All subprocess and filesystem calls are mocked so tests run without real GPUs,
bpftrace, or root privileges.  Modules are loaded directly by file path to
avoid the torch-dependent aorta import chain.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Direct module loading (bypasses torch-dependent aorta.hw_queue_eval.__init__)
# ---------------------------------------------------------------------------

_CORE_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "src", "aorta", "hw_queue_eval", "core",
)


def _load_module(name: str, filename: str):
    filepath = os.path.join(_CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_ebpf_tracer = _load_module("ebpf_tracer", "ebpf_tracer.py")
_ebpf_memory_tracer = _load_module("ebpf_memory_tracer", "ebpf_memory_tracer.py")
_device_ebpf = _load_module("device_ebpf", "device_ebpf.py")

# Metrics module needs torch -- load compare_ebpf_vs_cuda separately
# since it only uses stdlib.  We'll define a local version for this test.


BPFQueueTracer = _ebpf_tracer.BPFQueueTracer
DriverQueueEvent = _ebpf_tracer.DriverQueueEvent
DriverQueueMetrics = _ebpf_tracer.DriverQueueMetrics
EBPFCapabilities = _ebpf_tracer.EBPFCapabilities
check_ebpf_capabilities = _ebpf_tracer.check_ebpf_capabilities

BPFMemoryTracer = _ebpf_memory_tracer.BPFMemoryTracer
MemoryTraceEvent = _ebpf_memory_tracer.MemoryTraceEvent
MemoryTraceMetrics = _ebpf_memory_tracer.MemoryTraceMetrics

DeviceEBPFConfig = _device_ebpf.DeviceEBPFConfig
DeviceEBPFMetrics = _device_ebpf.DeviceEBPFMetrics
DeviceEBPFProfiler = _device_ebpf.DeviceEBPFProfiler


# ---------------------------------------------------------------------------
# EBPFCapabilities / check_ebpf_capabilities
# ---------------------------------------------------------------------------

class TestEBPFCapabilities:

    def test_available_when_bpftrace_and_tracepoints(self):
        caps = EBPFCapabilities(
            bpftrace_path="/usr/bin/bpftrace",
            has_amdgpu_tracepoints=True,
        )
        assert caps.available is True

    def test_not_available_without_bpftrace(self):
        caps = EBPFCapabilities(has_amdgpu_tracepoints=True)
        assert caps.available is False

    def test_not_available_without_tracepoints(self):
        caps = EBPFCapabilities(bpftrace_path="/usr/bin/bpftrace")
        assert caps.available is False

    def test_to_dict(self):
        caps = EBPFCapabilities(kernel_version="6.8.0")
        d = caps.to_dict()
        assert d["kernel_version"] == "6.8.0"
        assert "available" in d

    @patch.object(_ebpf_tracer, "shutil")
    @patch.object(_ebpf_tracer, "subprocess")
    @patch.object(_ebpf_tracer, "os")
    def test_check_ebpf_capabilities(self, mock_os, mock_subprocess, mock_shutil):
        mock_shutil.which.return_value = "/usr/bin/bpftrace"
        mock_run_result = MagicMock(stdout="6.8.0-90-generic\n", returncode=0)
        mock_subprocess.run.return_value = mock_run_result
        mock_subprocess.SubprocessError = Exception
        mock_os.geteuid.return_value = 1000

        # Patch Path.is_dir to return False (no debugfs access)
        with patch.object(_ebpf_tracer.Path, "is_dir", return_value=False):
            caps = check_ebpf_capabilities()

        assert caps.bpftrace_path == "/usr/bin/bpftrace"


# ---------------------------------------------------------------------------
# DriverQueueEvent / DriverQueueMetrics
# ---------------------------------------------------------------------------

class TestDriverQueueMetrics:

    def test_empty_metrics(self):
        m = DriverQueueMetrics()
        assert m.avg_submit_to_dispatch_us == 0.0
        assert m.p99_submit_to_dispatch_us == 0.0
        assert m.rings_used == []

    def test_metrics_with_events(self):
        m = DriverQueueMetrics(
            total_submissions=10,
            total_dispatches=10,
            submission_to_dispatch_us=[1.0, 2.0, 3.0, 4.0, 5.0],
            per_ring_submissions={0: 5, 1: 5},
            per_ring_dispatches={0: 5, 1: 5},
        )
        assert m.avg_submit_to_dispatch_us == 3.0
        assert m.rings_used == [0, 1]

    def test_to_dict(self):
        m = DriverQueueMetrics(total_submissions=42)
        d = m.to_dict()
        assert d["total_submissions"] == 42
        assert "avg_submit_to_dispatch_us" in d

    def test_p99_single_value(self):
        m = DriverQueueMetrics(submission_to_dispatch_us=[5.0])
        assert m.p99_submit_to_dispatch_us == 5.0


# ---------------------------------------------------------------------------
# BPFQueueTracer
# ---------------------------------------------------------------------------

class TestBPFQueueTracer:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(
                target_pid=1234, output_dir=Path(tmpdir)
            )
            assert tracer._target_pid == 1234
            assert tracer.is_running is False

    def test_generate_script_with_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(target_pid=42, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid == 42" in content
            assert "SUBMIT" in content
            assert "DISPATCH" in content

    def test_generate_script_all_pids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(target_pid=None, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid ==" not in content
            assert "SUBMIT" in content

    def test_parse_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tracer = BPFQueueTracer(output_dir=tmpdir)
            log_file = tmpdir / "queue_trace.log"
            log_file.write_text(
                "SUBMIT|1000000000|42|python|0|5\n"
                "DISPATCH|1000100000|42|python|0|1\n"
                "SUBMIT|1000200000|42|python|1|3\n"
                "some garbage line\n"
                "DISPATCH|1000300000|42|python|1|2\n"
            )
            tracer._output_path = log_file
            events = tracer._parse_output()
            assert len(events) == 4
            assert events[0].event_type == "submit"
            assert events[1].event_type == "dispatch"
            assert events[0].ring == 0
            assert events[2].ring == 1

    def test_compute_metrics(self):
        events = [
            DriverQueueEvent(1000000000, "submit", 42, "python", ring=0, fence=1),
            DriverQueueEvent(1000100000, "dispatch", 42, "python", ring=0, fence=1),
            DriverQueueEvent(1000200000, "submit", 42, "python", ring=1, fence=2),
            DriverQueueEvent(1000300000, "dispatch", 42, "python", ring=1, fence=2),
        ]
        metrics = BPFQueueTracer._compute_metrics(events, 1_000_000_000)
        assert metrics.total_submissions == 2
        assert metrics.total_dispatches == 2
        assert metrics.per_ring_submissions == {0: 1, 1: 1}
        assert len(metrics.submission_to_dispatch_us) == 2
        assert metrics.submission_to_dispatch_us[0] == pytest.approx(100.0)
        assert metrics.trace_duration_ms == pytest.approx(1000.0)

    def test_compute_metrics_empty(self):
        metrics = BPFQueueTracer._compute_metrics([], 500_000_000)
        assert metrics.total_submissions == 0
        assert metrics.trace_duration_ms == pytest.approx(500.0)

    @patch.object(_ebpf_tracer, "shutil")
    def test_start_raises_without_bpftrace(self, mock_shutil):
        mock_shutil.which.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace is not installed"):
                tracer.start()

    def test_stop_without_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(output_dir=Path(tmpdir))
            metrics = tracer.stop()
            assert isinstance(metrics, DriverQueueMetrics)
            assert metrics.total_submissions == 0

    def test_event_timestamp_ms(self):
        ev = DriverQueueEvent(5_000_000, "submit", 1, "test")
        assert ev.timestamp_ms == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# MemoryTraceMetrics / BPFMemoryTracer
# ---------------------------------------------------------------------------

class TestMemoryTraceMetrics:

    def test_empty(self):
        m = MemoryTraceMetrics()
        d = m.to_dict()
        assert d["total_faults"] == 0
        assert d["fault_rate_per_sec"] == 0.0

    def test_to_dict(self):
        m = MemoryTraceMetrics(total_evictions=3, migration_bytes=4096)
        d = m.to_dict()
        assert d["total_evictions"] == 3
        assert d["migration_bytes"] == 4096


class TestBPFMemoryTracer:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(target_pid=1234, output_dir=Path(tmpdir))
            assert tracer._target_pid == 1234

    def test_generate_script_with_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(target_pid=99, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid == 99" in content
            assert "BO_MAP" in content

    def test_generate_script_all_pids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(target_pid=None, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid ==" not in content

    def test_parse_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tracer = BPFMemoryTracer(output_dir=tmpdir)
            log_file = tmpdir / "memory_trace.log"
            log_file.write_text(
                "BO_MAP|1000000000|42|python|65536\n"
                "BO_UNMAP|1000100000|42|python|65536\n"
                "EVICT|1000200000|42|python|0\n"
                "RESTORE|1000300000|42|python|0\n"
            )
            tracer._output_path = log_file
            events = tracer._parse_output()
            assert len(events) == 4
            assert events[0].event_type == "bo_map"
            assert events[0].size_bytes == 65536
            assert events[2].event_type == "evict"
            assert events[3].event_type == "restore"

    def test_compute_metrics(self):
        events = [
            MemoryTraceEvent(1000000000, "bo_map", 42, "python", size_bytes=4096),
            MemoryTraceEvent(1000100000, "evict", 42, "python"),
            MemoryTraceEvent(1000200000, "restore", 42, "python"),
        ]
        metrics = BPFMemoryTracer._compute_metrics(events, 1_000_000_000)
        assert metrics.total_bo_maps == 1
        assert metrics.total_evictions == 1
        assert metrics.total_restores == 1
        assert metrics.migration_bytes == 4096
        assert metrics.avg_fault_latency_us == pytest.approx(100.0)

    def test_compute_metrics_empty(self):
        metrics = BPFMemoryTracer._compute_metrics([], 500_000_000)
        assert metrics.total_faults == 0
        assert metrics.trace_duration_ms == pytest.approx(500.0)

    @patch.object(_ebpf_memory_tracer, "shutil")
    def test_start_raises_without_bpftrace(self, mock_shutil):
        mock_shutil.which.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace is not installed"):
                tracer.start()

    def test_stop_without_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(output_dir=Path(tmpdir))
            metrics = tracer.stop()
            assert isinstance(metrics, MemoryTraceMetrics)
            assert metrics.total_faults == 0

    def test_event_timestamp_ms(self):
        ev = MemoryTraceEvent(5_000_000, "bo_map", 1, "test", size_bytes=1024)
        assert ev.timestamp_ms == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# DeviceEBPFProfiler (stub)
# ---------------------------------------------------------------------------

class TestDeviceEBPFProfiler:

    def test_not_available(self):
        assert DeviceEBPFProfiler.is_available() is False

    def test_start_raises(self):
        profiler = DeviceEBPFProfiler()
        with pytest.raises(NotImplementedError, match="not yet available"):
            profiler.start()

    def test_stop_raises(self):
        profiler = DeviceEBPFProfiler()
        with pytest.raises(NotImplementedError):
            profiler.stop()

    def test_config_to_dict(self):
        config = DeviceEBPFConfig(enabled=True, sampling_rate=4)
        d = config.to_dict()
        assert d["enabled"] is True
        assert d["sampling_rate"] == 4

    def test_metrics_to_dict(self):
        metrics = DeviceEBPFMetrics(warp_occupancy=0.75)
        d = metrics.to_dict()
        assert d["warp_occupancy"] == 0.75


# ---------------------------------------------------------------------------
# PolicyConfig (unit tests, no GPU needed)
# ---------------------------------------------------------------------------

class TestPolicyConfig:

    def test_builtin_policies_exist(self):
        # policy_evaluator doesn't depend on torch, load directly
        _policy_eval = _load_module("policy_evaluator", "policy_evaluator.py")
        assert "baseline" in _policy_eval.BUILTIN_POLICIES
        assert "priority_lc" in _policy_eval.BUILTIN_POLICIES
        assert "priority_be" in _policy_eval.BUILTIN_POLICIES
        assert "multi_tenant_fair" in _policy_eval.BUILTIN_POLICIES

    def test_policy_to_dict(self):
        _policy_eval = _load_module("policy_evaluator", "policy_evaluator.py")
        p = _policy_eval.PolicyConfig(name="test", policy_type="scheduling", gpu_clock_level=5)
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["gpu_clock_level"] == 5

    def test_policy_comparison_summary(self):
        _policy_eval = _load_module("policy_evaluator", "policy_evaluator.py")

        mock_result_a = MagicMock()
        mock_result_a.throughput = 100.0
        mock_result_a.latency_ms = {"p50": 1.0, "p95": 2.0, "p99": 3.0}

        mock_result_b = MagicMock()
        mock_result_b.throughput = 150.0
        mock_result_b.latency_ms = {"p50": 0.8, "p95": 1.5, "p99": 2.0}

        comp = _policy_eval.PolicyComparison(workload_name="test", stream_count=4)
        comp.add(_policy_eval.PolicyResult(
            policy=_policy_eval.PolicyConfig(name="baseline"),
            harness_result=mock_result_a,
        ))
        comp.add(_policy_eval.PolicyResult(
            policy=_policy_eval.PolicyConfig(name="priority_lc"),
            harness_result=mock_result_b,
        ))

        assert comp.best_throughput().policy.name == "priority_lc"
        assert comp.best_latency().policy.name == "priority_lc"

        table = comp.summary_table()
        assert "baseline" in table
        assert "priority_lc" in table

    def test_policy_comparison_save(self):
        _policy_eval = _load_module("policy_evaluator", "policy_evaluator.py")

        mock_result = MagicMock()
        mock_result.throughput = 100.0
        mock_result.latency_ms = {"p50": 1.0, "p95": 2.0, "p99": 3.0}
        mock_result.to_dict.return_value = {"throughput": 100.0}

        comp = _policy_eval.PolicyComparison(
            workload_name="test", stream_count=4, timestamp="2026-03-10"
        )
        comp.add(_policy_eval.PolicyResult(
            policy=_policy_eval.PolicyConfig(name="baseline"),
            harness_result=mock_result,
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            comp.save(f.name)
            import json
            with open(f.name) as rf:
                data = json.load(rf)
            assert data["workload"] == "test"
            assert len(data["results"]) == 1

    def test_empty_comparison(self):
        _policy_eval = _load_module("policy_evaluator", "policy_evaluator.py")
        comp = _policy_eval.PolicyComparison(workload_name="test", stream_count=4)
        assert comp.best_throughput() is None
        assert comp.best_latency() is None
