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


def _load_compare_ebpf_vs_cuda():
    """Load ``compare_ebpf_vs_cuda`` without dragging in torch.

    ``aorta.hw_queue_eval.core.metrics`` imports torch at module top
    level, but ``compare_ebpf_vs_cuda`` itself only depends on stdlib.
    We strip the file down to that function and exec it in an isolated
    namespace so the comparison helper can be unit tested without a GPU.
    """
    metrics_path = os.path.join(_CORE_DIR, "metrics.py")
    src = open(metrics_path).read()
    marker = "def compare_ebpf_vs_cuda("
    idx = src.index(marker)
    end_marker = "\ndef "
    end_idx = src.find(end_marker, idx + len(marker))
    snippet = src[idx:] if end_idx == -1 else src[idx:end_idx]
    snippet = "from typing import Any, Dict\n" + snippet
    ns: dict = {}
    exec(compile(snippet, metrics_path, "exec"), ns)  # noqa: S102
    return ns["compare_ebpf_vs_cuda"]


compare_ebpf_vs_cuda = _load_compare_ebpf_vs_cuda()


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

    def test_available_when_bpftrace_and_tracepoints_and_privilege(self):
        caps = EBPFCapabilities(
            bpftrace_path="/usr/bin/bpftrace",
            has_amdgpu_tracepoints=True,
            has_root_or_cap=True,
        )
        assert caps.available is True

    def test_not_available_without_bpftrace(self):
        caps = EBPFCapabilities(
            has_amdgpu_tracepoints=True, has_root_or_cap=True,
        )
        assert caps.available is False

    def test_not_available_without_tracepoints(self):
        caps = EBPFCapabilities(
            bpftrace_path="/usr/bin/bpftrace", has_root_or_cap=True,
        )
        assert caps.available is False

    def test_not_available_without_privilege(self):
        caps = EBPFCapabilities(
            bpftrace_path="/usr/bin/bpftrace",
            has_amdgpu_tracepoints=True,
            has_root_or_cap=False,
        )
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
            with patch.object(_ebpf_tracer, "_check_ebpf_privilege", return_value=False):
                caps = check_ebpf_capabilities()

        assert caps.bpftrace_path == "/usr/bin/bpftrace"
        assert caps.has_root_or_cap is False
        # available must be False without privileges, even if bpftrace exists
        assert caps.available is False

    def test_available_requires_privilege(self):
        caps = EBPFCapabilities(
            bpftrace_path="/usr/bin/bpftrace",
            has_amdgpu_tracepoints=True,
            has_root_or_cap=False,
        )
        assert caps.available is False
        caps.has_root_or_cap = True
        assert caps.available is True

    def test_check_ebpf_privilege_root(self):
        with patch.object(_ebpf_tracer.os, "geteuid", return_value=0):
            assert _ebpf_tracer._check_ebpf_privilege() is True

    def test_check_ebpf_privilege_unprivileged(self):
        # Non-root with no eBPF capabilities in CapEff
        fake_status = "CapEff:\t0000000000000000\n"
        with patch.object(_ebpf_tracer.os, "geteuid", return_value=1000):
            with patch("builtins.open", lambda *a, **k: __import__("io").StringIO(fake_status)):
                assert _ebpf_tracer._check_ebpf_privilege() is False

    def test_check_ebpf_privilege_with_cap_bpf(self):
        # CAP_BPF (bit 39) set -> privileged
        cap_eff = 1 << 39
        fake_status = f"CapEff:\t{cap_eff:016x}\n"
        with patch.object(_ebpf_tracer.os, "geteuid", return_value=1000):
            with patch("builtins.open", lambda *a, **k: __import__("io").StringIO(fake_status)):
                assert _ebpf_tracer._check_ebpf_privilege() is True


# ---------------------------------------------------------------------------
# DriverQueueEvent / DriverQueueMetrics
# ---------------------------------------------------------------------------

class TestDriverQueueMetrics:

    def test_empty_metrics(self):
        m = DriverQueueMetrics()
        assert m.avg_submit_to_dispatch_us == 0.0
        assert m.p99_submit_to_dispatch_us == 0.0
        assert m.avg_inter_dispatch_gap_us == 0.0
        assert m.p99_inter_dispatch_gap_us == 0.0
        assert m.dispatch_rate_per_sec == 0.0
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

    def test_dispatch_only_metrics(self):
        m = DriverQueueMetrics(
            total_dispatches=100,
            inter_dispatch_gap_us=[10.0, 20.0, 30.0],
            per_ring_dispatches={0: 50, 1: 50},
            trace_duration_ms=1000.0,
        )
        assert m.avg_inter_dispatch_gap_us == 20.0
        assert m.dispatch_rate_per_sec == 100.0

    def test_to_dict(self):
        m = DriverQueueMetrics(total_submissions=42, trace_duration_ms=1000.0)
        d = m.to_dict()
        assert d["total_submissions"] == 42
        assert "avg_submit_to_dispatch_us" in d
        assert "avg_inter_dispatch_gap_us" in d
        assert "dispatch_rate_per_sec" in d

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

    @patch.object(_ebpf_tracer, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script_with_pid(self, _mock_probe):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(target_pid=42, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid == 42" in content
            assert "SUBMIT" in content
            assert "DISPATCH" in content
            # amdgpu_sched_run_job must NOT be PID-filtered (runs on kthreads).
            assert "amdgpu_sched_run_job\n/pid ==" not in content
            # With no format probing, fields should be safe defaults (0)
            assert "args->ring" not in content

    @patch.object(_ebpf_tracer, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script_all_pids(self, _mock_probe):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFQueueTracer(target_pid=None, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid ==" not in content
            assert "SUBMIT" in content

    def test_generate_script_with_probed_fields(self):
        """When format probing succeeds, the script uses real field names."""
        sched_fields = {"ring", "seqno", "sched_job_id", "num_ibs"}
        cs_fields = {"ring", "num_ibs", "seqno"}

        def mock_probe(category, name):
            if name == "amdgpu_sched_run_job":
                return sched_fields
            if name == "amdgpu_cs_ioctl":
                return cs_fields
            return None

        with patch.object(_ebpf_tracer, "_probe_tracepoint_fields", side_effect=mock_probe):
            with tempfile.TemporaryDirectory() as tmpdir:
                tracer = BPFQueueTracer(target_pid=42, output_dir=Path(tmpdir))
                script_path = tracer._generate_script()
                content = script_path.read_text()
                assert "args->ring" in content
                assert "args->seqno" in content
                assert "args->num_ibs" in content

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

    def test_compute_metrics_dispatch_only(self):
        """ROCm/KFD path: only dispatch events, no submit events.

        Inter-dispatch gaps are computed globally across all rings (the
        intended interpretation is "how often does *any* ring fire?"),
        not per-ring.  With four dispatches at 0/50/80/150 us we get
        three gaps: 50us, 30us, 70us -> avg 50us.
        """
        events = [
            DriverQueueEvent(1000000000, "dispatch", 99, "amdgpu_sched", ring=0, fence=1),
            DriverQueueEvent(1000050000, "dispatch", 99, "amdgpu_sched", ring=0, fence=2),
            DriverQueueEvent(1000080000, "dispatch", 99, "amdgpu_sched", ring=1, fence=1),
            DriverQueueEvent(1000150000, "dispatch", 99, "amdgpu_sched", ring=0, fence=3),
        ]
        metrics = BPFQueueTracer._compute_metrics(events, 1_000_000_000)
        assert metrics.total_submissions == 0
        assert metrics.total_dispatches == 4
        assert metrics.per_ring_dispatches == {0: 3, 1: 1}
        assert len(metrics.submission_to_dispatch_us) == 0
        assert metrics.inter_dispatch_gap_us == pytest.approx([50.0, 30.0, 70.0])
        assert metrics.avg_inter_dispatch_gap_us == pytest.approx(50.0)
        assert metrics.dispatch_rate_per_sec == pytest.approx(4.0)

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

    @patch.object(_ebpf_tracer, "_probe_tracepoint_fields", return_value=None)
    @patch.object(_ebpf_tracer, "time")
    @patch.object(_ebpf_tracer, "subprocess")
    @patch.object(_ebpf_tracer, "shutil")
    def test_start_raises_on_immediate_exit(
        self, mock_shutil, mock_subprocess, mock_time, _mock_probe
    ):
        """Health check detects bpftrace crash (e.g. bad field names)."""
        import io
        mock_shutil.which.return_value = "/usr/bin/bpftrace"
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited with error
        mock_proc.stderr = io.StringIO(
            "ERROR: tracepoint:amdgpu:amdgpu_sched_run_job: no field named 'ring'\n"
        )
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.SubprocessError = Exception

        mock_caps = MagicMock()
        mock_caps.bpftrace_path = "/usr/bin/bpftrace"
        with patch.object(_ebpf_tracer, "check_ebpf_capabilities", return_value=mock_caps):
            with tempfile.TemporaryDirectory() as tmpdir:
                tracer = BPFQueueTracer(output_dir=Path(tmpdir))
                with pytest.raises(RuntimeError, match="bpftrace exited immediately"):
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


class TestGroupIrqCompletions:
    """Regression tests for ``BPFQueueTracer._group_irq_completions``.

    The previous implementation compared each IRQ to the *group head*'s
    timestamp, which would split a long but evenly spaced burst into
    multiple groups even when consecutive gaps were always small.  The
    fixed implementation compares to the *previous* IRQ instead.
    """

    @staticmethod
    def _ev(ts_ns: int) -> "DriverQueueEvent":
        return DriverQueueEvent(ts_ns, "irq", 1, "test")

    def test_close_irqs_collapse_to_one_group(self):
        # Three IRQs within 100us each
        irqs = [self._ev(0), self._ev(100_000), self._ev(200_000)]
        groups = BPFQueueTracer._group_irq_completions(irqs, window_us=500.0)
        assert len(groups) == 1
        assert groups[0].timestamp_ns == 0  # earliest is canonical

    def test_distant_irqs_form_separate_groups(self):
        # Each IRQ separated by 1ms (>> 500us window)
        irqs = [self._ev(i * 1_000_000) for i in range(4)]
        groups = BPFQueueTracer._group_irq_completions(irqs, window_us=500.0)
        assert len(groups) == 4

    def test_long_evenly_spaced_burst_stays_one_group(self):
        """Regression for the group-head bug.

        20 IRQs spaced 100us apart span 1.9ms (>> 500us) but each
        consecutive gap is only 100us.  The fix tracks the last-seen IRQ
        rather than the group head, so all 20 should collapse into a
        single completion event.
        """
        irqs = [self._ev(i * 100_000) for i in range(20)]
        groups = BPFQueueTracer._group_irq_completions(irqs, window_us=500.0)
        assert len(groups) == 1
        assert groups[0].timestamp_ns == 0

    def test_burst_then_gap_then_burst(self):
        irqs = [
            self._ev(0), self._ev(100_000), self._ev(200_000),  # burst 1
            self._ev(2_000_000),                                  # gap > window
            self._ev(2_100_000), self._ev(2_200_000),             # burst 2
        ]
        groups = BPFQueueTracer._group_irq_completions(irqs, window_us=500.0)
        assert len(groups) == 2

    def test_empty_input(self):
        assert BPFQueueTracer._group_irq_completions([]) == []


class TestCompareEbpfVsCuda:
    """Unit tests for ``compare_ebpf_vs_cuda`` (covers #10, #30).

    Exercises both the submit-path (DRM scheduler) and the
    dispatch-gap fallback (ROCm/KFD path) plus the accuracy clamp.
    """

    def test_submit_path_used_when_submissions_present(self):
        ebpf = {
            "total_submissions": 100,
            "total_dispatches": 100,
            "avg_submit_to_dispatch_us": 12.0,
            "avg_inter_dispatch_gap_us": 999.0,
            "rings_used": [0, 1],
            "dispatch_rate_per_sec": 100.0,
        }
        cuda = {
            "inter_stream_gap_ms": 0.020,
            "estimated_switch_overhead_ms": 0.015,
        }
        out = compare_ebpf_vs_cuda(ebpf, cuda)
        # 12us == 0.012ms should drive the comparison, NOT 999us
        assert out["ebpf_avg_submit_to_dispatch_ms"] == pytest.approx(0.012)
        assert "ebpf_avg_dispatch_gap_ms" not in out
        # |0.012 - 0.015| = 0.003; accuracy = (1 - 0.003/0.015)*100 = 80%
        assert out["accuracy_pct"] == pytest.approx(80.0)
        assert out["delta_ms"] == pytest.approx(0.003)
        assert out["ebpf_total_submissions"] == 100

    def test_dispatch_gap_fallback_when_no_submissions(self):
        ebpf = {
            "total_submissions": 0,
            "total_dispatches": 200,
            "avg_submit_to_dispatch_us": 0.0,
            "avg_inter_dispatch_gap_us": 25.0,
            "rings_used": [0],
            "dispatch_rate_per_sec": 200.0,
        }
        cuda = {
            "inter_stream_gap_ms": 0.030,
            "estimated_switch_overhead_ms": 0.020,
        }
        out = compare_ebpf_vs_cuda(ebpf, cuda)
        # ROCm/KFD path: dispatch gap (0.025ms) is the comparable metric
        assert out["ebpf_avg_dispatch_gap_ms"] == pytest.approx(0.025)
        assert "ebpf_avg_submit_to_dispatch_ms" not in out
        # |0.025 - 0.020| = 0.005; accuracy = (1 - 0.005/0.020)*100 = 75%
        assert out["accuracy_pct"] == pytest.approx(75.0)

    def test_accuracy_clamped_to_zero_for_huge_delta(self):
        # eBPF reports much larger value than CUDA -> accuracy could go
        # negative without the max(0.0, ...) clamp.
        ebpf = {"total_submissions": 1, "avg_submit_to_dispatch_us": 1000.0}
        cuda = {"inter_stream_gap_ms": 0.1, "estimated_switch_overhead_ms": 0.1}
        out = compare_ebpf_vs_cuda(ebpf, cuda)
        # 1000us == 1.0ms vs 0.1ms -> 9x off, raw accuracy = -800%, clamped 0
        assert out["accuracy_pct"] == 0.0
        assert out["delta_ms"] == pytest.approx(0.9)

    def test_zero_cuda_overhead_avoids_divzero(self):
        ebpf = {"total_submissions": 5, "avg_submit_to_dispatch_us": 10.0}
        cuda = {"inter_stream_gap_ms": 0.0, "estimated_switch_overhead_ms": 0.0}
        out = compare_ebpf_vs_cuda(ebpf, cuda)
        # No reference -> accuracy is 0.0 and delta is 0.0
        assert out["accuracy_pct"] == 0.0
        assert out["delta_ms"] == 0.0
        assert out["cuda_estimated_switch_overhead_ms"] == 0.0

    def test_missing_keys_defaults_to_zero(self):
        out = compare_ebpf_vs_cuda({}, {})
        # No keys at all -> all zeros, dispatch-gap path taken
        assert out["ebpf_avg_dispatch_gap_ms"] == 0.0
        assert out["accuracy_pct"] == 0.0
        assert out["ebpf_total_submissions"] == 0
        assert out["ebpf_total_dispatches"] == 0
        assert out["ebpf_rings_used"] == []

    def test_partial_match_within_tolerance(self):
        # eBPF and CUDA agree within ~5%
        ebpf = {"total_submissions": 50, "avg_submit_to_dispatch_us": 19.0}
        cuda = {"inter_stream_gap_ms": 0.025, "estimated_switch_overhead_ms": 0.020}
        out = compare_ebpf_vs_cuda(ebpf, cuda)
        # |0.019 - 0.020| / 0.020 = 5% -> accuracy 95%
        assert out["accuracy_pct"] == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# MemoryTraceMetrics / BPFMemoryTracer
# ---------------------------------------------------------------------------

class TestMemoryTraceMetrics:

    def test_empty(self):
        m = MemoryTraceMetrics()
        d = m.to_dict()
        assert d["total_faults"] == 0
        assert d["total_bo_moves"] == 0
        assert d["fault_rate_per_sec"] == 0.0
        assert d["bo_move_rate_per_sec"] == 0.0

    def test_to_dict(self):
        m = MemoryTraceMetrics(total_evictions=3, migration_bytes=4096, total_bo_moves=5)
        d = m.to_dict()
        assert d["total_evictions"] == 3
        assert d["migration_bytes"] == 4096
        assert d["total_bo_moves"] == 5


class TestBPFMemoryTracer:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(target_pid=1234, output_dir=Path(tmpdir))
            assert tracer._target_pid == 1234

    @patch.object(_ebpf_memory_tracer, "_check_tracepoint_exists", return_value=True)
    @patch.object(_ebpf_memory_tracer, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script_with_pid(self, _mock_probe, _mock_exists):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(target_pid=99, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            # PID filter must be applied to ALL probes (BO + KFD evict/restore)
            assert "pid == 99" in content
            assert content.count("/pid == 99/") >= 5
            assert "BO_MOVE" in content
            assert "BO_MAP" in content
            assert "EVICT" in content
            assert "RESTORE" in content
            # The eviction/restore worker tracepoints must also be PID-scoped
            assert (
                "tracepoint:amdkfd:kfd_evict_process_worker_start /pid == 99/"
                in content
            )
            assert (
                "tracepoint:amdkfd:kfd_restore_process_worker_start /pid == 99/"
                in content
            )
            # Without format probing, bo_size field should not be accessed
            assert "args->bo_size" not in content

    @patch.object(_ebpf_memory_tracer, "_check_tracepoint_exists", return_value=True)
    @patch.object(_ebpf_memory_tracer, "_probe_tracepoint_fields", return_value=None)
    def test_generate_script_all_pids(self, _mock_probe, _mock_exists):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(target_pid=None, output_dir=Path(tmpdir))
            script_path = tracer._generate_script()
            content = script_path.read_text()
            assert "pid ==" not in content
            assert "BO_MOVE" in content

    @patch.object(_ebpf_memory_tracer, "_check_tracepoint_exists", return_value=True)
    def test_generate_script_with_probed_fields(self, _mock_exists):
        """When format probing finds bo_size, the script uses it."""
        def mock_probe(category, name):
            if name == "amdgpu_bo_move":
                return {"bo", "bo_size", "new_placement", "old_placement"}
            return None

        with patch.object(_ebpf_memory_tracer, "_probe_tracepoint_fields", side_effect=mock_probe):
            with tempfile.TemporaryDirectory() as tmpdir:
                tracer = BPFMemoryTracer(output_dir=Path(tmpdir))
                script_path = tracer._generate_script()
                content = script_path.read_text()
                assert "args->bo_size" in content

    def test_parse_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tracer = BPFMemoryTracer(output_dir=tmpdir)
            log_file = tmpdir / "memory_trace.log"
            log_file.write_text(
                "BO_MOVE|1000000000|42|amdgpu_sched|65536\n"
                "BO_MAP|1000050000|42|python|0\n"
                "BO_UNMAP|1000100000|42|python|0\n"
                "EVICT|1000200000|42|python|0\n"
                "RESTORE|1000300000|42|python|0\n"
            )
            tracer._output_path = log_file
            events = tracer._parse_output()
            assert len(events) == 5
            assert events[0].event_type == "bo_move"
            assert events[0].size_bytes == 65536
            assert events[1].event_type == "bo_map"
            assert events[3].event_type == "evict"
            assert events[4].event_type == "restore"

    def test_compute_metrics(self):
        events = [
            MemoryTraceEvent(1000000000, "bo_move", 42, "amdgpu_sched", size_bytes=4096),
            MemoryTraceEvent(1000050000, "bo_map", 42, "python"),
            MemoryTraceEvent(1000100000, "evict", 42, "python"),
            MemoryTraceEvent(1000200000, "restore", 42, "python"),
        ]
        metrics = BPFMemoryTracer._compute_metrics(events, 1_000_000_000)
        assert metrics.total_bo_moves == 1
        assert metrics.total_bo_maps == 1
        assert metrics.total_evictions == 1
        assert metrics.total_restores == 1
        assert metrics.total_eviction_restore_pairs == 1
        assert metrics.migration_bytes == 4096
        assert metrics.avg_eviction_restore_latency_us == pytest.approx(100.0)
        # legacy alias still works
        assert metrics.avg_fault_latency_us == pytest.approx(100.0)
        assert metrics.bo_move_rate_per_sec == pytest.approx(1.0)

    def test_legacy_alias_setters(self):
        m = MemoryTraceMetrics()
        m.total_faults = 7
        assert m.total_eviction_restore_pairs == 7
        m.fault_rate_per_sec = 1.5
        assert m.eviction_restore_rate_per_sec == pytest.approx(1.5)
        m.avg_fault_latency_us = 42.0
        assert m.avg_eviction_restore_latency_us == pytest.approx(42.0)
        d = m.to_dict()
        assert d["total_faults"] == 7
        assert d["total_eviction_restore_pairs"] == 7
        assert d["fault_rate_per_sec"] == pytest.approx(1.5)

    def test_compute_metrics_empty(self):
        metrics = BPFMemoryTracer._compute_metrics([], 500_000_000)
        assert metrics.total_eviction_restore_pairs == 0
        assert metrics.total_faults == 0
        assert metrics.trace_duration_ms == pytest.approx(500.0)

    @patch.object(_ebpf_memory_tracer, "shutil")
    def test_start_raises_without_bpftrace(self, mock_shutil):
        mock_shutil.which.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace is not installed"):
                tracer.start()

    @patch.object(_ebpf_memory_tracer, "_build_memory_trace_script", return_value="#!/usr/bin/env bpftrace\n")
    @patch.object(_ebpf_memory_tracer, "time")
    @patch.object(_ebpf_memory_tracer, "subprocess")
    @patch.object(_ebpf_memory_tracer, "shutil")
    def test_start_raises_on_immediate_exit(
        self, mock_shutil, mock_subprocess, mock_time, _mock_build
    ):
        """Health check detects bpftrace crash."""
        import io
        mock_shutil.which.return_value = "/usr/bin/bpftrace"
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        # The stderr drain thread iterates over readline() until empty;
        # supply a real iterator that emits one error line then EOF.
        mock_proc.stderr = io.StringIO("ERROR: tracepoint not found\n")
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.SubprocessError = Exception

        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BPFMemoryTracer(output_dir=Path(tmpdir))
            with pytest.raises(RuntimeError, match="bpftrace.*exited immediately"):
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
# Tracepoint format probing
# ---------------------------------------------------------------------------

class TestProbeTracepointFields:
    """Tests for _probe_tracepoint_fields (shared by both tracers)."""

    def test_returns_none_when_file_missing(self):
        probe = _ebpf_tracer._probe_tracepoint_fields
        result = probe("nonexistent_category", "nonexistent_tp")
        assert result is None

    def test_parses_format_file(self):
        probe = _ebpf_tracer._probe_tracepoint_fields
        format_content = (
            "name: amdgpu_sched_run_job\n"
            "ID: 1234\n"
            "format:\n"
            "\tfield:unsigned short common_type;\toffset:0;\tsize:2;\n"
            "\tfield:uint64_t sched_job_id;\toffset:8;\tsize:8;\n"
            "\tfield:unsigned int context;\toffset:16;\tsize:4;\n"
            "\tfield:unsigned int seqno;\toffset:20;\tsize:4;\n"
            "\tfield:char * ring_name;\toffset:24;\tsize:8;\n"
        )
        with patch.object(_ebpf_tracer.Path, "read_text", return_value=format_content):
            with patch.object(_ebpf_tracer.Path, "__truediv__", return_value=_ebpf_tracer.Path()):
                result = probe("amdgpu", "amdgpu_sched_run_job")
                assert result is not None
                assert "seqno" in result
                assert "sched_job_id" in result
                assert "ring_name" in result
                assert "ring" not in result  # 'ring' is not a field name here

    def test_returns_none_on_permission_error(self):
        probe = _ebpf_tracer._probe_tracepoint_fields
        with patch.object(_ebpf_tracer.Path, "read_text", side_effect=PermissionError):
            result = probe("amdgpu", "amdgpu_sched_run_job")
            assert result is None


class TestCheckTracepointExists:
    """Tests for _check_tracepoint_exists in memory tracer."""

    def test_returns_true_when_category_missing(self):
        """When parent category doesn't exist, assume tp exists."""
        check = _ebpf_memory_tracer._check_tracepoint_exists
        with patch.object(_ebpf_memory_tracer.Path, "is_dir", return_value=False):
            assert check("nonexistent", "some_tp") is True

    def test_returns_true_on_permission_error(self):
        check = _ebpf_memory_tracer._check_tracepoint_exists
        with patch.object(_ebpf_memory_tracer.Path, "is_dir", side_effect=PermissionError):
            assert check("amdgpu", "amdgpu_bo_move") is True


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
