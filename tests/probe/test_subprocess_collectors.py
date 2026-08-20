"""Collector wiring inside :class:`SubprocessWorkload` (no GPU, no rocprofv3).

The generic seam is what makes ``aorta probe -- <any command>`` profilable:
``setup()`` rewrites the opaque user argv, and ``run()`` merges the collectors'
parsed summaries into ``WorkloadResult.metrics``. These tests pin both,
plus the on-disk layout the artifacts and metrics actually land in -- which is
NOT inside the hand-written ``trial_<n>/`` directory, and is easy to get wrong
from reading the code alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.instrumentation import proton, rocprof
from aorta.run.collectors import (
    CONFIG_KEY_COLLECT,
    CONFIG_KEY_COLLECT_DIR,
    CONFIG_KEY_COLLECT_OPTIONS,
)
from aorta.workloads._subprocess import (
    CONFIG_KEY_LOG_PREFIX,
    CONFIG_KEY_PROBE_EXTRAS,
    CONFIG_KEY_SUBPROCESS_ARGV,
    SubprocessWorkload,
)


@pytest.fixture
def rocprofv3_on_path(tmp_path, monkeypatch):
    fake = tmp_path / "fakebin" / "rocprofv3"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('#!/bin/sh\nexec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:/usr/bin:/bin")
    monkeypatch.delenv(rocprof.ENV_ROCPROF_BIN, raising=False)
    return fake


def _make_workload(tmp_path: Path, argv: list[str], *, collect=(), options=None, collect_dir=None):
    """Build a workload with the log-prefix / collect-dir shape the runner sets.

    The dispatcher sets ``_aorta_log_prefix`` to
    ``<cell_dir>/<workload>/trial_d0_m0_t<N>`` and ``_aorta_collect_dir`` to the
    same stem, so the synthetic config mirrors that and the test exercises the
    real path decoding.
    """
    workload_subdir = tmp_path / "_subprocess"
    workload_subdir.mkdir(parents=True, exist_ok=True)
    prefix = workload_subdir / "trial_d0_m0_t0"
    config: dict = {
        CONFIG_KEY_SUBPROCESS_ARGV: argv,
        CONFIG_KEY_LOG_PREFIX: str(prefix),
        CONFIG_KEY_PROBE_EXTRAS: {
            "cell_name": "none-none",
            "env_passthrough_mode": "inherit",
            "timeout_per_trial": None,
            "cell_env_vars": {},
        },
    }
    if collect:
        config[CONFIG_KEY_COLLECT] = list(collect)
        config[CONFIG_KEY_COLLECT_DIR] = str(collect_dir if collect_dir is not None else prefix)
    if options:
        config[CONFIG_KEY_COLLECT_OPTIONS] = options
    return SubprocessWorkload(config)


# ---- setup(): argv rewrite ----------------------------------------------


def test_setup_leaves_argv_alone_without_a_collector(tmp_path):
    """A run without ``--collect`` must launch byte-for-byte what it did."""
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    assert list(wl._argv) == ["true"]


def test_setup_leaves_argv_alone_for_a_validated_only_collector(tmp_path):
    wl = _make_workload(tmp_path, ["true"], collect=["layer_numerics"])
    wl.setup()
    assert list(wl._argv) == ["true"]


def test_setup_wraps_argv_with_rocprof(tmp_path, rocprofv3_on_path):
    wl = _make_workload(tmp_path, ["/tmp/gemm", "512"], collect=["rocprof"])
    wl.setup()
    assert wl._argv is not None
    argv = list(wl._argv)
    assert Path(argv[0]).name == "rocprofv3"
    assert argv[argv.index("--") + 1 :] == ["/tmp/gemm", "512"]


def test_setup_points_rocprof_at_the_threaded_collect_dir(tmp_path, rocprofv3_on_path):
    wl = _make_workload(tmp_path, ["/tmp/gemm"], collect=["rocprof"])
    wl.setup()
    expected = tmp_path / "_subprocess" / "trial_d0_m0_t0" / rocprof.OUTPUT_SUBDIR
    assert list(wl._argv)[list(wl._argv).index("-d") + 1] == str(expected)
    assert expected.is_dir()


def test_setup_forwards_recipe_options(tmp_path, rocprofv3_on_path):
    wl = _make_workload(
        tmp_path,
        ["/tmp/gemm"],
        collect=["rocprof"],
        options={"rocprof": {"trace": "kernel,hip"}},
    )
    wl.setup()
    assert "--hip-trace" in list(wl._argv)


def test_setup_wraps_argv_with_proton(tmp_path):
    wl = _make_workload(tmp_path, ["python", "vecadd.py"], collect=["proton"])
    wl.setup()
    argv = list(wl._argv)
    assert argv[1:3] == ["-m", proton.PROTON_MODULE]
    assert argv[-1] == "vecadd.py"


def test_setup_surfaces_an_unattachable_collector_as_a_setup_failure(tmp_path, monkeypatch):
    """A requested measurement that cannot be taken is a clean setup failure,
    not a silently unprofiled run."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv(rocprof.ENV_ROCPROF_BIN, raising=False)
    wl = _make_workload(tmp_path, ["/tmp/gemm"], collect=["rocprof"])
    with pytest.raises(rocprof.RocprofUnavailableError):
        wl.setup()


def test_setup_rejects_a_proton_cli_wrap_of_a_non_python_command(tmp_path):
    wl = _make_workload(tmp_path, ["/tmp/gemm", "512"], collect=["proton"])
    with pytest.raises(proton.ProtonWrapError, match="mode: env"):
        wl.setup()


def test_setup_argv_validation_runs_before_the_collector_wrap(tmp_path, rocprofv3_on_path):
    """The wrap must not mask the plain 'you gave me no argv' error."""
    wl = _make_workload(tmp_path, [], collect=["rocprof"])
    with pytest.raises(RuntimeError, match="non-empty list"):
        wl.setup()


# ---- run(): metrics merge ----------------------------------------------


def _rocprof_artifacts(collect_dir: Path, total_ns: int = 539404, calls: int = 23) -> Path:
    out_dir = collect_dir / rocprof.OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n' f'"sgemm_tiled(float const*)",{calls},{total_ns}\n',
        encoding="utf-8",
    )
    return out_dir


def test_run_merges_collector_metrics(tmp_path):
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    _rocprof_artifacts(collect_dir)
    wl.setup()
    result = wl.run()
    assert result.metrics["rocprof_kernel_count"] == 23
    assert result.metrics["rocprof_gpu_time_ms"] == pytest.approx(0.539404)
    assert result.metrics["rocprof_artifact_dir"] == str(collect_dir / rocprof.OUTPUT_SUBDIR)


def test_run_metrics_unchanged_without_a_collector(tmp_path):
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    result = wl.run()
    assert set(result.metrics) == {
        "verdict",
        "exit_code",
        "result_json_path",
        "failure_detectors_fired",
        "error_detectors_fired",
        "warn_detectors_fired",
    }


def test_run_collector_cannot_shadow_the_platform_keys(tmp_path, monkeypatch):
    """The collector summary is merged UNDER the platform bookkeeping, so a
    collector emitting ``verdict`` / ``exit_code`` cannot rewrite the trial's
    outcome."""
    monkeypatch.setattr(
        rocprof,
        "parse_summary",
        lambda _out: {"verdict": "pass", "exit_code": 0, "rocprof_gpu_time_ms": 1.0},
    )
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    (collect_dir / rocprof.OUTPUT_SUBDIR).mkdir(parents=True)
    wl = _make_workload(tmp_path, ["false"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    assert result.metrics["verdict"] == "fail"
    assert result.metrics["exit_code"] != 0
    assert result.metrics["rocprof_gpu_time_ms"] == 1.0


def test_run_survives_a_collector_that_produced_nothing(tmp_path):
    """rocprofv3 writes no files at all for a command with no GPU work."""
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    assert result.passed is True
    assert "rocprof_kernel_count" not in result.metrics
    assert result.metrics["verdict"] == "pass"


def test_run_survives_a_malformed_capture(tmp_path):
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    out_dir = collect_dir / rocprof.OUTPUT_SUBDIR
    out_dir.mkdir(parents=True)
    (out_dir / "aorta_kernel_stats.csv").write_text("garbage\n", encoding="utf-8")
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    assert result.passed is True
    assert "rocprof_kernel_count" not in result.metrics


def test_run_still_reports_metrics_for_a_failing_trial(tmp_path):
    """A profiled crash is exactly the case an operator wants numbers for."""
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    _rocprof_artifacts(collect_dir)
    wl = _make_workload(tmp_path, ["false"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    assert result.passed is False
    assert result.metrics["rocprof_kernel_count"] == 23


# ---- Artifact / metrics location --------------------------------------


def test_collector_artifacts_are_a_sibling_of_the_hand_written_trial_dir(tmp_path):
    """Pin the real layout, which is not the obvious one.

    ``SubprocessWorkload`` hand-writes ``<cell>/trial_<n>/result.json`` from
    ``_aorta_log_prefix``, while the dispatcher threads ``_aorta_collect_dir``
    as ``<cell>/<workload>/trial_d<d>_m<m>_t<t>``. So the collector artifact
    directory is a SIBLING of the per-trial ``result.json`` tree, not inside
    it, and ``result.json`` carries no collector metrics -- those ride
    ``WorkloadResult.metrics`` into the dispatcher's trial JSON instead.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    _rocprof_artifacts(collect_dir)
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    wl.run()

    artifact_dir = collect_dir / rocprof.OUTPUT_SUBDIR
    hand_written = tmp_path / "trial_0"
    assert artifact_dir.is_dir()
    assert hand_written.is_dir()
    # Neither contains the other.
    assert hand_written not in artifact_dir.parents
    assert artifact_dir not in hand_written.parents
    # Both live under the same cell directory, so `aorta bundle` (which copies
    # everything under the run dir) picks up both.
    assert artifact_dir.is_relative_to(tmp_path)
    assert hand_written.is_relative_to(tmp_path)


def test_hand_written_result_json_carries_no_collector_metrics(tmp_path):
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    _rocprof_artifacts(collect_dir)
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert not [key for key in doc if key.startswith(("rocprof_", "proton_"))]
    # The metrics channel is where they live, and it points back at result.json.
    assert result.metrics["result_json_path"] == str(tmp_path / "trial_0" / "result.json")


def test_result_json_records_the_wrapped_argv(tmp_path, rocprofv3_on_path):
    """``result.json`` records the argv that actually ran, so an attached
    collector -- and any environment rewriting it did, such as Proton's
    HIP_VISIBLE_DEVICES translation -- is auditable from the artifact rather
    than only from a log line."""
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    _rocprof_artifacts(collect_dir)
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    recorded = doc.get("argv")
    assert recorded is not None
    assert Path(recorded[0]).name == "rocprofv3"
    assert recorded[-1] == "true"
