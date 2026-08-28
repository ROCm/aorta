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
import shutil
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

#: Stand-in for ``rocprofv3``: drop the profiler's own flags up to the ``--``
#: separator, then exec the profiled command so the trial observes the
#: payload's exit code. Every test in this module needs it -- a host that
#: happens to have ROCm installed must not change what these tests assert.
_FAKE_ROCPROFV3 = """#!/bin/sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        break
    fi
    shift
done
[ "$#" -eq 0 ] && exit 0
exec "$@"
"""


@pytest.fixture
def rocprofv3_on_path(tmp_path, monkeypatch):
    """Make ``rocprofv3`` resolution hermetic.

    ``resolve_binary()`` raises when rocprofv3 is absent, so without this the
    tests below pass only on a ROCm host and fail everywhere else (CI). The
    stub is a real executable rather than a patched ``resolve_binary`` because
    these tests run the wrapped argv end-to-end through ``run()``.
    """
    fake = tmp_path / "fakebin" / "rocprofv3"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(_FAKE_ROCPROFV3)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:/usr/bin:/bin")
    monkeypatch.delenv(rocprof.ENV_ROCPROF_BIN, raising=False)
    return fake


def _make_workload(
    tmp_path: Path,
    argv: list[str],
    *,
    collect=(),
    options=None,
    collect_dir=None,
    retain=None,
):
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
    if retain is not None:
        config[CONFIG_KEY_PROBE_EXTRAS]["retain"] = retain
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
    """Write the capture a profiled run would have produced.

    Call this *after* ``setup()``: the real rocprofv3 writes while the wrapped
    command runs, and ``setup()`` deliberately empties the collector directory
    so a resumed trial cannot summarise the attempt it is replacing. Calling it
    before ``setup()`` therefore seeds a stale previous attempt, which is what
    ``test_setup_discards_an_interrupted_attempts_artifacts`` wants.
    """
    out_dir = collect_dir / rocprof.OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n' f'"sgemm_tiled(float const*)",{calls},{total_ns}\n',
        encoding="utf-8",
    )
    return out_dir


def test_run_merges_collector_metrics(tmp_path, rocprofv3_on_path):
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    _rocprof_artifacts(collect_dir)
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


def test_run_collector_cannot_shadow_the_platform_keys(tmp_path, monkeypatch, rocprofv3_on_path):
    """The collector summary is merged UNDER the platform bookkeeping, so a
    collector emitting ``verdict`` / ``exit_code`` cannot rewrite the trial's
    outcome."""
    shadow = {"verdict": "pass", "exit_code": 0, "rocprof_gpu_time_ms": 1.0}
    # Patch both entrypoints: the fd-relative (streams) path used on POSIX and
    # the pathname fallback, so the merge-order contract is pinned on either.
    monkeypatch.setattr(rocprof, "parse_summary", lambda _out: shadow)
    monkeypatch.setattr(
        rocprof, "parse_summary_from_streams", lambda *_args, **_kwargs: shadow
    )
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    (collect_dir / rocprof.OUTPUT_SUBDIR).mkdir(parents=True)
    wl = _make_workload(tmp_path, ["false"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    assert result.metrics["verdict"] == "fail"
    assert result.metrics["exit_code"] != 0
    assert result.metrics["rocprof_gpu_time_ms"] == 1.0


def test_run_survives_a_collector_that_produced_nothing(tmp_path, rocprofv3_on_path):
    """rocprofv3 writes no files at all for a command with no GPU work."""
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    result = wl.run()
    assert result.passed is True
    assert "rocprof_kernel_count" not in result.metrics
    assert result.metrics["verdict"] == "pass"


def test_run_survives_a_malformed_capture(tmp_path, rocprofv3_on_path):
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    out_dir = collect_dir / rocprof.OUTPUT_SUBDIR
    (out_dir / "aorta_kernel_stats.csv").write_text("garbage\n", encoding="utf-8")
    result = wl.run()
    assert result.passed is True
    assert "rocprof_kernel_count" not in result.metrics


def test_run_still_reports_metrics_for_a_failing_trial(tmp_path, rocprofv3_on_path):
    """A profiled crash is exactly the case an operator wants numbers for."""
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["false"], collect=["rocprof"])
    wl.setup()
    _rocprof_artifacts(collect_dir)
    result = wl.run()
    assert result.passed is False
    assert result.metrics["rocprof_kernel_count"] == 23


def test_setup_discards_an_interrupted_attempts_artifacts(tmp_path, rocprofv3_on_path):
    """Probe resume replays a trial onto the same paths.

    Without a reset the retry would summarise the interrupted attempt's
    capture -- reporting kernel counts for work the resumed trial never did.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    stale = _rocprof_artifacts(collect_dir, total_ns=999_999_999, calls=4242)
    (stale / "leftover_rank_1.csv").write_text("stale\n", encoding="utf-8")

    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    assert list(stale.iterdir()) == []

    _rocprof_artifacts(collect_dir)
    result = wl.run()
    assert result.metrics["rocprof_kernel_count"] == 23


# ---- Retention ---------------------------------------------------------


def test_retention_prunes_the_collector_tree(tmp_path, rocprofv3_on_path):
    """Profiler traces are the artifact class retention exists for, but they
    land in a tree that is a *sibling* of the trial dir. Pruning only the
    trial dir would keep every capture in a sweep regardless of ``retain``.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"], retain={"on_pass": "none"})
    wl.setup()
    _rocprof_artifacts(collect_dir)
    result = wl.run()

    # Summaries are parsed before pruning, so the numbers outlive the trace.
    assert result.metrics["rocprof_kernel_count"] == 23
    assert not (collect_dir / rocprof.OUTPUT_SUBDIR / "aorta_kernel_stats.csv").exists()

    # The audit trail names the pruned file and where it came from.
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    recorded = doc.get("capture", {}).get("retention", {})
    deleted = recorded.get("deleted", [])
    assert any(entry.endswith("aorta_kernel_stats.csv") for entry in deleted)
    assert any(entry.startswith("..") for entry in deleted)
    assert recorded.get("freed_bytes", 0) > 0


def test_retention_refuses_a_collector_root_swapped_for_a_symlink(tmp_path, rocprofv3_on_path):
    """The payload can replace the collector root with a symlink while it runs.

    This is the time-of-check/time-of-use gap the pre-launch reset cannot close:
    the profiled command is handed this path (``rocprofv3 -d``), so a guard that
    only ran before launch proves nothing afterwards. ``apply_retention()``
    follows links, so pruning through one would delete files in its target,
    outside the results tree, at any level below ``full``.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    outside = tmp_path / "precious"
    (outside / rocprof.OUTPUT_SUBDIR).mkdir(parents=True)
    victim = outside / rocprof.OUTPUT_SUBDIR / "aorta_kernel_stats.csv"
    victim.write_text("not yours to delete", encoding="utf-8")

    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"], retain={"on_pass": "none"})
    wl.setup()  # creates the real directory, as a launch would
    shutil.rmtree(collect_dir)
    collect_dir.symlink_to(outside, target_is_directory=True)

    wl.run()

    # The whole point: the tree behind the symlink is untouched.
    assert victim.read_text(encoding="utf-8") == "not yours to delete"


def test_summary_refuses_a_collector_root_swapped_for_a_symlink(tmp_path, rocprofv3_on_path):
    """The read path has the same exposure: the parsers glob the tree.

    Not destructive, but it would pull file contents from outside the results
    tree into the trial metrics, so it is refused with the same guard.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    outside = tmp_path / "elsewhere"
    _rocprof_artifacts(outside)

    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"], retain={"on_pass": "full"})
    wl.setup()
    shutil.rmtree(collect_dir)
    collect_dir.symlink_to(outside, target_is_directory=True)

    result = wl.run()
    assert "rocprof_kernel_count" not in result.metrics
    assert "rocprof_artifact_dir" not in result.metrics


@pytest.mark.parametrize("retain_level", ["none", "full"])
def test_collector_subdir_swapped_for_a_symlink_is_refused(
    tmp_path, rocprofv3_on_path, retain_level
):
    """The guard must cover each collector *subdirectory*, not just the root.

    The parsers glob ``<root>/rocprof``, so swapping that one directory is the
    read exposure one level below the root. Mutation-verified: with the guard
    narrowed back to the root alone, this fails with ``rocprof_kernel_count:
    7`` -- a metric read straight out of a directory outside the run tree.

    Both retention levels are covered, but only the read half is known to be
    reachable: with the guard removed, ``apply_retention`` did *not* delete the
    planted file through the symlinked subdirectory. The deletion assertion is
    kept as a regression guard on that, not as a demonstrated exploit.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    # A *parseable* capture outside the run tree, so an unguarded read is
    # observable as a leaked metric rather than a silently empty one.
    outside = tmp_path / "outside_capture"
    outside.mkdir()
    victim = outside / "aorta_kernel_stats.csv"
    victim.write_text(
        '"Name","Calls","TotalDurationNs"\n"leaked_kernel",7,7000000\n',
        encoding="utf-8",
    )

    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"], retain={"on_pass": retain_level})
    wl.setup()
    subdir = collect_dir / rocprof.OUTPUT_SUBDIR
    shutil.rmtree(subdir)
    subdir.symlink_to(outside, target_is_directory=True)

    result = wl.run()

    # Nothing from outside the run tree may reach the metrics...
    assert "rocprof_kernel_count" not in result.metrics
    assert "leaked_kernel" not in str(result.metrics)
    # ...and nothing out there may be deleted.
    assert victim.read_text(encoding="utf-8").endswith("7,7000000\n")


def test_retention_prunes_rocprof_when_a_sibling_collector_wrote_nothing(
    tmp_path, rocprofv3_on_path
):
    """One never-created collector directory must not retain the others.

    ``layer_numerics`` is validated-only, so nothing in the platform makes its
    output directory. When absence counted as an unsafe path, the guard reported
    it, ``_prune_collector_tree`` bailed before pruning anything, and
    ``retain.on_pass: none`` silently kept the whole rocprof capture -- the
    hundreds of MB per trial retention exists to drop.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(
        tmp_path,
        ["true"],
        collect=["rocprof", "layer_numerics"],
        retain={"on_pass": "none"},
    )
    wl.setup()
    _rocprof_artifacts(collect_dir)
    assert not (collect_dir / "layer_numerics").exists()

    result = wl.run()

    # Parsed before pruning, so the metric survives the trace it came from.
    assert result.metrics.get("rocprof_kernel_count") == 23
    assert not (collect_dir / rocprof.OUTPUT_SUBDIR / "aorta_kernel_stats.csv").exists()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    deleted = doc.get("capture", {}).get("retention", {}).get("deleted", [])
    assert any(entry.endswith("aorta_kernel_stats.csv") for entry in deleted)


def test_retention_full_keeps_the_collector_tree(tmp_path, rocprofv3_on_path):
    """``full`` is the keep-everything default; the collector tree follows it."""
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"], retain={"on_pass": "full"})
    wl.setup()
    _rocprof_artifacts(collect_dir)
    wl.run()
    assert (collect_dir / rocprof.OUTPUT_SUBDIR / "aorta_kernel_stats.csv").exists()


def test_retention_without_a_collector_is_unchanged(tmp_path):
    """A run with no ``--collect`` must not gain a collector-tree scan."""
    wl = _make_workload(tmp_path, ["true"], retain={"on_pass": "none"})
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc.get("capture", {}).get("retention", {}).get("level") == "none"


# ---- Artifact / metrics location --------------------------------------


def test_collector_artifacts_are_a_sibling_of_the_hand_written_trial_dir(
    tmp_path, rocprofv3_on_path
):
    """Pin the real layout, which is not the obvious one.

    ``SubprocessWorkload`` hand-writes ``<cell>/trial_<n>/result.json`` from
    ``_aorta_log_prefix``, while the dispatcher threads ``_aorta_collect_dir``
    as ``<cell>/<workload>/trial_d<d>_m<m>_t<t>``. So the collector artifact
    directory is a SIBLING of the per-trial ``result.json`` tree, not inside
    it, and ``result.json`` carries no collector metrics -- those ride
    ``WorkloadResult.metrics`` into the dispatcher's trial JSON instead.
    """
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    _rocprof_artifacts(collect_dir)
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


def test_hand_written_result_json_carries_no_collector_metrics(tmp_path, rocprofv3_on_path):
    collect_dir = tmp_path / "_subprocess" / "trial_d0_m0_t0"
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    _rocprof_artifacts(collect_dir)
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
    wl = _make_workload(tmp_path, ["true"], collect=["rocprof"])
    wl.setup()
    _rocprof_artifacts(collect_dir)
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    recorded = doc.get("argv")
    assert recorded is not None
    assert Path(recorded[0]).name == "rocprofv3"
    assert recorded[-1] == "true"
