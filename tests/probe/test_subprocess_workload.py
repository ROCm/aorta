"""Unit tests for :class:`aorta.workloads._subprocess.SubprocessWorkload`.

Covers FR 1.11 (entry-point resolution), FR 1.12 (per-trial result.json
shape), and the Tier-1 verdict rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.run.discovery import get_workload_class
from aorta.workloads._subprocess import (
    CONFIG_KEY_LOG_PREFIX,
    CONFIG_KEY_PROBE_EXTRAS,
    CONFIG_KEY_SUBPROCESS_ARGV,
    SubprocessWorkload,
)

# ---- FR 1.17 (entry-point resolution) ------------------------------------


def test_resolved_via_entry_point():
    """``get_workload_class('_subprocess')`` returns SubprocessWorkload."""
    cls = get_workload_class("_subprocess")
    assert cls is SubprocessWorkload


# ---- Setup() guards ------------------------------------------------------


def test_setup_requires_subprocess_argv():
    """Direct invocation without the reserved argv key raises."""
    workload = SubprocessWorkload({})
    with pytest.raises(RuntimeError, match=CONFIG_KEY_SUBPROCESS_ARGV):
        workload.setup()


def test_setup_requires_log_prefix(tmp_path):
    """Missing ``_aorta_log_prefix`` raises -- the runner must set save_logs=True."""
    workload = SubprocessWorkload({CONFIG_KEY_SUBPROCESS_ARGV: ["echo", "hi"]})
    with pytest.raises(RuntimeError, match=CONFIG_KEY_LOG_PREFIX):
        workload.setup()


def test_setup_rejects_non_list_argv(tmp_path):
    workload = SubprocessWorkload(
        {
            CONFIG_KEY_SUBPROCESS_ARGV: "echo hi",
            CONFIG_KEY_LOG_PREFIX: str(tmp_path / "trial_d0_m0_t0"),
        }
    )
    with pytest.raises(RuntimeError, match="non-empty list"):
        workload.setup()


# ---- FR 1.12 (result.json shape + Tier 1 verdict) ------------------------


def _make_workload(tmp_path: Path, argv: list[str], **extras):
    """Build a SubprocessWorkload with a synthetic log_prefix that decodes to trial 0.

    The runner sets ``_aorta_log_prefix`` to ``<cell_dir>/<workload>/trial_d0_m0_t<N>``
    and SubprocessWorkload derives ``<cell_dir>/trial_<N>/`` from it
    (Path(prefix).parent.parent is the cell dir, the trial idx comes from
    the _t<N> suffix). The synthetic prefix mirrors that shape so the
    test exercises the real path-decoding.
    """
    workload_subdir = tmp_path / "_subprocess"
    workload_subdir.mkdir(parents=True, exist_ok=True)
    prefix = workload_subdir / "trial_d0_m0_t0"
    cfg = {
        CONFIG_KEY_SUBPROCESS_ARGV: argv,
        CONFIG_KEY_LOG_PREFIX: str(prefix),
        CONFIG_KEY_PROBE_EXTRAS: {
            "cell_name": "none-none",
            "env_passthrough_mode": "inherit",
            "timeout_per_trial": None,
            "cell_env_vars": {},
            **extras,
        },
    }
    return SubprocessWorkload(cfg)


def test_pass_minimum_result_shape(tmp_path):
    """Successful exit_code=0 yields verdict=pass + rubric-mandated fields."""
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    result = wl.run()
    trial_dir = tmp_path / "trial_0"
    result_path = trial_dir / "result.json"
    assert result_path.is_file()
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    # FR 1.12: minimum shape.
    for key in ("verdict", "exit_code", "walltime_sec", "argv", "cell_name", "trial_index"):
        assert key in doc, f"missing required key {key} in result.json"
    assert doc["verdict"] == "pass"
    assert doc["exit_code"] == 0
    assert doc["argv"] == ["true"]
    assert doc["cell_name"] == "none-none"
    assert doc["trial_index"] == 0
    assert isinstance(doc["walltime_sec"], (int, float))
    # stdout.log and stderr.log written.
    assert (trial_dir / "stdout.log").is_file()
    assert (trial_dir / "stderr.log").is_file()
    # WorkloadResult round-trip:
    assert result.passed is True


def test_fail_minimum_result_shape(tmp_path):
    """Non-zero exit yields verdict=fail."""
    wl = _make_workload(tmp_path, ["false"])
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "fail"
    assert doc["exit_code"] != 0
    assert result.passed is False


def test_missing_executable_yields_fail(tmp_path):
    """argv[0] not found surfaces as a Tier-1 fail with exit_code=127."""
    wl = _make_workload(tmp_path, ["definitely-not-a-real-binary-9d8f7s6"])
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "fail"
    assert doc["exit_code"] == 127
    assert result.passed is False


def test_timeout_kill_race_does_not_crash(tmp_path, monkeypatch):
    """Regression for PR #194 review: ``proc.kill()`` after a
    ``TimeoutExpired`` must not propagate ``ProcessLookupError``
    when the child happens to exit between the timeout firing and
    the kill landing. The workload should record a deterministic
    timed-out trial with ``exit_code=-1`` and a ``result.json``
    on disk -- crashing here would mean a trial silently disappears
    from the matrix.

    We simulate the race by monkeypatching ``Popen`` so ``wait()``
    raises ``TimeoutExpired`` and ``kill()``/``wait()`` raise
    ``ProcessLookupError`` (matching the Linux ESRCH behaviour
    when the child has already exited).
    """
    import subprocess as _subprocess

    real_popen = _subprocess.Popen

    class _RacingPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen(["true"], **{
                k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "env")
            })
            self.pid = self._real.pid
            self._wait_calls = 0

        def wait(self, timeout=None):
            self._wait_calls += 1
            if self._wait_calls == 1:
                # First ``proc.wait(timeout=...)`` -- pretend the
                # child is still running so the timeout branch fires.
                raise _subprocess.TimeoutExpired(cmd="true", timeout=timeout)
            # Second ``proc.wait()`` after kill() -- child is gone.
            raise ProcessLookupError(3, "No such process")

        def kill(self):
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(_subprocess, "Popen", _RacingPopen)

    wl = _make_workload(tmp_path, ["true"], timeout_per_trial=0.01)
    wl.setup()
    # The crash this test is pinning: previously the unguarded
    # proc.kill() would propagate ProcessLookupError out of run().
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "fail"
    assert doc["timed_out"] is True
    assert doc["exit_code"] == -1
    assert result.passed is False
