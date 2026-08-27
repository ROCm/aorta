"""Unit tests for :class:`aorta.workloads._subprocess.SubprocessWorkload`.

Covers FR 1.11 (entry-point resolution), FR 1.12 (per-trial result.json
shape), and the Tier-1 verdict rule.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from aorta.run.discovery import get_workload_class
from aorta.workloads._subprocess import (
    CONFIG_KEY_LOG_PREFIX,
    CONFIG_KEY_PROBE_EXTRAS,
    CONFIG_KEY_SUBPROCESS_ARGV,
    SubprocessWorkload,
    _coerce_disable_tokens,
    _tier1_only_fallback_verdict,
)

# ---- Issue #230: classifier-crash fallback honours exec_failed ----------


def test_fallback_verdict_exec_failed_resolves_to_error(tmp_path):
    # When the full classifier crashes AND the command never launched,
    # the Tier-1-only fallback must emit tier1:exec_failed -> error
    # (not exit_nonzero -> fail) from the synthetic 127 exit code.
    verdict, _ = _tier1_only_fallback_verdict(
        exit_code=127,
        timed_out=False,
        trial_dir=tmp_path,
        exec_failed=True,
        classifier_exc=RuntimeError("boom"),
    )
    assert verdict.verdict == "error"
    assert "tier1:exec_failed" in verdict.error_detectors_fired
    assert "tier1:exit_nonzero" not in verdict.failure_detectors_fired


def test_fallback_verdict_nonzero_exit_resolves_to_fail(tmp_path):
    # A genuine non-zero child exit (launched) still resolves to fail.
    verdict, _ = _tier1_only_fallback_verdict(
        exit_code=1,
        timed_out=False,
        trial_dir=tmp_path,
        exec_failed=False,
        classifier_exc=RuntimeError("boom"),
    )
    assert verdict.verdict == "fail"
    assert "tier1:exit_nonzero" in verdict.failure_detectors_fired


# ---- Issue #229: disable-token payload coercion --------------------------


def test_coerce_disable_tokens_none_is_empty():
    assert _coerce_disable_tokens(None, "disable_detectors") == frozenset()


def test_coerce_disable_tokens_accepts_sequence():
    assert _coerce_disable_tokens(["tier2:hang", "tier3"], "disable_detectors") == frozenset(
        {"tier2:hang", "tier3"}
    )


@pytest.mark.parametrize("bad", ["tier3", 5, {"a": 1}])
def test_coerce_disable_tokens_rejects_non_sequence(bad):
    # A bare string would otherwise iterate per-character into a set of
    # letters; fail fast like the sibling tier3_vram_growth knob.
    with pytest.raises(TypeError, match="disable_detectors"):
        _coerce_disable_tokens(bad, "disable_detectors")


@pytest.mark.parametrize("bad", [["tier3", 5], ("tier2:hang", None), [b"tier3"]])
def test_coerce_disable_tokens_rejects_non_string_elements(bad):
    # A non-string token would survive into the set and later crash
    # sorted(disabled_detectors) (mixed types) or silently no-op against
    # the string detector IDs -- reject per-entry, fail fast.
    with pytest.raises(TypeError, match="must all be strings"):
        _coerce_disable_tokens(bad, "disable_detectors")

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
    for key in ("verdict", "exit_code", "walltime_sec", "argv", "cell_name", "trial_index", "env"):
        assert key in doc, f"missing required key {key} in result.json"
    assert doc["verdict"] == "pass"
    assert doc["exit_code"] == 0
    assert doc["argv"] == ["true"]
    assert doc["cell_name"] == "none-none"
    assert doc["trial_index"] == 0
    assert isinstance(doc["env"], dict)
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


def test_env_file_failure_result_includes_env(tmp_path):
    """The env-file-validation failure result.json carries env (Copilot review).

    The normal path records ``env`` for audit/redaction; the failure path
    used to omit it, so a corrupted-env trial produced a result.json with
    no env to scrub. It now mirrors the normal shape.
    """
    wl = _make_workload(
        tmp_path,
        ["true"],
        env_passthrough_mode="file",
        cell_env_vars={"BAD_VALUE": "line1\nline2"},
    )
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    # Issue #230: a rejected probe.env is an infra/config error (no valid
    # observation), not a fail.
    assert doc["verdict"] == "error"
    assert doc["failure_type"] == "env_file_validation_failed"
    assert doc["env"] == {"BAD_VALUE": "line1\nline2"}
    # Schema parity with the normal probe path: the failure artifact carries
    # the same detector/capture/tier-timing/vram keys (empty, but present) so
    # downstream parsers never special-case env-file errors.
    for key in ("warn_detectors_fired", "capture", "tier_durations_ms", "peak_vram_mib"):
        assert key in doc, f"env-file failure result.json missing {key!r}"
    assert doc["warn_detectors_fired"] == []
    assert doc["capture"] == {}
    assert doc["tier_durations_ms"] == {}
    assert doc["peak_vram_mib"] is None


def test_missing_executable_yields_error(tmp_path):
    """argv[0] not found surfaces as an ``error`` verdict (issue #230).

    A command-not-found never launched, so it produced no valid
    observation -- ``tier1:exec_failed`` -> ``error`` (not ``fail``). The
    synthetic exit_code 127 is still recorded for the operator.
    """
    wl = _make_workload(tmp_path, ["definitely-not-a-real-binary-9d8f7s6"])
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "error"
    assert doc["exit_code"] == 127
    assert "tier1:exec_failed" in doc["error_detectors_fired"]
    assert doc["failure_detectors_fired"] == []
    assert result.passed is False


@pytest.mark.parametrize(
    "extras",
    [
        {"disable_detector_tiers": ["tier1"]},
        {"disable_detectors": ["tier1:exit_nonzero"]},
    ],
)
def test_disabled_tier1_does_not_pass_unlaunched_command(tmp_path, extras):
    """A command that never launches must not pass even if Tier 1 is disabled.

    Regression for oyazdanb's #234 review: honouring a Tier-1 disable on
    the exec-failure path would let a command-not-found resolve to a green
    verdict -- a run that did no real work yet looks like a pass. The
    workload forces Tier 1 back on for the unlaunched path; the disable
    still applies to launched trials (asserted in the sibling test).

    With the three-way verdict (#230) the unlaunched path fires
    ``tier1:exec_failed`` -> ``error`` (not ``exit_nonzero`` -> ``fail``),
    so the guard now asserts a non-green ``error`` verdict.
    """
    wl = _make_workload(tmp_path, ["definitely-not-a-real-binary-9d8f7s6"], **extras)
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "error"
    assert doc["exit_code"] == 127
    assert "tier1:exec_failed" in doc["error_detectors_fired"]
    assert "tier1:exit_nonzero" not in doc["failure_detectors_fired"]
    assert result.passed is False
    # Capture must echo the *effective* disabled set the classifier honoured,
    # not the requested one -- else the artifact would claim tier1 was
    # disabled while tier1:exit_nonzero shows as fired (Copilot #234 review).
    capture = doc["capture"]
    assert "tier1" not in capture.get("disabled_detector_tiers", [])
    assert "tier1:exit_nonzero" not in capture.get("disabled_detectors", [])


@pytest.mark.parametrize(
    "extras",
    [
        {"disable_detector_tiers": ["tier1"]},
        {"disable_detectors": ["tier1:exit_nonzero"]},
    ],
)
def test_disabled_tier1_still_passes_launched_nonzero_exit(tmp_path, extras):
    """The Tier-1 disable still applies on a launched trial.

    Pins the lower bound of the unlaunched-path guard above: a child that
    actually ran and exited non-zero is silenced by the disable as the
    operator intended (the override is scoped to the exec-failure path).
    """
    wl = _make_workload(tmp_path, ["false"], **extras)  # launches, exits 1
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "pass"
    assert "tier1:exit_nonzero" not in doc["failure_detectors_fired"]
    assert result.passed is True
    # On a launched trial the disable is honoured, so the effective set
    # equals the requested set and is recorded in capture for audit.
    capture = doc["capture"]
    recorded = capture.get("disabled_detector_tiers", []) + capture.get(
        "disabled_detectors", []
    )
    assert recorded == list(extras.values())[0]


def test_exec_time_failure_flags_main_work_not_started(tmp_path):
    """Exec-time ``Popen`` failures must report
    ``main_work_started=False`` / ``executed_iterations=0`` so the
    matrix outcome classifier doesn't conflate a command-not-found
    with a completed 1/1 trial.

    Regression for PR #194 round-5 review: the workload used to
    hard-code ``main_work_started=True`` / ``executed_iterations=1``
    even on the ``FileNotFoundError`` / ``PermissionError`` /
    ``OSError`` exec-time-failure branch. The ``result.json`` is
    still written (artifact contract from PR #194 round 4) but the
    WorkloadResult now reflects "we never actually ran the child".
    """
    wl = _make_workload(tmp_path, ["definitely-not-a-real-binary-9d8f7s6"])
    wl.setup()
    result = wl.run()
    # Artifact tree intact (round-4 contract).
    assert (tmp_path / "trial_0" / "result.json").is_file()
    # Round-5 contract: matrix-side semantics reflect the exec-time failure.
    assert result.main_work_started is False, (
        "main_work_started=True for an exec-time failure misrepresents "
        "command-not-found as a completed trial"
    )
    assert result.executed_iterations == 0, (
        "executed_iterations=1 for an exec-time failure misrepresents "
        "command-not-found as a completed iteration"
    )
    assert result.failure_count == 1
    assert result.failure_details[0]["type"] == "subprocess_exec_failed", (
        "failure_details should distinguish exec-time failure from a "
        "normal subprocess non-zero exit"
    )


def test_successful_trial_flags_main_work_started(tmp_path):
    """Normal subprocess exits (zero OR non-zero) keep
    ``main_work_started=True`` -- the child actually ran. Pins the
    upper bound of the launched-flag change: the matrix outcome
    classifier still sees a normal 1/1 trial when the user command
    just exits with a non-zero status.
    """
    wl = _make_workload(tmp_path, ["false"])  # exits 1
    wl.setup()
    result = wl.run()
    assert result.main_work_started is True
    assert result.executed_iterations == 1
    assert result.failure_count == 1
    assert result.failure_details[0]["type"] == "subprocess_nonzero_exit"


def test_failure_detail_type_is_detector_failure_on_clean_exit(tmp_path):
    """Issue #229: a trial that exits 0 but is failed by a non-Tier-1
    detector (here a Tier-4 NaN log pattern) must NOT be labelled
    ``subprocess_nonzero_exit`` -- that contradicts ``exit_code == 0``.

    This is the exact minimal repro from the issue body. The failure came
    purely from the classifier, so the type is ``detector_failure``.
    """
    wl = _make_workload(tmp_path, ["bash", "-c", "echo 'loss is NaN'; exit 0"])
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    # The classifier failed the trial on Tier 4, despite a clean exit.
    assert doc["exit_code"] == 0
    assert doc["verdict"] == "fail"
    assert "tier4:nan_signature" in doc["failure_detectors_fired"]
    # The failure-detail type now agrees with the actual exit state.
    assert result.failure_details[0]["exit_code"] == 0
    assert result.failure_details[0]["type"] == "detector_failure", (
        "failure_details[].type must not say 'subprocess_nonzero_exit' when "
        "exit_code == 0 (issue #229)"
    )


def test_failure_detail_type_is_timeout_when_timed_out(tmp_path):
    """Issue #229: a timed-out trial reports ``subprocess_timeout`` rather
    than ``subprocess_nonzero_exit`` -- the child did not voluntarily exit
    with a non-zero status; it was killed by the per-trial timeout."""
    wl = _make_workload(tmp_path, ["sleep", "30"], timeout_per_trial=0.5)
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["timed_out"] is True
    assert result.failure_details[0]["timed_out"] is True
    assert result.failure_details[0]["type"] == "subprocess_timeout"


def test_non_executable_script_yields_error(tmp_path):
    """A user command pointing at a file without the +x bit must land
    as an ``error`` verdict (exit_code=126; issue #230), NOT escape to
    the dispatcher as ``infrastructure_failed``.

    Regression for PR #194 review: previously only ``FileNotFoundError``
    was caught. ``PermissionError`` (EACCES, raised by ``Popen`` when
    argv[0] exists but isn't executable) escaped the handler, leaving
    the per-trial directory without a ``result.json`` and breaking
    the documented "every probe trial leaves an artifact" contract.
    """
    script = tmp_path / "no_exec_bit.sh"
    script.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    script.chmod(0o644)  # readable, but NOT executable
    wl = _make_workload(tmp_path, [str(script)])
    wl.setup()
    result = wl.run()
    result_path = tmp_path / "trial_0" / "result.json"
    assert result_path.exists(), (
        "result.json missing: PermissionError escaped instead of being "
        "captured as a Tier-1 fail (regression of PR #194 review fix)"
    )
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    assert doc["verdict"] == "error"
    assert doc["exit_code"] == 126
    assert "tier1:exec_failed" in doc["error_detectors_fired"]
    assert result.passed is False
    # stderr.log should carry the diagnostic so the operator knows
    # which exec-time error fired.
    stderr_text = (tmp_path / "trial_0" / "stderr.log").read_text(encoding="utf-8")
    assert "Permission" in stderr_text or "permitted" in stderr_text.lower()


def test_popen_oserror_yields_error(tmp_path, monkeypatch):
    """A generic ``OSError`` from ``Popen`` (e.g. ENOEXEC "Exec format
    error" for a shebang-less script) also lands as an ``error`` verdict
    (issue #230) with the artifact tree intact, rather than escaping to
    the dispatcher.

    Regression for PR #194 review: only ``FileNotFoundError`` and
    ``PermissionError`` were named explicitly in the previous handler;
    other ``OSError`` subclasses (ENOEXEC, ELOOP, ...) leaked through.
    """
    import subprocess as _subprocess

    def _raises_oserror(*args, **kwargs):
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(_subprocess, "Popen", _raises_oserror)
    wl = _make_workload(tmp_path, ["/some/path/with/bad/format"])
    wl.setup()
    result = wl.run()
    result_path = tmp_path / "trial_0" / "result.json"
    assert result_path.exists(), (
        "result.json missing: bare OSError escaped instead of being " "captured as a Tier-1 fail"
    )
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    assert doc["verdict"] == "error"
    # Exit code falls back to 1 for non-{FileNotFound,Permission} OSError.
    assert doc["exit_code"] == 1
    assert "tier1:exec_failed" in doc["error_detectors_fired"]
    assert result.passed is False
    stderr_text = (tmp_path / "trial_0" / "stderr.log").read_text(encoding="utf-8")
    assert "Exec format error" in stderr_text


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

    from aorta.workloads import _subprocess as workload_mod

    # The amd-smi pre/post snapshots in run() shell out via subprocess.run
    # (which uses ``with Popen(...)``). Neutralise the Tier-3 poll so the
    # global Popen patch below only governs the workload's own child launch
    # -- the fake Popen has no context-manager protocol.
    monkeypatch.setattr(workload_mod, "poll_amd_smi", lambda _state: None)

    real_popen = _subprocess.Popen

    class _RacingPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen(
                ["true"],
                start_new_session=True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("stdout", "stderr", "env")
                },
            )
            self.pid = self._real.pid
            self._wait_calls = 0

        def wait(self, timeout=None):
            self._wait_calls += 1
            if self._wait_calls == 1:
                # First ``proc.wait(timeout=...)`` -- pretend the
                # child is still running so the timeout branch fires.
                raise _subprocess.TimeoutExpired(cmd="true", timeout=timeout)
            # Second ``proc.wait()`` inside the teardown -- child is gone.
            raise ProcessLookupError(3, "No such process")

        def send_signal(self, sig):
            # The teardown fallback path (when the process group is gone)
            # signals the direct child; the child already exited -> ESRCH.
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(_subprocess, "Popen", _RacingPopen)

    wl = _make_workload(tmp_path, ["true"], timeout_per_trial=0.01)
    wl.setup()
    # The crash this test is pinning: previously the unguarded
    # proc.kill() would propagate ProcessLookupError out of run().
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    # Issue #230: a timeout with no recognised hang is an ``error`` (the
    # trial never produced a valid observation), not a ``fail``. A timeout
    # the hang monitor DOES latch co-fires tier2:hang and stays ``fail``
    # (see test_timed_out_keeps_latched_hang_flag).
    assert doc["verdict"] == "error"
    assert "tier1:timeout" in doc["error_detectors_fired"]
    assert doc["timed_out"] is True
    assert doc["exit_code"] == -1
    assert result.passed is False


# ---- #220 (orphaned child teardown on timeout / interrupt) ---------------


def test_child_started_in_new_session(tmp_path, monkeypatch):
    """The child must lead its own session so the whole tree is reapable.

    Without ``start_new_session=True`` a ``sudo -> bash -> docker run -> python3``
    tree cannot be torn down via ``os.killpg`` and survives an interrupted /
    timed-out trial, keeping its GPUs pinned (#220).
    """
    from aorta.workloads import _subprocess as workload_mod

    captured: dict = {}
    real_popen = workload_mod.subprocess.Popen

    def _spy_popen(*args, **kwargs):
        captured.update(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(workload_mod.subprocess, "Popen", _spy_popen)

    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    wl.run()
    assert captured.get("start_new_session") is True


def test_terminate_process_tree_escalates_term_then_kill(monkeypatch):
    """SIGTERM first, then SIGKILL after the grace period, on the group."""
    from aorta.workloads import _subprocess as workload_mod

    signals: list[int] = []
    fake_pgid = 424242

    monkeypatch.setattr(workload_mod.os, "getpgid", lambda pid: fake_pgid)
    # Ensure the self-group guard does not trip (pgid != our group).
    monkeypatch.setattr(workload_mod.os, "getpgrp", lambda: fake_pgid + 1)
    monkeypatch.setattr(
        workload_mod.os, "killpg", lambda pgid, sig: signals.append(sig)
    )

    class _Stubborn:
        pid = 777

        def __init__(self):
            self._waits = 0

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                # Ignored SIGTERM -> force the SIGKILL escalation.
                raise workload_mod.subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return -9

    workload_mod._terminate_process_tree(_Stubborn(), grace_sec=0.01)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_terminate_process_tree_refuses_own_group(monkeypatch):
    """Safety net: never signal the aorta process's own group via killpg."""
    from aorta.workloads import _subprocess as workload_mod

    killpg_calls: list = []
    sent: list = []
    our_group = 555

    monkeypatch.setattr(workload_mod.os, "getpgid", lambda pid: our_group)
    monkeypatch.setattr(workload_mod.os, "getpgrp", lambda: our_group)
    monkeypatch.setattr(
        workload_mod.os, "killpg", lambda pgid, sig: killpg_calls.append(sig)
    )

    class _Child:
        pid = 888

        def send_signal(self, sig):
            sent.append(sig)

        def wait(self, timeout=None):
            return 0

    workload_mod._terminate_process_tree(_Child(), grace_sec=0.01)
    assert killpg_calls == [], "must not killpg our own process group"
    assert sent == [signal.SIGTERM], "must fall back to signalling the child"


def test_terminate_process_tree_warns_on_non_esrch_killpg(monkeypatch, caplog):
    """A non-ESRCH killpg failure (e.g. EPERM) must be surfaced, not swallowed.

    The expected ESRCH race stays silent, but an EPERM means the group could
    not be torn down and the tree may leak (#220). The operator needs that
    visibility, and teardown still falls back to signalling the direct child.
    """
    import logging

    from aorta.workloads import _subprocess as workload_mod

    fake_pgid = 626262
    sent: list = []

    monkeypatch.setattr(workload_mod.os, "getpgid", lambda pid: fake_pgid)
    monkeypatch.setattr(workload_mod.os, "getpgrp", lambda: fake_pgid + 1)

    def _eperm_killpg(pgid, sig):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(workload_mod.os, "killpg", _eperm_killpg)

    class _Child:
        pid = 999

        def send_signal(self, sig):
            sent.append(sig)

        def wait(self, timeout=None):
            return 0

    with caplog.at_level(logging.WARNING, logger=workload_mod.log.name):
        workload_mod._terminate_process_tree(_Child(), grace_sec=0.01)

    assert any("killpg" in r.message for r in caplog.records), (
        "non-ESRCH killpg failure must be logged at WARNING"
    )
    assert sent == [signal.SIGTERM], "must still fall back to the direct child"


def test_terminate_process_tree_escalates_to_kill_on_interrupt(monkeypatch):
    """A Ctrl-C during the SIGTERM grace wait must still SIGKILL, then re-raise.

    Regression for the #220 follow-up: if ``KeyboardInterrupt`` propagated out
    of the first ``proc.wait()`` the function would skip the SIGKILL
    escalation and re-orphan a SIGTERM-ignoring group (e.g. a stubborn
    ``docker run`` client). It must force-kill the group before propagating.
    """
    from aorta.workloads import _subprocess as workload_mod

    signals: list[int] = []
    fake_pgid = 313131

    monkeypatch.setattr(workload_mod.os, "getpgid", lambda pid: fake_pgid)
    monkeypatch.setattr(workload_mod.os, "getpgrp", lambda: fake_pgid + 1)
    monkeypatch.setattr(
        workload_mod.os, "killpg", lambda pgid, sig: signals.append(sig)
    )

    class _InterruptingChild:
        pid = 999

        def __init__(self):
            self._waits = 0

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                # Operator hits Ctrl-C again while we wait out the grace.
                raise KeyboardInterrupt()
            return -9

    with pytest.raises(KeyboardInterrupt):
        workload_mod._terminate_process_tree(_InterruptingChild(), grace_sec=0.01)
    assert signals == [signal.SIGTERM, signal.SIGKILL], (
        "interrupt during grace must still escalate to SIGKILL before re-raising"
    )


def test_keyboard_interrupt_reaps_tree_and_reraises(tmp_path, monkeypatch):
    """Ctrl-C during a trial must reap the child tree, then re-raise.

    Regression for #220: an interrupted probe previously left the wrapped
    child tree running. The workload must signal the group and propagate the
    ``KeyboardInterrupt`` so the run aborts as the operator intended.
    """
    from aorta.workloads import _subprocess as workload_mod

    real_popen = workload_mod.subprocess.Popen
    torn_down: list = []

    class _InterruptingPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen(
                ["true"],
                start_new_session=True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("stdout", "stderr", "env")
                },
            )
            self.pid = self._real.pid

        def wait(self, timeout=None):
            raise KeyboardInterrupt()

    def _spy_teardown(proc, grace_sec=workload_mod._TERMINATE_GRACE_SEC):
        torn_down.append(proc)

    monkeypatch.setattr(workload_mod, "poll_amd_smi", lambda _state: None)
    monkeypatch.setattr(workload_mod.subprocess, "Popen", _InterruptingPopen)
    monkeypatch.setattr(workload_mod, "_terminate_process_tree", _spy_teardown)

    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    with pytest.raises(KeyboardInterrupt):
        wl.run()
    assert len(torn_down) == 1, "child tree must be torn down on interrupt"


def test_keyboard_interrupt_during_monitor_start_reaps_tree(tmp_path, monkeypatch):
    """Ctrl-C during HangMonitor.start() (before proc.wait) must still reap.

    Regression for the #222 follow-up: the monitor wiring + start() now live
    in the same try as proc.wait(), so an interrupt landing between Popen and
    the wait reaches the ``except KeyboardInterrupt`` teardown instead of
    orphaning the child tree.
    """
    from aorta.workloads import _subprocess as workload_mod

    real_popen = workload_mod.subprocess.Popen
    torn_down: list = []

    class _OkPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen(
                ["true"],
                start_new_session=True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("stdout", "stderr", "env")
                },
            )
            self.pid = self._real.pid

        def wait(self, timeout=None):  # should never be reached
            return self._real.wait(timeout=timeout)

    class _InterruptingMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise KeyboardInterrupt()

        def stop(self):
            pass

    def _spy_teardown(proc, grace_sec=workload_mod._TERMINATE_GRACE_SEC):
        torn_down.append(proc)

    monkeypatch.setattr(workload_mod, "poll_amd_smi", lambda _state: None)
    monkeypatch.setattr(workload_mod.subprocess, "Popen", _OkPopen)
    monkeypatch.setattr(workload_mod, "HangMonitor", _InterruptingMonitor)
    monkeypatch.setattr(workload_mod, "_terminate_process_tree", _spy_teardown)

    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    with pytest.raises(KeyboardInterrupt):
        wl.run()
    assert len(torn_down) == 1, "tree must be reaped even if Ctrl-C precedes proc.wait()"


# ---- FR 2.9 (Phase 2 result.json shape) ----------------------------------


def test_result_json_phase_2_shape(tmp_path):
    """Phase 2 ``result.json`` carries the rubric §2.B FR 2.9 fields.

    The Phase 1 minimum shape is a subset; both shapes coexist (the
    Phase 1 test above continues to pass). Phase 2 adds:

    * ``failure_detectors_fired: list[str]``
    * ``warn_detectors_fired: list[str]``
    * ``capture: dict``
    * ``tier_durations_ms: dict``
    * ``peak_vram_mib: int | null``
    """
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    # Phase 2 keys present on every trial.
    assert isinstance(doc["failure_detectors_fired"], list)
    assert isinstance(doc["warn_detectors_fired"], list)
    assert isinstance(doc["capture"], dict)
    assert isinstance(doc["tier_durations_ms"], dict)
    # peak_vram_mib is int | null (null in env-less smoke runs).
    assert doc["peak_vram_mib"] is None or isinstance(doc["peak_vram_mib"], int)
    # tier_durations_ms records each tier's wall-clock contribution.
    for tier_key in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        assert tier_key in doc["tier_durations_ms"]


def test_phase_2_failure_path_lists_tier1_detector(tmp_path):
    """A non-zero exit fires ``tier1:exit_nonzero`` in failure_detectors_fired."""
    wl = _make_workload(tmp_path, ["false"])
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert "tier1:exit_nonzero" in doc["failure_detectors_fired"]


def test_phase_1_shape_still_present_in_phase_2_doc(tmp_path):
    """Phase 1's minimum-shape keys remain (rubric §2.B FR 2.9 'subset' clause)."""
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    for key in ("verdict", "exit_code", "walltime_sec", "argv", "cell_name", "trial_index"):
        assert key in doc


def test_tier3_actually_runs_per_trial(tmp_path, monkeypatch):
    """Regression for PR #197 review: Tier 3 used to be unreachable because
    SubprocessWorkload constructed a fresh Tier3State per trial and hard-
    coded ``dmesg_text=None`` + ``amd_smi_*=None``. Now the workload
    invokes ``poll_amd_smi`` and ``scan_dmesg`` through the module-level
    shared state, so a fake amd-smi snapshot is enough to make Tier 3
    surface its detector ID end-to-end.
    """
    from aorta.probe.classifier.tier3_kernel import DETECTOR_VRAM_GROWTH
    from aorta.workloads import _subprocess as workload_mod

    monkeypatch.setenv("AORTA_PROBE_AMDSMI_FAKE", "vram=0,throttle=0")
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    pre_calls: dict[str, int] = {"n": 0}

    real_poll = workload_mod.poll_amd_smi

    def _toggle_snapshot(state):
        # First call (pre) returns the env-supplied snapshot; the
        # second call (post) returns a snapshot with VRAM jumped past
        # the rubric's growth threshold so scan_amd_smi will fire.
        from aorta.probe.classifier.tier3_kernel import (
            VRAM_GROWTH_THRESHOLD_MIB,
            AmdSmiSnapshot,
        )

        pre_calls["n"] += 1
        if pre_calls["n"] == 1:
            return real_poll(state)
        return AmdSmiSnapshot(
            vram_used_mib=VRAM_GROWTH_THRESHOLD_MIB + 1,
            thermal_throttle_count=0,
        )

    monkeypatch.setattr(workload_mod, "poll_amd_smi", _toggle_snapshot)
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    fired = set(doc["failure_detectors_fired"]) | set(doc["warn_detectors_fired"])
    assert DETECTOR_VRAM_GROWTH in fired, (
        "Tier 3 vram-growth detector did not fire even though the workload "
        "supplied pre/post snapshots crossing the growth threshold; the "
        "SubprocessWorkload Tier-3 wiring is silently disabled"
    )


def test_disable_tier3_skips_probes(tmp_path, monkeypatch):
    """Issue #229 review: disabling the ``tier3`` tier must skip the
    side-effecting amd-smi / dmesg probes in the workload, not merely
    drop their verdict contribution. Spy on both and assert neither runs.
    """
    from aorta.workloads import _subprocess as workload_mod

    calls = {"poll": 0, "dmesg": 0}

    def _spy_poll(_state):
        calls["poll"] += 1
        return None

    def _spy_dmesg(_state, **_kw):
        calls["dmesg"] += 1
        return []

    monkeypatch.setattr(workload_mod, "poll_amd_smi", _spy_poll)
    monkeypatch.setattr(workload_mod, "scan_dmesg", _spy_dmesg)

    wl = _make_workload(tmp_path, ["true"], disable_detector_tiers=["tier3"])
    wl.setup()
    wl.run()
    assert calls == {"poll": 0, "dmesg": 0}, (
        "disabling tier3 still ran the amd-smi/dmesg probes; the knob only "
        "filtered the verdict instead of skipping the collection"
    )


def test_tier3_probes_run_when_not_disabled(tmp_path, monkeypatch):
    # Counterpart to the disable test: with tier3 active the probes run.
    # Spy on BOTH probes so the test stays hermetic -- the real scan_dmesg()
    # shells out to `dmesg` (host-permission dependent, flaky/slow).
    from aorta.workloads import _subprocess as workload_mod

    calls = {"poll": 0, "dmesg": 0}

    def _spy_poll(_state):
        calls["poll"] += 1
        return None

    def _spy_dmesg(_state, **_kw):
        calls["dmesg"] += 1
        return []

    monkeypatch.setattr(workload_mod, "poll_amd_smi", _spy_poll)
    monkeypatch.setattr(workload_mod, "scan_dmesg", _spy_dmesg)
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    wl.run()
    assert calls["poll"] >= 1
    assert calls["dmesg"] >= 1


def test_tier3_probes_skipped_when_not_launched(tmp_path, monkeypatch):
    """Copilot #234 review: an exec-time Popen failure (the subprocess
    never launched) must skip the Tier-3 *post*-snapshot + dmesg scan.

    A command-not-found trial did no GPU work, so running those probes only
    adds overhead / permission noise and risks misattributing an unrelated
    host kernel/GPU event to a trial that never ran. The pre-snapshot still
    fires once -- it happens before the Popen attempt, so ``launched`` is
    not yet known -- but the post-launch collection is gated on ``launched``.
    """
    from aorta.workloads import _subprocess as workload_mod

    calls = {"poll": 0, "dmesg": 0}

    def _spy_poll(_state):
        calls["poll"] += 1
        return None

    def _spy_dmesg(_state, **_kw):
        calls["dmesg"] += 1
        return []

    monkeypatch.setattr(workload_mod, "poll_amd_smi", _spy_poll)
    monkeypatch.setattr(workload_mod, "scan_dmesg", _spy_dmesg)

    # Tier 3 stays enabled: the gate under test is the unlaunched flag, not
    # an operator disable (covered by test_disable_tier3_skips_probes).
    wl = _make_workload(tmp_path, ["definitely-not-a-real-binary-9d8f7s6"])
    wl.setup()
    wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["exit_code"] == 127  # exec-time (ENOENT) failure path
    assert calls["poll"] == 1, (
        "expected only the pre-snapshot to run; the post-snapshot must be "
        "skipped when the subprocess never launched"
    )
    assert calls["dmesg"] == 0, (
        "scan_dmesg ran for a subprocess that never launched; an unrelated "
        "kernel event could be misattributed to a command-not-found trial"
    )


def test_hang_grace_zero_survives_runtime_extraction(tmp_path, monkeypatch):
    """Regression for PR #197 review: ``probe_extras["hang_grace_period_at_start"]
    == 0.0`` is a validated "no grace, fire as soon as the window
    elapses" value (see ``test_hang_grace_period_zero_is_accepted``
    in ``test_recipe.py``). The runtime extraction used to
    short-circuit through ``... or DEFAULT_HANG_GRACE_SEC``, which
    treats ``0.0`` as falsy and silently substitutes the default,
    defeating the entire knob.

    Pin the behavior by intercepting ``HangMonitor.__init__`` and
    asserting it observed the configured ``0.0`` exactly.
    """
    from aorta.workloads import _subprocess as workload_mod

    captured: dict[str, float] = {}
    real_hang_monitor = workload_mod.HangMonitor

    class _SpyHangMonitor(real_hang_monitor):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            captured["hang_grace_period_at_start"] = kwargs["hang_grace_period_at_start"]
            captured["hang_window_sec"] = kwargs["hang_window_sec"]
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(workload_mod, "HangMonitor", _SpyHangMonitor)

    wl = _make_workload(
        tmp_path,
        ["true"],
        hang_grace_period_at_start=0.0,
        hang_window_sec=30.0,
    )
    wl.setup()
    wl.run()
    assert captured["hang_grace_period_at_start"] == 0.0, (
        "Configured hang_grace_period_at_start=0.0 was silently clobbered to "
        f"{captured['hang_grace_period_at_start']!r} -- the `or DEFAULT` falsy "
        "shortcut is back. Use an explicit `is None` check at the extraction."
    )
    assert captured["hang_window_sec"] == 30.0


def test_tier3_vram_growth_threaded_into_classifier(tmp_path, monkeypatch):
    """Regression for PR #215 review (Sonbol): ``probe_extras["tier3_vram_growth"]``
    must thread through the workload into the ``TrialContext`` handed to
    ``classify_trial`` so the recipe opt-out actually reaches the Tier-3
    VRAM check. Spy on ``classify_trial`` and assert it observed ``False``.
    """
    from aorta.workloads import _subprocess as workload_mod

    captured: dict[str, object] = {}
    real_classify = workload_mod.classify_trial

    def _spy_classify(ctx):
        captured["tier3_vram_growth"] = ctx.tier3_vram_growth
        return real_classify(ctx)

    monkeypatch.setattr(workload_mod, "classify_trial", _spy_classify)

    wl = _make_workload(tmp_path, ["true"], tier3_vram_growth=False)
    wl.setup()
    wl.run()
    assert captured.get("tier3_vram_growth") is False, (
        "probe_extras['tier3_vram_growth']=False did not reach the "
        "TrialContext -- the recipe opt-out is silently ignored "
        "(or classify_trial was never called, so the spy never ran)."
    )


def test_tier3_vram_growth_non_bool_payload_raises(tmp_path):
    """A malformed ``tier3_vram_growth`` payload (e.g. the string "false")
    raises ``TypeError`` rather than being truthily coerced to ``True`` and
    silently re-enabling the detector. Mirrors the recipe-builder guard and
    the sibling ``float(...)`` knobs that surface a bad payload instead of
    swallowing it.
    """
    wl = _make_workload(tmp_path, ["true"], tier3_vram_growth="false")
    wl.setup()
    with pytest.raises(TypeError, match="tier3_vram_growth.*must be a bool"):
        wl.run()


def test_classifier_crash_still_writes_result_json(tmp_path, monkeypatch):
    """Regression for PR #197 round-3 review: if ``classify_trial``
    raises (regex catastrophe, future refactor edge case, etc.),
    the workload MUST still write a ``result.json`` so the trial
    doesn't silently disappear from the matrix. Falls back to a
    Tier-1-only verdict derived from the captured exit code, and
    records the classifier exception under
    ``capture['classifier_error']`` so the operator sees the
    cause.
    """
    from aorta.workloads import _subprocess as workload_mod

    boom = RuntimeError("synthetic classifier crash for PR #197 review")

    def _exploding_classify(_ctx):
        raise boom

    monkeypatch.setattr(workload_mod, "classify_trial", _exploding_classify)

    # Use ``false`` so Tier 1 fallback fires ``tier1:exit_nonzero``.
    wl = _make_workload(tmp_path, ["false"])
    wl.setup()
    result = wl.run()

    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "fail"
    assert "tier1:exit_nonzero" in doc["failure_detectors_fired"]
    assert "classifier_error" in doc["capture"]
    assert "synthetic classifier crash" in doc["capture"]["classifier_error"]
    # The workload result still reports the failure (not a hang) so
    # the dispatcher records a failed trial deterministically.
    assert result.passed is False


def test_classifier_crash_fallback_honours_tier1_disable(tmp_path, monkeypatch):
    """The Tier-1-only crash fallback must honour the disable knob too.

    Bugbot #234 follow-up: a launched trial with Tier 1 silenced should
    stay ``pass`` even when ``classify_trial`` crashes -- the fallback
    applies the same disabled set, so a disabled ``tier1:exit_nonzero``
    is not re-fired behind the operator's back.
    """
    from aorta.workloads import _subprocess as workload_mod

    def _exploding_classify(_ctx):
        raise RuntimeError("synthetic classifier crash")

    monkeypatch.setattr(workload_mod, "classify_trial", _exploding_classify)
    wl = _make_workload(tmp_path, ["false"], disable_detectors=["tier1:exit_nonzero"])
    wl.setup()
    result = wl.run()

    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "pass"
    assert "tier1:exit_nonzero" not in doc["failure_detectors_fired"]
    assert "classifier_error" in doc["capture"]
    assert result.passed is True


def test_classifier_crash_fallback_still_fails_unlaunched_with_tier1_disabled(
    tmp_path, monkeypatch
):
    """A command that never launches must not pass in the crash fallback too.

    The exec-failure path forces Tier 1 back on (the ``effective_*`` set),
    and that same set feeds the fallback, so a command-not-found can't
    resolve green even when Tier 1 is disabled and the classifier crashed.
    With the three-way verdict (#230) the fallback fires
    ``tier1:exec_failed`` -> ``error`` for the unlaunched path.
    """
    from aorta.workloads import _subprocess as workload_mod

    def _exploding_classify(_ctx):
        raise RuntimeError("synthetic classifier crash")

    monkeypatch.setattr(workload_mod, "classify_trial", _exploding_classify)
    wl = _make_workload(
        tmp_path,
        ["definitely-not-a-real-binary-9d8f7s6"],
        disable_detector_tiers=["tier1"],
    )
    wl.setup()
    result = wl.run()

    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "error"
    assert "tier1:exec_failed" in doc["error_detectors_fired"]
    assert result.passed is False


def test_classifier_crash_on_passing_trial_falls_back_to_pass(tmp_path, monkeypatch):
    """A ``true`` exit + classifier crash -> Tier-1-only verdict is
    ``pass`` (no Tier 1 detector fires for exit_code=0), but the
    classifier_error still gets recorded so the operator knows the
    full classifier didn't run.
    """
    from aorta.workloads import _subprocess as workload_mod

    def _exploding_classify(_ctx):
        raise ValueError("simulated tier-4 regex catastrophe")

    monkeypatch.setattr(workload_mod, "classify_trial", _exploding_classify)
    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    result = wl.run()

    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "pass"
    assert doc["failure_detectors_fired"] == []
    assert "classifier_error" in doc["capture"]
    assert result.passed is True


def test_peak_vram_mib_threaded_from_amd_smi_snapshots(tmp_path, monkeypatch):
    """Regression for PR #197 round-6 review:
    ``peak_vram_mib`` was hard-coded to ``None`` both in the
    ``TrialContext`` handed to ``classify_trial`` and in the emitted
    ``result.json``, which made Tier-5 sandbox conditions like
    ``peak_vram_mib > 70000`` permanently unreachable on real hosts.

    With the fix, the workload computes a coarse high-water mark
    from the two amd-smi snapshots it already collects
    (pre-Popen + post-Popen) and threads the value into BOTH the
    classifier context and ``result.json``.
    """
    from aorta.probe.classifier.tier3_kernel import AmdSmiSnapshot
    from aorta.workloads import _subprocess as workload_mod

    pre = AmdSmiSnapshot(vram_used_mib=4000, thermal_throttle_count=0)
    post = AmdSmiSnapshot(vram_used_mib=71234, thermal_throttle_count=0)

    calls = {"n": 0}

    def _two_snapshots(_state):
        calls["n"] += 1
        return pre if calls["n"] == 1 else post

    monkeypatch.setattr(workload_mod, "poll_amd_smi", _two_snapshots)

    seen_ctx_peak: dict[str, int | None] = {}
    real_classify = workload_mod.classify_trial

    def _spy_classify(ctx):
        seen_ctx_peak["value"] = ctx.peak_vram_mib
        return real_classify(ctx)

    monkeypatch.setattr(workload_mod, "classify_trial", _spy_classify)

    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    wl.run()

    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["peak_vram_mib"] == 71234, (
        "peak_vram_mib in result.json should be max(pre.vram, post.vram); "
        f"got {doc['peak_vram_mib']!r}"
    )
    assert seen_ctx_peak["value"] == 71234, (
        "TrialContext.peak_vram_mib should match the result.json value so "
        "Tier-5 sandbox conditions and the emitted doc agree"
    )


def test_peak_vram_mib_none_when_amd_smi_unavailable(tmp_path, monkeypatch):
    """Both snapshots returning ``None`` -> ``peak_vram_mib`` stays
    ``None`` (the sandbox's ``None -> 0`` shim then keeps Tier-5
    conditions deterministic).
    """
    from aorta.workloads import _subprocess as workload_mod

    monkeypatch.setattr(workload_mod, "poll_amd_smi", lambda _state: None)

    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    wl.run()

    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["peak_vram_mib"] is None


def test_read_log_text_uses_replace_not_backslashreplace(tmp_path):
    """Regression for PR #197 round-6 review: ``_read_log_text`` used
    ``errors="backslashreplace"`` which expands each invalid byte
    into up to four characters (``\\\\xff``). For a binary-heavy log,
    that inflates the decoded string past ``MAX_LOG_BYTES``,
    defeating the documented regex-DoS cap.

    The fix decodes with ``errors="replace"`` so each invalid byte
    becomes a single U+FFFD replacement char, holding the 1:1
    byte->char invariant.
    """
    from aorta.probe.sandbox import MAX_LOG_BYTES
    from aorta.workloads._subprocess import _read_log_text

    # A blob of pure invalid bytes the size of the cap. ``replace``
    # gives len == MAX_LOG_BYTES; ``backslashreplace`` would give
    # 4 * MAX_LOG_BYTES.
    binary = b"\xff" * MAX_LOG_BYTES
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_bytes(binary)
    stderr_path.write_bytes(b"")

    text = _read_log_text(stdout_path, stderr_path)
    # Allow a small overhead for the ``\n`` joiner and the empty
    # stderr piece; the load-bearing assertion is the 4x cap.
    assert len(text) <= MAX_LOG_BYTES + 8, (
        f"decoded log length {len(text)} exceeds the MAX_LOG_BYTES "
        f"({MAX_LOG_BYTES}) cap -- the decode path is back on "
        "backslashreplace and a binary-heavy log can quadruple the "
        "regex-input size, defeating the cap."
    )


# ---- Latched-hang reconciliation against exit (false-positive fix) -------


def _force_latched_hang(monkeypatch):
    """Replace ``HangMonitor`` with a stub whose ``hang_detected`` is True.

    The real monitor's stdout/io legs are blind to work delegated to a
    child process tree (``sudo`` -> ``bash`` -> ``docker run`` ->
    container -> ``python3``): the wrapper goes silent and does almost
    no I/O of its own, so the monitor latches ``hang_detected`` on a
    perfectly healthy run. This stub reproduces that latch
    deterministically -- no real subprocess hang, no /proc polling, no
    amd-smi -- so the reconciliation logic in ``run()`` can be pinned.
    """
    from aorta.workloads import _subprocess as workload_mod

    real_monitor = workload_mod.HangMonitor

    class _AlwaysHangMonitor(real_monitor):  # type: ignore[misc, valid-type]
        def start(self) -> None:
            self.hang_detected = True

        def stop(self) -> None:
            pass

    monkeypatch.setattr(workload_mod, "HangMonitor", _AlwaysHangMonitor)


def _spy_classify_hang(monkeypatch, captured: dict):
    """Spy on ``classify_trial`` to capture the ``hang_detected`` it sees."""
    from aorta.workloads import _subprocess as workload_mod

    real_classify = workload_mod.classify_trial

    def _spy(ctx):
        captured["hang_detected"] = ctx.hang_detected
        return real_classify(ctx)

    monkeypatch.setattr(workload_mod, "classify_trial", _spy)


def test_clean_exit_discards_latched_hang_flag(tmp_path, monkeypatch):
    """A wrapped command that exits 0 within the timeout is NEVER
    classified ``tier2:hang``, even when the in-flight monitor latched
    ``hang_detected`` (issue: false-positive ``tier2:hang`` on
    docker/sudo-delegated workloads). A voluntary exit 0 within the
    timeout cannot have been hung, so the advisory flag is dropped
    before it reaches the classifier.
    """
    captured: dict = {}
    _force_latched_hang(monkeypatch)
    _spy_classify_hang(monkeypatch, captured)

    wl = _make_workload(tmp_path, ["true"])
    wl.setup()
    result = wl.run()

    assert captured["hang_detected"] is False, (
        "latched hang flag reached classify_trial despite a clean exit 0 "
        "within the timeout -- the reconciliation gate is missing"
    )
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "pass"
    assert doc["exit_code"] == 0
    assert doc["timed_out"] is False
    assert "tier2:hang" not in doc["failure_detectors_fired"]
    # Durable breadcrumb: the verdict is a clean pass, but the discarded
    # latch is recorded for the #224 follow-up to study.
    assert doc["capture"].get("tier2_hang_latched_but_reconciled") is True
    assert result.passed is True


def test_nonzero_exit_keeps_latched_hang_flag(tmp_path, monkeypatch):
    """A latched hang flag survives when the process exits non-zero.

    Reconciliation only discards the flag for a *clean* exit
    (``exit_code == 0 and not timed_out``). A non-zero exit is a real
    failure, so the hang signal is preserved and ``tier2:hang`` still
    surfaces -- the fix must not blanket-suppress the detector.
    """
    captured: dict = {}
    _force_latched_hang(monkeypatch)
    _spy_classify_hang(monkeypatch, captured)

    wl = _make_workload(tmp_path, ["false"])
    wl.setup()
    result = wl.run()

    assert captured["hang_detected"] is True, (
        "the reconciliation gate over-reached: it discarded the latched "
        "hang flag on a non-zero exit, where the signal is still meaningful"
    )
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "fail"
    assert doc["exit_code"] != 0
    assert "tier2:hang" in doc["failure_detectors_fired"]
    # No reconciliation happened -> no breadcrumb on a preserved-flag fail.
    assert "tier2_hang_latched_but_reconciled" not in doc["capture"]
    assert result.passed is False


def test_timed_out_keeps_latched_hang_flag(tmp_path, monkeypatch):
    """A latched hang flag survives a genuine timeout.

    When ``proc.wait(timeout=...)`` fires, the process is killed
    (``exit_code == -1``, ``timed_out == True``) -- a real hang. The
    reconciliation gate keys off ``exit_code == 0 and not timed_out``,
    so a timed-out trial keeps ``tier2:hang``.
    """
    captured: dict = {}
    _force_latched_hang(monkeypatch)
    _spy_classify_hang(monkeypatch, captured)

    wl = _make_workload(tmp_path, ["sleep", "30"], timeout_per_trial=0.2)
    wl.setup()
    result = wl.run()

    assert captured["hang_detected"] is True
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["timed_out"] is True
    assert doc["verdict"] == "fail"
    assert "tier2:hang" in doc["failure_detectors_fired"]
    assert "tier2_hang_latched_but_reconciled" not in doc["capture"]
    assert result.passed is False
