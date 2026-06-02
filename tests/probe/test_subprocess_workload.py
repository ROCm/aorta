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
    assert doc["verdict"] == "fail"
    assert doc["failure_type"] == "env_file_validation_failed"
    assert doc["env"] == {"BAD_VALUE": "line1\nline2"}


def test_missing_executable_yields_fail(tmp_path):
    """argv[0] not found surfaces as a Tier-1 fail with exit_code=127."""
    wl = _make_workload(tmp_path, ["definitely-not-a-real-binary-9d8f7s6"])
    wl.setup()
    result = wl.run()
    doc = json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "fail"
    assert doc["exit_code"] == 127
    assert result.passed is False


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


def test_non_executable_script_yields_fail(tmp_path):
    """A user command pointing at a file without the +x bit must land
    as a Tier-1 fail (exit_code=126), NOT escape to the dispatcher as
    ``infrastructure_failed``.

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
    assert doc["verdict"] == "fail"
    assert doc["exit_code"] == 126
    assert result.passed is False
    # stderr.log should carry the diagnostic so the operator knows
    # which exec-time error fired.
    stderr_text = (tmp_path / "trial_0" / "stderr.log").read_text(encoding="utf-8")
    assert "Permission" in stderr_text or "permitted" in stderr_text.lower()


def test_popen_oserror_yields_fail(tmp_path, monkeypatch):
    """A generic ``OSError`` from ``Popen`` (e.g. ENOEXEC "Exec format
    error" for a shebang-less script) also lands as a Tier-1 fail
    with the artifact tree intact, rather than escaping to the
    dispatcher.

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
    assert doc["verdict"] == "fail"
    # Exit code falls back to 1 for non-{FileNotFound,Permission} OSError.
    assert doc["exit_code"] == 1
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

    real_popen = _subprocess.Popen

    class _RacingPopen:
        def __init__(self, *args, **kwargs):
            self._real = real_popen(
                ["true"], **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "env")}
            )
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
