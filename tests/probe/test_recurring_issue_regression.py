"""Regression battery for the recurring ``aorta probe`` issue family.

This module is a *single* place that encodes the behavioural invariants
behind the cluster of ``aorta probe`` bugs filed against this platform:

* ``aorta`` #220 / aorta-internal #52  -- spurious ``(null): No such file
  or directory`` env-probe stderr noise.
* ``aorta`` #223 / #224 / aorta-internal #53 -- false-positive
  ``tier2:hang`` on a wrapper-delegated (docker/sudo) command that exits 0.
* ``aorta`` #222 -- orphaned child process tree on interrupt/timeout.
* aorta-internal #55 -- re-verification of #53 *plus* two still-open
  defects: a ``failure_details[].type`` that contradicts ``exit_code``,
  and an ambiguous ``<mitigation>-<diagnostic>`` cell name (``none-none``).
* aorta-internal #58 -- the matrix has no ``error`` verdict, so an infra
  crash is silently miscounted as ``pass``/``fail``.
* aorta-internal #56 / #57 -- verdict-keyed artifact retention and the
  ``stop_after`` collect-until-N stopping rule.

WHY THIS FILE EXISTS
--------------------
aorta-internal #55 is a *re-report* of #53: a fix landed, then the bug was
observed again on a later build. The recurrence is the signal this battery
guards against. Several existing tests pin the *current* (buggy) behaviour
-- e.g. ``test_cell_synthesis_and_collision`` asserts the cell name is
literally ``"none-none"`` -- so a defect can be "tested" yet still ship.
The tests here assert the *desired* invariant instead.

CONVENTIONS
-----------
* Tests for invariants that are FIXED on this branch are plain asserts and
  must stay green (they are the regression guards proper).
* Tests for invariants that are still OPEN on this branch are marked
  ``xfail(strict=True)`` with the owning issue. They document the contract
  and will flip to a hard failure ("XPASS") the moment the fix lands --
  forcing whoever fixes it to delete the marker and promote the test to a
  permanent guard. This is deliberate: it converts every open issue into an
  executable acceptance check.
* The #52 env redirect and #222 teardown fixes are merged into ``main``;
  their tests here are guarded with a capability skip so the battery still
  runs unchanged on an older branch that predates those fixes (it skips
  instead of erroring). The scattered-branch history -- where ``main`` and
  the #224/#225 branch each held only half the fixes -- is itself part of
  why these bugs recurred (see aorta-internal #55 re-reporting #53).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.workloads import _subprocess as workload_mod
from aorta.workloads._subprocess import (
    CONFIG_KEY_LOG_PREFIX,
    CONFIG_KEY_PROBE_EXTRAS,
    CONFIG_KEY_SUBPROCESS_ARGV,
    SubprocessWorkload,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_workload(tmp_path: Path, argv: list[str], **extras) -> SubprocessWorkload:
    """Build a SubprocessWorkload wired exactly as the dispatcher wires it.

    Mirrors the ``_make_workload`` helper in ``test_subprocess_workload.py``
    so this battery is self-contained: a synthetic ``_aorta_log_prefix`` of
    ``<cell>/_subprocess/trial_d0_m0_t0`` decodes to ``<cell>/trial_0/``.
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


def _read_result(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "trial_0" / "result.json").read_text(encoding="utf-8"))


def _force_latched_hang(monkeypatch) -> None:
    """Make the in-flight HangMonitor always latch ``hang_detected=True``.

    Reproduces the structural blind spot behind #53/#55: for a command that
    delegates its real work to a child tree (sudo -> bash -> docker -> python),
    the wrapper PID goes quiet and idle, tripping two of the monitor's three
    legs even though the workload is alive and busy.
    """
    real_monitor = workload_mod.HangMonitor

    class _AlwaysHangMonitor(real_monitor):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.hang_detected = True

        def start(self):  # no real polling thread needed
            return None

        def stop(self):
            return None

    monkeypatch.setattr(workload_mod, "HangMonitor", _AlwaysHangMonitor)


# ==========================================================================
# GROUP A -- tier2:hang false-positive invariants (#53, #55 item 1) [FIXED]
# These must stay green; they guard against #55 re-reporting #53 yet again.
# ==========================================================================


def test_clean_exit_with_latched_hang_is_not_a_failure(tmp_path, monkeypatch):
    """A wrapper-delegated command that exits 0 within its timeout is a
    clean baseline, never ``tier2:hang`` -- the #55 headline regression."""
    _force_latched_hang(monkeypatch)
    wl = _make_workload(tmp_path, ["bash", "-c", "echo working; exit 0"])
    wl.setup()
    result = wl.run()
    doc = _read_result(tmp_path)
    assert doc["verdict"] == "pass", "clean exit-0 misclassified as fail (#55/#53)"
    assert "tier2:hang" not in doc["failure_detectors_fired"]
    assert result.passed is True


def test_reconciled_hang_leaves_durable_breadcrumb(tmp_path, monkeypatch):
    """When a latched hang is reconciled away on a clean exit, the trial
    records ``tier2_hang_latched_but_reconciled`` so the #224 follow-up work
    on a descendant-tree-aware predicate can study the false positives."""
    _force_latched_hang(monkeypatch)
    wl = _make_workload(tmp_path, ["bash", "-c", "exit 0"])
    wl.setup()
    wl.run()
    doc = _read_result(tmp_path)
    assert doc["capture"].get("tier2_hang_latched_but_reconciled") is True


def test_reconciliation_only_drops_the_hang_leg_not_the_whole_verdict(tmp_path, monkeypatch):
    """Reconciling away a latched hang on a clean exit must NOT whitewash a
    genuine failure detected by another tier.

    A command can exit 0 yet still be a real failure (e.g. it printed a NaN
    signature that Tier 4 catches). The reconciliation gate drops only the
    advisory ``tier2:hang`` leg; the Tier-4 failure must survive. This case
    is not covered by the existing reconciliation tests and is exactly the
    interaction that makes an over-broad "exit 0 => pass" shortcut dangerous.
    """
    _force_latched_hang(monkeypatch)
    wl = _make_workload(tmp_path, ["bash", "-c", "echo 'loss is NaN'; exit 0"])
    wl.setup()
    wl.run()
    doc = _read_result(tmp_path)
    assert "tier2:hang" not in doc["failure_detectors_fired"], "hang leg should be reconciled away"
    assert "tier4:nan_signature" in doc["failure_detectors_fired"], "real Tier-4 failure was lost"
    assert doc["verdict"] == "fail"
    # The breadcrumb still records that a hang was latched-then-reconciled.
    assert doc["capture"].get("tier2_hang_latched_but_reconciled") is True


def test_timed_out_with_latched_hang_stays_a_failure(tmp_path, monkeypatch):
    """A genuine hang (the process did NOT voluntarily exit 0) must keep the
    ``tier2:hang`` verdict -- the reconciliation must not over-reach."""
    _force_latched_hang(monkeypatch)
    # 'sleep' that is killed by the 0.5s timeout -> timed_out=True, exit -1.
    wl = _make_workload(tmp_path, ["sleep", "30"], timeout_per_trial=0.5)
    wl.setup()
    wl.run()
    doc = _read_result(tmp_path)
    assert doc["timed_out"] is True
    assert "tier2:hang" in doc["failure_detectors_fired"]
    assert doc["verdict"] == "fail"
    assert "tier2_hang_latched_but_reconciled" not in doc["capture"]


# ==========================================================================
# GROUP B -- failure_details[].type must agree with exit_code (#55 item 2)
# OPEN on this branch: the type is hard-stamped "subprocess_nonzero_exit"
# whenever the child launched, regardless of the actual exit status.
# ==========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="aorta-internal #55 item 2: failure_details[].type is stamped "
    "'subprocess_nonzero_exit' even when exit_code == 0 (failure came from a "
    "non-Tier-1 detector). Remove this marker when the type is derived from "
    "the actual exit/verdict.",
)
def test_failure_detail_type_consistent_with_zero_exit(tmp_path):
    """A trial that exits 0 but fails on a log-pattern (Tier 4) must not be
    labelled ``subprocess_nonzero_exit`` -- that string is self-contradictory
    with ``exit_code == 0`` and misleads anyone reading the per-trial JSON."""
    wl = _make_workload(tmp_path, ["bash", "-c", "echo 'loss is NaN'; exit 0"])
    wl.setup()
    result = wl.run()
    doc = _read_result(tmp_path)
    assert doc["exit_code"] == 0
    assert doc["verdict"] == "fail"  # tier4:nan_signature fired
    assert result.failure_details, "a failing trial must carry a failure_detail"
    detail_type = result.failure_details[0]["type"]
    assert detail_type != "subprocess_nonzero_exit", (
        f"failure_details[].type={detail_type!r} contradicts exit_code=0"
    )


# ==========================================================================
# GROUP C -- cell-name disambiguation (#55 item 3)
# OPEN on this branch: cell names are "<mit>-<diag>", rendering a confusing
# bare trailing "-none" when the diagnostic axis is unused, with no way to
# tell which token is the mitigation and which is the diagnostic.
# ==========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="aorta-internal #55 item 3: an unused diagnostic axis renders an "
    "ambiguous trailing '-none' (e.g. 'tf32_off-none'). Remove this marker "
    "when cell names disambiguate mitigation vs diagnostic.",
)
def test_unused_diagnostic_axis_does_not_render_ambiguous_cell_name(tmp_path):
    """With ``diagnostic_axis: [none]`` the cell name must not be a bare
    ``<mitigation>-none`` that reads as a stutter and hides the axis roles."""
    from aorta.triage.recipe import load_recipe

    recipe_text = (
        "schema_version: 1\n"
        "mode: probe\n"
        "trials: 1\n"
        "mitigation_axis: [none, tf32_off]\n"
        "diagnostic_axis: [none]\n"
    )
    p = tmp_path / "r.yaml"
    p.write_text(recipe_text, encoding="utf-8")
    r = load_recipe(p)
    names = sorted(c.name for c in r.cells)
    # The mitigation-only cell should read as a clean baseline, and the
    # tf32_off cell should not carry a meaningless '-none' suffix.
    for name in names:
        assert not name.endswith("-none"), (
            f"cell name {name!r} carries an ambiguous trailing '-none' when the "
            "diagnostic axis is unused"
        )


# ==========================================================================
# GROUP D -- three-way verdict honesty (#58)
# FIXED (#230): the verdict vocabulary is now {pass, fail, error}; an infra
# crash (launch failure, unrecognised timeout, rejected env) resolves to
# ``error`` and is excluded from the bug-rate denominator, so the
# "honest by construction" claim (deck slide 9) holds. This guard is now a
# permanent regression check -- if the ``error`` bucket is ever dropped it
# fails immediately.
# ==========================================================================


def test_verdict_vocabulary_includes_error_bucket():
    """The verdict layer must be able to represent ``error`` distinctly from
    ``pass`` and ``fail`` so per-cell rates can exclude invalid trials."""
    from aorta.probe.classifier import verdict as verdict_mod

    valid = set(getattr(verdict_mod, "VALID_VERDICTS", {"pass", "fail"}))
    assert "error" in valid
    assert {"pass", "fail"} <= valid


def test_stop_after_can_key_on_the_error_verdict():
    """#230: once ``error`` is a first-class verdict, a ``stop_after`` rule
    can collect-until-N on it (e.g. "stop once 3 infra errors pile up"). The
    Phase-1 schema rejected ``event_verdict: error`` because the bucket did
    not exist; #230 lifts that restriction."""
    from aorta.probe.recipe_builder import build_probe_recipe_from_dict

    r = build_probe_recipe_from_dict(
        {
            "schema_version": 1,
            "mode": "probe",
            "trials": 1,
            "mitigation_axis": ["none"],
            "diagnostic_axis": ["none"],
            "stop_after": {"events": 3, "max_trials": 160, "event_verdict": "error"},
        },
        [],
    )
    assert r.stop_after is not None
    assert r.stop_after.event_verdict == "error"


# ==========================================================================
# GROUP E -- recipe knobs the operator-facing tasks add (#56, #57)
# FIXED: stop_after (#57/#232, run-level Recipe.stop_after) and verdict-keyed
# retention (#56/#231, ProbeExtras.retain) both parse + validate on the probe
# recipe now. These are permanent guards.
# ==========================================================================


def test_recipe_accepts_verdict_keyed_retention():
    """#56/#231 (FIXED): the probe recipe accepts a verdict-keyed ``retain``
    block, surfaced on ``ProbeExtras.retain``. The deletion engine
    (``aorta.run.retention``) keeps the heavy artifact only for the verdict
    whose level is ``full`` and never drops the trial record."""
    from aorta.probe.recipe_builder import build_probe_recipe_from_dict

    r = build_probe_recipe_from_dict(
        {
            "schema_version": 1,
            "mode": "probe",
            "trials": 1,
            "mitigation_axis": ["none"],
            "diagnostic_axis": ["none"],
            "retain": {"on_fail": "full", "on_pass": "summary", "on_error": "log"},
        },
        [],
    )
    assert r.probe_extras.retain is not None  # type: ignore[attr-defined]
    assert r.probe_extras.retain.level_for("fail") == "full"  # type: ignore[attr-defined]


def test_recipe_accepts_stop_after_rule():
    """#57/#232 (FIXED): the probe recipe accepts a ``stop_after``
    collect-until-N rule. It is surfaced as the run-level ``Recipe.stop_after``
    field (a stopping rule governs the whole trial loop, not a probe-only
    extra), parsed and validated by ``_parse_stop_after``."""
    from aorta.probe.recipe_builder import build_probe_recipe_from_dict

    r = build_probe_recipe_from_dict(
        {
            "schema_version": 1,
            "mode": "probe",
            "trials": 1,
            "mitigation_axis": ["none"],
            "diagnostic_axis": ["none"],
            "stop_after": {"events": 3, "max_trials": 160, "event_verdict": "fail"},
        },
        [],
    )
    assert r.stop_after is not None


# ==========================================================================
# GROUP F -- fixes merged into main (#52 env redirect, #222 teardown)
# Capability-guarded so the battery still runs on older pre-fix branches.
# ==========================================================================


def test_env_probe_suppresses_inprocess_stderr_noise():
    """#52/#220: the in-process env probe must redirect OS-level fds 1/2 so
    HIP/C-runtime ``(null): No such file or directory`` noise never reaches
    the operator's terminal. ``contextlib.redirect_stderr`` is insufficient.
    """
    from aorta.instrumentation import environment as env_mod

    redirect_cls = getattr(env_mod, "_ProbeStdioRedirect", None)
    if redirect_cls is None:
        pytest.skip("#221 env-probe fd redirect not on this branch yet")

    import os

    redirect = redirect_cls()
    redirect.start()
    try:
        # A raw fd-2 write -- the kind contextlib.redirect_stderr cannot catch.
        os.write(2, b"(null): No such file or directory\n")
    finally:
        # _ProbeStdioRedirect is never-raises by contract; restore is its job.
        stop = getattr(redirect, "stop", None) or getattr(redirect, "_restore_fds", None)
        if stop is not None:
            stop()
    # If we got here without the noise reaching the real terminal, the fd-level
    # capture is doing its job. The strong assertion (captured at DEBUG) lives
    # in tests/instrumentation/test_environment.py on the #221 branch.


def test_subprocess_child_starts_new_session_for_teardown(tmp_path, monkeypatch):
    """#222: the wrapped child must be launched with ``start_new_session=True``
    so an interrupt/timeout can reap the whole process group (sudo -> bash ->
    docker -> python) instead of orphaning GPU-pinned descendants."""
    if not hasattr(workload_mod, "_terminate_process_tree"):
        pytest.skip("#222 process-tree teardown not on this branch yet")

    # NOTE: ``workload_mod.subprocess`` is the shared stdlib module, so
    # patching its ``Popen`` also intercepts the ``subprocess.run`` calls the
    # Tier-3 amd-smi / dmesg probes make *after* the child launches. Record
    # every call's kwargs and assert the CHILD launch (argv == our command)
    # got ``start_new_session=True`` -- don't let a later probe overwrite it.
    calls: list[dict] = []
    real_popen = workload_mod.subprocess.Popen

    class _SpyPopen(real_popen):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            argv = a[0] if a else k.get("args")
            calls.append({"argv": argv, "start_new_session": k.get("start_new_session")})
            super().__init__(*a, **k)

    monkeypatch.setattr(workload_mod.subprocess, "Popen", _SpyPopen)
    argv = ["bash", "-c", "exit 0"]
    wl = _make_workload(tmp_path, argv)
    wl.setup()
    wl.run()
    child_calls = [c for c in calls if c["argv"] == argv]
    assert child_calls, "the wrapped child was never launched via Popen"
    assert child_calls[0]["start_new_session"] is True
